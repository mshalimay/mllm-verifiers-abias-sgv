import asyncio
import functools
import re
import time
from concurrent.futures import TimeoutError as FutureTimeoutError
from threading import Lock
from typing import Any, List

import aiolimiter
import httpx
from google.genai import types as genai_types
from google.genai.errors import APIError
from pydantic import BaseModel

from core_utils.image_utils import any_to_pil
from core_utils.logger_utils import logger
from core_utils.timing_utils import timeit
from llms.generation_config import GenerationConfig
from llms.providers.google.constants import (
    BASE_DELAY,
    BASE_DELAY_THROTTLED,
    MAX_API_WAIT_TIME,
    MAX_CALLS_PER_KEY,
    MAX_DELAY,
    MAX_GENERATION_PER_BATCH,
    MAX_REQUESTS_PER_MINUTE,
    MAX_RETRIES,
    MAX_RETRIES_THROTTLED,
    MAX_THINKING_BUDGETS,
    MAX_WAIT_PER_GEN,
    MIN_DELAY_RESET,
    MIN_SECS_BTW_GEN,
    MODEL_TEST_API_KEY,
    THINKING_MODELS_STRS,
    safety_settings,
)
from llms.providers.google.error_utils import PromptFeedbackError, TestAPIError, parse_quota_error
from llms.providers.google.google_client_manager import get_client_manager
from llms.providers.google.prompter import GooglePrompter
from llms.retry_utils import retry_with_exponential_backoff
from llms.types import Cache, ContentItem, Message

# ===============================================================================
# Globals
# ===============================================================================

# --- State control flow ---
PAYLOAD_TOO_LARGE = False  # If true, upload parts of the prompt to the cloud.
RESET_PROMPT = False  # If true, reset the prompt messages. Important if rotating keys with uploaded files.
NUM_CALLS_PER_KEY = {}  # Keep track of number of calls per key. Counts are reset when key is rotated.
LAST_API_CALL_PER_KEY = {}  # Keep track of last API call time per key.

# Global cache storing the provider-specific prompt messages, gen configs, api responses.
# This reduces overhead of prompt conversions and helps control flow during multiple generations.
cache = Cache()

# --- Batch generation / throttled execution configs ---
# THROTTLED_EXECUTION concurrency synchronization
LOCKS_PER_PROCESS = {}  # Lock for each process.
THROTTLED_EXECUTION = False  # Controls whether throttled execution is in use.
API_KEY_ERROR_COUNT = {}  # Keep track of number of errors per API key.

# ==============================================================================
# LINK: Provider-specific Error handling and retry logic
# ==============================================================================
# Google API error documentation: https://github.com/googleapis/python-genai/blob/main/google/genai/errors.py


def handle_custom_errors(e: Exception, *args: Any, **kwargs: Any) -> tuple[Exception, bool, bool, bool]:
    """Handle errors that are not due to the API call.

    Args:
        e (Exception): Error to handle

    Returns:
        tuple[Exception, bool, bool, bool]:
        `e`: Error to raise in case of no retry
        `should_retry`: Whether to retry the API call
        `apply_delay`: Whether to apply exp backoff delay before retrying
        `increment_retries`: Whether to increment the number of retries
    """
    global PAYLOAD_TOO_LARGE

    # If API just took too long to respond, and num_retries < max_retries, retry without delay
    if isinstance(e, FutureTimeoutError):
        logger.info(f"Google API didn't respond after {MAX_API_WAIT_TIME} seconds. Retrying...")
        should_retry, apply_delay, increment_retries = True, False, True

    # If HTTP error, upload prompt to cloud and retry with delay
    elif isinstance(e, httpx.HTTPError):
        PAYLOAD_TOO_LARGE = True
        logger.error(f"HTTP error: {e}")
        should_retry, apply_delay, increment_retries = True, True, True
        # note: this error sometimes seems to be due to payload size (Jun-2025, genai API)

    else:
        # All other errors: raise
        should_retry, apply_delay, increment_retries = False, False, False

    return e, should_retry, apply_delay, increment_retries


def handle_api_errors(e: APIError | TestAPIError, *args: Any, **kwargs: Any) -> tuple[Exception, bool, bool, bool]:
    """Handle errors from the provider API.

    Args:
        e (APIError): Error due to the API call

    Returns:
        tuple[Exception, bool, bool, bool]:
        `e`: Error to raise in case of no retry
        `should_retry`: Whether to retry the API call
        `apply_delay`: Whether to apply exp backoff delay before retrying
        `increment_retries`: Whether to increment the number of retries
    """

    global PAYLOAD_TOO_LARGE, API_KEY_ERROR_COUNT
    try:
        p_id = kwargs.get("process_id", 0)
        meta_data = kwargs.get("meta_data", {})
        model = kwargs.get("model", "")
        call_id = meta_data.get("call_id", 0)
        api_key: str = meta_data.get("api_key_used_in_call", "") or get_client_manager(p_id).api_key or ""
        quota_limit, retry_delay = parse_quota_error(e)

        # If error due to payload too large, upload prompt to cloud and retry without exponential backoff.
        if e.message and re.search("payload", e.message, re.IGNORECASE):
            logger.error(f"Google API error: call_id {call_id}, proccess {p_id}: {e.message}. Payload too large.")
            PAYLOAD_TOO_LARGE = True
            should_retry, apply_delay, increment_retries = True, True, False

        elif quota_limit or (e.message and "quota" in e.message.lower()):
            logger.error(
                f"Google API error: call_id {call_id}, proccess {p_id}: {e}.\nQuota limit: {quota_limit}, retry_delay: {retry_delay}.\nAPI key: {api_key[-5:] if api_key else None}",
            )

            if quota_limit == "day":
                logger.info(f"Day quota limit error for API key: {api_key[-5:] if api_key else None}. Resetting client.")
                reset_client(p_id, increment_api_key_retry_count=1, meta_data=meta_data, model=model)
                should_retry, apply_delay, increment_retries = True, False, False
            else:
                if retry_delay:
                    logger.info(f"Retry delay: {retry_delay}. Sleeping for {retry_delay + 1} seconds.")
                    time.sleep(retry_delay + 1)
                    should_retry, apply_delay, increment_retries = True, False, True
                else:
                    should_retry, apply_delay, increment_retries = True, True, True

        elif e.message and re.search("key expired", e.message, re.IGNORECASE):
            logger.error(f"Google API error: call_id {call_id}, proccess {p_id}: {e.message}. Key expired.")
            reset_client(p_id, increment_api_key_retry_count=float("inf"), meta_data=meta_data, model=model)
            should_retry, apply_delay, increment_retries = True, False, False

        elif e.message and re.search("Deadline", e.message, re.IGNORECASE):
            logger.error(f"Google API error: call_id {call_id}, proccess {p_id}: {e}.")
            PAYLOAD_TOO_LARGE = True
            should_retry, apply_delay, increment_retries = True, False, True
            # note: this error seems due to payload size (Jun-2025, genai API)

        # If other invalid argument error, do not retry.
        elif e.status and re.search("invalid", e.status, re.IGNORECASE):
            logger.error(f"Google API error: call_id {call_id}, proccess {p_id}: {e}. Stopping generation.")
            should_retry, apply_delay, increment_retries = False, False, True

        # All other API errors: retry with exponential backoff and increment the number of retries
        else:
            logger.error(f"Google API error: call_id {call_id}, proccess {p_id}: {e}. Retrying with exponential backoff...")
            should_retry, apply_delay, increment_retries = True, True, True

        return e, should_retry, apply_delay, increment_retries

    except Exception as ex:
        logger.error(ex)
        return ex, False, False, True


def handle_max_retries(e: Exception, *args: Any, **kwargs: Any) -> tuple[Exception, bool, bool, bool]:
    """Specific logic in case number of exp backoff retries is hit.

    Args:
        e (Exception): Error to handle

    Returns:
        tuple[Exception, bool, bool]: (`e`, `should_retry`, `apply_delay`)
        `e`: Error to raise in case of no retry
        `should_retry`: Whether to retry the API call
        `apply_delay`: Whether to apply exp backoff delay before retrying
    """

    try:
        logger.info(f"Max retries reached for API key. Last error: {e}.")

        # Update retry count for the current API key
        p_id = kwargs.get("process_id", 0)
        model = kwargs.get("model", "")
        meta_data = kwargs.get("meta_data", {})
        reset_client(p_id, model=model, meta_data=meta_data)
        should_retry, apply_delay, increment_retries = True, False, True

        return e, should_retry, apply_delay, increment_retries
    # If no API keys left or other errors, do not retry
    except Exception as e:
        logger.error(f"{e}")
        return e, False, False, True


retry_exp_backoff = functools.partial(
    retry_with_exponential_backoff,
    base_delay=BASE_DELAY,
    max_delay=MAX_DELAY,
    exp_base=2,
    jitter=True,
    max_retries=MAX_RETRIES,
    api_errors=(APIError, TestAPIError),
    custom_errors=(Exception,),
    handle_custom_errors=handle_custom_errors,
    handle_api_errors=handle_api_errors,
    handle_max_retries=handle_max_retries,
    max_workers=MAX_GENERATION_PER_BATCH,
    logger=logger,
)


# Tells `retry_exp_backoff` to retry if API call doesnt return in min(MAX_WAIT_PER_GEN * num_generations, MAX_API_WAIT_TIME) seconds
def timeout_getter(args: Any, kwargs: Any, key: str = "provider_gen_config") -> float:
    provider_gen_config: genai_types.GenerateContentConfig = kwargs.get(key)
    n = provider_gen_config.candidate_count if provider_gen_config.candidate_count else MAX_API_WAIT_TIME
    return min(MAX_WAIT_PER_GEN * n, MAX_API_WAIT_TIME)


def id_getter(args: Any, kwargs: Any, key: str = "process_id") -> str:
    process_id: int = kwargs.get(key, "")
    return str(process_id)


# ==============================================================================
# LINK: Client reset
# ==============================================================================
def _test_api_key(p_id: int = 0, model: str = MODEL_TEST_API_KEY):
    valid_key = False
    try:
        client = get_client_manager(p_id).get_client()

        prompt = "What's the capital of France?"

        _ = client.models.generate_content(model=model, contents=[prompt])
        valid_key = True
    except APIError as _:
        valid_key = False

    except Exception as e:
        raise e
    return valid_key


def reset_client(
    p_id: int = 0,
    increment_api_key_retry_count: int | float = 1,
    meta_data: dict[str, Any] = {},
    model: str = "",
):
    global THROTTLED_EXECUTION, RESET_PROMPT, API_KEY_ERROR_COUNT, NUM_CALLS_PER_KEY
    call_id = meta_data.get("call_id", 0)

    if not THROTTLED_EXECUTION:
        client_manager = get_client_manager(p_id)
        prev_api_key = client_manager.api_key
        NUM_CALLS_PER_KEY[prev_api_key] = 0
        client_manager.api_keys_retry_count[prev_api_key] += increment_api_key_retry_count
        client_manager.reset_api_key()
        NUM_CALLS_PER_KEY[client_manager.api_key] = 0
        RESET_PROMPT = True
        logger.info(f"prev_api_key: {prev_api_key[-5:] if prev_api_key else None}. New api_key: {client_manager.api_key[-5:] if client_manager.api_key else None}")

    else:
        global LOCKS_PER_PROCESS
        if p_id not in LOCKS_PER_PROCESS:
            LOCKS_PER_PROCESS[p_id] = Lock()

        sleep = False
        with LOCKS_PER_PROCESS[p_id]:
            client_manager = get_client_manager(p_id)
            api_key = client_manager.api_key
            start_time = client_manager.get_start_time(api_key)
            elapsed = time.time() - start_time

            # If other async call reset the client, just wait for it to finish
            if api_key != meta_data.get("api_key_used_in_call", ""):
                sleep = True
                logger.info(f"process {p_id}, call_id {call_id}: API key changed. Resetting prompt.")
                _ = GooglePrompter.reset_prompt(meta_data["provider_msgs"], p_id=p_id)

            # Elif, reset the key if sufficient time has passed
            elif elapsed > MIN_DELAY_RESET or API_KEY_ERROR_COUNT.get(api_key, 0) == 0:
                API_KEY_ERROR_COUNT[api_key] = 1

                valid_key = False
                num_tries = 1
                client_manager.api_keys_retry_count[client_manager.api_key] += increment_api_key_retry_count
                while not valid_key:
                    client_manager.reset_api_key()
                    add_text = "" if num_tries == 1 else "Previous api_key not valid. "
                    logger.info(f"Proccess {p_id}, call_id {call_id}: {add_text}Resetting client. Attempt {num_tries}.")
                    num_tries += 1
                    valid_key = _test_api_key(p_id, model=model)
                    new_key = client_manager.api_key
                    if not valid_key:
                        client_manager.api_keys_retry_count[new_key] += 1
                    else:
                        API_KEY_ERROR_COUNT[new_key] = API_KEY_ERROR_COUNT.get(new_key, 0)
                        logger.info(f"Proccess {p_id}, call_id {call_id}: API reset successful.")
                        logger.info(f"Proccess {p_id}, call_id {call_id}: Resetting prompt.")
                        _ = GooglePrompter.reset_prompt(meta_data["provider_msgs"], p_id=p_id)
            else:
                sleep = True
                API_KEY_ERROR_COUNT[api_key] = API_KEY_ERROR_COUNT.get(api_key, 0) + 1

        if sleep:
            if API_KEY_ERROR_COUNT[api_key] > 1:
                # Safeguard; shouldn't happen if code is correct
                logger.warning(f"ATTENTION: retried API key that is not working more than 1 times without reset.")
                time.sleep(MIN_DELAY_RESET - elapsed)
            else:
                time.sleep(10)


# ==============================================================================
# LINK: Output conversion: provider-specific -> uniform format
# ==============================================================================
def convert_single_part(part: genai_types.Part) -> ContentItem | None:
    """Convert a single part to a list of content items."""
    if part.text is not None:
        if part.thought:
            return ContentItem(type="reasoning", data=part.text, raw_model_output=part)
        else:
            return ContentItem(type="text", data=part.text, raw_model_output=part)

    elif part.inline_data is not None and part.inline_data.data is not None:
        try:
            img = any_to_pil(part.inline_data.data)
            return ContentItem(type="image", data=img, raw_model_output=part)
        except Exception as e:
            logger.warning(f"Error converting inline_data to PIL Image: {e}")
            # raise e
            return None

    elif part.function_call is not None:
        return ContentItem(type="function_call", data=part.to_json_dict(), raw_model_output=part)

    elif part.thought is not None:
        return ContentItem(type="reasoning", data=part.to_json_dict(), raw_model_output=part)

    elif part.executable_code is not None:
        logger.warning(f"Executable code generated but not implemented yet: {part.executable_code}")
        return None

    else:
        logger.warning(f"Part type not implemented: type {type(part)}; part: {part}")
        return None


def convert_single_generation(
    candidate: genai_types.Candidate,
) -> Message | None:
    """
    Convert a single candidate to a list of content items.
    """
    if candidate.content is None:
        return None
    if candidate.content.parts is None:
        return None

    all_parsed_parts = []
    # Convert all outputs of a single generation
    for part in candidate.content.parts:
        parsed_part = convert_single_part(part)
        if parsed_part is not None:
            all_parsed_parts.append(parsed_part)
    if all_parsed_parts:
        return Message(role="assistant", name="", contents=all_parsed_parts)
    else:
        return None


def convert_generations(response: genai_types.GenerateContentResponse, response_schema: bool = False) -> list[Message]:
    all_generations = []
    if response.candidates:
        # FIXME: for some reason, on Nov-2025 gemini started to return the content for canditate 1,2,..,n
        # as 'part' objects in candidate 0. + one candidate with their own part (except for the 0, which is the first entry)
        # This is a temporary fix prevent repetition of messages.
        if len(response.candidates) > 1 and response.candidates[0].content is not None:
            if response.candidates[0].content.parts is not None:
                response.candidates[0].content.parts = response.candidates[0].content.parts[0:1]
        for candidate in response.candidates:
            msg = convert_single_generation(candidate)
            if msg and msg.is_empty():
                continue
            all_generations.append(msg) if msg else None

    return all_generations


# ==============================================================================
# LINK: Prompt messages conversion: uniform -> provider-specific format
# ==============================================================================


def get_provider_msgs(
    messages: List[Message],
    use_cache: bool = True,
    p_id: int = 0,
    force_upload: bool = False,
) -> List[genai_types.Content]:
    """
    Processes the input messages:
    - Converts the prompt using GooglePrompter
    - Resets the prompt if needed
    - Uploads images if payload is too large
    """
    global RESET_PROMPT, PAYLOAD_TOO_LARGE, cache, THROTTLED_EXECUTION

    if use_cache:
        provider_msgs = cache.messages_to_provider
        # If no preprocessed messages, create them
        if not provider_msgs:
            provider_msgs = GooglePrompter.convert_prompt(messages, p_id=p_id, force_upload=force_upload)
            cache.messages_to_provider = provider_msgs
    else:
        provider_msgs = GooglePrompter.convert_prompt(messages, p_id=p_id, force_upload=force_upload)

    # Re-create prompt only on specific flags
    if RESET_PROMPT:
        RESET_PROMPT = False
        logger.info("Resetting prompt...")
        provider_msgs = GooglePrompter.reset_prompt(provider_msgs, p_id=p_id, flush_cache=True)
        if use_cache:
            cache.messages_to_provider = provider_msgs

    elif PAYLOAD_TOO_LARGE:
        logger.info("Payload too large. Uploading images...")
        provider_msgs = GooglePrompter.upload_all_images_for_prompt(provider_msgs, p_id=p_id, force_upload=False)
        PAYLOAD_TOO_LARGE = False
        if use_cache:
            cache.messages_to_provider = provider_msgs

    return provider_msgs


# ===============================================================================
# LINK: Generation config conversion: uniform -> provider-specific format
# ===============================================================================


def regularize_thinking_budget(gen_config: GenerationConfig):
    if gen_config.thinking_budget is not None:
        for k, v in MAX_THINKING_BUDGETS.items():
            if k in gen_config.model:
                if gen_config.thinking_budget > v:
                    logger.warning(f"Thinking budget regularized for model {gen_config.model}: {gen_config.thinking_budget} -> {v}")
                    gen_config.thinking_budget = v
                    break
        return gen_config.thinking_budget
    else:
        return None


def clean_schema(schema: dict) -> dict:
    """
    Recursively remove keys that are not allowed by the Schema model,
    such as "additionalProperties".
    """
    cleaned = {}
    for key, value in schema.items():
        if key == "additionalProperties":
            # Remove this key to prevent validation errors.
            continue
        if isinstance(value, dict):
            cleaned[key] = clean_schema(value)
        elif isinstance(value, list):
            cleaned[key] = [clean_schema(item) if isinstance(item, dict) else item for item in value]
        else:
            cleaned[key] = value
    return cleaned


def regularize_response_schema(original_schema: Any) -> dict[str, Any]:
    # Import tools for handling generic types and inspection
    import inspect
    from typing import get_args, get_origin

    # If already a valid JSON Schema dict, pass it through.
    if isinstance(original_schema, dict):
        return original_schema

    # If the schema is of the form list[SomeModel] (a generic alias)
    elif get_origin(original_schema) is list:
        # Get the inner type of the list (e.g. Translation)
        inner = get_args(original_schema)[0]
        if inspect.isclass(inner) and issubclass(inner, BaseModel):
            return {
                "type": "array",
                "items": inner.model_json_schema(),
            }
        elif isinstance(inner, dict):
            # In case someone passes list[dict] where dict is a valid JSON schema
            return {"type": "array", "items": inner}
        else:
            raise ValueError(f"Invalid list response schema type: {inner}")

    # If the schema is provided as a Pydantic model class
    elif inspect.isclass(original_schema) and issubclass(original_schema, BaseModel):
        return original_schema.model_json_schema()

    # If the schema is provided as an instance of a Pydantic model
    elif isinstance(original_schema, BaseModel):
        return type(original_schema).model_json_schema()

    else:
        raise ValueError(f"Invalid response schema: {original_schema}")


def gen_config_to_provider(gen_config: GenerationConfig) -> genai_types.GenerateContentConfig:
    """
    Convert the uniform generation configuration to Google API format.
    """
    # Generation arguments
    gen_args = {
        "candidate_count": gen_config.num_generations,
        "max_output_tokens": gen_config.max_tokens,
        "top_p": gen_config.top_p,
        "temperature": gen_config.temperature,
        "stop_sequences": gen_config.stop_sequences,
        "top_k": gen_config.top_k,
        "seed": gen_config.seed,
        "presence_penalty": gen_config.presence_penalty,
        "frequency_penalty": gen_config.frequency_penalty,
        "safety_settings": safety_settings,
        "response_modalities": gen_config.modalities,
    }
    if gen_config.response_schema:
        gen_args["response_schema"] = clean_schema(regularize_response_schema(gen_config.response_schema))
        gen_args["response_mime_type"] = "application/json"

    if gen_config.thinking_budget is not None:
        thinking_budget = regularize_thinking_budget(gen_config)
        if thinking_budget is not None:
            gen_args["thinking_config"] = genai_types.ThinkingConfig(thinking_budget=thinking_budget, include_thoughts=bool(gen_config.include_thoughts))

    if gen_config.tools:  # TODO: support other tools
        for tool in gen_config.tools:
            if tool == "web_search":
                web_tool = genai_types.Tool(google_search=genai_types.GoogleSearch())
                gen_args["tools"] = [web_tool]

    if gen_config.include_logprobs:
        gen_args["response_logprobs"] = True

    if gen_config.top_logprobs is not None:
        gen_args["top_logprobs"] = gen_config.top_logprobs

    provider_gen_config = genai_types.GenerateContentConfig(**gen_args)  # type: ignore
    return provider_gen_config


def regularize_provider_gen_config_for_model(
    model: str,
    provider_gen_config: genai_types.GenerateContentConfig,
) -> genai_types.GenerateContentConfig:
    """Regularize the provider generation configuration for model-specific settings.

    Args:
        model (str): Model name
        provider_gen_config (genai_types.GenerateContentConfig): Provider-specific generation configuration

    Returns:
        genai_types.GenerateContentConfig: Regularized generation configuration
    """

    # Model specific regularization
    if model == "gemini-2.0-flash-exp-image-generation":
        provider_gen_config.system_instruction = None
        provider_gen_config.candidate_count = 1
        # logger.warning(
        #     f"Warning: model {model} arguments regularized: {provider_gen_config.system_instruction} -> {None} and {provider_gen_config.candidate_count} -> {1}"
        # )

    else:
        provider_gen_config.response_modalities = ["Text"]

    if not any(model_str in model for model_str in THINKING_MODELS_STRS):
        provider_gen_config.thinking_config = None

    return provider_gen_config


def get_provider_gen_config(
    gen_config: GenerationConfig,
    provider_msgs: List[genai_types.Content],
    use_cache: bool = True,
) -> genai_types.GenerateContentConfig:
    """
    Constructs the generation configuration to be used in the API call.
    """
    global cache
    if not use_cache:
        return gen_config_to_provider(gen_config)

    # Else, try to get from cache
    if not cache.gen_config:
        provider_gen_config = gen_config_to_provider(gen_config)
        cache.gen_config = provider_gen_config
    else:
        provider_gen_config = cache.gen_config

    return provider_gen_config


# ==============================================================================
# LINK: Non-batch Generation
# ==============================================================================
def generate_from_google_chat_completion(
    messages: List[Message],
    gen_config: GenerationConfig,
    meta_data: dict[str, Any] = {},
) -> tuple[list[dict[str, Any]], list[Message]]:
    """Synchronous generation from Google API.

    This function:
     - Converts prompt messages and genconfig from uniform to provider-specific format.
     - Applies model-specific regularizations to genconfigs and messages.
     - Handles multiple generations of different modalities.
     - Converts model outputs back to uniform format.

    Args:
        messages (List[Message]): List of Message objects in uniform format to send to the model.
        gen_config (GenerationConfig): Generation configuration.

    Returns:
        tuple[list[Dict[str, Any]], list[Message]]: List of API responses and list of generated messages in uniform format
    """
    global MAX_GENERATION_PER_BATCH, cache
    cache.reset()

    # Number of generations remaining to be generated
    remaining_generation_count = gen_config.num_generations

    # Build provider messages and generation config on first call
    provider_gen_config = get_provider_gen_config(gen_config, [])
    # (obs.: needs to rebuild prompt on each retry to cover API resets; see `sync_api_call`)

    # Generate outputs
    logger.info(f"CALLING MODEL: `{gen_config.model}`: generating {gen_config.num_generations} outputs...")

    while remaining_generation_count > 0:
        provider_gen_config.candidate_count = min(MAX_GENERATION_PER_BATCH, remaining_generation_count)

        # Regularizing here handles where model supports only `num_generations=1` by calling the API `n` times
        provider_gen_config = regularize_provider_gen_config_for_model(gen_config.model, provider_gen_config)

        # Call the API
        response: genai_types.GenerateContentResponse
        response = _sync_api_call(model=gen_config.model, messages=messages, provider_gen_config=provider_gen_config, meta_data={})
        model_messages = convert_generations(response, gen_config.response_schema is not None)

        # Update cache and decrement remaining generation count
        if model_messages:
            cache.api_responses.append(response.model_dump(mode="json"))
            cache.model_messages.extend(model_messages)
            remaining_generation_count -= len(model_messages)

    return cache.api_responses, cache.model_messages


def maybe_rotate_api_key(process_id: int, meta_data: dict[str, Any]):
    global NUM_CALLS_PER_KEY
    if get_client_manager(process_id).api_key not in NUM_CALLS_PER_KEY:
        NUM_CALLS_PER_KEY[get_client_manager(process_id).api_key] = 0

    # Check if we need to reset before making the call
    if NUM_CALLS_PER_KEY[get_client_manager(process_id).api_key] >= MAX_CALLS_PER_KEY:
        api_key = get_client_manager(process_id).api_key
        logger.info(f"Max calls per key reached for API key {api_key[-5:] if api_key else None}. Resetting client.")
        reset_client(p_id=process_id, increment_api_key_retry_count=0, meta_data=meta_data)
        new_api_key = get_client_manager(process_id).api_key
        logger.info(f"New API key: {new_api_key[-5:] if new_api_key else None}")

    # Increment the count for the current key (original or new)
    api_key = get_client_manager(process_id).api_key
    logger.debug(f"Incrementing count for API key: {api_key[-5:] if api_key else None}")
    NUM_CALLS_PER_KEY[get_client_manager(process_id).api_key] += 1


def maybe_wait_before_gen(process_id: int, meta_data: dict[str, Any]):
    global LAST_API_CALL_PER_KEY
    # If last call was too recent, wait and update last call time
    if get_client_manager(process_id).api_key not in LAST_API_CALL_PER_KEY:
        LAST_API_CALL_PER_KEY[get_client_manager(process_id).api_key] = time.time()
    else:
        if time.time() - LAST_API_CALL_PER_KEY[get_client_manager(process_id).api_key] < MIN_SECS_BTW_GEN:
            api_key = get_client_manager(process_id).api_key
            logger.info(f"Waiting {MIN_SECS_BTW_GEN} secs before generation with API key {api_key[-5:] if api_key else None}.")  # fmt:off
            time.sleep(MIN_SECS_BTW_GEN)
        LAST_API_CALL_PER_KEY[get_client_manager(process_id).api_key] = time.time()


# Keeping the final API call separate from main generation function for isolated application of `retry_with_exponential_backoff`.
@timeit(custom_name=f"LLM:_sync_api_call")
@retry_exp_backoff(timeout_getter=timeout_getter, id_getter=id_getter)
def _sync_api_call(
    model: str,
    messages: List[Message],
    provider_gen_config: genai_types.GenerateContentConfig,
    process_id: int = 0,
    use_cache: bool = True,
    meta_data: dict[str, Any] = {},
) -> genai_types.GenerateContentResponse:
    """Synchronous API call to Google API."""

    global NUM_CALLS_PER_KEY
    if "api_key" in meta_data:
        process_id = meta_data["api_key"]
        use_cache = False

    # Get global client
    _ = get_client_manager(process_id).get_client()

    # If max num calls per key is reached, rotate the API key
    maybe_rotate_api_key(process_id, meta_data)

    # If last call was too recent, wait and update last call time
    maybe_wait_before_gen(process_id, meta_data)

    api_key = get_client_manager(process_id).api_key
    logger.debug(f"Get provider messages. API key: {api_key[-5:] if api_key else None}. Counts: {NUM_CALLS_PER_KEY[get_client_manager(process_id).api_key]}")

    # Get provider messages. Obs.: This caches, re-upload, reset prompts if needed.
    provider_msgs = get_provider_msgs(messages, use_cache=use_cache, p_id=process_id)

    # For Google, system prompt goes into the generation config
    if provider_msgs and provider_msgs[0].parts:
        provider_gen_config.system_instruction = provider_msgs[0].parts[0]

    # Regularize provider generation config for model-specific settings.
    provider_gen_config = regularize_provider_gen_config_for_model(model, provider_gen_config)

    api_key = get_client_manager(process_id).api_key
    logger.debug(f"generate_content. API key: {api_key[-5:] if api_key else None}. Counts: {NUM_CALLS_PER_KEY[get_client_manager(process_id).api_key]}")

    client = get_client_manager(process_id).get_client()

    api_key = get_client_manager(process_id).api_key
    logger.debug(f"Generating content for model {model} with API key {api_key[-5:] if api_key else None}")
    response = client.models.generate_content(
        model=model,
        contents=provider_msgs[1:],  # all msgs except sys_prompt, # type: ignore
        config=provider_gen_config,
    )
    api_key = get_client_manager(process_id).api_key
    logger.debug(f"Generate content sucessful. API key: {api_key[-5:] if api_key else None}")
    if hasattr(response, "text"):
        logger.debug(f"Response: {response.text}")

    # Check if the response was blocked and raise an error if so
    json_response = response.model_dump(mode="json")
    if json_response.get("prompt_feedback", {}) and json_response.get("prompt_feedback", {}).get("block_reason"):
        block_reason = json_response["prompt_feedback"]["block_reason"]
        if block_reason == "PROHIBITED_CONTENT":
            logger.warning(f"Prompt blocked by API: {json_response['prompt_feedback']}.\n messages: {messages}. Retrying by replacing system prompt with user prompt.")
            provider_gen_config.system_instruction = None
            provider_msgs[0].role = "user"
            time.sleep(MIN_SECS_BTW_GEN)
            response = client.models.generate_content(model=model, contents=provider_msgs, config=provider_gen_config)  # type: ignore
            json_response = response.model_dump(mode="json")
            if json_response.get("prompt_feedback", {}) and json_response.get("prompt_feedback", {}).get("block_reason"):
                raise PromptFeedbackError("Prompt feedback: " + json_response["prompt_feedback"]["block_reason"])
        else:
            logger.warning(f"Prompt feedback: {json_response['prompt_feedback']}")
    return response


# ==============================================================================
# LINK: Batch generation
# ==============================================================================
async def _async_api_call(
    model: str,
    messages: List[Message],
    provider_gen_config: genai_types.GenerateContentConfig,
    process_id: int = 0,
    use_cache: bool = False,
    meta_data: dict[str, Any] = {},
) -> genai_types.GenerateContentResponse:
    """Asynchronous API call to Google API."""

    # Get provider messages. Obs.: This caches, re-upload, reset prompts if needed.
    provider_msgs = get_provider_msgs(messages, use_cache=use_cache, p_id=process_id)

    # Obs.: this redundancy in system_instruction and regularization handle cases
    # Where API is redefined which must reset system_prompt message
    # (not currently the case, but possible if contain files or context caching)

    # For Google, system prompt goes into the generation config
    if provider_msgs and provider_msgs[0].parts:
        provider_gen_config.system_instruction = provider_msgs[0].parts[0]

    # Regularize provider generation config for model-specific settings.
    # (some models don't support a system_instruction)
    provider_gen_config = regularize_provider_gen_config_for_model(model, provider_gen_config)

    # Get global client
    api_key = None
    start_time = time.time()
    while not api_key:
        client_manager = get_client_manager(process_id)
        client = client_manager.get_client()
        api_key = client_manager.api_key
        if time.time() - start_time > 30:
            raise Exception("Failed to get API key")
    meta_data["api_key_used_in_call"] = api_key
    meta_data["provider_msgs"] = provider_msgs

    # Call the API
    response = client.models.generate_content(
        model=model,
        contents=provider_msgs[1:],  # type: ignore # all msgs except sys_prompt
        config=provider_gen_config,
    )
    return response


async def _throttled_google_agenerate(
    limiter: aiolimiter.AsyncLimiter,
    messages: List[Message],
    gen_config: GenerationConfig,
    dump_conversation_fun=None,
    dump_usage_fun=None,
    process_id: int = 0,
    call_id: int = 0,
) -> genai_types.GenerateContentResponse | dict[str, Any]:
    async with limiter:
        provider_msgs = get_provider_msgs(messages, use_cache=False, p_id=process_id)
        provider_gen_config = get_provider_gen_config(gen_config, provider_msgs, use_cache=False)
        num_retries = 0
        try:
            while num_retries < MAX_RETRIES_THROTTLED:
                num_retries += 1
                try:
                    logger.info(f"Calling model {gen_config.model} with call_id {call_id}")
                    meta_data: dict[str, Any] = {"call_id": call_id}
                    resp = await asyncio.wait_for(
                        _async_api_call(
                            model=gen_config.model,
                            messages=messages,
                            provider_gen_config=provider_gen_config,
                            process_id=process_id,
                            use_cache=False,
                            meta_data=meta_data,
                        ),
                        timeout=min(MAX_WAIT_PER_GEN * gen_config.num_generations, MAX_API_WAIT_TIME),
                    )
                    logger.info(f"1 generation successful for model {gen_config.model}")
                    _model_msgs = convert_generations(resp)
                    if not _model_msgs:
                        logger.warning(f"No model messages returned for {resp.model_dump(mode='json')}. Retrying...")
                        continue
                    if dump_conversation_fun or dump_usage_fun:
                        gen_config_dict = gen_config.to_dict()
                        if dump_conversation_fun:
                            model_messages = convert_generations(resp)
                            if model_messages:
                                dump_conversation_fun(messages, model_messages, gen_config_dict)
                        if dump_usage_fun:
                            dump_usage_fun([resp.model_dump(mode="json")], gen_config_dict)
                    return resp

                except APIError as e:
                    e, should_retry, apply_delay, increment_retries = handle_api_errors(e, **{"model": gen_config.model, "meta_data": meta_data, "process_id": process_id})
                    if not should_retry:
                        logger.error(f"Error in async generation: {e}")
                        return {"failed": True, "request_kwargs": provider_gen_config}

                    if hasattr(e, "status") and re.search("permission", e.status, re.IGNORECASE):  # type: ignore
                        logger.info(f"Proccess {process_id}, call_id {call_id}: Permission denied during throttled generation.")
                        try:
                            logger.info(f"Proccess {process_id}, call_id {call_id}: Resetting prompt.")
                            _ = GooglePrompter.reset_prompt(meta_data["provider_msgs"], p_id=process_id)
                            apply_delay, increment_retries = False, True
                        except Exception as e:
                            logger.error(f"Proccess {process_id}, call_id {call_id}: Error in async generation: {e}")
                            continue

                    if apply_delay:
                        logger.info(f"Proccess {process_id}, call_id {call_id}: Sleeping for {BASE_DELAY_THROTTLED} seconds")
                        await asyncio.sleep(BASE_DELAY_THROTTLED)

                except TimeoutError as e:
                    e, should_retry, apply_delay, increment_retries = handle_custom_errors(e, process_id)
                    if not should_retry:
                        logger.error(f"Error in async generation: {e}")
                        return {"failed": True, "request_kwargs": provider_gen_config}

                except Exception as e:
                    if "Server disconnected" in str(e):
                        logger.warning(f"Proccess {process_id}, call_id {call_id}: {e}. Sleeping for {BASE_DELAY_THROTTLED} seconds")
                        await asyncio.sleep(BASE_DELAY_THROTTLED // 3)
                    elif "peer closed connection" in str(e):
                        logger.warning(f"Proccess {process_id}, call_id {call_id}: {e}. Sleeping for {BASE_DELAY_THROTTLED} seconds")
                        await asyncio.sleep(BASE_DELAY_THROTTLED // 3)
                    else:
                        logger.error(f"Proccess {process_id}, call_id {call_id}: Error in async generation: {e}")
                        return_dict = {"failed": True, "request_kwargs": provider_gen_config}
                        return return_dict
            logger.error(f"Proccess {process_id}, call_id {call_id}: Max retries reached for model {gen_config.model}")
            return {"failed": True, "request_kwargs": provider_gen_config}
        except Exception as e:
            logger.error(f"Error in async generation: {e}")
            return {"failed": True, "request_kwargs": provider_gen_config}


def batch_generate_from_google(
    messages_list: list[list[Message]],
    gen_config: GenerationConfig,
    requests_per_minute: int = MAX_REQUESTS_PER_MINUTE,
    dump_conversation_funs=None,
    dump_usage_funs=None,
    process_id: int = 0,
) -> tuple[List[dict[str, Any]], List[List[Message]]]:
    """
    Args:
        prompt_batches: A list where each element is a list of messages to be sent to the API.
        gen_config: Generation configuration.
        requests_per_minute: Rate-limit for async requests.

    Returns:
        A tuple of:
          - List of raw JSON response objects.
          - List of generated text contents.
    """
    global cache, THROTTLED_EXECUTION
    cache.reset()
    THROTTLED_EXECUTION = True

    try:
        limiter = aiolimiter.AsyncLimiter(requests_per_minute)

        async def _async_generate() -> tuple[List[dict[str, Any]], List[List[Message]]]:
            tasks = [
                _throttled_google_agenerate(
                    limiter=limiter,
                    messages=messages,
                    gen_config=gen_config,
                    dump_conversation_fun=None if dump_conversation_funs is None else dump_conversation_funs[i],
                    dump_usage_fun=None if dump_usage_funs is None else dump_usage_funs[i],
                    process_id=process_id,
                    call_id=i,
                )
                for i, messages in enumerate(messages_list)
            ]
            logger.info(f"Generating {len(messages_list)} calls in batch mode for model {gen_config.model}")
            results: list[genai_types.GenerateContentResponse | dict[str, Any]] = await asyncio.gather(*tasks)
            all_api_responses = []
            all_model_messages = []
            for response in results:
                if isinstance(response, dict) and response.get("failed", False):
                    logger.warning(f"No generations returned for {response['request_kwargs']}")
                    continue

                assert isinstance(response, genai_types.GenerateContentResponse)
                model_messages = convert_generations(response)
                if model_messages:
                    all_api_responses.append(response.model_dump(mode="json"))
                    all_model_messages.extend(model_messages)
                else:
                    # logger.warning(f"No generations returned for {response['request_kwargs']}")
                    logger.warning(f"No generations returned for {response.model_dump(mode='json')}")
                    continue
            return all_api_responses, all_model_messages

        # Run the asyncio task synchronously.
        return asyncio.run(_async_generate())
    except Exception as e:
        THROTTLED_EXECUTION = False
        raise e


# ==============================================================================
# Token counting
# ==============================================================================


def google_count_tokens(model: str, lm_input: str) -> int | None:
    client = get_client_manager().get_client()
    token_count = client.models.count_tokens(model=model, contents=lm_input)
    return token_count.total_tokens

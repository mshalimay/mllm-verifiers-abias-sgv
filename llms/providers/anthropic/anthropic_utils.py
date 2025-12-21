import asyncio
import functools
import os
import re
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, Dict, List

from anthropic import AnthropicError, APIError, BadRequestError, NotFoundError
from anthropic._exceptions import RequestTooLargeError
from anthropic.types import Message as AnthropicMessage

from core_utils.logger_utils import logger
from core_utils.timing_utils import timeit
from llms.generation_config import GenerationConfig
from llms.providers.anthropic.anthropic_client_manager import get_client_manager
from llms.providers.anthropic.constants import DEFAULT_ANTHROPIC_MAX_TOKENS, DEFAULT_ANTHROPIC_MODE
from llms.providers.anthropic.error_utils import TestAPIError
from llms.providers.anthropic.prompter import AnthropicPrompter
from llms.retry_utils import retry_with_exponential_backoff
from llms.types import Cache, ContentItem, Message

# TODO: handle RequestTooLargeError uploading prompts?

# ===============================================================================
# Globals
# ===============================================================================
# --- State control flow ---
# OBS: Not implemented for OpenAI, but may be useful in the future.
RESET_PROMPT = False  # Whether to reset the prompt messages. Important if changing API keys and uploading files.
PAYLOAD_TOO_LARGE = False  # Whether to upload parts of the prompt to the cloud.

# Global cache storing the provider-specific prompt messages, gen configs, api responses.
# This reduces overhead of prompt conversions and also helps control flow during multiple generations.
cache = Cache()

# --- Handling retries with exponential backoff ---

MAX_API_WAIT_TIME = 5 * 60  # Maximum wait time for overall API call before flagging as failed
MAX_WAIT_PER_GEN = 2 * 60  # Maximum wait time for each generation
MAX_RETRIES = 3  # Max retries before declaring failure for an API key
MAX_DELAY = 60 * 2  # Maximum delay between retries

# --- Provider configs ---
# Max size of generation batch. # TODO: use throttled generation instead
MAX_GENERATION_PER_BATCH = 8


# Global persistent thread pool to manage API call timeout. Create it here to avoid overhead of creating it on each call.


# ===============================================================================
# LINK Provider-specific Error handling and retry logic
# ===============================================================================


def handle_custom_errors(e: Exception, *args, **kwargs) -> tuple[Exception, bool, bool, bool]:
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
    # By default, retry, apply exponential backoff, increment retries
    should_retry, apply_delay, increment_retries = True, True, True

    if isinstance(e, FutureTimeoutError):
        # If API just took too long to respond, and num_retries < max_retries, retry without delay
        logger.info(f"Anthropic API didn't respond after {MAX_API_WAIT_TIME} seconds. Retrying...")

        # Re-set the client (this cover cases where IP of machine changes).
        get_client_manager().set_client()
        should_retry, apply_delay, increment_retries = True, False, True
    else:
        # All other errors: retry with exponential backoff and increment the number of retries
        should_retry, apply_delay, increment_retries = True, True, True

    return e, should_retry, apply_delay, increment_retries


def handle_api_errors(e: APIError | TestAPIError, *args, **kwargs) -> tuple[Exception, bool, bool, bool]:
    """
    Handle errors raised by the OpenAI API call.
    """
    global RESET_PROMPT
    # By default, retry, apply exponential backoff, increment retries
    should_retry, apply_delay, increment_retries = True, True, True

    if isinstance(e, BadRequestError):
        logger.error(f"BadRequestError during Anthropic API call: {e}. Stopping generation.")
        # Do not retry on bad requests.
        should_retry, apply_delay, increment_retries = False, False, False

    elif hasattr(e, "message") and re.search("image exceeds", e.message, re.IGNORECASE):
        logger.error(f"Image exceeds max tokens. Resetting prompt and retrying...")
        RESET_PROMPT = True
        should_retry, apply_delay, increment_retries = True, False, True

    elif isinstance(e, NotFoundError):
        logger.error(f"NotFoundError during Anthropic API call: {e}. Stopping generation.")
        # Do not retry on not found errors.
        should_retry, apply_delay, increment_retries = False, False, False

    elif isinstance(e, RequestTooLargeError):
        logger.error(f"RequestTooLargeError during Anthropic API call: {e}. Stopping generation.")
        # Do not retry on request too large errors.
        should_retry, apply_delay, increment_retries = False, False, False

    else:
        # All other errors: retry with exponential backoff and increment the number of retries
        logger.error(f"Error: {e}. Retrying...")
        should_retry, apply_delay, increment_retries = True, True, True

    return e, should_retry, apply_delay, increment_retries


def handle_max_retries(e: Exception, *args, **kwargs) -> tuple[Exception, bool, bool, bool]:
    """Specific logic in case number of exp backoff retries is hit.

    Args:
        e (Exception): Error to handle

    Returns:
        tuple[Exception, bool, bool]: (`e`, `should_retry`, `apply_delay`)
        `e`: Error to raise in case of no retry
        `should_retry`: Whether to retry the API call
        `apply_delay`: Whether to apply exp backoff delay before retrying
    """

    # global RESET_PROMPT
    try:
        # Update retry count for the current API key
        client_manager = get_client_manager()
        client_manager.api_keys_retry_count[client_manager.api_key] += 1
        client_manager.reset_api_key()
        # RESET_PROMPT = True

        # If manages to redefine API key, retry without delay
        should_retry, apply_delay, increment_retries = True, False, True
        return e, should_retry, apply_delay, increment_retries

    # If no API keys left or other errors, do not retry
    except Exception as e:
        logger.error(f"{e}")
        return e, False, False, True


retry_exp_backoff = functools.partial(
    retry_with_exponential_backoff,
    base_delay=1.0,
    max_delay=MAX_DELAY,
    exp_base=2,
    jitter=True,
    max_retries=MAX_RETRIES,
    api_errors=(AnthropicError, TestAPIError),
    custom_errors=(FutureTimeoutError,),
    handle_custom_errors=handle_custom_errors,
    handle_api_errors=handle_api_errors,
    handle_max_retries=handle_max_retries,
    max_workers=MAX_GENERATION_PER_BATCH,
)


# If API call doesnt return in min(MAX_WAIT_PER_GEN * num_generations, MAX_API_WAIT_TIME) seconds, retry
# This should be passed to the `retry_exp_backoff` decorator (see `sync_api_call`)
def timeout_getter(args: Any, kwargs: Any, key: str = "provider_gen_config") -> float:
    provider_gen_config: dict[str, Any] = kwargs.get(key)
    n: float = provider_gen_config.get("n", MAX_API_WAIT_TIME)
    return min(MAX_WAIT_PER_GEN * n, MAX_API_WAIT_TIME)


# ==============================================================================
# LINK: Output conversion: provider-specific -> uniform format
# ==============================================================================


def convert_single_generation(anthropic_message: AnthropicMessage) -> Message | None:
    """
    Convert a single API completion to a uniform Message.
    """
    all_contents: List[ContentItem] = []
    if anthropic_message is None:
        return None

    content = anthropic_message.content

    if content is None or len(content) == 0:
        return None

    for c in content:
        # if c.type == "thinking":
        # TODO
        # if c.type == "redacted_thinking":

        # if c.type == "tool_use":

        if c.type == "text":
            all_contents.append(ContentItem(type="text", data=c.text))  # type: ignore

    if hasattr(anthropic_message, "role"):
        role = anthropic_message.role.lower()
    else:
        role = "assistant"

    if len(all_contents) == 0:
        return None

    return Message(role=role, contents=all_contents, name="")


def convert_generations(api_response: AnthropicMessage | dict[str, Any]) -> List[Message]:
    """
    Convert the API response into a list of Message objects.
    """
    all_generations = []

    # Anthropic is always a single generation / message, but keep this for future use
    if isinstance(api_response, AnthropicMessage):
        converted = convert_single_generation(api_response)
        if converted:
            all_generations.append(converted)

    return all_generations


# ==============================================================================
# LINK: Prompt messages conversion: uniform -> provider-specific format
# ==============================================================================


def get_provider_msgs(messages: List[Message], gen_config: GenerationConfig) -> List[Dict[str, Any]]:
    """
    Process the input messages:
      - Use OpenAIPrompter to convert the unified List[Message] to provider-specific format.
      - Reset the prompt or trigger image upload if needed.
    """
    global cache
    provider_msgs = cache.messages_to_provider
    if not provider_msgs:
        provider_msgs = AnthropicPrompter.convert_prompt(messages, gen_config.mode)
        cache.messages_to_provider = provider_msgs

    # global RESET_PROMPT # Not implemented for Anthropic
    # if RESET_PROMPT:
    #     logger.info("Resetting prompt...")
    #     provider_msgs = OpenAIPrompter.reset_prompt(provider_msgs)
    #     RESET_PROMPT = False
    #     cache.messages_to_provider = provider_msgs

    return provider_msgs


# ===============================================================================
# LINK: Generation config conversion: uniform -> provider-specific format
# ===============================================================================


def regularize_provider_gen_config_for_model(provider_gen_config: dict[str, Any]) -> dict[str, Any]:
    """
    Regularize generation arguments for different models.
    """
    if provider_gen_config.get("temperature") is not None:
        if provider_gen_config.get("top_p") is not None:
            logger.warning(f"Cannot set top_p and temperature for model {provider_gen_config.get('model')}. Ignoring top_p.")
            del provider_gen_config["top_p"]
        if provider_gen_config.get("top_k") is not None:
            logger.warning(f"Cannot set top_k and temperature for model {provider_gen_config.get('model')}. Ignoring top_k.")
            del provider_gen_config["top_k"]

    return provider_gen_config


def _build_chat_completion_config(gen_config: GenerationConfig) -> dict[str, Any]:
    """
    Builds the portion of the config shared by both 'chat_completion'
    and 'response' modes.
    """
    base_kwargs: dict[str, Any] = {
        "model": gen_config.model,
    }

    if gen_config.temperature is not None:
        base_kwargs["temperature"] = gen_config.temperature

    if gen_config.top_p is not None:
        base_kwargs["top_p"] = gen_config.top_p

    if gen_config.top_k is not None:
        base_kwargs["top_k"] = gen_config.top_k

    if gen_config.tools is not None:
        base_kwargs["tools"] = gen_config.tools

    if gen_config.tool_choice is not None:
        base_kwargs["tool_choice"] = gen_config.tool_choice

    if gen_config.stop_sequences is not None:
        base_kwargs["stop_sequences"] = gen_config.stop_sequences

    if gen_config.metadata is not None:
        base_kwargs["metadata"] = gen_config.metadata

    if gen_config.max_tokens is not None:
        base_kwargs["max_tokens"] = gen_config.max_tokens
    else:
        # Anthropic requires max_tokens to be set
        base_kwargs["max_tokens"] = DEFAULT_ANTHROPIC_MAX_TOKENS

    if gen_config.thinking_budget is not None:
        if gen_config.thinking_budget > 0:
            base_kwargs["thinking"] = {"type": "enabled", "budget_tokens": gen_config.thinking_budget}
            if gen_config.temperature != 1:
                logger.warning(f"Thinking requires temperature to be 1. Setting temperature to 1.")
                base_kwargs["temperature"] = 1
        else:
            base_kwargs["thinking"] = {"type": "disabled"}

    # if gen_config.web_search_options is not None:
    #     base_kwargs["web_search_options"] = gen_config.web_search_options

    # if gen_config.seed is not None:
    #     base_kwargs["seed"] = gen_config.seed

    # if gen_config.response_format is not None:
    #     base_kwargs["response_format"] = gen_config.response_format

    # chat_kwargs["n"] = gen_config.num_generations

    # base_kwargs["modalities"] = [m.lower() for m in gen_config.modalities]
    # base_kwargs["reasoning_effort"] = gen_config.reasoning_effort

    # base_kwargs["frequency_penalty"] = gen_config.frequency_penalty
    # base_kwargs["presence_penalty"] = gen_config.presence_penalty
    # base_kwargs["logprobs"] = gen_config.logprobs

    return base_kwargs


def gen_config_to_provider(gen_config: GenerationConfig) -> Dict[str, Any]:
    """
    Build Anthropic-specific generation arguments from a unified GenerationConfig,
    dispatching to the appropriate builder based on `mode`.
    """
    if not gen_config.mode:
        gen_config.mode = DEFAULT_ANTHROPIC_MODE

    if gen_config.mode == "chat_completion":
        return _build_chat_completion_config(gen_config)

    else:
        raise ValueError(f"Unsupported mode: {gen_config.mode}")


def get_provider_gen_config(
    gen_config: GenerationConfig,
) -> Dict[str, Any]:
    """
    Constructs the generation configuration to be used in the API call.
    """
    global cache
    if not cache.gen_config:
        provider_gen_config = gen_config_to_provider(gen_config)
        cache.gen_config = provider_gen_config
    else:
        provider_gen_config = cache.gen_config

    return provider_gen_config


# ===============================================================================
# LINK Synchronous Generation
# ===============================================================================
def generate_from_anthropic(
    messages: List[Message],
    gen_config: GenerationConfig,
) -> tuple[List[dict[str, Any]], List[Message]]:
    """
    Synchronous function that hides asynchronous parallelization.
    """

    global MAX_GENERATION_PER_BATCH, cache
    cache.reset()

    provider_gen_config = get_provider_gen_config(gen_config)
    _ = get_provider_msgs(messages, gen_config)

    logger.info(f"[{__file__}] CALLING MODEL: `{gen_config.model}`: generating {gen_config.num_generations} output(s)...")

    if gen_config.num_generations == 1:
        regularize_provider_gen_config_for_model(provider_gen_config)
        response = sync_call(messages=messages, provider_gen_config=provider_gen_config, gen_config=gen_config)
        cache.api_responses.append(response.to_dict())
        cache.model_messages.extend(convert_generations(response))
        return cache.api_responses, cache.model_messages

    # Else: multiple generations => use async call
    async def _async_generate() -> None:
        remaining_generation_count = gen_config.num_generations
        while remaining_generation_count > 0:
            # Decide how many to request in this round
            n = min(MAX_GENERATION_PER_BATCH, remaining_generation_count)
            regularize_provider_gen_config_for_model(provider_gen_config)

            # Launch n concurrent calls
            tasks = [asyncio.create_task(async_call(messages=messages, provider_gen_config=provider_gen_config, gen_config=gen_config)) for _ in range(n)]

            # Wait for completion
            results: list[AnthropicMessage] = await asyncio.gather(*tasks)

            num_generations = 0
            for response in results:
                model_messages = convert_generations(response)
                if model_messages:
                    # Cache responses; adjust this as needed based on how your cache works.
                    cache.api_responses.append(response.to_dict())
                    cache.model_messages.extend(model_messages)
                    num_generations += len(model_messages)

            if len(cache.model_messages) == 0:
                logger.warning("No generations returned from API call; breaking out of loop.")
                break

            remaining_generation_count -= num_generations

    # Run the coroutine in a fresh event loop (blocks here)
    asyncio.run(_async_generate())

    return cache.api_responses, cache.model_messages


@timeit(custom_name=f"LLM:sync_{os.path.basename(__file__)}_api_call")
@retry_exp_backoff(timeout_getter=timeout_getter)  # type: ignore
def sync_call(
    messages: List[Message],
    provider_gen_config: Dict[str, Any],
    gen_config: GenerationConfig,
) -> AnthropicMessage:
    # Get global client
    client = get_client_manager().get_client()

    # Get provider messages. Obs.: This caches, re-upload, reset prompts if needed.
    provider_msgs = get_provider_msgs(messages, gen_config)
    sys_prompt = provider_msgs[0]["text"]
    if sys_prompt:
        provider_gen_config["system"] = sys_prompt

    return client.messages.create(messages=provider_msgs[1:], **provider_gen_config)  # type:ignore


@timeit(custom_name=f"LLM:async_{os.path.basename(__file__)}_api_call")
@retry_exp_backoff(timeout_getter=timeout_getter)  # type: ignore
async def async_call(
    messages: List[Message],
    provider_gen_config: Dict[str, Any],
    gen_config: GenerationConfig,
) -> AnthropicMessage:
    # Get global client
    aclient = get_client_manager().get_aclient()

    # Get provider messages. Obs.: This caches, re-upload, reset prompts if needed.
    provider_msgs = get_provider_msgs(messages, gen_config)
    sys_prompt = provider_msgs[0]["text"]
    if sys_prompt:
        provider_gen_config["system"] = sys_prompt

    response = await aclient.messages.create(messages=provider_msgs[1:], **provider_gen_config)  # type:ignore
    return response

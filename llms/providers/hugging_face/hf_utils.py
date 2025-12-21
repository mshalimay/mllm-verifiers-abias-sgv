import asyncio
import functools
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, Callable, Dict, List

import aiolimiter
import openai
import requests
from openai import APIError, AsyncOpenAI, BadRequestError, NotFoundError, OpenAIError
from openai.types.chat import ChatCompletion

from core_utils.file_utils import flatten_dict
from core_utils.logger_utils import logger
from llms.generation_config import GenerationConfig
from llms.prompt_utils import get_message
from llms.providers.hugging_face.constants import (
    DEFAULT_HF_MODE,
    VLLM_DEFAULT_PARAMS_PER_MODEL,
    get_api_provider_arg,
)
from llms.providers.hugging_face.error_utils import EmptyResponseError, TestAPIError
from llms.providers.hugging_face.hugging_face_client_manager import get_client_manager
from llms.providers.hugging_face.model_specific.model_processor import ModelProcessor
from llms.providers.hugging_face.parsing_utils import count_tokens, get_trim_prompt_idxs
from llms.providers.hugging_face.prompter import HuggingFacePrompter
from llms.retry_utils import retry_with_exponential_backoff
from llms.types import Cache, Message

# ===============================================================================
# Globals
# ===============================================================================
# NOTE: It has been more convenient to have a file of functions and globals than
# an object-oriented solution due to multiple differences between providers that
# makes little of the code re-usable between them.
# The general structure is similar among providers though:
# 1) Convert prompt messages and generation config from uniform to provider-specific format
# 2) Call the API 'num_generations' times
# 3) Convert the API response back to list of messages in uniform format
# +: Provider-specific logic for error handling and retry with exponential backoff

# --- State control flow ---
RESET_PROMPT = False  # Controls whether to reset the prompt messages. Important if uploading files.
PAYLOAD_TOO_LARGE = False  # Controls if should upload parts of the prompt to the cloud.

# Global cache storing the provider-specific prompt messages, gen configs, api responses.
# This reduces overhead of prompt conversions + helps control flow during multiple generations.
cache = Cache()

# --- Handling retries with exponential backoff ---

MAX_API_WAIT_TIME = 10 * 60  # Maximum wait time for overall API call before flagging as failed
MAX_WAIT_PER_GEN = 5 * 60  # Maximum wait time for each generation
MAX_RETRIES = 2  # Max retries before switching to a new API key
MAX_DELAY = 60  # Maximum delay between retries

# --- Provider configs ---
# Max size of generation batch. # TODO: implement throttled generation (see openai_utils.py)
MAX_GENERATION_PER_BATCH = 8

# ==============================================================================
# LINK: Provider-specific Error handling and retry logic
# ==============================================================================

# By default, always retry, apply exponential backoff, increment retries
# The handlers below can override this behavior depending on error


def handle_custom_errors(
    e: Exception,
    *args,
    **kwargs,
) -> tuple[Exception, bool, bool, bool]:
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

    if isinstance(e, TimeoutError):
        # If API just took too long to respond, and num_retries < max_retries, retry without delay
        logger.info(f"Hugging Face didn't respond after {MAX_API_WAIT_TIME} seconds. Retrying...")
        should_retry, apply_delay, increment_retries = True, False, True
    else:
        # All other errors: retry with exponential backoff and increment the number of retries
        logger.error(f"Error during Hugging Face call: {e}. Retrying...")
        should_retry, apply_delay, increment_retries = True, True, True

    return e, should_retry, apply_delay, increment_retries


def handle_api_errors(e: OpenAIError | TestAPIError, *args, **kwargs) -> tuple[Exception, bool, bool, bool]:
    # By default, retry, apply exponential backoff, increment retries
    should_retry, apply_delay, increment_retries = True, True, True

    if isinstance(e, OpenAIError) and isinstance(e, BadRequestError):
        logger.error(f"BadRequestError during Hugging Face call: {e}. Stopping generation.")
        # Do not retry on bad requests.
        should_retry, apply_delay, increment_retries = False, False, False

    elif isinstance(e, OpenAIError) and isinstance(e, NotFoundError):
        logger.error(f"NotFoundError during Hugging Face call: {e}. Stopping generation.")
        # Do not retry on not found errors.
        should_retry, apply_delay, increment_retries = False, False, False
    else:
        # All other errors: retry with exponential backoff and increment the number of retries
        logger.error(f"Error during Hugging Face call: {e}. Retrying...")
        should_retry, apply_delay, increment_retries = True, True, True

    return e, should_retry, apply_delay, increment_retries


def handle_max_retries(
    e: Exception,
    *args,
    **kwargs,
) -> tuple[Exception, bool, bool, bool]:
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
        logger.info(f"Max number of retries for API key reached. Resetting client and retrying. Last error: {e}.")
        # Update retry count for the current API key
        client_manager = get_client_manager(client_manager_idx=0)
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
    base_delay=30,
    max_delay=MAX_DELAY,
    exp_base=2,
    jitter=True,
    max_retries=MAX_RETRIES,
    api_errors=(APIError, TestAPIError),
    custom_errors=(FutureTimeoutError, EmptyResponseError),
    handle_custom_errors=handle_custom_errors,
    handle_api_errors=handle_api_errors,
    handle_max_retries=handle_max_retries,
)


# If API call doesnt return in min(MAX_WAIT_PER_GEN * num_generations, MAX_API_WAIT_TIME) seconds, retry
# This should be passed to the `retry_exp_backoff` decorator (see `sync_api_call`)
def timeout_getter(args: Any, kwargs: Any, key: str = "provider_gen_config") -> float:
    provider_gen_config: Dict[str, Any] = kwargs.get(key)
    n = MAX_API_WAIT_TIME
    n = n or provider_gen_config.get("num_return_sequences", None) or provider_gen_config.get("n", None)
    return min(MAX_WAIT_PER_GEN * n, MAX_API_WAIT_TIME)


# ==============================================================================
# LINK: Output conversion: provider-specific -> uniform format
# ==============================================================================


def convert_generations(response: Any, engine: str) -> List[Message]:
    """
    Convert the provider-specific response to a list of Message objects.
    """
    if engine == "vllm" or engine == "openai":
        from llms.providers.openai.openai_utils import convert_generations as openai_convert_generations

        return openai_convert_generations(response)  # type: ignore
    else:
        if not isinstance(response, list):
            response = [response]

        converted_messages = []
        for msg in response:
            if isinstance(msg, str):
                msg = get_message(msg, role="assistant", name="")
            else:
                raise NotImplementedError(f"Not implemented output parsing for {type(msg)} yet.")
            converted_messages.append(msg)
        return converted_messages


# ==============================================================================
# LINK: Prompt messages conversion: uniform -> provider-specific format
# ==============================================================================


def get_provider_msgs(messages: List[Message], gen_config: GenerationConfig, use_cache: bool = True) -> List[Dict[str, Any]]:
    """
    Process the input messages:
      - Use OpenAIPrompter to convert the unified List[Message] to provider-specific format.
      - Reset the prompt or trigger image upload if needed.
    """
    global cache
    if use_cache:
        provider_msgs = cache.messages_to_provider
        if provider_msgs:
            return provider_msgs

    if gen_config.engine == "vllm":
        from llms.providers.openai.prompter import OpenAIPrompter

        provider_msgs = OpenAIPrompter.convert_prompt(messages, mode="chat_completion", gen_config=gen_config)
    else:
        provider_msgs = HuggingFacePrompter.convert_prompt(messages, gen_config)

    cache.messages_to_provider = provider_msgs
    return provider_msgs


# ===============================================================================
# LINK: Generation config conversion: uniform -> provider-specific format
# ===============================================================================
def get_default_params_vllm(model_path: str) -> Dict[str, Any]:
    default_params = {}
    for k in VLLM_DEFAULT_PARAMS_PER_MODEL:
        if k in model_path:
            default_params = VLLM_DEFAULT_PARAMS_PER_MODEL[k]
            break
    return default_params


def _gen_config_to_api_gen_args(gen_config: GenerationConfig) -> Dict[str, Any]:
    provider_gen_args = _gen_config_to_vllm_gen_args(gen_config)
    provider_gen_args["model"] = gen_config.model_path

    if gen_config.metadata.get("api_extra_args"):
        extra_args = gen_config.metadata["api_extra_args"]
        if "extra_body" not in provider_gen_args:
            provider_gen_args["extra_body"] = {}

        if isinstance(extra_args, dict):
            for k, v in extra_args.items():
                provider_gen_args["extra_body"][k] = v
        else:
            raise ValueError("api_extra_args must be a dictionary")

    if gen_config.extra_body:
        extra_args = gen_config.extra_body
        if "extra_body" not in provider_gen_args:
            provider_gen_args["extra_body"] = {}
        if isinstance(extra_args, dict):
            for k, v in extra_args.items():
                provider_gen_args["extra_body"][k] = v
        else:
            raise ValueError("api_extra_args must be a dictionary")
    return provider_gen_args


def _gen_config_to_vllm_gen_args(gen_config: GenerationConfig) -> Dict[str, Any]:
    """
    Convert the uniform generation configuration to VLLM format.
    """
    from llms.providers.openai.openai_utils import gen_config_to_provider as openai_gen_config_to_provider

    # Get gen args compatible with OpenAI client
    provider_gen_args = openai_gen_config_to_provider(gen_config)

    # Add extra parameters not supported by OpenAI client
    provider_gen_args["extra_body"] = {
        "top_k": gen_config.top_k,
    }
    if "top_p" in provider_gen_args and provider_gen_args["top_p"] == 0:
        # Set top_p to 0.01 to avoid errors
        provider_gen_args["top_p"] = 0.001

    # If repetition penalty not provided, override with model creator's recommendation if available
    # For some models, VLLM openai client sets undesired values. Override if recommended by model creator.
    if gen_config.frequency_penalty is not None:
        provider_gen_args["extra_body"]["repetition_penalty"] = gen_config.frequency_penalty  # type: ignore
    else:
        default_vllm_params = get_default_params_vllm(gen_config.model_path)
        if "repetition_penalty" in default_vllm_params and gen_config.frequency_penalty is None:
            provider_gen_args["extra_body"]["repetition_penalty"] = default_vllm_params["repetition_penalty"]

    if gen_config.presence_penalty is not None:
        provider_gen_args["extra_body"]["presence_penalty"] = gen_config.presence_penalty  # type: ignore
    else:
        default_vllm_params = get_default_params_vllm(gen_config.model_path)
        if "presence_penalty" in default_vllm_params and gen_config.presence_penalty is None:
            provider_gen_args["extra_body"]["presence_penalty"] = default_vllm_params["presence_penalty"]

    return provider_gen_args


def gen_config_to_provider(gen_config: GenerationConfig) -> Dict[str, Any]:
    """
    Convert the uniform generation configuration to HF format.
    """
    if not gen_config.mode:
        gen_config.mode = DEFAULT_HF_MODE

    if gen_config.engine == "vllm":
        return _gen_config_to_vllm_gen_args(gen_config)

    elif api_provider := gen_config.metadata.get("provider", ""):
        if gen_config.engine != "openai":
            raise NotImplementedError(f"API provider `{api_provider}` only supported with `openai` engine.")
        else:
            gen_kwargs = _gen_config_to_api_gen_args(gen_config)
            return gen_kwargs

    # else: default to hugging face generation

    # Hugging Face generation arguments
    do_sample = True

    # Determine if do_sample should stay True
    # If None, HF will use default so we set dummies to not change behavior
    temperature = 0.01 if gen_config.temperature is None else gen_config.temperature
    top_p = 0.01 if gen_config.top_p is None else gen_config.top_p
    top_k = 0.01 if gen_config.top_k is None else gen_config.top_k

    do_sample = do_sample and temperature != 0 and top_p != 0 and top_k != 0
    do_sample = do_sample and gen_config.do_sample

    # Fill in generation arguments
    provider_gen_args: Dict[str, Any] = {
        "num_return_sequences": gen_config.num_generations,
        "do_sample": do_sample,
    }
    if gen_config.max_tokens is not None:
        provider_gen_args["max_new_tokens"] = gen_config.max_tokens

    if do_sample:
        if gen_config.temperature is not None:
            provider_gen_args["temperature"] = gen_config.temperature
        if gen_config.top_p is not None:
            provider_gen_args["top_p"] = gen_config.top_p
        if gen_config.top_k is not None:
            provider_gen_args["top_k"] = gen_config.top_k

    if gen_config.frequency_penalty is not None:
        provider_gen_args["repetition_penalty"] = float(gen_config.frequency_penalty)
    return provider_gen_args


def regularize_provider_gen_config_for_model(
    model: str,
    provider_gen_config: Dict[str, Any],
) -> Dict[str, Any]:
    # Model specific regularization
    return provider_gen_config


def get_provider_gen_config(gen_config: GenerationConfig) -> Dict[str, Any]:
    """
    Get the generation configuration to be used in the API call.
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


def generate_from_huggingface(
    messages: List[Message],
    gen_config: GenerationConfig,
) -> tuple[List[dict[str, Any]], List[Message]]:
    """
    Synchronous generation from Hugging Face's API for both 'chat_completion' and 'response' modes.

    Both modes use a loop to repeatedly call the API until the requested number of generations are obtained.

    Returns:
        A tuple containing:
         - A list of raw API responses
         - A list of uniform Message objects generated from the responses.
    """
    global MAX_GENERATION_PER_BATCH, cache
    cache.reset()

    # Number of generations remaining
    remaining_generation_count = gen_config.num_generations

    # Build provider-specific configuration and messages.
    provider_gen_config = get_provider_gen_config(gen_config)

    if "do_sample" in provider_gen_config and not provider_gen_config["do_sample"]:
        # HF throws an error if do_sample is False and num_return_sequences > 1
        remaining_generation_count = 1
        logger.warning("'num_generations' > 1 but not sampling; setting num_generations to 1. Check the `temperature`, `top_p`, `top_k` and `do_sample` parameters.")

    logger.info(f"[{__file__}] CALLING MODEL: `{gen_config.model}` with engine `{gen_config.engine}`: generating {gen_config.num_generations} output(s)...")

    provider_messages = get_provider_msgs(messages, gen_config)

    while remaining_generation_count > 0:
        # Adjust batch size in the provider configuration.
        if "num_return_sequences" in provider_gen_config:
            provider_gen_config["num_return_sequences"] = min(MAX_GENERATION_PER_BATCH, remaining_generation_count)
        elif "n" in provider_gen_config:
            provider_gen_config["n"] = min(MAX_GENERATION_PER_BATCH, remaining_generation_count)

        # Regularize the provider configuration based on model and mode.
        provider_gen_config = regularize_provider_gen_config_for_model(gen_config.model, provider_gen_config)

        model_messages, response_dict = sync_call(
            provider_messages=provider_messages,
            provider_gen_config=provider_gen_config,
            gen_config=gen_config,
        )

        model_messages = convert_generations(model_messages, gen_config.engine)

        response_dict["gen_config"] = provider_gen_config
        response_dict["prompt"] = provider_messages
        if model_messages:
            cache.api_responses.append(response_dict)
            cache.model_messages.extend(model_messages)
            remaining_generation_count -= len(model_messages)
        else:
            logger.warning("No generations returned from API call; breaking out of loop.")
            break

    return cache.api_responses, cache.model_messages


def batch_generate_from_huggingface(
    messages_list: list[list[Message]],
    gen_config: GenerationConfig,
    dump_conversation_funs: list[Callable] | None = None,
    dump_usage_funs: list[Callable] | None = None,
    process_id: int = 0,
) -> tuple[list[dict[str, Any]], list[list[Message]]]:
    provider_gen_config = get_provider_gen_config(gen_config)

    if gen_config.engine == "openai":
        all_api_responses, all_model_messages = _batch_generate_from_openai(
            messages_list=messages_list,
            gen_config=gen_config,
            requests_per_minute=get_api_provider_arg(provider_gen_config.get("provider", "default"), "max_requests_per_minute"),
            dump_conversation_funs=dump_conversation_funs,
            dump_usage_funs=dump_usage_funs,
            process_id=process_id,
        )

    elif gen_config.engine == "automodel" or gen_config.engine == "server":
        all_provider_messages = [get_provider_msgs(message, gen_config, use_cache=False) for message in messages_list]

        if gen_config.engine == "automodel":
            _all_model_messages, _response_dict = _generate_from_automodel(all_provider_messages, provider_gen_config, gen_config)
        else:
            _all_model_messages, _response_dict = _generate_from_local_server(all_provider_messages, provider_gen_config, gen_config)

        # TODO: clean this up
        try:
            _all_api_responses = _batch_response_dict_to_list(_response_dict)
            if len(_all_api_responses) != len(_all_model_messages):
                _all_api_responses = [_response_dict] * len(_all_model_messages)

        except Exception as _:
            _all_api_responses = [_response_dict] * len(_all_model_messages)

        all_api_responses, all_model_messages = [], []
        for msgs, responses in zip(_all_model_messages, _all_api_responses):
            model_messages = convert_generations(msgs, gen_config.engine)

            if model_messages:
                all_model_messages.append(model_messages)
                all_api_responses.append(responses)
            else:
                logger.warning("No generations returned from API call; breaking out of loop.")
                break

    else:
        raise ValueError(f"Engine: {gen_config.engine} batch generation not supported yet.")

    return all_api_responses, all_model_messages


def sync_call(
    provider_messages: List[Dict[str, Any]],
    provider_gen_config: Dict[str, Any],
    gen_config: GenerationConfig,
) -> tuple[List[Any], Dict[str, Any]]:
    if gen_config.engine == "automodel":
        return _generate_from_automodel([provider_messages], provider_gen_config, gen_config)

    elif gen_config.engine == "vllm":
        return _generate_from_vllm(provider_messages, provider_gen_config, gen_config)  # type: ignore

    elif gen_config.engine == "server":
        return _generate_from_local_server([provider_messages], provider_gen_config, gen_config)

    elif gen_config.engine == "tgi":
        raise NotImplementedError("TGI support discontinued.")

    elif gen_config.engine == "openai":
        return _generate_from_openai(provider_messages, provider_gen_config, gen_config)
    else:
        raise ValueError(f"Unsupported mode: {gen_config.engine}")


def _generate_from_automodel(
    provider_messages: list[list[Dict[str, Any]]],
    gen_kwargs: Dict[str, Any],
    gen_config: GenerationConfig,
) -> tuple[List[Any], Dict[str, Any]]:
    # Get model
    client_manager = get_client_manager(gen_config.model_path)
    model = client_manager.get_model(gen_config, engine="automodel")

    # Build model inputs
    inputs = ModelProcessor.get_inputs(provider_messages, gen_config.model_path)

    # Generate output tokens
    gen_kwargs.update({"return_dict_in_generate": True, "output_scores": True})
    device = model.device
    if hasattr(inputs, "shape"):
        response = model.generate(inputs.to(device), **gen_kwargs)
    else:
        response = model.generate(**(inputs.to(device)), **gen_kwargs)

    # Index to trim the prompt from the output
    trim_prompt_idxs = get_trim_prompt_idxs(inputs, len(response.sequences))

    # Decode outputs
    natural_outputs = ModelProcessor.decode_outputs(response.sequences, gen_config.model_path, start_idxs=trim_prompt_idxs, skip_special_tokens=True)

    # Convert `response` to dict
    response_dict = dict(response)
    response_dict["usage"] = count_tokens(response, inputs)

    return natural_outputs, response_dict


def _generate_from_local_server(
    provider_messages: list[list[Dict[str, Any]]],
    gen_kwargs: Dict[str, Any],
    gen_config: GenerationConfig,
) -> tuple[List[Any], Dict[str, Any]]:
    # Get or launch local server
    _ = get_client_manager(gen_config.endpoint).get_model(gen_config, engine="server")

    # Send generation request to local server
    response = requests.post(gen_config.endpoint, json={"messages": provider_messages, "gen_kwargs": gen_kwargs})  # type:ignore
    response_dict = response.json()
    if "model_messages" not in response_dict:
        logger.warning(f"No model messages returned from local server: {response_dict}")
        return [], {}
    return response_dict["model_messages"], response_dict["api_response"]


def _generate_from_vllm(
    provider_messages: List[Dict[str, Any]],
    gen_kwargs: Dict[str, Any],
    gen_config: GenerationConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    # Get or launch vLLM server
    endpoint = get_client_manager(gen_config.endpoint).get_model(gen_config, engine="vllm")
    openai_client = get_client_manager(gen_config.endpoint).get_openai_client(gen_config.model_path, endpoint)

    # Send generation request to vLLM server
    response = openai_client.chat.completions.create(messages=provider_messages, **gen_kwargs)

    model_messages = {"choices": response.choices}
    response_dict = response.to_dict()
    return model_messages, response_dict


@retry_exp_backoff(timeout_getter=timeout_getter)
def _generate_from_openai(
    provider_messages: List[Dict[str, Any]],
    gen_kwargs: Dict[str, Any],
    gen_config: GenerationConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    # Generate using OpenAI client for providers that support it (e.g.: openrouter, alibaba)
    endpoint = get_client_manager(gen_config.endpoint).get_model(gen_config, engine="openai")
    openai_client = get_client_manager(gen_config.endpoint).get_openai_client(gen_config.model_path, endpoint)

    response: ChatCompletion = openai_client.chat.completions.create(messages=provider_messages, **gen_kwargs)

    if not hasattr(response, "choices") or not response.choices:
        raise EmptyResponseError(f"Empty response from provider `{gen_config.metadata['provider']}`.")

    model_messages = {"choices": response.choices}
    response_dict = response.to_dict()
    return model_messages, response_dict


async def _throttled_openai_agenerate(
    aclient: AsyncOpenAI,
    limiter: aiolimiter.AsyncLimiter,
    messages: List[Message],
    gen_config: GenerationConfig,
    provider_gen_config: Dict[str, Any],
    dump_conversation_fun=None,
    dump_usage_fun=None,
) -> ChatCompletion | dict[str, Any]:
    """Call OpenAI asynchronously with built-in retry logic and rate-limiting."""
    provider_msgs = get_provider_msgs(messages, gen_config, use_cache=False)
    throttled_sleep = get_api_provider_arg(gen_config.metadata.get("provider", "default"), "throttled_sleep")

    async with limiter:
        for _ in range(MAX_RETRIES):
            try:
                logger.info(f"async generation call for model {gen_config.model}...")
                resp = await aclient.chat.completions.create(messages=provider_msgs, **provider_gen_config)  # type: ignore
                logger.info(f"1 generation successful for model {gen_config.model}. Response: {resp}")
                if dump_conversation_fun or dump_usage_fun:
                    if dump_conversation_fun:
                        model_messages = convert_generations(resp, gen_config.engine)
                        if model_messages:
                            dump_conversation_fun(messages, model_messages, provider_gen_config)
                    if dump_usage_fun:
                        dump_usage_fun([resp.to_dict()], provider_gen_config)
                return resp
            except openai.RateLimitError as e:
                logger.warning(f"OpenAI API rate limit exceeded: {e}. Sleeping for {throttled_sleep} seconds.")
                await asyncio.sleep(throttled_sleep)
            except asyncio.exceptions.TimeoutError:
                logger.warning(f"OpenAI API timeout. Sleeping for {throttled_sleep} seconds.")
                await asyncio.sleep(throttled_sleep)
            except openai.APIError as e:
                if "timeout" in e.message.lower():
                    logger.warning(f"OpenAI API timeout. Sleeping for {throttled_sleep} seconds.")
                    await asyncio.sleep(throttled_sleep)
                elif "connection" in e.message.lower():
                    logger.warning(f"OpenAI API connection error. Sleeping for {throttled_sleep} seconds.")
                    await asyncio.sleep(throttled_sleep)
                else:
                    logger.warning(f"OpenAI API error: {e}", exc_info=True)
                    break
            except Exception as e:
                logger.error(f"Error in async generation: {e}", exc_info=True)
                break
        return_dict = {"failed": True, "request_kwargs": provider_gen_config}
        return return_dict


def _batch_generate_from_openai(
    messages_list: list[list[Message]],
    gen_config: GenerationConfig,
    requests_per_minute: int,
    dump_conversation_funs: list[Callable] | None = None,
    dump_usage_funs: list[Callable] | None = None,
    process_id: int = 0,
) -> tuple[list[dict[str, Any]], list[List[Message]]]:
    """
    Asynchronous generation from OpenAI's Chat Completion API.

    Args:
        prompt_batches: A list where each element is a list of messages to be sent to the API.
        gen_config: Generation configuration.
        requests_per_minute: Rate-limit for async requests.

    Returns:
        A tuple of:
          - List of raw JSON response objects.
          - List of generated text contents.
    """

    provider_gen_config = get_provider_gen_config(gen_config)

    if provider_gen_config.get("n", 1) > 1:
        provider_gen_config["n"] = 1
        logger.warning("Setting num_generations to 1 for batch async generation.")

    # Create a rate-limiter
    limiter = aiolimiter.AsyncLimiter(requests_per_minute)

    async def _async_generate() -> tuple[List[dict[str, Any]], List[List[Message]]]:
        # TODO: fix api key management when multiple keys
        hf_client_manager = get_client_manager(model_id=gen_config.endpoint, client_manager_idx=process_id)
        _ = hf_client_manager.get_model(gen_config, engine="openai", p_id=str(process_id))
        hf_client_manager.openai_client_manager.set_aclient()
        openai_aclient = hf_client_manager.openai_aclient
        api_key = openai_aclient.api_key
        logger.info(f"[Process ID: {process_id}] Using OpenAI async client with API KEY: {api_key[-5:] if api_key else None}...")
        tasks = [
            _throttled_openai_agenerate(
                aclient=openai_aclient,
                limiter=limiter,
                dump_conversation_fun=None if dump_conversation_funs is None else dump_conversation_funs[i],
                dump_usage_fun=None if dump_usage_funs is None else dump_usage_funs[i],
                messages=messages,
                gen_config=gen_config,
                provider_gen_config=provider_gen_config,
            )
            for i, messages in enumerate(messages_list)
        ]
        api_key = openai_aclient.api_key
        logger.info(f"[{__file__}] Generating {len(messages_list)} calls in batch mode for model {gen_config.model}. API KEY: {api_key[-5:] if api_key else None}")
        results: list[ChatCompletion | dict[str, Any]] = await asyncio.gather(*tasks)
        all_api_responses = []
        all_model_messages = []
        for response in results:
            if isinstance(response, dict) and response.get("failed", False):
                logger.warning(f"No generations returned for {response['request_kwargs']}")
                continue

            model_messages = convert_generations(response, engine=gen_config.engine)
            if not isinstance(response, dict):
                response_dict = response.to_dict()
            else:
                response_dict = response

            if model_messages:
                all_api_responses.append(response_dict)
                all_model_messages.extend(model_messages)
            else:
                logger.warning(f"No generations returned for {response_dict['request_kwargs']}")
                continue
        return all_api_responses, all_model_messages

    # Run the asyncio task synchronously.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_async_generate())
    finally:
        loop.close()


def _batch_response_dict_to_list(response_dict: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Convert a response dictionary with list values into a list of dictionaries,
    where each dictionary combines elements from each key's list by index.

    Given:
        {"usage": [data0, data1], "sequences": [data0, data1], "model": "gpt-3"}
    the output will be:
        [
            {"usage": data0, "sequences": data0, "model": "gpt-3"},
            {"usage": data1, "sequences": data1, "model": "gpt-3"}
        ]
    If a value is not a list, the same value is repeated in every dictionary.
    If any list is shorter than the maximum list length, missing entries are filled with None.
    """
    # Determine the maximum length among values that are lists.
    max_len = 0
    if "sequences" in response_dict:
        max_len = len(response_dict["sequences"])
    elif "usage" in response_dict:
        max_len = len(response_dict["usage"]["input_tokens"])
    else:
        raise ValueError("No sequences or usage found in response dict.")

    flat_response_dict = flatten_dict(response_dict)

    all_responses = []
    for i in range(max_len):
        item = {}
        for key, value in flat_response_dict.items():
            try:
                item[key] = value[i]

            except IndexError:
                item[key] = None

            except TypeError as _:
                item[key] = value

            except Exception as _:
                item[key] = None

        all_responses.append(item)

    return all_responses

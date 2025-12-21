import datetime
import gc
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, List, cast

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from core_utils.file_utils import flatten_dict
from core_utils.logger_utils import logger
from llms.constants import MODEL_REPO_PATH
from llms.generation_config import GenerationConfig, get_fields, make_generation_config
from llms.prompt_utils import (
    build_message,
    conversation_to_html,
    conversation_to_txt,
    flatten_generations,
    get_messages,
)
from llms.prompt_utils import visualize_prompt as visualize_prompt_utils
from llms.providers.anthropic.anthropic_utils import generate_from_anthropic
from llms.providers.google.google_utils import batch_generate_from_google, generate_from_google_chat_completion
from llms.providers.hugging_face.hf_utils import batch_generate_from_huggingface, generate_from_huggingface
from llms.providers.openai.openai_utils import batch_generate_from_openai, generate_from_openai
from llms.setup_utils import infer_provider, safe_count_api_keys
from llms.types import Message, ValidInputs

# ==============================================================================
# LINK High level functions
# ==============================================================================


# Main user facing function for LLM calling
def call_llm(
    gen_kwargs: dict[str, Any],
    prompt: ValidInputs | List[Message],
    meta_data: dict[str, Any] = {},
    conversation_dir: str = "",
    usage_dir: str = "",
    call_id: str = "",
    verbose: bool = False,
    dump_html: bool = True,
    dump_txt: bool = True,
    ovewrite_txt: bool = False,
) -> tuple[list[dict[str, Any]], List[Message]]:
    """
    Call an LLM.

    Args:
        gen_kwargs (dict[str, Any]): The generation arguments.
        prompt (List[ImageInput | Dict[str, Any] | Message | List[Message]]): The prompt to send to the LLM.
        meta_data (dict[str, Any], optional): Optional arguments for additional behavior.
        conversation_dir (str, optional): Directory path to store conversation logs.
            e.g. "results/experiment_1"
        usage_dir (str, optional): Directory path to store usage logs.
            e.g. "results/experiment_1"
        call_id (str, optional): Unique identifier for the LLM call; used to name the files stored in `conversation_dir` and `usage_dir`.
            e.g. "task_0" => `results/experiment_1/task_0.html`, `results/experiment_1/task_0.txt`, `results/experiment_1/task_0.csv`.
        dump_html (bool, optional): If true, creates an HTML file to store the conversation.
        dump_txt (bool, optional): If true, creates a TXT file to store the conversation.
        verbose (bool, optional): If true, prints verbose output.

    Returns:
        tuple[list[Any], list[Message]]: A tuple of the API response and the model generations.
    """
    if call_id:
        call_id = str(call_id)
    elif conversation_dir or usage_dir:
        call_id = get_call_id(gen_kwargs)

    if not isinstance(prompt, list):
        prompt = [prompt]  # type: ignore

    prompt_uniform_format = get_messages(inputs=prompt)  # type:ignore

    # Regularize/add arguments for the LLM call
    reg_gen_kwargs = _regularize_gen_kwargs(gen_kwargs)

    # If manual input signal given, get input from user
    if meta_data.get("manual_input", False):
        model_generation = _get_manual_input(prompt_uniform_format, meta_data, conversation_dir, reg_gen_kwargs, verbose, dump_html, dump_txt)
        if not model_generation[0].contents[0].data == "llm":
            return [], model_generation

    # Get generation config
    gen_config = make_generation_config(reg_gen_kwargs)

    # Generate from LLM
    api_responses, model_generations = _generate_from_llm(
        messages=prompt_uniform_format,
        gen_config=gen_config,
        provider=reg_gen_kwargs["provider"],
        meta_data=meta_data,
    )
    if len(api_responses) == 0 or len(model_generations) == 0:
        print(f"No API responses or model generations returned from LLM call. Returning empty lists.")
        return [], []

    # if conversation_dir == usage_dir:
    #     conversation_dir = f"{conversation_dir}/conversation"
    #     usage_dir = f"{usage_dir}/usage"

    if conversation_dir:
        dump_llm_output(
            prompt_uniform_format,
            model_generations,
            conversation_dir,
            call_id,
            reg_gen_kwargs,
            verbose,
            dump_html,
            dump_txt,
            ovewrite_txt,
        )

    if usage_dir:
        dump_usage(reg_gen_kwargs["provider"], api_responses, usage_dir, call_id, reg_gen_kwargs)

    return api_responses, model_generations


def batch_call_llm(
    gen_kwargs: dict[str, Any],
    prompts: list[ValidInputs] | list[list[Message]],
    conversation_dirs: list[str | None] | None = None,
    usage_dirs: list[str | None] | None = None,
    call_ids: list[str | None] | None = None,
    verbose: bool = False,
    dump_html: bool = True,
    dump_txt: bool = True,
    max_batch_size: int = 10,
    num_workers: int = 2,
    return_outputs: bool = True,
    multiprocess_mode: bool = False,
    max_api_keys: int = 0,
    order_by_payload_size: bool = False,
    ovewrite_txt: bool = False,
) -> tuple[list[dict[str, Any]], list[list[Message]]]:
    """
    Batch call an LLM.

    Args:
        gen_kwargs (dict[str, Any]): The generation arguments.
        prompts (list[ValidInputs | List[Message]]): The prompts to send to the LLM.
        conversation_dirs (list[str] | None, optional): The directories to save the conversation logs.
        usage_dirs (list[str] | None, optional): The directories to save the usage logs.
        call_ids (list[str] | None, optional): The unique identifiers for each call.
        max_batch_size (int, optional): Maximum batch size per worker at each iteration.
        num_workers (int, optional): Number of workers for parallel processing.
        return_outputs (bool, optional): If true, return the API responses and model generations. Use false if to save memory if only dumping outputs.
        verbose (bool, optional): If true, print verbose output.
        dump_html (bool, optional): If true, dump the conversation to an HTML file.
        dump_txt (bool, optional): If true, dump the conversation to a TXT file.

    Returns:
        tuple[list[Any], list[Message]]: A tuple of the API responses and the model generations.
    """

    # Argument validation
    if call_ids:
        if len(prompts) != len(call_ids):
            raise ValueError("prompts and call_ids must be the same length")
        call_ids = [str(call_id) for call_id in call_ids]

    if conversation_dirs and len(prompts) != len(conversation_dirs):
        raise ValueError("prompts and conversation_dirs must be the same length")

    if usage_dirs and len(prompts) != len(usage_dirs):
        raise ValueError("prompts and usage_dirs must be the same length")

    if not call_ids and (conversation_dirs or usage_dirs):
        call_ids = [get_call_id(gen_kwargs) for _ in prompts]

    if n := gen_kwargs.get("num_generations", None) is not None:
        if int(n) > 1:
            raise NotImplementedError("Batch generation with num_generations > 1 is not implemented yet.")

    # Regularize arguments for the LLM call and get generation config
    reg_gen_kwargs = _regularize_gen_kwargs(gen_kwargs)
    gen_config = make_generation_config(reg_gen_kwargs)

    if not multiprocess_mode:
        reg_prompts, conversation_dirs, usage_dirs, call_ids = _batch_regularize_prompts(
            prompts=prompts,
            conversation_dirs=conversation_dirs,
            usage_dirs=usage_dirs,
            call_ids=call_ids,
            order_by_payload_size=order_by_payload_size,
        )  # type:ignore
    else:
        reg_prompts = prompts

    if multiprocess_mode:
        all_api_responses, all_model_generations = _batch_generate_multiprocess(
            prompts=reg_prompts,  # type: ignore
            conversation_dirs=conversation_dirs,
            usage_dirs=usage_dirs,
            call_ids=call_ids,
            gen_config=gen_config,
            num_workers=num_workers,
            verbose=verbose,
            dump_html=dump_html,
            dump_txt=dump_txt,
            return_outputs=return_outputs,
            max_batch_size=max_batch_size,
            ovewrite_txt=ovewrite_txt,
        )
    else:
        all_api_responses, all_model_generations = _batch_generate_from_provider(
            prompts=reg_prompts,  # type: ignore
            conversation_dirs=conversation_dirs,
            usage_dirs=usage_dirs,
            call_ids=call_ids,
            gen_config=gen_config,
            verbose=verbose,
            dump_html=dump_html,
            dump_txt=dump_txt,
            return_outputs=return_outputs,
            max_batch_size=max_batch_size,
            max_api_keys=max_api_keys,
            ovewrite_txt=ovewrite_txt,
        )
    return all_api_responses, all_model_generations


def visualize_prompt(messages: ValidInputs | List[Message], output_path: str | Path = "", verbose: bool = False) -> None:
    if not output_path:
        output_path = f"./{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_messages.html"
    visualize_prompt_utils(messages, output_path, verbose)


def get_gen_config_fields() -> list[str]:
    gen_fields = get_fields()
    return gen_fields


def get_payload_size(messages: ValidInputs | list[Message] | list[ValidInputs] | list[list[Message]]) -> int:
    reg_prompt = get_messages(inputs=messages)  # type: ignore[arg-type]
    return sum(message.payload_size or 0 for message in reg_prompt)


# ==============================================================================
# LINK LLM call - low level functions
# ==============================================================================


def _generate_from_llm(
    messages: List[Message],
    gen_config: GenerationConfig,
    provider: str,
    meta_data: dict[str, Any] = {},
) -> tuple[List[Any], List[Message]]:
    if provider == "openai":
        api_responses, model_generations = generate_from_openai(
            messages=messages,
            gen_config=gen_config,
        )
    elif provider == "google":
        api_responses, model_generations = generate_from_google_chat_completion(
            messages=messages,
            gen_config=gen_config,
            meta_data=meta_data,
        )
    elif provider == "huggingface":
        api_responses, model_generations = generate_from_huggingface(
            messages=messages,
            gen_config=gen_config,
        )
    elif provider == "anthropic":
        api_responses, model_generations = generate_from_anthropic(
            messages=messages,
            gen_config=gen_config,
        )
    else:
        raise NotImplementedError(f"Provider {provider} not implemented")
    return api_responses, model_generations


def _get_manual_input(
    prompt: List[Message],
    meta_data: dict[str, Any],
    conversation_dir: str,
    gen_kwargs: dict[str, Any],
    verbose: bool = False,
    dump_html: bool = True,
    dump_txt: bool = True,
) -> List[Message]:
    # Generate a call id to save logs
    call_id = f"manual_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if conversation_dir:
        pass
        # Save a firtst version of the conversation to help in the manual input
        # dump_llm_output(prompt, "", conversation_dir, call_id, gen_kwargs, verbose, dump_html, dump_txt)

    # Save an image obs if available, to help in the manual input
    if "trajectory" in meta_data:
        # Save the last observation as an image with text overlay
        img = Image.fromarray(meta_data["trajectory"].states[-1]["observation"]["image"])
        txt = meta_data["trajectory"].states[-1]["observation"]["text"]
        # Save HTML file
        visualize_prompt(build_message(contents=[txt, img], role="user", name="manual_input"), "observation.html", False)

    # Get the manual input from the user
    print("Enter text followed by 'ctrl+d'. If enter 'llm' followed by 'ctrl+d', calls LLM.", end="", flush=True)
    utterance = sys.stdin.read()
    model_generations = build_message(contents=[utterance], role="assistant", name="manual_input")

    # If the user inputs "llm", then the next utterance will come from an LLM call
    # If not "llm", then dump conversation and return utterance
    if not utterance == "llm" and conversation_dir:
        dump_llm_output(prompt, [model_generations], conversation_dir, call_id, gen_kwargs, verbose, dump_html, dump_txt)
    return [model_generations]


# ==============================================================================
# LINK Batch generation - low level functions
# ==============================================================================


def _batch_regularize_prompts(
    prompts: list[ValidInputs] | list[list[Message]],
    conversation_dirs: list[str | None] | None = None,
    usage_dirs: list[str | None] | None = None,
    call_ids: list[str | None] | None = None,
    order_by_payload_size: bool = False,
    descending: bool = False,
) -> list[list[Message]] | tuple[list[list[Message]], list[str | None], list[str | None], list[str | None]]:
    """
    Regularize the prompts and optionally sort them along with their associated directories and call_ids.

    Args:
        prompts (list): A list of prompts, where a prompt can be a ValidInputs or list of Messages.
    conversation_dirs (list[str | None], optional): List of conversation directory paths; use None per-item to skip dumping.
    usage_dirs (list[str | None], optional): List of usage directory paths; use None per-item to skip dumping.
    call_ids (list[str | None], optional): List of call ids.
        order_by_payload_size (bool): If true, sort the regularized prompts by the total payload size.
        descending (bool): If true, sort in descending order.

    Returns:
        If conversation_dirs, usage_dirs, or call_ids are provided, returns a tuple:
            (regularized_prompts, sorted_conversation_dirs, sorted_usage_dirs, sorted_call_ids).
        Otherwise, returns a list of regularized prompts.

    Sorting is done by computing, for each prompt, the sum:
        sum(msg.payload_size or 0 for msg in prompt)
    """
    regularized_prompts = []
    for prompt in prompts:
        if not isinstance(prompt, list):
            prompt = [prompt]
        regularized_prompts.append(get_messages(prompt))  # type: ignore

    # Normalize aux lists so downstream indexing & zip() are always safe.
    # Per-item None means "skip dumping" for that prompt.

    if not conversation_dirs:
        conversation_dirs = cast(list[str | None], [None] * len(prompts))
    if not usage_dirs:
        usage_dirs = cast(list[str | None], [None] * len(prompts))
    if call_ids is None:
        call_ids = cast(list[str | None], [None] * len(prompts))

    if order_by_payload_size:
        # Zip the regularized prompts together with the aux lists
        zipped = list(zip(regularized_prompts, conversation_dirs, usage_dirs, call_ids))
        # Define the sort key as the sum of payload_size for each message in the prompt.
        zipped.sort(key=lambda tup: sum((msg.payload_size or 0) for msg in tup[0]), reverse=descending)
        reg_prompts_sorted, conv_dirs_sorted, use_dirs_sorted, c_ids_sorted = zip(*zipped)
        regularized_prompts = list(reg_prompts_sorted)
        conversation_dirs = list(conv_dirs_sorted)
        usage_dirs = list(use_dirs_sorted)
        call_ids = list(c_ids_sorted)
        return regularized_prompts, conversation_dirs, usage_dirs, call_ids

    return regularized_prompts, conversation_dirs, usage_dirs, call_ids


def _process_batch(
    prompts: list[ValidInputs] | list[List[Message]],
    gen_config: GenerationConfig,
    conversation_dirs: list[str | None] | None = None,
    usage_dirs: list[str | None] | None = None,
    call_ids: list[str | None] | None = None,
    verbose: bool = False,
    dump_html: bool = True,
    dump_txt: bool = True,
    return_outputs: bool = True,
    meta_data: dict[str, Any] = {},
) -> tuple[list[Any], list[Message]]:
    import builtins
    import functools

    # fmt: off
    builtins.print = functools.partial(builtins.print, flush=True)
    batch_api_responses = []
    batch_model_generations = []

    print(f"Batch LLM call [{os.getpid()}]: Processing batch of size {len(prompts)}")

    # For each conversation in the batch, generate from LLM and dump outputs.
    for i, prompt in enumerate(prompts):
        try:
            if not isinstance(prompt, list):
                prompt = [prompt]
            prompt_uniform_format = get_messages(inputs=prompt)  # type: ignore

            api_response, model_generations = _generate_from_llm(
                messages=prompt_uniform_format,
                gen_config=gen_config,
                provider=gen_config.provider,
                meta_data=meta_data,
            )
            
            if conversation_dirs:
                dump_llm_output(prompt_uniform_format, model_generations, conversation_dirs[i], call_ids[i], gen_config.to_dict(), verbose, dump_html, dump_txt)  # type: ignore
            if usage_dirs:
                dump_usage(gen_config.provider, api_response, usage_dirs[i], call_ids[i], gen_config.to_dict(), verbose)  # type: ignore

            # If no need to return outputs, skip accummulation to save memory.
            if return_outputs:
                batch_api_responses.append(api_response)
                batch_model_generations.append(model_generations)
        except Exception as e:
            logger.warning(f"Error during batch generation: {e}")
            continue
    # fmt: on
    return batch_api_responses, batch_model_generations


def _compute_batches(
    prompts: list[ValidInputs] | list[List[Message]],
    max_batch_size: int = 10,
    conversation_dirs: list[str | None] | None = None,
    usage_dirs: list[str | None] | None = None,
    call_ids: list[str | None] | None = None,
    dump_html: bool = True,
    dump_txt: bool = True,
    verbose: bool = False,
    add_dumping_funs: bool = False,
    gen_config: GenerationConfig | None = None,
    ovewrite_txt: bool = False,
) -> list[dict[str, Any]]:
    """
    Returns a list of dictionaries, each containing the data for a batch of prompts to send to the LLM.
    Example: output:
    output[0] = {
        "prompts": [prompts[0], prompts[1]],
        "conversation_dirs": [conversation_dirs[0], conversation_dirs[1]],
        "usage_dirs": [usage_dirs[0], usage_dirs[1]],
        "call_ids": [call_ids[0], call_ids[1]],
    }
    """
    # Calculate the number of batches; each batch will have at most max_batch_size items.
    total_tasks = len(prompts)

    if max_batch_size > 0:
        num_batches = (total_tasks + max_batch_size - 1) // max_batch_size
    else:
        num_batches = 1

    # Create batches: each sublist is at most max_batch_size long.
    batch_idxs = np.array_split(range(total_tasks), num_batches)

    all_job_data = []
    for j, batch_idx in enumerate(batch_idxs):
        job_data = {}
        job_data["batch_id"] = j
        job_data["prompts"] = [prompts[i] for i in batch_idx.tolist()]

        job_data["conversation_dirs"] = None
        if conversation_dirs:
            job_data["conversation_dirs"] = [conversation_dirs[i] for i in batch_idx.tolist()]

        job_data["usage_dirs"] = None
        if usage_dirs:
            job_data["usage_dirs"] = [usage_dirs[i] for i in batch_idx.tolist()]

        job_data["call_ids"] = None
        if call_ids:
            job_data["call_ids"] = [call_ids[i] for i in batch_idx.tolist()]

        if add_dumping_funs and call_ids and gen_config:
            # Create a wrapper for dumping conversations
            def create_dump_llm_output_wrapper(conv_dir, call_id, verbose, dump_html, dump_txt):
                def wrapper(messages, model_generations, gen_kwargs):
                    return dump_llm_output(
                        messages=messages,
                        model_generations=model_generations,
                        conversation_dir=conv_dir,
                        call_id=call_id,
                        gen_kwargs=gen_kwargs,
                        verbose=verbose,
                        dump_html=dump_html,
                        dump_txt=dump_txt,
                        ovewrite_txt=ovewrite_txt,
                    )

                return wrapper

            # Similarly, create a wrapper for dumping usage
            def create_dump_usage_wrapper(usage_dir, call_id, provider, verbose):
                def wrapper(api_responses, gen_kwargs):
                    return dump_usage(
                        provider=provider,
                        api_responses=api_responses,
                        usage_dir=usage_dir,
                        call_id=call_id,
                        gen_kwargs=gen_kwargs,
                        verbose=verbose,
                    )

                return wrapper

            if conversation_dirs:
                job_data["dump_conversation_funs"] = [
                    create_dump_llm_output_wrapper(
                        conversation_dirs[i],
                        call_ids[i],
                        verbose,
                        dump_html,
                        dump_txt,
                    )
                    for i in batch_idx.tolist()
                ]

            if usage_dirs:
                job_data["dump_usage_funs"] = [
                    create_dump_usage_wrapper(
                        usage_dirs[i],
                        call_ids[i],
                        gen_config.provider,
                        verbose,
                    )
                    for i in batch_idx.tolist()
                ]

        all_job_data.append(job_data)

    return all_job_data


def _batch_generate_multiprocess(
    # TODO: deprecate this
    prompts: list[ValidInputs] | list[list[Message]],
    gen_config: GenerationConfig,
    conversation_dirs: list[str | None] | None = None,
    usage_dirs: list[str | None] | None = None,
    call_ids: list[str | None] | None = None,
    num_workers: int = 2,
    verbose: bool = False,
    dump_html: bool = True,
    dump_txt: bool = True,
    return_outputs: bool = True,
    max_batch_size: int = 10,
    ovewrite_txt: bool = False,
) -> tuple[list[dict[str, Any]], list[list[Message]]]:
    """
    Process a batch call for Google LLM in parallel.

    This function splits the messages_list into batches of size at most max_batch_size,
    then uses Joblib to process those batches using num_workers in parallel.

    Args:
        messages_list (list[List[Message]]): A list where each element is a conversation (a list of Message).
        gen_config (GenerationConfig): The generation configuration.
        max_batch_size (int, optional): Maximum number of messages to process per iteration.
        num_workers (int, optional): Number of workers for parallel processing.
        return_outputs (bool, optional): If true, return the API responses and model generations. Use false if to save memory if only dumping outputs.
    Returns:
        tuple[list[list[Any]], list[list[Message]]]: Tuple of all API responses and all model generations.
    """
    if not prompts:
        logger.warning("No prompts provided to batch generation.")
        return [], []

    from joblib import Parallel, delayed

    all_jobs_data = _compute_batches(
        prompts=prompts,
        conversation_dirs=conversation_dirs,
        usage_dirs=usage_dirs,
        call_ids=call_ids,
        max_batch_size=max_batch_size,
        ovewrite_txt=ovewrite_txt,
    )

    if len(all_jobs_data) < num_workers:
        num_workers = len(all_jobs_data)

    # Why joblib: needs isolated environments for some functionalities in providers code (e.g.: isolated clients, worker pools for timeout)
    results = Parallel(n_jobs=num_workers)(
        delayed(_process_batch)(
            **job_data,
            gen_config=gen_config,
            verbose=verbose,
            dump_html=dump_html,
            dump_txt=dump_txt,
            return_outputs=return_outputs,
        )
        for job_data in all_jobs_data
    )
    all_api_responses, all_model_generations = [], []
    # Flatten the results from all batches.
    if results is not None and return_outputs:
        for responses, generations in results:  # type: ignore
            all_api_responses.extend(responses)
            all_model_generations.extend(generations)
    else:
        # Try to release memory
        del results
        gc.collect()
    return all_api_responses, all_model_generations


def _batch_generate_from_provider(
    prompts: list[list[Message]],
    gen_config: GenerationConfig,
    conversation_dirs: list[str | None] | None = None,
    usage_dirs: list[str | None] | None = None,
    call_ids: list[str | None] | None = None,
    verbose: bool = False,
    dump_html: bool = True,
    dump_txt: bool = True,
    max_batch_size: int = 10,
    return_outputs: bool = True,
    max_api_keys: int = 1,
    ovewrite_txt: bool = False,
) -> tuple[list[dict[str, Any]], list[list[Message]]]:
    """
    Batch generate from provider.
    """
    all_api_responses, all_model_generations = [], []
    gen_config_dict = gen_config.to_dict()

    if gen_config.provider == "openai" or (gen_config.engine == "openai") or gen_config.provider == "google":
        add_dumping_funs = True
    else:
        add_dumping_funs = False

    resolved_provider = gen_config.metadata.get("provider", "") or gen_config.provider
    gen_config_dict = gen_config.to_dict()

    if max_api_keys <= 0:
        num_keys = safe_count_api_keys(provider=resolved_provider)
        max_api_keys = max(num_keys, 1)

    if max_batch_size <= 0:
        max_batch_size = max(1, len(prompts) // max_api_keys)

    logger.info(f"Using {max_api_keys} api keys in batch generation")
    payload_in_batches = _compute_batches(
        prompts=prompts,
        conversation_dirs=conversation_dirs,
        usage_dirs=usage_dirs,
        call_ids=call_ids,
        max_batch_size=max_batch_size,
        dump_html=dump_html,
        dump_txt=dump_txt,
        verbose=verbose,
        add_dumping_funs=add_dumping_funs,
        gen_config=gen_config,
        ovewrite_txt=ovewrite_txt,
    )

    if gen_config.provider == "google":
        # TODO: move to _route_batch_generation
        max_workers = max(1, max_api_keys)
        # One thread per batch running concurrently
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit each prompt batch to be processed concurrently
            futures = [
                executor.submit(
                    batch_generate_from_google,
                    messages_list=batch_data["prompts"],
                    gen_config=gen_config,
                    dump_conversation_funs=batch_data.get("dump_conversation_funs", None),
                    dump_usage_funs=batch_data.get("dump_usage_funs", None),
                    process_id=batch_data["batch_id"],
                )
                for batch_data in payload_in_batches
            ]
            # As each future finishes, extract its results and extend the overall lists
            for future in as_completed(futures):
                if return_outputs:
                    api_responses, model_generations = future.result()
                    all_api_responses.extend(api_responses)
                    all_model_generations.extend(model_generations)
                else:
                    future.result()
                gc.collect()

    elif gen_config.provider == "huggingface":
        # TODO: move to _route_batch_generation
        max_workers = max(1, max_api_keys)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit each prompt batch to be processed concurrently
            futures = [
                executor.submit(
                    batch_generate_from_huggingface,
                    messages_list=batch_data["prompts"],
                    gen_config=gen_config,
                    dump_conversation_funs=batch_data.get("dump_conversation_funs", None),
                    dump_usage_funs=batch_data.get("dump_usage_funs", None),
                    process_id=batch_data["batch_id"],
                )
                for batch_data in payload_in_batches
            ]
            # As each future finishes, extract its results and extend the overall lists
            for future in as_completed(futures):
                if return_outputs:
                    api_responses, model_generations = future.result()
                    all_api_responses.extend(api_responses)
                    all_model_generations.extend(model_generations)
                else:
                    future.result()
                gc.collect()

    else:
        for batch_data in payload_in_batches:
            batch_api_responses, batch_model_generations = _route_batch_generation(
                batch_data=batch_data,
                gen_config=gen_config,
                provider=gen_config.provider,
                return_outputs=return_outputs,
            )
            if return_outputs:
                all_api_responses.extend(batch_api_responses)
                all_model_generations.extend(batch_model_generations)

    if not add_dumping_funs:
        _dump_batch_outputs(
            prompts=batch_data["prompts"],
            all_api_responses=batch_api_responses,
            all_model_generations=batch_model_generations,
            conversation_dirs=batch_data["conversation_dirs"],
            usage_dirs=batch_data["usage_dirs"],
            call_ids=batch_data["call_ids"],
            gen_config=gen_config_dict,
            verbose=verbose,
            dump_html=dump_html,
            dump_txt=dump_txt,
            ovewrite_txt=ovewrite_txt,
        )
        # fmt: on
    return all_api_responses, all_model_generations


def _route_batch_generation(
    batch_data: dict[str, Any],
    gen_config: GenerationConfig,
    provider: str,
    return_outputs: bool = True,
) -> tuple[list[dict[str, Any]], list[list[Message]]]:
    if provider == "huggingface":
        return batch_generate_from_huggingface(
            batch_data["prompts"],
            gen_config,
            dump_conversation_funs=batch_data.get("dump_conversation_funs", None),
            dump_usage_funs=batch_data.get("dump_usage_funs", None),
        )

    elif provider == "openai":
        return batch_generate_from_openai(
            batch_data["prompts"],
            gen_config,
            dump_conversation_funs=batch_data.get("dump_conversation_funs", None),
            dump_usage_funs=batch_data.get("dump_usage_funs", None),
        )

    elif provider == "google":
        raise NotImplementedError("TODO: move batch from _batch_generate_from_provider to here")

    else:
        raise NotImplementedError(f"Provider {provider} not implemented")

    # Batch generation for other cases


# ==============================================================================
# LINK Get data per provider or model
# ==============================================================================


def _regularize_gen_kwargs(gen_kwargs: dict[str, Any], inplace: bool = False) -> dict[str, Any]:
    if not inplace:
        gen_kwargs = gen_kwargs.copy()

    if "model" not in gen_kwargs:
        raise ValueError("Please specify a model in gen_kwargs. Run get_avail_models() to see available models.")

    model = gen_kwargs["model"]

    with open(MODEL_REPO_PATH, "r") as file:
        model_config = yaml.safe_load(file)  # type: ignore
        if model not in model_config["models"]:
            model_config = {}
        else:
            model_config = model_config["models"][model]

    # If provider not specified, infer from model
    if "provider" not in gen_kwargs:
        provider = model_config.get("provider", None) or infer_provider(gen_kwargs["model"])
        if provider is None:
            raise ValueError("Unable to infer provider from model name. Please specify a provider in gen_kwargs or add a provider to the model repository file.")
        gen_kwargs["provider"] = provider

    # If `mode` not specified, optionally infer from model (`mode` not needed for all providers)
    mode = gen_kwargs.get("mode", None) or model_config.get("mode", None)
    gen_kwargs["mode"] = mode

    model_path = gen_kwargs.get("model_path", None) or model_config.get("model_path", None) or model
    gen_kwargs["model_path"] = model_path

    # Reg parameters dependent on  `engine`
    if engine := gen_kwargs.get("engine", None):
        if re.search(r"server", engine, flags=re.IGNORECASE):
            endpoint = gen_kwargs.get("endpoint", None)
            if not endpoint:
                raise ValueError("Please specify an endpoint in gen_kwargs when hosting a model on a server.")
            gen_kwargs["endpoint"] = endpoint
            # Extract host and port from endpoint
            match = re.search(r"^(?:https?://)?([^:/]+):(\d+)", endpoint)
            if match and len(match.groups()) == 2:
                host = match.group(1)
                port = match.group(2)
                gen_kwargs["endpoint"] = f"http://{host}:{port}/generate"
            else:
                raise ValueError("Invalid endpoint format. Please use the format 'host:port'.")
    return gen_kwargs


# TODO: clean this up: redefine normalized names, add img output tokens
def _get_usage(provider: str, gen_kwargs: dict[str, Any], api_response: dict[str, Any]) -> dict[str, Any]:
    try:
        if provider == "openai":
            usage = flatten_dict(api_response["usage"])

            reasoning_tokens = usage.get("completion_tokens_details:reasoning_tokens", None) or usage.get("output_tokens_details:reasoning_tokens", None) or float("nan")
            full_out_tokens = usage.get("completion_tokens", None) or usage.get("output_tokens", None) or float("nan")
            usage_normalized = {
                "_input_tokens": usage.get("prompt_tokens", None) or usage.get("input_tokens", None) or float("nan"),
                "_completion_tokens": full_out_tokens - reasoning_tokens,
                "_reasoning_tokens": reasoning_tokens,
            }
            usage.update(usage_normalized)

        elif provider == "google":
            usage = flatten_dict(api_response["usage_metadata"])
            usage_normalized = {}
            if "prompt_token_count" in usage:
                usage_normalized["_input_tokens"] = usage["prompt_token_count"]
            if "candidates_token_count" in usage:
                usage_normalized["_completion_tokens"] = usage["candidates_token_count"]
            if "thoughts_token_count" in usage:
                usage_normalized["_reasoning_tokens"] = usage["thoughts_token_count"]
            if "total_token_count" in usage:
                usage_normalized["_total_tokens"] = usage["total_token_count"]

            if "prompt_tokens_details" in usage:
                for item in usage["prompt_tokens_details"]:
                    modality = item.get("modality", None)
                    if not modality:
                        continue
                    token_count = item.get("token_count", None)
                    if not token_count:
                        continue
                    if modality.lower() == "text":
                        usage_normalized["_input_tokens:text"] = token_count
                    elif modality.lower() == "image":
                        usage_normalized["_input_tokens:image"] = item["token_count"]
                    else:
                        continue
            if usage.get("tool_use_prompt_token_count", None):
                usage_normalized["_tool_use_prompt_tokens"] = usage["tool_use_prompt_token_count"]
            usage.update(usage_normalized)
        elif provider == "huggingface":
            if "usage" in api_response:
                usage = flatten_dict(api_response["usage"])
            elif "usage:input_tokens" in api_response:
                usage = {k: v for k, v in api_response.items() if "usage" in k}
            else:
                usage = {}
            if gen_kwargs.get("engine", "") == "openai":
                reasoning_tokens = usage.get("completion_tokens_details:reasoning_tokens", 0)
                completion_tokens = usage["completion_tokens"] - reasoning_tokens
                usage_normalized = {
                    "_input_tokens": usage["prompt_tokens"],
                    "_completion_tokens": completion_tokens,
                    "_reasoning_tokens": usage.get("completion_tokens_details:reasoning_tokens", None),
                    "_total_tokens": reasoning_tokens + completion_tokens + usage["prompt_tokens"],
                }

        elif provider == "anthropic":
            usage = api_response["usage"]
            usage_normalized = {
                "_input_tokens": usage["input_tokens"],
                "_completion_tokens": usage["output_tokens"],
                "_reasoning_tokens": usage.get("completion_tokens_details:reasoning_tokens", None),
                "_output_tokens": usage["completion_tokens"],
            }
            usage.update(usage_normalized)
        else:
            raise NotImplementedError(f"Provider {provider} not implemented")

    except Exception as e:
        print(f"Warning: Error getting usage for provider {provider}: {e}")
        return {}

    return usage


# ==============================================================================
# LINK Dumping output helpers
# ==============================================================================


def get_call_id(gen_kwargs: dict[str, Any]) -> str:
    model = gen_kwargs["model"]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model}_{timestamp}"


def dump_llm_output(
    messages: list[Message],
    model_generations: list[Message],
    conversation_dir: str | Path,
    call_id: str,
    gen_kwargs: dict[str, Any],
    verbose: bool = False,
    dump_html: bool = True,
    dump_txt: bool = True,
    ovewrite_txt: bool = False,
) -> None:
    try:
        if not isinstance(model_generations, list):
            model_generations = [model_generations]

        if dump_html:
            out_path = Path(conversation_dir) / f"{call_id}.html"
            dump_to_html(messages, model_generations, out_path, gen_kwargs, verbose=verbose)
        if dump_txt:
            out_path = Path(conversation_dir) / f"{call_id}.txt"
            if ovewrite_txt and out_path.exists():
                out_path.unlink()
            dump_to_txt(messages, model_generations, out_path, gen_kwargs, verbose=verbose)
    except Exception as e:
        logger.error(f"Error dumping LLM output associated to {conversation_dir}, call_id: {call_id}: {e}")


def dump_to_txt(
    messages: list[Message],
    model_generations: List[Message],
    output_path: str | Path,
    gen_kwargs: dict[str, Any],
    verbose: bool = False,
) -> None:
    conversation_to_txt(
        prompt_messages=messages,
        model_messages=model_generations,
        output_path=output_path,
        verbose=verbose,
        gen_kwargs=gen_kwargs,
    )


def dump_to_html(
    messages: list[Message],
    model_generations: List[Message],
    output_path: str | Path,
    gen_kwargs: dict[str, Any],
    verbose: bool = False,
) -> None:
    flat_generations = flatten_generations(model_generations)
    conversation_to_html(messages=messages + [flat_generations], output_path=output_path, verbose=verbose, gen_kwargs=gen_kwargs)


def dump_usage(
    provider: str,
    api_responses: list[Any],
    usage_dir: str | Path,
    call_id: str,
    gen_kwargs: dict[str, Any],
    verbose: bool = False,
    ovewrite_csv: bool = False,
) -> None:
    try:
        os.makedirs(usage_dir, exist_ok=True)
        call_id = re.sub(r"/", "_", call_id)
        csv_path = Path(usage_dir) / f"{call_id}.csv"
        if ovewrite_csv and csv_path.exists():
            csv_path.unlink()

        # For each API response, get its usage information
        usage_list = [_get_usage(provider, gen_kwargs, response) for response in api_responses]

        # Convert the list of usage dicts to DataFrame
        df_new = pd.DataFrame(usage_list)
        df_new["gen_config"] = None
        df_new.loc[0, "gen_config"] = str(gen_kwargs)
        # Column with number of generations
        df_new["num_generations"] = len(api_responses)

        if csv_path.exists():
            # Read existing CSV
            df_existing = pd.read_csv(csv_path)

            # Combine existing columns with new columns
            all_columns = list(set(df_existing.columns) | set(df_new.columns))

            # Update existing DataFrame with new columns
            for col in all_columns:
                if col not in df_existing.columns:
                    df_existing[col] = None
                if col not in df_new.columns:
                    df_new[col] = None

            # Exclude empty or all-NA columns for compatiblity with newer versions of pandas
            df_existing = df_existing.dropna(axis=1, how="all")
            df_new = df_new.dropna(axis=1, how="all")

            # Append new data
            df_updated = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_new = df_new.dropna(axis=1, how="all")
            df_updated = df_new

        # Save to CSV
        df_updated.to_csv(csv_path, index=False)
        if verbose:
            logger.info(f"API usage saved to {csv_path}")
    except Exception as e:
        logger.error(f"Error dumping usage for provider {provider}, call_id: {call_id}: {e}")


def _dump_batch_outputs(
    prompts,
    all_api_responses,
    all_model_generations,
    conversation_dirs,
    usage_dirs,
    call_ids,
    gen_config,
    verbose=False,
    dump_html=True,
    dump_txt=True,
    ovewrite_txt=False,
) -> None:
    if not isinstance(gen_config, dict):
        gen_config_dict = gen_config.to_dict()
    else:
        gen_config_dict = gen_config

    if not call_ids:
        return

    for i, call_id in enumerate(call_ids):
        try:
            conversation_dir = "" if not conversation_dirs else conversation_dirs[i]
            usage_dir = "" if not usage_dirs else usage_dirs[i]
            if not (conversation_dir or usage_dir):
                continue

            # if conversation_dir == usage_dir:
            #     conversation_dir = f"{conversation_dir}/conversation"
            #     usage_dir = f"{usage_dir}/usage"

            api_response = all_api_responses[i]
            model_generations = all_model_generations[i]
            if conversation_dir:
                # fmt: off
                dump_llm_output(
                    prompts[i], model_generations, conversation_dir, call_id, gen_config_dict, verbose, dump_html, dump_txt, ovewrite_txt
                )
                # fmt: on
            if usage_dir:
                dump_usage(gen_config["provider"], [api_response], usage_dir, call_id, gen_config_dict, verbose)
        except Exception as e:
            logger.error(f"Error dumping batch outputs for {conversation_dir}, {usage_dir}, {call_id}: {e}")

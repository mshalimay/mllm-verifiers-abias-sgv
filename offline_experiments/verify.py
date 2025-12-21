import os
import random
import re

from core_utils.logger_utils import logger
from core_utils.string_utils import clean_spaces
from llms.llm_utils import visualize_prompt
from llms.prompt_utils import get_conversation_payload_size, get_messages
from offline_experiments.prompts.build_prompt import get_k_injection, get_prompts_eval
from offline_experiments.utils_offline.utils_offline_exper import (
    get_intent_message,
    get_response_from_file,
    get_trace_data,
    get_trajectory_msgs,
    maybe_swap_sys_user_role,
)

cache_k_responses = {}

# Set random seed
random.seed(42)


def condense_k_responses(mode: str, k_resps: list[str], k_max: int = -1) -> str:
    # shuffle k_resps
    random.shuffle(k_resps)
    if mode == "concat":
        condensed_ks = "\n".join(k_resps)
        return condensed_ks
    elif mode == "multiprior":
        condensed_ks = ""
        for i, k_resp in enumerate(k_resps):
            # remove "#" from k_resp
            k_resp = re.sub(r"#+\s*", "", k_resp)
            k_resp = clean_spaces(k_resp)
            if k_max != -1 and i >= k_max:
                break
            condensed_ks += f"### Source {i + 1}:\n{k_resp}\n\n"
        condensed_ks.rstrip("\n\n")
        return condensed_ks
    else:
        raise ValueError(f"Unknown condense mode: {mode}")


def _get_k_from_cache(cache_key: str):
    try:
        if cache_key and cache_key in cache_k_responses:
            return cache_k_responses[cache_key]
        return None
    except Exception as _:
        return None


def _get_k_resp_from_cache_or_file(
    file_path: str = "",
):
    # Try to get K from cache
    k_resp = _get_k_from_cache(file_path)
    if k_resp:
        return k_resp

    if file_path:
        k_resp = get_response_from_file(file_path)
        cache_k_responses[file_path] = k_resp
        return k_resp

    k_resp = get_response_from_file(file_path)
    if k_resp:
        cache_k_responses[file_path] = k_resp
    return k_resp


def _extract_generations(k_resp: str) -> list[str]:
    """
    Extract all generations from a k_resp string that contains generation markers.

    Args:
        k_resp: String containing generation markers like "---------- GENERATION 0 ----------"

    Returns:
        List of strings, where each string is the content between generation markers
    """
    pattern_match = re.search(r"---------- GENERATION \d+ ----------\n", k_resp)
    if not pattern_match:
        return [k_resp.strip()]

    # Split by generation markers and extract content
    generations = re.split(r"---------- GENERATION \d+ ----------\n", k_resp)
    # Remove empty first element if it exists
    if generations and not generations[0].strip():
        generations = generations[1:]
    # Strip whitespace from each generation
    return [gen.strip() for gen in generations if gen.strip()]


def build_llm_call_args(trace_path, task_id, config, run_config) -> tuple[list[dict] | None, str | None, str | None, str | None]:
    try:
        conversation_dir = f"{config['out_dir'].strip('-')}/conversation"
        usage_dir = f"{config['out_dir'].strip('-')}/usage"
        combined_ks = []
        single_pass = True

        # If cached K dirs, load responses from previous generations
        if config["prompt_args"].get("k_configs", []):
            single_pass = False
            k_resps_per_id = []

            if config.get("additional_config", {}).get("multimodel_k"):
                k_resps = []
                single_pass = False
                k_resps_per_id = []
                # For each previous response for a given k_prompt_id, get the cached responses
                # e.g.: 1 generation to caption the trajectory; 3 generations to extract knowledge
                for k_config in config["prompt_args"]["k_configs"]:
                    k_id, k_dir_name = k_config["k_prompt_id"], k_config["cached_k_dir"]

                    file_path = f"{k_dir_name}/conversation/{task_id}.txt"
                    cached_k_resp = _get_k_resp_from_cache_or_file(file_path=file_path)
                    if not cached_k_resp:
                        logger.error(f"Failed to build prompt for {config}:\n  task_id: {task_id}\n  Msg: Unable to get first pass response from {file_path}")
                        return None, None, None, None

                    # Get (possible) N generations for this k_prompt_id (e.g.: generating N responses for knowledge extraction)
                    # e.g.: k_resps = [gen_0] for the captioning step; k_resps = [gen_0, gen_1, gen_2] for the knowledge extraction step
                    k_resps.extend(_extract_generations(cached_k_resp))
                condensed_ks = condense_k_responses("multiprior", k_resps, -1)
                final_ks = [get_k_injection(config["env"], "k_2p_expert_multiprior").format(k=condensed_ks)]
                k_resps_per_id.append(final_ks)

            else:
                # For each previous response for a given k_prompt_id, get the cached responses
                # e.g.: 1 generation to caption the trajectory; 3 generations to extract knowledge
                for k_config in config["prompt_args"]["k_configs"]:
                    k_id, k_dir_name = k_config["k_prompt_id"], k_config["cached_k_dir"]

                    file_path = f"{k_dir_name}/conversation/{task_id}.txt"
                    cached_k_resp = _get_k_resp_from_cache_or_file(file_path=file_path)
                    if not cached_k_resp:
                        logger.error(f"Failed to build prompt for {config}:\n  task_id: {task_id}\n  Msg: Unable to get first pass response from {file_path}")
                        return None, None, None, None

                    # Get (possible) N generations for this k_prompt_id (e.g.: generating N responses for knowledge extraction)
                    # e.g.: k_resps = [gen_0] for the captioning step; k_resps = [gen_0, gen_1, gen_2] for the knowledge extraction step
                    k_resps = _extract_generations(cached_k_resp)
                    k_injection = get_k_injection(config["env"], k_id)
                    if k_config.get("additional_config", {}).get("condense_k_mode", ""):
                        # Condense multiple generations into one by joining with newlines
                        condensed_ks = condense_k_responses(k_config["additional_config"]["condense_k_mode"], k_resps, k_max=k_config["additional_config"].get("k_max", -1))
                        final_ks = [k_injection.format(k=condensed_ks)]
                    else:
                        final_ks = [k_injection.format(k=k_resp) for k_resp in k_resps]
                    k_resps_per_id.append(final_ks)

            # Combine the generations for each prompt ID.
            # Create combined_ks by taking the ith element from each k_prompt_id, or the last if it doesn't exist
            max_generations = max(len(final_ks) for final_ks in k_resps_per_id) if k_resps_per_id else 0
            for i in range(max_generations):
                combined_k = []
                for final_ks in k_resps_per_id:
                    # If the ith element exists, use it; otherwise use the last element
                    if i < len(final_ks):
                        combined_k.append(final_ks[i])
                    else:
                        combined_k.append(final_ks[-1])
                combined_ks.append(combined_k)

        # Get trajectory data
        trace_data = get_trace_data(config["env"], trace_path, task_id, img_ann_types=config.get("img_ann_types", []))
        if not trace_data:
            logger.error(f"Failed to build prompt for {config} Unable to get trajectory data.)")
            return None, None, None, None

        # Build prompt messages for objective and trajectory
        msg_intent = get_intent_message(config, trace_data, add_state_idxs=[], state_img_intros=[])
        trajectory_msgs = get_trajectory_msgs(config, trace_data, img_ann_types=config.get("img_ann_types", []))

        # Build prompts for Verifier evaluation
        verifier_prompts = get_prompts_eval(config, single_pass=single_pass)
        sys_prompt, eval_prompt = verifier_prompts["sys_prompt"], verifier_prompts["eval_prompt"]

        # Combine all parts into full prompts
        full_prompts = []
        user_or_sys_role = maybe_swap_sys_user_role(config["gen_config"]["model"])
        if combined_ks:
            for combined_k in combined_ks:
                full_prompt = [
                    {"role": user_or_sys_role, "content": sys_prompt},
                    msg_intent,
                    trajectory_msgs,
                    *combined_k,
                    eval_prompt,
                ]
                full_prompts.append(full_prompt)
        else:
            full_prompts.append([{"role": user_or_sys_role, "content": sys_prompt}, msg_intent, trajectory_msgs, eval_prompt])

        if run_config.get("skip_payload", 0):
            payload_size = get_conversation_payload_size(get_messages(full_prompts[0]))
            if payload_size > run_config["skip_payload"]:
                logger.info(f"Skipping {task_id} because payload size {payload_size / 1024 / 1024} MB is greater than {run_config['skip_payload'] / 1024 / 1024} MB")
                return None, None, None, None

        # IDs and directory to log the MLLM responses, usage data
        if len(combined_ks) > 1:
            call_ids = [f"{task_id}_k-{i}" for i in range(len(combined_ks))]
        else:
            call_ids = [task_id] * len(full_prompts)
        conversation_dirs = [conversation_dir] * len(full_prompts)
        usage_dirs = [usage_dir] * len(full_prompts)

        # Skip any generation for which there is already a conversation file
        if not run_config["overwrite"]:
            for call_id in call_ids:
                full_conversation_path = f"{conversation_dir}/{call_id}.txt"
                if os.path.exists(full_conversation_path):
                    logger.info(f"Skipping {call_id} because {full_conversation_path} exists.")
                    return None, None, None, None

        logger.info(f"VERIFY: Finished building llm call args for task {task_id}, config {config}")
        return full_prompts, conversation_dirs, usage_dirs, call_ids  # type: ignore

    except Exception as e:
        logger.error(f"Failed to build prompt for {config}:\n  task_id: {task_id}\n  Msg: {repr(e)}", exc_info=True)
        return None, None, None, None

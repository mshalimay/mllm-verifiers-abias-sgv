import os

from core_utils.logger_utils import logger
from offline_experiments.prompts.build_prompt import get_prompts_first_pass
from offline_experiments.utils_offline.utils_offline_exper import get_intent_message, get_trace_data, get_trajectory_msgs, maybe_swap_sys_user_role


def build_prompt_first_pass(
    trace_data,
    config,
    trajectory_msgs=None,
):
    env = config["env"]
    prompt_config = config["prompt_args"]
    k_config = prompt_config["k_config"]
    thoughts_actions_idxs = k_config.get("trace_info", {}).get("idxs", None)
    trajectory_msgs = get_trajectory_msgs(config, trace_data, img_ann_types=k_config.get("img_ann_types", []))
    k_retrieval_query, _sys_prompt = get_prompts_first_pass(env, config)

    state_img_intros = []
    if thoughts_actions_idxs is not None:
        add_state_idxs = []
        state_img_intros = []
    else:
        add_state_idxs = [0]
        if "vwa" in env:
            state_img_intros = ["Initial Webpage Screenshot"]
        elif env == "osw":
            state_img_intros = ["Initial Computer Screenshot"]

    objective_msg = get_intent_message(
        config=config,
        trace_data=trace_data,
        add_state_idxs=add_state_idxs,
        state_img_intros=state_img_intros,
    )

    user_or_sys_role = maybe_swap_sys_user_role(config["gen_config"]["model"])
    if thoughts_actions_idxs is not None:
        prompt = [{"role": user_or_sys_role, "content": _sys_prompt}, objective_msg, trajectory_msgs, k_retrieval_query]
    else:
        prompt = [{"role": user_or_sys_role, "content": _sys_prompt}, objective_msg, k_retrieval_query]

    return prompt


def build_llm_call_args(trace_path, task_id, config, run_config) -> tuple[list[dict] | None, str | None, str | None, str | None]:
    conversation_dir = f"{config['out_dir']}/conversation"
    usage_dir = f"{config['out_dir']}/usage"
    full_conversation_path = f"{conversation_dir}/{task_id}.txt"
    try:
        # If output already exists and not overwrite, skip
        if not run_config["overwrite"]:
            if os.path.exists(full_conversation_path):
                logger.info(f"Skipping {task_id}: {full_conversation_path} exists.")
                return None, None, None, None
        else:
            if os.path.exists(full_conversation_path):
                os.remove(full_conversation_path)
            full_usage_path = f"{usage_dir}/{task_id}.csv"
            if os.path.exists(full_usage_path):
                os.remove(full_usage_path)

        k_config = config["prompt_args"].get("k_config", None)
        if not k_config:
            return None, None, None, None

        trace_data = get_trace_data(config["env"], trace_path, task_id, img_ann_types=k_config.get("img_ann_types", []))
        prompt = build_prompt_first_pass(trace_data, config)

        logger.info(f"FIRST PASS: Building llm call args for {full_conversation_path}, config {config}")

        return [prompt], [conversation_dir], [usage_dir], [task_id]  # type: ignore
    except Exception as e:
        logger.warning(f"FIRST PASS: Error creating llm call args for {full_conversation_path}, config {config}: {e}", exc_info=True)
        return None, None, None, None

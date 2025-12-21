# ===============================================
# VWA-specific
# ===============================================

import json
import re
from pathlib import Path
from typing import Any

from llms.prompt_utils import get_messages
from offline_experiments.config.eval_configs import VWA_DOCKER_INSTANCE_ID
from vwa.agent.prompt_constructor import PromptConstructor
from vwa.agent.request_refiner import RequestRefiner

request_refiner = None
caption_image_fn = None
prompt_constructor = None


def _get_captioning_fn():
    from vwa.utils_vwa.captioner_utils import define_captioning_fn

    caption_image_fn, eval_caption_image_fn = define_captioning_fn(
        agent_captioning_model="Salesforce/blip2-flan-t5-xl",
        agent_captioning_model_device="server-cuda",
        eval_captioning_model_device="server-cuda",
        eval_captioning_model="Salesforce/blip2-flan-t5-xl",
        observation_type="image_som",
    )
    return caption_image_fn, eval_caption_image_fn


def _get_prompt_constructor(
    name_user="user",
    name_assistant="assistant",
    img_detail="auto",
    prompt_constructor_class="PromptConstructor",
    text_first=True,
    image_expert: dict[str, Any] = {},
) -> PromptConstructor:
    global prompt_constructor
    if not prompt_constructor:
        _build_prompt_constructor(
            name_user=name_user,
            name_assistant=name_assistant,
            img_detail=img_detail,
            prompt_constructor_class=prompt_constructor_class,
            text_first=text_first,
            image_expert=image_expert,
        )
    return prompt_constructor  # type: ignore


def _build_prompt_constructor(
    name_user="user",
    name_assistant="assistant",
    img_detail="auto",
    prompt_file="p_verifier",
    prompt_constructor_class="PromptConstructor",
    text_first=True,
    image_expert: dict[str, Any] = {},
):
    import importlib

    prompt_constructor_module = importlib.import_module("vwa.agent.prompt_constructor")
    PromptConstructorClass = getattr(prompt_constructor_module, prompt_constructor_class)

    lm_config_dict = {
        "name_user": name_user,
        "name_assistant": name_assistant,
        "img_detail": img_detail,
        "text_first": text_first,
    }

    agent_config = {
        "use_text_observation": False,
        "use_img_observation": True,
        "prompt": prompt_file,
        "lm_config": lm_config_dict,
    }
    if image_expert:
        agent_config["image_expert"] = image_expert

    global prompt_constructor
    prompt_constructor = PromptConstructorClass(lm_config=lm_config_dict, agent_config=agent_config)
    return prompt_constructor


def _get_traj_attempt_num(traj_data, file_path) -> int | None:
    traj_attempt_num = traj_data.get("metadata", {}).get("reflexion_data", {}).get("attempt_num", None)
    if traj_attempt_num is None:
        traj_attempt_num = Path(file_path).parent.name.split("_")[-1]
        if traj_attempt_num.isdigit():
            traj_attempt_num = int(traj_attempt_num)
        else:
            traj_attempt_num = None

    return traj_attempt_num


def _get_interaction_history_message(
    trace_data: dict[str, Any],
    use_a: bool = False,
    use_u: bool = True,
    state_idxs: list[int] = [],
    aeval_refine: bool = False,
    thoughts_actions_idxs=[],
    shuffle_states: bool = False,
    image_expert: dict[str, Any] = {},
):
    global prompt_constructor
    prompt_constructor = _get_prompt_constructor(image_expert=image_expert)

    trajectory = trace_data["trajectory"]
    meta_data = trace_data["meta_data"]

    inputs = []

    if aeval_refine:
        action_history = ""
        for idx, act in enumerate(trajectory.actions):
            action_history += f"{idx + 1}: {act['extracted_action']}\n"

        prompt_action_history = """\n## Action History:\n{action_history}""".format(action_history=action_history)

        inputs = [
            prompt_action_history,
            "## Last snapshot of the webpage:",
            trajectory.states[-1]["observation"]["image"],
        ]
        msg = get_messages(
            inputs,
            role="user",
            name="",
            concatenate_text=True,
        )
        return msg

    prompt_constructor.instruction["meta_data"]["use_low_level_actions_env_parsed"] = use_a
    prompt_constructor.instruction["meta_data"]["use_assistant_utterance"] = use_u

    # Trajectory includes all states if no state_idxs are provided
    msg_img_links = None
    if not state_idxs:
        state_idxs = list(range(len(trajectory.states)))

    # For answers providing only URLs, give additional context to be possible to evaluate

    if use_a or use_u:
        if re.findall(
            r"https?://[^\s,\]]+?\.(?:jpg|jpeg|png|gif|bmp|webp)",
            trajectory.actions[-1].get("extracted_action", ""),
            re.IGNORECASE,
        ):
            msg_img_links, img_metadata = prompt_constructor.get_data_for_urls(
                trajectory.actions[-1].get("extracted_action", ""),
                trajectory.states[-1].get("observation", {}).get("text", ""),
                env_id=str(VWA_DOCKER_INSTANCE_ID),
            )
            if not msg_img_links:
                raise ValueError(f"A URL is provided in the action, but not able to get image from the link: {trajectory.actions[-1].get('extracted_action', '')}. Maybe no container is running.")

            # If raw link answers, provide associated data from the text observation if any
            # Obs.: No need to add text metadata if using text observation
            if img_metadata and not prompt_constructor.use_text_observation:
                add_text_metadata = {"intro": "STATE `t-{t}` TEXT REPRESENTATION:\n", "text": img_metadata}
                # This adds a text observation to the specific state
                trajectory.states[-1]["add_text_metadata"] = add_text_metadata

    # Build interaction history
    interaction_history_msgs = prompt_constructor.build_interaction_history(trajectory, meta_data, idxs_history=state_idxs, thoughts_actions_idxs=thoughts_actions_idxs, shuffle_states=shuffle_states)

    # If image link answers, provide the image or images
    if msg_img_links:
        interaction_history_msgs.append(msg_img_links)  # add the image or images

    return interaction_history_msgs


def parse_trajectory_json_data(file_path, attempt_num: int | None = None) -> tuple[str, str, bool]:
    with open(file_path, "r") as f:
        traj_data = json.load(f)
        task_id = traj_data["task_id"]
        domain = traj_data["domain"]

        if attempt_num is not None:
            traj_attempt_num = _get_traj_attempt_num(traj_data, file_path)
            if traj_attempt_num is None:
                return task_id, domain, False

            if traj_attempt_num and int(traj_attempt_num) != int(attempt_num):
                return task_id, domain, False

        episode_completed = traj_data.get("episode_completed", False)

        if not episode_completed:
            print(f"Episode not completed for traj: {file_path}")
        return task_id, domain, episode_completed


def get_trace_data_vwa(
    trajectory_path: str,
    task_id: str | int,
    img_ann_types: list[str] = [],
):
    from vwa.utils_vwa.extract_trajectory_from_log import extract_trajectory_data

    try:
        objective, trajectory, meta_data = extract_trajectory_data(file_path=trajectory_path, stop_at_verifier_loop=True, ann_types=img_ann_types)

        return {
            "objective": objective,
            "trajectory": trajectory,
            "meta_data": meta_data,
            "task_id": str(task_id),
            "trajectory_path": trajectory_path,
        }

    except Exception as e:
        import traceback

        traceback.print_exc()
        return None


def get_intent_message_vwa(
    trace_data: dict[str, Any],
    add_state_idxs: list[int] = [],
    state_img_intros: list[str] = [],
    aeval_refine: bool = False,
    caption_input_img: bool = True,
    config: dict[str, Any] = {},
):
    global caption_image_fn, request_refiner, prompt_constructor

    if caption_input_img:
        if not caption_image_fn:
            caption_image_fn, _ = _get_captioning_fn()

        if not request_refiner:
            request_refiner = RequestRefiner(
                agent_config={"caption_input_img": caption_input_img},
                captioning_fn=caption_image_fn,
            )

    trajectory = trace_data["trajectory"]
    intent = trace_data["objective"]

    task_id = str(trace_data["task_id"])
    prompt_constructor = _get_prompt_constructor(image_expert=config.get("additional_config", {}).get("image_expert", {}))

    meta_data = {"task_id": task_id}
    if request_refiner:
        request_refiner.next_action(trajectory, intent["text"], intent["images"], meta_data)

    objective_image_captions = prompt_constructor.get_image_captions(intent["images"], meta_data)

    prefix = f"## OBJECTIVE:\n{intent['text']}"
    if aeval_refine:
        prefix = f"## User Intent:\n{intent['text']}"

    return prompt_constructor.build_intent_message(
        trajectory,
        prefix,
        intent["images"],
        {},
        add_states_idxs=add_state_idxs,
        state_img_intros=state_img_intros,
        objective_image_captions=objective_image_captions,
        add_state_img=True,
    )


def get_trajectory_vwa(config, trace_data, aeval_refine=False):
    prompt_config = config["prompt_args"]

    if prompt_config and prompt_config.get("trace_info") is not None:
        trace_info_type = prompt_config.get("trace_info", {}).get("type", "none")
        thoughts_actions_idxs = prompt_config.get("trace_info", {}).get("idxs", [])
        use_a = True if "actions" in trace_info_type else False
        use_u = True if "utt" in trace_info_type else False
        state_idxs = prompt_config.get("state_idxs", [])
    else:
        use_a = False
        use_u = False
        state_idxs = []
        thoughts_actions_idxs = []

    # Else, create messages and return
    trajectory_msgs = _get_interaction_history_message(
        trace_data=trace_data,
        use_a=use_a,
        use_u=use_u,
        state_idxs=state_idxs,
        aeval_refine=aeval_refine,
        thoughts_actions_idxs=thoughts_actions_idxs,
        shuffle_states=prompt_config.get("trace_info", {}).get("shuffle", False),
        image_expert=config.get("additional_config", {}).get("image_expert", {}),
    )
    return trajectory_msgs

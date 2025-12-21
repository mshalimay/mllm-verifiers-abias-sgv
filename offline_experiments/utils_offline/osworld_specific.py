import random
import re
import sys

cache_trajectory_msgs = {}

import os
from pathlib import Path

from core_utils.logger_utils import logger
from llms.prompt_utils import get_message
from osw.utils_osw.utils_osw import TrajectoryView, annotate_action_on_image, trace_to_english


def get_trace_data_osw(
    trajectory_path: str,
    task_id: str | int,
):
    trace_data = trace_to_english(trace_path=trajectory_path)
    trace_data["base_path"] = Path(trajectory_path).parent
    trace_data["trajectory_path"] = trajectory_path
    return trace_data


def get_intent_message_osw(
    trace_data,
    add_state_idxs: list[int] = [],
    state_img_intros: list[str] = [],
    objective_template: str = "## OBJECTIVE: {objective}",
    objective_first: bool = True,
):
    trajectory = TrajectoryView(trace_data, to_english=True)
    states = trajectory.states
    intent_str = trace_data["objective"]

    inputs = []
    if len(state_img_intros) > 0:
        inputs.append("## IMAGES:\n")
    for i, state_idx in enumerate(add_state_idxs):
        img_obs = states[state_idx]["observation"]["images"][-1]

        img_intro = f"Image ({i + 1})"
        if state_img_intros:
            img_intro += f": {state_img_intros[state_idx]}"
        inputs.append(img_intro)
        inputs.append(img_obs)

    str_objective = objective_template.format(objective=intent_str)

    if objective_first:
        return [str_objective] + inputs
    else:
        return inputs + [str_objective]


def _parse_thoughts_actions(text_response: str, splitters: list[str] = ["Action:"]) -> list[dict[str, str]]:
    split_pattern = "|".join(re.escape(s) for s in splitters)
    pattern = re.compile(rf"^(.*?)^(?:{split_pattern})(.*)$", re.DOTALL | re.MULTILINE)

    results = []
    for match in pattern.finditer(text_response):
        thought = match.group(1).strip()
        action_body = match.group(2).strip()
        # Find which splitter triggered the match (for clarity)
        matched_splitter = next((s for s in splitters if action_body.startswith(s) or text_response.find(s + action_body) >= 0), splitters[0])
        action = f"{matched_splitter} {action_body}" if not action_body.startswith(matched_splitter) else action_body
        results.append({"thought": thought, "action": action})

    return results


def get_interaction_history_message(
    trace_data,
    add_state_idxs: list[int] = [],
    num_states: int = 0,
    last_img_per_state: bool = True,
    use_text_obs=False,
    use_img_obs=True,
    intro_txt_obs_template="",
    intro_img_obs_template="## STATE t-{t} screenshot",
    use_thoughts: bool = True,
    use_actions: bool = True,
    thought_actions_idxs: list[int] = [],
    name_user="user",
    name_assistant="assistant",
    img_detail: str = "auto",
    img_ann_types: list[str] = [],
    shuffle_states: bool = False,
    reversed_idxs: bool = True,
    intro_execution_history: str = """"## Here is the trace of execution so far:""",
):
    trajectory = TrajectoryView(trace_data, to_english=True)

    states = trajectory.states
    if not add_state_idxs:
        add_state_idxs = list(range(len(states) - num_states))

    if thought_actions_idxs == [-1]:
        thought_actions_idxs = [add_state_idxs[-1]]

    elif not thought_actions_idxs:
        thought_actions_idxs = add_state_idxs

    idxs = add_state_idxs.copy()
    ts = range(len(idxs), 0, -1) if reversed_idxs else idxs
    if shuffle_states:
        ts, idxs = map(list, zip(*random.sample(list(zip(ts, idxs)), len(idxs))))
    else:
        idxs = sorted(idxs)

    messages = []
    if intro_execution_history:
        messages.append(
            get_message(
                inputs=intro_execution_history,
                role="user",
                name=name_user,
            )
        )

    for idx, t in zip(idxs, ts):
        action = trajectory.actions[idx]
        state = trajectory.states[idx]

        text_obs = state["observation"]["text"] if use_text_obs else ""
        img_observations = state["observation"]["images"] if use_img_obs else []
        if last_img_per_state and img_observations:
            img_observations = [img_observations[-1]]

        intro_txt_obs = ""
        if use_text_obs and intro_txt_obs_template:
            intro_txt_obs = intro_txt_obs_template.format(t=t)

        intro_img_obs = ""
        if use_img_obs and intro_img_obs_template:
            intro_img_obs = intro_img_obs_template.format(t=t)

        if eng_texts := action.get("texts_en"):
            text_generation = eng_texts[0]
        else:
            text_generation = action["texts"][0]

        try:
            if any(supported_ann in img_ann_types for supported_ann in ["coord", "coordinates", "dot"]):
                img = img_observations[-1]
                img = annotate_action_on_image(img, text_generation)
                img_observations = [img]
        except Exception as e:
            logger.warning(f"{__file__}: Failed to annotate action on image: {e}. Action: {text_generation}. Image: {img_observations[-1]}")

        # Otherwise, parse actions and thoughts as requested
        if idx in thought_actions_idxs:
            if use_actions and not use_thoughts:
                # Exclude the thought part; if unable to parse, keep the original generation
                splitted_generation = _parse_thoughts_actions(text_generation, splitters=["Action:"])[-1]
                if "action" in splitted_generation:
                    text_generation = splitted_generation["action"]

            elif use_thoughts and not use_actions:
                # Exclude the action part; if unable to parse, keep the original generation
                splitted_generation = _parse_thoughts_actions(text_generation, splitters=["Action:"])[-1]
                if "thought" in splitted_generation:
                    text_generation = splitted_generation["thought"]
            elif not use_thoughts and not use_actions:
                # Exclude both thoughts and actions
                text_generation = ""
        else:
            text_generation = ""

        user_msg = get_message(
            inputs=[intro_txt_obs, text_obs, intro_img_obs, *img_observations],
            role="user",
            name=name_user,
            img_detail=img_detail,
        )
        assistant_msg = get_message(
            inputs=text_generation,
            role="assistant",
            name=name_assistant,
            img_detail=img_detail,
        )

        messages.extend([user_msg, assistant_msg])
    return messages


def get_trajectory_osw(config, trace_data, img_ann_types: list[str] = []):
    # global cache_trajectory_msgs

    prompt_config = config["prompt_args"]

    if prompt_config and prompt_config.get("trace_info") is not None:
        trace_info_type = prompt_config.get("trace_info", {}).get("type", "none")
        thoughts_action_idxs = prompt_config.get("trace_info", {}).get("idxs", [])
        use_a = True if "actions" in trace_info_type else False
        use_u = True if "utt" in trace_info_type else False
        state_idxs = prompt_config.get("state_idxs", [])
    else:
        use_a = False
        use_u = False
        state_idxs = []
        thoughts_action_idxs = []

    state_idxs = prompt_config.get("state_idxs", [])
    img_detail = config.get("img_detail", "auto")
    msgs = get_interaction_history_message(
        trace_data,
        state_idxs,
        img_detail=img_detail,
        img_ann_types=img_ann_types,
        use_actions=use_a,
        use_thoughts=use_u,
        thought_actions_idxs=thoughts_action_idxs,
        shuffle_states=prompt_config.get("trace_info", {}).get("shuffle", False),
    )

    # cache_trajectory_msgs[cache_key] = msgs
    return msgs

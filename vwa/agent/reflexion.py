import json
import os
import re
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
from utils_vwa.extract_trajectory_from_log import extract_trajectory_data
from utils_vwa.utils_vwa import TrajectoryView

from agent.agent import Agent
from agent.prompt_constructor import LMParsingError, PromptConstructor
from core_utils.logger_utils import logger
from core_utils.string_utils import clean_spaces, partial_format
from core_utils.types import ImageInput
from llms.prompt_utils import get_message, get_messages
from llms.types import Message

# TODO: log source_dir in reflections json


# Enum for Critic Modes
class ReflexionMode(Enum):
    SGV = "sgv"
    NO_SGV = "no_sgv"


class EvalMode(Enum):
    ORACLE = "oracle"
    MODEL = "model"


def get_trajectory_from_summary_csv(
    summary_csv_path: str,
    attempt_num: int,
    domain: str,
    task_id: int,
    env_name: str,
    ann_types: list[str] = ["som"],
) -> tuple[dict[str, Any] | None, TrajectoryView | None, dict[str, Any] | None, str]:
    if not os.path.exists(summary_csv_path):
        return None, None, None, ""

    df = pd.read_csv(summary_csv_path)
    domain_task_id_attempt_num = f"{domain}{task_id}{attempt_num}"
    # Add column to df if not exists
    if "domain_task_id_attempt_id" not in df.columns:
        df["domain_task_id_attempt_id"] = df["domain_task_id"] + df["attempt_id"].astype(str)

    # Try to get row for domain_task_id, attempt_num
    row = df.loc[df["domain_task_id_attempt_id"] == domain_task_id_attempt_num]
    if len(row) == 0:
        return None, None, None, ""

    else:
        # Try to get traj_source_path
        if "traj_source_path" in row.columns:
            json_path = row["traj_source_path"].iloc[0]
        else:
            # Get source_summary_path
            source_summary_path = row["source_dir"].iloc[0]
            # Build trajectory.json path
            # Example: experiments/gemini-2.5-flash-001/reflexion-thinking_oracle/classifieds/tasks_1/trajectories/47_0/trajectory-classifieds-47.json
            json_path = f"{source_summary_path}/trajectories/{task_id}_{attempt_num}/trajectory-{domain}-{task_id}.json"

        intent, trajectory, meta_data = extract_trajectory_data(
            json_path,
            stop_at_verifier_loop=True,
            stop_at_loop_idx=0,
            ann_types=ann_types,
        )

        return intent, trajectory, meta_data, json_path


def load_reflections_from_disk(memory_dir_path: str | Path, uid: str = "") -> dict[str, list[dict[str, Any]]]:
    reflection_data: dict[str, list[dict[str, Any]]] = {}
    json_path = Path(memory_dir_path) / f"{uid}.json"
    if not os.path.exists(json_path):
        return {}

    with open(json_path, "r") as f:
        reflection_data = json.load(f)

    reflections_data: dict[str, list[dict[str, Any]]] = {}

    for _uid, data in reflection_data.items():
        if uid and _uid != uid:
            continue
        reflections_data[_uid] = data

    return reflections_data


class ReflexionPromptConstructor(PromptConstructor):
    def __init__(self, lm_config: dict[str, Any], agent_config: dict[str, Any]):
        super().__init__(lm_config, agent_config)
        self.mode: str = ReflexionMode(agent_config["mode"]).value
        self.parsing_error_msg_key: str = self.instruction["meta_data"]["parsing_error_msg_key"]
        self.parsing_error_msg_template: str = self.instruction["parsing_error_msg_template"]
        self.agent_config = agent_config
        self.image_info: str = agent_config.get("image_info", "som")

    def build_execution_history(self, trajectory: TrajectoryView, meta_data: dict[str, Any], idxs_trajectory: list[int]) -> list[Message]:
        return self.build_interaction_history(trajectory, meta_data, idxs_trajectory)

    def construct(
        self,
        trajectory: TrajectoryView,
        objective: str,
        objective_imgs: list[ImageInput],
        meta_data: dict[str, Any],
        idxs_trajectory: list[int] = [],
        previous_reflections: list[str] = [],
    ) -> list[Message]:
        messages: list[Message | str | ImageInput] = []

        # -----------------------------------------------------------------------
        # System prompt, evaluation prompt
        # -----------------------------------------------------------------------
        sys_prompt_template: str = self.instruction[f"system_prompt_{self.mode}"]

        sys_prompt = partial_format(
            sys_prompt_template,
            image_info=self.img_obs_info_trace,
            text_obs_info=self.text_obs_info,
            trace_info=self.trace_info,
        )

        # -----------------------------------------------------------------------
        # Examples
        # -----------------------------------------------------------------------

        # -----------------------------------------------------------------------
        # Build intent input
        # -----------------------------------------------------------------------
        # text intro for objective
        text_input = self.instruction["objective_template"].format(objective=objective)

        # Add intent images and corresponding textual prefixes
        state_img_intros, add_state_idxs = [], []

        # If SGV, and is the first pass, initial page is necessary to define the intent
        if self.mode == ReflexionMode.SGV.value:
            raise NotImplementedError("SGV mode not implemented yet.")

        messages.extend(
            self.build_intent_message(
                trajectory,
                text_input,
                objective_imgs,
                meta_data,
                objective_image_captions=self.get_image_captions(objective_imgs, meta_data),
                state_img_intros=state_img_intros,
                add_states_idxs=add_state_idxs,
                add_state_text=False,
                add_state_img=True,
                role="user",
                name=self.lm_config.get("name_user", ""),
            )
        )

        # -----------------------------------------------------------------------
        # Execution history
        # -----------------------------------------------------------------------
        # For answers providing only URLs, give additional context for being possible to evaluate
        msg_img_links, img_metadata = self.get_data_for_urls(
            trajectory.actions[-1].get("parsed_action", ""),
            trajectory.states[-1].get("observation", {}).get("text", ""),
        )
        # If raw link answers, provide associated data from the text observation (if any)
        if img_metadata and not self.use_text_observation:
            add_text_metadata = {"intro": "STATE `t-{t}` TEXT REPRESENTATION:\n", "text": img_metadata}
            trajectory.states[-1]["add_text_metadata"] = add_text_metadata  # This adds a text observation to the specific state
        execution_history_msgs = self.build_interaction_history(
            trajectory,
            meta_data,
            idxs_trajectory,
        )
        messages.extend(execution_history_msgs)

        # If image link answers, provide the image or images
        if msg_img_links:
            messages.append(msg_img_links)  # add the image or images
            # get_message(inputs = execution_history_msgs[-1].contents + msg_img_links.contents, role = "assistant", name = "assistant")

        # -----------------------------------------------------------------------
        # Get Reflexion Request
        # -----------------------------------------------------------------------
        # Inject previous reflections, if any
        reflection_injection_prompt = self.instruction.get("reflections_injection_template", "")
        if previous_reflections:
            previous_reflections_str = ""
            for i, reflection in enumerate(previous_reflections):
                previous_reflections_str += f"\n### REFLECTION {i + 1}:\n{reflection}"
        else:
            previous_reflections_str = " None"

        if reflection_injection_prompt:
            messages.append(
                get_message(
                    inputs=reflection_injection_prompt.format(previous_reflections=previous_reflections_str),
                    role="user",
                    name=self.lm_config.get("name_user", ""),
                )
            )

        reflection_request_template: str = self.instruction[f"reflection_request_{self.mode}"]
        reflection_request = partial_format(
            reflection_request_template,
            previous_reflections=previous_reflections_str,
        )

        # Parsing error feedback
        parsing_error_msg = ""
        if meta_data.get(self.parsing_error_msg_key):
            parsing_error_msg = meta_data[self.parsing_error_msg_key]
            meta_data[self.parsing_error_msg_key] = ""

        messages.append(
            get_message(
                inputs=reflection_request + parsing_error_msg,
                role="user",
                name=self.lm_config.get("name_user", ""),
            )
        )

        return get_messages(
            inputs=messages,  # type: ignore
            sys_prompt=sys_prompt,
            role="user",
            name=self.lm_config.get("name_user", ""),
        )

    def parse_reflection(self, response: str) -> dict[str, Any]:
        splitters: list[str] = self.instruction["meta_data"]["splitters"]
        parsed_data: dict[str, Any] = {}
        splitters_group: str = "|".join(map(re.escape, splitters))

        # Iterate over splitters and parse corresponding sections
        for splitter in splitters:
            pattern = rf"{re.escape(splitter)}(.*?)(?=\n(?:{splitters_group})|$)"
            matches = re.findall(pattern, response, re.IGNORECASE | re.DOTALL)
            if matches:
                content = matches[-1].strip()
                parsed_data[re.sub(r":$", "", splitter)] = content  # Remove trailing ":"

        if not all(req_splitter in parsed_data for req_splitter in self.instruction["meta_data"]["required_splitters"]):
            raise LMParsingError(f"Cannot find all required splitters in {response}")

        return parsed_data


class Reflexion(Agent):
    def __init__(self, lm_config: dict[str, Any], agent_config: dict[str, Any]) -> None:
        self.lm_config = lm_config
        self.prompt_constructor: ReflexionPromptConstructor = ReflexionPromptConstructor(lm_config=lm_config, agent_config=agent_config)
        self.eval_mode: EvalMode = EvalMode(agent_config["eval_mode"])
        self.max_reflexion_attempts = agent_config["max_reflexion_attempts"]
        self.agent_config = agent_config
        self.num_previous_state_actions = agent_config["num_previous_state_actions"]
        self.mode = agent_config["mode"]
        self.out_utterance = agent_config["out_utterance"]
        self.max_model_call = agent_config["max_model_call"]
        self.conversation_dir = agent_config.get("conversation_dir", "")
        self.usage_dir = agent_config.get("usage_dir", "")
        self.num_attempts_per_task = {}
        self.reflection_memory: dict[str, list[dict[str, Any]]] = {}
        self.num_previous_reflections = agent_config["num_previous_reflections"]
        self.reflections_memory_path: str | Path = agent_config.get("reflections_memory_path", "")
        self.get_from_memory_key: str = agent_config["get_from_memory_key"]
        self.csv_previous_executions = agent_config.get("csv_previous_executions", "")
        self.lm_logs_dir = agent_config.get("lm_logs_dir", None)
        self.name = "reflexion"

        # Create the directory if it doesn't exist
        if self.reflections_memory_path:
            os.makedirs(self.reflections_memory_path, exist_ok=True)

    def get_reflection_from_parsed_response(self, parsed_response: dict[str, Any]) -> str:
        reflection_str = parsed_response[self.prompt_constructor.instruction["meta_data"]["reflection_key"]]  # type: ignore
        # strip left ":" and space
        reflection_str = reflection_str.lstrip(":").strip()
        reflection_str = clean_spaces(reflection_str)

        # Remove "#" markers
        reflection_str = re.sub(r"#+", "", reflection_str)
        reflection_str = re.sub(r"(?m)^\s+", "", reflection_str)
        return reflection_str

    def should_retry(self, uid: str) -> bool:
        return self.get_num_attempts_per_task(uid) < self.max_reflexion_attempts

    def get_eval_mode(self) -> str:
        return self.eval_mode.value

    def select_states(self, trajectory: TrajectoryView, num_states: int = -1) -> list[int]:
        # Return indices of the last `num_previous_state_actions` states
        num_states = min(self.num_previous_state_actions, len(trajectory.states))
        return list(range(len(trajectory.states) - num_states, len(trajectory.states)))

    def select_reflections(self, uid: str, num_previous_reflections: int = -1, key: str = "REFLECTION") -> list[str]:
        prev_reflections_list = self.reflection_memory.get(uid, [])
        if not prev_reflections_list:
            return []
        if num_previous_reflections == -1:
            return [reflection_data.get(key, "") for reflection_data in prev_reflections_list]
        # Else, get last K reflections
        return [reflection_data.get(key, "") for reflection_data in prev_reflections_list[-num_previous_reflections:]]

    def next_action(
        self,
        trajectory: TrajectoryView,
        intent: str,
        intent_images: list[ImageInput],
        meta_data: dict[str, Any],
    ) -> tuple[Any, str]:
        idxs_trajectory = self.select_states(trajectory)
        num_tries = 0
        logger.info("Reflexion Agent: generating reflection...")
        prev_reflections = self.select_reflections(
            meta_data["uid"],
            num_previous_reflections=self.num_previous_reflections,
            key=self.get_from_memory_key,
        )
        parsed_response: dict[str, str] = {}
        parsed_response, raw_response, _ = self.act_parse_retry(
            trajectory=trajectory,
            intent=intent,
            intent_images=intent_images,
            meta_data=meta_data,
            idxs_trajectory=idxs_trajectory,
            parser_fn=self.prompt_constructor.parse_reflection,
            max_tries=self.max_model_call - num_tries,
            error_msg_template=self.prompt_constructor.instruction["parsing_error_msg_template"],
            error_msg_key=self.prompt_constructor.instruction["meta_data"]["parsing_error_msg_key"],
            previous_reflections=prev_reflections,
        )

        if self.out_utterance:
            logger.info(f"\n[Reflexion Agent]: {raw_response}")

        reflection_entry = {k: v.lstrip(":").strip() for k, v in parsed_response.items()}
        reflection_entry["raw_response"] = raw_response
        self.update_reflection_memory(meta_data["uid"], [reflection_entry])
        self.save_reflection_to_disk(meta_data["uid"], reflection_entry)

        return parsed_response, raw_response

    def save_reflection_to_disk(self, uid: str, reflection_entry: dict[str, Any]) -> None:
        # Open existing json
        if not self.reflections_memory_path:
            return

        json_path = Path(self.reflections_memory_path) / f"{uid}.json"
        os.makedirs(json_path.parent, exist_ok=True)

        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                reflection_data = json.load(f)
        else:
            reflection_data = {}

        logger.debug(f"Saving reflection to disk for {uid}: {reflection_entry}, path: {json_path}")
        if uid not in reflection_data:
            reflection_data[uid] = []
        reflection_data[uid].append(reflection_entry)

        dest_file_temp = json_path.with_suffix(".tmp")
        with open(dest_file_temp, "w") as f:
            json.dump(reflection_data, f, indent=2)
        dest_file_temp.rename(json_path)

    def update_reflection_memory(self, uid: str, reflections: list[dict[str, Any]]) -> None:
        if uid not in self.reflection_memory:
            self.reflection_memory[uid] = []
        self.reflection_memory[uid].extend(reflections)

    def get_from_memory(self, uid: str) -> list[str]:
        if self.reflection_memory.get(uid, []):
            key = self.get_from_memory_key
            return [reflection.get(key, "") for reflection in self.reflection_memory.get(uid, [])]  # type: ignore
        return []

    def get_num_attempts_per_task(self, uid: str) -> int:
        self.populate_reflection_memory(uid)
        return len(self.reflection_memory.get(uid, []))

    def _maybe_generate_reflection_for_existing_trajectory(self, attempt_num: int, uid: str) -> None:
        # At attempt i, executor has access to i reflections based on up to i-1 trajectories
        # e.g.: at attempt 1, there must be reflection for attempt 0
        # e.g.: at attempt 2, there must be reflection for attempt 1 and attempt 0

        # If already have reflections for this attempt, return
        # e.g.: attempt=2, and there are 3 reflections, no need to generate reflection
        # e.g.: attempt=3, and there are 3 reflections, no need to generate reflection
        if uid in self.reflection_memory and attempt_num <= len(self.reflection_memory[uid]) and attempt_num <= self.max_reflexion_attempts:
            return

        if not self.csv_previous_executions or not Path(self.csv_previous_executions).exists():
            return

        # If no reflections for this attempt, generate reflection for existing trajectory
        logger.info(f"Building trajectory for {uid} at attempt {attempt_num} using csv: {self.csv_previous_executions} for reflection")
        domain, task_id, env_name = uid.split("_")  # FIXME: clean this

        # Get trajectory for reflection
        intent, trajectory, meta_data, trajectory_json_path = get_trajectory_from_summary_csv(
            self.csv_previous_executions,
            attempt_num=attempt_num,
            domain=domain,
            task_id=int(task_id),
            env_name=env_name,
        )
        if uid in self.reflection_memory:
            self.reflection_memory[uid] = self.reflection_memory[uid][:attempt_num]

        if intent is None or trajectory is None or meta_data is None:
            raise ValueError(f"Error getting trajectory from summary csv for {uid} at attempt {attempt_num}")

        logger.info(f"Generating reflection for trajectory: {trajectory_json_path}")
        meta_data["uid"] = uid
        self.reset(config_dict={"task_id": task_id}, args={"attempts_per_task": {task_id: attempt_num}})

        self.next_action(trajectory, intent["text"], intent["images"], meta_data)

    def _get_num_attempts_per_task(self, uid: str) -> int:
        self.populate_reflection_memory(uid)
        return len(self.reflection_memory.get(uid, []))

    def populate_reflection_memory(self, uid: str) -> None:
        self.reflection_memory = {}
        # If existing reflections from current experiment, load them
        if self.reflections_memory_path:
            self.reflection_memory = load_reflections_from_disk(self.reflections_memory_path, uid)
        if not self.csv_previous_executions:
            return
        try:
            attempt_num = len(self.reflection_memory.get(uid, []))
            self._maybe_generate_reflection_for_existing_trajectory(attempt_num=attempt_num, uid=uid)
        except Exception as e:
            logger.error(
                f"Error populating reflection memory for previous executions of {uid} using csv: {self.csv_previous_executions}.\nError: {e}",
                exc_info=True,
            )

    def reset(self, config_dict: dict[str, Any], args: dict[str, Any] = {}) -> None:
        super().reset(config_dict, args)

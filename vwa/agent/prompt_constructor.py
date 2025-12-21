import random
import re
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

from browser_env.env_utils import map_url_to_local, map_url_to_real
from trajectory_utils.trajectory_utils import annotate_marker_on_image
from utils_vwa.utils_vwa import TrajectoryView

from agent.constants import PATH_RAW_PROMPTS
from core_utils.image_utils import any_to_pil
from core_utils.logger_utils import logger
from core_utils.string_utils import clean_spaces, extract_urls, safe_format
from core_utils.types import ImageInput
from llms.prompt_utils import get_interleaved_img_txt_msg, get_message
from llms.tokenizer_utils import Tokenizer
from llms.types import Message


class LMParsingError(Exception):
    def __init__(self, message: str, raw_response: str = "") -> None:
        self.message = message
        self.raw_response = raw_response
        super().__init__(self.message)


def load_instruction(prompt_id: str, path_raw_prompts: str = PATH_RAW_PROMPTS) -> dict[str, Any]:
    full_prompt_path = Path(path_raw_prompts) / f"{prompt_id}.py"
    if not full_prompt_path.exists():
        raise ValueError(f"Prompt {prompt_id} not found in {path_raw_prompts}")

    module_name = full_prompt_path.stem
    spec = spec_from_file_location(module_name, str(full_prompt_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {full_prompt_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "prompt"):
        raise ValueError(f"Prompt {prompt_id} not found in {full_prompt_path}")
    return module.prompt


class PromptConstructor(object):
    def __init__(self, lm_config: dict[str, Any], agent_config: dict[str, Any]):
        self.tokenizer: Tokenizer | None = None
        self.lm_config: dict[str, Any] = lm_config
        self.text_first: bool = self.lm_config.get("text_first", True)
        self.instruction: dict[str, Any] = load_instruction(agent_config["prompt"])
        self.parsing_error_msg_key: str | None = self.instruction["meta_data"].get("parsing_error_msg_key", None)
        self.img_detail: str = self.lm_config.get("img_detail", "auto")
        self.use_text_observation: bool = self.instruction["meta_data"].get("use_text_observation", False) or agent_config["use_text_observation"]
        self.use_img_observation: bool = self.instruction["meta_data"].get("use_img_observation", True) or agent_config["use_img_observation"]

        text_obs_info_key: str = agent_config.get("text_obs_info", "")

        if not self.use_text_observation:
            self.text_obs_info = ""
        elif text_obs_info_key:
            self.text_obs_info = self.instruction["text_obs_infos"][text_obs_info_key]
        else:
            self.text_obs_info = ""

        trace_config = agent_config.get("trace_config", {})
        if trace_config:
            if "use_low_level_actions" in trace_config:
                self.instruction["meta_data"]["use_low_level_actions"] = trace_config["use_low_level_actions"]
            if "use_low_level_actions_env_parsed" in trace_config:
                self.instruction["meta_data"]["use_low_level_actions_env_parsed"] = trace_config["use_low_level_actions_env_parsed"]
            if "use_assistant_utterance" in trace_config:
                self.instruction["meta_data"]["use_assistant_utterance"] = trace_config["use_assistant_utterance"]

        # TODO: clean this up; some code due to backwards compatibility
        # Information about images in execution trace
        if trace_config:
            self.img_obs_info_trace_key: str = trace_config["image_info"]
            self.img_obs_info_trace = self.instruction["image_infos"][self.img_obs_info_trace_key]
            self.use_raw_screenshot = self.img_obs_info_trace_key == "raw"
        else:
            self.img_obs_info_trace_key: str = ""
            self.img_obs_info_trace = ""
            self.use_raw_screenshot = False

        # Prompt part about execution trace
        if not trace_config:
            self.trace_info = ""

        elif self.instruction["meta_data"].get("use_assistant_utterance", False):
            self.trace_info = self.instruction["trace_infos"]["utt"]

        elif self.instruction["meta_data"].get("use_low_level_actions", False) or self.instruction["meta_data"].get("use_low_level_actions_env_parsed", False):
            self.trace_info = self.instruction["trace_infos"]["actions"]

        elif self.instruction["meta_data"].get("use_image", False) or self.instruction["meta_data"].get(
            "use_img_observation",
            False,  # TODO: clear second condition, backwards compatibility
        ):
            self.trace_info = self.instruction["trace_infos"]["img_only"]

        else:
            self.trace_info = ""

    def get_open_tabs(self, state_info: dict[str, Any]) -> str:
        if "info" not in state_info:
            return ""
        if "observation_metadata" not in state_info["info"]:
            return ""

        if "text" in state_info["info"]["observation_metadata"]:
            if "open_tabs" in state_info["info"]["observation_metadata"]["text"]:
                return state_info["info"]["observation_metadata"]["text"]["open_tabs"]

        if "image" in state_info["info"]["observation_metadata"]:
            if "open_tabs" in state_info["info"]["observation_metadata"]["image"]:
                return state_info["info"]["observation_metadata"]["image"]["open_tabs"]
        return ""

    def map_links_to_real(self, action_or_utterance: str) -> str:
        # Find all complete localhost URLs and map them to real counterparts
        localhost_urls = re.findall(r"https?://localhost:[^\s,\]]+", action_or_utterance, re.IGNORECASE)

        for localhost_url in localhost_urls:
            real_url = map_url_to_real(localhost_url)  # type: ignore
            action_or_utterance = action_or_utterance.replace(localhost_url, real_url)
        return action_or_utterance

    def map_links_to_local(self, action_or_utterance: str) -> str:
        # Find urls, if any
        urls = extract_urls(action_or_utterance)
        for url in urls:
            local_url = map_url_to_local(url)  # type: ignore
            action_or_utterance = action_or_utterance.replace(url, local_url)
        return action_or_utterance

    def get_data_for_urls(self, action_or_utterance: str, text_obs: str = "", env_id: str = ""):
        try:
            # Find all content inside stop [ ... ]
            matches = re.findall(
                r"stop\s*\[\s*([^\]]+)\s*\]",
                action_or_utterance,
                re.IGNORECASE,
            )
            all_img_links, all_other_links = [], []
            for match in matches:
                # Find all image URLs by extension in the matched content
                image_urls = re.findall(
                    r"https?://[^\s,\]]+\.(?:jpg|jpeg|png|gif|bmp|webp)",
                    match,
                    re.IGNORECASE,
                )
                all_img_links.extend(image_urls)

                # Find all other URLs
                other_urls = re.findall(
                    r"https?://[^\s,\]]+",
                    match,
                    re.IGNORECASE,
                )
                all_other_links.extend(url for url in other_urls if url not in all_img_links)

            if len(all_img_links) == 0 and len(all_other_links) == 0:
                return None, None

            inputs = []
            # If providing non-image links, provide minimum metadata for context
            img_metadata = ""
            if len(all_other_links) > 0 and text_obs:
                # extract img tags from text_obs
                img_pattern = r"\[(\d+)\] \[IMG\] \[([^,]+),.*?url: ([^\]]+)\]"
                img_tags = re.findall(img_pattern, text_obs)
                img_metadata = "\n".join([f"[{num}] [IMG] [{label}, url: {url}]" for num, label, url in img_tags])

            if len(all_img_links) == 1:
                inputs.append(f"Here is the image corresponding to the link returned by the assistant:\n")
            elif len(all_img_links) > 1:
                inputs.append(f"Here are the images corresponding to the links returned by the assistant:\n")

            for i, img_link in enumerate(all_img_links):
                local_link = map_url_to_local(img_link, docker_instance_id=env_id)
                img = any_to_pil(local_link)
                inputs.extend([f"- Image ({i}), url: {img_link}", img])

            msg = get_message(
                inputs,
                role="user",
                name="",
            )

            return msg, img_metadata
        except Exception as e:
            logger.error(f"Error getting URL from Agent response.: {e}", exc_info=True)
            return None, None

    def construct(
        self,
        trajectory: TrajectoryView,
        objective: str,
        objective_imgs: list[ImageInput],
        meta_data: dict[str, Any],
        idxs_trajectory: list[int] = [],
    ) -> list[Message]:
        raise NotImplementedError("Subclasses must implement the construct method")

    def build_interaction_history(
        self,
        trajectory: TrajectoryView,
        meta_data: dict[str, Any],
        idxs_history: list[int] = [],
        text_first: bool = False,
        raw_screenshot: bool | None = None,
        add_tab_info: bool = False,
        thoughts_actions_idxs: list[int] = [],
        shuffle_states: bool = False,
        reversed_idxs: bool = True,
    ) -> list[Message]:
        messages: list[Message] = []

        if raw_screenshot is None:
            raw_screenshot = "raw" in self.img_obs_info_trace_key

        if not idxs_history:
            return messages

        # Do not add invalid actions to the history
        valid_indices = [idx for idx in sorted(idxs_history) if "invalid" not in trajectory.actions[idx]]
        if not valid_indices:
            return messages

        # If there is an intro execution history, add it
        if self.instruction.get("intro_execution_history"):
            messages.append(
                get_message(
                    inputs=self.instruction["intro_execution_history"],
                    role="user",
                    name=self.lm_config.get("name_user", ""),
                )
            )

        if thoughts_actions_idxs == [-1]:
            thoughts_actions_idxs = [1]

        ins_metadata = self.instruction["meta_data"]
        ts = range(len(valid_indices), 0, -1) if reversed_idxs else valid_indices
        if shuffle_states:
            ts, valid_indices = map(list, zip(*random.sample(list(zip(ts, valid_indices)), len(valid_indices))))

        for idx, t in zip(valid_indices, ts):
            action = trajectory.actions[idx]
            state = trajectory.states[idx]
            text_obs = state["observation"]["text"] if self.use_text_observation else ""
            if raw_screenshot:
                img_obs = state["observation"]["raw_screenshot"] if self.use_img_observation else ""
            else:
                img_obs = state["observation"]["image"] if self.use_img_observation else ""

            if any(ann in self.img_obs_info_trace_key for ann in ["coord", "coordinates", "dot"]) and action.get("element_center"):
                x_coord, y_coord = action["element_center"]
                if x_coord and y_coord:
                    img_obs, _ = annotate_marker_on_image(img_obs, {"x": action["element_center"][0], "y": action["element_center"][1], "relative": True})

            intro_txt_obs = ""
            if self.use_text_observation and "intro_txt_obs_history" in self.instruction:
                intro_txt_obs = safe_format(self.instruction["intro_txt_obs_history"], t=t)

            if add_tab_info:
                if text_obs:
                    text_obs = text_obs + "\n" + f"Open tabs: {self.get_open_tabs(state)}"
                else:
                    text_obs = f"Open tabs: {self.get_open_tabs(state)}"

            if "add_text_metadata" in state:
                intro_txt_data = safe_format(state["add_text_metadata"]["intro"], t=t)
                text_obs = intro_txt_data + text_obs + state["add_text_metadata"]["text"]

            intro_img_obs = ""
            if self.use_img_observation and "intro_img_obs_history" in self.instruction:
                intro_img_obs = self.instruction["intro_img_obs_history"].format(t=t)

            utterance = ""
            no_prediction_flag = False

            # Add assistant responses
            if thoughts_actions_idxs and t not in thoughts_actions_idxs:
                utterance = ""
            else:
                if ins_metadata.get("use_assistant_utterance") or ins_metadata.get("last_u"):
                    if ins_metadata.get("last_u") and idx == len(valid_indices) - 1:
                        utterance = clean_spaces(action["raw_prediction"])
                        no_prediction_flag = not utterance
                    elif self.instruction["meta_data"].get("use_assistant_utterance"):
                        utterance = clean_spaces(action["raw_prediction"])
                        no_prediction_flag = not utterance

                if ins_metadata.get("use_low_level_actions"):
                    u = clean_spaces(action["extracted_action"])
                    utterance = u if not utterance else f"{utterance}\n\n{u}"
                    # if re.match(r"^stop\s*\[[Ee]arly stop:.*\]$", utterance, re.IGNORECASE):
                    #     utterance = "stop[]"

                # Add action strings parsed by the environment. OR condition: if no prediction, this is added as fallback
                elif ins_metadata.get("use_low_level_actions_env_parsed") or no_prediction_flag:
                    if not (ins_metadata.get("last_u") and idx == len(valid_indices) - 1):
                        # Use action_str_history. Obs: this is environment specific to VWA
                        u = clean_spaces(meta_data["action_str_history"][idx + 1])
                        u = re.sub(r"\n]", "]", u)  # FIXME backward compatibility for old format: 'type [text\n]' represented 'type [text] [1]'
                        if ins_metadata.get("no_label"):
                            u = re.sub(r"label\s+'.*?'\s+and\s+", "", u)
                        utterance = u if not utterance else f"{utterance}\n\n{u}"

                utterance = self.map_links_to_real(utterance)

            user_msg = get_message(
                inputs=[intro_txt_obs, text_obs, intro_img_obs, img_obs],
                role="user",
                name=self.lm_config.get("name_user", ""),
                img_detail=self.lm_config.get("img_detail", "auto"),
            )
            assistant_msg = get_message(
                inputs=utterance,
                role="assistant",
                name=self.lm_config.get("name_assistant", ""),
                img_detail=self.lm_config.get("img_detail", "auto"),
            )

            if text_first:
                messages.extend([assistant_msg, user_msg])
            else:
                messages.extend([user_msg, assistant_msg])
        return messages

    def build_rationale_action_history(
        self,
        trajectory: TrajectoryView,
        meta_data: dict[str, Any],
        idxs_history: list[int] = [],
    ) -> str:
        # If no history yet, "None"
        if idxs_history is None or len(idxs_history) == 0:
            return "None"

        # Else, build prompt with previous actions and utterances
        prompt_parts: list[str] = []
        t = 0
        for idx in reversed(idxs_history):
            action = trajectory.actions[idx]
            # If invalid action, skip (action,state) pair
            if "invalid" in action:
                continue
            t += 1

            utterance = clean_spaces(action["raw_prediction"])
            parsed_action = self.map_links_to_real(clean_spaces(meta_data["action_str_history"][idx + 1]))
            url = map_url_to_real(trajectory.states[idx]["info"]["page"].url)
            if action.get("thought_summary"):
                thought_summary = clean_spaces(action["thought_summary"])
            else:
                thought_summary = ""

            prompt = safe_format(
                self.instruction["intro_txt_obs_history"],
                t=t,
                utterance=utterance,
                parsed_action=parsed_action,
                url=url,
                thought_summary=thought_summary,
            )

            prompt_parts.append(prompt)
        if len(prompt_parts) == 0:
            return "None"

        return "\n".join(reversed(prompt_parts))

    def get_image_captions(
        self,
        images: list[ImageInput],
        meta_data: dict[str, Any],
        key: str = "intent_images_captions",
    ) -> list[str]:
        img_captions: list[str] = []

        # If no image captions, return empty list
        if key not in meta_data:
            return img_captions

        # Else, get captions for each image
        for _, img in enumerate(images):
            img = any_to_pil(img)
            img_caption = meta_data[key].get(hash(img.tobytes()), "")
            img_captions.append(img_caption)
        return img_captions

    # TODO: make this more general; refine function; modularize away the textual intros to instruction
    def build_intent_message(
        self,
        trajectory: TrajectoryView,
        text_intent_input: str,
        objective_imgs: list[ImageInput],
        meta_data: dict[str, Any],
        objective_image_captions: list[str] = [],
        add_states_idxs: list[int] = [],
        add_state_text: bool = False,
        add_state_img: bool = False,
        state_img_intros: list[str] = [],
        state_text_intros: list[str] = [],
        role: str = "user",
        name: str = "",
        raw_screenshot: bool | None = None,
        add_tab_info: bool = False,
    ) -> list[Message]:
        if raw_screenshot is None:
            raw_screenshot = self.use_raw_screenshot

        imgs: list[ImageInput] = []
        full_img_captions: list[str] = []
        img_idx: int = 0
        text_observations_input: str = ""

        # Add environment state as part of current intent
        if add_states_idxs:
            for i, idx in enumerate(add_states_idxs):
                # Select the state
                state = trajectory.states[idx]

                # Add screenshot of current state
                if add_state_img and self.use_img_observation:
                    if raw_screenshot:
                        imgs.append(state["observation"]["raw_screenshot"])
                        full_img_captions.append(f"Image {img_idx}: {state_img_intros[i]}")
                    else:
                        imgs.append(state["observation"]["image"])
                        full_img_captions.append(f"Image {img_idx}: {state_img_intros[i]}")

                # Add text observation of current state
                if add_state_text and self.use_text_observation:
                    if state_text_intros:
                        text_observations_input += f"{state_text_intros[i]}"
                    text_observations_input += f"{state['observation']['text']}\n\n"
                if add_tab_info:
                    text_observations_input += f"Open tabs: {self.get_open_tabs(state)}\n\n"
                img_idx += 1

        # Full textual input = text observations + text intent
        final_text_input = f"{text_observations_input}{text_intent_input}"
        if self.parsing_error_msg_key and meta_data.get(self.parsing_error_msg_key):
            final_text_input += meta_data[self.parsing_error_msg_key]

        # Get intent images and corresponding captions
        if objective_imgs:
            for i, img in enumerate(objective_imgs):
                imgs.append(img)
                # Index of image in the full list of images
                img_caption = f"Image {img_idx}: objective image {i + 1}"

                # If image was captioned by previous agents, add it to the full caption
                additional_caption = objective_image_captions[i] if objective_image_captions else ""
                if additional_caption:
                    img_caption = f"{img_caption}; description: {additional_caption}."
                else:
                    img_caption = f"{img_caption}."

                full_img_captions.append(img_caption)
                img_idx += 1

        # Prepend image captions with "IMAGES:"
        if full_img_captions:
            full_img_captions[0] = f"IMAGES:\n{full_img_captions[0]}"

        # Build message
        msg = get_interleaved_img_txt_msg(
            images=imgs,
            img_captions=full_img_captions,
            role=role,
            name=name,
            img_detail=self.lm_config.get("img_detail", "auto"),
            text_prefix=final_text_input,
            text_first=self.lm_config.get("text_first", True),
        )
        return [msg]

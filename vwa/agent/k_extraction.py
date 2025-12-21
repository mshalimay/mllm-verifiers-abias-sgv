import re
from enum import Enum
from typing import Any

from utils_vwa.utils_vwa import TrajectoryView

from agent.agent import Agent
from agent.prompt_constructor import LMParsingError, PromptConstructor
from core_utils.logger_utils import logger
from core_utils.string_utils import clean_spaces, partial_format
from core_utils.types import ImageInput
from llms.prompt_utils import get_messages
from llms.types import Message


# Enum for Critic Modes
class KExtractionMode(Enum):
    SGV = "sgv"
    NO_SGV = "no_sgv"


class KExtractionPromptConstructor(PromptConstructor):
    mode: str

    def __init__(self, lm_config: dict[str, Any], agent_config: dict[str, Any]):
        super().__init__(lm_config, agent_config)
        self.mode = KExtractionMode(agent_config["mode"]).value
        self.parsing_error_msg_key: str = self.instruction["meta_data"]["parsing_error_msg_key"]
        self.parsing_error_msg_template: str = self.instruction["parsing_error_msg_template"]
        self.expert: bool = agent_config["expert"]
        self.agent_config = agent_config
        self.img_obs_info: str = agent_config["trace_config"]["image_info"]
        self.text_obs_info: str = agent_config["trace_config"].get("text_obs_info", "none")
        self.trace_info: str = agent_config["trace_config"].get("trace_info", "none")
        self.caption_input_images: bool = agent_config.get("caption_input_images", False)

    def build_execution_history(self, trajectory: TrajectoryView, meta_data: dict[str, Any], idxs_trajectory: list[int]) -> list[Message]:
        return self.build_interaction_history(trajectory, meta_data, idxs_trajectory)

    def construct(
        self,
        trajectory: TrajectoryView,
        objective: str,
        objective_imgs: list[ImageInput],
        meta_data: dict[str, Any],
        idxs_trajectory: list[int] = [],
    ) -> list[Message]:
        messages: list[Message | str | ImageInput] = []

        # -----------------------------------------------------------------------
        # System prompt, evaluation prompt
        # -----------------------------------------------------------------------
        if self.expert:
            sys_prompt: str = partial_format(
                self.instruction["sys_prompt_k_expert"],
                image_info=self.instruction["image_infos"][self.img_obs_info],
                text_obs_info=self.instruction["text_obs_infos"][self.text_obs_info],
                trace_info=self.instruction["trace_infos"][self.trace_info],
                rules=self.instruction["rules"][self.img_obs_info],
            ).strip()
        else:
            raise NotImplementedError("TODO: Pass the Verifier's system prompt")
            sys_prompt: str = self.instruction["sys_prompt"]

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

        # -----------------------------------------------------------------------
        # Execution history
        # -----------------------------------------------------------------------
        # If SGV, initial page is necessary to define the intent
        if self.mode == KExtractionMode.SGV.value:
            add_state_idxs.append(0)
            if self.use_img_observation:
                state_img_intros.append("description: Initial webpage screenshot")

            if self.use_text_observation:
                text_input += f"\nTEXT OBSERVATION: {trajectory.states[0]['observation']['text']}"

            objective_image_captions = []
            if self.caption_input_images:
                objective_image_captions = self.get_image_captions(objective_imgs, meta_data)
            messages.extend(
                self.build_intent_message(
                    trajectory,
                    text_input,
                    objective_imgs,
                    meta_data,
                    objective_image_captions=objective_image_captions,
                    state_img_intros=state_img_intros,
                    add_states_idxs=add_state_idxs,
                    add_state_text=False,
                    add_state_img=True,
                    role="user",
                    name=self.lm_config.get("name_user", ""),
                )
            )

        else:
            # Build full interaction history so far

            # For answers providing only URLs, give additional context
            msg_img_links, img_metadata = self.get_data_for_urls(
                trajectory.actions[-1].get("parsed_action", ""),
                trajectory.states[-1].get("observation", {}).get("text", ""),
            )
            # If raw link answers, provide associated data from the text observation (if any)
            if img_metadata and not self.use_text_observation:
                add_text_metadata = {"intro": "STATE `t-{t}` TEXT REPRESENTATION:\n", "text": img_metadata}
                trajectory.states[-1]["add_text_metadata"] = add_text_metadata  # This adds a text observation to the specific state
            execution_history_msgs = self.build_interaction_history(trajectory, meta_data, idxs_trajectory)
            messages.extend(execution_history_msgs)

            # If image link answers, provide the image or images
            if msg_img_links:
                messages.append(msg_img_links)  # add the image or images
                # get_message(inputs = execution_history_msgs[-1].contents + msg_img_links.contents, role = "assistant", name = "assistant")

        # -----------------------------------------------------------------------
        # Get K Extraction Request
        # -----------------------------------------------------------------------
        if self.expert:
            k_retrieval_prompt = self.instruction["k_retrieval_expert"]
        else:
            k_retrieval_prompt = self.instruction["k_retrieval_no_expert"]

        if meta_data.get(self.parsing_error_msg_key):
            k_retrieval_prompt = k_retrieval_prompt + meta_data[self.parsing_error_msg_key]
        messages.append(k_retrieval_prompt)

        return get_messages(
            inputs=messages,  # type: ignore
            sys_prompt=sys_prompt,
            role="user",
            name=self.lm_config.get("name_user", ""),
        )

    def parse_k_extraction(self, response: str) -> str:
        splitter = self.instruction["meta_data"]["splitters"][0]
        if not splitter:
            return clean_spaces(response)
        pattern = rf"{re.escape(splitter)}(.*)"
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return clean_spaces(match.group(1))
        else:
            raise LMParsingError(f"Cannot find {splitter} in {response}")


class KExtraction(Agent):
    def __init__(self, lm_config: dict[str, Any], agent_config: dict[str, Any]) -> None:
        self.lm_config = lm_config
        self.prompt_constructor: KExtractionPromptConstructor = KExtractionPromptConstructor(lm_config=lm_config, agent_config=agent_config)
        self.agent_config = agent_config
        self.num_previous_state_actions = agent_config["num_previous_state_actions"]
        self.mode = agent_config["mode"]
        self.out_utterance = agent_config.get("out_utterance", True)
        self.max_model_call = agent_config["max_model_call"]
        self.k_extraction_response = {}
        self.conversation_dir = agent_config.get("conversation_dir", "")
        self.usage_dir = agent_config.get("usage_dir", "")
        self.score_per_round: dict[str, Any] = {}
        self.name = "k_extraction"

    def select_states(self, trajectory: TrajectoryView, num_states: int = -1) -> list[int]:
        # Return indices of the last `num_previous_state_actions` states
        num_states = min(self.num_previous_state_actions, len(trajectory.states))
        return list(range(len(trajectory.states) - num_states, len(trajectory.states)))

    def get_k_extraction_response(self):
        return self.k_extraction_response["parsed_response"]

    def next_action(
        self,
        trajectory: TrajectoryView,
        intent: str,
        intent_images: list[ImageInput],
        meta_data: dict[str, Any],
    ) -> tuple[Any, str, int]:
        idxs_trajectory = self.select_states(trajectory)

        # NOTE: retrieve first step response just once for experiments with critique at STOP action only.
        # If extending to critique other actions besides STOP, might allow multiple retrievals.
        if not self.k_extraction_response:
            logger.info("K Extractor: generating...")
            parsed_response, raw_response, num_tries = self.act_parse_retry(
                trajectory=trajectory,
                intent=intent,
                intent_images=intent_images,
                meta_data=meta_data,
                idxs_trajectory=idxs_trajectory,
                parser_fn=self.prompt_constructor.parse_k_extraction,
                max_tries=self.max_model_call,
                error_msg_template=self.prompt_constructor.instruction["parsing_error_msg_template"],
                error_msg_key=self.prompt_constructor.instruction["meta_data"]["parsing_error_msg_key"],
            )
            trajectory.states[-1]["retrieved_knowledge"] = parsed_response
            self.k_extraction_response = {"raw_response": raw_response, "parsed_response": parsed_response}

            if meta_data.get("thought_summary"):
                self.k_extraction_response["thought_summary"] = meta_data["thought_summary"]
                meta_data["thought_summary"] = ""

            if self.out_utterance:
                if "thought_summary" in self.k_extraction_response:
                    logger.info(f"K Extractor: Thought summary: {self.k_extraction_response['thought_summary']}")
                logger.info(f"K Extractor: {parsed_response}")

            return parsed_response, raw_response, num_tries
        else:
            return self.k_extraction_response["parsed_response"], self.k_extraction_response["raw_response"], 0

    def reset(self, config_dict: dict[str, Any], args: dict[str, Any] = {}) -> None:
        super().reset(config_dict, args)
        self.k_extraction_response = {}

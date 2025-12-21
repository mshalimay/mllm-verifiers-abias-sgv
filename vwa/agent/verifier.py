import re
from enum import Enum
from typing import Any

from browser_env.actions import ActionTypes
from utils_vwa.utils_vwa import TrajectoryView

from agent.agent import Agent
from agent.prompt_constructor import LMParsingError, PromptConstructor
from core_utils.logger_utils import logger
from core_utils.string_utils import clean_spaces, partial_format
from core_utils.types import ImageInput
from llms.prompt_utils import get_message, get_messages
from llms.types import Message


class VerifierMode(Enum):
    SGV = "sgv"
    NO_SGV = "no_sgv"


class EvalMode(Enum):
    ORACLE_MODEL_FEEDBACK = "oracle_model_feedback"
    ORACLE_NO_FEEDBACK = "oracle_no_feedback"
    MODEL = "model"


class VerifierPromptConstructor(PromptConstructor):
    mode: str

    def __init__(self, lm_config: dict[str, Any], agent_config: dict[str, Any]):
        super().__init__(lm_config, agent_config)
        self.mode = VerifierMode(agent_config["mode"]).value
        self.eval_key: str = self.instruction["meta_data"]["eval_key"]
        self.eval_scores: list[str] = self.instruction["meta_data"]["eval_scores"]
        self.parsing_error_msg_key: str = self.instruction["meta_data"]["parsing_error_msg_key"]
        self.parsing_error_msg_template: str = self.instruction["parsing_error_msg_template"]
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
        add_tab_info: bool = False,
    ) -> list[Message]:
        messages: list[Message | str | ImageInput] = []

        # -----------------------------------------------------------------------
        # System prompt, evaluation prompt
        # -----------------------------------------------------------------------
        sys_prompt: str = self.instruction[f"system_prompt_{self.mode}"]
        eval_prompt: str = self.instruction[f"eval_prompt_{self.mode}"]

        sys_prompt = partial_format(
            sys_prompt,
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

        objective_imgs_captions = []
        if self.caption_input_images:
            objective_imgs_captions = self.get_image_captions(objective_imgs, meta_data)

        messages.extend(
            self.build_intent_message(
                trajectory,
                text_input,
                objective_imgs,
                meta_data,
                objective_image_captions=objective_imgs_captions,
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
            add_tab_info=add_tab_info,
        )
        messages.extend(execution_history_msgs)

        # If image link answers, provide the image or images
        if msg_img_links:
            messages.append(msg_img_links)  # add the image or images
            # get_message(inputs = execution_history_msgs[-1].contents + msg_img_links.contents, role = "assistant", name = "assistant")

        # -----------------------------------------------------------------------
        # Get Verifier Request
        # -----------------------------------------------------------------------
        if self.mode == VerifierMode.SGV.value:
            assert meta_data.get("k_extraction_response"), "K extraction response is required for SGV mode"
            knowledge_injection_prompt = self.instruction["knowledge_injection_prompt"].format(
                knowledge_retrieval_response=meta_data["k_extraction_response"],
            )
            messages.append(knowledge_injection_prompt)

        # Parsing error feedback
        parsing_error_msg = ""
        if meta_data.get(self.parsing_error_msg_key):
            parsing_error_msg = meta_data[self.parsing_error_msg_key]
            meta_data[self.parsing_error_msg_key] = ""

        messages.append(
            get_message(
                inputs=eval_prompt + parsing_error_msg,
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

    def parse_verifier_response(self, response: str) -> dict[str, Any]:
        splitters: list[str] = self.instruction["meta_data"]["splitters"]
        parsed_data: dict[str, Any] = {}
        splitters_group: str = "|".join(map(re.escape, splitters))

        # Iterate over splitters and parse corresponding sections
        for splitter in splitters:
            # Use regex to extract the section content
            pattern = rf"{re.escape(splitter)}(.*?)(?=\n(?:{splitters_group})|$)"
            matches = re.findall(pattern, response, re.IGNORECASE | re.DOTALL)
            if matches:
                content = matches[-1].strip()
                if self.eval_key in splitter:
                    # Map evaluation criteria like SUCCESS or FAILURE
                    parsed_data[self.eval_key] = self.parse_evaluation(content)
                else:
                    # General parsing for all sections, including COMPARISON
                    parsed_data[re.sub(r":$", "", splitter)] = content  # Remove trailing ":"

        if not all(req_splitter in parsed_data for req_splitter in self.instruction["meta_data"]["required_splitters"]):
            raise LMParsingError(f"Cannot find all required splitters in {response}")

        return parsed_data

    def parse_evaluation(self, content: str) -> str:
        """
        Extracts the evaluation score (e.g., SUCCESS, PARTIAL SUCCESS, FAILURE) from the EVALUATION section.
        """
        content = clean_spaces(content)
        eval_scores = "|".join(sorted(self.eval_scores, key=len, reverse=True))  # Join criteria with OR operator
        status_pattern = rf"(?i)\s*({eval_scores})"

        match = re.search(status_pattern, content)
        if match:
            return match.group(1).upper()  # Normalize to uppercase

        else:
            raise LMParsingError(f"Cannot determine evaluation score in: {content}")


class Verifier(Agent):
    def __init__(self, lm_config: dict[str, Any], agent_config: dict[str, Any]) -> None:
        self.lm_config = lm_config
        self.prompt_constructor: VerifierPromptConstructor = VerifierPromptConstructor(lm_config=lm_config, agent_config=agent_config)
        self.agent_config = agent_config
        self.num_previous_state_actions = agent_config["num_previous_state_actions"]
        self.mode = VerifierMode(agent_config["mode"]).value
        self.out_utterance = agent_config["out_utterance"]
        self.success_keyword = self.prompt_constructor.instruction["meta_data"]["success_keyword"]
        self.max_model_call = agent_config["max_model_call"]
        self.max_verifier_executor_loop = agent_config["max_verifier_executor_loop"]
        self.conversation_dir = agent_config.get("conversation_dir", "")
        self.usage_dir = agent_config.get("usage_dir", "")
        self.online_verify = agent_config["online_verify"]
        eval_mode = "model" if not agent_config.get("eval_mode", "") else agent_config.get("eval_mode")
        self.eval_mode = EvalMode(eval_mode).value
        self.name = "verifier"
        self.last_verify_step = 0
        self.verify_every = agent_config.get("verify_every", -1)

    def select_states(self, trajectory: TrajectoryView, num_states: int = -1) -> list[int]:
        # Return indices of the last `num_previous_state_actions` states
        num_states = min(self.num_previous_state_actions, len(trajectory.states))
        return list(range(len(trajectory.states) - num_states, len(trajectory.states)))

    def should_verify(self, trajectory: TrajectoryView) -> bool:
        if not self.online_verify:
            return False

        # If stop action, verify
        if trajectory.actions[-1]["action_type"] == ActionTypes.STOP:
            return True

        # Otherwise, verify every N steps if specified
        if self.verify_every > 0:
            valid_actions = trajectory.valid_actions
            if len(valid_actions) - self.last_verify_step >= self.verify_every:
                self.last_verify_step = len(valid_actions)
                return True
        return False

    def next_action(
        self,
        trajectory: TrajectoryView,
        intent: str,
        intent_images: list[ImageInput],
        meta_data: dict[str, Any],
    ) -> tuple[Any, str]:
        idxs_trajectory = self.select_states(trajectory)

        if self.mode == VerifierMode.NO_SGV.value or self.mode == VerifierMode.SGV.value:
            logger.info(f"Verifier, evaluating...")
            parsed_response: dict[str, Any]
            parsed_response, raw_response, num_tries = self.act_parse_retry(
                trajectory=trajectory,
                intent=intent,
                intent_images=intent_images,
                meta_data=meta_data,
                idxs_trajectory=idxs_trajectory,
                parser_fn=self.prompt_constructor.parse_verifier_response,
                max_tries=self.max_model_call,
                error_msg_template=self.prompt_constructor.instruction["parsing_error_msg_template"],
                error_msg_key=self.prompt_constructor.instruction["meta_data"]["parsing_error_msg_key"],
            )
            if meta_data.get("thought_summary"):
                parsed_response["thought_summary"] = meta_data["thought_summary"]
                meta_data["thought_summary"] = ""

            if self.out_utterance:
                if "thought_summary" in parsed_response:
                    logger.info(f"Verifier: <thought_summary> {parsed_response['thought_summary']} </thought_summary>")
                logger.info(f"\n[Verifier]: {raw_response}")

            return parsed_response, raw_response
        else:
            raise ValueError(f"Unknown verifier mode {self.mode}")

    def get_feedback(self, parsed_response: dict[str, Any]) -> str:
        return parsed_response[self.prompt_constructor.instruction["meta_data"]["feedback_key"]]  # type: ignore

    def get_eval_score(self, parsed_response: dict[str, Any]) -> str:
        return parsed_response[self.prompt_constructor.instruction["meta_data"]["eval_key"]]  # type: ignore

    def is_infeasible(self, parsed_response: dict[str, Any]) -> bool:
        feedback = clean_spaces(parsed_response[self.prompt_constructor.instruction["meta_data"]["feedback_key"]])
        # Look for feedback that contains only "NA" or "NONE" (with optional whitespace/punctuation)
        match = re.search(r"^\s*(NONE|NA)\s*[.,;!?]?\s*$", feedback, re.IGNORECASE)
        return match is not None

    def is_success(self, parsed_response: dict[str, Any]) -> bool:
        return self.get_eval_score(parsed_response) == self.success_keyword  # type: ignore

    def should_stop(self, parsed_response: dict[str, Any]) -> bool:
        if self.is_success(parsed_response):
            return True
        elif self.is_infeasible(parsed_response):
            return True
        else:
            return False

    def reset(self, config_dict: dict[str, Any], args: dict[str, Any] = {}) -> None:
        super().reset(config_dict, args)

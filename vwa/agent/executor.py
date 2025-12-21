import re
from typing import Any

from browser_env.actions import Action, ActionParsingError, create_id_based_action, create_none_action, create_playwright_action
from browser_env.env_utils import map_url_to_real
from utils_vwa.utils_vwa import TrajectoryView

from agent.agent import Agent
from agent.prompt_constructor import LMParsingError, PromptConstructor
from core_utils.logger_utils import logger
from core_utils.string_utils import clean_spaces, extract_urls, partial_format, safe_format
from core_utils.types import ImageInput
from llms.prompt_utils import get_message
from llms.tokenizer_utils import Tokenizer
from llms.types import Message


class ExecutorPromptConstructor(PromptConstructor):
    def __init__(self, lm_config: dict[str, Any], agent_config: dict[str, Any]):
        super().__init__(lm_config, agent_config)
        self.action_splitter: str = self.instruction["meta_data"]["action_splitter"]
        self.template_cur_observation: str = self.instruction["template_cur_observation"]
        self.ins_metadata: dict[str, Any] = self.instruction["meta_data"]
        self.parsing_error_msg_key: str = self.instruction["meta_data"]["parsing_error_msg_key"]
        self.parsing_error_msg_template: str = self.instruction["parsing_error_msg_template"]
        self.use_k_extraction = agent_config.get("use_k_extraction", False)
        self.agent_config = agent_config
        self.caption_input_images = agent_config.get("caption_input_images", True)

    def construct(
        self,
        trajectory: TrajectoryView,
        objective: str,
        objective_imgs: list[ImageInput],
        meta_data: dict[str, Any],
        idxs_trajectory: list[int] = [],
        **kwargs: Any,
    ) -> list[Message]:
        messages: list[Message] = []

        # System prompt
        sys_prompt = self.instruction.get("system_prompt", "")
        if meta_data.get("reflections") and self.agent_config.get("use_reflexion", True):
            sys_prompt = partial_format(
                sys_prompt,
                reflexion_info=self.instruction["reflexion_info"],
                reflexion_rule=self.instruction["reflexion_rule"],
            ).strip()
        else:
            sys_prompt = partial_format(sys_prompt, reflexion_info="", reflexion_rule="").strip()
        messages.append(get_message(inputs=sys_prompt, role="system"))

        # --- examples ---
        examples: list[list[dict[str, Any]]] = self.select_examples()

        # If Gemini, add a hint that examples are coming (cannot hint with name like 'example_user')
        if examples and "gemini" in self.lm_config["model"]:
            messages.append(
                get_message(
                    inputs="The following are examples of observations and the corresponding responses:\n",
                    role="user",
                    name=self.lm_config.get("name_user", ""),
                )
            )

        examples: list[list[dict[str, Any]]] = self.select_examples()
        # examples = [example1, example2, ...];
        # example = [{"text": str, "image": str, "utterance": str}, {"text": str, "image": str, "utterance": str}, ...]
        for _, example in enumerate(examples):
            intro_img_obs_user = "IMAGES: (1) current page screenshot"
            for state in example:
                user_msg = get_message(
                    inputs=[state["text"], intro_img_obs_user, state["image"]],
                    role="user",
                    name="example_user",
                    img_detail=self.lm_config.get("img_detail", "auto"),
                )
                assistant_msg = get_message(inputs=state["utterance"], role="assistant", name="example_assistant")
                messages.append(user_msg)
                messages.append(assistant_msg)

        if self.use_k_extraction and meta_data.get("k_extraction_response"):
            assert meta_data.get("k_extraction_response"), "K extraction response is required when use_k_extraction is True"
            messages.append(
                get_message(
                    inputs=f"""## General web knowledge:\n{meta_data["k_extraction_response"]}""",
                    role="user",
                    name=self.lm_config.get("name_user", ""),
                )
            )

        # --- Execution history ---
        prompt_prev_utterances = ""
        if self.ins_metadata["history_type"] == "interaction_history":
            messages.extend(self.build_interaction_history(trajectory, meta_data, idxs_trajectory))

        elif self.ins_metadata["history_type"] == "rationale_action":
            prompt_prev_utterances = self.build_rationale_action_history(trajectory, meta_data, idxs_trajectory)
        else:
            raise ValueError(f"Unknown history type: {self.ins_metadata['history_type']}")

        # -- Build current text input --
        text_input = self.build_current_text_input(trajectory, meta_data, objective, prompt_prev_utterances)
        # If the last action is invalid, add the error message to the text input
        if len(trajectory.actions) > 0 and "invalid" in trajectory.actions[-1]:
            text_input += self.instruction["env_parsing_error_msg_template"].format(error_msg=meta_data["action_str_history"][-1])

        if meta_data.get("verifier_feedback"):
            text_input += self.instruction["feedback_template"].format(
                previous_response=meta_data["prev_utterance_for_feedback"],
                feedback=meta_data["verifier_feedback"],
            )
            # Clear feedback fields
            meta_data["verifier_feedback"] = ""
            meta_data["prev_utterance_for_feedback"] = ""

        if meta_data.get("reflections"):
            reflection_str = ""
            if len(meta_data["reflections"]) == 1:
                reflection_str = meta_data["reflections"][0]
            else:
                for i, reflection in enumerate(meta_data["reflections"]):
                    reflection_str += f"### REFLECTION {i + 1}\n{reflection}\n"

            meta_data["reflections"] = ""
            text_input += self.instruction["reflection_injection_template"].format(reflections=reflection_str)

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
                state_img_intros=["current webpage screenshot at state `t`"],
                add_states_idxs=[-1],
                add_state_text=False,
                add_state_img=True,
                role="user",
                name=self.lm_config.get("name_user", ""),
            )
        )
        return messages

    def select_examples(self) -> list[list[dict[str, Any]]]:
        # NOTE: possible to retrieve examples dynamically
        return self.instruction["examples"]  # type: ignore

    def build_current_text_input(self, trajectory: TrajectoryView, meta_data: dict[str, Any], objective: str, prompt_prev_utterances: str = "") -> str:
        state_info = trajectory.states[-1]
        last_action = trajectory.actions[-1] if trajectory.actions else {}
        text_obs = state_info["observation"]["text"]
        url = state_info["info"]["page"].url

        use_text_obs = self.use_text_observation
        if url.endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp")):
            use_text_obs = False

        if not use_text_obs:
            # If not using text observation, add only the open tabs
            open_tabs = state_info["info"]["observation_metadata"]["text"]["open_tabs"]
            if open_tabs:
                text_obs = clean_spaces(open_tabs)
            else:
                text_obs = ""

        if last_action.get("action_output"):
            last_action_str = clean_spaces(meta_data["action_str_history"][-1])
            text_obs = text_obs + "\n\n" + f"## {last_action_str}"

        if self.ins_metadata["history_type"] == "rationale_action":
            text_input = safe_format(
                string_template=self.template_cur_observation,
                rationale_action_history=prompt_prev_utterances,
                text_obs=text_obs,
                url=map_url_to_real(url),
                objective=objective,
            )
        else:
            raise ValueError("History type not implemented")
        return text_input

    def regularize_action(self, parsed_response: str) -> str:
        try:
            # regularize ```goto``` action
            if re.search("goto", parsed_response, re.IGNORECASE):
                # Added because gemini 2.5 has a lot of trouble in generating goto [url] actions, despite direct instruction
                # Obs: Not necessary for other models (incl. gemini 2.0, 1.5, etc)

                action_wrapper = self.ins_metadata.get("args_wrapper", "[]")
                urls = extract_urls(parsed_response)
                if urls:
                    action_wrapper = self.ins_metadata.get("args_wrapper", "[]")
                    url = clean_spaces(urls[-1]).strip("[]")
                    if url:
                        return f"goto {action_wrapper[0]}{url}{action_wrapper[-1]}"
                    else:
                        return parsed_response
                else:
                    # If no urls, return as is
                    return parsed_response

            elif re.search("select", parsed_response, re.IGNORECASE):
                # Replace select for click
                return re.sub("select", "click", parsed_response, flags=re.IGNORECASE)

            else:
                return parsed_response
        except Exception as _:
            # If error, return as is.
            return parsed_response

    def extract_action(self, response: str) -> str:
        # find the first occurence of action
        action_splitter = self.action_splitter
        pattern = rf"{action_splitter}((.|\n)*?){action_splitter}"
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            parsed_action = match.group(1).strip()
            return self.regularize_action(parsed_action)
        else:
            raise LMParsingError(f'Cannot find the action identifier "{action_splitter}" in "{response}"')


class Executor(Agent):
    """prompt-based agent that emits action given the history"""

    def __init__(self, lm_config: dict[str, Any], agent_config: dict[str, Any]) -> None:
        self.lm_config = lm_config
        self.tokenizer = Tokenizer(model_name=lm_config["model"])
        self.num_previous_state_actions = agent_config["num_previous_state_actions"]
        self.action_set_tag = agent_config["action_set_tag"]
        self.max_model_call = agent_config["max_model_call"]
        self.out_utterance = agent_config.get("out_utterance", True)
        self.conversation_dir = agent_config.get("conversation_dir", "")
        self.usage_dir = agent_config.get("usage_dir", "")
        self.prompt_constructor: ExecutorPromptConstructor = ExecutorPromptConstructor(lm_config=lm_config, agent_config=agent_config)
        self.agent_config = agent_config
        self.use_k_extraction = agent_config.get("use_k_extraction", False)
        self.use_reflexion = agent_config.get("use_reflexion", False)
        self.num_previous_reflections = agent_config["num_previous_reflections"]
        self.name = "executor"

    def select_states(self, trajectory: TrajectoryView, num_states: int = -1) -> list[int]:
        # Return indices of all states except the last one
        num_states = min(self.num_previous_state_actions, len(trajectory.states) - 1)
        return list(range(len(trajectory.states) - num_states - 1, len(trajectory.states) - 1))

    def select_previous_reflections(self, prev_reflections: list[str], num_previous_reflections: int) -> list[str]:
        if not prev_reflections:
            return []
        if num_previous_reflections == -1:
            return prev_reflections
        if num_previous_reflections > 0:
            prev_reflections = prev_reflections[-num_previous_reflections:]
        return prev_reflections

    def next_action(
        self,
        trajectory: TrajectoryView,
        intent: str,
        intent_images: list[ImageInput],
        meta_data: dict[str, Any],
    ) -> Action:
        idxs_trajectory = self.select_states(trajectory)
        total_tries = self.max_model_call

        if not self.use_reflexion:
            meta_data["reflections"] = []
        else:
            meta_data["reflections"] = self.select_previous_reflections(meta_data.get("reflections", []), self.num_previous_reflections)

        while True:
            # Call model and parse response. Obs.: This is just text parsing, done by the PromptConstructor.
            try:
                parsed_response, raw_response, num_tries = self.act_parse_retry(
                    trajectory=trajectory,
                    intent=intent,
                    intent_images=intent_images,
                    meta_data=meta_data,
                    idxs_trajectory=idxs_trajectory,
                    parser_fn=self.prompt_constructor.extract_action,
                    max_tries=total_tries,
                    error_msg_template=self.prompt_constructor.parsing_error_msg_template,
                    error_msg_key=self.prompt_constructor.parsing_error_msg_key,
                )

            # Failed to parse the action with the splitters more than `max_model_call` times
            except LMParsingError as e:
                # If fails more than `max_model_call` times, create a NONE action for environment feedback in next iteration
                raw_response = e.raw_response
                logger.info(e.message)
                action = create_none_action()
                action.update({"raw_prediction": raw_response, "wait_for": 0, "early_stop": e.message})  # type: ignore
                break

            total_tries = total_tries - num_tries

            # Create action -- environment-specific; environment will parse the action.
            try:
                parsed_response = self.prompt_constructor.map_links_to_local(parsed_response)
                action = self.create_action(parsed_response)
                action.update({"raw_prediction": raw_response})
                if meta_data.get("thought_summary"):
                    action.update({"thought_summary": meta_data["thought_summary"]})  # type: ignore
                    meta_data["thought_summary"] = ""
                break

            except ActionParsingError as _:
                # If emits an invalid action, create a NONE action for environment feedback in next iteration
                action = create_none_action()
                action.update({"raw_prediction": raw_response, "wait_for": 0})  # type: ignore
                logger.warning(f"Invalid action emitted by {self.__str__()}.\nRaw generation: {raw_response}")
                break

            except Exception as e:
                raise e

        logger.info(f"\n[Executor Agent]: {raw_response}") if self.out_utterance else None
        return action

    def create_action(self, action_str: str) -> Action:
        try:
            if self.action_set_tag == "id_accessibility_tree":
                action = create_id_based_action(action_str)
            elif self.action_set_tag == "playwright":
                action = create_playwright_action(action_str)
            elif self.action_set_tag == "som":
                action = create_id_based_action(action_str)
            else:
                raise ValueError(f"Unknown action type {self.action_set_tag}.")
            action["parsed_action"] = action_str  # type: ignore
            return action
        except ActionParsingError as e:
            raise e
        except Exception as e:
            raise e

    def get_action_splitter(self, agent_id: str = "executor") -> str:
        return self.prompt_constructor.instruction["meta_data"]["action_splitter"]  # type:ignore

    def get_tokenizer(self, agent_id: str = "executor") -> Tokenizer:
        return self.prompt_constructor.tokenizer  # type: ignore

    def set_action_set_tag(self, tag: str) -> None:
        self.action_set_tag = tag

    def reset(self, config_dict: dict[str, Any], args: dict[str, Any] = {}) -> None:
        super().reset(config_dict, args)

from typing import Any, Callable

from browser_env.actions import ActionTypes, action2str
from utils_vwa.score_logger import ScoreLogger
from utils_vwa.utils_vwa import TrajectoryView

from agent.agent import Agent
from agent.executor import Executor
from agent.k_extraction import KExtraction
from agent.prompt_constructor import PromptConstructor
from agent.reflexion import Reflexion
from agent.request_refiner import RequestRefiner
from agent.text_refiner import TextRefiner
from agent.verifier import Verifier
from core_utils.logger_utils import logger
from core_utils.string_utils import clean_spaces
from core_utils.timing_utils import timeit
from core_utils.types import ImageInput
from llms.tokenizer_utils import Tokenizer


class ModularAgent(Agent):
    def __init__(
        self,
        text_refiner: TextRefiner | None,
        executor_agent: Executor,
        request_refiner: RequestRefiner | None,
        verifier: Verifier | None,
        reflexion: Reflexion | None,
        k_extraction: KExtraction | None,
    ):
        self.text_refiner = text_refiner
        self.executor_agent = executor_agent
        self.request_refiner = request_refiner
        self.verifier = verifier
        self.reflexion = reflexion
        self.k_extraction = k_extraction
        self.all_modules = [
            self.text_refiner,
            self.executor_agent,
            self.request_refiner,
            self.verifier,
            self.reflexion,
            self.k_extraction,
        ]
        self.score_logger: ScoreLogger = ScoreLogger()

    @timeit(custom_name="AGENT:next_action")
    def next_action(
        self,
        trajectory: TrajectoryView,
        intent: str,
        intent_images: list[ImageInput],
        meta_data: dict[str, Any],
    ) -> Any:
        # Refine user request
        if self.request_refiner is not None:
            # NOTE currently just caption input img if provided; possible to do more, like paraphrase intent, ask for clarification, etc.
            new_intent = self.request_refiner.next_action(trajectory, intent, intent_images, meta_data)

        # Refine text obs
        if self.text_refiner is not None:
            self.text_refiner.next_action(trajectory, intent, intent_images, meta_data)

        # Execute low-level action on environment
        verifier_executor_loop = 0
        utterance_history: dict[str, list[str]] = {"executor": [], "verifier": []}

        # If executor using web knowledge, retrieve web knowledge at initiation
        if self.k_extraction is not None and self.executor_agent.use_k_extraction:
            self.k_extraction.next_action(trajectory, intent, intent_images, meta_data)
            meta_data["k_extraction_response"] = self.k_extraction.get_k_extraction_response()

        # If executor using reflexion, retrieve reflections from memory
        if self.reflexion is not None and self.executor_agent.use_reflexion:
            meta_data["reflections"] = self.reflexion.get_from_memory(meta_data["uid"])

        while True:
            logger.info(f"Executor Agent action")
            low_level_action = self.executor_agent.next_action(trajectory, intent, intent_images, meta_data)
            if "early_stop" in low_level_action:
                break

            utterance_history["executor"].append(low_level_action["raw_prediction"])

            # If Verifier provided, start executor-verifier loop
            if self.verifier is not None:
                verifier_executor_loop += 1

                # Stop the executor-verifier loop if max_verifier_executor_loop is reached
                if verifier_executor_loop > self.verifier.max_verifier_executor_loop:
                    break

                # Create a temporary trajectory with the low-level action
                temp_trajectory = TrajectoryView(trajectory.trajectory + [low_level_action])
                action_str = action2str(low_level_action, self.executor_agent.action_set_tag)
                temp_meta_data = meta_data.copy()
                temp_meta_data["action_str_history"] = meta_data["action_str_history"] + [action_str]

                if not self.verifier.should_verify(temp_trajectory):
                    break

                # Call Verifier
                verifier_should_stop, verifier_feedback, raw_verifier_response, verifier_parsed_resp = self.get_verifier_response(temp_trajectory, intent, intent_images, temp_meta_data)
                if "thought_summary" in verifier_parsed_resp:
                    raw_verifier_response = f"<thought_summary> {verifier_parsed_resp['thought_summary']} </thought_summary>\n{raw_verifier_response}"

                utterance_history["verifier"].append(raw_verifier_response)

                low_level_action.update({"verifier_executor_loop_utterances": utterance_history})  # type: ignore

                # If it is a stop action, log scores (this provides a no-verifier baseline for free when running experiments with verifier)
                if self.score_logger and "oracle" not in self.verifier.eval_mode and low_level_action["action_type"] == ActionTypes.STOP:
                    oracle_score = self.score_logger.log_scores_per_round(temp_trajectory, intent, meta_data, int(verifier_should_stop))
                    # if NaN, raise an exception
                    if oracle_score != oracle_score:
                        raise Exception(f"Failed to compute oracle score during verifier-executor loop: {oracle_score}")

                if verifier_should_stop:
                    break
                else:
                    meta_data["verifier_feedback"] = clean_spaces(verifier_feedback)
                    raw_prediction = low_level_action["raw_prediction"]
                    if "thought_summary" in low_level_action:
                        raw_prediction = f"<thought>{low_level_action['thought_summary']}</thought>\n{raw_prediction}"
                    meta_data["prev_utterance_for_feedback"] = clean_spaces(raw_prediction)
            else:
                break

        if utterance_history["executor"] and utterance_history["verifier"]:
            low_level_action["verifier_executor_loop_utterances"] = utterance_history  # type: ignore

        return low_level_action

    def get_verifier_response(self, trajectory: TrajectoryView, intent: str, intent_images: list[Any], meta_data: dict[str, Any]) -> tuple[bool, str, str, dict[str, Any]]:
        if self.verifier is None:
            raise ValueError("No verifier agent provided")

        if self.verifier.mode == "sgv" and "model" in self.verifier.eval_mode:
            if not self.k_extraction:
                raise ValueError("Verifier SGV mode requires k extractor")
            self.k_extraction.next_action(trajectory, intent, intent_images, meta_data)
            meta_data["k_extraction_response"] = self.k_extraction.get_k_extraction_response()

        if self.verifier.eval_mode == "model":
            verifier_response, raw_verifier_response = self.verifier.next_action(trajectory, intent, intent_images, meta_data)
            return (
                self.verifier.should_stop(verifier_response),
                self.verifier.get_feedback(verifier_response),
                raw_verifier_response,
                verifier_response,
            )

        elif "oracle" in self.verifier.eval_mode:
            score = self.score_logger.log_scores_per_round(trajectory, intent, meta_data)
            if score == 1:
                should_stop = True
            elif score == 0:
                should_stop = False
            else:
                raise Exception(f"Failed to compute oracle score: {score}")
            feedback = raw_verifier_response = ""
            verifier_response: dict[str, Any] = {}

            # If not should stop, get feedback
            if not should_stop:
                if "model_feedback" in self.verifier.eval_mode:
                    verifier_response, raw_verifier_response = self.verifier.next_action(trajectory, intent, intent_images, meta_data)
                    feedback = f"The task is likely not accomplished. {self.verifier.get_feedback(verifier_response)}"
                    logger.info(f"Oracle verifier, model feedback: {feedback}")
                else:
                    feedback = "The task is likely not accomplished. Please revise your plan and continue."
                    raw_verifier_response = f"Oracle feedback: {feedback}"
                    logger.info(f"Oracle verifier, no model feedback: {feedback}")

            return (
                should_stop,
                feedback,
                raw_verifier_response,
                verifier_response,
            )
        else:
            raise ValueError(f"Unknown eval mode: {self.verifier.eval_mode}")

    def generate_reflection(self, trajectory: TrajectoryView, intent: str, intent_images: list[Any], meta_data: dict[str, Any]) -> Any:
        if self.reflexion is not None:
            return self.reflexion.next_action(trajectory, intent, intent_images, meta_data)
        else:
            raise ValueError("No reflexion agent provided")

    def get_prompt_constructor(self) -> PromptConstructor:
        if self.executor_agent:
            return self.executor_agent.prompt_constructor
        else:
            raise ValueError("No executor agent provided")

    def get_provider(self, module_name: str = "executor") -> str:
        if module_name == "executor" and self.executor_agent:
            return self.executor_agent.tokenizer.provider
        else:
            raise NotImplementedError(f"Only implemented for 'executor'")

    def get_action_splitter(self, agent_id: str = "executor") -> str:
        if agent_id == "executor" and self.executor_agent:
            return self.executor_agent.get_action_splitter()
        else:
            raise ValueError(f"Unknown agent {agent_id}")

    def get_tokenizer(self, agent_id: str = "executor") -> Tokenizer:
        if agent_id == "executor" and self.executor_agent:
            return self.executor_agent.get_tokenizer()
        else:
            raise ValueError(f"Unknown agent {agent_id}")

    def reset(self, config_dict: dict[str, Any], args: dict[str, Any] = {}) -> None:
        for module in self.all_modules:
            if module is not None:
                module.reset(config_dict, args)


def construct_modular_agent(
    agents_configs: dict[str, Any],
    caption_image_fn: Callable,  # type: ignore
) -> ModularAgent:
    # Load configurations

    if "executor" not in agents_configs:
        raise ValueError("Please provide a low-level executor module.")
    agent_config = agents_configs["executor"]
    executor_agent = Executor(
        lm_config=agent_config["lm_config"],
        agent_config=agent_config,
    )

    text_refiner = None
    if "text_refiner" in agents_configs:
        agent_config = agents_configs["text_refiner"]
        text_refiner = TextRefiner(
            lm_config=agent_config["lm_config"],
            agent_config=agent_config,
        )

    request_refiner = None
    if "request_refiner" in agents_configs:
        request_refiner = RequestRefiner(
            agent_config=agents_configs["request_refiner"],
            captioning_fn=caption_image_fn,
        )

    verifier = None
    if "verifier" in agents_configs:
        agent_config = agents_configs["verifier"]
        verifier = Verifier(
            lm_config=agent_config["lm_config"],
            agent_config=agent_config,
        )

    k_extraction = None
    if "k_extraction" in agents_configs:
        agent_config = agents_configs["k_extraction"]
        k_extraction = KExtraction(
            lm_config=agent_config["lm_config"],
            agent_config=agent_config,
        )

    reflexion = None
    if "reflexion" in agents_configs:
        agent_config = agents_configs["reflexion"]
        reflexion = Reflexion(
            lm_config=agent_config["lm_config"],
            agent_config=agent_config,
        )

    return ModularAgent(
        text_refiner=text_refiner,
        executor_agent=executor_agent,
        request_refiner=request_refiner,
        verifier=verifier,
        reflexion=reflexion,
        k_extraction=k_extraction,
    )

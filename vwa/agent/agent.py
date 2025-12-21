from typing import Any, Callable

import yaml
from utils_vwa.score_logger import ScoreLogger
from utils_vwa.utils_vwa import TrajectoryView

from agent.prompt_constructor import LMParsingError, PromptConstructor
from core_utils.logger_utils import logger
from core_utils.string_utils import clean_spaces
from core_utils.timing_utils import time_block
from core_utils.types import ImageInput
from llms.llm_utils import call_llm


class Agent:
    """Base class for the agent"""

    def __init__(self, *args: Any) -> None:
        """Initialize the agent

        Args:
            lm_config (dict[str, Any]): Configuration for the model used by the agent
            agent_config (dict[str, Any]): Agent-specific configuration settings
            prompt_constructor (PromptConstructor): Prompt constructor used by the agent to construct prompts during navigation
            score_logger (ScoreLogger | None): Computes and logs oracle scores at chosen states
            usage_dir (str): Directory path to store API-specific metrics
            conversation_dir (str): Directory path to store user-model conversation logs
            call_id (str): Logs are stored as {call_id}.html, {call_id}.txt, {call_id}.csv
            name (str): Name of the agent
            dump_html (bool): Whether to an HTML with conversation history
        """
        self.lm_config: dict[str, Any] = {}
        self.agent_config: dict[str, Any] = {}
        self.prompt_constructor: PromptConstructor = PromptConstructor(self.lm_config, self.agent_config)
        self.score_logger: ScoreLogger | None = None
        self.dump_html: bool = True
        self.conversation_dir: str = ""
        self.usage_dir: str = ""
        self.call_id: str = ""
        self.name: str = ""

    def next_action(
        self,
        trajectory: TrajectoryView,
        intent: str,
        intent_images: list[ImageInput],
        meta_data: dict[str, Any],
    ) -> Any:
        """Predict the next action given the observation"""
        raise NotImplementedError

    def select_states(self, trajectory: TrajectoryView, num_states: int) -> list[Any]:
        """Basic state selection of the last `num_states` states.
        Agents can override this method with more sophisticated state selection logic.

        Args:
            trajectory (TrajectoryView): Trajectory of states
            num_states (int): Number of states to select

        Returns:
            list: List of states to select
        """
        num_states = max(1, num_states)
        if num_states == 1:
            return [trajectory.states[-1]]
        else:
            return trajectory.states[-num_states:]

    def act_parse_retry(
        self,
        trajectory: TrajectoryView,
        intent: str,
        intent_images: list[ImageInput],
        meta_data: dict[str, Any],
        idxs_trajectory: list[int],
        parser_fn: Callable[[str], Any],
        max_tries: int,
        error_msg_template: str,
        error_msg_key: str,
        **kwargs: Any,
    ) -> tuple[Any, str, int]:
        """Build prompt, call model, parse response, and retry on failure.
        Try for `max_tries` times, appending error message to prompt in case of parsing failure.

        Args:
            trajectory (TrajectoryView): Trajectory of states
            intent (str): User objective
            intent_images (list[Image.Image]): Images of the objective
            meta_data (dict[str, Any]): Metadata
            idxs_trajectory (list[int]): Indices of the trajectory to use
            parser_fn (callable): Function to parse the model response
            max_tries (int): Maximum number of tries
            error_msg_key (str): Key used to store the error message in meta_data for the agent's prompt constructor retrieval during prompt construction
            error_msg_template (str): Template for the error message
            **kwargs: Additional arguments for the prompt constructor

        Raises:
            LMParsingError: If the response cannot be parsed after `max_tries` attempts

        Returns:
            tuple[Any, str, int]: Parsed response, raw response, number of tries
        """
        num_tries = 0
        while num_tries < max_tries:
            try:
                # Build API messages with feedback from previous attempts
                with time_block("AGENT:construct_prompt"):
                    prompt = self.prompt_constructor.construct(
                        trajectory=trajectory,
                        objective=intent,
                        objective_imgs=intent_images,
                        meta_data=meta_data,
                        idxs_trajectory=idxs_trajectory,
                        **kwargs,
                    )

                # Clean feedback from previous attempts
                meta_data[error_msg_key] = "" if error_msg_key in meta_data else None

                # Call model and parse response
                with time_block("AGENT:call_llm"):
                    _, model_generations = call_llm(
                        gen_kwargs=self.lm_config,
                        prompt=prompt,
                        meta_data=meta_data,
                        call_id=self.call_id,
                        conversation_dir=self.conversation_dir,
                        usage_dir=self.usage_dir,
                        dump_html=self.dump_html,
                    )

                # TODO: reintroduce multiple generations handling
                raw_response = model_generations[0].text()
                meta_data["thought_summary"] = model_generations[0].thoughts()
                parsed_response = parser_fn(raw_response)
                if hasattr(self, "dump_html"):
                    self.dump_html = False  # Dump only one time to save execution time
                return parsed_response, raw_response, num_tries

            # If parsing fails, append error message to prompt and try again
            except LMParsingError as e:
                error_message = error_msg_template.format(response=clean_spaces(raw_response))
                meta_data[error_msg_key] = error_message
                logger.warning(f"Failed to parse model response for {self.__str__()}: {e}.\nRaw generation: {raw_response}")
                num_tries += 1

            # If any other error occurs, raise it
            except Exception as e:
                logger.error(f"Error in act_parse_retry: {e}")
                meta_data[error_msg_key] = "" if error_msg_key in meta_data else None
                raise e

        # If parsing still fails after `max_tries` attempts, raise error
        meta_data[error_msg_key] = "" if error_msg_key in meta_data else None
        raise LMParsingError(f"Failed to parse {self.__str__()} response after {max_tries} attempts.", raw_response=raw_response)

    def __str__(self) -> str:
        """String method to print agent name"""
        if self.name:
            return self.name
        return f"{self.__class__.__name__}"

    def reset(self, config_dict: dict[str, Any], args: dict[str, Any] = {}) -> None:
        """Reset the agent. Agents can override this method with agent-specific reset logic.

        Args:
            config_dict (dict[str, Any]): Configuration dictionary
        """
        self.dump_html = True
        task_id = config_dict["task_id"]
        domain = config_dict["domain"]
        env_name = config_dict["env"]
        uid = f"{domain}_{task_id}_{env_name}"
        results_dir = args["result_dir"]
        attempt_num = args["attempts_per_task"][uid]
        task_id_attempt = f"{task_id}_{attempt_num}"
        self.conversation_dir = f"{results_dir}/trajectories/{task_id_attempt}/conversations"
        self.usage_dir = f"{results_dir}/trajectories/{task_id_attempt}/lm_usage"
        self.call_id = f"{self.name}_{task_id}"


def load_agent_config(agent_config_path: str) -> dict[str, Any]:
    with open(agent_config_path, "r") as f:
        agent_config = yaml.safe_load(f)  # type: ignore
    return agent_config  # type: ignore

from typing import Any

from utils_vwa.utils_vwa import TrajectoryView

from agent.agent import Agent
from core_utils.types import ImageInput
from llms.tokenizer_utils import Tokenizer


class TextRefiner(Agent):
    def __init__(self, lm_config: dict[str, Any], agent_config: dict[str, Any]) -> None:
        self.tokenizer = Tokenizer(model_name=lm_config["model"])
        self.lm_config = lm_config
        self.max_obs_length = agent_config["max_obs_length"]
        self.agent_config = agent_config
        self.name = "text_refiner"

    def select_states(self, trajectory: TrajectoryView, num_states: int = 1) -> list[dict[str, Any]]:
        return super().select_states(trajectory, num_states=num_states)

    def next_action(self, trajectory: TrajectoryView, intent: str, intent_images: list[ImageInput], meta_data: dict[str, Any]) -> None:
        env_state = self.select_states(trajectory)[0]
        text_obs = env_state["observation"]["text"]
        env_state["observation"]["text"] = self.refine_text_obs(text_obs)

    def refine_text_obs(self, text_observation: str) -> str:
        # Trim text observation to `max_obs_length`
        if self.max_obs_length > 0:
            if self.tokenizer.provider == "google":
                # If Gemini, trim per character
                text_observation = text_observation[: self.max_obs_length * self.agent_config["chars_per_token"]]
            elif self.tokenizer.provider == "anthropic":
                # If Anthropic, trim per character
                text_observation = text_observation[: self.max_obs_length * self.agent_config["chars_per_token"]]
            else:
                # If other, tokenize and trim
                tokenized_obs = self.tokenizer.encode(text_observation, add_special_tokens=False)
                text_observation = self.tokenizer.decode(tokenized_obs[: self.max_obs_length], skip_special_tokens=False)
        return text_observation

    def reset(self, config_dict: dict[str, Any], args: dict[str, Any] = {}) -> None:
        super().reset(config_dict, args)

from typing import Any, Callable, OrderedDict

from utils_vwa.utils_vwa import TrajectoryView

from agent.agent import Agent
from core_utils.image_utils import any_to_pil
from core_utils.types import ImageInput

# This is an Agent responsible to rewriting the user request.
# In theory, it should be able to do anything with the intent, such as paraphrasing it, asking for clarification from the user, etc.


class RequestRefiner(Agent):
    def __init__(self, agent_config: dict[str, Any], captioning_fn: Callable | None = None) -> None:  # type: ignore
        self.agent_config = agent_config
        self.captioning_fn = captioning_fn
        self.caption_input_img = agent_config["caption_input_img"]
        self.image_captions: OrderedDict[int, str] = OrderedDict()
        self.name = "request_refiner"

    def next_action(
        self,
        trajectory: TrajectoryView,
        intent: str,
        intent_images: list[ImageInput],
        meta_data: dict[str, Any],
    ) -> str:
        # Possibly other actions. Example: paraphrase intent, model-user clarification loop based on uncertainty measure
        # [..code..]
        if self.caption_input_img:
            return self.caption_request_images(intent, intent_images, meta_data)
        else:
            return intent

    def caption_request_images(self, intent: str, intent_images: list[ImageInput], meta_data: dict[str, Any]) -> str:
        # Caption query input images, if provided.
        if intent_images is None or len(intent_images) == 0 or self.captioning_fn is None:
            return intent

        # Caption images
        image_input_caption = ""
        for image_i, image in enumerate(intent_images):
            image = any_to_pil(image)
            img_hash = hash(image.tobytes())
            if img_hash not in self.image_captions:
                self.image_captions[img_hash] = self.captioning_fn([image])[0]

            image_input_caption += f'image {image_i + 1}: "{self.image_captions[img_hash]}"'
            if len(intent_images) > 1:
                image_input_caption += "; "

        # Update intent to include captions of query images.
        intent = f"{intent}\nObjective image descriptions: {image_input_caption}"
        meta_data["intent_images_captions"] = self.image_captions
        return intent

    def reset(self, config_dict: dict[str, Any], args: dict[str, Any] = {}) -> None:
        super().reset(config_dict, args)
        self.image_captions = OrderedDict()

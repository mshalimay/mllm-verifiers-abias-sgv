import re
from typing import Any, Dict

from core_utils.image_utils import any_to_b64, any_to_pil, compress_to_target_size, get_image_size_bytes, resize_image
from core_utils.logger_utils import logger
from llms.providers.hugging_face.setup_utils import get_max_image_size_per_model, get_max_img_dims_per_model
from llms.types import ContentItem


class ModelPrompter:
    """
    Encapsulates methods to convert model inputs to the correct format for Hugging Face models.
    `openai` mode: assumes it is using a third-party provider that uses the OpenAI client and chat completion format,
    NOTE: hugging face uses the openai chat completion, but with different keys and dict format.
    """

    def __init__(self) -> None:
        pass

    @classmethod
    def to_model(
        cls,
        model_id: str,
        input: ContentItem,
        engine: str,
        provider: str = "",
    ) -> Dict[str, Any]:
        """Route to the correct model prompter based on the model ID."""

        if re.match(r"Qwen/Qwen2\.5-VL-.*", model_id, flags=re.IGNORECASE):
            return cls.to_qwen_vl_2_5(model_id, input, engine, provider)

        elif re.search(r"Kimi-VL", model_id, flags=re.IGNORECASE):
            return cls.to_kimi_vl(model_id, input, engine, provider)

        else:
            return cls.to_open_ai_chat_completion(model_id, input, engine, provider)

    @classmethod
    def to_open_ai_chat_completion(
        cls,
        model_id: str,
        input: ContentItem,
        engine: str,
        provider: str = "",
    ) -> Dict[str, Any]:
        # Textual input
        if input.type == "text":
            return {"type": "text", "text": input.data}

        # Image input
        elif input.type == "image":
            if engine == "openai":
                # Vanilla OpenAI format
                max_img_size_mb = get_max_image_size_per_model(model_id, provider)
                max_img_dims = get_max_img_dims_per_model(model_id, provider)
                imgb64 = any_to_b64(input.data, add_header=True)
                if not max_img_size_mb and not max_img_dims:
                    return {"type": "image_url", "image_url": {"url": imgb64}}

                if max_img_dims:
                    img_dims = any_to_pil(imgb64).size
                    if img_dims[0] > max_img_dims[0] or img_dims[1] > max_img_dims[1]:
                        logger.info(f"Image dimensions {img_dims} exceed max dimensions {max_img_dims}, compressing...")
                        imgb64 = any_to_b64(resize_image(imgb64, target_size=(max_img_dims[0], max_img_dims[1])), add_header=True)

                img_size_mb = get_image_size_bytes(imgb64) / 1024 / 1024
                if img_size_mb > max_img_size_mb:
                    logger.info(f"Image size {img_size_mb} exceeds max size {max_img_size_mb} MB, compressing...")
                    imgb64 = any_to_b64(compress_to_target_size(input.data, target_size_mb=max_img_size_mb, encoding="base64"), add_header=True)

                # Update input data to not compress/resize again later
                input.data = imgb64
                return {"type": "image_url", "image_url": {"url": imgb64}}

            else:
                # Hugging Face OpenAI format
                img_b64 = any_to_b64(input.data, add_header=True)
                # https://huggingface.co/docs/transformers/main/en/chat_templating_multimodal#image-inputs
                return {"type": "image", "base64": img_b64}

        # Video input
        elif input.type == "video":
            return {"type": "video", "path": input.data}

        else:
            raise NotImplementedError(f"{__file__}: Type: {input.type} not implemented for OpenAI chat completion with Hugging Face")

    @classmethod
    def to_kimi_vl(cls, model_id: str, input: ContentItem, engine: str, provider: str) -> Dict[str, Any]:
        """Kimivl specific prompt format."""

        if engine == "openai":
            return cls.to_open_ai_chat_completion(model_id=model_id, input=input, engine=engine, provider=provider)

        if input.type == "text":
            return {"type": "text", "text": input.data}

        elif input.type == "image":
            if engine == "server" or engine == "vllm" or engine == "tgi":
                img = any_to_b64(input.data, add_header=True)
            else:
                img = any_to_pil(input.data)
            return {"type": "image", "image": img}
        else:
            raise NotImplementedError(f"{__file__}: Type: {input.type} not implemented for KimiVL with Hugging Face")

    @classmethod
    def to_qwen_vl_2_5(cls, model_id: str, input: ContentItem, engine: str, provider: str) -> Dict[str, Any]:
        """Qwen2.5 VL specific prompt format."""

        if engine == "openai":
            return cls.to_open_ai_chat_completion(model_id=model_id, input=input, engine=engine, provider=provider)

        if input.type == "text":
            return {"type": "text", "text": input.data}

        elif input.type == "image":
            add_args = {}
            if min_pixels := input.meta_data.get("min_pixels", None):
                add_args["min_pixels"] = min_pixels
            if max_pixels := input.meta_data.get("max_pixels", None):
                add_args["max_pixels"] = max_pixels

            if engine == "server" or engine == "vllm" or engine == "tgi" or engine == "openai":
                img = any_to_b64(input.data, add_header=True)
            else:
                img = any_to_pil(input.data)
            return {"type": "image", "image": img, **add_args}
        else:
            raise NotImplementedError(f"{__file__}: Type: {input.type} not implemented for Qwen2.5 VL with Hugging Face")

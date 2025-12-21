from typing import Any, Dict, List

from core_utils.image_utils import any_to_b64, compress_to_target_size
from core_utils.logger_utils import logger
from llms.generation_config import GenerationConfig
from llms.prompt_utils import get_conversation_img_payload_size
from llms.providers.openai.constants import MAX_IMG_PAYLOAD_SIZE_MB, ROLE_MAPPINGS
from llms.types import ContentItem, Message


class OpenAIPrompter:
    """
    A class to encapsulate prompt adjustments for OpenAI API generation.

    This class handles:
      - Converting a content item to a genai Part (text, image, etc.)
      - Regularizing a list of content items
      - Regularizing messages (including handling system vs. non-system messages)
      - Adjusting the message role according to OpenAI's API
    """

    def __init__(self) -> None:
        pass

    @classmethod
    def _compress_image(cls, input: ContentItem, target_size_mb: float) -> str:
        try:
            compressed_img = compress_to_target_size(input.data, target_size_mb=target_size_mb, encoding="base64")
            img_b64 = any_to_b64(compressed_img, add_header=True)
        except Exception as e:
            img_b64 = any_to_b64(input.data, add_header=True)
            logger.warning(f"Image compression failed ({e}); using original image payload")
        return img_b64

    @classmethod
    def img_to_provider(cls, input: ContentItem, mode: str, role: str, compress_img: tuple[bool, float]) -> Dict[str, Any]:
        # Maybe compress image for model payload size limits
        if compress_img[0] and compress_img[1] > 0:
            target_size_mb = float(compress_img[1])
            logger.info(f"Compressing image to target size {target_size_mb} MB for OpenAI prompt")
            img_b64 = cls._compress_image(input, target_size_mb)
        else:
            img_b64 = any_to_b64(input.data, add_header=True)

        img_detail = input.meta_data.get("img_detail", "auto")
        if mode == "chat_completion":
            return {"type": "image_url", "image_url": {"url": img_b64, "detail": img_detail}}
        elif mode == "response":
            _ty = "input_image" if role != "assistant" else "output_image"
            return {"type": _ty, "image_url": img_b64, "detail": img_detail}
        else:
            raise ValueError(f"Unknown mode: {mode}")

    @classmethod
    def text_to_provider(cls, input: ContentItem, mode: str, role: str) -> Dict[str, Any]:
        if mode == "chat_completion":
            return {"type": "text", "text": input.data}
        elif mode == "response":
            _type = "input_text" if role != "assistant" else "output_text"
            return {"type": _type, "text": input.data}
        else:
            raise ValueError(f"Unknown mode: {mode}")

    @classmethod
    def reasoning_to_provider(cls, input: ContentItem, mode: str, role: str) -> Dict[str, Any]:
        if isinstance(input.data, str):
            return cls.text_to_provider(input, mode, role)
        else:
            raise ValueError(f"Reasoning type {type(input.data)} not supported yet")

    @classmethod
    def img_text_to_provider(cls, input: ContentItem, mode: str, role: str, compress_img: tuple[bool, float]) -> Dict[str, Any]:
        if input.type == "text":
            return cls.text_to_provider(input, mode, role)
        elif input.type == "image":
            return cls.img_to_provider(input, mode, role, compress_img)
        else:
            raise ValueError(f"Unknown content item type: {input.type}")

    @staticmethod
    def func_output_to_provider(input: ContentItem, mode: str, role: str) -> Dict[str, Any]:
        if mode == "chat_completion":
            return {"type": "function_call_output", "call_id": input.meta_data["call_id"], "output": input.data}
        elif mode == "response":
            return {"type": "function_call_output", "call_id": input.meta_data["call_id"], "output": input.data}
        else:
            raise ValueError(f"Unknown mode: {mode}")

    @classmethod
    def computer_output_to_provider(cls, input: ContentItem, mode: str, role: str, compress_img: tuple[bool, float]) -> Dict[str, Any]:
        if mode == "chat_completion":
            raise ValueError("Computer output not supported in chat completion")
        elif mode == "response":
            # https://platform.openai.com/docs/guides/tools-computer-use

            if len(input.data) > 1:
                raise ValueError("As of March-2025: Computer output must be a single item")

            payload = {
                "type": "computer_call_output",
                "call_id": input.meta_data["call_id"],
                "acknowledged_safety_checks": input.meta_data.get("acknowledged_safety_checks", []),
                "output": cls.img_text_to_provider(input.data[0], mode, role, compress_img=compress_img),
            }
            if input.meta_data.get("url", None):
                payload["current_url"] = input.meta_data["url"]
            return payload
        else:
            raise ValueError(f"Unknown mode: {mode}")

    @classmethod
    def convert_message(
        cls, message: Message, mode: str = "chat_completion", compress_img: tuple[bool, float] = (False, 0)
    ) -> List[Dict[str, Any]]:
        role = cls.convert_role(message.role)
        all_msgs = []

        all_contents = []
        for content_item in message.contents:
            if not content_item.data and not content_item.raw_model_output:
                # Ignore if no data or raw model output is available
                continue

            # For image and text, accumulate contents to write a single message
            if content_item.type == "text" or content_item.type == "image":
                all_contents.append(cls.img_text_to_provider(content_item, mode, message.role, compress_img))
                continue
            elif content_item.type == "reasoning":
                all_contents.append(cls.reasoning_to_provider(content_item, mode, message.role))
                continue
            else:
                # Flush content once find a type that is not image or text
                if len(all_contents) > 0:
                    all_msgs.append({"role": role, "content": all_contents})
                    all_contents = []

            # For function output, write a single message
            if content_item.type == "function_output":
                all_msgs.append(cls.func_output_to_provider(content_item, mode, role))

            # For computer output, write a single message
            elif content_item.type == "computer_output":
                all_msgs.append(cls.computer_output_to_provider(content_item, mode, role, compress_img=compress_img))

            # Other cases of model messages, create message with data as is
            elif content_item.type == "computer_call" or content_item.type == "function_call":
                if content_item.raw_model_output is not None:
                    # Use raw model output if available
                    all_msgs.append(content_item.raw_model_output)
                elif content_item.data is not None:
                    # Try to use data if raw model output is not available
                    all_msgs.append(content_item.data)
                else:
                    # Ignore if no data or raw model output is available
                    continue
            else:
                raise ValueError(f"Unknown content item type: {content_item.type}")

        # Flush any remaining contents
        if len(all_contents) > 0:
            all_msgs.append({"role": role, "content": all_contents})

        return all_msgs

    @classmethod
    def convert_prompt(cls, prompt: List[Message], mode: str, gen_config: GenerationConfig) -> List[Dict[str, Any]]:
        reg_prompt = []
        compress_img = (False, 0)
        if max_img_payload_size := MAX_IMG_PAYLOAD_SIZE_MB.get(gen_config.model, None):
            # get_conversation_img_payload_size returns (bytes, count)
            img_payload_size, total_imgs = get_conversation_img_payload_size(prompt)
            img_payload_size_mb = img_payload_size / 1024 / 1024
            # If total image payload in MB exceeds allowed max for this model, set a per-image
            # compression target so that the total will be <= max_img_payload_size.
            if total_imgs > 0 and img_payload_size_mb > max_img_payload_size:
                per_img_target_mb = float(max_img_payload_size) / float(total_imgs)
                # Avoid zero or extremely small targets; set a reasonable floor (10KB ~ 0.01MB)
                per_img_target_mb = max(per_img_target_mb, 0.01)
                logger.info(
                    f"Conversation image payload {img_payload_size_mb:.3f} MB exceeds max {max_img_payload_size} MB for model {gen_config.model}; "
                    f"setting per-image target to {per_img_target_mb:.3f} MB for {total_imgs} images"
                )
                compress_img = (True, per_img_target_mb)
            else:
                compress_img = (False, 0)

        for message in prompt:
            reg_prompt.extend(cls.convert_message(message, mode, compress_img=compress_img))
        return reg_prompt

    @staticmethod
    def reset_prompt(prompt: List[Dict[str, Any]]) -> None:
        raise NotImplementedError("Reset prompt is not implemented for OpenAI")

    @staticmethod
    def upload_all_images(prompt: List[Dict[str, Any]]) -> None:
        raise NotImplementedError("Upload all images is not implemented for OpenAI")

    @staticmethod
    def convert_role(role: str) -> str:
        return ROLE_MAPPINGS[role]

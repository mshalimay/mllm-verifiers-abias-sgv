from typing import Any, Dict, List

from core_utils.image_utils import any_to_b64, get_mime_type
from llms.providers.openai.constants import ROLE_MAPPINGS
from llms.types import ContentItem, Message


class AnthropicPrompter:
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
    def img_to_provider(cls, input: ContentItem) -> Dict[str, Any]:
        img_type = get_mime_type(input.data, fallback="PNG")
        img_b64 = any_to_b64(input.data, add_header=False, format=img_type)
        return {"type": "image", "source": {"type": "base64", "media_type": f"image/{img_type.lower()}", "data": img_b64}}

    @classmethod
    def text_to_provider(cls, input: ContentItem) -> Dict[str, Any]:
        return {"type": "text", "text": input.data}

    @classmethod
    def img_text_to_provider(cls, input: ContentItem) -> Dict[str, Any]:
        if input.type == "text":
            return cls.text_to_provider(input)
        elif input.type == "image":
            return cls.img_to_provider(input)
        else:
            raise ValueError(f"Unknown content item type: {input.type}")

    @staticmethod
    def func_output_to_provider(input: ContentItem) -> Dict[str, Any]:
        # TODO
        raise NotImplementedError("Function output not supported in Anthropic")
        # return {"type": "function_call_output", "call_id": input.meta_data["call_id"], "output": input.data}

    @classmethod
    def computer_output_to_provider(cls, input: ContentItem) -> Dict[str, Any]:
        raise NotImplementedError("Computer output not supported in Anthropic")

    @classmethod
    def convert_message(cls, message: Message) -> List[Dict[str, Any]]:
        role = cls.convert_role(message.role)
        all_msgs = []

        all_contents = []
        for content_item in message.contents:
            if not content_item.data and not content_item.raw_model_output:
                # Ignore if no data or raw model output is available
                continue

            # For image and text, accumulate contents to write a single message
            if content_item.type == "text" or content_item.type == "image":
                all_contents.append(cls.img_text_to_provider(content_item))
                continue
            else:
                # Flush content once find a type that is not image or text
                if len(all_contents) > 0:
                    all_msgs.append({"role": role, "content": all_contents})
                    all_contents = []

            if content_item.type == "video":
                # TODO
                raise NotImplementedError("Video not supported in Hugging Face")

            # For function output, write a single message
            if content_item.type == "function_output":
                # TODO
                all_msgs.append(cls.func_output_to_provider(content_item))

            # For computer output, write a single message
            elif content_item.type == "computer_output":
                # TODO
                raise NotImplementedError("Computer output not supported in Hugging Face")

            # Other cases of model messages, create message with data as is
            elif content_item.type == "computer_call" or content_item.type == "function_call" or content_item.type == "reasoning":
                # TODO
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
    def convert_prompt(cls, prompt: List[Message], mode: str = "chat_completion") -> List[Dict[str, Any]]:
        reg_prompt = [{"role": "system", "text": ""}]
        for message in prompt:
            if message.role == "system":
                reg_prompt[0]["text"] = message.contents[0].data
            else:
                reg_prompt.extend(cls.convert_message(message))
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

"""Converts messages from uniform format to Hugging Face API format."""

from typing import Any, Dict, List

from llms.generation_config import GenerationConfig
from llms.providers.hugging_face.constants import ROLE_MAPPINGS
from llms.providers.hugging_face.model_specific.model_prompter import ModelPrompter
from llms.types import ContentItem, Message


class HuggingFacePrompter:
    """
    A class to encapsulate prompt adjustments for Hugging Face API generation.
    """

    provider: str = "hugging_face"

    def __init__(self) -> None:
        pass

    @classmethod
    def img_text_to_provider(
        cls,
        model_id: str,
        input: ContentItem,
        engine: str,
        provider: str,
    ) -> Dict[str, Any]:
        return ModelPrompter.to_model(model_id, input, engine, provider)

    @staticmethod
    def func_output_to_provider(
        model_id: str,
        input: ContentItem,
        engine: str,
        provider: str,
    ) -> Dict[str, Any]:
        # TODO
        return {"type": "function_call_output", "call_id": input.meta_data["call_id"], "output": input.data}

    @classmethod
    def convert_message(
        cls,
        message: Message,
        model_id: str,
        engine: str,
        provider: str,
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
                all_contents.append(
                    cls.img_text_to_provider(
                        model_id=model_id,
                        input=content_item,
                        engine=engine,
                        provider=provider,
                    )
                )
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
                all_msgs.append(
                    cls.func_output_to_provider(
                        model_id=model_id,
                        input=content_item,
                        engine=engine,
                        provider=provider,
                    )
                )

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
    def convert_prompt(cls, prompt: List[Message], gen_config: GenerationConfig) -> List[Dict[str, Any]]:
        reg_prompt = []
        for message in prompt:
            reg_prompt.extend(
                cls.convert_message(
                    message,
                    model_id=gen_config.model_path,
                    engine=gen_config.engine,
                    provider=gen_config.metadata.get("provider", ""),
                )
            )
        return reg_prompt

    @classmethod
    def convert_role(cls, role: str) -> str:
        return ROLE_MAPPINGS[role]

    @classmethod
    def reset_prompt(cls, prompt: List[Dict[str, Any]]) -> None:
        raise NotImplementedError(f"Reset prompt is not implemented for {cls.provider}")

    @classmethod
    def upload_all_images(cls, prompt: List[Dict[str, Any]]) -> None:
        raise NotImplementedError(f"Upload all images is not implemented for {cls.provider}")

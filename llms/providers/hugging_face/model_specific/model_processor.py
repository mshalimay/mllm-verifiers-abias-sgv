import re
from typing import Any, Dict, List

from qwen_vl_utils import (
    process_vision_info,
)

from core_utils.image_utils import any_to_pil, is_image
from llms.providers.hugging_face.hugging_face_client_manager import get_client_manager


def flatten_text_contents(messages: list[dict[str, Any]], concat_str: str = "\n", error_msg: str = "") -> list[dict[str, Any]]:
    """Flatten the text contents of a list of messages."""

    for message in messages:
        content = message.get("content", None)
        if not content:
            continue

        if not isinstance(content, list):
            content = [content]

        str_msgs = []
        for item in content:
            if item.get("type") == "text":
                str_msgs.append(item["text"])
            else:
                raise ValueError(f"{__name__}: error while flattening text content. If this function was triggered, the expected model is text-only. Original error: {error_msg}")

        if str_msgs:
            conatenated_string = concat_str.join(str_msgs)
        else:
            continue

        message["content"] = conatenated_string
    return messages


def extract_media_from_chat_completion(messages: list[dict[str, Any]]) -> tuple[list[Any], list[Any]]:
    """Extract media from a chat completion."""
    images = []
    videos = []
    for message in messages:
        content = message.get("content", None)
        if not content:
            continue

        if not isinstance(content, list):
            content = [content]

        for item in content:
            if not isinstance(item, dict):
                if is_image(item):
                    images.append(item)
                # elif is_video(item):
                # videos.append(item)
                else:
                    continue

            if item.get("type") == "image":
                if "image" in item:
                    images.append(item["image"])
                elif "base64" in item:
                    img = any_to_pil(item["base64"])
                    images.append(img)
                elif "url" in item:
                    img = any_to_pil(item["url"])
                    images.append(img)
            if item.get("type") == "video":
                if "path" in item:
                    videos.append(item["path"])
                elif "url" in item:
                    videos.append(item["url"])
    return images, videos


class ModelProcessor:
    """
    Encapsulates methods to process model inputs and outputs for Hugging Face models.
    """

    def __init__(self) -> None:
        pass

    # ===============================================================
    # Get model inputs
    # ===============================================================
    @classmethod
    def get_inputs(
        cls,
        provider_messages: list[list[Dict[str, Any]]],
        model_path: str,
    ) -> Any:
        """Route to the correct model input getter based on the model path."""
        if re.match(r"Qwen/Qwen2\.5-VL-.*", model_path, flags=re.IGNORECASE):
            return cls.get_inputs_qwen2_5_vl(
                messages=provider_messages,
                model_path=model_path,
            )

        elif re.search(r"Kimi-VL", model_path, flags=re.IGNORECASE):
            return cls.get_inputs_default(
                provider_messages=provider_messages,
                model_path=model_path,
                truncation=True,
            )

        else:
            return cls.get_inputs_default(
                provider_messages=provider_messages,
                model_path=model_path,
            )

    @classmethod
    def get_inputs_default(
        cls,
        provider_messages: list[list[Dict[str, Any]]],
        model_path: str,
        add_generation_prompt: bool = True,
        padding: bool = True,
        truncation: bool = False,
    ) -> Any:
        """Default preparation for model inputs."""
        processor = get_client_manager(model_path).get_processor()

        images, videos = [], []
        for message in provider_messages:
            imgs_batch, videos_batch = extract_media_from_chat_completion(message)
            images.extend(imgs_batch)
            videos.extend(videos_batch)

        try:
            texts = [
                processor.apply_chat_template(  # type: ignore
                    message, add_generation_prompt=add_generation_prompt, return_tensors="pt"
                )
                for message in provider_messages
            ]
        except TypeError as e:
            if "can only concatenate str" in str(e).lower():
                # If this error happens, probably it is a language-only model, for which
                # HF requires a different format than the traditional chat completion
                # So we modify the provider_messages by concatenating the text contents
                # and removing "type":"text"
                _ = [flatten_text_contents(provider_messages, error_msg=str(e)) for provider_messages in provider_messages]
                # Apply the template without tokenization to pass to the processor below
                texts = [
                    processor.apply_chat_template(  # type: ignore
                        message, add_generation_prompt=add_generation_prompt, tokenize=False
                    )
                    for message in provider_messages
                ]
            else:
                raise e

        if len(images) > 0:
            inputs = processor(text=texts, images=images, return_tensors="pt", padding=padding, truncation=truncation)  # type: ignore
        else:
            inputs = processor(text=texts, return_tensors="pt")  # type: ignore

        return inputs

    @classmethod
    def get_inputs_qwen2_5_vl(
        cls,
        messages: list[list[Dict[str, Any]]],
        model_path: str,
    ) -> Any:
        processor = get_client_manager(model_path).get_processor()

        texts = [
            processor.apply_chat_template(  # type: ignore
                message,
                tokenize=False,
                add_generation_prompt=True,
            )
            for message in messages
        ]

        image_inputs, video_inputs = process_vision_info(messages, return_video_kwargs=False)  # type: ignore

        inputs = processor(  # type: ignore
            text=texts, images=image_inputs, video=video_inputs, padding=True, return_tensors="pt"
        )
        return inputs

    # ===============================================================
    # Decoding
    # ===============================================================
    @classmethod
    def decode_outputs(
        cls,
        outputs: Any,
        model_path: str,
        start_idxs: List[int] = [],
        skip_special_tokens: bool = True,
    ) -> Any:
        outputs_trimmed = outputs
        if start_idxs:
            outputs_trimmed = [output[start_idx:] for output, start_idx in zip(outputs, start_idxs)]

        if re.match(r"Qwen/Qwen2\.5-VL-.*", model_path, flags=re.IGNORECASE):
            return cls.decode_outputs_qwen2_5_vl(outputs_trimmed, model_path, skip_special_tokens)
        else:
            return cls.decode_outputs_default(outputs_trimmed, model_path, skip_special_tokens)

    @classmethod
    def decode_outputs_default(
        cls,
        outputs: Any,
        model_path: str,
        skip_special_tokens: bool = True,
    ) -> Any:
        processor = get_client_manager(model_path).get_processor()
        decoded_outputs = processor.batch_decode(  # type: ignore
            outputs,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )

        return decoded_outputs

    @classmethod
    def decode_outputs_qwen2_5_vl(
        cls,
        outputs: Any,
        model_path: str,
        skip_special_tokens: bool = True,
    ) -> Any:
        processor = get_client_manager(model_path).get_processor()
        decoded_outputs = processor.batch_decode(  # type: ignore
            outputs,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )

        return decoded_outputs

from typing import Any, List

from google.genai import types as genai_types
from google.genai.types import Part

from core_utils.image_utils import any_to_bytes, get_mime_type, is_image
from llms.prompt_utils import mark_for_upload
from llms.providers.google.constants import MAX_PAYLOAD_SIZE, ROLE_MAPPINGS, UPLOAD_ALL, UPLOAD_IMAGES
from llms.providers.google.file_manager import get_file_manager
from llms.types import ContentItem, Contents, Message


# Obs.: didn't use `self` logic for convenience as typically use the same prompter.
# This way doesnt have to instantiate the class every time.
class GooglePrompter:
    """
    A class to encapsulate prompt adjustments for Google API generation.

    This class handles:
      - Converting a content item to a genai Part (text, image, etc.)
      - Regularizing a list of content items
      - Regularizing messages (including handling system vs. non-system messages)
      - Adjusting the message role according to Google's API
    """

    upload_images = UPLOAD_IMAGES

    def __init__(self) -> None:
        pass

    @classmethod
    def convert_content_item(cls, content_item: ContentItem, p_id: int = 0) -> Any:
        if content_item.type == "text":
            return Part.from_text(text=content_item.data)

        elif content_item.type == "image":
            if cls.upload_images or content_item.meta_data.get("upload", False):
                input_file = get_file_manager(p_id).get_upload_image_file(image=content_item.data)
                return Part.from_uri(file_uri=input_file.uri, mime_type=input_file.mime_type)  # type:ignore
            else:
                img_fmt = get_mime_type(content_item.data, fallback="PNG")
                if img_fmt == "gif":
                    img_fmt = "PNG"  # Google API does not support GIF inline
                img_bytes = any_to_bytes(content_item.data, format=img_fmt)
                return Part.from_bytes(data=img_bytes, mime_type=f"image/{img_fmt.lower()}")

        elif content_item.type == "function_call":
            # https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling#python_3
            if content_item.raw_model_output is not None:
                return content_item.raw_model_output
            else:
                # return Part.from_function_call(content_item.meta_data["name"], function_call=content_item.data)
                raise ValueError("Function call not found in content item")

        elif content_item.type == "function_output":
            return Part.from_function_response(
                name=content_item.meta_data["name"],  # must have a name
                response={"content": content_item.data},
            )

        elif content_item.type == "reasoning":
            if content_item.raw_model_output is not None:
                return content_item.raw_model_output
            else:
                raise ValueError(f"Reasoning not found in content item: {content_item}")

        else:
            raise ValueError(f"Unknown content type: {content_item.type}")

    @classmethod
    def convert_contents(cls, contents: Contents, p_id: int = 0) -> List[Part]:
        parts = []
        for content in contents:
            parts.append(cls.convert_content_item(content, p_id=p_id))
        return parts

    @classmethod
    def convert_message(cls, message: Message, p_id: int = 0) -> genai_types.Content:
        role = cls.convert_role(message.role)
        parts = cls.convert_contents(message.contents, p_id=p_id)
        return genai_types.Content(role=role, parts=parts)

    @staticmethod
    def convert_role(role: str) -> str:
        return ROLE_MAPPINGS[role]

    @classmethod
    def convert_prompt(
        cls,
        prompt: List[Message],
        p_id: int = 0,
        force_upload: bool = False,
    ) -> List[genai_types.Content]:
        reg_prompt = [genai_types.Content(role="system", parts=[])]

        prompt = mark_for_upload(prompt, MAX_PAYLOAD_SIZE, upload_all=UPLOAD_ALL)

        for message in prompt:
            if message.role == "system":
                reg_prompt[0] = cls.convert_message(message, p_id=p_id)
            else:
                reg_prompt.append(cls.convert_message(message, p_id=p_id))

        # If greater than 20MB, raise an error
        # if payload_size > MAX_PAYLOAD_SIZE:
        #     logger.info(f"Payload exceeded {MAX_PAYLOAD_SIZE / 1024 / 1024} MB, uploading images.")
        #     reg_prompt = cls.upload_all_images(reg_prompt, p_id=p_id, force_upload=force_upload)
        return reg_prompt

    @staticmethod
    def reset_prompt(prompt: List[genai_types.Content], p_id: int = 0, flush_cache: bool = False) -> List[genai_types.Content]:
        old_to_new = get_file_manager(p_id).reupload_images_for_prompt(prompt)
        for i, content in enumerate(prompt):
            if not content.parts:
                continue
            for j, part in enumerate(content.parts):
                if hasattr(part, "file_data") and part.file_data is not None:
                    new_file = old_to_new[part.file_data.file_uri]  # type: ignore

                    if not new_file.uri or not new_file.mime_type:
                        raise ValueError("Error uploading file: no uri or mime type")
                    content.parts[j] = Part.from_uri(
                        file_uri=new_file.uri,
                        mime_type=new_file.mime_type,
                    )
        if flush_cache:
            get_file_manager(p_id).flush_cache(keep_list=list(old_to_new.values()))
        return prompt

    @staticmethod
    def upload_all_images_for_prompt(
        prompt: List[genai_types.Content],
        p_id: int = 0,
        force_upload=False,
    ) -> List[genai_types.Content]:
        for i, content in enumerate(prompt):
            if not content.parts:
                continue
            for j, part in enumerate(content.parts):
                if part.inline_data is not None:
                    if (
                        part.inline_data.mime_type
                        and "image" in part.inline_data.mime_type  # short-circuit to prevent is_image if not necessary
                        or is_image(part.inline_data.data)
                    ):
                        new_file = get_file_manager(p_id).get_upload_image_file(image=part.inline_data.data, force_upload=force_upload)
                        if not new_file.uri or not new_file.mime_type:
                            raise ValueError("Error uploading file: no uri or mime type")
                        content.parts[j] = Part.from_uri(
                            file_uri=new_file.uri,
                            mime_type=new_file.mime_type,
                        )
        return prompt


def get_content_item_payload_size(content_item: ContentItem) -> int:
    """
    Computes the payload size (in bytes) for a given content item.

    The size is computed based on the type:
      - For text: returns the length of the text encoded in UTF-8.
      - For image: converts the image to bytes (PNG format) and returns its length.
      - For function_call and reasoning: if 'raw_model_output' is provided, returns the length
        of its string representation.
      - For function_output: returns the length of the string representation of the data.

    Args:
        content_item (ContentItem): The content item for which to compute the payload size.

    Returns:
        int: The size of the payload in bytes.
    """

    if content_item.type == "text":
        return len(content_item.data.encode("utf-8"))

    elif content_item.type == "image":
        try:
            # Convert image data to bytes using PNG format.
            img_fmt = get_mime_type(content_item.data, fallback="PNG")
            img_bytes = any_to_bytes(content_item.data, format=img_fmt)
            return len(img_bytes)
        except Exception:
            return 0

    elif content_item.type == "function_call":
        if content_item.raw_model_output is not None:
            return len(str(content_item.raw_model_output))
        else:
            return 0

    elif content_item.type == "function_output":
        return len(str(content_item.data))

    elif content_item.type == "reasoning":
        if content_item.raw_model_output is not None:
            return len(str(content_item.raw_model_output))
        else:
            return 0

    else:
        raise ValueError(f"Unknown content type: {content_item.type}")

# This Module provides functions to build prompts for LLMs among different providers using a common interface.

import html
import io
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypedDict

from core_utils.file_utils import is_empty
from core_utils.image_utils import any_to_b64, is_image, is_string
from core_utils.logger_utils import logger
from core_utils.types import ImageInput
from llms.constants.constants import GENERATION_PREFIX_PATTERN
from llms.types import ContentItem, Message, ValidInputs


# TODO: Refactor and remove this; see TODO.md
class Instruction(TypedDict):
    """Instruction for constructing prompt"""

    intro: str
    examples: List[Tuple[str, str]]
    template: str
    meta_data: Dict[str, Any]


# ==============================================================================
# LINK Prompting functions - low level
# ==============================================================================
def get_other_k_v_pairs(item: dict[str, Any], exclude_keys: List[str] = []) -> Dict[str, Any]:
    """
    Get other key-value pairs from a dictionary.
    """
    return {k: v for k, v in item.items() if k not in exclude_keys}


def parse_img_url_dict_item(subitem: str | dict[str, Any]) -> List[ContentItem]:
    """
    Get an image from a URL.
    """
    if isinstance(subitem, dict):
        if "image_url" in subitem:
            img_input = subitem["image_url"]
        elif "url" in subitem:
            img_input = subitem["url"]
        else:
            return []
        if img := is_image(img_input, return_image=True):
            meta_data = get_other_k_v_pairs(subitem, ["image_url", "url"])
            return [ContentItem(type="image", data=img, meta_data=meta_data)]
        else:
            return []
    elif img := is_image(subitem, return_image=True):
        return [ContentItem(type="image", data=img)]
    else:
        return []


def contents_from_dict(item: Dict[str, Any], recursion_idx: int = 0) -> List[ContentItem]:
    """
    Extracts content from a dictionary and returns a list of standardized ContentItem objects.

    Args:
        item: A dictionary that may contain content under various keys.

    Returns:
        A list of ContentItem objects representing the extracted content.
    """

    # Max number of nesting in the dictionaries is 1
    # E.g.: {"contents": [{"image": "..."}, {"text": "..."}, {"image_url": "..."}]} this is allowed
    # E.g.: {"contents": {"images": []}} this is not allowed
    if recursion_idx > 1:
        return []

    # Define the list of potential keys that can hold the primary content.
    inputs_keys = ["inputs", "data", "contents", "content", "input", "parts"]

    # Check for all keys
    count = 0
    input_data = []
    for key in inputs_keys:
        if key in item:
            count += 1
            input_data = item[key]

    # Enforce one of {'inputs': [...]} or {'text': '...', 'image': '...'}
    if count > 1:
        raise ValueError(f"Please provide just one entry among {inputs_keys!r}.")

    if len(input_data) > 0 and ("text" in item or "image" in item):
        raise ValueError(f"Please do not provide `text` and `image` keys along with `inputs`, `data`, `contents`, or `content` keys.")

    # List to hold the resulting content items.
    all_contents = []

    # Case 1: provided as list of inputs in `inputs_keys`
    if len(input_data) > 0:
        if not isinstance(input_data, list):
            input_data = [input_data]

        # Process each item in the list
        for input_item in input_data:
            if not input_item:
                continue
            # Nested dict input: if item is like {"contents": [{"image": "..."}, {"text": "..."}, {"image_url": "..."}]}
            # => process {"image": "..."}, {"text": "..."}, {"image_url": "..."} by recursively calling itself.
            if isinstance(input_item, dict):
                all_contents.extend(contents_from_dict(input_item, recursion_idx + 1))

            # Raw image input: directly create content item
            elif img := is_image(input_item, return_image=True):
                all_contents.append(ContentItem(type="image", data=img))

            # Raw text input: directly create content item
            elif isinstance(input_item, str):
                all_contents.append(ContentItem(type="text", data=input_item))

    # Case 2: atomic unit of data provided with explicit and known keys
    # Eg.: {"text": "...", "image": "...", "video": "..."}
    # E.g.:{"type": "image", "image_url": "..."}
    else:
        # {"text": "<string>"} => directly create from string
        if "text" in item and item["text"]:
            all_contents.append(ContentItem(type="text", data=item["text"]))

        # {"image": "<ImageInput>"} => directly create from ImageInput
        if "image" in item and item["image"]:
            if img := is_image(item["image"], return_image=True):
                all_contents.append(ContentItem(type="image", data=img))
            else:
                pass
        if "video" in item and item["video"]:
            # all_contents.append(ContentItem(type="video", data=item["video"]))
            raise NotImplementedError("Video content is not supported yet.")

        # Deal with other variations for image inputs
        if "image_url" in item and item["image_url"]:
            # This allow one level of additional nesting
            # OpenAI chat completion uses: {"image_url":{"<key>":"<ImageInput>"}})
            # Also parse {"image_url":<ImageInput>}
            all_contents.extend(parse_img_url_dict_item(item["image_url"]))

        elif "type" in item and "image" in item["type"]:
            # {"type":"image", "key":"<ImageInput>"} => directly create from ImageInput
            if "data" in item:
                img_input = item["data"]
            elif "file" in item:
                img_input = item["file"]
            elif "b64" in item:
                img_input = item["b64"]
            elif "url" in item:
                img_input = item["url"]
            elif "base64" in item:
                img_input = item["base64"]
            else:
                img_input = None
            if img := is_image(img_input, return_image=True):
                meta_data = get_other_k_v_pairs(item, ["data", "file", "b64", "url", "base64"])
                all_contents.append(ContentItem(type="image", data=img, meta_data=meta_data))
            else:
                pass

        # Deal with other variations for video inputs
        if "type" in item and "video" in item["type"]:
            raise NotImplementedError("Video content is not supported yet.")

    return all_contents


def get_content_item(
    input: ImageInput | str | Dict[str, Any] | ContentItem | List[ImageInput | str | Dict[str, Any] | ContentItem],
    img_detail: str = "auto",
) -> ContentItem:
    if img := is_image(input, return_image=True):
        return ContentItem(type="image", data=img, meta_data={"img_detail": img_detail})

    elif isinstance(input, str):
        return ContentItem(type="text", data=input)

    else:
        raise ValueError(f"Invalid input type: {type(input)}")


def get_contents(
    inputs: ImageInput | str | Dict[str, Any] | ContentItem | List[ImageInput | str | Dict[str, Any] | ContentItem],
) -> List[ContentItem]:
    if not isinstance(inputs, list):
        inputs = [inputs]

    all_contents = []

    for item in inputs:
        if img := is_image(item, return_image=True):
            all_contents.append(ContentItem(type="image", data=img))

        elif isinstance(item, str):
            all_contents.append(ContentItem(type="text", data=item))

        elif isinstance(item, dict):
            all_contents.extend(contents_from_dict(item))

        elif isinstance(item, ContentItem):
            all_contents.append(item)

        else:
            raise ValueError(f"Invalid input type: {type(item)}")

    return all_contents


def build_message(contents: List[Any], role: str, name: str = "", message_metadata: Dict[str, Any] = {}) -> Message:
    """
    Creates a single `Message` object given `Contents`, `role`, and `name`.

    Args:
        contents: the input data to send to the model.
        role: role of the entity producing the content.
        name: name of the entity producing the content.

    Returns:
        A message object in the universal format.
    """
    if not isinstance(contents, list):
        contents = [contents]
    contents = get_contents(contents)
    return Message(role=role, name=name, contents=contents, meta_data=message_metadata)


# ==============================================================================
# LINK Prompting functions - higher level
# ==============================================================================


def get_func_out_content(
    output: Dict[str, Any],
    name: str = "",
    call_id: str | None = None,
    meta_data: Dict[str, Any] = {},
) -> ContentItem:
    """ """
    meta_data["call_id"] = call_id if call_id else None
    meta_data["name"] = name
    return ContentItem(type="function_output", data=output, meta_data=meta_data)


def get_computer_out_content(
    outputs: Any,
    call_id: str | None = None,
    url: str | None = None,
    pending_safety_checks: List[str] = [],
    acknowledged_safety_checks: List[str] = [],
    meta_data: Dict[str, Any] = {},
    img_detail: str = "auto",
) -> ContentItem:
    """ """
    meta_data["url"] = url if url else None
    meta_data["pending_safety_checks"] = pending_safety_checks if pending_safety_checks else []
    meta_data["acknowledged_safety_checks"] = acknowledged_safety_checks if acknowledged_safety_checks else []
    meta_data["call_id"] = call_id if call_id else None

    if not isinstance(outputs, list):
        outputs = [outputs]

    all_contents = []
    for out in outputs:
        if is_image(out):
            all_contents.append(ContentItem(type="image", data=out, meta_data={"img_detail": img_detail}))
        elif isinstance(out, str):
            all_contents.append(ContentItem(type="text", data=out))
        else:
            raise ValueError(f"Invalid output type for computer output: {type(out)}")

    return ContentItem(type="computer_output", data=all_contents, meta_data=meta_data)


def get_message(
    inputs: Dict[str, Any] | str | ImageInput | ContentItem | Sequence[str | ImageInput | Dict[str, Any] | ContentItem],
    role: str = "",
    name: str = "",
    img_detail: str = "auto",
    message_metadata: Dict[str, Any] = {},
) -> Message:
    """
    Convenient function to create a single `Message` given a list of text, images, dicts, or `ContentItem` objects.

    Args:
        role (str): role of the entity sending the message.
        inputs (List[str, ImageInput, Dict[str, Any], ContentItem]): list of inputs to be part of the message.
        name (str): name of the entity sending the message.
        img_detail (str): detail level of the image (OpenAI only).
    Returns:
        Message: a single `Message` object in the uniform format.
    """
    contents = []

    if not role and isinstance(inputs, dict):
        try:
            role = inputs["role"]
        except KeyError:
            raise ValueError(f"Add role as parameter or as a key in the input dict: {inputs!r}")

    inputs_list = inputs if isinstance(inputs, list) else [inputs]

    for item in inputs_list:
        if is_empty(item):
            continue

        if img := is_image(item, return_image=True):
            contents.append(ContentItem(type="image", data=img, meta_data={"img_detail": img_detail}))

        elif isinstance(item, str):
            contents.append(ContentItem(type="text", data=item))

        elif isinstance(item, ContentItem):
            contents.append(item)

        elif isinstance(item, dict):
            contents.extend(contents_from_dict(item))

    return build_message(contents=contents, role=role, name=name, message_metadata=message_metadata)


def get_messages(
    inputs: ValidInputs,
    sys_prompt: str = "",
    role: str = "user",
    name: str = "",
    concatenate_text: bool = False,
    concatenate_sep: str = "\n",
    img_detail: str = "auto",
) -> List[Message]:
    """
    Builds a list of `Message`s ready to send to a model given a list of inputs in flexible formats.

    Args:
        inputs: A list of inputs to send to the model, which can be:
                1) raw strings
                2) raw images
                3) an already-created `Message`
                4) a dict with a `role` and `inputs` key
                5) a list of `Message` objects
                6) a list of other types (which will be converted into a single `Message` with all items)
        sys_prompt: system prompt to send to the model (if any). If system prompts also provided in other inputs, they will be concatenated.
        role: role of the entity; all inputs without a role will be assigned this role.
        name: name of the entity; all inputs without a name will be assigned this name.
        concatenate_text: if True, consecutive string inputs are concatenated into a single message.
        concatenate_sep: separator string when concatenating text.
        img_detail: detail level of the image (for providers such as OpenAI); all inputs without an `img_detail` will be assigned this value.

    Returns:
        List[Message]: List of `Message` objects in the uniform format.
    """

    # Regularize inputs to a list
    if not isinstance(inputs, list):
        inputs_list = [inputs]
    else:
        inputs_list = inputs

    # Initialize the list of messages to be returned
    messages: List[Message] = []

    # Cache to concatenate system prompts
    sys_prompts: List[ContentItem | str | Dict[str, Any]] = []
    if sys_prompt:
        sys_prompts.append(sys_prompt)

    # Cache to concatenate consecutive text inputs, if requested
    conc_string_cache: list[str] = []
    for item in inputs_list:
        if is_empty(item):
            continue

        # If it's a raw string => handle text input
        if is_string(item):
            if concatenate_text:
                conc_string_cache.append(item)  # type:ignore
            else:
                messages.append(build_message(contents=[item], role=role, name=name))
            continue
        else:
            # Flush the accumulated text if any when a non-string item is encountered.
            if concatenate_text and conc_string_cache:
                messages.append(build_message(contents=[concatenate_sep.join(conc_string_cache)], role=role, name=name))
                conc_string_cache = []

        # If it's an image => create a `Message` with the image content
        if img := is_image(item, return_image=True):
            content_item = ContentItem(type="image", data=img, meta_data={"img_detail": img_detail})
            messages.append(build_message(contents=[content_item], role=role, name=name))

        # If it's a list:
        # - If all items are `Message` objects => extend messages as is
        # - If all items are other types => create a single message combining them
        elif isinstance(item, list):
            temp_items = []
            has_message, has_other_type = False, False
            for i in item:
                if is_empty(i):
                    continue
                if isinstance(i, Message):
                    has_message = True
                    # If system message, proccess later
                    if i.role == "system" or i.role == "developer":
                        sys_prompts.extend(i.contents)
                    else:
                        messages.append(i)
                else:
                    has_other_type = True
                    temp_items.append(i)
                if has_message and has_other_type:
                    raise ValueError("Behavior undefined for Message objects mixed with other types within a list.")
            if has_other_type:
                # Create a single message combining all items
                messages.append(build_message(contents=temp_items, role=role, name=name))
            else:
                # Empty list or all items are `Message` objects (handled in the loop if no error)
                continue

        # If it's a single `Message` object => add it directly
        elif isinstance(item, Message):
            if item.role == "system" or item.role == "developer":
                sys_prompts.extend(item.contents)
            else:
                messages.append(item)

        # If it's a dict
        elif isinstance(item, dict):
            contents = contents_from_dict(item)
            if not contents:
                continue
            _role = item.get("role", role)
            _name = item.get("name", "")
            msg = build_message(contents=contents, role=_role, name=_name)
            if _role == "system" or _role == "developer":
                sys_prompts.append(msg.text())
                continue
            else:
                messages.append(msg)

        elif isinstance(item, ContentItem):
            messages.append(build_message(contents=[item], role=role, name=name))
        else:
            raise ValueError(f"Unknown input type: {type(item)}")

    # Flush any remaining concatenated text items into a message.
    if concatenate_text and conc_string_cache:
        new_item = concatenate_sep.join(conc_string_cache)
        messages.append(build_message(contents=[new_item], role=role, name=name))

    # If a system prompt was provided, prepend it as the first message.
    if sys_prompts:
        system_msg = build_message(contents=sys_prompts, role="system", name="")
        messages = [system_msg] + messages

    return messages


def get_interleaved_img_txt_msg(
    images: List[ImageInput],
    img_captions: Optional[List[str]] = None,
    role: str = "user",
    name: str = "",
    text_first: bool = True,
    img_detail: str = "auto",
    text_prefix: str = "",
) -> Message:
    """
    Creates a message by interleaving images and captions.
    """
    # Ensure img_captions is a list with the same length as images.
    if img_captions is None:
        img_captions = [""] * len(images)
    elif len(img_captions) < len(images):
        img_captions.extend([""] * (len(images) - len(img_captions)))
    elif len(img_captions) > len(images):
        img_captions = img_captions[: len(images)]

    # Interleave based on the text_first flag.
    inputs: list[Any] = []

    if text_prefix and text_first:
        inputs.append(text_prefix)

    if text_first:
        # [caption0, image0, caption1, image1, ...]
        inputs.extend([x for pair in zip(img_captions, images) for x in pair])
    else:
        # [image0, caption0, image1, caption1, ...]
        inputs.extend([x for pair in zip(images, img_captions) for x in pair])

    if text_prefix and not text_first:
        inputs.append(text_prefix)

    return get_message(inputs=inputs, role=role, name=name, img_detail=img_detail)


def get_conversation_payload_size(messages: List[Message]) -> int:
    """
    Get the payload size of a conversation.
    """
    return sum(message.payload_size or 0 for message in messages)


def get_conversation_img_payload_size(messages: List[Message]) -> tuple[int, int]:
    """
    Get the image payload size of a conversation.
    """
    total_size = 0
    total_imgs = 0
    for message in messages:
        for content_item in message.contents:
            if content_item.type == "image":
                total_size += content_item.payload_size or 0
                total_imgs += 1
    return total_size, total_imgs


def mark_for_upload(messages: List[Message], max_payload_size: float, upload_all: bool = False) -> List[Message]:
    """
    Mark messages for upload.
    """
    # Determine the amount of payload to offload:
    payload_size = get_conversation_payload_size(messages)

    if payload_size <= max_payload_size:
        return messages

    if upload_all:
        upload_until = 0
    else:
        upload_until = max_payload_size

    logger.info(f"Payload size: {payload_size / 1024 / 1024} MB is greater than max payload size: {max_payload_size / 1024 / 1024} MB. Uploading part of the conversation.")
    required_upload = payload_size - upload_until
    logger.info(f"Uploading {required_upload / 1024 / 1024} MB of {payload_size / 1024 / 1024} MB")

    uploaded_payload = 0
    for msg in messages:
        for content_item in msg.contents:
            if content_item.type == "image":
                uploaded_payload += content_item.payload_size or 0
                content_item.meta_data["upload"] = True
            if uploaded_payload >= required_upload:
                break
    return messages


# ==============================================================================
# LINK Logging functions
# ==============================================================================


def conversation_to_txt(
    prompt_messages: List[Message],
    model_messages: List[Message],
    output_path: str | Path,
    gen_kwargs: dict[str, Any] = {},
    verbose: bool = False,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Read the existing content (if any)
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            existing_content = f.read()
    except FileNotFoundError:
        existing_content = ""

    buffer = io.StringIO()

    thought_prefix = "\n----- THOUGHT START -----\n"
    thought_suffix = "\n----- THOUGHT END -----\n"

    # Create generation arguments block if needed and not already present in file
    if gen_kwargs and not re.search(r"=+\s*\nGENERATION\s+ARGS", existing_content, re.IGNORECASE):
        buffer.write("==================================\n")
        buffer.write("GENERATION ARGS\n")
        buffer.write("==================================\n")
        for key, value in gen_kwargs.items():
            buffer.write(f"{key}: {value}\n")

    # Append the existing file content
    buffer.write(existing_content)

    # Build prompt block
    buffer.write("==================================\nPROMPT\n==================================\n")
    for message in prompt_messages:
        role_text = message.role.upper() + (f" ({message.name})" if message.name else "")
        buffer.write(f"\n---------------\nROLE: {role_text}\n---------------")
        for c in message.contents:
            if c.type == "reasoning" and c.data:
                write_text = f"{thought_prefix}{c.data}{thought_suffix}"
            elif c.type == "text":
                write_text = c.data
            else:
                write_text = f"<{c.type}>"
            buffer.write(f"\n\nCONTENT TYPE: {c.type}\n{write_text}\n")

    # Build generation block
    buffer.write("\n==================================\n")
    buffer.write("GENERATION\n")
    buffer.write("==================================\n")
    for i, message in enumerate(model_messages):
        buffer.write(GENERATION_PREFIX_PATTERN.format(i))
        for c in message.contents:
            if c.type == "reasoning":
                write_text = f"{thought_prefix}{c.data}{thought_suffix}"
            elif c.type == "text":
                write_text = c.data
            else:
                write_text = f"<{c.type}>"
            buffer.write(f"\n\nCONTENT TYPE: {c.type}\n{write_text}\n")
    buffer.write("\n\n\n")

    # Get the final string from the buffer and write it to the file
    final_output = buffer.getvalue()
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_output)

    if verbose:
        print(f"Conversation saved to {output_path}")


def conversation_to_html(
    messages: List[Message],
    output_path: str | Path,
    html_header: str = "",
    verbose: bool = False,
    gen_kwargs: dict[str, Any] = {},
) -> None:
    """Dump a series of user-assistant messages in API format to an HTML file for visualization.
    Args:
        messages (list[dict[str, Any]]): list of messages in OpenAI API format
        output_path (str | Path): path to the HTML file
        html_header (str, optional): header to be displayed at the top of the HTML file.
        verbose (bool, optional): whether to print a message when the file is saved.
    """
    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        # Start HTML document
        f.write("""
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {
                    font-family: monospace;
                    margin: 20px;
                    max-width: 1200px;
                    margin: 0 auto;
                }
                .section {
                    border: 1px solid #ccc;
                    margin: 10px 0;
                    padding: 10px;
                    background: #fafafa;
                }
                .role {
                    font-weight: bold;
                    background: #e8e8e8;
                    padding: 5px;
                    border: 1px solid #ccc;
                }
                .name {
                    color: #666;
                    font-size: 0.9em;
                    margin-left: 10px;
                }
                .content {
                    margin: 10px 0;
                }
                pre {
                    white-space: pre-wrap;
                    word-wrap: break-word;
                    overflow-wrap: break-word;
                    max-width: 100%;
                }
                img {
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                    display: block;
                    margin: 10px 0;
                }
            </style>
        </head>
        <body>
        """)

        if html_header:
            f.write(f'<div class="section"><h2>{html_header}</h2></div>')

        if gen_kwargs:
            f.write('<div class="section">')
            f.write("<h3>Generation Arguments</h3>")
            f.write("<pre>")
            for key, value in gen_kwargs.items():
                f.write(f"{key}: {value}\n")
            f.write("</pre>")
            f.write("</div>")

        # Process each message
        for message in messages:
            f.write('<div class="section">')

            # Get name from any available source
            name = message.name

            role_text = f"{message.role.upper()}"
            role_text += f'<span class="name">({name})</span>' if name else ""
            f.write(f'<div class="role">{role_text}</div>')

            for part in message.contents:
                f.write('<div class="content">')
                if part.type == "image":
                    img_data = any_to_b64(part.data, add_header=True)
                    f.write(f'<img src="{img_data}" alt="Prompt Image">')

                elif part.type == "reasoning" and part.data:
                    f.write(f"<pre> ====== THOUGHT START ======<br>{html.escape(part.data)}<br>====== THOUGHT END ======</pre>")

                elif part.type == "text":
                    f.write(f"<pre>{html.escape(part.data)}</pre>")

                else:
                    try:
                        f.write(f"<pre>{html.escape(part.data)}</pre>")
                    except Exception as _:
                        continue
                f.write("</div>")

            f.write("</div>")

        f.write("</body></html>")

    if verbose:
        print(f"Conversation saved to {output_path}")


def visualize_prompt(messages, output_path: str | Path, verbose: bool = False) -> None:
    messages = get_messages(messages)

    out_file_obj = Path(output_path)
    # Add .html suffix if not present
    if not out_file_obj.suffix:
        out_file_obj = out_file_obj.with_suffix(".html")

    # Ensure that the parent directory exists.
    out_file_obj.parent.mkdir(parents=True, exist_ok=True)

    conversation_to_html(messages, out_file_obj, verbose=verbose)


# ==============================================================================
# LINK: LLM output parsing
# ==============================================================================


def flatten_generations(generations: List[Message], text_prefix_pattern: str = GENERATION_PREFIX_PATTERN) -> Message:
    flat_generation: Message
    all_contents: List[ContentItem | str] = []

    if not isinstance(generations, list):
        generations = [generations]

    if len(generations) == 1:
        return generations[0]

    for i, msg in enumerate(generations):
        if text_prefix_pattern:
            all_contents.append(text_prefix_pattern.format(i))
        all_contents.extend(msg.contents)
    flat_generation = build_message(
        contents=all_contents,
        role="assistant",
    )
    return flat_generation

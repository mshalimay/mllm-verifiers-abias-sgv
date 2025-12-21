from typing import Any, Dict, List, Tuple

from core_utils.image_utils import any_to_pil


def count_tokens(response: Any, inputs: Any) -> Dict[str, List[int]]:
    token_counts = {}

    # Get input token counts
    if hasattr(inputs, "input_ids"):
        if len(inputs.input_ids) > 1:
            token_counts["input_tokens"] = [len(seq) for seq in inputs.input_ids]
        else:
            token_counts["input_tokens"] = [len(inputs.input_ids[0])] * len(response.sequences)
    elif hasattr(inputs, "shape"):
        if len(inputs.shape) > 1:
            token_counts["input_tokens"] = [len(seq) for seq in inputs.shape]
        else:
            token_counts["input_tokens"] = [inputs.shape[0]] * len(response.sequences)

    # Get total and completion tokens
    if hasattr(response, "sequences"):
        token_counts["total_tokens"] = [len(seq) for seq in response.sequences]
        if "input_tokens" in token_counts:
            token_counts["completion_tokens"] = [total - prompt for total, prompt in zip(token_counts["total_tokens"], token_counts["input_tokens"])]

    return token_counts


def get_other_k_v(ele: Dict[str, Any], keys_exclude: List[str]) -> Dict[str, Any]:
    other_keys = [key for key in ele.keys() if key not in keys_exclude]
    return {key: ele[key] for key in other_keys}


def extract_vision_chat_msgs(
    conversations: list[dict[str, Any]] | list[list[dict[str, Any]]],
) -> Tuple[List[Any], List[Any]]:
    images, videos = [], []
    if isinstance(conversations[0], dict):
        conversations = [conversations]  # type: ignore

    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):  # type: ignore
                for ele in message["content"]:  # type: ignore
                    # Images
                    if ele["type"] == "image" and "url" in ele:
                        new_ele = {"type": "image", "image": any_to_pil(ele["url"])}
                        new_ele.update(get_other_k_v(ele, ["type", "image", "url"]))
                        images.append(new_ele)

                    elif ele["type"] == "image" and "b64" in ele:
                        new_ele = {"type": "image", "image": any_to_pil(ele["b64"])}
                        new_ele.update(get_other_k_v(ele, ["type", "image", "b64"]))
                        images.append(new_ele)

                    elif "image" in ele:
                        new_ele = {"type": "image", "image": any_to_pil(ele["image"])}
                        new_ele.update(get_other_k_v(ele, ["type", "image"]))
                        images.append(new_ele)

                    elif "image_url" in ele:
                        new_ele = {"type": "image", "image": any_to_pil(ele["image_url"])}
                        new_ele.update(get_other_k_v(ele, ["type", "image_url"]))
                        images.append(new_ele)

                    # Videos
                    elif ele["type"] == "video" or "video" in ele:
                        videos.append(ele)
    return images, videos


def get_trim_prompt_idxs(inputs: Any, num_generations: int) -> list[int]:
    if hasattr(inputs, "input_ids"):
        if len(inputs.input_ids) > 1:
            trim_prompt_idxs = [inputs.input_ids[i].shape[0] for i in range(num_generations)]
        else:
            trim_prompt_idxs = [inputs.input_ids[0].shape[0]] * num_generations
    elif hasattr(inputs, "shape"):
        if len(inputs.shape) > 1:
            trim_prompt_idxs = [in_ids.shape[1] for in_ids in inputs]
        else:
            trim_prompt_idxs = [inputs.shape[1]] * num_generations
    else:
        trim_prompt_idxs = [0] * num_generations
        print("WARNING: Could not determine input shape. Generated answer may include the prompt")

    return trim_prompt_idxs

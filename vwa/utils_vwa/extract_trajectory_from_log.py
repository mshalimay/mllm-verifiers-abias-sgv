import os
import re
from pathlib import Path
from typing import Any

from browser_env.env_config import Website
from bs4 import BeautifulSoup
from trajectory_utils.trajectory import VWA_Trajectory

from core_utils.image_utils import any_to_pil
from utils_vwa.utils_vwa import TrajectoryView


def maybe_get_domain_from_path(path: str | Path) -> str:
    for domain in Website._value2member_map_.keys():
        if domain in path:
            return domain
    return ""


def resolve_input_image_paths(image_paths: list[str] | list[Any]) -> list[str]:
    """
    Resolve image paths from VWA config files to absolute paths.

    VWA config files use paths relative to the vwa/ directory (e.g., "config_files/vwa_input_images/...").
    This function converts them to absolute paths that can be accessed from the project root.

    Args:
        image_paths: List of image paths (strings or other types like PIL Images)

    Returns:
        List of resolved image paths as strings
    """
    from benchmark_config.constants import LOCAL_INPUT_IMAGES_DIR_TEMPLATE

    resolved_images = []
    for img in image_paths:
        if isinstance(img, str):
            # If it's a relative path starting with "config_files/", it's from VWA config
            # VWA config files use paths relative to the vwa/ directory
            if img.startswith("config_files/"):
                # Use the LOCAL_INPUT_IMAGES_DIR_TEMPLATE to build the correct path
                # Extract domain and task_id from the path
                # Pattern: config_files/vwa_input_images/{domain}/task_{task_id}/...
                parts = img.split("/")
                if len(parts) >= 4:
                    domain = parts[2]
                    task_dir = parts[3]  # e.g., "task_0"
                    task_id = task_dir.replace("task_", "")
                    filename = parts[-1] if len(parts) > 4 else ""

                    # Build the correct path using the template
                    img_dir = LOCAL_INPUT_IMAGES_DIR_TEMPLATE.format(domain=domain, task_id=task_id)
                    img = os.path.join(img_dir, filename)

            # Convert to absolute path if it's still relative
            img_path = Path(img)
            if not img_path.is_absolute():
                # Resolve relative to current working directory
                img_path = img_path.resolve()

            resolved_images.append(str(img_path))
        else:
            # If already a PIL Image or other type, keep as is
            resolved_images.append(img)

    return resolved_images


def get_input_images_for_task(task_id: int | str, domain: str) -> list[str]:
    """
    Retrieve absolute input image paths for a given VWA task using domain and task_id.

    Uses the benchmark config template to construct the directory and lists images.

    Args:
        task_id: Task ID (int or str)
        domain: VWA domain (e.g., "reddit", "shopping")

    Returns:
        List of absolute file paths to input images (PNG/JPG/JPEG/GIF), sorted by name.
    """
    from benchmark_config.constants import LOCAL_INPUT_IMAGES_DIR_TEMPLATE

    task_id_str = str(task_id)
    img_dir = LOCAL_INPUT_IMAGES_DIR_TEMPLATE.format(domain=domain, task_id=task_id_str)

    resolved: list[str] = []
    try:
        if not os.path.isdir(img_dir):
            return resolved
        for fname in sorted(os.listdir(img_dir)):
            fpath = os.path.join(img_dir, fname)
            if not os.path.isfile(fpath):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext in {".png", ".jpg", ".jpeg", ".gif"}:
                resolved.append(str(Path(fpath).resolve()))
    except Exception:
        # Silently fall back; callers may choose alternative resolution
        return resolved

    return resolved


def parse_html_trajectory_data(html_file: str | Path, stop_at_verifier_loop: bool = False, stop_at_loop_idx: int = 0) -> dict:
    if isinstance(html_file, Path):
        html_file = str(html_file)

    with open(html_file, "r", encoding="utf-8") as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, "html.parser")
    data = {}
    data["config"] = {}
    data["config"]["intent_images"] = []
    steps = []

    # Extract intent from <pre> tag
    pre_div = soup.find("pre")
    prev_key = None
    if pre_div:
        # Break by newlines
        pred_div_contents = pre_div.text.strip().split("\n")

        for i, item in enumerate(pred_div_contents):
            if ":" not in item and prev_key == "intent":
                data["config"]["intent"] += "\n" + item.strip()
                continue

            try:
                key, value = item.split(":", 1)
                key = key.strip()
                value = value.strip()
            except ValueError:
                print(f"Error parsing {item} in {html_file}")
                continue

            if key == "intent":
                data["config"]["intent"] = value

            elif key == "image":
                img_content = value
                if img_content.lower() != "none":
                    if img_content.startswith("["):
                        img_urls = eval(img_content)
                        data["config"]["intent_images"].extend(img_urls)
                    else:
                        data["config"]["intent_images"].append(img_content)
            elif "task_id" in key:
                data["config"]["task_id"] = value
            else:
                data["config"][key] = value
            prev_key = key

    # Try to extract score from annotation_banner div
    annotation_banner = soup.find("div", id="annotation_banner")
    if annotation_banner:
        banner_text = annotation_banner.get_text()
        # Extract SCORE:value
        score_match = re.search(r"SCORE:\s*(\S+)", banner_text)
        if score_match:
            data["score"] = score_match.group(1)

    # Try to extract domain and source from the lightgray div (if not already found)
    if not data["config"].get("domain", ""):
        for div in soup.find_all("div"):
            style_attr = div.get("style", "")  # type: ignore
            if "background:lightgray" in style_attr:  # type: ignore
                div_text = div.get_text()
                if "DOMAIN:" in div_text:
                    domain_match = re.search(r"DOMAIN:\s*(\S+)", div_text)
                    if domain_match:
                        data["config"]["domain"] = domain_match.group(1)

    if not data["config"].get("domain", ""):
        data["config"]["domain"] = maybe_get_domain_from_path(html_file)

    # Split content by "New Page" headers
    page_sections = soup.find_all("h2")
    verifier_loop_idx = 0
    for section in page_sections:
        state = {}
        current_element = section.next_sibling
        skip_elements = []
        current_element: Any
        while current_element and not (current_element.name == "h2"):
            # 1. URL (found in h3 with class 'url')
            if current_element.name == "h3" and "url" in current_element.get("class", []):
                state["url"] = current_element.find("a").text.replace("URL: ", "")

            if current_element in skip_elements:
                current_element = current_element.next_sibling
                continue

            if current_element.name == "div" and "executor_critique_loop" in current_element.get("class", []):
                skip_elements.append(current_element)

            # 2. State observation and nested elements
            elif current_element.name == "div" and "state_obv" in current_element.get("class", []):
                # Extract action tree
                state["action_tree"] = current_element.find("pre").text if current_element.find("pre") else None

                # Extract image if present
                imgs = current_element.find_all("img")
                state_img = None
                for img in imgs:
                    # Get last b64 image available
                    if img.get("src", "").startswith("data:image/png;base64"):
                        state_img = img["src"]
                state["screenshot"] = state_img

                # Extract previous action
                prev_action = current_element.find("div", class_="prev_action")
                if prev_action:
                    state["prev_action"] = prev_action.text.strip()

                # Extract prediction elements
                predict_div = current_element.find("div", class_="predict_action")
                if predict_div:
                    # Raw parsed prediction
                    raw_pred = predict_div.find("div", class_="raw_parsed_prediction")
                    if raw_pred:
                        state["raw_utterance"] = raw_pred.find("pre").text if raw_pred.find("pre") else None

                    # Parsed action
                    parsed = predict_div.find("div", class_="parsed_action")
                    if parsed:
                        state["parsed_action"] = parsed.find("pre").text if parsed.find("pre") else None

                exec_crit_loop_div = current_element.find("div", class_="executor_critique_loop")

                if exec_crit_loop_div:
                    state["scores_per_round"] = []

                    verifier_divs = current_element.find_all("div", class_="critique_utterance")
                    executor_divs = current_element.find_all("div", class_="executor_utterance")

                    round_idx = 0
                    for i in range(min(len(verifier_divs), len(executor_divs))):
                        verifier_div = verifier_divs[i] if i < len(verifier_divs) else None
                        executor_div = executor_divs[i] if i < len(executor_divs) else None

                        if verifier_div:
                            verifier_raw_response = verifier_div.text.strip()
                        else:
                            verifier_raw_response = None

                        if executor_div:
                            raw_utt = executor_div.text.strip()
                            parsed_action = extract_action(raw_utt)

                        state["scores_per_round"].append(
                            {
                                "verifier_raw_response": verifier_raw_response,
                                "round_idx": round_idx,
                                "raw_prediction": raw_utt,
                                "parsed_action": parsed_action,
                            }
                        )

                        if verifier_loop_idx == stop_at_loop_idx and stop_at_verifier_loop:
                            state["raw_utterance"] = raw_utt
                            state["parsed_action"] = parsed_action

                            steps.append(state)
                            data["steps"] = steps
                            return data
                    verifier_loop_idx += 1

            current_element = current_element.next_sibling

        if state:  # Only append if we found any data
            steps.append(state)

    data["steps"] = steps
    return data


def extract_action(response: str, fallback: str = "") -> str:
    # find the first occurence of action
    action_splitter = "```"
    pattern = rf"{action_splitter}((.|\n)*?){action_splitter}"
    match = re.search(pattern, response)
    if match:
        return match.group(1).strip()
    else:
        print(f"Unable to extract action from response: {response}. Using: {fallback}")
        return fallback


class Page:
    def __init__(self, url: str):
        self.url = url


def rebuild_trajectory_vwa_format_from_html(
    html_path: str | Path,
    stop_at_verifier_loop: bool = False,
) -> tuple[dict, dict, TrajectoryView]:
    trajectory_data = parse_html_trajectory_data(html_path, stop_at_verifier_loop)

    trajectory_vwa_format = []
    meta_data = {"action_str_history": ["None"]}
    for step in trajectory_data["steps"]:
        observation = {"text": step["action_tree"], "image": step["screenshot"]}

        info = {"url": step["url"], "page": Page(step["url"])}

        extracted_action = extract_action(step["raw_utterance"], fallback=step["parsed_action"])

        state_vwa_format = {"observation": observation, "info": info, "scores_per_round": step.get("scores_per_round", [])}
        action_vwa_format = {
            "raw_prediction": step["raw_utterance"],
            "extracted_action": extracted_action,
            "_parsed_action": step["parsed_action"],
        }
        meta_data["action_str_history"].append(step["parsed_action"])

        trajectory_vwa_format.append(state_vwa_format)
        trajectory_vwa_format.append(action_vwa_format)

    trajectory_view = TrajectoryView(trajectory_vwa_format)
    trajectory_view.task_config = trajectory_data["config"]
    return trajectory_data, meta_data, trajectory_view


def rebuild_trajectory_vwa_format_from_json(
    trajectory: VWA_Trajectory,
    stop_at_verifier_loop=False,
    stop_at_loop_idx=0,
    ann_types=[],
    verifier_round_stop_idx=0,
):
    trajectory_vwa_format = []
    meta_data: dict[str, Any] = {"action_str_history": ["None"]}

    stop_idx = 0
    for state, action in zip(trajectory.states, trajectory.actions):
        text_obs = state["text"]
        raw_screenshot = any_to_pil(state["images"][-1])
        if ann_types:
            trajectory.annotate_images(state=state, actions=action, ann_types=ann_types)
            img_obs = state["images"][-1]
        else:
            img_obs = raw_screenshot

        state_vwa_format = {
            "observation": {"text": text_obs, "image": img_obs, "raw_screenshot": raw_screenshot},
            "info": {"url": state["url"], "page": Page(state["url"])},
        }

        executor_agent_action = action["agents"]["executor"][-1]  # type: ignore
        raw_prediction = executor_agent_action["generated_text"]
        action_str_env_parsed = executor_agent_action["action_str_env_parsed"]
        extracted_action = extract_action(raw_prediction, fallback=action_str_env_parsed)

        if "stop" in action_str_env_parsed and "localhost" in action_str_env_parsed:
            action_str_env_parsed = extracted_action

        if state.get("scores_per_round", None) is not None and stop_at_verifier_loop:
            if stop_idx == stop_at_loop_idx:
                data = state.get("scores_per_round")[verifier_round_stop_idx]  # type: ignore
                raw_prediction = data["raw_prediction"]
                extracted_action = extract_action(raw_prediction, fallback=action_str_env_parsed)
            stop_idx += 1

        action_vwa_format = {
            "raw_prediction": raw_prediction,
            "extracted_action": extracted_action,
        }
        meta_data["action_str_history"].append(action_str_env_parsed)
        trajectory_vwa_format.append(state_vwa_format)
        trajectory_vwa_format.append(action_vwa_format)

        if stop_idx > stop_at_loop_idx:
            break

    return trajectory_vwa_format, meta_data, TrajectoryView(trajectory_vwa_format)


def extract_trajectory_data(file_path, stop_at_verifier_loop=False, stop_at_loop_idx=0, ann_types=[]) -> tuple[dict, TrajectoryView, dict]:
    if isinstance(file_path, Path):
        file_path = str(file_path)

    if file_path.endswith(".html"):
        trajectory_data, meta_data, trajectory_view = rebuild_trajectory_vwa_format_from_html(file_path)

        # Prefer resolving by domain/task_id from the config
        domain = trajectory_data["config"].get("domain", "") or maybe_get_domain_from_path(file_path)
        task_id = trajectory_data["config"].get("task_id", "") or ""
        images_by_task = get_input_images_for_task(task_id=task_id, domain=domain)

        if images_by_task:
            resolved_images = images_by_task
        else:
            # Fallback: resolve whatever was listed in the config
            intent_images = trajectory_data["config"].get("intent_images", [])
            resolved_images = resolve_input_image_paths(intent_images)

        objective = {
            "text": trajectory_data["config"]["intent"],
            "images": resolved_images,
        }
        return objective, trajectory_view, meta_data

    elif file_path.endswith(".json"):
        trajectory = VWA_Trajectory.from_json(file_path)
        _, meta_data, trajectory_view = rebuild_trajectory_vwa_format_from_json(
            trajectory=trajectory,
            stop_at_verifier_loop=stop_at_verifier_loop,
            stop_at_loop_idx=stop_at_loop_idx,
            ann_types=ann_types,
        )
        # Prefer resolving by domain/task_id from the trajectory
        domain = trajectory.domain or maybe_get_domain_from_path(file_path)
        task_id = trajectory.task_id or ""
        images_by_task = get_input_images_for_task(task_id=task_id, domain=domain)

        if images_by_task:
            resolved_images = images_by_task
        else:
            # Fallback: resolve existing objective image paths
            objective_images = trajectory.objective.get("images", [])
            resolved_images = resolve_input_image_paths(objective_images)

        objective = {"text": trajectory.objective.get("text", ""), "images": resolved_images}

        config_file = trajectory.config
        meta_data["config_file"] = config_file

        return objective, trajectory_view, meta_data

    else:
        raise ValueError(f"Unsupported file type: {file_path}")

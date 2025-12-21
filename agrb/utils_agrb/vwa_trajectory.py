#!/usr/bin/env python3
import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from PIL import Image

from agrb.constants import MAX_STEPS
from agrb.utils_agrb.agrb_utils import (
    annotate_action_on_image,
    get_local_input_image_path,
    get_vwa_domain_task_id,
    update_task_intent,
    write_bbox_to_screenshot,
)
from agrb.utils_agrb.parse_ax_tree import AXTreeParser
from core_utils.image_utils import any_to_pil
from core_utils.logger_utils import logger
from core_utils.string_utils import clean_spaces

# scroll(delta_x: float, delta_y: float)
# keyboard_press(key: str)
# click(bid: str, button: Literal['left', 'middle', 'right'] = 'left', modifiers: list[typing.Literal['Alt', 'Control', 'ControlOrMeta', 'Meta', 'Shift']] = [])
# fill(bid: str, value: str)
# hover(bid: str)
# tab_focus(index: int)
# new_tab()
# go_back()
# go_forward()
# goto(url: str)
# tab_close()
# select_option(bid: str, options: str | list[str])
# send_msg_to_user(text: str)
# upload_file(bid: str, file: str | list[str])
# report_infeasible(reason: str)

_ACTION_RE = re.compile(r"^\s*(\w+)\s*\((.*)\)")


def _convert_bid_to_int_if_possible(bid_str: str) -> str | int:
    """Convert bid string to int if it represents a number, otherwise keep as string."""
    try:
        # Try to convert to int if it's a numeric string
        return int(bid_str)
    except (ValueError, TypeError):
        # Keep as string if conversion fails
        return bid_str


def _parse_args_tuple(raw_args: str) -> Tuple[Any, ...]:
    """Safely parse a comma-separated argument list into a tuple."""
    raw = raw_args.strip()
    if raw == "":
        return tuple()
    # Wrap in parentheses so literal_eval parses as a tuple when needed
    try:
        parsed = ast.literal_eval(f"({raw},)") if raw[-1] != ")" else ast.literal_eval(raw)
    except Exception:
        # Fallback: try as a single literal
        parsed = ast.literal_eval(raw)
    if isinstance(parsed, tuple):
        return parsed
    return (parsed,)


def _map_url_to_real(url: str) -> str:
    """
    Map VWA URLs to their real domain counterparts.

    Example:
        https://vwa-shopping-xl-1.mcgill-nlp.org/long-path -> http://onestopmarket.com/long-path
    """
    from vwa.browser_env.env_config import URL_MAPPINGS, Website

    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        path = parsed.path
        query = parsed.query
        fragment = parsed.fragment

        # Build query and fragment parts
        query_part = f"?{query}" if query else ""
        fragment_part = f"#{fragment}" if fragment else ""

        # Check if this is a VWA domain that needs mapping
        mapped_domain = ""
        if "shopping" in domain:
            # Map to shopping domain
            mapped_domain = URL_MAPPINGS[Website.SHOPPING][0]  # "http://onestopmarket.com"
            return f"{mapped_domain}{path}{query_part}{fragment_part}"
        elif "forum" in domain or "reddit" in domain:
            # Map to reddit domain
            mapped_domain = URL_MAPPINGS[Website.REDDIT][0]  # "http://reddit.com"
            return f"{mapped_domain}{path}{query_part}{fragment_part}"
        elif "wikipedia" in domain:
            # Map to wikipedia domain
            mapped_domain = URL_MAPPINGS[Website.WIKIPEDIA][0]  # "http://wikipedia.org"
            return f"{mapped_domain}{path}{query_part}{fragment_part}"
        elif "classifieds" in domain:
            # Map to classifieds domain
            mapped_domain = URL_MAPPINGS[Website.CLASSIFIEDS][0]  # "http://classifieds.com"
            return f"{mapped_domain}{path}{query_part}{fragment_part}"
        elif "homepage" in domain:
            # Map to homepage domain
            mapped_domain = URL_MAPPINGS[Website.HOMEPAGE][0]  # "http://homepage.com"
            return f"{mapped_domain}{path}{query_part}{fragment_part}"
        elif "gitlab" in domain:
            # Map to gitlab domain
            mapped_domain = URL_MAPPINGS[Website.GITLAB][0]  # "http://gitlab.com"
            return f"{mapped_domain}{path}{query_part}{fragment_part}"
        elif "map" in domain:
            # Map to map domain
            mapped_domain = URL_MAPPINGS[Website.MAP][0]  # "http://openstreetmap.org"

        if not mapped_domain:
            logger.warning(f"Not able to infer domain from {url}. Using original URL.")
            return url
        else:
            return f"{mapped_domain}{path}{query_part}{fragment_part}"

    except Exception:
        # Fallback: return original URL if parsing fails
        return url


def _map_all_urls_to_real(generated_text: str) -> str:
    """
    Map all URLs in an action string to their real domain counterparts.
    """
    urls = re.findall(
        # r"https?://[^\s,\]]+?\.(?:jpg|jpeg|png|gif|bmp|webp)",
        r"https?://[^\s,\]]+",
        generated_text,
        re.IGNORECASE,
    )
    for url in urls:
        new_url = _map_url_to_real(url)
        generated_text = generated_text.replace(url, new_url)
    return generated_text


def parse_action(action: str) -> Dict[str, Any]:
    """
    Parse a single action string into a structured dict covering ALL supported actions.

    Returns a dict like:
      {"action": "<type>", "bid": Optional[str], "args": {...}}
    """
    if re.match(r"^\s*None\s*$", action, re.IGNORECASE):
        return {"action": "None", "bid": None, "args": {}}

    m = _ACTION_RE.match(action)
    if not m:
        raise ValueError(f"Invalid action format: {action!r}")

    action_type, raw_args = m.groups()
    action_type = action_type.lower()  # Make action type case-insensitive
    args_tuple = _parse_args_tuple(raw_args)

    bid: Optional[str | int] = None
    args: Dict[str, Any] = {}

    # --- Action-specific handling (ALL actions covered) ---

    if action_type == "noop":
        # noop(wait_ms: float = 1000)
        args["wait_ms"] = float(args_tuple[0]) if len(args_tuple) > 0 else 1000.0

    elif action_type == "scroll":
        # scroll(delta_x: float, delta_y: float)
        args["delta_x"] = float(args_tuple[0]) if len(args_tuple) > 0 else 0.0
        args["delta_y"] = float(args_tuple[1]) if len(args_tuple) > 1 else 0.0

    elif action_type == "keyboard_press":
        # keyboard_press(key: str)
        if len(args_tuple) < 1:
            raise ValueError("keyboard_press requires a key")
        args["key"] = str(args_tuple[0])

    elif action_type == "click":
        # click(bid: str, button: 'left'|'middle'|'right'='left', modifiers: list[...] = [])
        if len(args_tuple) < 1:
            raise ValueError("click requires a bid")
        bid = _convert_bid_to_int_if_possible(str(args_tuple[0]))
        args["button"] = str(args_tuple[1]) if len(args_tuple) > 1 else "left"
        mods = args_tuple[2] if len(args_tuple) > 2 else []
        if isinstance(mods, (list, tuple)):
            args["modifiers"] = [str(m) for m in mods]
        else:
            # Allow a single string as a convenience
            args["modifiers"] = [str(mods)]

    elif action_type == "fill":
        # fill(bid: str, value: str)
        if len(args_tuple) < 2:
            raise ValueError("fill requires bid and value")
        bid = _convert_bid_to_int_if_possible(str(args_tuple[0]))
        args["value"] = str(args_tuple[1])

    elif action_type == "hover":
        # hover(bid: str)
        if len(args_tuple) < 1:
            raise ValueError("hover requires a bid")
        bid = _convert_bid_to_int_if_possible(str(args_tuple[0]))

    elif action_type == "tab_focus":
        # tab_focus(index: int)
        if len(args_tuple) < 1:
            raise ValueError("tab_focus requires an index")
        args["index"] = int(args_tuple[0])

    elif action_type == "new_tab":
        # new_tab()
        pass

    elif action_type == "go_back":
        # go_back()
        pass

    elif action_type == "go_forward":
        # go_forward()
        pass

    elif action_type == "goto":
        # goto(url: str)
        if len(args_tuple) < 1:
            raise ValueError("goto requires a url")
        args["url"] = str(args_tuple[0])

    elif action_type == "tab_close":
        # tab_close()
        pass

    elif action_type == "select_option":
        # select_option(bid: str, options: str | list[str])
        if len(args_tuple) < 2:
            raise ValueError("select_option requires bid and options")
        bid = _convert_bid_to_int_if_possible(str(args_tuple[0]))
        opts = args_tuple[1]
        if isinstance(opts, (list, tuple)):
            args["options"] = [str(o) for o in opts]
        else:
            args["options"] = str(opts)

    elif action_type == "send_msg_to_user":
        # send_msg_to_user(text: str)
        if len(args_tuple) < 1:
            raise ValueError("send_msg_to_user requires text")
        args["text"] = str(args_tuple[0])

    elif action_type == "upload_file":
        # upload_file(bid: str, file: str | list[str])
        if len(args_tuple) < 2:
            raise ValueError("upload_file requires bid and file(s)")
        bid = _convert_bid_to_int_if_possible(str(args_tuple[0]))
        files = args_tuple[1]
        if isinstance(files, (list, tuple)):
            args["file"] = [str(f) for f in files]
        else:
            args["file"] = str(files)

    elif action_type == "report_infeasible":
        # report_infeasible(reason: str)
        if len(args_tuple) < 1:
            raise ValueError("report_infeasible requires a reason")
        args["reason"] = str(args_tuple[0])

    elif action_type == "None":
        # None
        pass

    else:
        raise ValueError(f"Unknown action: {action_type}")

    return {"action": action_type, "bid": bid, "args": args}


def _semantic2str(node: dict) -> str:
    """
    Turn a semantic info dict into a string description.
    """
    _role = node.get("role", "")
    _content = node.get("name", "").strip()

    pattern = (
        "[\ue000-\uf8ff\U000f0000-\U000ffffd\U00100000-\U0010fffd]"  # actual PUA chars
        r"|\\u(?:e[0-9A-Fa-f]{3}|f[0-8][0-9A-Fa-f]{2})"  # \uE000–\uF8FF escapes
        r"|\\U(?:000F[0-9A-Fa-f]{4}|0010[0-9A-Fa-f]{4})"  # \U000F0000–\U0010FFFF escapes
    )

    # Remove control characters and private use area characters including \ue622
    _content = clean_spaces(re.sub(pattern, "", _content).strip())

    _role_str = ""
    match _role:
        case "button":
            _role_str = "BUTTON"
        case "input" | "textbox":
            _role_str = "INPUT"
        case "link":
            _role_str = "A"
        case _:
            _role_str = _role.upper()
    return f"[{_role_str}] element with content [{_content}]"


def action2str(parsed_action: dict, semantic_info: dict, add_semantic_info: bool = True) -> str:
    """
    Turn a parsed action dict into a semantic string description.

    Args:
        action: dict with keys "action", "bid", "args"
        semantic_element: string describing what the bid refers to

    Returns:
        str: semantic action string
    """

    if parsed_action["action"] == "None":
        return "None"

    node = semantic_info.get("node", {})
    semantic_str = ""
    if node and add_semantic_info:
        semantic_str = _semantic2str(node)

    action_type = parsed_action["action"]
    bid = parsed_action.get("bid")

    # convenience
    args = parsed_action.get("args", {})

    match action_type:
        case "click":
            if semantic_str:
                return f"click [{bid}] where [{bid}] is {semantic_str}"
            else:
                return f"click [{bid}]"

        case "fill" | "type":
            if semantic_str:
                return f"type [{bid}] [{args.get('value', '')}] where [{bid}] is {semantic_str}"
            else:
                return f"type [{bid}] [{args.get('value', '')}]"

        case "hover":
            if semantic_str:
                return f"hover [{bid}] where [{bid}] is {semantic_str}"
            else:
                return f"hover [{bid}]"

        case "scroll":
            if args.get("delta_x", 0) == 0 and args.get("delta_y", 0) > 0:
                return "scroll [down]"
            elif args.get("delta_x", 0) == 0 and args.get("delta_y", 0) < 0:
                return "scroll [up]"
            elif args.get("delta_x", 0) > 0 and args.get("delta_y", 0) == 0:
                return "scroll [right]"
            elif args.get("delta_x", 0) < 0 and args.get("delta_y", 0) == 0:
                return "scroll [left]"
            else:
                return f"scroll [dx={args.get('delta_x', 0)}, dy={args.get('delta_y', 0)}]"

        case "keyboard_press":
            return f"press [{args.get('key', '')}]"

        case "goto":
            return f"goto [{args.get('url', '')}]"

        case "new_tab":
            return "new_tab"

        case "tab_close":
            return "close_tab"

        case "go_back":
            return "go_back"

        case "go_forward":
            return "go_forward"

        case "tab_focus":
            return f"page_focus [{args.get('index', 0)}]"

        case "noop":
            return f"wait [{args.get('wait_ms', 1000)}ms]"

        case "upload_file":
            files = args.get("file")
            if isinstance(files, list):
                files_str = ", ".join(files)
            else:
                files_str = str(files)
            if semantic_str:
                return f"upload [{files_str}] to [{bid}] where [{bid}] is {semantic_str}"
            else:
                return f"upload [{files_str}] to [{bid}]"

        case "select_option":
            opts = args.get("options")
            if isinstance(opts, list):
                opts_str = ", ".join(opts)
            else:
                opts_str = str(opts)
            if semantic_str:
                return f"select [{opts_str}] from [{bid}] where [{bid}] is {semantic_str}"
            else:
                return f"select [{opts_str}] from [{bid}]"

        case "send_msg_to_user":
            return f"stop [{args.get('text', '')}]"

        case "report_infeasible":
            return f"stop [{args.get('reason', '')}]"

        case _:
            raise ValueError(f"Unknown action type {action_type}")


class VWATrajectoryView:
    """
    A view class for VisualWebArena trajectories that provides easy access to trajectory data.
    """

    def __init__(self, trajectory_path: str, ann_types: list[str] = [], trajectories_dir: str = "", update_intent: bool = True):
        """
        Initialize the trajectory view.

        Args:
            trajectory_path: Path to the trajectory JSON file
            trajectory_data: Raw trajectory data dictionary
        """
        self._states: List[Dict] = []
        self._actions: List[Dict] = []
        self._screenshots: List[str] = []
        self.trajectory_path = trajectory_path

        with open(trajectory_path, "r", encoding="utf-8") as f:
            self.trajectory = json.load(f)

        # Extract basic trajectory information
        self.benchmark = self.trajectory.get("benchmark", "")
        self.agent = self.trajectory.get("agent", "")
        self.model = self.trajectory.get("model", "")
        self.experiment = self.trajectory.get("experiment", "")
        self.goal = self.trajectory.get("goal", "")
        self.valid = self.trajectory.get("valid", False)
        self.seed = self.trajectory.get("seed", None)
        self.trajectories_dir = trajectories_dir
        self.task_id = Path(trajectory_path).stem
        self.vwa_domain, self.vwa_task_id = get_vwa_domain_task_id(self.task_id, self.experiment)

        self.objective = {"text": "", "images": []}
        self.objective["text"] = self.get_clean_goal()

        if update_intent:
            new_intent = update_task_intent(self.vwa_domain, self.vwa_task_id)
            if not new_intent:
                logger.warning(f"Could not updated intent for: {self.vwa_domain}_{self.vwa_task_id}. Using original goal.")
            else:
                self.objective["text"] = new_intent

        input_images = self.extract_input_images()

        if input_images:
            # input_img_paths = [img["url"] for img in input_images]
            input_img_paths = get_local_input_image_path(self.task_id, self.experiment)
            if not input_img_paths:
                raise ValueError(f"No input image found for task {self.task_id} in experiment {self.experiment}")
            self.objective["images"] = self.get_images(input_img_paths)

        self.agent_prompt = self.trajectory.get("steps", [{}])[0].get("chat_messages", None)

        # Extract summary info
        self.summary_info = self.trajectory.get("summary_info", {})
        self.cum_reward = self.summary_info.get("cum_reward", None)
        self.n_steps = self.summary_info.get("n_steps", 0)

        self.annotate_images = len(ann_types) > 0
        self.ann_types = ann_types

        # Build the trajectory view
        self._build_trajectory_view()

    def sanitize_reasoning_str(self, reasoning_str: str) -> str:
        """
        Sanitize the reasoning string by removing anything enclosed in <action> </action> tags.
        """
        import re

        # Remove everything between <action> and </action> tags (including the tags)
        pattern = r"<action>.*?</action>"
        sanitized = re.sub(pattern, "", reasoning_str, flags=re.DOTALL)
        # Clean up any extra whitespace that might be left
        return sanitized.strip()

    def _build_trajectory_view(self):
        steps = self.trajectory.get("steps", [])
        if not steps:
            return

        self._states = []
        self._actions = []
        self._screenshots = []
        ax_tree_parser = AXTreeParser()

        for i in range(len(steps)):
            step = steps[i]  # Get the current step

            state_dict, action_dict = {}, {}

            # Get screenshots
            ss_paths = step.get("screenshot_path", [])
            if ss_paths:
                images = self.get_images(ss_paths, trajectory_dir=self.trajectories_dir)
            else:
                images = []

            # Populate state dict
            state_dict = {
                "observation": {"images": images, "texts": ""},
                "screenshot_path": step.get("screenshot_path", ""),
                "url": step.get("url", ""),
                "focused_element": step.get("focused_element", ""),
                "last_action_error": step.get("last_action_error", ""),
                "open_pages_urls": step.get("open_pages_urls", []),
                "axtree": step.get("axtree", ""),
                "axtree_obj": step.get("axtree_obj", {}),
            }

            # Action and reasoning in original format
            action_str: str = step.get("action") if step.get("action") else ""
            reasoning_str: str = step.get("reasoning") if step.get("reasoning") else ""
            reasoning_str = self.sanitize_reasoning_str(reasoning_str)

            # Parse action type and arguments
            if i == len(steps) - 1 and not action_str:
                # If no action and last step, it is a stop action
                if i == MAX_STEPS:
                    parsed_action = parse_action("send_msg_to_user('Max steps reached')")
                else:
                    parsed_action = parse_action("send_msg_to_user('')")

            elif i == MAX_STEPS and not action_str:
                # If no action and max steps, it is a stop action due to max steps
                parsed_action = parse_action("send_msg_to_user('Max steps reached')")

            elif action_str:
                # If action is present, parse it
                parsed_action = parse_action(action_str)

            else:
                # If no action and not last step, it is an error
                raise ValueError(f"No action found for step {i}, trajectory: {self.trajectory_path}")

            # Get semantic info for the action
            if parsed_action.get("bid"):
                semantic_info = ax_tree_parser.get_semantic_info(step, parsed_action["bid"])
                if not semantic_info.get("node"):
                    action_dict["invalid"] = True
            else:
                semantic_info = {}

            # Transform the action to VWA original format
            action_str_semantic = action2str(parsed_action, semantic_info, add_semantic_info=True)

            # Populate the action dict
            action_dict = {
                "action_str": action_str,
                "reasoning_str": reasoning_str,
                "parsed_action": parsed_action,
                "images": [],
                "num": step.get("num", i),
                "stats": step.get("stats", {}),
                "semantic_info": semantic_info,
                "action_str_semantic": action_str_semantic,
                "action_str_no_semantic": action2str(parsed_action, semantic_info, add_semantic_info=False),
            }

            # Post-processing: annotate images if requested
            if self.annotate_images:
                ann_imgs = []

                for img in images:
                    ann_img = img

                    # Annotate with dot if requested
                    if any(supported_ann in self.ann_types for supported_ann in ["coord", "coordinates", "dot"]):
                        if bid := action_dict["parsed_action"].get("bid"):
                            bboxes = step.get("bounding_boxes", {})
                            bbox = bboxes.get(str(bid), {}) or bboxes.get(bid, {})
                            if bbox:
                                ann_img, _ = self.annotate_dot_on_image(img, bbox)

                    # Annotate with SOM if requested
                    if any(supported_ann in self.ann_types for supported_ann in ["bbox", "som"]):
                        ann_img = write_bbox_to_screenshot(img, step.get("bounding_boxes", {}))
                    ann_imgs.append(ann_img)
                state_dict["observation"]["images"] = ann_imgs

            # Add the state and action to the trajectory view
            self._states.append(state_dict)
            self._actions.append(action_dict)

            # Stop if the action is a stop action or the max steps is reached
            if parsed_action["action"] == "send_msg_to_user" or parsed_action["action"] == "report_infeasible":
                break

    def add_data_vwa_format(self):
        """
        Add data in VWA format.
        """
        self.meta_data = {"action_str_history": ["None"]}
        # state_vwa_format = {"observation": observation, "info": info}
        # action_vwa_format = {"raw_prediction": state["raw_utterance"], "extracted_action": extracted_action}
        for state, action in zip(self._states, self._actions):
            state["url"] = _map_url_to_real(state["url"])
            state["observation"]["image"] = state["observation"]["images"][-1]
            action["raw_prediction"] = _map_all_urls_to_real(action["reasoning_str"])
            action["extracted_action"] = _map_all_urls_to_real(action["action_str_no_semantic"])
            self.meta_data["action_str_history"].append(_map_all_urls_to_real(action["action_str_semantic"]))
            state["observation"]["info"] = {"url": state["url"], "page": None}

    def get_images(self, img_paths, trajectory_dir: str = "") -> List[Image.Image]:
        if not isinstance(img_paths, list):
            img_paths = [img_paths]

        images = []
        for img_path in img_paths:
            try:
                if trajectory_dir:
                    img_path = Path(trajectory_dir) / str(img_path).split("trajectories/")[1]

                img = any_to_pil(img_path)

                if img is not None:
                    images.append(img)
                else:
                    raise ValueError(f"Not able to load image {img_path}")

            except Exception as e:
                raise ValueError(f"Not able to load image {img_path}: {e}")

        return images

    def extract_input_images(self) -> List[Dict]:
        """
        Extract input image information from the goal text.

        Returns:
            List of dictionaries with image information
        """
        input_images = []

        if "Input image" in self.goal:
            import re

            pattern = r"Input image (\d+)/(\d+) below.*?url: \'(.*?)\'"
            matches = re.findall(pattern, self.goal, re.DOTALL)

            for match in matches:
                input_images.append({"image_number": int(match[0]), "total_images": int(match[1]), "url": match[2]})

        return input_images

    def get_clean_goal(self) -> str:
        """
        Get the goal text without input image information.

        Returns:
            Clean goal text
        """
        input_images = self.extract_input_images()
        if not input_images:
            return self.goal
        clean_goal = re.sub(r"Input image \d+/\d+ below.*?url: \'.*?\'.*?(?=\n|$)", "", self.goal, flags=re.DOTALL)
        return clean_goal.strip()

    def annotate_dot_on_image(self, img, bbox):
        """
        Annotate the given image based on the click action.
        """
        return annotate_action_on_image(img, bbox, marker_style="dot")

    @property
    def states(self) -> List[Dict]:
        """Get the list of states."""
        return self._states

    @property
    def actions(self) -> List[Dict]:
        """Get the list of actions."""
        return self._actions

    @property
    def screenshots(self) -> List[str]:
        """Get the list of screenshot paths."""
        return self._screenshots

    @property
    def task_id(self) -> str:
        """Get the task ID (extracted from filename if available)."""
        return getattr(self, "_task_id", "")

    @task_id.setter
    def task_id(self, value: str):
        """Set the task ID."""
        self._task_id = value

    def write_actions_to_txt(self, output_path: str):
        """
        Write the actions to a txt file.
        """
        with open(output_path, "w") as f:
            for action in self._actions:
                f.write(f"{action['num']}: {action['action_str']}\n | {action['action_str_som']}\n")

    def get_all_roles(self) -> Dict:
        """
        Get all the roles in the trajectory.
        """
        all_roles = {}
        for action in self._actions:
            node = action.get("semantic_info", {}).get("node", {})
            if node:
                all_roles[node.get("role", "")] = node.get("name", "")
        return all_roles

    def get_all_actions_som(self) -> Dict:
        """
        Get all the actions in the trajectory.
        """
        all_actions_som = {}
        for action in self._actions:
            all_actions_som[action["action_str_som"]] = action["action_str"]
        return all_actions_som

    def get_all_parsed_actions(self) -> List[Dict]:
        """
        Get all the parsed actions in the trajectory.
        """
        all_parsed_actions = []
        for action, state in zip(self._actions, self._states):
            parsed_action = action["parsed_action"]
            img_path = state["screenshot_path"]
            all_parsed_actions.append({"parsed_action": parsed_action, "img_path": img_path})
        return all_parsed_actions

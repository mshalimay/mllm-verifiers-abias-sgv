# NOTE[mandrade]: some refactoring for better env feedback hints + marking invalid actions

import base64
import copy
import html
import io
import json
import re
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from PIL import Image

from browser_env import Action, ActionTypes, ObservationMetadata, StateInfo, action2str

HTML_TEMPLATE = """
<!DOCTYPE html>
<head>
    <style>
        pre {{
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
    </style>
</head>
<html>
    <body>
     {body}
    </body>
</html>
"""


def get_render_action(
    action: Action,
    observation_metadata: dict[str, ObservationMetadata],
    action_set_tag: str,
) -> str:
    """Parse the predicted actions for rendering purpose. More comprehensive information"""
    match action_set_tag:
        case "id_accessibility_tree":
            text_meta_data = observation_metadata["text"]
            if action["element_id"] in text_meta_data["obs_nodes_info"]:
                node_content = text_meta_data["obs_nodes_info"][action["element_id"]]["text"]
            else:
                node_content = "No match found"

            action_str = f"<div class='raw_parsed_prediction' style='background-color:grey'><pre>{action['raw_prediction']}</pre></div>"
            action_str += f"<div class='action_object' style='background-color:grey'><pre>{repr(action)}</pre></div>"
            action_str += f"<div class='parsed_action' style='background-color:yellow'><pre>{action2str(action, action_set_tag, node_content)}</pre></div>"

        case "som":
            meta_data = observation_metadata["image"]
            if meta_data["obs_nodes_semantic_info"] and action["element_id"] in meta_data["obs_nodes_semantic_info"]:
                node_content = meta_data["obs_nodes_semantic_info"][action["element_id"]]
            else:
                node_content = "No match found"

            action_str = f"<div class='raw_parsed_prediction' style='background-color:grey'><pre>{action['raw_prediction']}</pre></div>"
            action_str += f"<div class='action_object' style='background-color:grey'><pre>{repr(action)}</pre></div>"
            action_str += f"<div class='parsed_action' style='background-color:yellow'><pre>{action2str(action, action_set_tag, node_content)}</pre></div>"

        case "playwright":
            action_str = action["pw_code"]
        case _:
            raise ValueError(f"Unknown action type {action['action_type'], action_set_tag}")

    return action_str


def is_typeable_element(node_content: str) -> bool:
    # Any button that is not a BUTTON or ANCHOR is deemed valid for type actions.
    # Obs.: not precise; can include more elements
    return not re.search(r"\[BUTTON\]|\[A\]", node_content)


def get_action_description(
    action: Action,
    observation_metadata: dict[str, ObservationMetadata],
    action_set_tag: str,
    action_splitter: str | None = None,
) -> str:
    """Generate the text version of the predicted actions to store in action history for prompt use.
    May contain hint information to recover from the failures"""
    # NOTE[mandrade]: some refactoring for better env feedback hints + marking invalid actions

    match action_set_tag:
        case "id_accessibility_tree":
            text_meta_data = observation_metadata["text"]
            # REVIEW: removed  `id_accessibility_tree` + `som_image` possibility
            # Why: if the observation type is `som_image` observation_metadata["text"] gets empty, giving wrong hint
            if action["action_type"] in [
                ActionTypes.CLICK,
                ActionTypes.HOVER,
                ActionTypes.TYPE,
            ]:
                action_name = str(action["action_type"]).split(".")[1].lower()
                if action["element_id"] in text_meta_data["obs_nodes_info"]:
                    node_content = text_meta_data["obs_nodes_info"][action["element_id"]]["text"]
                    node_content = " ".join(node_content.split()[1:])
                    action_str = action2str(action, action_set_tag, node_content)

                else:
                    action_str = f'Attempt to perfom "{action_name}" on element "[{action["element_id"]}]" but no matching element found. Please check the observation more carefully.'
                    action["invalid"] = True  # type: ignore
            else:
                if action["action_type"] == ActionTypes.NONE:
                    if action_splitter is not None:
                        action_str = f'The previous prediction you issued was "{action["raw_prediction"]}". However, the format was incorrect. Ensure that the action is wrapped inside a pair of {action_splitter} and enclose arguments within [] as follows: {action_splitter}action [arg] {action_splitter}.'
                    else:
                        action_str = f'The previous prediction you issued was "{action["raw_prediction"]}". However, the format was incorrect. Ensure to issue the action in the required format.'
                    action["invalid"] = True  # type: ignore
                else:
                    action_str = action2str(action, action_set_tag, "")

        case "som":
            meta_data = observation_metadata["image"]
            if action["action_type"] in [
                ActionTypes.CLICK,
                ActionTypes.HOVER,
                ActionTypes.TYPE,
                ActionTypes.SELECT_OPTION,
            ]:
                action_name = str(action["action_type"]).split(".")[1].lower()
                if meta_data["obs_nodes_semantic_info"] and action["element_id"] in meta_data["obs_nodes_semantic_info"]:
                    node_content = meta_data["obs_nodes_semantic_info"][action["element_id"]]

                    # Handling cases where type action do not match element type. Info is not given in the prompt, and typing into a button maps to a click due to auto-enter after type.
                    if action["action_type"] == ActionTypes.TYPE and not is_typeable_element(node_content):
                        action_str = f'Attempt to perfom "{action_name}" on "{node_content}" but this action is not valid for this element. Please check the observation more carefully.'
                        action["invalid"] = True  # type: ignore
                    else:
                        action_str = action2str(action, action_set_tag, node_content)
                else:
                    action_str = f'Attempt to perfom "{action_name}" on element "[{action["element_id"]}]" but no matching element found. Please check the observation more carefully.'
                    action["invalid"] = True  # type: ignore
            else:
                if action["action_type"] == ActionTypes.NONE:
                    if action_splitter is not None:
                        action_str = f'The previous prediction you issued was "{action["raw_prediction"]}". However, the format was incorrect. Ensure that the action is wrapped inside a pair of {action_splitter} and enclose arguments within [] as follows: {action_splitter}action [arg] {action_splitter}.'
                    else:
                        action_str = f'The previous prediction you issued was "{action["raw_prediction"]}". However, the format was incorrect. Ensure to issue the action in the required format.'
                    action["invalid"] = True  # type: ignore
                else:
                    action_str = action2str(action, action_set_tag, "")

        case "playwright":
            action_str = action["pw_code"]

        case _:
            raise ValueError(f"Unknown action type {action['action_type']}")
    action_str = re.sub(r"\s+", " ", action_str.replace("\n", "")).strip()
    return action_str


class RenderHelper(object):
    """Helper class to render text and image observations and meta data in the trajectory"""

    def __init__(self, config_file: str | dict, result_dir: str, action_set_tag: str, prefix: str = "", postfix: str = "") -> None:
        if isinstance(config_file, str | Path):
            # Backward compatibility
            with open(config_file, "r") as f:
                _config = json.load(f)
        else:
            _config = config_file

        _config_str = ""
        for k, v in _config.items():
            _config_str += f"{k}: {v}\n"
        _config_str = f"<pre>{_config_str}</pre>\n"
        task_id = _config["task_id"]

        self.action_set_tag = action_set_tag

        str_part = str(task_id)
        if prefix:
            str_part = f"{prefix}_{str_part}"
        if postfix:
            str_part += f"_{postfix}"

        self.render_file = open(Path(result_dir) / f"render_{str_part}.html", "a+", encoding="utf-8")

        self.render_file.truncate(0)
        # write init template
        self.render_file.write(HTML_TEMPLATE.format(body=f"{_config_str}"))
        self.render_file.read()
        self.render_file.flush()

    def render(
        self,
        action: Action,
        state_info: StateInfo,
        meta_data: dict[str, Any],
        render_screenshot: bool = False,
        additional_text: list[str] | None = None,
    ) -> None:
        """Render the trajectory"""

        # Deep copy the action
        action_copy = copy.deepcopy(action)

        # text observation
        observation = state_info["observation"]
        text_obs = observation["text"]
        info = state_info["info"]
        new_content = f"<h2>New Page</h2>\n"
        new_content += f"<h3 class='url'><a href={state_info['info']['page'].url}>URL: {state_info['info']['page'].url}</a></h3>\n"

        text_obs_escaped = html.escape(text_obs)

        new_content += f"<div class='state_obv'><pre>{text_obs_escaped}</pre><div>\n"

        if render_screenshot:
            # image observation
            img_obs = observation["image"]
            image = Image.fromarray(img_obs)
            byte_io = io.BytesIO()
            image.save(byte_io, format="PNG")
            byte_io.seek(0)
            image_bytes = base64.b64encode(byte_io.read())
            image_str = image_bytes.decode("utf-8")
            new_content += f"<img src='data:image/png;base64,{image_str}' style='width:60vw; height:auto;'/><br>"
        # meta data
        new_content += f"<div class='prev_action' style='background-color:pink'>{meta_data['action_str_history'][-1]}</div>\n"

        if "early_stop" in action_copy:
            new_content += f"<div class='early_stop' style='background-color:red'>Early stop: {action_copy['early_stop']}</div>\n"

        # additional text
        if additional_text:
            for text_i, text in enumerate(additional_text):
                # Alternate background color between light green and light blue
                if text_i % 2 == 0:
                    bg_color = "#87CEFA"
                else:
                    bg_color = "#98FB98"
                new_content += f"<div class='additional_text' style='background-color: {bg_color}'>#{text_i + 1}: {text}</div>\n"

        # If verifier_executor_loop_utterances is not empty, extract it and delete the key from Action (to render the action object using less space)
        executor_utterances, critique_utterances = None, None
        if "verifier_executor_loop_utterances" in action_copy:
            executor_utterances = action_copy["verifier_executor_loop_utterances"]["executor"]
            critique_utterances = action_copy["verifier_executor_loop_utterances"]["verifier"]
            del action_copy["verifier_executor_loop_utterances"]

        # Add raw utterance and action object
        action_str = get_render_action(
            action_copy,
            info["observation_metadata"],
            action_set_tag=self.action_set_tag,
        )
        # Add string representation of action with yellow background
        action_str = f"<div class='predict_action'>{action_str}</div>"
        new_content += f"{action_str}\n"

        # Add critique-executor loop
        if executor_utterances is not None and critique_utterances is not None:
            new_content += "<div class='executor_critique_loop'>"
            new_content += "<h4>Executor Critique Loop</h4>"
            for i in range(len(executor_utterances)):
                executor_text = executor_utterances[i].replace("\n", "<br>")
                new_content += (
                    f"<div class='executor_utterance' style='background-color: #D8CEF6'><strong>Executor:</strong> {executor_text}</div>\n"
                )
                if i < len(critique_utterances):
                    critique_text = critique_utterances[i].replace("\n", "<br>")
                    new_content += f"<div class='critique_utterance' style='background-color: #F0E6FF'><strong>Critique:</strong> {critique_text}</div>\n"
            new_content += "</div>"

        # add new content
        self.render_file.seek(0)
        html_content = self.render_file.read()
        soup = BeautifulSoup(html_content, "html.parser")
        body_tag = soup.body

        if body_tag:
            html_body = str(body_tag)
            html_body += new_content
        else:
            # Handle the case when <body> tag is not found
            html_body = new_content

        html_content = HTML_TEMPLATE.format(body=html_body)
        self.render_file.seek(0)
        self.render_file.truncate()
        self.render_file.write(html_content)
        self.render_file.flush()

    def close(self) -> None:
        self.render_file.close()

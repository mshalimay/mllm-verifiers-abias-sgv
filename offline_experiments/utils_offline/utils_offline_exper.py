import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from flask import json

from agrb.constants import AGRB_TO_VWA
from core_utils.string_utils import clean_spaces
from offline_experiments.config.eval_configs import DOMAINS
from offline_experiments.utils_offline.agrb_specific import get_trace_data_agrb
from offline_experiments.utils_offline.osworld_specific import get_intent_message_osw, get_trace_data_osw, get_trajectory_osw

from .vwa_specific import get_intent_message_vwa, get_trace_data_vwa, get_trajectory_vwa


def maybe_swap_sys_user_role(model: str):
    """Swap system role by user role for specific models as recommended by model authors."""

    substrings_to_swap = ["qwen3-vl"]
    # https://www.alibabacloud.com/help/en/model-studio/vision#:~:text=Model%20configuration%3A%20For%20best%20results%2C%20do%20not%20set%20a%20system%20message%20or%20modify%20the%20default%20hyperparameters%20for%20Qwen3%2DVL.
    model_str = model.lower()
    if any(substr in model_str for substr in substrings_to_swap):
        return "user"
    return "system"


def safe_format(string_template: str, fill_with: str = "", **kwargs) -> str:
    """
    Formats a given template using the provided keyword arguments.
    Missing keys in the template are replaced with an empty string.

    Args:
        template (str): The string template with placeholders.
        **kwargs: Key-value pairs for formatting.

    Returns:
        str: The formatted string with missing keys as empty strings.
    """

    class DefaultDict(dict):
        def __missing__(self, key):
            return fill_with

    return string_template.format_map(DefaultDict(**kwargs))


def is_aeval_refine(config):
    if "aeval_refine" in config["prompt_args"].get("sys_prompt", {}):
        return True
    else:
        return False


def get_intent_message(
    config,
    trace_data,
    add_state_idxs: list[int] = [],
    state_img_intros: list[str] = [],
):
    if "vwa" in config["env"]:
        caption_input_images = config.get("additional_config", {}).get("caption_input_images", False)
        return get_intent_message_vwa(
            trace_data=trace_data,
            add_state_idxs=add_state_idxs,
            state_img_intros=state_img_intros,
            aeval_refine=is_aeval_refine(config),
            caption_input_img=caption_input_images,
            config=config,
        )

    elif config["env"] == "osw":
        return get_intent_message_osw(
            trace_data=trace_data,
            add_state_idxs=add_state_idxs,
            state_img_intros=state_img_intros,
        )
    else:
        raise ValueError(f"Environment {config['env']} not supported")


def get_trace_data(env, trace_path, task_id, img_ann_types: list[str] = []):
    if env == "vwa":
        return get_trace_data_vwa(trajectory_path=trace_path, task_id=task_id, img_ann_types=img_ann_types)

    elif env == "osw":
        return get_trace_data_osw(trajectory_path=trace_path, task_id=task_id)

    elif "agrb" in env:
        return get_trace_data_agrb(trajectory_path=trace_path, env=env, img_ann_types=img_ann_types)

    else:
        raise ValueError(f"Environment {env} not supported")


def get_trajectory_msgs(config, trace_data, img_ann_types: list[str] = []):
    env = config["env"]

    if "vwa" in env:
        return get_trajectory_vwa(config, trace_data, is_aeval_refine(config))

    elif env == "osw":
        return get_trajectory_osw(config, trace_data, img_ann_types=img_ann_types)

    else:
        raise ValueError(f"Environment {env} not supported")


def parse_evaluation(response: str) -> list[str]:
    splitters = ["EVALUATION:", "FEEDBACK:"]
    utterances = []
    splitters_group = "|".join(map(re.escape, splitters))
    for splitter in splitters:
        pattern = rf"{splitter}(.*?)(?:\n|{splitters_group}|$)"
        # pattern = rf"(?:\*+|#+|\s*)({splitters_group}):\s*(.*?)(?=\n(?:\s*(?:\*+|#+|\s*)({splitters_group}):)|$)"

        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            utterances.append(clean_spaces(match.group(1)))
        else:
            utterances.append("error")
            # raise ValueError(f"Cannot find {splitter} in {response}")
    return utterances


def get_response_from_html_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        return ""

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            html_content = file.read()

        soup = BeautifulSoup(html_content, "html.parser")

        # Get all section-like elements
        sections = soup.find_all("div", class_="section")

        if sections:
            last_section = sections[-1]
            all_text = last_section.get_text(separator="\n", strip=True)
            all_text = re.sub(r"ASSISTANT\n", "", all_text)
            return all_text
        else:
            return ""

    except Exception as e:
        print(f"Error: {e}")
        return ""


def get_response_from_txt_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        return ""

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        # Remove think blocks
        THINKING_BLOCK_REGEX = re.compile(
            r"^\s*-{3,}\s*THOUGHT\s+START\s*-{3,}\s*$.*?^\s*-{3,}\s*THOUGHT\s+END\s*-{3,}\s*$",
            re.IGNORECASE | re.MULTILINE | re.DOTALL,
        )

        pattern_match = r"==================================\nGENERATION\n==================================(.*)"
        match = re.search(pattern_match, content, re.DOTALL)
        if match:
            response = match.group(1)
            response = response.strip()
            response = re.sub(r"CONTENT TYPE: .*\n", "", response)
            response = re.sub(THINKING_BLOCK_REGEX, "", response)
            return response
        else:
            return ""
    except Exception as e:
        print(f"Error: {e}")
        return ""


def get_response_from_file(file_path: str) -> str:
    if file_path.endswith(".html"):
        return get_response_from_html_file(file_path)
    elif file_path.endswith(".txt"):
        return get_response_from_txt_file(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")


def get_domain_from_path(path: Path | str) -> str:
    path = str(path)
    if "agrb" in str(path):
        if "agrb_mapping" not in globals():
            global agrb_mapping
            agrb_mapping = pd.read_csv(AGRB_TO_VWA)

        task_id = Path(path).stem
        task_id = str(task_id)
        line = agrb_mapping[agrb_mapping["task_id"] == task_id]
        return line["domain"].values[0]

    for domain in DOMAINS:
        match = re.match(rf".*/({domain})/.*", path)
        if match:
            return match.group(1)
    raise ValueError(f"Unable to determine domain from {path}")


def dump_exper_args(
    config: dict,
    gen_config: dict,
):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = config["out_dir"]
    if os.path.exists(f"{output_dir}/exper_args.json"):
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(f"{output_dir}/exper_args.json", "w") as f:
        args_to_dump = {
            "gen_config": gen_config,
            "date": timestamp,
        }

        if config is not None:
            args_to_dump.update(config)

        json.dump(args_to_dump, f, indent=4)


# Deprecated functions
# def _terminate_process_group(process: subprocess.Popen, timeout: float = 5.0):
#     """Terminate the entire process group started for `process`.

#     Sends SIGTERM to the group, waits up to `timeout` seconds, and escalates to SIGKILL if needed.
#     Safe to call multiple times.
#     """
#     try:
#         # When start_new_session=True, the child becomes a new session and process group leader (PGID=PID)
#         pgid = os.getpgid(process.pid)
#     except Exception:
#         # If we cannot get a pgid, fall back to terminating just the child
#         pgid = None

#     try:
#         if pgid is not None:
#             os.killpg(pgid, signal.SIGTERM)
#         else:
#             process.terminate()
#     except ProcessLookupError:
#         return
#     except Exception:
#         # Best-effort; don't raise
#         pass

#     try:
#         process.wait(timeout=timeout)
#         return
#     except Exception:
#         # Still alive; escalate
#         try:
#             if pgid is not None:
#                 os.killpg(pgid, signal.SIGKILL)
#             else:
#                 process.kill()
#         except Exception:
#             pass
#         try:
#             process.wait(timeout=timeout)
#         except Exception:
#             pass


# def run_and_wait(command, logfile_path):
#     if logfile_path:
#         with open(logfile_path, "w") as logfile:
#             process = subprocess.Popen(
#                 command,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.STDOUT,
#                 text=True,
#                 start_new_session=True,  # Make child the leader of a new process group for clean teardown
#             )
#             # Ensure termination tears down the whole process group
#             signal_manager.register_termination_signals(lambda: _terminate_process_group(process))  # type:ignore
#             while True:
#                 output_line = process.stdout.readline()  # type:ignore
#                 if output_line == "" and process.poll() is not None:
#                     break
#                 if output_line:
#                     print(output_line, end="")  # Print to command line
#                     logfile.write(output_line)
#                     logfile.flush()
#             process.wait()
#     else:
#         process = subprocess.Popen(
#             command,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.STDOUT,
#             text=True,
#             start_new_session=True,  # Make child the leader of a new process group for clean teardown
#         )
#         # Ensure termination tears down the whole process group
#         signal_manager.register_termination_signals(lambda: _terminate_process_group(process))  # type:ignore
#         while True:
#             output_line = process.stdout.readline()  # type:ignore
#             if output_line == "" and process.poll() is not None:
#                 break
#             if output_line:
#                 print(output_line, end="")  # Print to command line
#         process.wait()

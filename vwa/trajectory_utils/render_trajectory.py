import argparse
import concurrent.futures
import html
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
from trajectory_utils.trajectory import VWA_Trajectory

from core_utils.file_utils import find_files
from core_utils.image_utils import any_to_b64
from core_utils.string_utils import clean_spaces

refelxion_location_template = f"{{parent_dir}}/{{task_id}}_{{attempt_id}}/{{json_file_name}}"

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        pre {{
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        body {{
            font-family: monospace;
            background-color: #ffffff;
            color: #000000;
        }}
        .step {{
            border: 1px solid #ddd;
            margin: 10px 0;
            padding: 10px;
        }}
        .url a {{
            color: #0645ad;
            text-decoration: none;
        }}
        .text_obs {{
            border: 1px solid #eee;
            padding: 5px;
        }}
        .screenshot {{
            border: 1px solid #eee;
            padding: 5px;
        }}
        .action_prediction {{
            background-color: #e0e0e0;
        }}
        .parsed_action {{
            background-color: #ffd6d6;
        }}
        .verifier_loop {{
            border: 1px dotted #d6b4fc;
            margin: 10px 0;
            padding: 5px;
            margin-left: 10px;
        }}
        .generator_utterance {{
            background-color: #D8CEF6;
            padding: 5px;
            margin: 5px 0;
        }}
        .verifier_utterance {{
            background-color: #F0E6FF;
            padding: 5px;
            margin: 5px 0;
        }}

    </style>
</head>
<body>
{body}
</body>
</html>
"""


def parse_evaluation(content: str, eval_scores: list[str]) -> str:
    """
    Extracts the evaluation score (e.g., SUCCESS, PARTIAL SUCCESS, FAILURE) from the EVALUATION section.
    """
    content = clean_spaces(content)
    eval_scores_str = "|".join(sorted(eval_scores, key=len, reverse=True))  # Join criteria with OR operator
    status_pattern = rf"(?i)\s*({eval_scores_str})"

    match = re.search(status_pattern, content)
    if match:
        return match.group(1).upper()  # Normalize to uppercase

    else:
        raise Exception(f"Cannot determine evaluation score in: {content}")


def textify_verifier_response(
    raw_verifier_response: str,
    splitters: list[str] = [],
    required_splitters: list[str] = [],
    eval_key: str = "EVALUATION",
    eval_scores: list[str] = ["SUCCESS", "PARTIAL SUCCESS", "FAILURE"],
) -> str:
    if not raw_verifier_response:
        return ""

    if not splitters:
        return raw_verifier_response

    if not required_splitters:
        required_splitters = splitters

    parsed_data = parse_verifier_response(
        raw_verifier_response,
        splitters=splitters,
        eval_key=eval_key,
        required_splitters=required_splitters,
        eval_scores=eval_scores,
    )
    joined_str = "\n".join([f"{k}: {v}" for k, v in parsed_data.items()])
    return f"\n{joined_str}"


def parse_verifier_response(
    response: str,
    splitters: list[str],
    eval_key: str,
    required_splitters: list[str],
    eval_scores: list[str],
) -> dict[str, Any]:
    parsed_data: dict[str, Any] = {}
    splitters_group: str = "|".join(map(re.escape, splitters))

    # Iterate over splitters and parse corresponding sections
    for splitter in splitters:
        # Use regex to extract the section content
        pattern = rf"{re.escape(splitter)}(.*?)(?=\n(?:{splitters_group})|$)"
        matches = re.findall(pattern, response, re.IGNORECASE | re.DOTALL)
        if matches:
            content = clean_spaces(matches[-1].strip().strip(":"))
            if eval_key in splitter:
                # Map evaluation criteria like SUCCESS or FAILURE
                parsed_data[eval_key] = parse_evaluation(content, eval_scores=eval_scores)
            else:
                # General parsing for all sections, including COMPARISON
                parsed_data[re.sub(r":$", "", splitter)] = content  # Remove trailing ":"

    if not all(req_splitter in parsed_data for req_splitter in required_splitters):
        raise Exception(f"Cannot find all required splitters in {response}")

    return parsed_data


def try_get_reflexion_data(traj_file: Path, task_id: int | str) -> dict:
    task_id = str(task_id)
    if not task_id:
        return {}

    try:
        parent_dir = traj_file.parent.parent
        cur_attempt_id = traj_file.parent.name.split("_")[1]
        prev_attempt_id = str(int(cur_attempt_id) - 1)
        if prev_attempt_id == "-1":
            return {}
        prev_traj_file = refelxion_location_template.format(
            parent_dir=parent_dir, task_id=task_id, attempt_id=prev_attempt_id, json_file_name=traj_file.name
        )
        prev_trajectory = VWA_Trajectory.from_json(str(prev_traj_file))
        return prev_trajectory.metadata.get("reflexion_data", {})
    except Exception as e:
        print(f"Error getting reflexion data from {prev_traj_file}: {e}")
        return {}


def format_evaluator_html(evaluator):
    if not evaluator:
        return "<p><strong>Evaluator:</strong> <em>Not found</em></p>"
    html = ["<div class='evaluator'><h4>Evaluator</h4>"]
    html.append(f"<b>Function:</b> {evaluator.get('func', '')}<br>")
    if "result" in evaluator:
        html.append(f"<b>Result:</b> <pre>{json.dumps(evaluator['result'], indent=2)}</pre>")
    if "expected" in evaluator:
        html.append(f"<b>Expected:</b> <pre>{json.dumps(evaluator['expected'], indent=2)}</pre>")
    html.append("</div>")
    return "\n".join(html)


def trajectory_to_html(
    traj_file: Path,
    render_text_obs: bool = False,
    ann_types: list[str] = [],
    skip_not_completed: bool = False,
    overwrite: bool = False,
    out_file_template: str = "",
) -> tuple[str, VWA_Trajectory | None]:
    try:
        trajectory = VWA_Trajectory.from_json(traj_file)
    except Exception as e:
        print(f"Error reading {traj_file}: {e}")
        return "", None

    if not trajectory.episode_completed and skip_not_completed:
        print(f"Trajectory {traj_file} is not completed. Skipping")
        return "", None

    # Gather header info.
    task_id = trajectory.task_id or ""
    score = trajectory.cum_reward
    domain = trajectory.domain
    objective_text = ""

    if not overwrite and Path(out_file_template.format(domain=domain, task_id=task_id)).exists():
        print(f"Trajectory {traj_file} already exists. Skipping")
        return "", trajectory

    objective_text = trajectory.objective["text"]
    objective_imgs = trajectory.objective["images"]
    objective_img_captions = trajectory.objective.get("img_captions", [""] * len(objective_imgs))

    if trajectory.config:
        _config = trajectory.config
        config_text = ""
        for k, v in _config.items():
            config_text += f"{k}: {v}\n"
        task_id = _config["task_id"]

    # Compute number of verifier loops by checking each state's keys.
    num_verifier_loops = 0
    no_verifier_score = None
    retrieved_knowledge = ""
    for state in trajectory.states:
        if state.get("scores_per_round"):
            num_verifier_loops += 1
            if "score" in state["scores_per_round"][0] and no_verifier_score is None:
                no_verifier_score = state["scores_per_round"][0]["score"]
        if "retrieved_knowledge" in state and not retrieved_knowledge:
            retrieved_knowledge = state["retrieved_knowledge"].replace("\n", "<br>")

    html_parts = []

    # Header information including the verifier loops count.
    html_parts.append(f"<p><strong>Task id:</strong> {task_id}</p>")
    html_parts.append(f"<p><strong>Domain:</strong> {domain}</p>")
    html_parts.append(f"<p><strong>Score:</strong> {score}</p>")
    # html_parts.append(f"<p><strong>Verifier Loops:</strong> {num_verifier_loops}</p>")
    html_parts.append(f"<p><strong>Objective:</strong> {objective_text}</p>")
    if objective_imgs:
        for img, caption in zip(objective_imgs, objective_img_captions):
            html_parts.append(f"<img src='{any_to_b64(img)}' style='width:20vw; height:auto;'/><br>")
            if caption:
                html_parts.append(f"<p><strong>Caption:</strong> {html.escape(caption)}</p>")

    html_parts.append(f"<div><strong><br>Task Config:</strong><pre>{html.escape(config_text.strip())}</pre></div>")

    if retrieved_knowledge is not None:
        retrieved_knowledge = re.sub("\n", "<br>", retrieved_knowledge)
        html_parts.append(f"<div><strong>Retrieved Knowledge:</strong><pre>{retrieved_knowledge}</pre></div>")

    reflexion_data = try_get_reflexion_data(traj_file, task_id)
    if reflexion_data:
        reflexion_str = html.escape(json.dumps(reflexion_data, indent=2))
        html_parts.append(f"<div><strong>Reflexion Data:</strong><pre>{reflexion_str}</pre></div>")

    # Process each step and generate a block for each.
    for t, (state, action) in enumerate(zip(trajectory.states, trajectory.actions)):
        html_parts.append("<div class='step'>")
        html_parts.append(f"<h2>Step {t}</h2>")

        # URL of the page
        url = state.get("url", "")
        if url:
            html_parts.append(f"<h4 class='url'><a href='{url}'>URL: {url}</a></h4>")

        # Render text observation optionally
        if render_text_obs:
            text_obs = state.get("text", "")
            if text_obs:
                text_obs_escaped = html.escape(text_obs)
                html_parts.append("<div class='text_obs'>")
                html_parts.append("<h3>Text Observation:</h3>")
                html_parts.append(f"<pre>{text_obs_escaped}</pre>")
                html_parts.append("</div>")

        # Get & annotate screenshots for state.
        if state.get("images", []):
            trajectory.annotate_images(state, action, ann_types=ann_types)
            html_parts.append("<div class='screenshot'>")
            html_parts.append("<h3>Screenshots:</h3>")
            html_parts.append(f"<img src='{any_to_b64(state['images'][-1])}' style='max-width:50vw; height:auto;'/><br>")
            html_parts.append("</div>")
        else:
            # print(f"No screenshot provided for state {t}, traj_file: {traj_file}")
            # html_parts.append("<p>[No screenshot provided]</p>")
            raise Exception(f"No screenshot provided for state {t}, traj_file: {traj_file}")

        # Last executor atomic action
        action_str = ""
        executor_action = action["agents"]["executor"][-1]  # type: ignore
        html_parts.append("<div class='executor_action'>")
        html_parts.append("<h3>Action Prediction:</h3>")
        if executor_action.get("thought_summary", ""):
            action_str += (
                f"<strong> === Thought Summary: === </strong> {executor_action['thought_summary']} === End of Thought Summary === <br>"
            )
        action_str += executor_action["generated_text"]
        html_parts.append(f"<div class='action_prediction'><pre>{action_str}</pre></div>")
        html_parts.append(f"<div class='parsed_action'><pre>{executor_action['action_str_env_parsed']}</pre></div>")
        html_parts.append("</div>")

        # Add generator-verifier loop.
        if state.get("scores_per_round", None):
            loop = state["scores_per_round"]
            html_parts.append("<div class='verifier_loop'>")
            html_parts.append("<h4>Generator-Verifier Loop:</h4>")
            for item in loop:
                verifier_text = textify_verifier_response(item.get("verifier_raw_response", ""), splitters=[])
                generator_text = item.get("raw_prediction", "")
                if generator_text:
                    generator_text_html = generator_text.replace("\n", "<br>")
                    html_parts.append(f"<div class='generator_utterance'><strong>Generator:<br></strong> {generator_text_html}<br></div>")
                else:
                    raise Exception(f"No generator text in loop for state {t}, traj_file: {traj_file}")
                if "score" in item:
                    html_parts.append(f"<div class='score' style='background-color: #ffffe0'>Score: {item['score']}</div>")

                if verifier_text:
                    verifier_text_html = verifier_text.replace("\n", "<br>")
                    html_parts.append(f"<div class='verifier_utterance'><strong>Verifier:</strong> {verifier_text_html}</div>")
                else:
                    raise Exception(f"No verifier text provided for state {t}, traj_file: {traj_file}")

            # Finish with predicted action if not the last step.
            if t < len(trajectory.states) - 1:
                html_parts.append(f"<div class='generator_utterance'><strong>Generator:</strong> {action_str}</div>")
            html_parts.append("</div>")

        html_parts.append("</div>")

    return "\n".join(html_parts), trajectory


def process_trajectory(
    traj_file: Path | str,
    out_file_template: str,
    render_text_obs: bool = False,
    ann_types: list[str] = [],
    skip_not_completed: bool = False,
    overwrite: bool = False,
    skip_html: bool = False,
) -> tuple[str, VWA_Trajectory | None, Path | None]:
    """
    Process a single trajectory file: convert it to HTML and write it to disk.
    Returns the path of the output HTML file as a string if successful, or an empty string if not.
    """
    try:
        traj_file = Path(traj_file)
        if skip_html:
            trajectory = VWA_Trajectory.from_json(traj_file)
            if skip_not_completed and not trajectory.episode_completed:
                return "", None, traj_file
            return "", trajectory, traj_file

        html_fragment, trajectory = trajectory_to_html(
            traj_file,
            render_text_obs=render_text_obs,
            ann_types=ann_types,
            skip_not_completed=skip_not_completed,
            overwrite=overwrite,
            out_file_template=out_file_template,
        )
        if not html_fragment or not trajectory:
            return "", trajectory, traj_file
        final_html = HTML_TEMPLATE.format(body=html_fragment)

        output_file = Path(out_file_template.format(domain=trajectory.domain, task_id=trajectory.task_id))
        os.makedirs(output_file.parent, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(final_html)
            print(f"{output_file}")
        return str(output_file), trajectory, traj_file
    except Exception as e:
        print(f"Error processing {traj_file}: {e}, {traceback.format_exc()}")
        return "", None, None


def is_valid_score(score: Any) -> bool:
    if isinstance(score, (float, int)):
        if score != score:
            return False
        return True
    return False


def main(args):
    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        print(f"Error: Base directory '{base_dir}' does not exist.")
        sys.exit(1)

    traj_files = find_files(
        base_dir,
        "*trajectory*.json",
        include_any_strs=args.must_include_strs,
        must_exclude_strs=args.must_exclude_strs,
    )

    if not traj_files:
        print(f"No trajectory.json files found under '{base_dir}'.")
        sys.exit(0)

    print(f"Found {len(traj_files)} trajectory file(s). Processing...")
    # Parallelize processing using ProcessPoolExecutor.

    scores_per_domain = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all trajectory files for processing.
        futures = {
            executor.submit(
                process_trajectory,
                traj_file,
                args.out_file_template,
                args.render_text_obs,
                args.ann_types,
                args.skip_not_completed,
                args.overwrite,
                args.skip_html,
            ): traj_file
            for traj_file in traj_files
        }
        for future in concurrent.futures.as_completed(futures):
            _, trajectory, traj_file = future.result()
            if not trajectory or not is_valid_score(trajectory.cum_reward):
                continue
            domain = trajectory.domain

            if domain not in scores_per_domain:
                scores_per_domain[domain] = {}

            if trajectory.task_id in scores_per_domain[domain]:
                if trajectory.cum_reward > scores_per_domain[domain][trajectory.task_id]["score"]:
                    scores_per_domain[domain][trajectory.task_id] = {
                        "score": trajectory.cum_reward,
                        "traj_file": str(traj_file),
                    }
            else:
                scores_per_domain[domain][trajectory.task_id] = {
                    "score": trajectory.cum_reward,
                    "traj_file": str(traj_file),
                }
    # Dump to csv
    rows = []
    for domain, task_scores in scores_per_domain.items():
        for task_id, data in task_scores.items():
            rows.append({"score": data["score"], "task_id": task_id, "domain": domain, "traj_file": data["traj_file"]})
    df = pd.DataFrame(rows, columns=["score", "task_id", "domain", "traj_file"])
    df.to_csv(f"{base_dir}/trajectory_scores.csv", index=False)
    print(f"Saved scores to {base_dir}/trajectory_scores.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir", nargs="?", default="", help="Base directory to search for trajectories")
    parser.add_argument("--render_text_obs", action="store_true", help="Render text observations")
    parser.add_argument(
        "--ann_types",
        type=list[str],
        default=[],
        help="Annotation types to render. Options: som: set-of-marks, coord: annotates a dot on elements interacted with. Separate multiple types with commas.",
        nargs="+",
    )
    parser.add_argument(
        "--must_include_strs",
        type=list[str],
        default=[],
        help="Only include trajectories that contain any of these strings. Separate multiple strings with commas.",
        nargs="+",
    )
    parser.add_argument(
        "--must_exclude_strs",
        type=list[str],
        default=[],
        help="Exclude trajectories that contain any of these strings. Separate multiple strings with commas.",
        nargs="+",
    )
    parser.add_argument("--skip_not_completed", action="store_true", help="Skip trajectories that failed execution")
    parser.add_argument(
        "--max_workers", type=int, default=16, help="Max number of worker processes to use for parallel trajectory processing"
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing rendered HTML files")
    parser.add_argument("--skip_html", action="store_true", help="Skip HTML rendering")
    args = parser.parse_args()

    args.out_file_template = f"{args.base_dir}/{{domain}}/{{task_id}}.html"
    main(args)

# Usage examples:
# python trajectory_utils/render_trajectory.py <base_dir> [options]

# Basic (render HTML for all `*trajectory*.json` under a directory):
#   python vwa/trajectory_utils/render_trajectory.py \
#     /path/to/experiments \
#
# Render text observations + annotate screenshots (multiple annotation types):
#   python vwa/trajectory_utils/render_trajectory.py \
#     /path/to/experiments \
#     --render_text_obs \
#     --ann_types som coord \
#
# Only include / exclude certain trajectories by substring match on their file paths (only include classifieds-61 and shopping-1, exclude any trajectory with debug or tmp in the path):
#   python vwa/trajectory_utils/render_trajectory.py \
#     /path/to/experiments \
#     --must_include_strs "classifieds-61" "shopping-1" \
#     --must_exclude_strs "debug" "tmp" \

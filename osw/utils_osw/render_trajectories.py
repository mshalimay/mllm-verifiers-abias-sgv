import concurrent.futures
import json
import os
import sys
import traceback
from pathlib import Path

import pandas as pd

# Updated HTML template with escaped curly braces in the CSS.
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Trajectories</title>
    <style>
        pre {{
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        .trajectory {{
            border: 1px solid #ccc;
            margin: 20px;
            padding: 10px;
        }}
        .trajectory h2 {{
            margin-top: 0;
        }}
        .step {{
            border: 1px solid #ddd;
            margin: 10px 0;
            padding: 10px;
        }}
        .executor_utterance {{
            background-color: #D8CEF6;
            padding: 5px;
            margin: 5px 0;
        }}
        .critique_utterance {{
            background-color: #F0E6FF;
            padding: 5px;
            margin: 5px 0;
        }}
        .executor_critique_loop {{
            border: 1px dashed #aaa;
            margin: 10px 0;
            padding: 5px;
        }}
        .verifier_loop {{
            border: 1px dashed #ff9900;
            margin: 10px 0;
            padding: 5px;
        }}
        .verifier_output {{
            background-color: #ffe6cc;
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

from core_utils.file_utils import find_files
from core_utils.image_utils import any_to_b64
from osw.utils_osw.utils_osw import TrajectoryView, annotate_action_on_image


def load_evaluator_info(domain, task_id):
    """Load evaluator info from the task config JSON, or return None if not found."""
    config_path = Path("osw_traces/task_configs") / domain / f"{task_id}.json"
    if not config_path.exists():
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config.get("evaluator")
    except Exception as e:
        print(f"Error loading evaluator for {domain}/{task_id}: {e}\n{traceback.format_exc()}")
        return None


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


def trajectory_to_html(traj_file: Path) -> str:
    try:
        trajectory_view = TrajectoryView(trajectory_path=traj_file, to_english=False)
    except Exception as e:
        print(f"Error reading {traj_file}: {e}")
        return ""

    # Gather header info.
    task_id = trajectory_view.task_id
    score = trajectory_view.score
    domain = trajectory_view.domain
    objective = trajectory_view.objective

    # Load evaluator info
    evaluator = load_evaluator_info(domain, task_id)

    # Compute number of verifier loops by checking each state's keys.
    num_verifier_loops = sum(1 for state in trajectory_view.states if "verifier_loop" in state)

    # Load the raw trajectory JSON to access 'first_pass_knowledge' if present
    try:
        with open(traj_file, "r", encoding="utf-8") as f:
            raw_traj = json.load(f)
        first_pass_knowledge = raw_traj.get("first_pass_knowledge", None)
    except Exception as e:
        first_pass_knowledge = None

    html_parts = []
    # Start a single container for both header and step details.
    html_parts.append("<div class='trajectory'>")

    # Header information including the verifier loops count.
    html_parts.append(f"<p><strong>Task id:</strong> {task_id}</p>")
    html_parts.append(f"<p><strong>Score:</strong> {score}</p>")
    html_parts.append(f"<p><strong>Domain:</strong> {domain}</p>")
    html_parts.append(f"<p><strong>Objective:</strong> {objective}</p>")
    html_parts.append(f"<p><strong>Verifier Loops:</strong> {num_verifier_loops}</p>")
    # Add evaluator info
    html_parts.append(format_evaluator_html(evaluator))
    # Add first_pass_knowledge at the end, preserving newlines
    if first_pass_knowledge is not None:
        html_parts.append(f"<div><strong>First Pass Knowledge:</strong><pre>{first_pass_knowledge}</pre></div>")

    # Process each step and generate a block for each.
    for t, (state, action) in enumerate(zip(trajectory_view.states, trajectory_view.actions)):
        html_parts.append("<div class='step'>")
        html_parts.append(f"<h3>State {t}</h3>")

        generated_text = action["texts"][-1]

        # For the screenshot, check the 'screenshots' list if available.
        screenshots = state["observation"]["images"]
        if screenshots:
            # Pick the last screenshot.
            last_ss = screenshots[-1]
            last_ss = annotate_action_on_image(last_ss, generated_text)
            html_parts.append(f"<img src='{any_to_b64(last_ss, add_header=True)}' style='max-width:50vw; height:auto;'/><br>")
        else:
            html_parts.append("<p>[No screenshot provided]</p>")

        # Add the generated text.
        html_parts.append(f"<pre>{generated_text}</pre>")

        # Check if a verifier loop exists in the 'verifier_loop' key.
        if "verifier_loop" in state:
            loop = state["verifier_loop"]
            html_parts.append("<div class='verifier_loop'>")
            html_parts.append("<h4>Verifier Loop</h4>")
            for item in loop:
                verifier_text = item.get("verifier", "")
                generator_text = item.get("agent", "")
                if verifier_text:
                    verifier_text_html = verifier_text.replace("\n", "<br>")
                    html_parts.append(f"<div class='verifier_output'><pre>{verifier_text_html}</pre></div>")
                if generator_text:
                    generator_text_html = generator_text.replace("\n", "<br>")
                    html_parts.append(f"<div class='generator_output'><pre>{generator_text_html}</pre></div>")
            html_parts.append("</div>")
        html_parts.append("</div>")
    html_parts.append("</div>")  # end .trajectory

    return "\n".join(html_parts)


def process_trajectory(traj_file: Path | str) -> str:
    """
    Process a single trajectory file: convert it to HTML and write it to disk.
    Returns the path of the output HTML file as a string if successful, or an empty string if not.
    """
    traj_file = Path(traj_file)
    html_fragment = trajectory_to_html(traj_file)
    if not html_fragment:
        return ""
    final_html = HTML_TEMPLATE.format(body=html_fragment)
    output_file = traj_file.with_suffix(".html")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(final_html)
    return str(output_file)


def extract_first_verifier_steps(traj_files, output_csv="first_verifier_steps.csv"):
    """
    For each trajectory, extract (domain, task_id, first_verifier_step) and write to a CSV.
    """
    records = []
    for traj_file in traj_files:
        try:
            trajectory_view = TrajectoryView(trajectory_path=traj_file, to_english=False)
        except Exception as e:
            print(f"Error reading {traj_file}: {e}")
            continue
        domain = trajectory_view.domain
        task_id = trajectory_view.task_id
        # Find the first step index (1-based) where verifier_loop appears
        first_verifier_step = None
        for idx, state in enumerate(trajectory_view.states):
            if "verifier_loop" in state:
                # states[0] is the initial state, so step index in trajectory is idx
                first_verifier_step = idx
                break
        records.append({"domain": domain, "task_id": task_id, "first_verifier_step": first_verifier_step})
    # Write to CSV
    try:
        df = pd.DataFrame(records)
        df.to_csv(output_csv, index=False)
        print(f"Wrote first verifier steps to {output_csv}")
    except ImportError:
        # Fallback to csv module
        import csv

        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["domain", "task_id", "first_verifier_step"])
            writer.writeheader()
            writer.writerows(records)
        print(f"Wrote first verifier steps to {output_csv} (csv module)")


def main():
    # Use the first argument as the base directory; default to "trace_osworld"
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir", nargs="?", default="trace_osworld", help="Base directory to search for trajectories")
    parser.add_argument("--extract_verifier_steps", action="store_true", help="Extract first verifier steps to CSV and exit")
    parser.add_argument("--verifier_csv", type=str, default="first_verifier_steps.csv", help="CSV output for verifier steps")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        print(f"Error: Base directory '{base_dir}' does not exist.")
        sys.exit(1)

    traj_files = find_files(base_dir, "trajectory.json")

    if not traj_files:
        print(f"No trajectory.json files found under '{base_dir}'.")
        sys.exit(0)

    if args.extract_verifier_steps:
        extract_first_verifier_steps(traj_files, output_csv=args.verifier_csv)
        return

    print(f"Found {len(traj_files)} trajectory file(s). Processing...")
    # Parallelize processing using ProcessPoolExecutor.
    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        # Submit all trajectory files for processing.
        futures = {executor.submit(process_trajectory, traj_file): traj_file for traj_file in traj_files}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                print(f"HTML trace written to: {result}")


if __name__ == "__main__":
    main()

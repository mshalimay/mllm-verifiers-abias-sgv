#!/usr/bin/env python3
"""
Script to render visualwebarena trajectories into HTML files.
Uses the VisualWebArenaTrajectoryView class for trajectory processing.
"""

import concurrent.futures
import html
import json
import os
import sys
from pathlib import Path

from agrb.constants import TRAJECTORIES_DIR

# Import our custom trajectory view
from agrb.utils_agrb.agrb_utils import find_trajectory_files, get_human_annotation
from agrb.utils_agrb.vwa_trajectory import VWATrajectoryView
from core_utils.image_utils import any_to_b64
from core_utils.logger_utils import logger

# HTML template for visualwebarena trajectories
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>VisualWebArena Trajectory</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .trajectory {{
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            margin: 20px 0;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .trajectory h1 {{
            color: #333;
            border-bottom: 2px solid #007acc;
            padding-bottom: 10px;
        }}
        .trajectory h2 {{
            color: #555;
            margin-top: 30px;
        }}
        .metadata {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
            border-left: 4px solid #007acc;
        }}
        .metadata p {{
            margin: 5px 0;
        }}
        .step {{
            border: 1px solid #e0e0e0;
            border-radius: 5px;
            margin: 20px 0;
            padding: 15px;
            background-color: #fafafa;
        }}
        .step h3 {{
            color: #007acc;
            margin-top: 0;
            border-bottom: 1px solid #e0e0e0;
            padding-bottom: 10px;
        }}
        .screenshot {{
            text-align: center;
            margin: 15px 0;
        }}
        .screenshot img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .action {{
            background-color: #e3f2fd;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            border-left: 4px solid #2196f3;
        }}
        .parsed-action {{
            background-color: #e3f2fd;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            border-left: 4px solid #2196f3;
        }}
        .reasoning {{
            background-color: #fff3e0;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            border-left: 4px solid #ff9800;
            font-family: monospace;
            white-space: pre-wrap;
        }}
        .url {{
            background-color: #e8f5e8;
            padding: 8px;
            border-radius: 3px;
            margin: 5px 0;
            font-family: monospace;
            word-break: break-all;
        }}
        .stats {{
            background-color: #f3e5f5;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            border-left: 4px solid #9c27b0;
        }}
        .stats pre {{
            margin: 0;
            font-size: 12px;
            overflow-x: auto;
        }}
        .goal {{
            background-color: #e8f5e8;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
            border-left: 4px solid #4caf50;
            font-style: italic;
        }}
        .input-images {{
            background-color: #fff8e1;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            border-left: 4px solid #ffc107;
        }}
        .input-images img {{
            max-width: 200px;
            height: auto;
            margin: 5px;
            border: 1px solid #ddd;
            border-radius: 3px;
        }}
        pre {{
            white-space: pre-wrap;
            word-wrap: break-word;
            background-color: #f5f5f5;
            padding: 10px;
            border-radius: 3px;
            border: 1px solid #ddd;
        }}
    </style>
</head>
<body>
{body}
</body>
</html>
"""
OUTPUT_DIR = Path("trajectories/cleaned/html")

all_roles = {}
all_actions_som = {}
all_parsed_actions = {}
failed_trajectories = []


def trajectory_to_html(
    trajectory: Path | VWATrajectoryView,
    ann_types: list[str] = ["som_coord"],
    output_dir: Path | str = "",
) -> str:
    """Convert a visualwebarena trajectory file to HTML using the TrajectoryView class."""
    # try:
    # Create trajectory view

    if isinstance(trajectory, Path):
        trajectory_view = VWATrajectoryView(
            trajectory_path=str(trajectory),
            ann_types=ann_types,
            trajectories_dir=TRAJECTORIES_DIR,
        )
    else:
        trajectory_view = trajectory

    # except Exception as e:
    #     logger.info(f"Error creating trajectory view for {traj_file}: {e}")
    #     return ""

    # Get trajectory summary
    html_parts = []
    html_parts.append("<div class='trajectory'>")

    # Header information
    html_parts.append(f"<h1>VisualWebArena Trajectory: {trajectory_view.task_id}</h1>")

    # Metadata section
    html_parts.append("<div class='metadata'>")
    if hasattr(trajectory_view, "task_id"):
        html_parts.append(f"<p><strong>Task ID:</strong> {trajectory_view.task_id}</p>")
    html_parts.append(f"<p><strong>Benchmark:</strong> {trajectory_view.benchmark}</p>")
    html_parts.append(f"<p><strong>Agent:</strong> {trajectory_view.agent}</p>")
    html_parts.append(f"<p><strong>Model:</strong> {trajectory_view.model}</p>")
    html_parts.append(f"<p><strong>Experiment:</strong> {trajectory_view.experiment}</p>")
    html_parts.append(f"<p><strong>Valid:</strong> {trajectory_view.valid}</p>")
    html_parts.append(f"<p><strong>Total Steps:</strong> {len(trajectory_view.states)}</p>")
    html_parts.append(f"<p><strong>Cumulative Reward:</strong> {trajectory_view.cum_reward}</p>")

    human_annotations = get_human_annotation(trajectory_view.task_id, trajectory_view.benchmark, trajectory_view.experiment)
    if human_annotations is not None:
        html_parts.append(f"<p><strong>Human Mean Score:</strong> {human_annotations['human_mean_score']}</p>")

    if trajectory_view.seed is not None:
        html_parts.append(f"<p><strong>Seed:</strong> {trajectory_view.seed}</p>")
    html_parts.append("</div>")

    # Goal section
    html_parts.append("<div class='goal'>")
    html_parts.append(f"<strong>Goal:</strong> {trajectory_view.get_clean_goal()}")
    html_parts.append("</div>")

    # Input images section (if any)
    input_images = trajectory_view.extract_input_images()
    if input_images:
        html_parts.append("<div class='input-images'>")
        html_parts.append("<h3>Input Images</h3>")
        for img_info in input_images:
            html_parts.append(f"<p><strong>Image {img_info['image_number']}/{img_info['total_images']}:</strong></p>")
            html_parts.append(f"<div class='url'>{img_info['url']}</div>")
        html_parts.append("</div>")

    # Process each step
    for i in range(len(trajectory_view.states)):
        state = trajectory_view.states[i]
        action = trajectory_view.actions[i]

        html_parts.append(f"<div class='step'>")
        html_parts.append(f"<h3>Step {i}</h3>")

        # Screenshot
        url = state.get("url", "")
        if url:
            html_parts.append(f"<div class='url'><strong>URL:</strong> {html.escape(url)}</div>")

        # Screenshot

        images = state["observation"]["images"]
        if images:
            html_parts.append("<div class='screenshot'>")
            b64_img = any_to_b64(images[-1], add_header=True)
            html_parts.append(f"<img src='{b64_img}' alt='Screenshot step {i}' />")
            html_parts.append("</div>")

        # Last action error
        last_action_error = state.get("last_action_error", "")
        if last_action_error:
            html_parts.append(f"<div class='reasoning'><strong>Last Action Error:</strong> {html.escape(last_action_error)}</div>")

        # Reasoning
        reasoning = action.get("reasoning_str", "")
        if reasoning:
            html_parts.append(f"<div class='reasoning'><strong>Reasoning:</strong> {html.escape(reasoning)}</div>")

        # Action
        action_str = action.get("action_str", "")
        if action_str:
            html_parts.append(f"<div class='action'><strong>Action:</strong> {html.escape(action_str)}</div>")

        # Parsed action
        parsed_action = action.get("parsed_action", {})
        if parsed_action:
            html_parts.append(f"<div class='parsed-action'><strong>Parsed Action:</strong> {html.escape(str(parsed_action))}</div>")

        # Semantic action
        som_action = action.get("action_str_som", "")
        if som_action:
            html_parts.append(f"<div class='som-action'><strong>Semantic Action:</strong> {html.escape(str(som_action))}</div>")

        # Focused element
        focused_element = state.get("focused_element", "")
        if focused_element:
            html_parts.append(f"<div class='action'><strong>Focused Element:</strong> {html.escape(str(focused_element))}</div>")

        html_parts.append("</div>")

    html_parts.append("</div>")  # end .trajectory

    return "\n".join(html_parts)


def process_trajectory(
    traj_file: Path | str,
    annotate_images: bool = True,
    ann_types: list[str] = ["som_coord"],
    output_dir: Path | str = "",
) -> tuple[str, VWATrajectoryView]:
    """
    Process a single trajectory file: convert it to HTML and write it to disk.
    Returns the path of the output HTML file as a string if successful, or an empty string if not.
    """
    traj_file = Path(traj_file)
    trajectory_view = VWATrajectoryView(
        trajectory_path=str(traj_file),
        ann_types=ann_types,
        trajectories_dir=TRAJECTORIES_DIR,
    )
    trajectory_view.task_id = traj_file.stem
    html_fragment = trajectory_to_html(trajectory_view, ann_types, output_dir)

    if not html_fragment:
        return "", trajectory_view

    final_html = HTML_TEMPLATE.format(body=html_fragment)
    output_file = traj_file.with_suffix(".html")
    os.makedirs(output_file.parent, exist_ok=True)

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(final_html)
        return str(output_file), trajectory_view
    except Exception as e:
        logger.error(f"Error writing HTML file {output_file}: {e}", exc_info=True)
        return "", trajectory_view


def main():
    """Main function to process all trajectory files."""
    import argparse

    global all_roles, all_actions_som, all_parsed_actions, failed_trajectories

    parser = argparse.ArgumentParser(description="Render VisualWebArena trajectories to HTML")
    parser.add_argument(
        "base_dir",
        nargs="?",
        default="trajectories/cleaned/visualwebarena/GenericAgent-anthropic_claude-3.7-sonnet",
        help="Base directory to search for trajectory files (default: trajectories/cleaned)",
    )
    parser.add_argument("--output_dir", type=str, help="Output directory for HTML files (default: same as trajectory files)")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of worker processes (default: 4)")

    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        logger.info(f"Error: Base directory '{base_dir}' does not exist.")
        sys.exit(1)

    logger.info(f"Searching for trajectory files in '{base_dir}'...")
    trajectory_files = find_trajectory_files(str(base_dir))

    if not trajectory_files:
        logger.info(f"No JSON trajectory files found under '{base_dir}'.")
        sys.exit(0)

    logger.info(f"Found {len(trajectory_files)} trajectory file(s). Processing...")

    if not args.output_dir:
        args.output_dir = base_dir

    # Process files in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all trajectory files for processing
        futures = {executor.submit(process_trajectory, traj_file): traj_file for traj_file in trajectory_files}

        successful_count = 0
        for future in concurrent.futures.as_completed(futures):
            traj_file = futures[future]
            try:
                out_path, trajectory_view = future.result()

                if out_path:
                    logger.info(f"✓ HTML written to: {out_path}")
                    successful_count += 1
                    task_id = Path(traj_file).stem
                    all_actions_som[task_id] = trajectory_view.get_all_actions_som()
                    all_parsed_actions[task_id] = trajectory_view.get_all_parsed_actions()
                    all_roles[task_id] = trajectory_view.get_all_roles()
                else:
                    logger.info(f"✗ Failed to process: {traj_file}")
            except Exception as e:
                logger.info(f"✗ Error processing {traj_file}: {e}")
                failed_trajectories.append(traj_file)

    with open("all_roles.json", "w") as f:
        json.dump(all_roles, f, indent=2)
    with open("all_actions_som.json", "w") as f:
        json.dump(all_actions_som, f, indent=2)
    with open("all_parsed_actions.json", "w") as f:
        json.dump(all_parsed_actions, f, indent=2)

    logger.info(f"\nProcessing complete! Generated {successful_count} HTML files.")
    logger.info(f"Failed to process {len(failed_trajectories)} trajectory files.")
    with open("failed_trajectories.json", "w") as f:
        json.dump(failed_trajectories, f, indent=2)


if __name__ == "__main__":
    main()

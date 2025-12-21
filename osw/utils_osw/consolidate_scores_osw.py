import json
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from osw.utils_osw.utils_osw import get_experiments_dirs, load_trajectory_data

conversation_path_template = "{source_dir}/conversations/{domain}_{task_id}.html"
TRAJ_FILE_PATTERN = "trajectory.json"
DEFAULT_FILENAME = "consolidated_results.csv"


def flatten_nested_dict(d, parent_key="", sep="_"):
    rows = []

    if isinstance(d, dict):
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k

            if isinstance(v, dict):
                # If value has 'score', 'domain', etc., it's a leaf node
                if any(key in v for key in ["score", "domain", "objective"]):
                    row = {"task_id": k}  # k is the task_id in this case
                    row.update(v)
                    rows.append(row)
                else:
                    # Continue recursion
                    rows.extend(flatten_nested_dict(v, new_key, sep))

    return rows


def main(
    base_dir: str = "osw_traces",
    output_path: str = "",
    should_update_excel: bool = False,
    traj_file_pattern: str = TRAJ_FILE_PATTERN,
):
    source_dirs, source_to_files = get_experiments_dirs(base_dir=base_dir, traj_file_pattern=traj_file_pattern)
    consolidated_results = {}

    # For each experiment dir
    for source_dir in source_dirs:
        # Get all trajectory files for the current experiment dir
        traj_files = source_to_files[source_dir]

        # For each trajectory file
        for traj_file in traj_files:
            traj_data = load_trajectory_data(traj_file)
            if "score" not in traj_data:
                print(f"Score not found in {traj_file}. Skipping...")
                continue

            # Get score, task_id, domain, objective
            score = traj_data["score"]
            task_id = traj_data["task_id"]
            domain = traj_data["domain"]
            objective = traj_data["objective"]

            # Create nested dict entries if they don't exist
            consolidated_results.setdefault(source_dir, {})
            consolidated_results[source_dir].setdefault(task_id, {})

            consolidated_results[source_dir][task_id] = {
                "score": score,
                "domain": domain,
                "objective": objective,
                "conversation_path": conversation_path_template.format(source_dir=source_dir, domain=domain, task_id=task_id),
                "trace_path": traj_file,
            }
        # Save a version at source_dir/consolidated_results.csv
        flat_dict = flatten_nested_dict(consolidated_results[source_dir])
        df = pd.DataFrame(flat_dict)
        df.to_csv(f"{source_dir}/consolidated_results.csv", index=False)

        print(f"Saved consolidated results to {source_dir}/consolidated_results.csv")
        mean_score = df["score"].mean()
        num_tasks = len(df)
        # Format decimals to 2 decimal places
        mean_score = round(mean_score * 100, 2)
        print(f"Mean score: {mean_score}, Number of tasks: {num_tasks}")

        scores_per_domain = df.groupby("domain")["score"].mean()
        # Format decimals to 2 decimal places
        scores_per_domain = round(scores_per_domain * 100, 2)
        print(f"Scores per domain: {scores_per_domain}")
    # Convert to csv

    # Flatten the consolidated_results dictionary into a list of rows to create a pandas DataFrame.
    rows = []
    for source_dir, tasks in consolidated_results.items():
        for task_id, data in tasks.items():
            row = {
                "task_id": task_id,
                "domain": data["domain"],
                "objective": data["objective"],
                "score": data["score"],
                "conversation_path": data["conversation_path"],
                "trace_path": data.get("trace_path", ""),
                "html_trajectory_path": str(Path(data.get("trace_path", "")).with_suffix(".html")),
                "source_dir": source_dir,
            }
            rows.append(row)

    # Create a DataFrame from the flattened data.
    df = pd.DataFrame(rows)

    # Specify the output CSV file path.
    if output_path:
        csv_output_path = output_path
    else:
        csv_output_path = f"{base_dir}/{DEFAULT_FILENAME}"
    df.to_csv(csv_output_path, index=False)
    print(f"CSV file has been saved to: {csv_output_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_dir", "-d", type=str, default="osw_traces")
    parser.add_argument("--update_excel", "-ux", action="store_true", help="Update Excel file with results")
    parser.add_argument("--out_path", "-o", type=str, default="")
    args = parser.parse_args()
    main(base_dir=args.base_dir, should_update_excel=args.update_excel, output_path=args.out_path)

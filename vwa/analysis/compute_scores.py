"""Compute aggregated scores from trajectory JSON files discovered under a base directory.

This script scans for files matching `trajectory*.json`, loads them as `VWA_Trajectory` objects,
and aggregates the cumulative reward per (domain, task_id). If multiple trajectories exist for the
same task, the maximum cumulative reward is kept. It then prints per-domain averages and an overall
average. Optionally, it can dump the flattened results to a CSV file.

Usage:
    python -m analysis.compute_scores --base-dir experiments/gemini-2.5-flash/noverifier \
        --out scores.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from trajectory_utils.trajectory import VWA_Trajectory

from core_utils.file_utils import find_files


def collect_scores(base_dir: str) -> dict[str, dict[str, float]]:
    """Return nested mapping: domain -> task_id -> best_score."""
    scores_per_domain: dict[str, dict[str, float]] = {}
    trajectory_files = find_files(base_dir, filename="trajectory*.json")

    for traj_path in trajectory_files:
        traj_source_path = traj_path
        # Load JSON string (kept mainly for validation; we then use the class loader)
        try:
            with open(traj_source_path, "r") as f:
                json_str = f.read()
            _ = json.loads(json_str)  # ensure valid JSON; errors will raise
        except Exception as e:
            print(f"[WARN] Skipping invalid JSON file: {traj_source_path} ({e})")
            continue

        try:
            traj = VWA_Trajectory.from_json(traj_source_path)
        except Exception as e:
            print(f"[WARN] Failed to parse trajectory into object: {traj_source_path} ({e})")
            continue

        domain = getattr(traj, "domain", "") or "unknown_domain"
        task_id = getattr(traj, "task_id", "") or Path(traj_source_path).stem
        score = float(getattr(traj, "cum_reward", float("nan")))

        if domain not in scores_per_domain:
            scores_per_domain[domain] = {}

        # Keep max score if repeated task
        if task_id not in scores_per_domain[domain]:
            scores_per_domain[domain][task_id] = score
        else:
            prev_score = scores_per_domain[domain][task_id]
            scores_per_domain[domain][task_id] = max(prev_score, score)

    return scores_per_domain


def scores_to_dataframe(scores_per_domain: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows = []
    for domain, task_scores in scores_per_domain.items():
        for task_id, score in task_scores.items():
            rows.append({"domain": domain, "task_id": task_id, "score": score})
    if not rows:
        return pd.DataFrame(columns=["domain", "task_id", "score"])
    return pd.DataFrame(rows)


def print_summary(scores_per_domain: dict[str, dict[str, float]]):
    if not scores_per_domain:
        print("No trajectory files found. Nothing to aggregate.")
        return

    # Per-domain average
    str_to_print = ""
    for domain, task_scores in scores_per_domain.items():
        valid_scores = [s for s in task_scores.values() if pd.notnull(s)]
        if not valid_scores:
            avg_score = float("nan")
        else:
            avg_score = sum(valid_scores) / len(valid_scores)
        str_to_print += f"Domain: {domain}: {avg_score:.4f} | "
    print(str_to_print)

    # Overall average
    all_scores = [s for domain_scores in scores_per_domain.values() for s in domain_scores.values() if pd.notnull(s)]
    if all_scores:
        overall_avg = sum(all_scores) / len(all_scores)
        print(f"Overall Average Score: {overall_avg:.4f}")
    else:
        print("Overall Average Score: NaN (no valid scores)")


def main():
    parser = argparse.ArgumentParser(description="Aggregate scores from trajectory JSON files.")
    parser.add_argument(
        "--base-dir",
        type=str,
        default="experiments/gemini-2.5-flash/noverifier",
        help="Base directory to search for trajectory*.json files",
    )
    parser.add_argument("--out", type=str, default="", help="Optional path to write aggregated (domain, task_id, score) CSV")
    args = parser.parse_args()

    scores_per_domain = collect_scores(args.base_dir)
    print_summary(scores_per_domain)

    df = scores_to_dataframe(scores_per_domain)
    if args.out:
        out_path = Path(args.out)
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path, index=False)
            print(f"Wrote aggregated scores to {out_path}")
        except Exception as e:
            print(f"[WARN] Failed to write CSV to {out_path}: {e}")

    # Also print DataFrame-based domain means for cross-check
    if not df.empty:
        mean_scores_df = df.groupby("domain")["score"].mean().reset_index()
        str_to_print = ""
        for _, row in mean_scores_df.iterrows():
            str_to_print += f"Domain: {row['domain']}: {row['score']:.4f} | "
        print(str_to_print)
        overall_mean_score_df = df["score"].mean()
        print(f"(DataFrame) Overall Mean Score: {overall_mean_score_df:.4f}")


if __name__ == "__main__":
    main()

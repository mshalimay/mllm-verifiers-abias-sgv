"""Compute aggregated scores from verification and no-verification results discovered under a base directory.

This script scans for files matching `summary_data.csv` and `scores_per_round.json`, loads them as pandas DataFrames,
and aggregates the scores per domain and task. It then prints per-domain averages and an overall average.
Optionally, it can dump the flattened results to a CSV file.

Obs.: the 'noverifier' stats equals the first score before the verifier intervention.
This is correct if outcome-based verifier (i.e, called after `stop` actions only).

"""

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from core_utils.file_utils import find_files

# fmt:off
args = [
    {"path": "experiments/gemini-2.5-flash/noverifier", "add_traj_json_file": False},    
]
# fmt:on
saved_paths = []


def _print_success_block(label: str, stats: dict[str, Any]) -> None:
    print_with_color(f" {label}", color="magenta", bold=True)
    if "verif" in stats:
        print(f"  Verif: {stats['verif']['success_rate']}, Total Tasks: {stats['verif']['total_tasks']}")
    if "no_verif" in stats:
        print(f"  No Verif: {stats['no_verif']['success_rate']}, Total Tasks: {stats['no_verif']['total_tasks']}")
    if "task_subsets_only" in stats:
        sub = stats["task_subsets_only"]
        if isinstance(sub, dict):
            if "verif" in sub:
                print(f"  Subset Verif: {sub['verif']['success_rate']}, Total Tasks: {sub['verif']['total_tasks']}")
            if "no_verif" in sub:
                print(f"  Subset No Verif: {sub['no_verif']['success_rate']}, Total Tasks: {sub['no_verif']['total_tasks']}")


def print_with_color(text: str, color: str = "reset", bold: bool = False) -> None:
    """Print text with ANSI color and optional bold.

    color: one of {reset, black, red, green, yellow, blue, magenta, cyan, white}
    """
    color_codes = {
        "reset": "",
        "black": "30",
        "red": "31",
        "green": "32",
        "yellow": "33",
        "blue": "34",
        "magenta": "35",
        "cyan": "36",
        "white": "37",
    }
    parts = []
    if bold:
        parts.append("1")
    code = color_codes.get(color.lower(), "")
    if code:
        parts.append(code)
    prefix = f"\033[{';'.join(parts) if parts else '0'}m"
    print(f"{prefix}{text}\033[0m")


def _load_task_ids_from_subset_file(subset_file_path: Path) -> set[str]:
    """Load numeric task IDs (as strings) from a subset file.

    The subset files have the first line as a config path and subsequent lines as numeric IDs.
    Non-numeric lines are ignored.
    """
    task_ids: set[str] = set()
    if not subset_file_path.exists():
        return task_ids

    for line_index, raw_line in enumerate(subset_file_path.read_text().splitlines()):
        line = raw_line.strip()
        if not line:
            continue
        # Skip the first config line or any non-numeric line
        if line_index == 0 and not line.isdigit():
            continue
        if line.isdigit():
            # Normalize to string of int to avoid leading-zero mismatch
            task_ids.add(str(int(line)))
    return task_ids


def _get_domain_task_id_sets() -> dict[str, set[str]]:
    """Return mapping of domain name to the set of task IDs (as strings) for that domain."""
    base_dir = Path(__file__).resolve().parent / "evaluation_harness" / "task_subsets"

    classifieds_ids = _load_task_ids_from_subset_file(base_dir / "classifieds.txt")
    shopping_ids = _load_task_ids_from_subset_file(base_dir / "shopping.txt")
    reddit_ids = _load_task_ids_from_subset_file(base_dir / "reddit.txt")

    return {
        "classifieds": classifieds_ids,
        "shopping": shopping_ids,
        "reddit": reddit_ids,
    }


def _infer_domain_from_path(results_dir: str) -> Optional[str]:
    """Infer domain name from the provided results directory path.

    Returns one of {"classifieds", "shopping", "reddit"} when detected, otherwise None.
    """
    path = Path(results_dir)
    parts = {p.lower() for p in path.parts}
    for candidate in ("classifieds", "shopping", "reddit"):
        if candidate in parts:
            return candidate
    # Fallback to substring search on the full string
    lower_str = str(path).lower()
    for candidate in ("classifieds", "shopping", "reddit"):
        if f"/{candidate}" in lower_str or lower_str.endswith(candidate):
            return candidate
    return None


def infer_domain_from_task_id(task_id: str) -> str | None:
    if "classifieds" in task_id:
        return "classifieds"
    elif "shopping" in task_id:
        return "shopping"
    elif "reddit" in task_id:
        return "reddit"
    else:
        return None


def regularize_task_id(task_id: str) -> str:
    # Remove repeated domain prefix
    # e.g.: reddit_reddit_177 -> reddit_177
    domain = infer_domain_from_task_id(task_id)
    if domain is None:
        raise ValueError(f"Domain not found for {task_id}")
    x = re.sub(f"{domain}", "", task_id)
    x = re.sub(f"_", "", x)
    x = f"{domain}_{x}"
    return x


def regularize_scores_per_round(scores_per_round_path: str | Path) -> dict[str, dict[str, dict[str, float]]]:
    with open(scores_per_round_path, "r") as file:
        data = json.load(file)
    new_data = {}

    # Add attempt_num to the data if not present
    for task_id, task_data in list(data.items()):
        if "scores" in task_data:
            new_task_data = {}
            new_task_data["0"] = task_data
            new_data[task_id] = new_task_data
        else:
            new_data[task_id] = task_data

    for task_id, task_data in list(new_data.items()):
        domain_val: Optional[str] = None
        for attempt_num, attempt_data in task_data.items():
            if "domain" not in attempt_data:
                parent_dir = Path(scores_per_round_path).parent
                domain = infer_domain_from_task_id(task_id)
                if domain is None:
                    domain = _infer_domain_from_path(str(parent_dir))
                if domain is None:
                    raise ValueError(f"Domain not found for {parent_dir}")
                attempt_data["domain"] = domain
            # Track any domain we see
            domain_val = attempt_data.get("domain", domain_val)

        if domain_val is None:
            # Fallbacks if no attempts present
            domain_val = infer_domain_from_task_id(task_id) or _infer_domain_from_path(str(Path(scores_per_round_path).parent))
        if domain_val is not None:
            new_key = regularize_task_id(f"{domain_val}_{task_id}")
            new_data[new_key] = task_data

    with open(scores_per_round_path, "w") as file:
        json.dump(new_data, file, indent=2)
    return new_data


def find_traj_json_file(parent_dir: str, task_id: str, domain: str) -> str:
    traj_files = find_files(
        parent_dir,
        f"**/*trajectory-{domain}-{task_id}.json",
        upwards=False,
        downwards=True,
    )
    if not traj_files:
        # Try to get via domain dir
        match = re.search(r"^(.*/(?:shopping|reddit|classifieds)/)", parent_dir)
        if not match:
            return ""
        domain_dir_path = Path(match.group(0).rstrip("/"))
        if domain_dir_path.exists():
            traj_files = find_files(
                str(domain_dir_path),
                f"**/*trajectory-{domain}-{task_id}.json",
                upwards=False,
                downwards=True,
            )
            # If didn't find, try old format:
            if not traj_files:
                traj_files = find_files(
                    str(domain_dir_path),
                    f"**/*render_{task_id}.html",
                    upwards=False,
                    downwards=True,
                )
    if traj_files:
        return traj_files[0]
    return ""


def get_critique_round_scores(
    json_file_path: str,
    round_idx: int = 0,
    attempt_num: str = "0",
) -> dict[int, dict[str, float]]:
    """Given a JSON with scores for the `critic-executor` loops per task ID, return all scores for the `round_idx` round of the loops.
    Args:
        json_file_path (str): the path to the JSON file formatted as {task_id:scores:[{state_idx, score, round}, {state_idx, score, round}, ...]}
        round_idx (int, optional): The i'th round of a critic-agent loop for a single state.

    Returns:
        dict[int, dict[str, float]]: {task_id: {score: score}}

    # NOTE: always return the scores for the first call to the critic.
    """

    # Read the JSON file
    with open(json_file_path, "r") as file:
        data = json.load(file)

    scores = {}

    # Iterate through each task and get the score observed for the `round_idx` round of the first call to the critic.
    for task_id, task_data in data.items():
        if "scores" in task_data:
            for score_entry in task_data["scores"]:
                if score_entry["round"] == round_idx:
                    scores[task_id] = {
                        "score": score_entry["score"],
                    }
                    break
        elif attempt_num in task_data:
            task_attempt_data = task_data[attempt_num]
            if "scores" in task_attempt_data:
                for score_entry in task_attempt_data["scores"]:
                    if score_entry["round"] == round_idx:
                        scores[task_id] = {
                            "score": score_entry["score"],
                        }
                        break
    return scores


def compute_success_rate(dir: str, add_traj_json_file: bool = False) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Compute success rate and counts from summary_data.csv files under a results directory.
    Returns a tuple of (stats_dict, dataframe).
    """
    dir = str(Path(dir))
    args_files = find_files(dir, "args.json", upwards=False, downwards=True)
    df = pd.DataFrame()
    df_no_verif = pd.DataFrame()
    aggregated_round_scores = {}

    for args_file in args_files:
        parent_dir = Path(args_file).parent
        summary_data_path = parent_dir / "summary_data.csv"
        scores_per_round_path = parent_dir / "scores_per_round.json"

        if summary_data_path.exists():
            part_df = pd.read_csv(summary_data_path)
            # Tag with domain and normalize task_id early to avoid cross-domain collisions
            if "domain" not in part_df.columns:
                domain = _infer_domain_from_path(str(parent_dir))
                if domain is None:
                    raise ValueError(f"Domain not found for {parent_dir}")
                part_df["domain"] = domain

            # Create unique domain_task_id key
            part_df["domain_task_id"] = part_df["domain"].astype(str) + "_" + part_df["task_id"].astype(str)
            if add_traj_json_file:
                # Ensure the column exists and uses a string-compatible dtype to avoid dtype-incompatible assignment warnings
                if "traj_json_file" not in part_df.columns:
                    part_df["traj_json_file"] = pd.Series(pd.NA, index=part_df.index, dtype="string")
                else:
                    # Coerce to pandas nullable string dtype for consistent handling of missing values and strings
                    part_df["traj_json_file"] = part_df["traj_json_file"].astype("string")

                for row in part_df.itertuples():
                    if pd.notna(row.traj_json_file):
                        traj_json_file = Path(str(row.traj_json_file))
                        if traj_json_file.exists():
                            continue
                    traj_json_file = find_traj_json_file(str(parent_dir), str(row.task_id), str(row.domain))
                    if traj_json_file:
                        # Add traj_json_file to part_df
                        part_df.loc[row.Index, "traj_json_file"] = str(traj_json_file)
            part_df.to_csv(summary_data_path, index=False)

            df = pd.concat([df, part_df])

            scores_no_verif = part_df.copy()

            scores_no_verif.rename(columns={"score": "no_verif"}, inplace=True)

            scores_no_verif["domain_task_id"] = scores_no_verif["domain"].astype(str) + "_" + scores_no_verif["task_id"].astype(str)

            if scores_per_round_path.exists():
                new_data = regularize_scores_per_round(scores_per_round_path)
                round_scores_data = get_critique_round_scores(str(scores_per_round_path), round_idx=0, attempt_num="0")
                aggregated_round_scores.update(new_data)
                round_scores_df = pd.DataFrame.from_dict(round_scores_data, orient="index")
                if not round_scores_df.empty:
                    round_scores_df.reset_index(inplace=True)
                    round_scores_df.rename(columns={"index": "domain_task_id", "score": "no_verif"}, inplace=True)

                    # Align on unified key
                    scores_no_verif = scores_no_verif.set_index(["domain_task_id"]).sort_index()
                    round_scores_df = round_scores_df.set_index(["domain_task_id"]).sort_index()
                    scores_no_verif.update(round_scores_df[["no_verif"]])
                    scores_no_verif.reset_index(inplace=True)

            df_no_verif = pd.concat([df_no_verif, scores_no_verif])

    # Keep only valid rows
    df = df.dropna(subset=["score", "domain_task_id"]) if not df.empty else df
    if df.empty:
        return {}, pd.DataFrame()

    json.dump(aggregated_round_scores, open(f"{dir}/aggregated_round_scores.json", "w"), indent=2)
    # Rename traj_json_file to traj_source_path for clarity
    if "traj_json_file" in df.columns:
        df.rename(columns={"traj_json_file": "traj_source_path"}, inplace=True)
    df.to_csv(f"{dir}/df_verif.csv")
    print(f"Wrote {len(df)} verif scores to {dir}/df_verif.csv")
    df_no_verif.to_csv(f"{dir}/df_no_verif.csv")
    print(f"Wrote {len(df_no_verif)} no_verif scores to {dir}/df_no_verif.csv")

    global saved_paths
    if len(df) > 300:
        saved_paths.append(f"{dir}/df_verif.csv")

    # If duplicate domain_task_id, pick the max score (unified key means global uniqueness)
    df = (
        df.sort_values("score", ascending=False)
        .groupby("domain_task_id", as_index=False)
        .agg(
            {
                "score": "max",
                "domain": "first",
                "task_id": "first",
            }
        )
    )

    # Join with no_verif scores if available
    if not df_no_verif.empty:
        df_no_verif = df_no_verif.dropna(subset=["no_verif", "domain_task_id"]).copy()
        df_no_verif = df_no_verif.groupby("domain_task_id", as_index=False).agg({"no_verif": "max"})
        df = df.merge(df_no_verif[["domain_task_id", "no_verif"]], on="domain_task_id", how="left")

    df["success"] = df["score"] >= 1
    success_rate = df["success"].mean() * 100
    num_tasks = len(df["success"])

    crit_stats = {
        "success_rate": f"{success_rate:.3f}%",
        "total_tasks": num_tasks,
    }

    all_stats = {"verif": crit_stats}

    if not df_no_verif.empty and "no_verif" in df.columns:
        df_no_crit_filtered = df.dropna(subset=["no_verif"]).copy()
        if not df_no_crit_filtered.empty:
            df_no_crit_filtered.loc[:, "success_no_crit"] = df_no_crit_filtered["no_verif"] >= 1
            success_rate_no_crit = df_no_crit_filtered["success_no_crit"].mean() * 100
            num_tasks_no_crit = len(df_no_crit_filtered["success_no_crit"])

            no_crit_stats = {
                "success_rate": f"{success_rate_no_crit:.3f}%",
                "total_tasks": num_tasks_no_crit,
            }
            all_stats["no_verif"] = no_crit_stats

    # Compute and print per-domain stats
    def _compute_metrics(df_slice: pd.DataFrame) -> dict[str, Any]:
        if df_slice.empty:
            return {}
        local = df_slice.dropna(subset=["score", "domain_task_id"]).copy()
        # Ensure one row per unified domain_task_id
        agg_dict: dict[str, str] = {"score": "max"}
        if "no_verif" in local.columns:
            agg_dict["no_verif"] = "max"
        local = local.groupby("domain_task_id", as_index=False).agg(agg_dict)
        local["success"] = local["score"] >= 1
        out: dict[str, Any] = {
            "verif": {
                "success_rate": f"{(local['success'].mean() * 100):.3f}%",
                "total_tasks": len(local),
            }
        }
        if "no_verif" in local.columns:
            nc = local.dropna(subset=["no_verif"]).copy()
            if not nc.empty:
                nc.loc[:, "success_no_crit"] = nc["no_verif"] >= 1
                out["no_verif"] = {
                    "success_rate": f"{(nc['success_no_crit'].mean() * 100):.3f}%",
                    "total_tasks": len(nc),
                }
        return out

    # Per-domain metrics from single canonical dataframe
    per_domain_stats: dict[str, dict[str, Any]] = {}
    if "domain" in df.columns:
        for d in sorted(df["domain"].dropna().unique()):
            stats_d = _compute_metrics(df[df["domain"] == d])
            if stats_d:
                per_domain_stats[str(d)] = stats_d

    # Subset-only success rates (per domain and overall) using domain + task_id (raw id)
    domain_task_sets = _get_domain_task_id_sets()

    # Per-domain subset-only
    for d, stats_d in list(per_domain_stats.items()):
        ids = domain_task_sets.get(d, set())
        if ids:
            mask = (df["domain"] == d) & (df["task_id"].astype(str).isin(ids))
            stats_subset = _compute_metrics(df[mask])
            if stats_subset:
                stats_d["task_subsets_only"] = {}
                if "verif" in stats_subset:
                    stats_d["task_subsets_only"]["verif"] = stats_subset["verif"]
                if "no_verif" in stats_subset:
                    stats_d["task_subsets_only"]["no_verif"] = stats_subset["no_verif"]

    # Overall subset-only across domains (union by domain + task_id)
    mask_overall = pd.Series(False, index=df.index)
    for d, ids in domain_task_sets.items():
        if ids:
            mask_overall = mask_overall | ((df["domain"] == d) & (df["task_id"].astype(str).isin(ids)))
    overall_subset_stats = _compute_metrics(df[mask_overall])

    # Print per-domain then overall for this dir
    for d in ("classifieds", "shopping", "reddit"):
        if d in per_domain_stats:
            _print_success_block(d.upper(), per_domain_stats[d])
    if overall_subset_stats:
        all_stats["task_subsets_only"] = {}
        if "verif" in overall_subset_stats:
            all_stats["task_subsets_only"]["verif"] = overall_subset_stats["verif"]
        if "no_verif" in overall_subset_stats:
            all_stats["task_subsets_only"]["no_verif"] = overall_subset_stats["no_verif"]
    # Include per-domain stats for downstream consolidation
    all_stats["per_domain"] = per_domain_stats
    _print_success_block(dir, all_stats)
    return all_stats, df


def _run_one_arg(arg: dict[str, Any]) -> tuple[str, dict[str, Any], pd.DataFrame] | None:
    path = arg.get("path")
    add_traj_json_file = bool(arg.get("add_traj_json_file", False))
    if not path:
        return None
    print_with_color(f"==== Experiment: {path} ========", color="cyan", bold=True)
    stats, df = compute_success_rate(path, add_traj_json_file)
    return (path, stats, df)


collected_results: list[tuple[str, dict[str, Any], pd.DataFrame]] = []
max_workers = min(8, len(args)) if args else 1
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(_run_one_arg, arg) for arg in args if arg.get("path")]
    for fut in as_completed(futures):
        res = fut.result()
        if res is not None:
            collected_results.append(res)

# Consolidated per-path success rates
print("\n OVERALL")
for path, stats, _ in collected_results:
    _print_success_block(path, stats)


# Consolidated csv with all the results
consolidated_rows = []
for path, _, df in collected_results:
    if df.empty:
        continue
    # Select relevant columns and add experiment path
    task_df = df[["task_id", "domain", "score"]].copy()
    task_df["experiment_path"] = path
    consolidated_rows.append(task_df)

if consolidated_rows:
    consolidated_df = pd.concat(consolidated_rows, ignore_index=True)
    # Create concatenated column
    consolidated_df["domain_id_experiment"] = (
        consolidated_df["domain"] + consolidated_df["task_id"].astype(str) + consolidated_df["experiment_path"]
    )
    # Reorder columns to match requested format
    consolidated_df = consolidated_df[["experiment_path", "task_id", "domain", "score", "domain_id_experiment"]]
    output_path = "consolidated_results.csv"
    consolidated_df.to_csv(output_path, index=False)
    print_with_color(f"\nWrote consolidated results to {output_path}", color="green", bold=True)
    print(f"Total rows: {len(consolidated_df)}")

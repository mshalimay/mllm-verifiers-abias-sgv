"""
Compute per-model, per-config statistics broken down by trajectory length ranges
using offline_experiments/all_evals_consolidated.csv.
Output: offline_experiments/trajectory_len_stats.csv
"""

from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

from offline_experiments.analysis.eval_mapping import (
    extract_eval_template_from_variation_name,
    map_eval_to_score_with_context,
)

MAX_TRAJ_LEN = 30
ALL_EVALS_PATH = Path("offline_experiments") / "results" / "all_evals_consolidated.csv"
OUTPUT_PATH = Path("offline_experiments") / "results" / "trajectory_len_stats.csv"


def _compute_distance_skewness(X: Iterable[float], theta: float) -> float:
    X = np.array(list(X), dtype=float)
    if X.size == 0:
        return np.nan
    # pairwise distances between the elements of X
    pairwise_distances = np.abs(np.subtract.outer(X, X))

    # numerator and denominator of the distance skewness formula
    numerator = np.sum(pairwise_distances)
    denominator = np.sum(np.abs(np.add.outer(X, X) - 2 * theta))

    # handle the case when Pr(X=theta) = 1
    if denominator == 0:
        return 0.0
    else:
        return 1 - numerator / denominator


def _ranges(max_len: int = MAX_TRAJ_LEN, step: int = 5) -> List[Tuple[int, int | None]]:
    """Generate ranges as requested.

    Starts: 1, 5, 10, ..., max
    Ends per start: multiples of `step` >= start, and an open-ended 'max' (None).

    Example for max=100, step=5:
      (1, 5), (1, 10), ..., (1, None)
      (5, 10), (5, 15), ..., (5, None)
      ...
    """
    multiples = [i for i in range(step, max_len + step, step)]  # 5,10,...,max
    starts = [1] + [m for m in multiples if m >= step]

    result: List[Tuple[int, int | None]] = []
    for s in starts:
        ends = [m for m in multiples if m >= s]
        # numeric ends
        for e in ends:
            result.append((s, e))
        # 'max' end as None (interpreted as >= s)
        result.append((s, None))
    return result


def _prepare_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Ensure numeric types
    if "traj_len" in df.columns:
        df["traj_len"] = pd.to_numeric(df["traj_len"], errors="coerce")

    # Ensure gold_score numeric if present
    if "gold_score" in df.columns:
        df["gold_score"] = pd.to_numeric(df["gold_score"], errors="coerce")

    # Ensure predicted_score present; if not, compute from eval and config_name
    if "predicted_score" not in df.columns and "eval" in df.columns:

        def _map_row(row):
            template = extract_eval_template_from_variation_name(row.get("config_name", ""))
            return map_eval_to_score_with_context(row.get("eval", ""), template)

        df["predicted_score"] = pd.to_numeric([_map_row(r) for _, r in df.iterrows()], errors="coerce")

    return df


def _compute_metrics(sub: pd.DataFrame) -> dict:
    # Keep rows with available gold and predicted scores
    used = sub.dropna(subset=["gold_score", "predicted_score"]).copy()
    n = int(len(used))
    if n == 0:
        return {
            "tpr": np.nan,
            "tnr": np.nan,
            "acc": np.nan,
            "bias": np.nan,
            "dskew": np.nan,
            "success_rate": np.nan,
            "n": 0,
        }

    # Confusion counts
    fp = int(((used["predicted_score"] == 1) & (used["gold_score"] == 0)).sum())
    fn = int(((used["predicted_score"] == 0) & (used["gold_score"] == 1)).sum())
    tp = int(((used["predicted_score"] == 1) & (used["gold_score"] == 1)).sum())
    tn = int(((used["predicted_score"] == 0) & (used["gold_score"] == 0)).sum())

    pos = tp + fn
    neg = tn + fp

    tpr = (tp / pos) if pos > 0 else np.nan
    tnr = (tn / neg) if neg > 0 else np.nan
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else np.nan

    # Bias and distance skewness on prediction error
    diff = (used["predicted_score"] - used["gold_score"]).astype(float)
    bias = float(diff.mean()) if not diff.empty else np.nan
    dskew = float(_compute_distance_skewness(diff, 0)) if not diff.empty else np.nan

    # Success rate (predicted)
    success_rate = float(used["predicted_score"].mean()) if not used.empty else np.nan

    # Mirror compute_stats scaling (percentages x100; bias and dskew x100)
    return {
        "tpr": round(tpr * 100, 1) if pd.notna(tpr) else np.nan,
        "tnr": round(tnr * 100, 1) if pd.notna(tnr) else np.nan,
        "acc": round(acc * 100, 1) if pd.notna(acc) else np.nan,
        "bias": round(bias * 100, 2) if pd.notna(bias) else np.nan,
        "dskew": round(dskew * 100, 2) if pd.notna(dskew) else np.nan,
        "success_rate": round(success_rate * 100, 1) if pd.notna(success_rate) else np.nan,
        "n": n,
    }


def main(max_traj_len: int = MAX_TRAJ_LEN, step: int = 5) -> None:
    if not ALL_EVALS_PATH.exists():
        raise FileNotFoundError(f"Input not found: {ALL_EVALS_PATH}")

    df = pd.read_csv(ALL_EVALS_PATH)
    df = _prepare_scores(df)

    if "traj_len" not in df.columns:
        raise ValueError("Column 'traj_len' is not present in all_evals_consolidated.csv. Re-run compute_stats to include it.")

    # Only keep rows with a trajectory length
    df = df.dropna(subset=["traj_len"]).copy()
    df["traj_len"] = pd.to_numeric(df["traj_len"], errors="coerce")
    df = df.dropna(subset=["traj_len"]).copy()

    # Ensure needed columns exist
    required_cols = ["model", "config_name", "gold_score", "predicted_score", "traj_len"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    results = []
    ranges = _ranges(max_len=max_traj_len, step=step)

    # Group by model configuration
    for (model, config), g in df.groupby(["model", "config_name"], dropna=False):
        for start, end in ranges:
            if end is None:
                sub = g[g["traj_len"] >= start]
                range_label = f"{start}-max"
            else:
                sub = g[(g["traj_len"] >= start) & (g["traj_len"] <= end)]
                range_label = f"{start}-{end}"

            if sub.empty:
                continue

            metrics = _compute_metrics(sub)
            row = {
                "model": model,
                "config_name": config,
                "range": range_label,
                **metrics,
            }
            results.append(row)

    out_df = pd.DataFrame(results)
    out_df = out_df.sort_values(by=["model", "config_name", "range"]).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved trajectory length stats to {OUTPUT_PATH} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from offline_experiments.analysis.eval_mapping import (
    extract_eval_template_from_variation_name,
    map_response_to_category,
)

# Import the modular calibrator
from offline_experiments.calibration.calibration import Calibrator, PlattParams

allowed_methods = {"platt", "isotonic", "histogram", "logistic", "prior"}

# =============================
# Helpers: scoring and splits
# =============================


def _safe_float(x: str | float | int) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def response_to_raw_score(eval_text: str, eval_template: Optional[str]) -> Optional[float]:
    """
    Map an eval response to a raw continuous score in [0,1].
    - Numeric 1-10 -> divide by 10
    - Numeric 0/50/75/100 -> divide by 100
    - Numeric 0-1 -> keep as is
    - Categorical -> SUCCESS=1, PARTIAL SUCCESS=0.5, PARTIAL FAILURE=0.25,
                     FAILURE=0, UNCERTAIN=None
    """
    if eval_text is None or (isinstance(eval_text, float) and np.isnan(eval_text)):
        return None

    s = str(eval_text).strip().upper()
    if s == "":
        return None

    if s == "INFEASIBLE":
        return 1.0

    # Numeric-like: accept integers/floats as strings possibly with sign/decimal
    num_like = s.replace(".", "").replace("-", "").isdigit()
    if num_like:
        val = _safe_float(s)
        if val is None:
            return None
        if eval_template is None:
            raise ValueError("eval_template must be provided for numeric responses")

        if eval_template in {"four_room_score", "four_room_score_2", "range_num_likelihood_0_100", "range_num_likelihood_0_100_2"}:
            # Expect 0/50/75/100
            return float(val) / 100.0

        elif eval_template in {"range_num_0_1", "range_num_confidence_0_1"}:
            return float(val)

        elif "s_prob" in eval_template:
            return float(val)

        elif eval_template in {"range_num", "range_num_vague"} or any(_id in eval_template for _id in ["tri_num", "four_num", "four_degree", "bin_num"]):
            return float(val) / 10.0
        else:
            raise ValueError(f"Unexpected numeric eval_template for numeric response: {eval_template}")

    # Categorical: map via centralized category mapping first
    category = map_response_to_category(s, eval_template)
    if category is None:
        return None
    if category == "SUCCESS":
        return 1.0
    if category == "PARTIAL SUCCESS":
        return 0.5
    if category == "PARTIAL FAILURE":
        return 0.25
    if category == "FAILURE":
        return 0.0
    # UNCERTAIN and others -> None
    return None


def _frac_to_token(frac: float) -> str:
    """Convert a fraction to a filename-safe token, e.g., 0.333333 -> '0p333333'."""
    s = f"{frac:.6f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def build_random_train_by_env(
    df: pd.DataFrame,
    seed: int = 42,
    frac: float = 1.0 / 3.0,
    split_label: Optional[str] = None,
    min_neg_frac: Optional[float] = None,
    max_neg_frac: Optional[float] = None,
    target_neg_frac: Optional[float] = None,
    neg_frac_tol: float = 0.0,
    max_tries: int = 200,
) -> pd.DataFrame:
    """
    Return a DataFrame with columns [domain_task_id, env, split, train].
    The split label is either provided via split_label or auto-generated as
    'random_train_{token}_seed{seed}', where token encodes the training fraction.
    Here, train==1 indicates TRAIN (random fraction), and 0 indicates TEST (remaining examples).
    """
    rng = np.random.default_rng(seed)
    rows: List[Dict] = []

    if split_label is not None:
        split_name = split_label
    else:
        # Construct informative default split name
        parts = [f"random_train_{_frac_to_token(frac)}"]
        if target_neg_frac is not None:
            pos_frac = 1.0 - target_neg_frac
            parts.append(f"pos{int(round(pos_frac * 100))}")
            parts.append(f"neg{int(round(target_neg_frac * 100))}")
            if neg_frac_tol > 0:
                parts.append(f"tol{_frac_to_token(neg_frac_tol)}")
        else:
            if min_neg_frac is not None:
                parts.append(f"minneg_{_frac_to_token(min_neg_frac)}")
            if max_neg_frac is not None:
                parts.append(f"maxneg_{_frac_to_token(max_neg_frac)}")
        parts.append(f"seed{seed}")
        split_name = "_".join(parts)

    group_keys = ["env"] + (["gold_source"] if "gold_source" in df.columns else [])

    for gvals, sub_df in df.groupby(group_keys, as_index=False):
        if isinstance(gvals, tuple):
            env = gvals[0]
            gold_src = gvals[1] if len(gvals) > 1 else None
        else:
            env = gvals
            gold_src = None

        env_ids = sub_df["domain_task_id"].unique()
        if env_ids.size == 0:
            continue
        k = max(1, int(np.floor(len(env_ids) * frac)))

        vc = sub_df.groupby(["domain_task_id", "gold_score"]).size().unstack(fill_value=0)
        n0 = vc[0] if 0 in vc.columns else pd.Series(0, index=vc.index)
        n1 = vc[1] if 1 in vc.columns else pd.Series(0, index=vc.index)
        idx = vc.index.values
        n0_arr = n0.reindex(idx).to_numpy()
        n1_arr = n1.reindex(idx).to_numpy()
        total_arr = n0_arr + n1_arr
        total_arr = np.where(total_arr == 0, 1, total_arr)

        if min_neg_frac is None and max_neg_frac is None and target_neg_frac is None:
            choose_ids = set(rng.choice(idx, size=min(k, len(idx)), replace=False))
        else:
            density = n0_arr / total_arr
            if target_neg_frac is not None:
                # Aim for a specific negative fraction with optional tolerance
                mid = target_neg_frac
                w = 1.0 - np.abs(density - mid)
                w = np.clip(w, 1e-3, None)
                probs = w / w.sum()
            elif min_neg_frac is not None and max_neg_frac is not None:
                mid = 0.5 * (min_neg_frac + max_neg_frac)
                w = 1.0 - np.abs(density - mid)
                w = np.clip(w, 1e-3, None)
                probs = w / w.sum()
            elif min_neg_frac is not None:
                probs = (n0_arr + 1.0) / (total_arr + 2.0)
                probs = probs / probs.sum()
            else:
                probs = (n1_arr + 1.0) / (total_arr + 2.0)
                probs = probs / probs.sum()

            choose_ids = None
            best_ids = None
            best_penalty = float("inf")
            best_frac = None

            for _ in range(max_tries):
                sample = rng.choice(idx, size=min(k, len(idx)), replace=False, p=probs)
                sel = np.isin(idx, sample)
                sel_n0 = n0_arr[sel].sum()
                sel_n1 = n1_arr[sel].sum()
                denom = sel_n0 + sel_n1
                frac_neg = (sel_n0 / denom) if denom > 0 else 0.0

                ok_min = (min_neg_frac is None) or (frac_neg >= min_neg_frac)
                ok_max = (max_neg_frac is None) or (frac_neg <= max_neg_frac)
                ok_target = True
                if target_neg_frac is not None:
                    ok_target = abs(frac_neg - target_neg_frac) <= neg_frac_tol
                if ok_min and ok_max and ok_target:
                    choose_ids = set(sample.tolist())
                    break

                penalty = 0.0
                if (min_neg_frac is not None) and (frac_neg < min_neg_frac):
                    penalty += min_neg_frac - frac_neg
                if (max_neg_frac is not None) and (frac_neg > max_neg_frac):
                    penalty += frac_neg - max_neg_frac
                if penalty < best_penalty:
                    best_penalty = penalty
                    best_ids = set(sample.tolist())
                    best_frac = frac_neg

            if choose_ids is None:
                if best_ids is not None and best_penalty < float("inf"):
                    if (min_neg_frac is not None) and (best_frac is not None) and (best_frac < min_neg_frac):
                        order = np.argsort(-density)
                        greedy = set(idx[order[: min(k, len(idx))]].tolist())
                        choose_ids = greedy
                    elif (max_neg_frac is not None) and (best_frac is not None) and (best_frac > max_neg_frac):
                        order = np.argsort(density)
                        greedy = set(idx[order[: min(k, len(idx))]].tolist())
                        choose_ids = greedy
                    else:
                        choose_ids = best_ids
                else:
                    if target_neg_frac is not None:
                        mid = target_neg_frac
                        order = np.argsort(np.abs(density - mid))
                    elif min_neg_frac is not None and max_neg_frac is not None:
                        mid = 0.5 * (min_neg_frac + max_neg_frac)
                        order = np.argsort(np.abs(density - mid))
                    elif min_neg_frac is not None:
                        order = np.argsort(-density)
                    else:
                        order = np.argsort(density)
                    choose_ids = set(idx[order[: min(k, len(idx))]].tolist())

        choose_ids = choose_ids or set()
        for dtid in env_ids:
            rows.append(
                {
                    "domain_task_id": dtid,
                    "env": env,
                    **({"gold_source": gold_src} if gold_src is not None else {}),
                    "split": split_name,
                    "train": 1 if dtid in choose_ids else 0,
                    "origin": "random",
                }
            )

    base_cols = ["domain_task_id", "env", "split", "train", "origin"]
    if "gold_source" in df.columns:
        base_cols.insert(2, "gold_source")
    return pd.DataFrame(rows)[base_cols]


def load_train_sets(path: Path | str) -> pd.DataFrame:
    """
    Load provided TRAIN-sets file with columns
    [domain_task_id, env, <split1>, <split2>, ...].
    Returns standardized long-form DataFrame:
    [domain_task_id, env, split, train].
    """
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["domain_task_id", "env", "split", "train", "origin"])
    df = pd.read_csv(p)
    if not {"domain_task_id", "env"}.issubset(set(df.columns)):
        cols_lower = {c.lower(): c for c in df.columns}
        if "domain_task_id" in cols_lower and "env" in cols_lower:
            df = df.rename(
                columns={
                    cols_lower["domain_task_id"]: "domain_task_id",
                    cols_lower["env"]: "env",
                }
            )
        else:
            return pd.DataFrame(columns=["domain_task_id", "env", "split", "train", "origin"])

    split_cols = [c for c in df.columns if c not in {"domain_task_id", "env"}]
    if not split_cols:
        return pd.DataFrame(columns=["domain_task_id", "env", "split", "train", "origin"])

    long_df = df.melt(
        id_vars=["domain_task_id", "env"],
        value_vars=split_cols,
        var_name="split",
        value_name="train",
    )
    long_df["train"] = long_df["train"].fillna(0).astype(int)
    long_df["origin"] = "provided"
    return long_df[["domain_task_id", "env", "split", "train", "origin"]]


# =============================
# Metrics
# =============================


def _log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(((y - p) ** 2).mean())


def _compute_distance_skewness(X: np.ndarray | List[float], theta: float) -> float:
    X = np.array(X, dtype=float)
    if X.size == 0:
        return float("nan")
    pairwise_distances = np.abs(np.subtract.outer(X, X))
    numerator = np.sum(pairwise_distances)
    denominator = np.sum(np.abs(np.add.outer(X, X) - 2 * theta))
    if denominator == 0:
        return 0.0
    return float(1 - numerator / denominator)


def _threshold(p: np.ndarray, thr: float = 0.5) -> np.ndarray:
    return (p >= thr).astype(int)


def _metrics_from_probs(y_true: np.ndarray, p: np.ndarray, thr: float = 0.5) -> dict:
    y_true = y_true.astype(float)
    p = p.astype(float)
    mask = np.isfinite(p) & np.isfinite(y_true)
    if mask.sum() == 0:
        return {
            "tpr": float("nan"),
            "tnr": float("nan"),
            "accuracy": float("nan"),
            "bias": float("nan"),
            "dskew": float("nan"),
        }
    y_true_m = y_true[mask].astype(int)
    p_m = p[mask]
    y_hat = _threshold(p_m, thr)

    tp = int(((y_hat == 1) & (y_true_m == 1)).sum())
    tn = int(((y_hat == 0) & (y_true_m == 0)).sum())
    fp = int(((y_hat == 1) & (y_true_m == 0)).sum())
    fn = int(((y_hat == 0) & (y_true_m == 1)).sum())

    tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    tnr = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else float("nan")

    diff = y_hat - y_true_m
    bias = float(np.mean(diff)) if diff.size > 0 else float("nan")
    dskew = _compute_distance_skewness(diff, 0)

    return {
        "tpr": tpr,
        "tnr": tnr,
        "accuracy": acc,
        "bias": bias,
        "dskew": dskew,
    }


def _normalize_prob_array(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Ensure probabilities are in (0,1). If values look like 0..100, scale down.
    Clips to [eps, 1-eps].
    """
    if p.size == 0:
        return p
    p = p.astype(float)
    finite = np.isfinite(p)
    if finite.any():
        maxv = float(np.nanmax(p[finite]))
        if maxv > 1.0 + 1e-9:
            p = p / 100.0
    return np.clip(p, eps, 1 - eps)


# =============================
# Main calibration routine
# =============================


def compute_calibration(
    csv_path: str | Path,
    train_path: Optional[str | Path] = None,
    out_dir: str | Path = "./offline_experiments",
    seed: int = 42,
    max_groups: Optional[int] = None,
    random_fracs: Optional[List[float]] = None,
    neg_fracs: Optional[List[float]] = None,
    pos_fracs: Optional[List[float]] = None,
    neg_frac_tol: float = 0.05,
    random_min_neg_frac: Optional[float] = 0.4,
    random_max_neg_frac: Optional[float] = 0.5,
    random_max_tries: int = 200,
    calibration_methods: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main entry point. Behavior is identical to the original Platt script
    except that calibration is modular and multiple methods can be used.

    calibration_methods: list of methods, e.g. ["platt", "isotonic", "histogram"]
    """
    if calibration_methods is None or len(calibration_methods) == 0:
        calibration_methods = ["platt"]

    # Normalize methods list
    calibration_methods = [m for m in calibration_methods if m in allowed_methods]
    if not calibration_methods:
        raise ValueError("No valid calibration methods specified.")

    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load consolidated evals
    df = pd.read_csv(csv_path)

    required = {
        "domain_task_id",
        "config_name",
        "eval",
        "model",
        "env",
        "gold_score",
    }
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    # Derive eval_template and raw_score
    df["eval_template"] = df["config_name"].map(lambda v: extract_eval_template_from_variation_name(str(v)))
    df["raw_score"] = [response_to_raw_score(ev, tpl) for ev, tpl in zip(df["eval"], df["eval_template"])]

    # Keep only rows with valid gold and raw score
    df = df[df["gold_score"].isin([0, 1])].copy()
    df = df[df["raw_score"].notna()].copy()

    # Provided training sets (wide -> long)
    provided_long = load_train_sets(train_path) if train_path else pd.DataFrame(columns=["domain_task_id", "env", "split", "train", "origin"])

    # Random TRAIN splits per env (only if random_fracs provided)
    if random_fracs is None or len(random_fracs) == 0:
        random_long = pd.DataFrame(columns=["domain_task_id", "env", "split", "train", "origin"])
    else:
        random_long_list: List[pd.DataFrame] = []
        for f in random_fracs:
            # Unbiased split at this fraction
            random_long_list.append(
                build_random_train_by_env(
                    df,
                    seed=seed,
                    frac=float(f),
                )
            )

            # Range-based negative fraction control (existing behavior)
            if (random_min_neg_frac is not None) or (random_max_neg_frac is not None):
                random_long_list.append(
                    build_random_train_by_env(
                        df,
                        seed=seed,
                        frac=float(f),
                        min_neg_frac=float(random_min_neg_frac) if random_min_neg_frac is not None else None,
                        max_neg_frac=float(random_max_neg_frac) if random_max_neg_frac is not None else None,
                        max_tries=int(random_max_tries),
                    )
                )

            # Explicit target negative fractions (each creates its own biased split)
            if neg_fracs:
                for nf in neg_fracs:
                    try:
                        nf_val = float(nf)
                    except Exception:
                        continue
                    random_long_list.append(
                        build_random_train_by_env(
                            df,
                            seed=seed,
                            frac=float(f),
                            target_neg_frac=nf_val,
                            neg_frac_tol=neg_frac_tol,
                            max_tries=int(random_max_tries),
                        )
                    )

            # Explicit target positive fractions (alias via 1 - pos_frac)
            if pos_fracs:
                for pf in pos_fracs:
                    try:
                        pf_val = float(pf)
                    except Exception:
                        continue
                    nf_val = 1.0 - pf_val
                    random_long_list.append(
                        build_random_train_by_env(
                            df,
                            seed=seed,
                            frac=float(f),
                            target_neg_frac=nf_val,
                            neg_frac_tol=neg_frac_tol,
                            max_tries=int(random_max_tries),
                        )
                    )

        random_long = pd.concat(random_long_list, ignore_index=True) if random_long_list else pd.DataFrame(columns=["domain_task_id", "env", "split", "train", "origin"])

    # Union of provided + random
    splits_long = pd.concat([provided_long, random_long], ignore_index=True, sort=False)
    if splits_long.empty:
        splits_long = random_long

    results: List[Dict] = []
    preds_rows: List[Dict] = []

    split_names = splits_long["split"].unique().tolist() if not splits_long.empty else []

    groups = list(df.groupby(["config_name", "model", "env"]))
    if max_groups is not None:
        groups = groups[: max(1, int(max_groups))]

    for split_name in split_names:
        split_rows = splits_long[splits_long["split"] == split_name]

        join_keys = ["domain_task_id", "env"]
        if "gold_source" in split_rows.columns and split_rows["gold_source"].notna().any() and "gold_source" in df.columns:
            join_keys = ["domain_task_id", "env", "gold_source"]

        split_map = split_rows[join_keys + ["train"]]
        gdf_full = df.merge(split_map, on=join_keys, how="left")
        gdf_full["train"] = gdf_full["train"].fillna(0).astype(int)

        for (config_name, model, env), gdf in gdf_full.groupby(["config_name", "model", "env"]):
            train_mask = gdf["train"] == 1
            test_mask = ~train_mask
            train = gdf[train_mask]
            test = gdf[test_mask]

            if len(train) < 5 or len(test) < 3:
                continue

            s_train = train["raw_score"].astype(float).to_numpy()
            y_train = train["gold_score"].astype(int).to_numpy()

            s_test = test["raw_score"].astype(float).to_numpy()
            y_test = test["gold_score"].astype(int).to_numpy()

            # Skip degenerate cases where TRAIN has only one class
            # Platt/logistic-style calibration requires both positives and negatives.
            if len(np.unique(y_train)) < 2:
                continue

            # Baseline "before" probabilities: predicted_score if available
            if "predicted_score" in test.columns:
                p_test_before = test["predicted_score"].astype(float).to_numpy()
                p_test_before = _normalize_prob_array(p_test_before, eps=1e-6)
            else:
                p_test_before = np.full(len(test), np.nan, dtype=float)

            before_stats = _metrics_from_probs(y_test, p_test_before, thr=0.5)
            logloss_before = _log_loss(y_test, p_test_before)
            brier_before = _brier(y_test, p_test_before)

            # Metrics using predicted_score on all data (train+test)
            if "predicted_score" in gdf.columns:
                p_all = _normalize_prob_array(gdf["predicted_score"].astype(float).to_numpy(), eps=1e-6)
            else:
                p_all = np.full(len(gdf), np.nan, dtype=float)
            y_all = gdf["gold_score"].astype(int).to_numpy()
            all_stats = _metrics_from_probs(y_all, p_all, thr=0.5)

            # Gold label distribution stats (as percentages)
            n_train0 = int((y_train == 0).sum())
            n_train1 = int((y_train == 1).sum())
            n_test0 = int((y_test == 0).sum())
            n_test1 = int((y_test == 1).sum())
            train_pct_gold0 = 100.0 * n_train0 / len(train) if len(train) > 0 else float("nan")
            train_pct_gold1 = 100.0 * n_train1 / len(train) if len(train) > 0 else float("nan")
            test_pct_gold0 = 100.0 * n_test0 / len(test) if len(test) > 0 else float("nan")
            test_pct_gold1 = 100.0 * n_test1 / len(test) if len(test) > 0 else float("nan")

            # Run each calibration method
            for method in calibration_methods:
                calibrator = Calibrator(method=method)
                calibrator.fit(s_train, y_train)

                p_test_after = calibrator.predict(s_test)
                # Also compute calibrated probabilities for TRAIN examples
                p_train_after = calibrator.predict(s_train)
                logloss_after = _log_loss(y_test, p_test_after)
                brier_after = _brier(y_test, p_test_after)
                after_stats = _metrics_from_probs(y_test, p_test_after, thr=0.5)

                # A,B are only meaningful for Platt; NaN otherwise
                if method == "platt" and isinstance(calibrator.platt_params, PlattParams):
                    A = calibrator.platt_params.A
                    B = calibrator.platt_params.B
                else:
                    A = float("nan")
                    B = float("nan")

                results.append(
                    {
                        "calibration_method": method,
                        "config_name": config_name,
                        "model": model,
                        "env": env,
                        "split": split_name,
                        # metrics
                        "tpr_after": after_stats["tpr"] * 100,
                        "tpr_before": before_stats["tpr"] * 100,
                        "tpr_pred_all": all_stats["tpr"] * 100,
                        "tnr_after": after_stats["tnr"] * 100,
                        "tnr_before": before_stats["tnr"] * 100,
                        "tnr_pred_all": all_stats["tnr"] * 100,
                        "accuracy_after": after_stats["accuracy"] * 100,
                        "accuracy_before": before_stats["accuracy"] * 100,
                        "acc_pred_all": all_stats["accuracy"] * 100,
                        "bias_after": after_stats["bias"] * 100,
                        "bias_before": before_stats["bias"] * 100,
                        "bias_pred_all": all_stats["bias"] * 100,
                        "dskew_after": after_stats["dskew"] * 100,
                        "dskew_before": before_stats["dskew"] * 100,
                        "dskew_pred_all": all_stats["dskew"] * 100,
                        # Other data
                        "n_train": int(len(train)),
                        "n_test": int(len(test)),
                        "train_pct_gold0": train_pct_gold0,
                        "train_pct_gold1": train_pct_gold1,
                        "test_pct_gold0": test_pct_gold0,
                        "test_pct_gold1": test_pct_gold1,
                        "logloss_before": logloss_before,
                        "logloss_after": logloss_after,
                        "brier_before": brier_before,
                        "brier_after": brier_after,
                        "A": A,
                        "B": B,
                    }
                )

                # Per-row predictions for this method (TEST set)
                # Use vectorized test predictions already computed.
                for dtid, raw_s, gold, tpl, prob in zip(
                    test["domain_task_id"],
                    s_test,
                    y_test,
                    test["eval_template"],
                    p_test_after,
                ):
                    preds_rows.append(
                        {
                            "domain_task_id": dtid,
                            "env": env,
                            "model": model,
                            "config_name": config_name,
                            "eval_template": tpl,
                            "split": split_name,
                            "calibration_method": method,
                            "raw_score": float(raw_s),
                            "calibrated_prob": float(prob),
                            "gold_score": int(gold),
                            "train": 0,
                        }
                    )

                # Per-row predictions for TRAIN set (optional downstream use: overfitting analysis)
                for dtid, raw_s, gold, tpl, prob in zip(
                    train["domain_task_id"],
                    s_train,
                    y_train,
                    train["eval_template"],
                    p_train_after,
                ):
                    preds_rows.append(
                        {
                            "domain_task_id": dtid,
                            "env": env,
                            "model": model,
                            "config_name": config_name,
                            "eval_template": tpl,
                            "split": split_name,
                            "calibration_method": method,
                            "raw_score": float(raw_s),
                            "calibrated_prob": float(prob),
                            "gold_score": int(gold),
                            "train": 1,
                        }
                    )

    params_df = pd.DataFrame(results).sort_values(["calibration_method", "env", "model", "config_name", "split"]) if results else pd.DataFrame()
    preds_df = pd.DataFrame(preds_rows) if preds_rows else pd.DataFrame()

    params_out = out_dir / "calibration_params.csv"
    preds_out = out_dir / "calibration_predictions.csv"
    # Round floats for readability
    float_cols = params_df.select_dtypes(include=["float"]).columns.tolist()
    params_df[float_cols] = params_df[float_cols].round(2)
    params_df.to_csv(params_out, index=False)
    preds_df.to_csv(preds_out, index=False)

    print(f"Saved params to {params_out} ({len(params_df)} rows)")
    print(f"Saved calibrated predictions to {preds_out} ({len(preds_df)} rows)")

    return params_df, preds_df


# =============================
# CLI
# =============================


def main():
    import argparse

    p = argparse.ArgumentParser(description="Calibration (Platt / Isotonic / Histogram) for evaluation outputs")
    p.add_argument(
        "--csv",
        default="offline_experiments/calibration/calibration_data/verifications.csv",
        help="Path to model evaluations to calibrate",
    )
    p.add_argument(
        "--train",
        default="offline_experiments/calibration/calibration_data/train_set_calibration.csv",
        help="Path to training set for calibration",
    )
    p.add_argument(
        "--out-dir",
        default="offline_experiments/results",
        help="Directory to save outputs",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for random training sets",
    )
    p.add_argument(
        "--random-fracs",
        type=str,
        default="",
        help="Comma-separated list of training fractions for random splits (e.g., '0.2,1/3,0.5')",
    )
    p.add_argument(
        "--max-groups",
        type=int,
        default=None,
        help="Optional cap on number of groups to process",
    )
    p.add_argument(
        "--calibration-methods",
        type=str,
        default="platt",
        help="Comma-separated list of calibration methods to use. Supported: platt,isotonic,histogram,logistic,prior",
    )
    p.add_argument(
        "--neg-fracs",
        type=str,
        default="",
        help="Comma-separated list of target negative label fractions for biased TRAIN splits (e.g. '0.3,0.7' to create 70/30 and 30/70 pos/neg splits). Each value generates an additional split per training fraction.",
    )
    p.add_argument(
        "--pos-fracs",
        type=str,
        default="",
        help="Comma-separated list of target positive label fractions for biased TRAIN splits (e.g. '0.3,0.7'). Each value generates an additional split per training fraction; implemented as 1 - pos_frac on the negative side.",
    )
    p.add_argument(
        "--neg-frac-tol",
        type=float,
        default=0.05,
        help="Tolerance for matching target negative fraction in biased splits (abs(actual_neg_frac - target) <= tol).",
    )

    args = p.parse_args()

    def _parse_fraction_token(tok: str) -> float:
        t = tok.strip()
        if "/" in t:
            try:
                n, d = t.split("/", 1)
                return float(n) / float(d)
            except Exception:
                return float(t)
        return float(t)

    def _parse_fracs_arg(arg: str) -> Optional[List[float]]:
        if not arg:
            return None
        values = [_parse_fraction_token(piece) for piece in arg.split(",") if isinstance(piece, str) and piece.strip() != ""]
        return values if len(values) > 0 else None

    random_fracs = _parse_fracs_arg(args.random_fracs)
    neg_fracs = _parse_fracs_arg(args.neg_fracs)
    pos_fracs = _parse_fracs_arg(args.pos_fracs)

    methods_raw = [m.strip() for m in args.calibration_methods.split(",") if m.strip()]
    # Reuse the global allowed_methods (now includes logistic and prior) instead of shadowing it.
    calibration_methods = [m for m in methods_raw if m in allowed_methods]
    if not calibration_methods:
        raise ValueError(f"No valid calibration methods in {methods_raw}. Allowed: {sorted(allowed_methods)}")

    compute_calibration(
        args.csv,
        args.train,
        args.out_dir,
        seed=args.seed,
        max_groups=args.max_groups,
        random_fracs=random_fracs,
        neg_fracs=neg_fracs,
        pos_fracs=pos_fracs,
        neg_frac_tol=float(args.neg_frac_tol),
        calibration_methods=calibration_methods,
    )


if __name__ == "__main__":
    main()

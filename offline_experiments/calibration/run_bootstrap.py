import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_curve, accuracy_score, recall_score, roc_auc_score


def optimal_threshold_youden(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    Compute ROC-based optimal threshold using Youden's J statistic (TPR - FPR).
    """
    # roc_curve requires at least one positive and one negative
    if len(np.unique(y_true)) < 2:
        return 0.5  # fallback; shouldn't happen often if you filtered properly

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    J = tpr - fpr
    idx = np.argmax(J)
    return thresholds[idx]


def bootstrap_thresholds(
    y_train: np.ndarray,
    p_train: np.ndarray,
    B: int = 200,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Bootstrap optimal thresholds on the TRAIN set.
    Returns an array of length B with one threshold per bootstrap sample.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(y_train)
    thrs = []

    for _ in range(B):
        idx = rng.integers(0, n, size=n)  # sample with replacement
        y_b = y_train[idx]
        p_b = p_train[idx]

        thr = optimal_threshold_youden(y_b, p_b)
        thrs.append(thr)

    return np.array(thrs, dtype=float)


def evaluate_on_test(
    y_test: np.ndarray,
    p_test: np.ndarray,
    thr: float,
) -> dict:
    """Apply threshold and compute metrics: accuracy, TPR, TNR, bias, dskew.

    bias = mean(y_hat - y_true) (positive means over-predict positives)
    dskew = distance skewness of (y_hat - y_true) differences using same
            definition as in run_calibration.py.
    """
    y_hat = (p_test >= thr).astype(int)

    acc = accuracy_score(y_test, y_hat)
    tpr = recall_score(y_test, y_hat, pos_label=1, zero_division=0)
    tnr = recall_score(y_test, y_hat, pos_label=0, zero_division=0)

    diff = y_hat - y_test
    bias = float(np.mean(diff)) if diff.size > 0 else float("nan")
    # Distance skewness (reuse logic from run_calibration)
    if diff.size == 0:
        dskew = float("nan")
    else:
        pairwise_distances = np.abs(np.subtract.outer(diff, diff))
        numerator = np.sum(pairwise_distances)
        denominator = np.sum(np.abs(np.add.outer(diff, diff) - 2 * 0))
        dskew = float(1 - numerator / denominator) if denominator != 0 else 0.0

    return {"accuracy": acc, "tpr": tpr, "tnr": tnr, "bias": bias, "dskew": dskew}


def run_bootstrap_for_file(
    preds_csv: str | Path = "offline_experiments/calibration_predictions.csv",
    out_csv: str | Path = "offline_experiments/bootstrap_thresholds.csv",
    B: int = 200,
    min_train: int = 20,
    min_test: int = 20,
) -> pd.DataFrame:
    """
    Main entry point:
      - reads calibration_predictions.csv
      - for each (split, env, model, config_name, calibration_method),
        runs bootstrap on TRAIN rows and evaluates quantile thresholds on TEST rows
      - writes summary csv and returns it as a DataFrame
    """
    preds_csv = Path(preds_csv)
    out_csv = Path(out_csv)

    df = pd.read_csv(preds_csv)

    required_cols = {
        "split",
        "env",
        "model",
        "config_name",
        "calibration_method",
        "gold_score",
        "calibrated_prob",
        "train",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {preds_csv}: {missing}")

    results = []
    rng = np.random.default_rng()

    group_keys = ["split", "env", "model", "config_name", "calibration_method"]

    for key, g in df.groupby(group_keys):
        split, env, model, config_name, method = key

        g_train = g[g["train"] == 1]
        g_test = g[g["train"] == 0]

        if len(g_train) < min_train or len(g_test) < min_test:
            # Not enough data for a meaningful bootstrap
            continue

        y_train = g_train["gold_score"].to_numpy(dtype=int)
        p_train = g_train["calibrated_prob"].to_numpy(dtype=float)

        y_test = g_test["gold_score"].to_numpy(dtype=int)
        p_test = g_test["calibrated_prob"].to_numpy(dtype=float)

        # Skip degenerate cases (e.g., only one class in train)
        if len(np.unique(y_train)) < 2:
            continue

        # Class distribution stats (counts and percentages)
        n_train0 = int((y_train == 0).sum())
        n_train1 = int((y_train == 1).sum())
        n_test0 = int((y_test == 0).sum())
        n_test1 = int((y_test == 1).sum())

        train_pos_pct = 100.0 * n_train1 / len(y_train) if len(y_train) > 0 else float("nan")
        train_neg_pct = 100.0 * n_train0 / len(y_train) if len(y_train) > 0 else float("nan")
        test_pos_pct = 100.0 * n_test1 / len(y_test) if len(y_test) > 0 else float("nan")
        test_neg_pct = 100.0 * n_test0 / len(y_test) if len(y_test) > 0 else float("nan")

        # --- AUC on TEST set (threshold-free performance) ---
        try:
            auc_test = float(roc_auc_score(y_test, p_test)) if len(np.unique(y_test)) == 2 else float("nan")
        except Exception:
            auc_test = float("nan")

        # --- Optimal threshold on full TRAIN (Youden) ---
        thr_opt_train = optimal_threshold_youden(y_train, p_train)
        metrics_opt = evaluate_on_test(y_test, p_test, thr_opt_train)

        # --- Optimal threshold on TEST (diagnostic only; not for deployment) ---
        thr_opt_test = optimal_threshold_youden(y_test, p_test)
        metrics_opt_test = evaluate_on_test(y_test, p_test, thr_opt_test)

        # --- Bootstrap thresholds on TRAIN ---
        thrs = bootstrap_thresholds(y_train, p_train, B=B, rng=rng)

        thr_mean = float(thrs.mean())
        thr_std = float(thrs.std())
        thr_p05 = float(np.percentile(thrs, 5))
        thr_p25 = float(np.percentile(thrs, 25))
        thr_p50 = float(np.percentile(thrs, 50))
        thr_p75 = float(np.percentile(thrs, 75))
        thr_p95 = float(np.percentile(thrs, 95))

        # --- Diagnostic: Bootstrap thresholds on TEST (data leakage; for analysis only) ---
        thrs_test = bootstrap_thresholds(y_test, p_test, B=B, rng=rng)
        thr_test_mean = float(thrs_test.mean())
        thr_test_std = float(thrs_test.std())
        thr_test_p05 = float(np.percentile(thrs_test, 5))
        thr_test_p25 = float(np.percentile(thrs_test, 25))
        thr_test_p50 = float(np.percentile(thrs_test, 50))
        thr_test_p75 = float(np.percentile(thrs_test, 75))
        thr_test_p95 = float(np.percentile(thrs_test, 95))

        # --- Evaluate percentile thresholds on TEST (train-derived thresholds) ---
        # Fixed threshold 0.5 metrics (commonly used baseline)
        metrics_fixed_05 = evaluate_on_test(y_test, p_test, 0.5)
        metrics_p05 = evaluate_on_test(y_test, p_test, thr_p05)
        metrics_p25 = evaluate_on_test(y_test, p_test, thr_p25)
        metrics_p50 = evaluate_on_test(y_test, p_test, thr_p50)
        metrics_p75 = evaluate_on_test(y_test, p_test, thr_p75)
        metrics_p95 = evaluate_on_test(y_test, p_test, thr_p95)

        results.append(
            {
                "split": split,
                "env": env,
                "model": model,
                "config_name": config_name,
                "calibration_method": method,
                "n_train": len(g_train),
                "n_test": len(g_test),
                "auc_test": auc_test,
                "train_pct_gold1": train_pos_pct,
                "train_pct_gold0": train_neg_pct,
                "test_pct_gold1": test_pos_pct,
                "test_pct_gold0": test_neg_pct,
                # Optimal threshold metrics (train-derived)
                "thr_opt_train": thr_opt_train,
                "test_acc_thr_opt": metrics_opt["accuracy"],
                "test_tpr_thr_opt": metrics_opt["tpr"],
                "test_tnr_thr_opt": metrics_opt["tnr"],
                "test_bias_thr_opt": metrics_opt["bias"],
                "test_dskew_thr_opt": metrics_opt["dskew"],
                # Optimal threshold metrics (test-derived, diagnostic)
                "thr_opt_test": thr_opt_test,
                "test_acc_thr_opt_test": metrics_opt_test["accuracy"],
                "test_tpr_thr_opt_test": metrics_opt_test["tpr"],
                "test_tnr_thr_opt_test": metrics_opt_test["tnr"],
                "test_bias_thr_opt_test": metrics_opt_test["bias"],
                "test_dskew_thr_opt_test": metrics_opt_test["dskew"],
                # Bootstrap summary statistics (train-derived thresholds)
                "thr_mean": thr_mean,
                "thr_std": thr_std,
                "thr_p05": thr_p05,
                "thr_p25": thr_p25,
                "thr_p50": thr_p50,
                "thr_p75": thr_p75,
                "thr_p95": thr_p95,
                # Test bootstrap threshold distribution (diagnostic only)
                "thr_test_mean": thr_test_mean,
                "thr_test_std": thr_test_std,
                "thr_test_p05": thr_test_p05,
                "thr_test_p25": thr_test_p25,
                "thr_test_p50": thr_test_p50,
                "thr_test_p75": thr_test_p75,
                "thr_test_p95": thr_test_p95,
                # Test metrics at train-derived percentile thresholds
                "test_acc_thr_0_5": metrics_fixed_05["accuracy"],
                "test_tpr_thr_0_5": metrics_fixed_05["tpr"],
                "test_tnr_thr_0_5": metrics_fixed_05["tnr"],
                "test_bias_thr_0_5": metrics_fixed_05["bias"],
                "test_dskew_thr_0_5": metrics_fixed_05["dskew"],
                "test_acc_thr_p05": metrics_p05["accuracy"],
                "test_tpr_thr_p05": metrics_p05["tpr"],
                "test_tnr_thr_p05": metrics_p05["tnr"],
                "test_bias_thr_p05": metrics_p05["bias"],
                "test_dskew_thr_p05": metrics_p05["dskew"],
                "test_acc_thr_p25": metrics_p25["accuracy"],
                "test_tpr_thr_p25": metrics_p25["tpr"],
                "test_tnr_thr_p25": metrics_p25["tnr"],
                "test_bias_thr_p25": metrics_p25["bias"],
                "test_dskew_thr_p25": metrics_p25["dskew"],
                "test_acc_thr_p50": metrics_p50["accuracy"],
                "test_tpr_thr_p50": metrics_p50["tpr"],
                "test_tnr_thr_p50": metrics_p50["tnr"],
                "test_bias_thr_p50": metrics_p50["bias"],
                "test_dskew_thr_p50": metrics_p50["dskew"],
                "test_acc_thr_p75": metrics_p75["accuracy"],
                "test_tpr_thr_p75": metrics_p75["tpr"],
                "test_tnr_thr_p75": metrics_p75["tnr"],
                "test_bias_thr_p75": metrics_p75["bias"],
                "test_dskew_thr_p75": metrics_p75["dskew"],
                "test_acc_thr_p95": metrics_p95["accuracy"],
                "test_tpr_thr_p95": metrics_p95["tpr"],
                "test_tnr_thr_p95": metrics_p95["tnr"],
                "test_bias_thr_p95": metrics_p95["bias"],
                "test_dskew_thr_p95": metrics_p95["dskew"],
            }
        )

    out_df = pd.DataFrame(results)
    out_df.to_csv(out_csv, index=False)
    print(f"Saved bootstrap threshold analysis to {out_csv} ({len(out_df)} rows)")

    return out_df


if __name__ == "__main__":
    # Adjust paths / B as needed
    run_bootstrap_for_file(
        preds_csv="offline_experiments/calibration_predictions.csv",
        out_csv="offline_experiments/bootstrap_thresholds.csv",
        B=200,
    )

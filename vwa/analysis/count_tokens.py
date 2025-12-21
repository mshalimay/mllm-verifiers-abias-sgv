#!/usr/bin/env python3
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from trajectory_utils.trajectory import VWA_Trajectory


def parse_domain_task_id(x: Optional[str], traj_source_path: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Returns (domain_task_id, domain, task_id).
    Tries to parse from domain_task_id; if missing, falls back to traj_source_path (expects trajectory-<domain>-<task_id>.json).
    """
    if x and isinstance(x, str) and "_" in x:
        domain, task_id = x.split("_", 1)
        return f"{domain}_{task_id}", domain, task_id

    if traj_source_path:
        m = re.search(r"trajectory-([a-zA-Z]+)-(\d+)\.json$", str(traj_source_path))
        if m:
            domain, task_id = m.group(1), m.group(2)
            return f"{domain}_{task_id}", domain, task_id

    return None, None, None


def get_task_id_from_filename(file_path: str | Path) -> Optional[str]:
    p = Path(file_path)
    filename = p.name
    # Extract int from filename
    match = re.search(r"(\d+)\.csv$", filename)
    if match:
        return match.group(1)
    return None


def find_lm_usage_csvs(traj_source_path: str, domain: str, task_id: str) -> List[Path]:
    """
    From the trajectory JSON path, locate all CSVs inside any 'lm_usage' subdirectories under its directory.
    """
    tpath = Path(traj_source_path)
    if not tpath.exists():
        print(f"[WARN] traj_source_path does not exist: {traj_source_path}")
        return []

    domain_dir = re.search(rf".*/{re.escape(domain)}/", str(tpath))
    if not domain_dir:
        print(f"[WARN] Could not find domain directory for traj_source_path: {traj_source_path}")
        return []
    base_dir = Path(domain_dir.group(0))

    # New format for directories
    # traj_dir = re.search(rf"^(.*)/trajectories/", str(tpath))
    # if not traj_dir:
    #     return []
    # base_dir = Path(traj_dir.group(1)) / "trajectories" / task_id
    # if not base_dir.exists():
    #     base_dir = Path(traj_dir.group(1)) / "trajectories" / f"{task_id}_0"
    #     if not base_dir.exists():
    #         return []

    csvs: List[Path] = []
    for lm_dir in base_dir.rglob("lm_usage"):
        if lm_dir.is_dir():
            csvs.extend(sorted(lm_dir.glob("*.csv")))
    pruned_csvs = [p for p in csvs if p.is_file() and get_task_id_from_filename(p) == str(task_id)]
    if not pruned_csvs:
        print(f"[WARN] No lm_usage CSVs found for traj_source_path: {traj_source_path}")

    return pruned_csvs


def source_type_from_filename(csv_path: Path) -> str:
    """
    Extracts source type from filename prefix before the first underscore.
    Examples:
      executor_189.csv -> executor
      critic_123.csv   -> critic
      first_step_77.csv -> first_step
      foo_bar_9.csv -> foo (prefix before first underscore)
      If no underscore, uses stem.
    """
    stem = csv_path.stem
    # Strip integer
    stem = re.sub(r"_\d+$", "", stem)
    return stem.strip("_")


def get_sum_no_verif(csv_path: Path, stop_idx):
    """Sum numeric columns for the first stop_idx rows (inclusive) of the CSV.

    If stop_idx is None or CSV can't be read, return empty Series and 0 rows.
    """
    if stop_idx is None:
        return pd.Series(dtype="float64"), 0

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] Failed to read CSV for no-verif sum: {csv_path} ({e})")
        return pd.Series(dtype="float64"), 0

    if df.empty:
        return pd.Series(dtype="float64"), 0

    # Clamp slice to available rows; stop_idx is inclusive (0-based)
    end = min(int(stop_idx) + 1, len(df))
    if end <= 0:
        return pd.Series(dtype="float64"), 0

    sliced = df.iloc[:end]
    numeric_df = sliced.apply(pd.to_numeric, errors="coerce")
    sums = numeric_df.sum(numeric_only=True)
    return sums, len(sliced)


def sum_numeric_columns(csv_path: Path) -> Tuple[pd.Series, int]:
    """
    Reads a CSV and returns:
      - a Series with sums of numeric columns
      - number of rows counted
    Any non-numeric columns are ignored.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] Failed to read CSV: {csv_path} ({e})")
        return pd.Series(dtype="float64"), 0

    if df.empty:
        return pd.Series(dtype="float64"), 0

    # Convert all columns to numeric when possible
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    sums = numeric_df.sum(numeric_only=True)
    row_count = len(df)
    return sums, row_count


def aggregate_for_traj(domain_task_id: str, traj_source_path: str, domain: str, task_id: str) -> Tuple[List[Dict], List[str]]:
    """
    For a single traj_source_path:
      - find all lm_usage CSVs
      - group by source_type
      - sum numeric columns within each type
    Returns:
      - list of row dicts (detailed totals per type)
      - list of csv file paths processed
    """
    csvs = find_lm_usage_csvs(traj_source_path, domain, task_id)
    try:
        trajectory = VWA_Trajectory.from_json(traj_source_path)
        _, noverif_score_entry = trajectory.get_scores_for_round_step(0)
        stop_idx = None
        if noverif_score_entry:
            stop_idx = noverif_score_entry.get("state_idx", None)
    except Exception as e:
        stop_idx = None

    if not csvs:
        return [], []

    by_type: Dict[str, List[Path]] = {}
    for p in csvs:
        stype = source_type_from_filename(p)
        by_type.setdefault(stype, []).append(p)

    rows: List[Dict] = []
    processed_paths: List[str] = []

    for stype, files in sorted(by_type.items()):
        total_series = pd.Series(dtype="float64")
        total_rows = 0
        for f in files:
            sums, nrows = sum_numeric_columns(f)
            # Align indexes and add
            total_series = total_series.add(sums, fill_value=0.0)
            total_rows += nrows
            processed_paths.append(str(f))

        row = {
            "domain_task_id": domain_task_id,
            "source_type": stype,
            "file_count": len(files),
            "row_count": total_rows,
            "domain": domain,
            "task_id": task_id,
        }
        # Attach numeric totals (convert to python scalars)
        for k, v in total_series.items():
            row[str(k)] = float(v) if pd.notna(v) else 0.0
        rows.append(row)

        # Add executor_noverif row using the first stop_idx rows (inclusive)
        if stype == "executor" and stop_idx is not None:
            noverif_series = pd.Series(dtype="float64")
            noverif_rows = 0
            for f in files:
                nsums, nnrows = get_sum_no_verif(f, stop_idx)
                noverif_series = noverif_series.add(nsums, fill_value=0.0)
                noverif_rows += nnrows
                # processed_paths already includes f

            noverif_row = {
                "domain_task_id": domain_task_id,
                "source_type": "executor_noverif",
                "file_count": len(files),
                "row_count": noverif_rows,
                "domain": domain,
                "task_id": task_id,
            }
            for k, v in noverif_series.items():
                noverif_row[str(k)] = float(v) if pd.notna(v) else 0.0
            rows.append(noverif_row)

    # Add an 'ALL' row for this domain_task_id summing across all types
    if rows:
        df_rows = pd.DataFrame(rows)
        numeric_cols = [
            c for c in df_rows.columns if c not in {"domain_task_id", "source_type"} and pd.api.types.is_numeric_dtype(df_rows[c])
        ]
        # Exclude executor_noverif from ALL to avoid double counting
        df_rows_for_all = df_rows[df_rows["source_type"] != "executor_noverif"].copy()
        all_row = {
            "domain_task_id": domain_task_id,
            "source_type": "ALL",
            "file_count": int(df_rows_for_all["file_count"].sum()),
            "row_count": int(df_rows_for_all["row_count"].sum()),
            "domain": domain,
            "task_id": task_id,
        }
        for c in numeric_cols:
            all_row[c] = float(df_rows_for_all[c].sum())
        rows.append(all_row)

    return rows, processed_paths


def pivot_totals(detailed_df: pd.DataFrame, id_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Creates a pivot with domain_task_id rows, source_type columns, and summed numeric values.
    Column names flattened as '<source_type>__<metric>'.
    """
    if detailed_df.empty:
        return detailed_df

    if id_cols is None:
        id_cols = ["domain_task_id"]

    numeric_cols = [
        c for c in detailed_df.columns if c not in set(id_cols + ["source_type"]) and pd.api.types.is_numeric_dtype(detailed_df[c])
    ]
    if not numeric_cols:
        return pd.DataFrame()

    pivot = detailed_df.pivot_table(
        index=id_cols,
        columns="source_type",
        values=numeric_cols,
        aggfunc="sum",
        fill_value=0.0,
    )
    # Flatten multi-index columns like (metric, source_type) -> f"{source_type}__{metric}"
    if isinstance(pivot.columns, pd.MultiIndex):
        pivot.columns = [f"{str(col[1])}__{str(col[0])}" for col in pivot.columns]
    else:
        pivot.columns = [str(c) for c in pivot.columns]

    pivot = pivot.reset_index()
    return pivot


def main():
    # Configure your inputs and outputs here
    INPUT_CSVS: List[str] = [
        # Example inputs; replace with your list
        "./experiments/gemini-2.5-flash-001/experiment_1/scores.csv",
    ]
    UNIQUE_PER_DOMAIN_TASK = True  # dedupe per CSV by domain_task_id
    OUTPUT_DETAILED = "data_analysis/token_counts.csv"
    OUTPUT_PIVOT = "data_analysis/token_counts_pivot.csv"

    if not INPUT_CSVS:
        print("[ERROR] Please populate INPUT_CSVS with one or more CSV file paths.")
        return

    consolidated_rows: List[Dict] = []
    all_processed_files: List[str] = []

    for csv_source in INPUT_CSVS:
        in_path = Path(csv_source)
        if not in_path.exists():
            print(f"[WARN] Input CSV not found, skipping: {in_path}")
            continue

        try:
            df = pd.read_csv(in_path)
        except Exception as e:
            print(f"[WARN] Failed to read input CSV, skipping: {in_path} ({e})")
            continue

        if "traj_source_path" not in df.columns:
            print(f"[WARN] Missing 'traj_source_path' in {in_path}, skipping.")
            continue

        print("Processing CSV:", in_path)
        # Ensure domain/task identifiers present or derive them
        if "domain" not in df.columns or "task_id" not in df.columns:
            # Try to derive domain_task_id if missing
            if "domain_task_id" not in df.columns:
                df["domain_task_id"], _, _ = zip(*df["traj_source_path"].map(lambda p: parse_domain_task_id(None, p)))

            # Split domain and task_id if possible
            def _split_domain_task_id(val: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
                if isinstance(val, str) and "_" in val:
                    d, tid = val.split("_", 1)
                    return d, tid
                return None, None

            if "domain" not in df.columns:
                df["domain"], _ = zip(*df["domain_task_id"].map(_split_domain_task_id))
            if "task_id" not in df.columns:
                _, df["task_id"] = zip(*df["domain_task_id"].map(_split_domain_task_id))

        # Drop invalid rows
        df = df.dropna(subset=["traj_source_path", "domain", "task_id"]).copy()
        # Ensure domain_task_id exists
        if "domain_task_id" not in df.columns:
            df["domain_task_id"] = df.apply(lambda r: f"{r['domain']}_{r['task_id']}", axis=1)

        # Dedupe per CSV if configured
        if UNIQUE_PER_DOMAIN_TASK:
            df = df.sort_values("traj_source_path").drop_duplicates(subset=["domain_task_id"], keep="first")

        for row in df.itertuples(index=False):
            domain = getattr(row, "domain")
            task_id = getattr(row, "task_id")
            dtid = f"{domain}_{task_id}"
            traj_path = getattr(row, "traj_source_path")
            if not isinstance(traj_path, str):
                continue
            rows, processed = aggregate_for_traj(dtid, traj_path, domain, task_id)
            # Attach csv_source metadata
            for r in rows:
                r["csv_source"] = str(in_path)
                r["domain_task_id_csv_source"] = f"{r['domain_task_id']}|{in_path}"
            consolidated_rows.extend(rows)
            all_processed_files.extend(processed)

    detailed_df = pd.DataFrame(consolidated_rows)
    Path(OUTPUT_DETAILED).parent.mkdir(parents=True, exist_ok=True)
    detailed_df.to_csv(OUTPUT_DETAILED, index=False)
    print(f"[INFO] Wrote consolidated totals: {OUTPUT_DETAILED} (rows={len(detailed_df)})")

    # Pivot by domain_task_id and csv_source for clarity
    id_cols = ["domain_task_id", "csv_source"] if "csv_source" in detailed_df.columns else ["domain_task_id"]
    pivot_df = pivot_totals(detailed_df, id_cols=id_cols)
    Path(OUTPUT_PIVOT).parent.mkdir(parents=True, exist_ok=True)
    pivot_df.to_csv(OUTPUT_PIVOT, index=False)
    print(f"[INFO] Wrote consolidated pivot: {OUTPUT_PIVOT} (rows={len(pivot_df)})")

    # Optional: write a manifest of processed CSV files
    manifest_path = Path(OUTPUT_DETAILED).with_suffix(".files.txt")
    if all_processed_files:
        manifest_path.write_text("\n".join(sorted(set(all_processed_files))) + "\n")
        print(f"[INFO] Wrote file manifest: {manifest_path}")


main()

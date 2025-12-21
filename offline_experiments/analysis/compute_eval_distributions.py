"""
Compute distribution tables of evaluation responses across different templates.

This script analyzes all_evals_env.csv files and creates distribution tables showing
the percentage of responses falling into different categories (SUCCESS, PARTIAL SUCCESS,
FAILURE, etc.) or numeric ranges (1-10) for each config_name + model + env combination.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

ALL_EVALS_PATH = Path("offline_experiments") / "results" / "all_evals_consolidated.csv"

from offline_experiments.analysis.eval_mapping import (
    extract_eval_template_from_variation_name,
    map_numeric_score_to_category,
    map_response_to_category,
)


def map_numeric_to_aggregated_category(score_str: str, eval_template: Optional[str]) -> Optional[str]:
    """
    Thin wrapper that defers to the centralized mapping in eval_mapping.py
    to convert a numeric score (1-10) to an aggregated category.
    """
    return map_numeric_score_to_category(score_str, eval_template)


def categorize_response(response: str, eval_template: Optional[str] = None) -> str:
    """
    Categorize an evaluation response based on its template and mapped score.

    Returns categories like:
    - "SUCCESS" (score=1)
    - "PARTIAL SUCCESS" (score=0 but not complete failure)
    - "PARTIAL FAILURE" (score=0 but partial work done)
    - "FAILURE" (score=0)
    - "UNCERTAIN" (score=nan)
    - Numeric values (for numeric templates)
    """
    if not response or pd.isna(response):
        return "INVALID"

    response = response.strip().upper()

    # For numeric responses: return the number string for numeric templates.
    # For non-numeric templates (e.g., four_room_score), map to categories centrally.

    if response.replace(".", "").replace("-", "").isdigit():
        if eval_template and any(eval_template.startswith(prefix) for prefix in ["tri_num", "four_num", "range_num", "four_degree", "four_room_score"]):
            try:
                num = float(response)
                if num == int(num):
                    return str(int(num))
                return response
            except Exception:
                pass
        else:
            mapped = map_response_to_category(response, eval_template)
            if mapped is not None:
                return mapped
    # Delegate categorical mapping to centralized function
    category = map_response_to_category(response, eval_template)
    if category is not None:
        return category

    # Default fallback if no category matched
    return "INVALID"


def compute_distribution_from_all_evals(all_evals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute distribution table from all_evals_env.csv file.

    Groups by config_name + model + env and computes distribution percentages.

    Args:
        all_evals_df: DataFrame from all_evals_env.csv with columns:
                     ['domain_task_id', 'eval', 'config_name', 'model', 'env', ...]

    Returns:
        DataFrame with rows as categories and columns as config_name--model--env combinations
    """
    required_cols = ["eval", "config_name", "model", "env"]
    missing_cols = [col for col in required_cols if col not in all_evals_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Group by config_name, model, env
    groups = all_evals_df.groupby(["config_name", "model", "env"])

    distributions = {}

    for (config_name, model, env), group_df in groups:
        # Extract template from config_name
        eval_template = extract_eval_template_from_variation_name(config_name)

        # Get responses
        responses = group_df["eval"].dropna()

        if len(responses) == 0:
            continue

        # Categorize each response
        categories = responses.apply(lambda x: categorize_response(x, eval_template))

        # Count categories
        category_counts = categories.value_counts()
        total = len(categories)
        category_pcts = category_counts / total * 100

        # Create column name: config_name--model--env
        col_name = f"{config_name}--{model}--{env}"
        distributions[col_name] = category_pcts.to_dict()
        distributions[col_name]["# evals"] = total
        # Record eval_template per column for a final summary row
        distributions[col_name]["eval_template"] = eval_template or ""

        # Skip aggregation if eval_template is None
        if not eval_template:
            print(f"No eval template found for {config_name}")
            continue

        # Compute aggregated categories for ALL templates
        if any(eval_template.startswith(prefix) for prefix in ["tri_num", "four_num", "range_num", "four_degree"]):
            # For numeric templates, map numeric values to categories
            aggregated_categories = responses.apply(lambda x: map_numeric_to_aggregated_category(x, eval_template))
            # Filter out None values (non-numeric responses)
            aggregated_categories = aggregated_categories[aggregated_categories.notna()]

            if len(aggregated_categories) > 0:
                aggregated_counts = aggregated_categories.value_counts()
                aggregated_total = len(aggregated_categories)
                aggregated_pcts = aggregated_counts / aggregated_total * 100

                # Add aggregated rows with [AGG] prefix
                for category, pct in aggregated_pcts.items():
                    if category in ("FAILURE", "PARTIAL FAILURE"):
                        prev = distributions[col_name].get("[AGG] FAILURE + PARTIAL FAILURE", 0)
                        distributions[col_name]["[AGG] FAILURE + PARTIAL FAILURE"] = prev + float(pct)
                    else:
                        distributions[col_name][f"[AGG] {category}"] = float(pct)
        else:
            # For non-numeric templates, aggregate the existing categorical percentages
            # [AGG] SUCCESS = SUCCESS
            # [AGG] PARTIAL SUCCESS = PARTIAL SUCCESS
            # [AGG] UNCERTAIN = UNCERTAIN
            # [AGG] FAILURE + PARTIAL FAILURE = FAILURE + PARTIAL FAILURE
            distributions[col_name]["[AGG] SUCCESS"] = category_pcts.get("SUCCESS", 0)
            distributions[col_name]["[AGG] PARTIAL SUCCESS"] = category_pcts.get("PARTIAL SUCCESS", 0)
            distributions[col_name]["[AGG] UNCERTAIN"] = category_pcts.get("UNCERTAIN", 0)

            # Combine FAILURE and PARTIAL FAILURE
            failure_pct = category_pcts.get("FAILURE", 0) + category_pcts.get("PARTIAL FAILURE", 0)
            distributions[col_name]["[AGG] FAILURE + PARTIAL FAILURE"] = float(failure_pct)

    # Create DataFrame
    dist_df = pd.DataFrame(distributions).fillna(0.0)

    # Sort rows by common order
    row_order = ["SUCCESS", "PARTIAL SUCCESS", "PARTIAL FAILURE", "FAILURE", "UNCERTAIN"]
    aggregated_row_order = ["[AGG] SUCCESS", "[AGG] PARTIAL SUCCESS", "[AGG] UNCERTAIN", "[AGG] FAILURE + PARTIAL FAILURE"]
    numeric_rows = [str(i) for i in range(1, 11)]

    # Determine which rows exist (excluding '# evals')
    existing_rows = []

    # Add categorical rows first
    for row in row_order:
        if row in dist_df.index:
            existing_rows.append(row)

    # Add aggregated rows (for numeric templates)
    for row in aggregated_row_order:
        if row in dist_df.index:
            existing_rows.append(row)

    # Add numeric rows if they exist
    for row in numeric_rows:
        if row in dist_df.index:
            existing_rows.append(row)

    # Add any remaining rows (except '# evals')
    for row in dist_df.index:
        if row not in existing_rows and row != "# evals":
            existing_rows.append(row)

    # Add '# evals' and 'eval_template' at the end (in that order)
    if "# evals" in dist_df.index:
        existing_rows.append("# evals")
    if "eval_template" in dist_df.index:
        existing_rows.append("eval_template")

    # Reindex
    dist_df = dist_df.reindex(existing_rows, fill_value=0)

    return dist_df


def analyze_all_evals_csv(csv_path: str, output_file: Optional[str] = None) -> pd.DataFrame:
    """
    Analyze a single all_evals_env.csv file and create distribution table.

    Args:
        csv_path: Path to all_evals_env.csv file
        output_file: Optional path to save the distribution table

    Returns:
        DataFrame with distribution statistics
    """
    print(f"Reading {csv_path}...")

    try:
        all_evals_df = pd.read_csv(csv_path)
        print(f"  Loaded {len(all_evals_df)} evaluations")

        # Compute distribution table
        dist_table = compute_distribution_from_all_evals(all_evals_df)

        print(f"  Created distribution table with {len(dist_table.columns)} combinations")

        # Save if output path provided
        if output_file:
            dist_table.to_csv(output_file)
            print(f"  Distribution table saved to {output_file}")

        return dist_table

    except Exception as e:
        print(f"  Error processing {csv_path}: {e}")
        raise


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = str(ALL_EVALS_PATH)

    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        # Generate output filename based on input
        input_path = Path(csv_path)
        output_file = str(input_path.parent / f"{input_path.stem}_distributions.csv")

    print(f"Analyzing: {csv_path}")
    print(f"Output file: {output_file}")
    print("=" * 80 + "\n")

    analyze_all_evals_csv(csv_path, output_file)

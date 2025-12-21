import concurrent.futures
import html as _html
import os
import re
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom, chi2

from core_utils.string_utils import clean_spaces
from offline_experiments.analysis.eval_mapping import EVAL_LABELS, extract_eval_template_from_variation_name, map_eval_to_score_with_context
from offline_experiments.utils_offline.utils_offline_exper import get_domain_from_path

# Suppress RuntimeWarning for division by zero and PerformanceWarning for DataFrame fragmentation
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# --------------------------------
# Paths & Config to Input Files
# --------------------------------
# fmt:off
# Directories with oracle/human labels for each env
GOLD_SCORES_DIRS = {
    "vwa": "offline_experiments/gold_scores/vwa",
    "osw": "offline_experiments/gold_scores/osw",
    "agrb_vwa": "offline_experiments/gold_scores/agrb_vwa",
}
# Environments to process
ENVS_TO_PROCESS = ["vwa"]  # ["vwa, "osw", "agrb_vwa"]

# Paths to the offline experiments containing MLLM verification traces
EXPERIMENTS_PATHS = ["offline_experiments/results"]

# Pattern used to find trace files in each experiments_path
FILE_TO_PARSE_PATTERN = "**/conversation/**/*.txt"
FILE_TO_PARSE_PATTERN_HTML = "**/*.html"


IGNORE_HTML = True  # Ignore trace files in HTML format
MUST_INCLUDE_STRS = []  # Include only trace_files that contain any of these strings
EXCLUDE_STRS = ["logs", "log_files", "gold_scores", "/k_", "/k-", "sys_aeval_refine"]  # Exclude trace files containing any of these substrings

# --------------------------------
# Output config
# --------------------------------
OVERWRITE = True                            # Overwrite existing output files
DELETE_INVALID_FILES = False                # If trace file contains no valid verification, delete trace and usage files
OUTPUT_DIR = "offline_experiments/results"  # Directory to save consolidated statistics file per env and for all envs
EVALUATIONS_CSV_NAME = "evaluations.csv"    # Model evaluations (likerts, scores, etc) are saved to "evaluations.csv" files inside each agent-verifier-configuration dir

EVAL_SET_VWA_PATH = "offline_experiments/config/task_lists/vwa_lite.txt"  # Compute stats for VWA-lite
TEST_SET_AGRB_PATH = "offline_experiments/utils_agrb/splits.csv"  # Compute stats for AGRB test set

np.random.seed(42)
# fmt:on
# ==============================================================
# I/O Helpers
# ==============================================================


def save_csv(df: pd.DataFrame, path: Path | str, gold_scores: pd.DataFrame | None = None, key_gold_score: str = "gold_score"):
    df_to_save = df.copy()
    if gold_scores is not None:
        # Ensure unique_id is of type string for both DataFrames
        gold_df = gold_scores.copy()
        # Create unique_id concatenating domain and task_id
        gold_df["domain_task_id"] = gold_df["domain"].astype(str) + "_" + gold_df["task_id"].astype(str)

        gold_df["domain_task_id"] = gold_df["domain_task_id"].astype(str)
        df_to_save["domain_task_id"] = df_to_save["domain_task_id"].astype(str)

        # Subset gold_scores to only include domain_task_id, score, gold_source and optionally traj_len if present
        subset_cols = ["domain_task_id", key_gold_score, "gold_source"]
        if "traj_len" in gold_df.columns:
            subset_cols.append("traj_len")
        gold_scores_subset = gold_df[subset_cols]

        # Drop the existing gold_score column if it already exists
        if key_gold_score in df_to_save.columns:
            df_to_save = df_to_save.drop(columns=[key_gold_score])

        df_to_save = pd.merge(df_to_save, gold_scores_subset, on="domain_task_id", how="left")

    # Sort by unique_id
    df_to_save = df_to_save.sort_values(by="domain_task_id")
    # Reorder to have metadata cols first
    cols_first = [col for col in df_to_save.columns if not col.endswith("_eval")]
    eval_cols = [col for col in df_to_save.columns if col.endswith("_eval")]
    df_to_save = df_to_save[cols_first + eval_cols]

    # Ensure 'traj_len' appears as the last column if present.
    if "traj_len" in df_to_save.columns:
        cols = [c for c in df_to_save.columns if c != "traj_len"]
        cols.append("traj_len")
        df_to_save = df_to_save[cols]

    # Add per-row confusion label (TP/FP/TN/FN) for each evaluation column
    if gold_scores is not None and "gold_score" in df_to_save.columns:
        df_confusion = pd.DataFrame()
        # Add non-eval columns to df_confusion
        for col in df_to_save.columns:
            if col not in eval_cols:
                df_confusion[col] = df_to_save[col]

        for col in eval_cols:
            eval_upper = df_to_save[col].astype(str).str.upper().str.strip()
            gold = df_to_save["gold_score"]
            confusion = np.where(
                (gold == 1) & (eval_upper == "SUCCESS"),
                "TP",
                np.where(
                    (gold == 0) & (eval_upper == "SUCCESS"),
                    "FP",
                    np.where(
                        (gold == 0) & (eval_upper.isin(["FAILURE", "PARTIAL FAILURE", "PARTIAL SUCCESS"])),
                        "TN",
                        np.where((gold == 1) & (eval_upper.isin(["FAILURE", "PARTIAL FAILURE", "PARTIAL SUCCESS"])), "FN", ""),
                    ),
                ),
            )
            df_to_save[f"{col}_confusion"] = confusion
            df_confusion[col] = confusion
        df_confusion.to_csv(Path(path).parent / "confusion.csv", index=False)

    df_to_save.to_csv(path, index=False)

    return df_to_save


def get_gold_score_path_source(model_subdir: str | Path, gold_scores_dir: str | Path, env: str) -> tuple[Path, str]:
    agent_dir = str(model_subdir).split(env)[-1].strip("/")
    gold_scores_filepath = Path(gold_scores_dir) / f"{agent_dir}.csv"
    if not gold_scores_filepath.exists():
        raise ValueError(f"No gold scores found in {gold_scores_dir} for {model_subdir}")
    return gold_scores_filepath, agent_dir


def load_gold_scores(path_to_csv: str | Path) -> pd.DataFrame:
    gold_scores = pd.read_csv(path_to_csv)
    gold_scores.rename(columns={"score": "gold_score"}, inplace=True)
    gold_scores["task_id"] = gold_scores["task_id"].astype(str)

    # Ensure gold_scores has a domain; if not, assign a default.
    if "domain" not in gold_scores.columns:
        raise ValueError("Gold scores must have a domain column.")

    gold_scores = gold_scores.copy()
    gold_scores["domain_task_id"] = gold_scores["domain"].astype(str) + "_" + gold_scores["task_id"].astype(str)

    return gold_scores


def load_test_set_agrb(test_set_path: str) -> pd.DataFrame:
    test_set = pd.read_csv(test_set_path)
    # Get only "test" split
    test_set = test_set[test_set["split"] == "test"]
    test_set["task_id"] = test_set["task_id"].astype(str)
    return test_set


def load_eval_set(eval_set_path: str) -> set:
    """Load eval set file as a set of domain_task_id strings (e.g., 'shopping_0')."""
    eval_set = set()
    with open(eval_set_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                domain, task_id = parts[0], parts[1]
                eval_set.add(f"{domain}_{task_id}")
    return eval_set


def get_task_id_from_trace_file(file: Path | str) -> str:
    if not isinstance(file, Path):
        file = Path(file)

    # 29_k-0.html -> 29 ; 29_k-0.txt -> 29
    filename = file.stem

    task_id = re.sub(r"_k-.*$", "", filename)
    task_id = re.sub(r"conversation_", "", task_id, flags=re.IGNORECASE)
    return task_id


def get_variation_name_from_trace_file(file: Path | str) -> str:
    if not isinstance(file, Path):
        file = Path(file)

    return file.parent.parent.parent.name


# ==============================================================
# LINK Parsing of evaluation strings from a trace file
# ==============================================================
EVAL_SECTION_REGEX = re.compile(
    r"^\s*(?:#+\s*)?(?:EVALUATION:|VERDICT:|Status:|ANSWER:|#\s*EVALUATION)\s*(.*?)(?=\n+\s*(?:#+\s*)?(?:FEEDBACK:|FEEDBACK\n|EVALUATION:|VERDICT:|Status:|ANSWER:|#\s*EVALUATION|</pre>)|$)",
    # r"^\s*(?:#+\s*)?(?:EVALUATION:?|VERDICT:?|Status:?|ANSWER:?|#\s*EVALUATION)\s*(.*?)(?=\n+\s*(?:#+\s*)?(?:FEEDBACK:?|FEEDBACK\n|EVALUATION:?|VERDICT:?|Status:?|ANSWER:?|#\s*EVALUATION|</pre>)|$)",
    re.S | re.IGNORECASE | re.MULTILINE,
)

FEEDBACK_SECTION_REGEX = re.compile(
    r"^\s*FEEDBACK(?:\s*(?:[:\-—–])|\s*\(.*?\)\s*:)?\s*\n?(.*?)(?=\n\s*\n|\Z)",
    re.MULTILINE | re.DOTALL | re.IGNORECASE,
)
EVAL_CRITERIA_REGEX = re.compile(
    rf"^\s*:?\s*({'|'.join(map(re.escape, sorted(EVAL_LABELS, key=len, reverse=True)))})(?::)?",
    re.IGNORECASE,
)
GENERATION_BLOCK_REGEX = re.compile(
    r"-+\s*GENERATION(?:\s*\d+)?\s*-+[\s\S]*?(?=-+\s*GENERATION(?:\s*\d+)?\s*-+|\Z)",
    re.IGNORECASE,
)
THINKING_BLOCK_REGEX = re.compile(
    r"^\s*-{3,}\s*THOUGHT\s+START\s*-{3,}\s*$.*?^\s*-{3,}\s*THOUGHT\s+END\s*-{3,}\s*$",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)


def _is_infeasible(feedback: str) -> bool:
    feedback = clean_spaces(feedback)
    # Look for feedback that contains only "NA" or "NONE" (with optional whitespace/punctuation)
    match = re.search(r"^\s*(NONE|NA)\s*[.,;!?]?\s*$", feedback, re.IGNORECASE)
    return match is not None


def _parse_eval(content: str, eval_template: str | None = None) -> str:
    """
    Extracts the evaluation score (e.g., SUCCESS, FAILURE) or a numeric value from the EVALUATION section.
    Returns (categorical, numeric) where one may be None/empty.
    """
    content = content.strip("[]").strip("<>").strip()
    # Remove a leading bullet like "- SUCCESS" or "• FAILURE"
    content = re.sub(r"^\s*[-•]\s*", "", content)
    # Remove common label prefixes (after bullet removal), e.g., "Result:", "Rating:", "Verdict:", etc.
    content = re.sub(r"^(?:RESULT|RATING|SCORE|VALUE|VERDICT|OUTCOME|LABEL)\s*:?\s*", "", content, flags=re.IGNORECASE)

    fallback_labels = [
        "SUCCESS WITH ROOM FOR IMPROVEMENT",
        "SUCCESS",
        "PARTIAL SUCCESS",
        "FAILURE",
        "PARTIAL FAILURE",
        "UNCERTAIN",
        "UNCLEAR",
    ]
    labels_pattern = rf"{'|'.join(map(re.escape, fallback_labels))}"
    # Use case-insensitive search for the fallback labels and normalize the match.
    fallback = re.search(labels_pattern, content, flags=re.IGNORECASE)
    if fallback:
        # Normalize: strip whitespace, uppercase, and remove trailing period if present.
        matched = fallback.group(0).strip()
        _eval = matched.upper().rstrip(".")
        if _eval == "FULLY ACCOMPLISHED":
            return "SUCCESS"
        elif _eval == "PARTIALLY ACCOMPLISHED":
            return "PARTIAL SUCCESS"
        return _eval

    # First, check for comma-separated T/F format (for four_statement_unc_letter_t_f template)
    if eval_template == "four_statement_unc_letter_t_f":
        # Pattern 1: Standard "T or F, T or F, ..." format (5 comma-separated T/F values)
        tf_pattern = re.match(
            r"^\s*([TF]|TRUE|FALSE)\s*(?:or\s+[TF])?\s*,\s*([TF]|TRUE|FALSE)\s*(?:or\s+[TF])?\s*,\s*([TF]|TRUE|FALSE)\s*(?:or\s+[TF])?\s*,\s*([TF]|TRUE|FALSE)\s*(?:or\s+[TF])?\s*,\s*([TF]|TRUE|FALSE)\s*(?:or\s+[TF])?\s*$",
            content.strip(),
            re.IGNORECASE,
        )
        if tf_pattern:
            # Extract just the T/F values (first captured group from each set)
            values = [tf_pattern.group(i) for i in range(1, 6)]
            return ", ".join(values).upper()

        # Pattern 2: Model uses letters (A, B, C, D, E) mixed with F's (e.g., "F, F, C, F, F")
        # This means the model marked which statement is correct using the letter
        letter_pattern = re.match(
            r"^\s*([A-F]|TRUE|FALSE)\s*,\s*([A-F]|TRUE|FALSE)\s*,\s*([A-F]|TRUE|FALSE)\s*,\s*([A-F]|TRUE|FALSE)\s*,\s*([A-F]|TRUE|FALSE)\s*$",
            content.strip(),
            re.IGNORECASE,
        )
        if letter_pattern:
            values = [letter_pattern.group(i).upper() for i in range(1, 6)]
            # Convert to T/F format based on position
            # A should be at position 0, B at 1, C at 2, D at 3, E at 4
            # If we see a letter that matches its position, mark as T; otherwise F
            result = []
            letter_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
            for i, val in enumerate(values):
                if val in letter_map and letter_map[val] == i:
                    # This letter is in its correct position, mark as T
                    result.append("T")
                elif val == "F" or val == "FALSE":
                    result.append("F")
                elif val == "T" or val == "TRUE":
                    result.append("T")
                else:
                    # Letter in wrong position - this shouldn't happen normally
                    # but if it does, treat as F
                    result.append("F")
            return ", ".join(result)

    # Try categorical first
    match = EVAL_CRITERIA_REGEX.search(content)
    if match:
        return match.group(1).upper()

    # Try numeric (int or float)
    num_match = re.match(r"^\s*:?-?\s*([0-9]+(?:\.[0-9]+)?)", content)
    if num_match:
        return str(num_match.group(1))

    return ""


def _sanitize_content(content: str) -> str:
    """Sanitize the content by removing unwanted characters or patterns."""
    # Remove thinking blocks
    content = THINKING_BLOCK_REGEX.sub("", content)
    # Remove CONTENT TYPE: reasoning
    content = re.sub(r"CONTENT TYPE:\s*reasoning\s*", "", content, flags=re.IGNORECASE)
    # Remove CONTENT TYPE: text
    content = re.sub(r"CONTENT TYPE:\s*text\s*", "", content, flags=re.IGNORECASE)

    return content.strip()


def _sanitize_gen_block(gen_block: str):
    gen_block = re.sub(r"##+", "", gen_block)  # Remove ##, ###, etc markers
    gen_block = re.sub(r"\*\*", "", gen_block)  # Remove ** markers
    gen_block = re.sub(r"\t", " ", gen_block)  # normalize tabs to spaces
    gen_block = re.sub(r"\n{2,}", "\n", gen_block)
    gen_block = re.sub(r"\n\s+(\S)", r"\n\1", gen_block)
    gen_block = clean_spaces(gen_block)

    return gen_block


def _extract_assistant_final_message_from_html(html_content: str) -> str:
    """Extract the last assistant message using a simple regex.

    Strategy:
    - Find all occurrences of an ASSISTANT role followed by a content <pre>...</pre> block and take the last one's inner text.
    - If none found, fall back to the last <pre> anywhere in the document.
    - Return a plain-text (HTML-unescaped) string.
    """
    # Match: <div class="role">ASSISTANT</div> ... <div class="content"> ... <pre>...</pre>
    pattern = re.compile(
        r"<div[^>]*class=\s*\"role\"[^>]*>\s*ASSISTANT\s*</div>.*?"
        r"<div[^>]*class=\s*\"content\"[^>]*>.*?<pre[^>]*>(.*?)</pre>",
        re.IGNORECASE | re.DOTALL,
    )
    matches = pattern.findall(html_content)
    if matches:
        return _html.unescape(matches[-1])

    # Fallback: last <pre>...</pre>
    pre_blocks = list(re.finditer(r"<pre[^>]*>(.*?)</pre>", html_content, re.IGNORECASE | re.DOTALL))
    if pre_blocks:
        raw = pre_blocks[-1].group(1)
        return _html.unescape(raw)

    return ""


def _extract_evaluations_from_gen(gen_blocks: list[str]) -> list[str]:
    """Return list containing only the last EVALUATION section from each GENERATION block."""
    eval_sections: list[str] = []
    for block in gen_blocks:
        block = _sanitize_gen_block(block)
        all_evals = EVAL_SECTION_REGEX.findall(block)
        if all_evals:
            # keep only the last evaluation in the block (ignores placeholders)
            raw_eval = all_evals[-1]
            cleaned_eval = re.sub(r"^\s*(?:Score|Value|Rating|Result)\s*:?\s*", "", raw_eval.strip(), flags=re.IGNORECASE)
            cleaned_eval = cleaned_eval.strip("[]").strip("<>").strip("'").strip('"').strip()

            # Clean up the evaluation content by removing common prefixes
            eval_sections.append(cleaned_eval)
    return eval_sections


def _extract_feedback_from_gen(gen_blocks: list[str]) -> list[str]:
    """Return list containing only the last FEEDBACK section from each GENERATION block."""
    feedback_sections: list[str] = []
    for block in gen_blocks:
        block = _sanitize_gen_block(block)
        all_feedbacks = FEEDBACK_SECTION_REGEX.findall(block)
        if all_feedbacks:
            raw_feedback = all_feedbacks[-1]
            cleaned_feedback = raw_feedback.strip("[]").strip("<>").strip("'").strip('"').strip()
            feedback_sections.append(cleaned_feedback)
    return feedback_sections


def _print_eval_parsing_msg(message: str, file: Path, gen_blocks: list[str] | None = None):
    gen_blocks_str = "\nContent:" + "\n".join(gen_blocks) if gen_blocks else ""
    print(f"{message} File: {file}{gen_blocks_str}", flush=True)


def maybe_delete_invalid_files(file: Path | str, delete_invalid_files: bool = DELETE_INVALID_FILES):
    if not delete_invalid_files:
        return
    # Split the extension from the file.
    if not isinstance(file, Path):
        file = Path(file)
    base_path = str(file.with_suffix(""))

    all_files = []
    all_files.append(f"{base_path}.html")
    all_files.append(f"{base_path}.txt")
    parent_dir = Path(file).parent.parent
    all_files.append(f"{parent_dir}/usage/{base_path}.csv")
    for file in all_files:
        if not os.path.exists(file):
            continue
        if os.path.isfile(file):
            os.remove(file)


def get_eval_from_trace_file(file: Path | str) -> tuple[str, str, list[str] | None, str, Path]:
    if not isinstance(file, Path):
        file = Path(file)

    task_id = get_task_id_from_trace_file(file)
    variation_name = get_variation_name_from_trace_file(file)
    domain = get_domain_from_path(file)
    try:
        with open(file, "r", encoding="utf-8") as f:
            content = f.read().strip()
            is_html_file = str(file).lower().endswith(".html") or "</html>" in content.lower() or "</pre>" in content.lower()
            if is_html_file:
                # Extract the assistant's final message from HTML and treat it as a single generation block
                extracted = _extract_assistant_final_message_from_html(content)
                # Keep the raw extracted block for section parsing; sanitize lightly to remove thought blocks etc.
                extracted = _sanitize_content(extracted)
                gen_blocks = [extracted] if extracted else []
            else:
                content = _sanitize_content(content)
                # Extract evaluation sections only from inside GENERATION blocks.
                gen_blocks = GENERATION_BLOCK_REGEX.findall(content)
            matches = _extract_evaluations_from_gen(gen_blocks)
            feedbacks = _extract_feedback_from_gen(gen_blocks)
            if not matches:
                _print_eval_parsing_msg("\nError in eval parsing: No evaluation found.", file, gen_blocks)
                maybe_delete_invalid_files(file)
                return task_id, variation_name, None, domain, file

            # Parse all eval results from all EVALUATION sections
            eval_results = []

            for i, eval_section_str in enumerate(matches):
                eval_template = extract_eval_template_from_variation_name(variation_name)
                eval_result = _parse_eval(eval_section_str, eval_template=eval_template)
                score = map_eval_to_score_with_context(eval_result, eval_template=eval_template)
                if score is None:
                    _print_eval_parsing_msg("\nError in eval parsing: Score mapping returned None.", file, gen_blocks)
                    maybe_delete_invalid_files(file)
                    return task_id, variation_name, None, domain, file

                # Ensure feedbacks list has an entry for this generation index to avoid index errors
                if len(feedbacks) < i + 1:
                    if score == 1:
                        feedbacks.append("NA")
                    else:
                        _print_eval_parsing_msg("Error in eval parsing: Score!=1 & no feedback section.", file, gen_blocks)
                        maybe_delete_invalid_files(file)
                        return task_id, variation_name, None, domain, file

                if score != 1:
                    if _is_infeasible(feedbacks[i]):
                        eval_result = "INFEASIBLE"
                        # _print_eval_parsing_msg(
                        #     f"eval parsing: Mapping to infeasible: eval_result: {eval_result}, feedback {feedbacks[i]}", file, gen_blocks
                        # )

                if eval_result:
                    eval_results.append(eval_result)

            if not eval_results:
                _print_eval_parsing_msg("Error in eval parsing: No valid eval results found.", file, gen_blocks)
                maybe_delete_invalid_files(file)
                return task_id, variation_name, None, domain, file

            # Take the most frequent answer as the final answer
            # majority_vote = get_majority_eval(eval_results)

            return task_id, variation_name, eval_results, domain, file
    except Exception as e:
        print(f"Error reading file {file}: {e}", flush=True)
        return task_id, variation_name, None, domain, file


# =============================================================
# LINK Parsing all trace files in an experiment directory
# =============================================================


def remove_files_to_parse_if_exists(files_to_parse: list[Path], existing_results_df: pd.DataFrame) -> list[Path]:
    variation_names = [get_variation_name_from_trace_file(file) for file in files_to_parse]
    task_ids = [get_task_id_from_trace_file(file) for file in files_to_parse]
    domains = [get_domain_from_path(file) for file in files_to_parse]
    unique_ids = [f"{domain}_{task_id}" for domain, task_id in zip(domains, task_ids)]

    final_files_to_parse = []

    # Create lookup dictionaries for faster access
    existing_values = {
        (str(row["domain_task_id"]), col): row[col]
        for _, row in existing_results_df.iterrows()
        for col in existing_results_df.columns
        if col not in ["domain_task_id", "source", "gold_score", "env", "domain", "task_id"]
    }

    # Create a list of files that need to be parsed
    for file, variation_name, uid in zip(files_to_parse, variation_names, unique_ids):
        if (uid, variation_name) in existing_values:
            val = existing_values[(uid, variation_name)]
            if val and pd.notna(val):
                continue
        final_files_to_parse.append(file)

    return final_files_to_parse


def _results_to_df(results: dict) -> pd.DataFrame:
    # Convert the nested dictionary to a vertical DataFrame
    rows = []
    for domain_task_id, variations in results.items():
        for variation_name, data in variations.items():
            row = {
                "domain_task_id": domain_task_id,
                "domain": data["domain"],
                "task_id": data["task_id"],
                "config_name": variation_name,
                "eval": data["eval_result"],
            }
            # Optionally include evaluation probability and the number of generations if present
            if "success_prob" in data:
                row["success_prob"] = data["success_prob"]
            if "num_generations" in data:
                row["num_generations"] = data["num_generations"]
            rows.append(row)

    result_df = pd.DataFrame(rows)
    return result_df


def compute_success_prob(eval_results: list[str]) -> float:
    num_fail = sum(1 for result in eval_results if result.upper() in {"FAILURE", "PARTIAL FAILURE", "PARTIAL SUCCESS"})
    num_success = sum(1 for result in eval_results if result.upper() in {"SUCCESS", "INFEASIBLE"})
    total = num_fail + num_success
    if total == 0:
        return np.nan
    return num_success / total


def get_majority_eval(eval_results: list[str], agg="mode", eval_template="") -> str:
    # If drop_int, randomly drop that many evaluations from eval_results
    if len(eval_results) == 0:
        return ""
    if len(eval_results) == 1:
        return eval_results[-1]

    if agg == "mode":
        counter = Counter(eval_results)
        most_common_eval, n = counter.most_common(1)[0]
        return most_common_eval

    elif agg == "min":
        # FIXME: make more generic
        min_label = ""
        eval_results_set = set(eval_results)
        if "FAILURE" in eval_results_set:
            return "FAILURE"
        elif "PARTIAL FAILURE" in eval_results_set:
            return "PARTIAL FAILURE"
        elif "PARTIAL SUCCESS" in eval_results_set:
            return "PARTIAL SUCCESS"
        elif "SUCCESS" in eval_results_set or "INFEASIBLE" in eval_results_set:
            return "SUCCESS"
        return min_label

    # Default fallback to avoid returning None on unrecognized 'agg'
    return eval_results[-1] if eval_results else ""


def parse_experiment_results(
    base_path,
    existing_results_df: pd.DataFrame | None = None,
):
    num_invalid_files = 0
    print(f"Parsing evaluations in {base_path}", flush=True)

    files_to_parse = list(Path(base_path).glob(FILE_TO_PARSE_PATTERN))

    if not IGNORE_HTML:
        files_to_parse_html = list(Path(base_path).glob(FILE_TO_PARSE_PATTERN_HTML))
    else:
        files_to_parse_html = []

    files_to_parse = [file for file in files_to_parse if not any(exclude in str(file) for exclude in EXCLUDE_STRS)]
    files_to_parse_html = [file for file in files_to_parse_html if not any(exclude in str(file) for exclude in EXCLUDE_STRS)]
    if MUST_INCLUDE_STRS:
        # Convert wildcard patterns to regex patterns
        regex_patterns = [re.compile(pattern.replace("*", ".*")) for pattern in MUST_INCLUDE_STRS]
        files_to_parse = [file for file in files_to_parse if any(regex.search(str(file)) for regex in regex_patterns)]
        files_to_parse_html = [file for file in files_to_parse_html if any(regex.search(str(file)) for regex in regex_patterns)]

    files_to_parse_search = set(re.sub(r"\.txt$", "", str(file)) for file in files_to_parse)
    # If file is in the files_to_parse already, ignore; else, add it.
    for file in files_to_parse_html:
        file_base_path = re.sub(r"\.html$", "", str(file))
        if not file_base_path:
            continue
        if file_base_path in files_to_parse_search:
            continue
        else:
            files_to_parse.append(file)

    if existing_results_df is not None:
        files_to_parse = remove_files_to_parse_if_exists(files_to_parse, existing_results_df)

    if len(files_to_parse) == 0:
        print(f"No files to parse in {base_path}", flush=True)
        return None

    else:
        print(f"Parsing {len(files_to_parse)} files in {base_path}", flush=True)

    results = {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        # Submit all file processing tasks concurrently.
        futures = {executor.submit(get_eval_from_trace_file, file): file for file in files_to_parse}
        for future in concurrent.futures.as_completed(futures):
            task_id, variation_name, eval_result, domain, file = future.result()
            if not eval_result:
                num_invalid_files += 1
                print(f"Not able to parse eval result for {file}")
                continue

            domain_task_id = f"{domain}_{task_id}"
            results.setdefault(domain_task_id, {})

            if variation_name not in results[domain_task_id]:
                results[domain_task_id][variation_name] = {
                    "eval_results": [],
                    "domain": domain,
                    "task_id": task_id,
                }
            if eval_result:
                results[domain_task_id][variation_name]["eval_results"].extend(eval_result)

    if not results:
        return existing_results_df

    # If multiple generations, handle majority vote and individual results
    final_results = {}
    for domain_task_id, data in results.items():
        final_results.setdefault(domain_task_id, {})
        for variation_name, variation_data in data.items():
            final_results[domain_task_id].setdefault(variation_name, {})

            # eval_template = extract_eval_template_from_variation_name(variation_name)
            n_generations = len(variation_data["eval_results"]) if "eval_results" in variation_data else 0
            s_prob = compute_success_prob(variation_data["eval_results"])

            final_results[domain_task_id][variation_name] = {
                "eval_result": get_majority_eval(variation_data["eval_results"], "mode"),
                "domain": variation_data["domain"],
                "task_id": variation_data["task_id"],
                # Add metadata for downstream consolidated files
                "num_generations": n_generations,
                "success_prob": s_prob,
            }
            if n_generations > 4:
                final_results[domain_task_id][f"{variation_name}-s_prob"] = {
                    "eval_result": str(s_prob),
                    "domain": variation_data["domain"],
                    "task_id": variation_data["task_id"],
                    # Add metadata for downstream consolidated files
                    "num_generations": n_generations,
                    "success_prob": s_prob,
                }

            if len(variation_data["eval_results"]) > 1:
                final_results[domain_task_id][f"{variation_name}-min"] = {
                    "eval_result": get_majority_eval(variation_data["eval_results"], "min"),
                    "domain": variation_data["domain"],
                    "task_id": variation_data["task_id"],
                }
                eval_results_dropped = variation_data["eval_results"][:]
                for i in range(1, len(variation_data["eval_results"]) - 1):
                    # Randomly drop i samples and compute majority vote
                    if len(eval_results_dropped) <= i:
                        break
                    random_indices = np.random.choice(len(eval_results_dropped), size=i, replace=False)
                    for idx in sorted(random_indices, reverse=True):
                        del eval_results_dropped[idx]
                    num_samples = len(eval_results_dropped)
                    final_results[domain_task_id][f"{variation_name}-min-{num_samples}"] = {
                        "eval_result": get_majority_eval(eval_results_dropped, "min"),
                        "domain": variation_data["domain"],
                        "task_id": variation_data["task_id"],
                    }
                # For each sample, also compute the evaluation result.
                n = len(variation_data["eval_results"])
                for i in range(n):
                    variation_name_i = f"{variation_name}-{i}"
                    cat = variation_data["eval_results"][i] if i < len(variation_data["eval_results"]) else ""
                    final_results[domain_task_id][variation_name_i] = {
                        "eval_result": cat,
                        "domain": variation_data["domain"],
                        "task_id": variation_data["task_id"],
                    }
            del variation_data["eval_results"]

    # # Merge additional data
    # for domain_task_id, additional_variations in additional_data.items():
    #     if domain_task_id not in results:
    #         results[domain_task_id] = {}
    #     results[domain_task_id].update(additional_variations)

    # Convert dictionary to DataFrame after processing all subdirs
    result_df = _results_to_df(final_results)

    if existing_results_df is not None:
        result_df = result_df.set_index("domain_task_id").combine_first(existing_results_df.set_index("domain_task_id")).reset_index()

    print(f"Number of invalid files: {num_invalid_files} of {len(files_to_parse)} for {base_path}", flush=True)
    return result_df


# ==============================================================
# LINK: Statistics computation
# ==============================================================


def _mcnemar(b: int, c: int, apply_correction: bool = False) -> float:
    b = int(b)
    c = int(c)
    check_valid = lambda n: isinstance(n, int) or (isinstance(n, float) and n.is_integer())
    if not all(map(check_valid, [b, c])):
        print(f"b: {b}, c: {c}")
        print(check_valid(b), check_valid(c))
        raise ValueError("b and c must be integers!")
    n_min, n_max = sorted([b, c])

    if (n_min + n_max) >= 25:
        chi2_statistic = (abs(n_min - n_max) - int(apply_correction)) ** 2 / (n_min + n_max)
        pvalue = chi2.sf(chi2_statistic, 1)
    else:
        pvalue = 2 * binom.cdf(n_min, n_min + n_max, 0.5) - binom.pmf(n_min, n_min + n_max, 0.5)
    return float(pvalue)


def _compute_distance_skewness(X, theta):
    X = np.array(X)
    # pairwise distances between the elements of X
    pairwise_distances = np.abs(np.subtract.outer(X, X))

    # numerator and denominator of the distance skewness formula
    numerator = np.sum(pairwise_distances)
    denominator = np.sum(np.abs(np.add.outer(X, X) - 2 * theta))

    # handle the case when Pr(X=theta) = 1
    if denominator == 0:
        return 0
    else:
        return 1 - numerator / denominator


def _compute_skewness(X):
    X = np.array(X)
    try:
        return sum([ele**3 for ele in X]) / len(X) / (sum([ele**2 for ele in X]) / len(X)) ** (3 / 2)
    except Exception as _:
        print(f"Error computing skewness for {X}")
        return np.nan


def compute_all_stats(gold_scores: pd.DataFrame, evals: pd.DataFrame, eval_template: str | None = None):
    # Convert task_id to string
    gold_scores.loc[:, "domain_task_id"] = gold_scores["domain_task_id"].astype(str)
    evals.loc[:, "domain_task_id"] = evals["domain_task_id"].astype(str)

    evals = evals.copy()

    # Use context-aware mapping; if mapper return None, coerce to NaN/numeric for downstream math
    evals["predicted_score"] = pd.to_numeric(evals["eval"].apply(lambda x: map_eval_to_score_with_context(x, eval_template)), errors="coerce")

    # Merge evals with the ground-truth scores on task_id
    merged = pd.merge(evals, gold_scores, on="domain_task_id", how="right", suffixes=("_eval", "_true"))
    merged["gold_score"] = pd.to_numeric(merged["gold_score"], errors="coerce")

    # Compute confusion matrix values based on the comparison of predicted vs. true
    false_positive = ((merged["predicted_score"] == 1) & (merged["gold_score"] == 0)).sum()
    false_negative = ((merged["predicted_score"] == 0) & (merged["gold_score"] == 1)).sum()
    true_positive = ((merged["predicted_score"] == 1) & (merged["gold_score"] == 1)).sum()
    true_negative = ((merged["predicted_score"] == 0) & (merged["gold_score"] == 0)).sum()
    num_pos = true_positive + false_negative
    num_neg = true_negative + false_positive

    all_diff = merged["predicted_score"] - merged["gold_score"]
    all_diff = all_diff.dropna()
    signed_bias = all_diff.mean()
    # Bias and distance skewness computed on subsets conditioned by gold_score
    unsigned_bias = abs(all_diff).mean()
    skewness = _compute_skewness(all_diff)
    distance_skewness = _compute_distance_skewness(all_diff, 0)

    tp_ratio = true_positive / (true_positive + false_negative)
    tn_ratio = true_negative / (false_positive + true_negative)
    fp_ratio = false_positive / (false_positive + true_negative)
    fn_ratio = false_negative / (true_positive + false_negative)
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    precision = true_positive / (true_positive + false_positive)
    f1_score = 2 * (precision * tp_ratio) / (precision + tp_ratio)

    f1_pos = 2 * true_positive / (2 * true_positive + false_positive + false_negative)
    f1_neg = 2 * true_negative / (2 * true_negative + false_positive + false_negative)
    f1_balanced = (f1_pos * num_pos + f1_neg * num_neg) / (num_pos + num_neg)
    f1_balanced_inv_weights = (f1_pos * num_neg + f1_neg * num_pos) / (num_pos + num_neg)

    # --- Add Balanced Accuracy and MCC ---
    # Balanced Accuracy
    balanced_accuracy = (tp_ratio + tn_ratio) / 2

    # Matthews Correlation Coefficient (MCC)
    denominator = np.sqrt((true_positive + false_positive) * (true_positive + false_negative) * (true_negative + false_positive) * (true_negative + false_negative))
    if denominator == 0:
        mcc = np.nan
    else:
        mcc = ((true_positive * true_negative) - (false_positive * false_negative)) / denominator
    # --- end addition ---

    # Added counts for SUCCESS, PARTIAL SUCCESS, and FAILURE based on the "eval" column.
    criteria_counts = {}
    for criteria in EVAL_LABELS:
        criteria_count = evals["eval"].str.upper().eq(criteria).sum()
        criteria_counts[criteria] = criteria_count

    # Filter the false negatives (i.e. cases where gold_score==1 but predicted 0)
    # Cast merged["eval"] to string and strip whitespace.
    merged["eval_upper"] = merged["eval"].astype(str).str.upper().str.strip()
    false_negatives = merged[(merged["predicted_score"] == 0) & (merged["gold_score"] == 1)]
    false_neg_partial_success = false_negatives[false_negatives["eval_upper"] == "PARTIAL SUCCESS"].shape[0]
    false_neg_failure = false_negatives[false_negatives["eval_upper"] == "FAILURE"].shape[0]

    total = len(gold_scores)
    na_count = total - false_positive - false_negative - true_positive - true_negative
    total_effectively_completed = total - na_count
    mcnemar_p_value = _mcnemar(false_positive, false_negative, apply_correction=True)

    all_data = {
        "tp_ratio": tp_ratio * 100,
        "tn_ratio": tn_ratio * 100,
        "accuracy": accuracy * 100,
        "precision": precision * 100,
        "f1_score": f1_score * 100,
        "distance_skewness": distance_skewness * 100,
        "signed_bias": signed_bias * 100,
        "total_effectively_completed": total_effectively_completed,
        "NA %": 100 * na_count / total if total > 0 else np.nan,
        "skewness": skewness * 100,
        "unsigned_bias": unsigned_bias * 100,
        "fn_ratio": fn_ratio,
        "fp_ratio": fp_ratio,
        "total": total,
        "f1_pos": f1_pos,
        "f1_neg": f1_neg,
        "f1_balanced": f1_balanced,
        "f1_balanced_inv_weights": f1_balanced_inv_weights,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "balanced_accuracy": balanced_accuracy,
        "mcnemar_p_value": mcnemar_p_value,
        "mcc": mcc,
        "false_neg_partial_success": false_neg_partial_success,
        "false_neg_failure": false_neg_failure,
        "NA": na_count,
        "perc failure": (true_negative + false_positive) / total_effectively_completed if total_effectively_completed > 0 else np.nan,
    }

    all_data.update(criteria_counts)

    return all_data


# ===============================================================
# Main
# ===============================================================
if __name__ == "__main__":
    all_evals = pd.DataFrame()
    confusion_per_env = {}
    raw_evals_per_env = {}
    all_evals_per_env = {}

    # 0) Process each env and create consolidated CSVs for each env with statistics and raw model evaluations (likerts, scores, etc)
    for env in ENVS_TO_PROCESS:
        out_dir_env = Path(OUTPUT_DIR) / env
        confusion_matrices = pd.DataFrame()
        all_evals_env = pd.DataFrame()
        model_subdirs = []

        for experiments_path in EXPERIMENTS_PATHS:
            experiments_path_env = Path(experiments_path) / env
            agent_subdirs = [str(s) for s in Path(experiments_path_env).iterdir() if s.is_dir()]
            model_subdirs.extend([subdir for agent_subdir in agent_subdirs for subdir in Path(agent_subdir).iterdir() if subdir.is_dir()])

        model_subdirs = [str(d) for d in model_subdirs if not any(exclude in str(d) for exclude in EXCLUDE_STRS)]

        for model_subdir in model_subdirs:
            if any(exclude in model_subdir for exclude in EXCLUDE_STRS) or not any(env in model_subdir for env in ENVS_TO_PROCESS):
                continue
            try:
                gold_scores_filepath, source = get_gold_score_path_source(Path(model_subdir).parent, GOLD_SCORES_DIRS[env], env)
                gold_scores = load_gold_scores(gold_scores_filepath)
                gold_scores["gold_source"] = source
            except ValueError:
                print(f"No gold scores found in {GOLD_SCORES_DIRS[env]} for {model_subdir}")
                continue

            if not OVERWRITE and (Path(model_subdir) / EVALUATIONS_CSV_NAME).exists():
                existing_results = pd.read_csv(Path(model_subdir) / EVALUATIONS_CSV_NAME)
                if existing_results.empty:
                    existing_results = None
            else:
                existing_results = None

            # Get the model name from the subdirectory.
            model = Path(model_subdir).name + "__" + Path(model_subdir).parent.name

            model_results_all_domains = parse_experiment_results(base_path=model_subdir, existing_results_df=existing_results)
            if model_results_all_domains is None:
                continue

            # Add model and env columns to the vertical data
            model_results_all_domains["model"] = model
            model_results_all_domains["env"] = env

            # Merge with gold scores
            # Merge gold metadata (including optional traj_len) into the vertical results
            _gold_merge_cols = ["domain_task_id", "gold_score", "gold_source"]
            if "traj_len" in gold_scores.columns:
                _gold_merge_cols.append("traj_len")
            model_results_all_domains = pd.merge(
                model_results_all_domains,
                gold_scores[_gold_merge_cols],
                on="domain_task_id",
                how="left",
            )

            # Append to all_evals_env for this environment
            all_evals_env = pd.concat([all_evals_env, model_results_all_domains], ignore_index=True)

            # For backward compatibility, still create consolidated_evals for save_csv
            consolidated_evals = model_results_all_domains.pivot_table(index="domain_task_id", columns="config_name", values="eval", aggfunc="first").reset_index()

            # Append "_eval" to the config column names
            config_cols = [col for col in consolidated_evals.columns if col != "domain_task_id"]
            rename_dict = {col: f"{col}_eval" for col in config_cols}
            consolidated_evals.rename(columns=rename_dict, inplace=True)

            # Add back metadata columns for save_csv
            consolidated_evals = pd.merge(
                consolidated_evals,
                model_results_all_domains[["domain_task_id", "domain", "task_id", "env"]].drop_duplicates(),
                on="domain_task_id",
                how="left",
            )

            # Add experiments_path column
            consolidated_evals["experiments_path"] = model_subdir

            # Saves `evaluations.csv` at model subdirectory.
            save_path = Path(model_subdir) / EVALUATIONS_CSV_NAME
            save_csv(consolidated_evals, save_path, gold_scores=gold_scores)
            print(f"Evaluation results consolidated for {model} at {save_path}")

            # Compute confusion stats using the vertical data
            for config_name in model_results_all_domains["config_name"].unique():
                config_data = model_results_all_domains[model_results_all_domains["config_name"] == config_name]

                # Extract evaluation template from config_name
                eval_template = extract_eval_template_from_variation_name(config_name)
                if eval_template:
                    print(f"Using template '{eval_template}' for config '{config_name}'")
                else:
                    print(f"No template found for config '{config_name}', using default mapping")

                # Prepare the evaluation subset for this config
                evals_subset = config_data[["domain_task_id", "eval", "task_id"]].copy()

                # Overall stats for the entire environment for this config
                overall_confusion_stats = compute_all_stats(gold_scores, evals_subset, eval_template)
                overall_col_name = f"{model}--{config_name}--all"
                confusion_matrices[overall_col_name] = pd.Series(overall_confusion_stats)

                # Stats per domain subset
                for domain in gold_scores["domain"].unique():
                    domain_evals_subset = config_data[config_data["domain"] == domain][["domain_task_id", "eval"]]
                    domain_gold_scores = gold_scores[gold_scores["domain"] == domain]
                    if not domain_evals_subset.empty:
                        domain_confusion_stats = compute_all_stats(domain_gold_scores, domain_evals_subset, eval_template)
                        domain_col_name = f"{model}--{config_name}--{domain}"
                        confusion_matrices[domain_col_name] = pd.Series(domain_confusion_stats)

                # Stats for eval_set_vwa subset
                if env == "vwa":
                    eval_set_vwa = load_eval_set(EVAL_SET_VWA_PATH)
                    gold_scores_vwa = gold_scores[gold_scores["domain_task_id"].isin(eval_set_vwa)]
                    evals_subset_vwa = evals_subset[evals_subset["domain_task_id"].isin(eval_set_vwa)]
                    if not gold_scores_vwa.empty and not evals_subset_vwa.empty:
                        vwa_confusion_stats = compute_all_stats(gold_scores_vwa, evals_subset_vwa, eval_template)
                        vwa_col_name = f"{model}--{config_name}--eval_set_vwa"
                        confusion_matrices[vwa_col_name] = pd.Series(vwa_confusion_stats)

                if env == "agrb_vwa":
                    test_set_agrb = load_test_set_agrb(TEST_SET_AGRB_PATH)
                    gold_scores["task_id"] = gold_scores["task_id"].apply(lambda x: x.replace(".resized.", "."))
                    evals_subset["task_id"] = evals_subset["task_id"].apply(lambda x: x.replace(".resized.", "."))

                    gold_scores_agrb = gold_scores[gold_scores["task_id"].isin(test_set_agrb["task_id"])]
                    evals_subset_agrb = evals_subset[evals_subset["task_id"].isin(test_set_agrb["task_id"])]
                    if not gold_scores_agrb.empty and not evals_subset_agrb.empty:
                        agrb_confusion_stats = compute_all_stats(gold_scores_agrb, evals_subset_agrb, eval_template)
                        agrb_col_name = f"{model}--{config_name}--test_set_agrb"
                        confusion_matrices[agrb_col_name] = pd.Series(agrb_confusion_stats)

        # Add a column for the model (LHS of '__') before saving all_evals_env.csv
        if not all_evals_env.empty and "model" in all_evals_env.columns:
            all_evals_env["model_name"] = all_evals_env["model"].apply(lambda x: x.split("__")[0] if isinstance(x, str) and "__" in x else x)

        # Add a numeric predicted score column mapped from the textual evaluation.
        # Use the config_name to detect an eval template when available.
        if not all_evals_env.empty and "eval" in all_evals_env.columns and "config_name" in all_evals_env.columns:
            try:
                # Map each eval string to a numeric score using context-aware mapping.
                # Use a list comprehension to avoid ambiguous typing for DataFrame.apply.
                all_evals_env["predicted_score"] = pd.to_numeric(
                    [
                        map_eval_to_score_with_context(
                            row.get("eval", ""),
                            extract_eval_template_from_variation_name(row.get("config_name", "")),
                        )
                        for _, row in all_evals_env.iterrows()
                    ],
                    errors="coerce",
                )
            except Exception:
                # Fall back to trying to map without template if something goes wrong.
                all_evals_env["predicted_score"] = pd.to_numeric([map_eval_to_score_with_context(x, None) for x in all_evals_env["eval"]], errors="coerce")

        all_evals_env.to_csv(out_dir_env / "all_evals_env.csv", index=False)
        print(f"All evals for {env} saved to", out_dir_env / "all_evals_env.csv")

        # Store the vertical data for this environment
        all_evals_per_env[env] = all_evals_env

        # === Convert the flat column names into a MultiIndex (domain, model, config) ===
        new_columns = []
        for col in confusion_matrices.columns:
            if "--" in col:
                parts = col.split("--")
                if len(parts) == 3:
                    # Note that parts are in order: model, config, domain.
                    # We want the MultiIndex in the order: (domain, model, config)
                    new_columns.append((parts[2], parts[0], parts[1]))
                else:
                    new_columns.append(("", col, ""))
        confusion_matrices.columns = pd.MultiIndex.from_tuples(new_columns, names=["domain", "model", "config"])
        # Optional: sort columns by the multi-index levels.
        confusion_matrices = confusion_matrices.sort_index(axis=1, level=["domain", "model", "config"])

        if env == "agrb_vwa":
            # Aggregate by env_domain, evaluator, config
            # get evaluator column gemini-2.5-flash-zero-random__agrb_claude-3.7 -> gemini-2.5-flash-zero-random

            # Filter for test cases only
            test_columns = [col for col in confusion_matrices.columns if "test" in str(col).lower()]
            confusion_matrices_test = confusion_matrices[test_columns]

            # Group by evaluator and config, then average the values
            grouped_data = {}
            for col in confusion_matrices_test.columns:
                domain, model, config = col
                # Extract evaluator from model (first part before '__')
                evaluator = model.split("__")[0] if isinstance(model, str) and "__" in model else model
                # Create new column key
                new_col = ("agrb_vwa_all", evaluator, config)

                if new_col not in grouped_data:
                    grouped_data[new_col] = []
                grouped_data[new_col].append(confusion_matrices_test[col])

            # Calculate averages for each group
            averaged_data = {}
            for new_col, data_list in grouped_data.items():
                if len(data_list) == 1:
                    averaged_data[new_col] = data_list[0]
                else:
                    # Average across the columns, handling NaN values
                    stacked_data = pd.concat(data_list, axis=1)
                    averaged_data[new_col] = stacked_data.mean(axis=1, skipna=True)

            # Create DataFrame with averaged data and add to existing confusion_matrices
            averaged_df = pd.DataFrame(averaged_data)
            averaged_df.columns = pd.MultiIndex.from_tuples(list(averaged_data.keys()), names=["domain", "model", "config"])

            # Concatenate the averaged data with the existing confusion_matrices
            confusion_matrices = pd.concat([confusion_matrices, averaged_df], axis=1)
            # Sort by the new multi-index levels
            confusion_matrices = confusion_matrices.sort_index(axis=1, level=["domain", "model", "config"])

        confusion_matrices.T.to_csv(out_dir_env / "consolidated_stats.csv", index=True)
        print(f"Offline stats for {env} saved to", out_dir_env / "consolidated_stats.csv")
        confusion_per_env[env] = confusion_matrices

    # Create consolidated CSVs with data for all environments
    # 1) Statistics for all environments in one CSV
    if confusion_per_env:
        # First, concatenate the confusion matrices from the different environments.
        consolidated_confusion = pd.concat(confusion_per_env, axis=1)

        # --- Merge only the env and domain levels ---
        # We assume the MultiIndex is structured as follows:
        # Level 0: env, Level 1: domain, Level 2: model, Level 3: config
        # Merge levels 0 and 1 (env and domain) into a single value.
        merged_env_domain = [f"{env}_{domain}" for env, domain in zip(consolidated_confusion.columns.get_level_values(0), consolidated_confusion.columns.get_level_values(1))]

        # Rebuild the MultiIndex with the merged env_domain level and preserve any remaining levels.
        if consolidated_confusion.columns.nlevels > 2:
            new_tuples = list(
                zip(
                    merged_env_domain,
                    consolidated_confusion.columns.get_level_values(2),
                    consolidated_confusion.columns.get_level_values(3),
                )
            )
            new_index = pd.MultiIndex.from_tuples(new_tuples, names=["env_domain", "model", "config"])
        else:
            new_index = pd.Index(merged_env_domain, name="env_domain")

        consolidated_confusion.columns = new_index

        # Transpose the DataFrame so that the merged env_domain (and remaining levels) become the row labels.
        transposed_confusion = consolidated_confusion.T
        transposed_output_path = Path(OUTPUT_DIR) / "consolidated_stats.csv"
        transposed_confusion.to_csv(transposed_output_path, index=True)
        print(f"Consolidated stats saved at {transposed_output_path}")

    else:
        print("No confusion matrices to consolidate.")

    # 2) Raw model evaluations (likerts, scores, etc) for all environments in one CSV
    if all_evals_per_env:
        all_evals_consolidated = pd.concat(all_evals_per_env.values(), ignore_index=True)
        all_evals_output_path = Path(OUTPUT_DIR) / "all_evals_consolidated.csv"
        all_evals_consolidated.to_csv(all_evals_output_path, index=False)
        print(f"All consolidated model evals saved at {all_evals_output_path}")
    else:
        print("No model evals to consolidate.")

import re
from typing import Callable

import numpy as np


def _make_continuous_eval_mapping(max_val: float, min_val: float, success_thr: float, uncertain_lower_thr: float, uncertain_upper_thr: float) -> Callable[[str], int | float | None]:
    """Factory returning a callable that maps a string response in [min_val,max_val] to score.

    Scoring bands:
        >= success_thr -> 1
        uncertain_lower_thr <= x < uncertain_upper_thr -> np.nan
        else -> 0 (covers partial-success / partial-failure / failure buckets collapsed to 0)
    'INFEASIBLE' bypasses and returns 1.
    """

    def _mapper(eval_response: str) -> int | float | None:
        if not eval_response:
            return None
        resp = eval_response.strip().upper()
        if resp == "INFEASIBLE":
            return 1
        try:
            val = float(resp)
        except ValueError:
            return None
        if not (min_val <= val <= max_val):
            return None
        if val >= success_thr:
            return 1
        if uncertain_lower_thr <= val < uncertain_upper_thr:
            return np.nan
        return 0

    return _mapper


def _create_numeric_mapping(success_range: tuple, failure_range: tuple, uncertain_range: tuple | None = None, partial_success_range: tuple | None = None) -> dict:
    """
    Helper function to create numeric mappings for 1-10 scale templates.

    Args:
        success_range: Range for success scores (inclusive)
        failure_range: Range for failure scores (inclusive)
        uncertain_range: Range for uncertain scores (inclusive)
        partial_success_range: Range for partial success scores (inclusive)

    Returns:
        Dictionary mapping string numbers to scores
    """
    mapping = {}

    # Map all values 1-10
    for i in range(1, 11):
        if success_range[0] <= i <= success_range[1]:
            mapping[str(i)] = 1
        elif failure_range[0] <= i <= failure_range[1]:
            mapping[str(i)] = 0
        elif uncertain_range and uncertain_range[0] <= i <= uncertain_range[1]:
            mapping[str(i)] = np.nan
        elif partial_success_range and partial_success_range[0] <= i <= partial_success_range[1]:
            mapping[str(i)] = 0
        else:
            # Default to uncertain for values not in any specified range
            mapping[str(i)] = np.nan

    return mapping


# ===============================================================
# Evaluation Criteria Definitions
# ===============================================================

EVAL_LABELS = [
    # Standard categorical responses
    "SUCCESS WITH ROOM FOR IMPROVEMENT",
    "SUCCESS",
    "PARTIAL SUCCESS",
    "FAILURE",
    "FAIL",
    "PARTIAL FAILURE",
    "UNCERTAIN",
    "UNCLEAR",
    # Letter-based responses (from four_statement_unc_letter, three_statement_unc_letter)
    "A",
    "B",
    "C",
    "D",
    "E",
    # True/False responses (from four_statement_unc_letter_t_f)
    "T",
    "F",
    "TRUE",
    "FALSE",
    # Roman numeral responses (from three_option_unc, four_option_unc_question)
    "I",
    "II",
    "III",
    "IV",
    "V",
    # Other possible responses
    "N",
    "NONE",
    "EXCELLENT",
    "GOOD",
    "FAIR",
    "POOR",
]


TEMPLATE_MAPPINGS: dict[str, dict[str, int | float | None] | Callable[[str], int | float | None]] = {
    "four_room": {
        "SUCCESS": 1,
        "SUCCESS WITH ROOM FOR IMPROVEMENT": 1,
        "PARTIAL SUCCESS": 0,
        "FAILURE": 0,
    },
    "four_room_2": {
        "SUCCESS": 1,
        "SUCCESS WITH ROOM FOR IMPROVEMENT": 1,
        "PARTIAL SUCCESS": 0,
        "FAILURE": 0,
    },
    "four_room_score": {
        "100": 1,
        "75": 1,
        "50": 0,
        "0": 0,
    },
    "four_room_score_2": {
        "100": 1,
        "75": 1,
        "50": 0,
        "0": 0,
    },
    "four_excellent": {
        "EXCELLENT": 1,
        "GOOD": 1,
        "FAIR": 0,
        "POOR": 0,
    },
    "four_excellent_2": {
        "EXCELLENT": 1,
        "GOOD": 1,
        "FAIR": 0,
        "POOR": 0,
    },
    "four_letter": {
        "A": 1,
        "B": 1,
        "C": 0,
        "D": 0,
    },
    "four_letter_2": {
        "A": 1,
        "B": 1,
        "C": 0,
        "D": 0,
    },
    "four_statement_unc_letter": {
        # A: all necessary -> SUCCESS
        # B: most necessary -> PARTIAL SUCCESS
        # C: part necessary -> PARTIAL FAILURE
        # D: mostly incorrect -> FAILURE
        # E: unclear -> UNCERTAIN
        "A": 1,
        "B": 0,
        "C": 0,
        "D": 0,
        "E": np.nan,
    },
    "three_statement_unc_letter": {
        # A: all necessary -> SUCCESS
        # B: most necessary -> PARTIAL SUCCESS
        # C: mostly incorrect -> FAILURE
        # D: unclear -> UNCERTAIN
        "A": 1,
        "B": 0,
        "C": 0,
        "D": np.nan,
    },
    "four_option_unc_question": {
        # I: very confident incorrect -> FAILURE
        # II: fairly confident incorrect -> FAILURE
        # III: uncertain -> UNCERTAIN
        # IV: fairly confident correct -> PARTIAL SUCCESS
        # V: very confident correct -> SUCCESS
        "I": 0,
        "II": 0,
        "III": np.nan,
        "IV": 0,
        "V": 1,
    },
    "three_option_unc": {
        # I: all necessary -> SUCCESS
        # II: most necessary -> PARTIAL SUCCESS
        # III: part necessary -> PARTIAL FAILURE
        "I": 1,
        "II": 0,
        "III": 0,
    },
    "four_statement_unc_letter_t_f": {
        # This template expects T/F responses for each statement
        # Default behavior: T=success, F=failure (context dependent)
        "T": 1,
        "F": 0,
        "TRUE": 1,
        "FALSE": 0,
    },
    # Additional templates from common_prompts.py
    "bin": {
        # Binary: SUCCESS/FAILURE only
        "SUCCESS": 1,
        "FAILURE": 0,
        "FAIL": 0,
    },
    "bin_unc": {
        # Binary with uncertainty: SUCCESS/FAILURE/UNCERTAIN
        "SUCCESS": 1,
        "FAILURE": 0,
        "UNCERTAIN": np.nan,
    },
    "tri": {
        # Tri-state: SUCCESS/PARTIAL SUCCESS/FAILURE
        "SUCCESS": 1,
        "PARTIAL SUCCESS": 0,
        "FAILURE": 0,
        "FAIL": 0,
    },
    "tri_unc": {
        # Tri-state with uncertainty: SUCCESS/PARTIAL SUCCESS/FAILURE/UNCERTAIN
        "SUCCESS": 1,
        "PARTIAL SUCCESS": 0,
        "FAILURE": 0,
        "UNCERTAIN": np.nan,
    },
    "four": {
        # Four-state: SUCCESS/PARTIAL SUCCESS/PARTIAL FAILURE/FAILURE
        "SUCCESS": 1,
        "PARTIAL SUCCESS": 0,
        "PARTIAL FAILURE": 0,
        "FAILURE": 0,
    },
    "four_unc": {
        # Four-state with uncertainty: SUCCESS/PARTIAL SUCCESS/PARTIAL FAILURE/FAILURE/UNCERTAIN
        "SUCCESS": 1,
        "PARTIAL SUCCESS": 0,
        "PARTIAL FAILURE": 0,
        "FAILURE": 0,
        "UNCERTAIN": np.nan,
    },
    "four_unclear": {
        "SUCCESS": 1,
        "PARTIAL SUCCESS": 0,
        "PARTIAL FAILURE": 0,
        "FAILURE": 0,
        "UNCLEAR": np.nan,
    },
    # Numeric templates from common_prompts.py
    "bin_num_unc": _create_numeric_mapping(
        success_range=(7, 10),  # confident correct
        failure_range=(1, 4),  # confident incorrect
        uncertain_range=(5, 6),  # uncertain
    ),
    "tri_num_accomplished_inv": _create_numeric_mapping(
        success_range=(1, 4),  # all necessary (inverted)
        failure_range=(7, 10),  # mostly incorrect (inverted)
        # 5-6: most necessary -> partial (treated as failure/0, not uncertain)
        partial_success_range=(5, 6),
    ),
    "tri_num_accomplished": _create_numeric_mapping(
        success_range=(7, 10),  # most/all necessary
        failure_range=(1, 6),  # mostly incorrect + part necessary
    ),
    "four_num_unc": _create_numeric_mapping(
        success_range=(7, 10),  # fairly/very confident correct
        failure_range=(1, 4),  # fairly/very confident incorrect
        uncertain_range=(5, 6),  # uncertain
    ),
    "four_num_unc_accomplished_inv": _create_numeric_mapping(
        # 1-4: confident/correct bands -> success
        success_range=(1, 4),
        # 5-6: uncertain
        uncertain_range=(5, 6),
        # 7-10: incorrect bands -> failure
        failure_range=(7, 10),
    ),
    "four_num_unc_accomplished": _create_numeric_mapping(
        # 9-10: success
        success_range=(9, 10),
        # 7-8: partial success (treated as 0)
        partial_success_range=(7, 8),
        # 5-6: uncertain
        uncertain_range=(5, 6),
        # 1-4: failure/partial (treated as 0)
        failure_range=(1, 4),
    ),
    "range_num": _create_numeric_mapping(
        # High (10) = success, 7-9 = partial success (treated as 0), 5-6 = uncertain, 1-4 = failure
        success_range=(10, 10),
        partial_success_range=(7, 9),
        uncertain_range=(5, 6),
        failure_range=(1, 4),
    ),
    "range_num_vague": _create_numeric_mapping(
        success_range=(10, 10),
        partial_success_range=(7, 9),
        uncertain_range=(5, 6),
        failure_range=(1, 4),
    ),
    "four_degree": _create_numeric_mapping(
        success_range=(9, 10),  # all necessary
        failure_range=(1, 8),  # mostly incorrect through most necessary (all partial)
    ),
    "four_degree_vague": _create_numeric_mapping(
        success_range=(9, 10),
        failure_range=(1, 8),
    ),
    # Continuous confidence (0-1). Callable mapping attached after factory definition; temporary dummy lambda keeps type.
    "range_num_confidence": (lambda _x: None),
    "range_num_likelihood_0_100": (lambda _x: None),
    # NOTE: range_num_confidence is continuous; we attach a callable below so generic
    # scoring logic can invoke it without a hard-coded special case.
}


# Attach callable for continuous confidence template
TEMPLATE_MAPPINGS["range_num_confidence_0_1"] = _make_continuous_eval_mapping(
    max_val=1.0,
    min_val=0.0,
    success_thr=0.80,
    uncertain_lower_thr=0.40,
    uncertain_upper_thr=0.60,
)
TEMPLATE_MAPPINGS["range_num_likelihood_0_100"] = _make_continuous_eval_mapping(
    max_val=100.0,
    min_val=0.0,
    success_thr=80,
    uncertain_lower_thr=40,
    uncertain_upper_thr=60,
)
TEMPLATE_MAPPINGS["range_num_likelihood_0_100_2"] = _make_continuous_eval_mapping(
    max_val=100.0,
    min_val=0.0,
    success_thr=80,
    uncertain_lower_thr=40,
    uncertain_upper_thr=60,
)
TEMPLATE_MAPPINGS["s_prob"] = _make_continuous_eval_mapping(
    max_val=1.0,
    min_val=0.0,
    success_thr=0.501,
    uncertain_lower_thr=0.40,
    uncertain_upper_thr=0.60,
)

for key, value in TEMPLATE_MAPPINGS.items():
    if isinstance(value, dict):
        value["INFEASIBLE"] = 1


def map_eval_to_score_with_context(eval_response: str, eval_template: str | None = None) -> int | float | None:
    """
    Map evaluation responses to scores with template-specific context.

    Args:
        eval_response: The evaluation response string
        eval_template: The evaluation template used (e.g., 'four_statement_unc_letter')

    Returns:
        1 for success, 0 for failure/partial, np.nan for uncertain/invalid
    """
    # print(f"eval_response: {eval_response}, eval_template: {eval_template}")

    # Handle empty/None responses
    if not eval_response:
        return None

    # Normalize response
    eval_response = eval_response.upper().strip()

    # Special handling for four_statement_unc_letter_t_f template
    # This template expects comma-separated T/F values (e.g., "F, F, F, T, F")
    # corresponding to statements A, B, C, D, E
    if eval_template == "four_statement_unc_letter_t_f":
        # Parse the comma-separated T/F values
        values = [v.strip() for v in eval_response.split(",")]

        # Check if we have exactly 5 values
        if len(values) != 5:
            print(f"Expected 5 T/F values for four_statement_unc_letter_t_f, got {len(values)}: {eval_response}")
            return None

        # Normalize T/F values
        normalized_values = []
        for v in values:
            if v.upper() in ["T", "TRUE"]:
                normalized_values.append(True)
            elif v.upper() in ["F", "FALSE"]:
                normalized_values.append(False)
            else:
                print(f"Invalid T/F value in four_statement_unc_letter_t_f: {v}")
                return None

        # If E is True, the evaluation is uncertain
        if normalized_values[4]:  # E is True
            return np.nan

        # If A is True, it's a success
        if normalized_values[0]:  # A is True
            return 1

        # Otherwise (B, C, or D is True), it's a failure/partial
        return 0

    # Use template-specific mapping (dict or callable) if available
    if eval_template and eval_template in TEMPLATE_MAPPINGS:
        template_mapping = TEMPLATE_MAPPINGS[eval_template]
        if callable(template_mapping):
            return template_mapping(eval_response)
        if eval_response in template_mapping:
            return template_mapping[eval_response]
        else:
            print(f"eval_response: {eval_response} not found in template_mapping: {template_mapping}. Eval template: {eval_template}")
            return None
    else:
        return None


def extract_eval_template_from_variation_name(variation_name: str, fallback="tri") -> str | None:
    """
    Extract evaluation template from variation name.

    The pattern is: [number]p-[template]-[other_parts...]
    For example: 2p-tri-desc_cot_v2-actions-...

    Args:
        variation_name: The variation name extracted from directory structure

    Returns:
        The evaluation template name or None if not found
    """
    if not variation_name:
        return fallback

    if "s_prob" in variation_name:
        return "s_prob"

    # Split by common delimiters
    parts = re.split(r"[-]", variation_name)
    if len(parts) < 2:
        return fallback

    # Check if first part matches pattern like "1p", "2p", etc.
    first_part = parts[0].lower()
    if re.match(r"\d+p", first_part):
        # Template should be the second part
        potential_template = parts[1].lower()
    else:
        # Fallback: search through all parts
        parts = re.split(r"[-_]", variation_name)
        for part in parts:
            potential_template = part.lower()
            if potential_template in TEMPLATE_MAPPINGS:
                return potential_template
        return fallback

    # Clean potential template from known suffixes
    known_suffixes = [
        "_random",
        "_rev",
    ]
    for suffix in known_suffixes:
        potential_template = re.sub(suffix, "", potential_template)

    if potential_template in TEMPLATE_MAPPINGS:
        return potential_template

    return fallback


# ===============================================================
# Category Mapping (centralized for distributions)
# ===============================================================
# Numeric bands -> aggregated categories for each numeric template
# These follow the textual criteria in common_prompts.py and mirror
# the usage in compute_eval_distributions (SUCCESS, PARTIAL SUCCESS,
# UNCERTAIN, PARTIAL FAILURE, FAILURE).
# TODO: this can be integrated with map_eval_to_score_with_context + adding partial_failure range to _create_numeric_mapping

NUMERIC_CATEGORY_BANDS: dict[str, list[tuple[range, str]]] = {
    "bin_num_unc": [
        (range(7, 11), "SUCCESS"),
        (range(5, 7), "UNCERTAIN"),
        (range(1, 5), "FAILURE"),
    ],
    "tri_num_accomplished": [
        (range(7, 11), "SUCCESS"),
        (range(5, 7), "PARTIAL SUCCESS"),
        (range(1, 5), "FAILURE"),
    ],
    "tri_num_accomplished_inv": [
        (range(1, 5), "SUCCESS"),
        (range(5, 7), "PARTIAL SUCCESS"),
        (range(7, 11), "FAILURE"),
    ],
    "four_num_unc": [
        (range(9, 11), "SUCCESS"),
        (range(7, 9), "PARTIAL SUCCESS"),
        (range(5, 7), "UNCERTAIN"),
        (range(3, 5), "PARTIAL FAILURE"),
        (range(1, 3), "FAILURE"),
    ],
    "four_num_unc_accomplished": [
        (range(9, 11), "SUCCESS"),
        (range(7, 9), "PARTIAL SUCCESS"),
        (range(5, 7), "UNCERTAIN"),
        (range(3, 5), "PARTIAL FAILURE"),
        (range(1, 3), "FAILURE"),
    ],
    "four_num_unc_accomplished_inv": [
        (range(1, 3), "SUCCESS"),
        (range(3, 5), "PARTIAL SUCCESS"),
        (range(5, 7), "UNCERTAIN"),
        (range(7, 9), "PARTIAL FAILURE"),
        (range(9, 11), "FAILURE"),
    ],
    "range_num": [
        (range(10, 11), "SUCCESS"),
        (range(7, 10), "PARTIAL SUCCESS"),
        (range(5, 7), "UNCERTAIN"),
        (range(1, 4), "FAILURE"),
    ],
    "range_num_vague": [
        (range(10, 11), "SUCCESS"),
        (range(7, 10), "PARTIAL SUCCESS"),
        (range(5, 7), "UNCERTAIN"),
        (range(1, 4), "FAILURE"),
    ],
    "four_degree": [
        (range(9, 11), "SUCCESS"),
        (range(7, 9), "PARTIAL SUCCESS"),
        (range(5, 7), "PARTIAL FAILURE"),
        (range(3, 5), "PARTIAL FAILURE"),
        (range(1, 3), "FAILURE"),
    ],
    "four_degree_vague": [
        (range(9, 11), "SUCCESS"),
        (range(7, 9), "PARTIAL SUCCESS"),
        (range(5, 7), "PARTIAL FAILURE"),
        (range(3, 5), "PARTIAL FAILURE"),
        (range(1, 3), "FAILURE"),
    ],
}


def map_numeric_score_to_category(score_str: str | int, eval_template: str | None) -> str | None:
    """
    Map a numeric score (1-10) to an aggregated category for a numeric template.
    Returns one of: SUCCESS, PARTIAL SUCCESS, UNCERTAIN, PARTIAL FAILURE, FAILURE
    or None if not applicable.
    """
    if eval_template is None:
        return None
    try:
        score = int(score_str)
    except (ValueError, TypeError):
        return None

    # Special-case discrete scoring templates using 0/50/75/100
    if eval_template in {"four_room_score", "four_room_score_2"}:
        if score == 100 or score == 75:
            return "SUCCESS"
        if score == 50:
            return "PARTIAL SUCCESS"
        if score == 0:
            return "FAILURE"
        return None

    if score < 1 or score > 10:
        return None
    bands = NUMERIC_CATEGORY_BANDS.get(eval_template)
    if not bands:
        return None
    for score_range, category in bands:
        if score in score_range:
            return category
    return None


def map_response_to_category(eval_response: str, eval_template: str | None = None) -> str | None:
    """
    Centralized mapping of a single evaluation response to a category label,
    leveraging TEMPLATE_MAPPINGS semantics and special parsing where needed.

    Returns: SUCCESS, PARTIAL SUCCESS, PARTIAL FAILURE, FAILURE, UNCERTAIN, or None.
    """
    if not eval_response:
        return None
    resp = eval_response.strip().upper()

    # Numeric templates: derive category from numeric bands
    # For standard 1-10 integer and discrete score templates
    if resp.replace(".", "").replace("-", "").isdigit() and eval_template != "range_num_confidence":
        return map_numeric_score_to_category(resp, eval_template)

    # Special continuous confidence template (0-1 float)
    if eval_template == "range_num_confidence":
        try:
            val = float(resp)
        except ValueError:
            return None
        if not (0.0 <= val <= 1.0):
            return None
        if val >= 0.80:
            return "SUCCESS"
        if 0.60 <= val < 0.80:
            return "PARTIAL SUCCESS"
        if 0.40 <= val < 0.60:
            return "UNCERTAIN"
        if 0.20 <= val < 0.40:
            return "PARTIAL FAILURE"
        return "FAILURE"

    # four_statement_unc_letter_t_f: interpret comma-separated booleans
    if eval_template == "four_statement_unc_letter_t_f":
        parts = [v.strip() for v in resp.split(",")]
        if len(parts) == 5:
            normalized = []
            for v in parts:
                if v.upper() in ["T", "TRUE"]:
                    normalized.append(True)
                elif v.upper() in ["F", "FALSE"]:
                    normalized.append(False)
                else:
                    return None
            if normalized[4]:  # E true => UNCERTAIN
                return "UNCERTAIN"
            if normalized[0]:
                return "SUCCESS"
            # B/C/D true => treat as partial/failure buckets
            if normalized[1]:
                return "PARTIAL SUCCESS"
            if normalized[2]:
                return "PARTIAL FAILURE"
            if normalized[3]:
                return "FAILURE"
            return None

    # four_statement_unc_letter (letter form)
    if eval_template == "four_statement_unc_letter":
        if resp == "A":
            return "SUCCESS"
        if resp == "B":
            return "PARTIAL SUCCESS"
        if resp == "C":
            return "PARTIAL FAILURE"
        if resp == "D":
            return "FAILURE"
        if resp == "E":
            return "UNCERTAIN"

    # three_statement_unc_letter
    if eval_template == "three_statement_unc_letter":
        if resp == "A":
            return "SUCCESS"
        if resp == "B":
            return "PARTIAL SUCCESS"
        if resp == "C":
            return "FAILURE"
        if resp == "D":
            return "UNCERTAIN"

    # four_option_unc_question
    if eval_template == "four_option_unc_question":
        if resp == "V":
            return "SUCCESS"
        if resp == "IV":
            return "PARTIAL SUCCESS"
        if resp == "III":
            return "UNCERTAIN"
        if resp in ["II", "I"]:
            return "FAILURE"

    # three_option_unc
    if eval_template == "three_option_unc":
        if resp == "I":
            return "SUCCESS"
        if resp == "II":
            return "PARTIAL SUCCESS"
        if resp == "III":
            return "PARTIAL FAILURE"

    # four_excellent
    if eval_template in ("four_excellent", "four_excellent_2"):
        if resp == "EXCELLENT" or resp == "GOOD":
            return "SUCCESS"
        if resp == "FAIR":
            return "FAILURE"
        if resp == "POOR":
            return "FAILURE"

    # four_letter
    if eval_template in ("four_letter", "four_letter_2"):
        if resp in ["A", "B"]:
            return "SUCCESS"
        if resp in ["C", "D"]:
            return "FAILURE"

    # General known categories
    if resp == "SUCCESS WITH ROOM FOR IMPROVEMENT":
        return "SUCCESS"
    if resp == "INFEASIBLE":
        return "SUCCESS"
    if resp in ["SUCCESS", "PARTIAL SUCCESS", "PARTIAL FAILURE", "FAILURE", "UNCERTAIN", "UNCLEAR"]:
        return "UNCERTAIN" if resp == "UNCLEAR" else resp

    return None


# ===============================================================
# Export Functions
# ===============================================================

__all__ = [
    "EVAL_LABELS",
    "TEMPLATE_MAPPINGS",
    "map_eval_to_score_with_context",
    "extract_eval_template_from_variation_name",
    "map_numeric_score_to_category",
    "map_response_to_category",
]

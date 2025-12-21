# ===============================================
# LINK: System Prompt parts
# ===============================================
import random
import re

from offline_experiments.prompts.common_prompts import common_prompts
from offline_experiments.prompts.prompts_osw import prompts_osw
from offline_experiments.prompts.prompts_vwa import prompts_vwa

prompts_per_env = {
    "osw": prompts_osw,
    "vwa": prompts_vwa,
    "agrb_vwa": prompts_vwa,
}

new_criterias_id: list[str] = ["num", "degree", "letter", "option"]


def _get_trace_info_str(trace_config: dict, prompts: dict) -> str:
    if not trace_config:
        return ""

    trace_info_type = trace_config.get("type", "")
    if not trace_info_type:
        return ""

    thought_action_indexes = trace_config.get("idxs", [])
    if trace_info_type.lower() == "none":
        return ""

    elif trace_config.get("type") == "img_only":
        key = "img_only"

    elif "img_expert" in trace_info_type and "actions" in trace_info_type:
        key = "actions_img_expert"

    elif "img_expert" in trace_info_type and "utt" in trace_info_type:
        key = "utt_img_expert"
    elif "utt" in trace_info_type and "actions" in trace_info_type:
        key = "utt"

    elif "actions" in trace_info_type:
        key = "actions"

    elif "utt" in trace_info_type:
        key = "utt"

    else:
        raise ValueError(f"Invalid trace config: {trace_config}")

    if thought_action_indexes == [-1]:
        return prompts["trace_infos"]["last_u"]

    trace_str = prompts["trace_infos"][key]
    return trace_str


def _get_image_info_str(img_ann_types: str, prompts: dict) -> str:
    if not img_ann_types or "raw" in img_ann_types:
        key = "raw"
    elif "som" in img_ann_types and "coord" in img_ann_types:
        key = "som_coord"
    elif "som" in img_ann_types:
        key = "som"
    elif "coord" in img_ann_types:
        key = "coord"
    else:
        raise ValueError(f"Annotation type {img_ann_types} not recognized")

    return prompts["image_infos"][key]


def add_noise_to_k(
    k: str,
    noise_key: str = "latin_noise",
    noise_injections: dict = common_prompts["noise_injections"],
) -> str:
    """
    Add noise to a given text by randomly inserting a noise line between content lines.
    Args:
        k: The input text to add noise to
        noise_key: Key to select which noise to use from noise_injections
        noise_injections: Dictionary containing different types of noise
    Returns:
        The text with randomly inserted noise
    """
    lines = k.split("\n")
    noise: str = f"{noise_injections[noise_key]}"
    # Choose a random line index to insert noise (can be at the start, between, or end)
    insert_idx = random.randint(0, len(lines))
    # Choose a random noise line from the noise list
    # Insert the noise line
    if insert_idx < len(lines) and not lines[insert_idx] == "":
        noise = noise + "\n"

    if insert_idx > 0 and not lines[insert_idx - 1] == "":
        noise = "\n" + noise

    new_lines = lines[:insert_idx] + [noise] + lines[insert_idx:]
    return "\n".join(new_lines)


def get_k_injection(env: str, k_prompt_id: str) -> str:
    prompts = prompts_per_env[env]
    k_injection = prompts["k_prompt_parts"][k_prompt_id]["injection"]
    return k_injection


def reassign_criteria(criteria_id: str, criteria_prompt: str, shuffle_type: str) -> str:
    if criteria_id in new_criterias_id:
        # Lettered options (A., B., ...)
        if "letter" in criteria_id:
            lines = criteria_prompt.strip().split("\n")
            option_start = None
            for i, line in enumerate(lines):
                if re.match(r"^[A-Z]\. ", line.strip()):
                    option_start = i
                    break
            if option_start is None:
                raise ValueError(f"Error shuffling criteria: No match found for criteria_id: {criteria_id}")
            intro_section = "\n".join(lines[:option_start])
            option_lines = lines[option_start:]
            # Remove the labels
            option_texts = [re.sub(r"^[A-Z]\. ", "", li.strip(), 1) for li in option_lines]
            if shuffle_type == "random":
                random.shuffle(option_texts)
            elif shuffle_type == "rev":
                option_texts = option_texts[::-1]
            # Reassign labels
            labels = [chr(ord("A") + i) for i in range(len(option_texts))]
            reassigned_options = [f"{label}. {text}" for label, text in zip(labels, option_texts)]
            new_criteria = intro_section + ("\n" if intro_section else "") + "\n".join(reassigned_options)
            return new_criteria
        # Roman numeral options (I., II., ...)
        elif "option" in criteria_id:
            lines = criteria_prompt.strip().split("\n")
            option_start = None
            for i, line in enumerate(lines):
                if re.match(r"^(I{1,3}|IV|V|VI{0,3}|IX|X)\. ", line.strip()):
                    option_start = i
                    break
            if option_start is None:
                raise ValueError(f"Error shuffling criteria: No match found for criteria_id: {criteria_id}")
            intro_section = "\n".join(lines[:option_start])
            option_lines = lines[option_start:]
            # Remove the labels
            option_texts = [re.sub(r"^(I{1,3}|IV|V|VI{0,3}|IX|X)\. ", "", li.strip(), 1) for li in option_lines]
            if shuffle_type == "random":
                random.shuffle(option_texts)
            elif shuffle_type == "rev":
                option_texts = option_texts[::-1]
            # Reassign labels (Roman numerals)
            roman_labels = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
            reassigned_options = [f"{roman_labels[i]}. {text}" for i, text in enumerate(option_texts)]
            new_criteria = intro_section + ("\n" if intro_section else "") + "\n".join(reassigned_options)
            return new_criteria
        # Numeric/degree: shuffle/reverse score lines, reassign score labels
        elif ("num" in criteria_id) or ("degree" in criteria_id):
            # Find the score lines (start with 'Score ...:')
            lines = criteria_prompt.strip().split("\n")
            score_start = None
            for i, line in enumerate(lines):
                if re.match(r"^Score ", line.strip()):
                    score_start = i
                    break
            if score_start is None:
                raise ValueError(f"Error shuffling criteria: No score section found for criteria_id: {criteria_id}")
            intro_section = "\n".join(lines[:score_start])
            score_lines = lines[score_start:]
            # Extract score labels and texts
            score_labels = []
            score_texts = []
            for lin in score_lines:
                m = re.match(r"^(Score [^:]+:)(.*)$", lin.strip())
                if m:
                    score_labels.append(m.group(1))
                    score_texts.append(m.group(2).strip())
            if not score_labels:
                raise ValueError(f"Error shuffling criteria: No valid score lines for criteria_id: {criteria_id}")
            # Shuffle or reverse the texts
            if shuffle_type == "random":
                random.shuffle(score_texts)
            elif shuffle_type == "rev":
                score_texts = score_texts[::-1]
            # Reassign score labels in order
            reassigned_scores = [f"{label}{' ' if text else ''}{text}" for label, text in zip(score_labels, score_texts)]
            new_criteria = intro_section + ("\n" if intro_section else "") + "\n".join(reassigned_scores)
            return new_criteria
        else:
            return criteria_prompt
    return criteria_prompt


def shuffle_criteria(criteria_id: str, criteria_prompt: str, shuffle_type: str) -> str:
    try:
        new_criteria = criteria_prompt
        # Numeric or degree criteria
        if ("num" in criteria_id) or ("degree" in criteria_id):
            pattern = re.compile(r"(Score[\s\S]*)")
            match = pattern.search(criteria_prompt)
            if not match:
                raise ValueError(f"Error shuffling criteria: No match found for criteria_id: {criteria_id}")
            score_section = match.group(1)
            intro_section = new_criteria.strip()[: match.start()]

            # Shuffle the score section
            lines = score_section.strip().split("\n")
            if shuffle_type == "random":
                random.shuffle(lines)
            elif shuffle_type == "rev":
                lines = lines[::-1]

            shuffled_section = "\n".join(lines)
            new_criteria = intro_section + "\n" + shuffled_section

        # Lettered options (A., B., ...)
        elif "letter" in criteria_id:
            # Find the first line that starts with a capital letter and a period (A., B., ...)
            lines = new_criteria.strip().split("\n")
            option_start = None
            for i, line in enumerate(lines):
                if re.match(r"^[A-Z]\. ", line.strip()):
                    option_start = i
                    break
            if option_start is None:
                raise ValueError(f"Error shuffling criteria: No match found for criteria_id: {criteria_id}")
            intro_section = "\n".join(lines[:option_start])
            option_lines = lines[option_start:]
            if shuffle_type == "random":
                random.shuffle(option_lines)
            elif shuffle_type == "rev":
                option_lines = option_lines[::-1]
            shuffled_section = "\n".join(option_lines)
            new_criteria = intro_section + ("\n" if intro_section else "") + shuffled_section

        # Roman numeral options (I., II., ...)
        elif "option" in criteria_id:
            # Find the first line that starts with a Roman numeral and a period (I., II., ...)
            lines = new_criteria.strip().split("\n")
            option_start = None
            for i, line in enumerate(lines):
                if re.match(r"^(I{1,3}|IV|V|VI{0,3}|IX|X)\. ", line.strip()):
                    option_start = i
                    break
            if option_start is None:
                raise ValueError(f"Error shuffling criteria: No match found for criteria_id: {criteria_id}")
            intro_section = "\n".join(lines[:option_start])
            option_lines = lines[option_start:]
            if shuffle_type == "random":
                random.shuffle(option_lines)
            elif shuffle_type == "rev":
                option_lines = option_lines[::-1]
            shuffled_section = "\n".join(option_lines)
            new_criteria = intro_section + ("\n" if intro_section else "") + shuffled_section

        # Default: shuffle all lines
        else:
            lines = new_criteria.strip().split("\n")
            if shuffle_type == "random":
                random.shuffle(lines)
            elif shuffle_type == "rev":
                lines = lines[::-1]
            shuffled_section = "\n".join(lines)
            new_criteria = shuffled_section
        return new_criteria
    except Exception as e:
        print(f"Error shuffling criteria: {e}")
        return criteria_prompt


def safe_format(string_template: str, fill_with: str = "", **kwargs) -> str:
    """
    Formats a given template using the provided keyword arguments.
    Missing keys in the template are replaced with an empty string.

    Args:
        template (str): The string template with placeholders.
        **kwargs: Key-value pairs for formatting.

    Returns:
        str: The formatted string with missing keys as empty strings.
    """

    class DefaultDict(dict):
        def __missing__(self, key):
            return fill_with

    return string_template.format_map(DefaultDict(**kwargs))


def _get_eval_request(
    env: str,
    config: dict,
    single_pass: bool,
) -> str:
    prompts = prompts_per_env[env]
    eval_criterias = prompts["eval_criterias"]
    response_format_template = prompts["response_format"]
    prompt_config = config["prompt_args"]

    if "aeval_refine" in prompt_config["sys_prompt"]:
        return ""

    # Get CoT parts
    if prompt_config.get("cot_part"):
        _cot_parts = prompts["cot_parts"][prompt_config["cot_part"]].strip()
    else:
        _cot_parts = ""

    # Get eval criteria
    eval_criteria = prompt_config.get("eval_criteria", "")
    if eval_criteria:
        if "random" in eval_criteria:
            _eval_criteria = eval_criterias[eval_criteria.replace("_random", "").strip("_")].strip()
            _eval_criteria = shuffle_criteria(eval_criteria, _eval_criteria, "random")
        elif "rev" in eval_criteria:
            _eval_criteria = eval_criterias[eval_criteria.replace("_rev", "").strip("_")].strip()
            _eval_criteria = shuffle_criteria(eval_criteria, _eval_criteria, "rev")
        elif "reassign_random" in eval_criteria:
            _eval_criteria = eval_criterias[eval_criteria.replace("_reassign_random", "").strip("_")].strip()
            _eval_criteria = reassign_criteria(eval_criteria, _eval_criteria, "random")
        elif "reassign_rev" in eval_criteria:
            _eval_criteria = eval_criterias[eval_criteria.replace("_reassign_rev", "").strip("_")].strip()
            _eval_criteria = reassign_criteria(eval_criteria, _eval_criteria, "rev")

        else:
            _eval_criteria = eval_criterias[eval_criteria].strip()
        if any(c in eval_criteria for c in new_criterias_id):
            eval_prompt_template = prompts["eval_prompt_templates"]["ins"]
        else:
            eval_prompt_template = prompts["eval_prompt_templates"]["cat"]
    else:
        _eval_criteria = ""
        eval_prompt_template = ""

    if single_pass and prompt_config.get("k_configs", None):
        k_steps = []
        for k_config in prompt_config["k_configs"]:
            k_prompt_id = k_config["k_prompt_id"]
            k_steps.append(prompts["k_prompt_parts"][k_prompt_id]["extraction"].strip())
        k_step = "\n".join(k_steps).strip()
    else:
        k_step = ""

    response_format = response_format_template.format(k_step=k_step, cot_parts=_cot_parts).strip()

    # Get eval prompt
    eval_prompt = safe_format(
        eval_prompt_template,
        eval_criteria=_eval_criteria,
        response_format=response_format,
    )
    return eval_prompt.strip()


def _get_sys_prompt_verifier(
    env: str,
    config: dict = {},
) -> str:
    # Get all prompts for the env
    prompts = prompts_per_env[env]
    prompt_config = config["prompt_args"]

    # Get sys_prompt template
    sys_prompt_verifier_template = prompts["sys_prompts_verifier"][prompt_config["sys_prompt"]]

    # -----------------------------------
    # System prompt parts
    # -----------------------------------
    # Rules
    if prompt_config.get("rule", ""):
        rule = prompts["sys_prompt_parts"]["rules"][prompt_config["rule"]]
    else:
        rule = ""

    # Trace and Image info
    trace_info_str = _get_trace_info_str(prompt_config.get("trace_info", {}), prompts)
    img_info_str = _get_image_info_str(config.get("img_ann_types", ""), prompts)

    # K info
    if k_configs := prompt_config.get("k_configs", None):
        _k_infos = []
        for k_config in k_configs:
            k_prompt_id = k_config["k_prompt_id"]
            _k_infos.append(prompts["k_prompt_parts"][k_prompt_id]["sys_k_info"])
        _k_info = "\n".join(_k_infos)
    else:
        _k_info = ""

    sys_prompt = safe_format(
        sys_prompt_verifier_template,
        trace_info=trace_info_str,
        k_info=_k_info,
        rule=rule,
        image_info=img_info_str,
    )
    return sys_prompt.strip()


def get_prompts_first_pass(
    env: str,
    config,
) -> tuple[str, str]:
    prompts = prompts_per_env[env]
    prompt_config = config["prompt_args"]
    k_prompt_config = prompt_config["k_config"]
    k_prompt_parts = prompts["k_prompt_parts"][prompt_config["k_config"]["k_prompt_id"]]

    # System prompt
    if "sys_prompt" in k_prompt_parts:
        trace_info_str = _get_trace_info_str(k_prompt_config.get("trace_info", {}), prompts)
        img_info_str = _get_image_info_str(k_prompt_config.get("img_ann_types", ""), prompts)
        sys_prompt_template = k_prompt_parts["sys_prompt"]
        rules = k_prompt_parts.get("rule", "")
        _sys_prompt = safe_format(
            sys_prompt_template,
            trace_info=trace_info_str,
            image_info=img_info_str,
            rule=rules,
        )
    else:
        raise ValueError(f"System prompt not found for k_prompt_id: {prompt_config['k_config']['k_prompt_id']}")

    k_extract_request = k_prompt_parts["extraction"]

    return k_extract_request, _sys_prompt


def get_prompts_eval(config: dict, single_pass: bool) -> dict[str, str]:
    env = config["env"]

    eval_prompt = _get_eval_request(env, config, single_pass)
    sys_prompt = _get_sys_prompt_verifier(env, config)
    return {"eval_prompt": eval_prompt, "sys_prompt": sys_prompt}

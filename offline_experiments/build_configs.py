import json
import os
import re
import warnings
from copy import deepcopy
from datetime import datetime
from itertools import product
from typing import Any

from core_utils.logger_utils import logger
from offline_experiments.config.eval_configs import all_env_configs
from offline_experiments.config.llm_configs import llm_configs

warnings.filterwarnings("ignore", message="Pydantic serializer warnings:")

timestamp = datetime.now().strftime("%Y-%m-%d")


def dump_exper_args(
    config: dict,
    gen_config: dict,
):
    output_dir = config["out_dir"]
    if os.path.exists(f"{output_dir}/exper_args.json"):
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(f"{output_dir}/exper_args.json", "w") as f:
        args_to_dump = {
            "gen_config": gen_config,
            "date": timestamp,
        }

        if config is not None:
            args_to_dump.update(config)

        json.dump(args_to_dump, f, indent=4)


def _build_config_name_first_pass(
    k_prompt_id: str,
    trace_info: dict,
    img_ann_types: str,
    add_config: dict | None = None,
    ann_k: str = "",
):
    k_config_str = ""
    k_config_str += f"{k_prompt_id}"
    trace_info_type = trace_info.get("type", "")
    if trace_info_type:
        k_config_str += f"-trace_{trace_info_type}"

    if img_ann_types:
        k_config_str += f"-imgann_{img_ann_types}"

    if add_config:
        if add_config.get("caption_input_images", False):
            k_config_str += "-cap_input_img"

    return f"{k_config_str.strip('-')}{ann_k}"


def _build_config_name_verify(
    eval_criteria: str | None = "",
    cot_part: str | None = "",
    trace_config: dict | None = None,
    k_configs: list[dict] | None = None,
    env="",
    rule=None,
    sys_prompt: str | None = "",
    img_ann_types: str | None = "",
    add_config: dict | None = None,
):
    config_str = ""
    if eval_criteria:
        config_str += f"{eval_criteria}"
    if cot_part:
        config_str += f"-{cot_part}"
    if trace_config:
        if trace_config.get("type", ""):
            trace_info = trace_config["type"]
            config_str += f"-{trace_info}"
        if trace_config.get("shuffle", False):
            config_str += f"-shuffle"

    rules_str = f"-{rule}" if rule else ""
    config_str += rules_str

    if sys_prompt:
        config_str += f"-sys_{sys_prompt}"

    if img_ann_types:
        config_str += f"-imgann_{img_ann_types}"

    if add_config:
        if add_config.get("caption_input_images", False):
            config_str += "-cap_input_img"

    k_config_str = ""
    for k_config in k_configs or []:
        k_prompt_id = k_config["k_prompt_id"]
        if "1p" in k_prompt_id:
            continue
        fp_model = re.sub("/", "-", k_config["fp_model"].split("/")[-1])
        k_trace_info = k_config.get("trace_info", {"type": ""})
        k_add_config = k_config.get("additional_config", {})
        k_config_str += f"__{_build_config_name_first_pass(k_prompt_id, k_trace_info, k_config.get('img_ann_types', ''), add_config=k_add_config)}-{fp_model}"

    final_str = f"{config_str}{k_config_str}"

    return final_str.strip("-")


def _get_data_for_agent(env_config: list[dict], agent_id: str, data_key: str):
    for entry in env_config:
        if entry["agent_id"] == agent_id:
            return entry[data_key]
    return None


def build_all_first_pass_configs(cases, out_dir_template, ann_k=""):
    configs_per_env = {}
    # Create configs to run per env
    for env in all_env_configs:
        all_configs = {}
        # Create configs to run per CASE input
        for case_config in cases:
            # Create run configs for each source of trajectories to evaluate
            for case_name, case_config in case_config.items():
                for trace_data in case_config["trajectories_data"]:
                    agent_id = trace_data["agent_id"]
                    source_file_list = _get_data_for_agent(
                        all_env_configs[env]["trajectories_data"],
                        agent_id=agent_id,
                        data_key="source_file_list",
                    )
                    if not source_file_list:
                        logger.warning(f"Source file list not found for agent {agent_id}, env {env}. Skipping.")
                        continue
                    else:
                        logger.info(f"Building first step config list for agent {agent_id}, env {env}.")

                    combinations = []
                    if not case_config.get("k_configs", []):
                        continue

                    # Flatten k_configs
                    flat_k_configs = {}
                    for k_configs in case_config["k_configs"]:
                        for k_config in k_configs:
                            k_config_str = str(k_config)
                            flat_k_configs[k_config_str] = k_config

                    combinations.extend(
                        list(
                            product(
                                all_env_configs[env]["domains"],
                                list(flat_k_configs.values()),
                            )
                        )
                    )

                    # Create run configs for  all combinations
                    for i, (domain, k_config) in enumerate(combinations):
                        model_fp = k_config["fp_model"]
                        gen_config_fp = {**llm_configs[model_fp]}
                        # Filter out single pass ("1p") configs; no first step generation.
                        if "1p" in k_config["k_prompt_id"]:
                            continue

                        k_prompt_id = k_config["k_prompt_id"]
                        fp_model = k_config["fp_model"]
                        trace_info = k_config.get("trace_info", {"type": ""})
                        img_ann_types = k_config.get("img_ann_types", [])
                        key = f"{env}_{agent_id}_{case_name}-"
                        full_config_name = _build_config_name_first_pass(
                            k_prompt_id,
                            trace_info,
                            img_ann_types,
                            add_config=k_config.get("additional_config", {}),
                            ann_k=ann_k,
                        )
                        key += full_config_name
                        subkey = domain

                        if key not in all_configs:
                            all_configs[key] = {}

                        config = {
                            "prompt_args": {
                                "eval_criteria": "",
                                "cot_part": "",
                                "trace_info": trace_info,
                                "k_config": k_config,
                                "rules": case_config.get("rules", []),
                            },
                            "env": env,
                            "domain": domain,
                            "additional_config": case_config.get("additional_config", {}),
                            "source_file_list": source_file_list,
                            "gen_config": gen_config_fp,
                        }
                        model_name = re.sub("/", "-", fp_model.split("/")[-1])
                        out_dir = f"{out_dir_template.format(env=env, model_name=model_name, domain=domain, agent_id=agent_id, full_config_name=full_config_name)}"
                        config["out_dir"] = out_dir
                        all_configs[key][subkey] = config
            configs_per_env[env] = all_configs
    return configs_per_env


def build_all_verif_configs(cases, model_v, gen_config_v, out_dir_template, ann_v=""):
    model_name = re.sub("/", "-", model_v.split("/")[-1])
    configs_per_env = {env: {} for env in all_env_configs}
    for env in all_env_configs:
        all_configs = {}
        for case_config in cases:
            case_config: dict[str, Any]
            for case_name, case_config in case_config.items():
                for trajectory_data in case_config["trajectories_data"]:
                    agent_id = trajectory_data["agent_id"]
                    source_file_list = _get_data_for_agent(
                        all_env_configs[env]["trajectories_data"],
                        agent_id=agent_id,
                        data_key="source_file_list",
                    )
                    if not source_file_list:
                        logger.warning(f"Source file list not found for agent {agent_id} in env {env}. Skipping.")
                        continue
                    else:
                        logger.info(f"Building verify config list for agent {agent_id} in env {env}.")

                    # Generate all combinations for each case
                    k_configs = case_config.get("k_configs", [[]])
                    combinations = list(
                        product(
                            all_env_configs[env]["domains"],
                            case_config["eval_criterias"],
                            case_config["cot_parts"],
                            case_config["trace_infos"],
                            case_config.get("k_configs", [[]]),
                            case_config.get("rules", [None]),
                            case_config["sys_prompts"],
                            ["raw"] if not case_config.get("img_ann_types", []) else case_config["img_ann_types"],
                        )
                    )

                    # Create new cases with all combinations
                    for i, (
                        domain,
                        eval_criteria,
                        cot_part,
                        trace_info,
                        k_configs,
                        rule,
                        sys_prompt,
                        img_ann_types,
                    ) in enumerate(combinations):
                        key = f"{env}__{agent_id}__{case_name}-"
                        config_name = _build_config_name_verify(
                            eval_criteria=eval_criteria,
                            cot_part=cot_part,
                            trace_config=trace_info,  # type: ignore
                            k_configs=k_configs,  # type: ignore
                            env=env,
                            rule=rule,
                            sys_prompt=sys_prompt,
                            img_ann_types=img_ann_types,
                            add_config=case_config.get("additional_config", {}),
                        )
                        key += config_name
                        if key not in all_configs:
                            all_configs[key] = {}

                        full_config_name = f"{case_name}-{config_name}{ann_v}"
                        k_configs = deepcopy(k_configs) if k_configs else []
                        config = {
                            "prompt_args": {
                                "eval_criteria": eval_criteria,
                                "cot_part": cot_part,
                                "trace_info": trace_info,
                                "rule": rule if rule else None,
                                "sys_prompt": sys_prompt,
                            },
                            "env": env,
                            "domain": domain,
                            "out_dir": f"{out_dir_template.format(env=env, model_name=model_name, domain=domain, agent_id=agent_id, full_config_name=full_config_name)}",
                            "img_ann_types": img_ann_types,
                            "additional_config": case_config.get("additional_config", {}),
                            "source_file_list": source_file_list,
                            "gen_config": gen_config_v,
                        }
                        for k_config in k_configs:
                            assert isinstance(k_config, dict)
                            if "1p" in k_config["k_prompt_id"]:
                                continue
                            k_prompt_id = k_config["k_prompt_id"]
                            fp_model = k_config["fp_model"]
                            trace_info = k_config.get("trace_info", {"type": ""})
                            img_ann_types = k_config.get("img_ann_types", [])
                            fp_config_name = _build_config_name_first_pass(k_prompt_id, trace_info, img_ann_types, add_config=k_config.get("additional_config", {}))
                            cached_k_dir = f"{out_dir_template.format(domain=domain, env=env, model_name=fp_model, agent_id=agent_id, full_config_name=fp_config_name)}"
                            k_config["cached_k_dir"] = cached_k_dir

                        config["prompt_args"]["k_configs"] = k_configs
                        all_configs[key][domain] = config
        configs_per_env[env] = all_configs
    return configs_per_env

# ===============================================================================
# Agent and Model config helpers
# ===============================================================================
import yaml

from agent.constants import LM_CONFIGS
from core_utils.file_utils import get_attribute_from_dict


def load_agent_config(agent_config_file: str) -> dict:
    with open(agent_config_file, "r") as file:
        agents_configs = yaml.safe_load(file)
    return agents_configs


def get_lm_config(model, gen_config_alias: str = "", lm_config_file: str = LM_CONFIGS) -> dict:
    if not gen_config_alias:
        return {}

    with open(lm_config_file, "r") as file:
        all_lm_configs = yaml.safe_load(file)
    return all_lm_configs[model][gen_config_alias]


def resolve_inheritance(agents_configs: dict):
    # For each agent
    for agent_str, agent_config in agents_configs.items():
        # For ach attribute
        for key, subconfig in agent_config.items():
            if not isinstance(subconfig, dict):
                continue

            if "_inherit_from" in subconfig:
                inherit_from = subconfig["_inherit_from"]
                agents_configs[agent_str][key] = agents_configs[inherit_from][key].copy()
    return agents_configs


def get_agent_attribute(agent_config_path: str, attribute: str):
    agents_configs = load_agent_config(agent_config_path)
    return get_attribute_from_dict(attribute, agents_configs)


def get_agent_config(agent_config_path: str):
    # Load general agent configurations
    agents_configs = load_agent_config(agent_config_path)

    # Set model and LM Config for each agent
    for config in agents_configs.values():
        # If not LLM-based agent, skip
        if "lm_config" not in config:
            continue

        # If LM configs inherited from other modules, inherit
        if "_inherit_from" in config["lm_config"]:
            mod_inherit_from = config["lm_config"]["_inherit_from"]
            config["lm_config"] = agents_configs[mod_inherit_from]["lm_config"]

        config["lm_config"].update(
            get_lm_config(
                model=config["lm_config"]["model"],
                gen_config_alias=config["lm_config"].get("gen_config_alias", ""),
            )
        )
    return agents_configs

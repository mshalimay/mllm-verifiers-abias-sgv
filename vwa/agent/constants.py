from pathlib import Path

_AGENT_DIR = Path(__file__).resolve().parent

# Directory containing prompt python files like `p_verifier.py`.
PATH_RAW_PROMPTS = str(_AGENT_DIR / "prompts" / "raw")

# YAML file containing LM configs.
LM_CONFIGS = str(_AGENT_DIR / "config" / "lm_configs.yaml")

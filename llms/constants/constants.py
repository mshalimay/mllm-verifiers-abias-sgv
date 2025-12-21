import os

current_dir = os.path.dirname(os.path.abspath(__file__))
llms_dir = os.path.dirname(current_dir)


# NOTE: Absolute path defaults under the llms/ package directory to avoid depending on the
# current working directory. You can override with environment variables.
API_KEYS_PATH = os.getenv("LLMS_API_KEYS_PATH", os.path.join(llms_dir, "api_keys.json"))
API_KEYS_BACKUP = os.getenv("LLMS_API_KEYS_BACKUP_PATH", os.path.join(llms_dir, "api_keys_backup.json"))

# Environment variable names for different providers' API keys
# e.g.: os.environ[API_KEY_ENV_VARS["openai"]] to get OpenAI API key
API_KEY_ENV_VARS = {
    "google": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "huggingface": "HF_TOKEN",
    "openrouter": "OPENROUTER_API_KEY",
    "alibaba": "DASHSCOPE_API_KEY",
    "together_ai": "TOGETHER_API_KEY",
}

ROLE_MAPPINGS = {
    "google": {
        # Assistant roles
        "assistant": "model",
        "model": "model",
        # User role
        "user": "user",
        # System role
        "system": "system",
    },
    "openai": {
        # User role
        "user": "user",
        # Assistant role
        "assistant": "assistant",
        # System roles
        "system": "system",
        "developer": "system",  # developer is the new key, but `system` is backwards compatible
    },
}


GENERATION_PREFIX_PATTERN = "-" * 10 + " GENERATION {} " + "-" * 10


# Backward compatibility
model_repo_relative_path = "./config/model_repo.yaml"
MODEL_REPO_PATH = os.path.join(llms_dir, model_repo_relative_path)

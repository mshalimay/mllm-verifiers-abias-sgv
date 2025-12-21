# Client Manager
MAX_API_KEY_RETRY = 10  # Maximum number of times to retry an API key during API calls in case of exceptions
MAX_KEY_PROCESS_COUNT = 10

# Prompter
ROLE_MAPPINGS = {
    "assistant": "assistant",
    "user": "user",
    "system": "system",  # Obs.: new role is `developer`, but system is backward and forward compatible
    "developer": "system",
}
UPLOAD_IMAGES = False  # Whether to upload images to cloud by default (not supported for OpenAI yet)

# Default mode for models from OpenAI. This is used to fill the `config_repo.yaml` file when fetching models from OpenAI.
DEFAULT_ANTHROPIC_MODE = "chat_completion"

# Default max tokens for models from OpenAI. This is used to fill the `config_repo.yaml` file when fetching models from OpenAI.
DEFAULT_ANTHROPIC_MAX_TOKENS = 8192

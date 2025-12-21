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
DEFAULT_OPENAI_MODE = "chat_completion"
# NOTE: Mar-2025: OpenAI introduced the the `response` API, but it's still more limited than `chat_completion`.
# However, some models are only supported via the `response` API.

MAX_REQUESTS_PER_MINUTE = 1000

DEFAULT_TIMEOUT = 60 * 10  # 10 minutes


DEFAULT_SUMMARY_DETAIL = "auto"

THROTTLED_SLEEP = 30


MAX_IMG_PAYLOAD_SIZE_MB = {"gpt-4.1-mini-2025-04-14": 50, "o4-mini-2025-04-16": 50}

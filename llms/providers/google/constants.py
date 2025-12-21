from google.genai.types import HarmBlockThreshold, HarmCategory, SafetySetting

# ---- Client Manager ----
MAX_API_KEY_RETRY = 3
MAX_KEY_PROCESS_COUNT = 1
API_VERSION = "v1alpha"
DEFAULT_REQUEST_TIMEOUT: int | None = 5 * 60 * 1000  # 10 minutes in milliseconds

# ---- Prompter ----
ROLE_MAPPINGS = {"assistant": "model", "user": "user", "system": "system"}
UPLOAD_IMAGES = False  # By default, images are not uploaded to the cloud unless payload is too large.
MAX_PAYLOAD_SIZE = 14 * 1024 * 1024 * 1  # If prompt larger than this, upload parts of the payload to the cloud.
UPLOAD_ALL = True  # controls upload once hits max payload size. False = uploads up until max_payload_size remaining; else, uploads everything. E.g.: max = 9mb, payload = 20mb => False uploads 11mb, True uploads 20mb


# ======================================================
# google_utils.py
# ======================================================
THINKING_MODELS_STRS = ["2.5", "thinking"]
MAX_THINKING_BUDGETS = {"gemini-2.5": 24576}
MAX_CALLS_PER_KEY = 60  # if num_calls > MAX_CALLS_PER_KEY, key is rotated.
MIN_SECS_BTW_GEN = 1  # if last_call - now = delta < MIN_SECS_BTW_GEN, sleep for delta.
MAX_GENERATION_PER_BATCH = 8  # When `num_generations` > 1, distributed into `n/MAX_GENERATION_PER_BATCH` calls
MAX_RETRIES = 2  # Max retries before switching to a new API key

# --- Handling retries with exponential backoff ---
MAX_DELAY = 60 * 1.5  # Maximum delay between retries
BASE_DELAY = 30 // 2  # Initial delay in the exponential backoffs = `num` seconds in `(num//2)`
BASE_DELAY_THROTTLED = 50  # Fixed delay in throttled executions
MAX_WAIT_PER_GEN = 10 * 60  # Maximum wait time for each generation
MAX_API_WAIT_TIME = 10 * 60  # Max wait time for overall API call before flagging as failed. Usually less than this.
# Obs.: final timeout = `min(MAX_WAIT_PER_GEN * num_generations, MAX_API_WAIT_TIME)`

# --- Batch generation / throttled execution configs ---
MIN_DELAY_RESET = MAX_WAIT_PER_GEN  # Minimum delay between API key resets.
MAX_RETRIES_THROTTLED = 1  # Max retries before switching to a new API key
MAX_REQUESTS_PER_MINUTE = 4  # Used in batch generation; limits to X concurrent calls per minute
# https://ai.google.dev/gemini-api/docs/rate-limits#free-tier
MODEL_TEST_API_KEY = "gemini-2.0-flash-001"  # Model used to test if API key is valid


# Safety settings
safety_settings = [
    # SafetySetting(
    #     category=HarmCategory.HARM_CATEGORY_UNSPECIFIED,
    #     threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
    # ),
    # SafetySetting(
    #     category=HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
    #     threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
    # ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
    ),
]

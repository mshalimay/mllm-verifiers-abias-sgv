# Prompter
from typing import Any

ROLE_MAPPINGS = {
    "assistant": "assistant",
    "user": "user",
    "system": "system",  # Obs.: new role is `developer`, but system is backward and forward compatible
    "developer": "system",
}
DEFAULT_HF_MODE = "chat_completion"


# VLLM defaults
VLLM_TENSOR_PARALLEL_SIZE = 1
VLLM_MAX_MODEL_LEN: int | None = None
VLLM_GPU_MEMORY_UTILIZATION = 0.9
VLLM_QUANTIZE = False
VLLM_ENFORCE_EAGER = False

# VLLM OpenAI Compatile server
VLLM_OPENAI_KEY = "EMPTY"
ENDPOINT_TEMPLATE = "http://{host}:{port}/v1"
VLLM_USE_V1 = 1
VLLM_DEFAULT_PARAMS_PER_MODEL = {
    "Qwen2.5-VL": {
        "repetition_penalty": 1.05,
    },
    "Qwen3-VL": {
        # https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
        "repetition_penalty": 1.0,
        "presence_penalty": 1.5,
        "top_k": 20,
    },
}


# Flash attention
USE_FLASH_ATTN = True

# API call generation (OpenRouter, alibaba, other OpenAI compatible providers)

API_PROVIDERS_ARGS = {
    "openrouter": {
        "max_api_key_retry": 10,
        "max_key_process_count": 100,
        "max_requests_per_minute": 20,
        "throttled_sleep": 30,
    },
    "alibaba": {
        "max_requests_per_minute": 40,  # https://www.alibabacloud.com/help/en/model-studio/rate-limit
        "throttled_sleep": 50,
        "max_key_process_count": 100,
        "max_api_key_retry": 10,
    },
    "default": {
        "max_api_key_retry": 10,
        "max_key_process_count": 1,
        "max_requests_per_minute": 20,
        "throttled_sleep": 30,
    },
}

MODEL_SPECIFIC_ARGS = {
    "qwen3-vl": {
        "max_image_size_mb": {
            "alibaba": 10,  # Max image size is 10mb: https://www.alibabacloud.com/help/en/model-studio/error-code. See Exceeded limit on max bytes per data-uri item
            "default": None,
        }
    },
    "Llama-4-Maverick": {
        "max_image_size_mb": {
            "together_ai": 10,
            "default": None,
        }
    },
    "qwen2.5-vl": {
        "max_image_size_mb": {
            "together_ai": 10,
            "openrouter": 10,
            "default": None,
        },
        "max_image_dims": {
            "together_ai": (512, 512),
            "openrouter": (512, 512),
            "default": None,
        },
    },
}


def get_api_provider_arg(provider: str, key: str) -> Any:
    provider_args = API_PROVIDERS_ARGS.get(provider, API_PROVIDERS_ARGS["default"])
    arg_value = provider_args.get(key, API_PROVIDERS_ARGS["default"].get(key))
    return arg_value

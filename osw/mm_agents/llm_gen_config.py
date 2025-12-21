# Use this for OpenRouter hosted model
gen_kwargs_openrouter = {
    "provider": "huggingface",
    "engine": "openai",
    "metadata": {
        "base_url": "https://openrouter.ai/api/v1",
        "provider": "openrouter",
    },
    "model": "bytedance/ui-tars-1.5-7b",
}


## Use this for VLLM hosted locally
gen_kwargs_vllm = {
    "provider": "huggingface",
    "engine": "vllm",
    "endpoint": "127.0.0.1:8000",
    "model": "ByteDance-Seed/UI-TARS-1.5-7B",
}

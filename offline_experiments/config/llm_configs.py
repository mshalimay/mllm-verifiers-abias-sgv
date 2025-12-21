# fmt:off

llm_configs = {
    "gpt-4.1-apr-zero-random": {
        "model": "gpt-4.1-2025-04-14",
        "temperature": 1,
        "top_p": 0.001,
        "max_tokens": 8192,
    },
    "gpt-4.1-apr-base": {
        "model": "gpt-4.1-2025-04-14",
        "temperature": 1,
        "top_p": 0.9,
        "max_tokens": 8192,
    },
    "gpt-4o-aug-zero-random": {
        "model": "gpt-4o-2024-08-06",
        "temperature": 1,
        "top_p": 0.001,
        "max_tokens": 8192,
    },
    "gpt-4o-aug-base": {
        "model": "gpt-4o-2024-08-06",
        "temperature": 1,
        "top_p": 0.9,
        "max_tokens": 8192,
    },
    "gemini-2.0-flash-001_base": {
        "model": "gemini-2.0-flash-001",
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_tokens": 8192,
        "num_generations": 1,
        "thinking_budget": 0,
    },
    "gemini-2.0-flash-001_zero-random": {
        "model": "gemini-2.0-flash-001",
        "temperature": 1,
        "top_p": 0.001,
        "top_k": 64,
        "max_tokens": 8192,
        "num_generations": 1,
        "thinking_budget": 0,
    },
    "qwen-2.5-72b-base": {
        # "model": "qwen/qwen2.5-vl-72b-instruct",
        "model": "Qwen/Qwen2.5-VL-72B-Instruct",
        "engine": "openai",
        "metadata": {
            # "base_url": "https://openrouter.ai/api/v1",
            "base_url": "https://api.together.xyz/v1",
            "provider": "together_ai",
            # "provider": "openrouter",
            # "api_extra_args": {"provider": {"only": ["novita/bf16"]}},
        },
        "num_generations": 1,
        "temperature": 0.7,
        "frequency_penalty": 1.05,
        "max_tokens": 8192,
    },
    "llama-4-maverick-base": {  # https://github.com/meta-llama/llama-models/blob/01dc8ce46fecf06b639598f715efbb4ab981fb4c/models/llama4/scripts/chat_completion.py#L30
        "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "engine": "openai",
        "metadata": {
            "base_url": "https://api.together.xyz/v1",
            "provider": "together_ai",
        },
        "num_generations": 1,
        "temperature": 0.6,
        "top_p": 0.9,
        "top_k": 64,
        "max_tokens": 8192,
    },
    "llama-4-maverick-zero-random": {
        "model": "meta-llama/llama-4-maverick-17b-128e-instruct",
        "engine": "openai",
        "metadata": {
            "base_url": "https://api.together.xyz/v1",
            "provider": "together_ai",
        },
        "temperature": 0,
        "top_p": 0,
        "num_generations": 1,
        "max_tokens": 8192,
    },


# Gemini 2.5
    "gemini-2.5-flash-base": {
        "model": "gemini-2.5-flash",
        "temperature": 1,
        "top_p": 0.9,
        "top_k": 64,
        "max_tokens": 8192,
        "num_generations": 1,
        "thinking_budget": 0,
    },
    "gemini-2.5-flash-zero-random": {
        "model": "gemini-2.5-flash",
        "temperature": 1,
        "top_p": 0.001,
        "top_k": 64,
        "max_tokens": 8192,
        "num_generations": 1,
        "thinking_budget": 0,
    },
    "gemini-2.5-flash-zero-random_web-search": {
        "model": "gemini-2.5-flash",
        "temperature": 1,
        "top_p": 0.001,
        "top_k": 64,
        "max_tokens": 8192,
        "num_generations": 1,
        "thinking_budget": 0,
        "tools": ["web_search"],
    },
    "gemini-2.5-flash-thinking-high-base": {
        "model": "gemini-2.5-flash",
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_tokens": 65535,
        "num_generations": 1,
        "thinking_budget": 24576,
    },
    "gemini-2.5-flash-thinking-high-base_web-search": {
        "model": "gemini-2.5-flash",
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_tokens": 65535,
        "num_generations": 1,
        "thinking_budget": 24576,
        "tools": ["web_search"],
    },
    "gemini-2.5-flash-thinking-high-zero-random": {
        "model": "gemini-2.5-flash",
        "temperature": 0.1,
        "top_p": 0.001,
        "top_k": 60,
        "max_tokens": 65535,
        "num_generations": 1,
        "thinking_budget": 24576,
    },
    "gemini-2.5-flash-thinking_auto-zero-random": {
        "model": "gemini-2.5-flash",
        "temperature": 0.1,
        "top_p": 0.001,
        "top_k": 64,
        "max_tokens": 65535,
        "num_generations": 1,
        "thinking_budget": -1,
    },
    "gemini-2.5-flash-thinking_auto-majority": {
        "model": "gemini-2.5-flash",
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_tokens": 65535,
        "num_generations": 8,
        "thinking_budget": -1,
    },
    "gemini-2.5-flash-majority": {
        "model": "gemini-2.5-flash",
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_tokens": 8192,
        "num_generations": 8,
        "thinking_budget": 0,
    },
    "gemini-2.5-flash-majority-high-temp": {
        "model": "gemini-2.5-flash",
        "temperature": 1.5,
        "top_p": 0.95,
        "top_k": 64,
        "max_tokens": 8192,
        "num_generations": 8,
        "thinking_budget": 0,
    },
    "gemini-2.5-flash-high-temp": {
        "model": "gemini-2.5-flash",
        "temperature": 1.5,
        "top_p": 0.95,
        "top_k": 64,
        "max_tokens": 8192,
        "num_generations": 1,
        "thinking_budget": 0,
    },
    "gemini-2.0-flash-001-high-temp": {
        "model": "gemini-2.0-flash-001",
        "temperature": 1.5,
        "top_p": 0.95,
        "top_k": 64,
        "max_tokens": 8192,
        "num_generations": 1,
        "thinking_budget": 0,
    },
    "gemini-2.5-flash-thinking-low-zero-random": {
        "model": "gemini-2.5-flash",
        "temperature": 0.1,
        "top_p": 0.001,
        "top_k": 64,
        "max_tokens": 65535,
        "num_generations": 1,
        "thinking_budget": 8192,
    },
    # Gemini 2.5 thinking high
    "gemini-2.5-flash-thinking-high-low-random": {
        "model": "gemini-2.5-flash",
        "temperature": 1,
        "top_p": 0.3,
        "top_k": 64,
        "max_tokens": 65535,
        "num_generations": 1,
        "thinking_budget": 24576,
    },


# Qwen 3 VL
# https://github.com/QwenLM/Qwen3-VL
# https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html#:~:text=multi%2Dturn%20conversations.-,Note,-For%20thinking%20mode
    "qwen3-vl-instruct-base": {
        "model": "qwen3-vl-235b-a22b-instruct",
        "engine": "openai",
        "metadata": {
            "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            "provider": "alibaba"
        },
        # qwen recommendations:
        "num_generations": 1,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "frequency_penalty": 1.0,
        "presence_penalty": 1.5,
        "max_tokens": 8192,
    },

    "qwen3-vl-thinking-base": {
        "model": "qwen3-vl-235b-a22b-thinking",
        "engine": "openai",
        "metadata": {
            "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            "provider": "alibaba"
        },
        # qwen recommendations:
        "num_generations": 1,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "frequency_penalty": 1.0,
        "presence_penalty": 0.0,
        "max_tokens": 8192,  # max is 131,072 // obs.: does not limit thinking budget, only response after <think>
        "extra_body":
            {"thinking_budget": 81920} # max is 81,920
    },

     "qwen3-vl-32b-instruct-base": {
        "model": "qwen3-vl-32b-instruct",
        "engine": "openai",
        "metadata": {
            "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            "provider": "alibaba"
        },
        # qwen recommendations:
        "num_generations": 1,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "frequency_penalty": 1.0,
        "presence_penalty": 1.5,
        "max_tokens": 8192,
    },
     
    "qwen3-vl-32b-thinking-base": {
        "model": "qwen3-vl-32b-thinking",
        "engine": "openai",
        "metadata": {
            "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            "provider": "alibaba"
        },
        # qwen recommendations:
        "num_generations": 1,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "frequency_penalty": 1.0,
        "presence_penalty": 0.0,
        "max_tokens": 8192,  # max is 131,072 // obs.: does not limit thinking budget, only response after <think>
        "extra_body":
            {"thinking_budget": 81920} # max is 81,920
    },
     "qwen3-vl-8b-instruct-base": {
        "model": "qwen3-vl-8b-instruct",
        "engine": "openai",
        "metadata": {
            "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            "provider": "alibaba"
        },
        # qwen recommendations:
        "num_generations": 1,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "frequency_penalty": 1.0,
        "presence_penalty": 1.5,
        "max_tokens": 8192,
    },
     
    "qwen3-vl-8b-thinking-base": {
        "model": "qwen3-vl-8b-thinking",
        "engine": "openai",
        "metadata": {
            "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            "provider": "alibaba"
        },
        # qwen recommendations:
        "num_generations": 1,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "frequency_penalty": 1.0,
        "presence_penalty": 0.0,
        "max_tokens": 8192,  # max is 131,072 // obs.: does not limit thinking budget, only response after <think>
        "extra_body":
            {"thinking_budget": 81920} # max is 81,920
    },


    "gemini-2.5-flash-preview-sep-zero-random": {
        "model": "gemini-2.5-flash-preview-09-2025",
        "temperature": 1,
        "top_p": 0.001,
        "top_k": 64,
        "max_tokens": 8192,
        "num_generations": 1,
        "thinking_budget": 0,
    },
    
    "gemini-2.5-pro-base-thinking-high": {
        "model": "gemini-2.5-pro",
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_tokens": 65535,
        "num_generations": 1,
        "thinking_budget": 32768,
    },

    "gemini-2.5-flash-lite-zero-random": {
        "model": "gemini-2.5-flash-lite",
        "temperature": 1,
        "top_p": 0.001,
        "top_k": 64,
        "max_tokens": 8192,
        "thinking_budget": 0,
        "num_generations": 1,
    },
    "gemini-2.5-flash-lite-high-temp": {
        "model": "gemini-2.5-flash-lite",
        "temperature": 1.5,
        "top_p": 0.95,
        "top_k": 64,
        "max_tokens": 8192,
        "num_generations": 1,
        "thinking_budget": 0,
    },
    "gpt-5-mini-high": {
        "model": "gpt-5-mini-2025-08-07",
        "reasoning_effort": "high",
        "mode": "response",
    },
    "gpt-5-nano-high": {
        "model": "gpt-5-nano-2025-08-07",
        "reasoning_effort": "high",
        "mode": "response",
    },
    "gpt-4.1-mini-apr-zero-random": {
        "model": "gpt-4.1-mini-2025-04-14",
        "temperature": 1,
        "top_p": 0.001,
        "max_tokens": 8192,
    },

    "gpt-o3-high": {
        "model":"o3-2025-04-16",
        "reasoning_effort":"high",
        "mode":"response",        
    },

    "gpt-o3-mini-high": {
        "model":"o3-mini-2025-01-31",
        "reasoning_effort":"high",
        "mode":"response",        
    },
    
    "gpt-o4-mini-high": {
        "model":"o4-mini-2025-04-16",
        "reasoning_effort":"high",
        "mode":"response",        
    },

  
}

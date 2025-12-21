# README

## 1) Summary
This is a set of tools facilitate inference calls to MLLMs across:
- Model providers (e.g.: OpenAI , HuggingFace) and model families, 
- Deployment engines (e.g.: VLLM, OpenRouter, local uvicorn hosted server), 
- API modes (OpenAI chat completions and response API)

Using a uniform format for messages and generation args and automatic handling of errors, API keys, etc.

The idea is to receive a list of multimodal generations from any model by simply providing a list of raw inputs (in flexible format) and a dict of generation arguments, abstracting away any other complexities.

Like below:
```
inputs = [
    PIL_image, 
    text, 
    {"role": "system", "text": "You are a helpful assistant"}, 
    ["image 1", "test_llm/dog.png"],
    numpy_image,
    url_image,
    video,
]

gen_args = {
    "model": "gemini-2.0-flash-001",
    "temperature": 1.0,
    "top_p": 0.9,
    "top_k": 40,
    "num_generations": 2
}

api_responses, model_messages = call_llm(gen_kwargs, inputs)
```

Main components:
1) Simple and flexible input-output format. In essence:
- (i) Provide a list of raw inputs (PIL/Numpy/URL/path images, text, videos, dictionaries) 
- (ii) Provide a dictionary of generation arguments (temperature, num_generations, etc).
- (iii) Receive a list of outputs (text, images, function calls, etc) back.

2) Automatic logging relevant information, handling errors, etc
Behind the scenes, during an inference call the tools also:
- Perform retries with exponential backoff handling provider-specific errors
- Manages and balance load among API keys in case of more than one
- Dynamically upload prompts to the cloud in case of large payloads and automatically retry
- Logging visual representations of the conversation
- Logging token usage
- Log timings for specific calls
- etc

3) Homogeneous representation for elements of a conversation and generation arguments.
- Any transformations needed to inputs to necessary for a provider, model, engine is done behind the scenes
- Similarly, any unsupported parameters for a model or provider are pruned away

## 2) Setting API keys
API keys can be set via environment variables or stored in a protected JSON file.

### 2.1) Option 1: Set API keys via environment variables:
```bash
export GOOGLE_API_KEY="your-google-api-key"
export OPENAI_API_KEY="your-openai-api-key"
export OPENROUTER_API_KEY="your-openrouter-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export HF_TOKEN="your-huggingface-token"
export DASHSCOPE_API_KEY="your-alibaba-api-key"
export TOGETHER_API_KEY="your-together-api-key"
```

### 2.2) Option 2: Store API keys in a protected JSON file:
If API keys are stored in `llms/api_keys.json`, the code will automatically load them and rotate keys if you have multiple keys for the same provider.
See `llms/api_keys_template.json` for a template.

**NOTES**:
- Protect the file permissions to avoid exposing them. Recommended: `chmod 600 llms/api_keys.json`.
- Create a `llms/api_keys_backup.json` file containing your API keys. During rotation, the code may remove keys from `llms/api_keys.json`. Although there are safeguards to re-add them, keep the backup file in case something goes wrong.
- Be aware of the security risks of storing API keys in files and make sure to add `llms/api_keys.json` and `llms/api_keys_backup.json` to `.gitignore`.
- The paths can be customized in `llms/constants/constants.py` API_KEYS_PATH and API_KEYS_BACKUP.

## 3) Additional notes

1. See `llms/examples/examples.ipynb` for a series of examples on how to use the tools, and `llms/examples/example.py` for a simple llm call.


2. This is package is constantly evolving. Most of the relevant utilities are working, however:
    - HuggingFace models have specific quirks. The tools make a best effort to support them, but may need to change the code in `llms/providers/hugging_face` to support a new model (typically one or two lines).
    - The repository may have some bugs and non-implemented arguments for particular providers/models
    - More documentation will be added in the future
    - Video, function calling, computer use are not fully tested.

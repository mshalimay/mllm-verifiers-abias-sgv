## README

This file contains instructions to run experiments on the OSWorld benchmarks.

Original instructions can be found at: [OSWorld](https://github.com/xlang-ai/OSWorld). 

# 1) Python Setup
1. Create a virtual environment using your preferred method.

    ```shell
    # Python venv:
    python3.11 -m venv abias-sgv
    source abias-sgv/bin/activate
    
    # Conda:
    conda create --name abias-sgv python=3.11  
    conda activate abias-sgv
    ```

2. Install required dependencies. From the repository's root:


# 2) Running experiments
Our main experiments utilize the UI-TARS-1.5-7B GUI agent. To use the model, you have two options:
(i) Host the model using vLLM, or (ii) Use an external model provider.

### 2.1) Host the agent using vLLM. (skip if using an external model provider).

    ```bash
    cd osw
    python host_vllm.py
    ```

    This will host the UI-TARS-1.5-7B model using vLLM. Requires ~24 GB VRAM with the current settings.

### 2.2) Set up the API keys for model providers.
The code in `llms` supports multiple model providers and engines, including OpenAI, Google, Alibaba, HuggingFace, vllm, and others. Before running experiments, make sure to set up your API keys for the providers you plan to use.

Two options are available:
1. Set the API keys via environment variables. Example:
    ```bash
    export GOOGLE_API_KEY="your-google-api-key"
    export OPENAI_API_KEY="your-openai-api-key"
    export OPENROUTER_API_KEY="your-openrouter-api-key"
    ```
    The experiments in OSWorld utilize the `gemini-2.5-flash` model for the verifier. Therefore, set up your API keys for Gemini by:

    ```bash
    export GOOGLE_API_KEY="your-google-api-key"
    ```

    The code is set up to use `ui-tars-1.5-7b` for the agent with openrouter. If you are not using vllm, set up your API keys for OpenRouter by:
    ```bash
    export OPENROUTER_API_KEY="your-openrouter-api-key"
    ```


2. Store keys in a protected json file. If API keys are stored in `llms/api_keys.json`, the code will automatically load them and rotate keys if you have multiple for the same provider. See `llms/api_keys_template.json` for a template.
   - **Note**: be aware of the security risks of storing API keys in files and make sure to add `llms/api_keys.json` to `.gitignore`.
   - See `llms/README.MD` for other instructions and details.


### 2.3) Sequential mode
To run experiments without environment parallelism, use the following command:
```bash
cd osw
python run_uitars.py --run_name <run_name> --max_steps <max_steps> --verifier <verifier> --verify_every_n_steps <verify_every_n_steps> --vllm <vllm>
```

Where:
- `<run_name>`: The name of the experiment in the results directory. Default: current timestamp.
- `<max_steps>`: Maximum number of steps per task. Default: 50.
- `<verifier>`: Defines the verifier implementation to use. Options: `two_pass` (SGV), `one_pass` (base MLLM verifier), `none` (no verifier). Default: `two_pass`.
- `<verify_every_n_steps>`: If set to a value `n` greater than 0, call the verifier every `n` steps. If set to 0, only call the verifier when the agent signals task completion. Default: 5.
- `<vllm>`: If set to `True`, use vLLM for UI-TARS-1.5-7B inference. Default: `False`.



### 2.4) Parallel mode
To run experiments with environment parallelism, use the following command:

```bash
./run_uitars_parallel.sh --run_name "1p_verify_every_5" --max_steps 50 --verifier one_pass --verify_every_n_steps 5 --test_all_meta_path evaluation_examples/test_all.json
```

Where:
- `<run_name>`: The name of the experiment in the results directory. Default: current timestamp.
- `<max_steps>`: Maximum number of steps per task. Default: 50.
- `<verifier>`: Defines the verifier implementation to use. Options: `two_pass` (SGV), `one_pass` (base MLLM verifier), `none` (no verifier). Default: `two_pass`.
- `<verify_every_n_steps>`: If set to a value `n` greater than 0, call the verifier every `n` steps. If set to 0, only call the verifier when the agent signals task completion. Default: 5.
- `<test_all_meta_path>`: Path to the OSW tasks to run. Default: `evaluation_examples/test_all.json`.



# 3) Other

## 3.1) Summarize results
The file `summarize_results.py` can be used to aggregate the results and output the success rates to the console and a `summary.txt` file in the results directory.
```bash
cd osw
python summarize_results.py <run_name>
```
Where: `<run_name>`: The name of the experiment in the results directory.

## 3.2) Visualize trajectories
Use the script `utils_osw/render_trajectories.py` to render `.json` trajectories to an HTML file for visualization.

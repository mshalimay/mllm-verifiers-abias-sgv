# README
This file contains instructions to run experiments on the (Visual)WebArena benchmarks. The steps were adapted to this codebase, which includes code to simplify environment setup, reset websites/cookies properly and automatically, etc.

Original instructions can be found at: [visualwebarena](https://github.com/web-arena-x/visualwebarena), [webarena](https://github.com/web-arena-x/webarena). 

Modifications to the original codebase are documented in `docs_env_updates.md`.

## 1) Python Setup
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

    ```bash
    pip install -e ".[vwa]"
    playwright install
    python -c "import nltk; nltk.download('punkt')"
    ```

## 2) Set up the (Visual)WebArena environments
To run experiments, it is necessary to set up the environments included in (Visual)WebArena.
The below sets up the environments using our provided scripts to donwload files, load images into Docker, etc. If you prefer to set up manually, see `environment_docker/README.MD`.

1. Download and load the necessary files:
    ```bash
    cd vwa
    # For all environments required in VisualWebArena:
    python -m scripts.environments.download_load_envs all_vwa 
    # Or shell: ./scripts/environments/download_load_envs.sh all_vwa
    ```
    **Important**: this step needs to be done **only one time**, and takes a while.

2. (Optional) You can test spawning websites by running the following command. This will start the websites in the list and print their links and statuses:
    ```bash
    cd vwa
    python -m scripts.environments.start_reset_envs -p 1 shopping reddit
    # or bash: ./scripts/environments/start_reset_envs.sh -p 1 shopping reddit
    # Obs.: -p is optional. It specifies an ID for the set of websites to keep them isolated. If not provided, a new ID will be created.

    # Output:
    # ...
    # > reddit-1        UP         URL: http://localhost:64840
    # > shopping-1      UP         URL: http://localhost:64841
    ```

3. (Optional) You can also check the status of the websites by running:
    ```bash
    ./vwa/scripts/environments/check_websites.sh -p 1
    # Output:
    # > reddit-1        UP         URL: http://localhost:64840
    # > shopping-1      UP         URL: http://localhost:64841
    ```

You can visit the websites in the browser by opening the URLs printed in the output.

Notes:
- The environment files are large so setting up the environments can take a while.
- Files are saved in `./environment_docker/docker_imgs`. If you need to change this path (e.g., storage restrictions), just modify `DOCKER_IMGS_PATH` in `vwa/benchmark_config/constants.py`
- The current codebase handles environments automatically (e.g., reset/start envs, etc). If you prefer do to some steps manually, see [link](environment_docker/README.md) for manual alternatives.
- See [5.4](#54-general-usage-of-environment-related-scripts) for general usage of environment-related scripts.

## 3) MLLM Provider API keys
The code in `llms` supports multiple model providers and engines, including OpenAI, Google, Alibaba, HuggingFace, vllm, and others. Before running experiments, make sure to set up your API keys for the providers you plan to use.

Two options are available:
1. Set the API keys via environment variables. Example:
    ```bash
    export GOOGLE_API_KEY="your-google-api-key"
    export OPENAI_API_KEY="your-openai-api-key"
    export OPENROUTER_API_KEY="your-openrouter-api-key"
    ```

2. Store keys in a protected json file. If API keys are stored in `llms/api_keys.json`, the code will automatically load them and rotate keys if you have multiple for the same provider. See `llms/api_keys_template.json` for a template.
   - **Note**: be aware of the security risks of storing API keys in files and make sure to add `llms/api_keys.json` to `.gitignore`.
   - See `llms/README.MD` for other instructions and details.


## 4) Running experiments
**Only proceed after setting up the environments as explained in (2).**

### 4.1) Sequential mode 
To run experiments without environment parallelism, use `run.py` or the wrapper `scripts/runs/run.sh`, which contains an extensive set of options with corresponding comments.

To replicate the experiments in the paper:

```shell
cd vwa
# Full VisualWebArena (910 tasks)
scripts/runs/run.sh -a <agent_config_file>

# VisualWebArena lite (305 tasks)
scripts/runs/run.sh -a <agent_config_file> -t evaluation_harness/task_lists/all_vwa-lite.txt 
```

Where `<agent_config_file>` is the path to the agent configuration file. The following are available in `agent/config/`:
- `agent/config/noverifier.yaml`: Base agent without verification
- `agent/config/nosgv.yaml`: Agent paired with the base MLLM verifier (outcome-based verifier, no SGV)
- `agent/config/sgv.yaml`: Agent paired with SGV (outcome-based verifier)
- `agent/config/sgv_every5.yaml`: Agent paired with SGV, periodic verification (every 5 steps)
- `agent/config/nosgv_every5.yaml`: Agent paired with the base MLLM verifier, periodic verification (every 5 steps)
- `agent/config/reflexion_oracle.yaml`: Reflexion. Oracle verifier for reflection generation
- `agent/config/reflexion_nosgv.yaml`: Reflexion. Base MLLM verifier for reflection generation
- `agent/config/reflexion_sgv.yaml`: Reflexion. SGV verifier for reflection generation

### 4.2) Parallel mode
The script `scripts/runs/p_run.py` configures and runs experiments in parallel, with each docker container running in a separate process.

The file is already set to run the base agent on VisualWebArena lite (see [4.3](#43-tasks-configs-and-visualwebarena-lite)). Simply run the following command to run in parallel:
```shell
cd vwa
# run in detached mode (recommended):
nohup python -u -m scripts.runs.prun > prun.log 2>&1 < /dev/null & disown

# foreground mode:
# python -u -m scripts.runs.prun
```
Check the log files created to see progress and results.

For other configurations, edit `prun.py` options for agents, tasks and other as instructed in the file.

Notes:
- Parallel experiments are run by: (i) splitting the tasks into batches, (ii) creating a new tmux session, and (iii) executing a separate `run.sh` instance within a tmux pane for each batch. 
- Therefore, configurations in `run.sh` will apply to each batch.
- The code uses a lazy initialization of environments, keeping them ready as new batches start. The number of processes and environments per process can be configured in `prun.py`.

### 4.3) Tasks configs and VisualWebArena lite
The configuration files for (Visual)WebArena are stored in `config_files/`. The file `test_vwa.raw.json` contains all tasks for VisualWebArena. 

The file `test_vwa_lite.raw.json` contains a subset of tasks in VisualWebArena that provides similar performance signals as the full benchmark.

You can specify the tasks considered for evaluation by either providing the `.json` directly or a `.txt` file containing the task IDs. In `evaluation_harness/task_lists/`, you can find the task lists for the full (Visual)WebArena lite and the subsets for each domain. 

See `scripts/runs/run.sh` for other options and details on how to specify the tasks considered for evaluation.
Also see `vwa/docs_env_updates.md` for more details on the changes relative to the original VisualWebArena codebase.

## 5) Other (Optional)

### 5.1) Customizing Agents
To customize the agents and model configurations, edit the configuration files in `agent/config/`. Below is a general description. Check the files in the folder for several examples. 

- The `agent.yaml` file, specifies each module used to build the agent. 
    - For example, the `executor` module is responsible for interacting with the environment, and the `verifier` module is the MLLM verifier that evaluates the agent's actions.
- If the modules are built using MLLMs, the `lm_config` field specifies which model and generation parameters to use for that module.
- The `lm_config.yaml` file centralizes the generation parameters for all models used in the agents.

For example, if `agent/config/agent.yaml` looks like this:
```yaml
executor:
  action_set_tag: som
  prompt: p_som_cot_id_actree_3s_prev_utterances  
  lm_config: # Model specific configs
    model: 'gpt-4o-2024-08-06'
    gen_config_alias: base

verifier:
  ...
  lm_config: # Model specific configs
    model: 'llava-next-1.6'
    gen_config_alias: low_random
```

The agent built will use `gpt-4o-2024-08-06` as the executor model with generation parameters defined under the `base` alias in `agent/config/lm_configs.yaml`, and `llava-next-1.6` as the verifier model with generation parameters defined under the `low_random` alias in `agent/config/lm_configs.yaml`. 

The `agent/config/lm_configs.yaml` should have an entry for the corresponding models and aliases, for example:

```yaml
gpt-4o-2024-08-06
  base:
    temperature: 1.0
    top_p: 0.9
    max_tokens: 768
    name_user: 'user'
    name_assistant: 'assistant'
    img_detail: 'auto'
llava-next-1.6
  low_random:
    max_tokens: 768

  base: # Base will not be used for the agent above; just an example
    temperature: 0.7
    top_p: 0.9
    ...
```

**Notes:**
- If certain parameters are not specified, the default values for the corresponding model will be used.
- Don't worry about including parameters that the models may not support (like `top_k` for GPT). The `llms` utilities will remove them if needed.
- You can add more models and aliases as needed. See `lm_config.yaml` for examples.

### 5.2) Human-readable trajectories
Use the script `trajectory_utils/render_trajectory.py` to render `.json` trajectories to an HTML file for visualization.
```bash
cd vwa
python trajectory_utils/render_trajectory.py <base_dir> [options]
```
Where:
- `<base_dir>`: The base directory containing the trajectories.
- [options]: See `trajectory_utils/render_trajectory.py` for options and examples.

### 5.3) Deploy the Captioner manually
(Visual)WebArena uses a BLIP2-based captioner to (i) support VQA-based evaluation and (ii) caption images included in tasks/pages.

The original codebase loads the model inside each run process. When running multiple experiments in parallel, this duplicates GPU/CPU memory usage and can become a bottleneck.

For this reason, this codebase can host the captioner as a server so all runs can share a single captioner instance.

The runner scripts will automatically start the captioner if it is not detected. To start it manually (default port: `9555`):

```shell
# from the `vwa/` directory
tmux new -s vwa_captioner
python utils_vwa/host_captioner.py
```

To detach from `tmux` without stopping the server, press `Ctrl-b` then `d`. To re-attach later: `tmux attach -t vwa_captioner`.


### 5.4) General usage of environment-related scripts

1. Download and load environment files:
    ```bash
    python -m scripts.environments.download_load_envs <site_1> ... <site_n>
    # where:
    # - <site_1> ... <site_n> are the websites to download and load. 
    # - sites in ['shopping' 'admin' 'reddit' 'gitlab' 'classifieds' 'wikipedia' 'homepagevwa' 'homepagewa' 'all_vwa' 'all_wa']
    # Special: all_vwa, all_wa (all VisualWebArena and WebArena websites, respectively)
    ``` 
2. Start and reset environments:
    ```bash
    python -m scripts.environments.start_reset_envs [-p <env_id>] <site_1> ... <site_n>
    # where:
    # - <site_1> ... <site_n> are the websites to start/reset. 
    # - sites in ['shopping' 'admin' 'reddit' 'gitlab' 'classifieds' 'wikipedia' 'homepagevwa' 'homepagewa' 'all_vwa' 'all_wa']
    # Special: all_vwa, all_wa (all VisualWebArena and WebArena websites, respectively)
    # - (optional) <env_id> is an integer ID for the set of websites. Used mostly in parallel execution to keep environments isolated.
    ```

3. Check website statuses
```bash
./vwa/scripts/environments/check_websites.sh [-p <env_id>]
# where:
# - (optional) <env_id> is an integer ID for the set of websites. If not provided, checks all environments running.

# Output:
# > reddit-1        UP         URL: http://localhost:64840
# > shopping-1      UP         URL: http://localhost:64841
```

4. Clean docker leftovers
```bash
./vwa/scripts/environments/clean_containers.sh [-p <env_id>]
# where:
# - (optional) <env_id> is the integer ID for the set of websites. If not provided, identifies all (Visual)WebArena environments and asks for confirmation before cleaning.
```

# Acknowledgements

This code makes extensive use of the original codebases of
[VisualWebArena](https://github.com/web-arena-x/visualwebarena),
[WebArena](https://github.com/web-arena-x/webarena), and
[Search-Agents](https://github.com/kohjingyu/search-agents).
We thank the authors of these projects for making their work publicly available.
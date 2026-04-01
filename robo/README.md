# README

This directory contains the code for running experiments on the RoboMimic benchmark (tool-hang task).

Original RoboMimic codebase: [RoboMimic](https://github.com/ARISE-Initiative/robomimic).

## 1) Installation

1. Follow the root [README](../README.md) to create a virtual environment and install base dependencies:

    ```bash
    pip install -e ".[robo]"
    ```

2. Clone and install the RoboMimic and RoboSuite dependencies:

    ```bash
    cd robo
    git clone https://github.com/ARISE-Initiative/robomimic robomimic
    pip install -e robomimic/
    git clone https://github.com/ARISE-Initiative/robosuite robosuite
    pip install -e robosuite/
    ```

3. Download policy models:
MLP, RNN, and diffusion models are available in this [link](https://drive.google.com/drive/folders/1HP8bsg0Z_U1bgb4sGIPYJgvYNNJkaZew?usp=sharing). 

For the experiments in the paper, replace `robo/models/tool_hang/ph/diffusion` with the corresponding directory provided in the link above.

4. Set up macros for both robomimic and robosuite (from `robo/`):

    ```bash
    cd robo
    python robo/robomimic/scripts/setup_macros.py
    python robo/robosuite/scripts/setup_macros.py
    ```

5. Run the unit tests to verify the setup (from `robo/`):

    ```bash
    cd robo
    python -m unittest
    ```

## 2) Experiments

Set up API keys:
```bash
export OPENAI_API_KEY="your-openai-api-key"
```


Then run experiments with or without the VLM verifier:

```bash
# Move to the robo directory
cd robo

# Without MLLM-based verifier
python run_vlm.py -m mlp
python run_vlm.py -m rnn
python run_vlm.py -m diffusion

# With MLLM-based verifier (SGV 2 steps)
python run_vlm.py -m rnn --vlm_name gpt-4o
python run_vlm.py -m diffusion --vlm_name gpt-4o

# With MLLM-based verifier (No SGV)
python run_vlm.py -m diffusion --vlm_name gpt-4o_1pass
```


## 3) Training (Optional)

If you'd like to train your own models:

- (i) Pre-trained MLP, RNN, and diffusion models are available at this [link](https://drive.google.com/drive/folders/1HP8bsg0Z_U1bgb4sGIPYJgvYNNJkaZew?usp=sharing). 

- (ii) To retrain models:

```bash
cd robo
python run.py -m mlp
python run.py -m rnn
python run.py -m diffusion
```


## 4) Notes
- This setup requires a Linux environment with MuJoCo for the RoboMimic/RoboSuite simulator.

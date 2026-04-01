## Let's Think in Two Steps: Mitigating Agreement Bias in MLLMs with Self-Grounded Verification

Official codebase accompanying the paper: "Let's Think in Two Steps: Mitigating Agreement Bias in MLLMs with Self-Grounded Verification".

<p align="center">
    🌐 <a href="https://mshalimay.github.io/agreement-bias-sgv/" target="_blank">Project</a> | 📃 <a href="https://arxiv.org/abs/2507.11662" target="_blank">Paper</a> | 🗂️ <a href="https://drive.google.com/drive/folders/15nAWNkyo6Jhsxl6RZg8cFRS_6jP-Xr-H?usp=drive_link" target="_blank">Data</a> <br>
</p>



## Project Structure

- **`core_utils/`**: General utilities for image handling, logging, file management, locking mechanisms, and more.
- **`llms/`**: Utilities to call models from multiple providers, batch inference, homogenize prompts and outputs, and more.
- **`vwa/`**: (Visual)WebArena updated environment. Contains: 
  - (i) implementation of our agents, reflexion, SGV; 
  - (ii) utilities to facilitate environment setup, proper environment reset, parallel execution, VisualWebArena-lite task subset, and more. 
  - (iii) documentation about changes to the environment.
- **`osw/`**: OSWorld codebase. Contains implementation of UI-Tars paired with MLLM verifiers and SGV.
- **`offline_experiments/`**: Evaluation of MLLMs as verifiers across models, prompt templates, test-time scaling, trajectory configurations, image annotations, environments, and more.
- **`robo/`**: robomimic-related code and models.

## Installation
1. Clone the repository:
```bash
git clone https://github.com/mshalimay/mllm-verifiers-abias-sgv
```

(Optional) To clone specific directories only, use sparse checkout:
```bash
# For (Visual)WebArena only:
git sparse-checkout init --cone
git sparse-checkout set core_utils llms vwa
git checkout main

# For OSWorld only:
git sparse-checkout init --cone
git sparse-checkout set core_utils llms osw
git checkout main

# For robomimic only:
git sparse-checkout init --cone
git sparse-checkout set core_utils llms robo
git checkout main
```

2. Install dependencies.

Create a virtual environment with your preferred method:

```bash
# Use python >= 3.11.
# Python venv:
python3.11 -m venv abias-sgv
source abias-sgv/bin/activate

# Conda:
conda create --name abias-sgv python=3.11  
conda activate abias-sgv
```

Install the desired dependencies:

```bash
# For all environments and experiments:
pip install -e ".[vwa,osw,offline,robo]"

# For (Visual)WebArena only:
pip install -e ".[vwa]"

# For OSWorld only:
pip install -e ".[osw]"

# For offline experiments only:
pip install -e ".[offline,vwa]"

# For robomimic only:
pip install -e ".[robo]"

# For LLMs utilities only:
pip install -e .
```

Then, follow the instructions in the README files of `vwa/`, `osw/`, `offline_experiments/`, and `robo/` directories for any additional steps to replicate the experiments.


## Data
- Trajectories utilized for the offline experiments are available at this [link](https://drive.google.com/drive/folders/1S8ic6JJ3h-iDucwsstgJJrDfelMILmBr?usp=drive_link). Download the directory to the root as `data`. See `offline_experiments/README.MD` for more details.

- SGV trajectories referenced in VisualWebArena leaderboard are available at this [link](https://drive.google.com/drive/folders/1iMpYpzKQQsWS46OmHgsr1T2yq6BdnICQ?usp=drive_link). 

- OSWorld trajectories with SGV are available at this [link](https://drive.google.com/drive/folders/1kzxyKcJMuz3FChPb375fTev-Qu5RJcZC?usp=drive_link).

- robomimic models are available at this [link](https://drive.google.com/drive/folders/1HP8bsg0Z_U1bgb4sGIPYJgvYNNJkaZew?usp=sharing).

For other data and trajectories, please contact the authors.

## Citation
If you find this useful and would like to cite this work, please use the following.


```bibtex
@misc{andrade2025letsthinkstepsmitigating,
      title={Let's Think in Two Steps: Mitigating Agreement Bias in MLLMs with Self-Grounded Verification}, 
      author={Moises Andrade and Joonhyuk Cha and Brandon Ho and Vriksha Srihari and Karmesh Yadav and Zsolt Kira},
      year={2025},
      eprint={2507.11662},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2507.11662}, 
}
```


## Acknowledgements
This project is built upon the following codebases:

- For (Visual)WebArena: [VisualWebArena](https://github.com/web-arena-x/visualwebarena),
  [WebArena](https://github.com/web-arena-x/webarena), and [Search-Agents](https://github.com/kohjingyu/search-agents).

- For OSWorld: [OSWorld](https://github.com/xlang-ai/OSWorld), and [UI-TARS](https://github.com/bytedance/UI-TARS).

- For robomimic: [RoboMimic](https://github.com/ARISE-Initiative/robomimic).

We thank the authors of these projects for making their work publicly available.

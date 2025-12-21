# fmt:off
# This template generates first-step responses from multiple models and uses them to generate the second-step response.
# Everything else is the same as `offline_experiments/config/experiment_config/gemini-2.5-flash.py`.


# If task list is provided, only evaluate for tasks in the list
task_list_vwa = ""
# task_list_vwa = "offline_experiments/config/task_lists/vwa_lite.txt" # Runs only on VisualWebArena-lite subset
task_list_osw = ""
task_list_agrb_vwa = ""

# Models for first-step:
models_fp = [
    "gemini-2.5-flash-zero-random",
    "gpt-5-mini-high",
    "qwen3-vl-instruct-base",
]

# model_v is the model used for Verification.
model_v = "gemini-2.5-flash-zero-random"
# See all available model aliases in llm_configs.py


# Append ann string to output dirs containing LLM outputs for first step and verification step
#  e.g.: if original dir = "outputs/gemini-25-flash" and ann_v = "_ann1" => final dir = "outputs/gemini-25-flash_ann1"
ann_v = ""
ann_k = ""

# Deployment configuration
batch_mode = True               # If True, build batch LLM inputs then send to `max_api_keys` LLM processes for generation. Else, build batchs sequentially and send when > sequential_min_fire_batch
max_batch_runners = 30          # Maximum number of LLM inputs to build before sending to LLM processes
sequential_min_fire_batch = 5   # Minimum number of LLM inputs to build before sending to LLM processes in sequential mode
max_api_keys = 2               # Distribute LLM inputs to `max_api_keys` LLM processes for generation in parallel

overwrite = False               # If False, skip tasks that have already been run
dump_html = True                # If True, also dump LLM conversations as HTML files (good for visualization, but takes more space)


# --------------------------------------------------------------------------------
# Configurations for each evaluation case
# --------------------------------------------------------------------------------
# CASES specifies:
# - how prompts are constructed for each evaluation.
# - what image annotations to use
# - what trajectories to evaluate on

# For each dictionary in the list, separate sets of experiments are run for the product of
# (eval_criterias * sys_prompts * cot_parts * rules * trace_infos * img_ann_types * trajectories_data * k_configs)


# ann_types: ["som", "soom_coord", "coord"]                                                     # annotate screenshots with different types of image annotations
# cot_parts: ["no_cot", "basic_cot"]; see prompts_osw.py or prompts_vwa.py for all options      # add a CoT instruction to the prompt
# eval_criterias: ["bin", "tri", "four_num_unc"]; see common_prompts.py for all options         # what evaluation template to use in the prompt

# trace_info: ["utt_actions", "actions", "utt", "none"]                                         # actions = provide only low-level actions; utt_actions = provide agent's utterance and low-level actions
   # if 'shuffle' is True, random shuffle (state-actions) order in the prompt
   # if idxs is non-empty, only include (state-actions) corresponding to idxs in the prompt (e.g.: [0,2] => first and third step in the trajectory)

traj_data_vwa = [{"agent_id": "gemini-2.5-sep25"}]
traj_data_agrb = [{"agent_id": "agrb_claude-3.7"}, {"agent_id": "agrb_gpt4o"}, {"agent_id": "agrb_qwen2.5-vl"}]
traj_data_osw = [{"agent_id": "uitars_50s_apr25"}]


CASES = [
    # VWA
    {
        "2p": {
            "eval_criterias": ["tri"],
            "sys_prompts": ["base"],
            "cot_parts": [
                "basic_cot",
            ],
            "rules": ["rule_2p_k"],
            "trace_infos": [{"type": "actions", "idxs": []}],
            "img_ann_types": ["som"],
            "k_configs": [
                 [
                    {"k_prompt_id": "k_2p_expert", "fp_model": models_fp[0], "img_ann_types": "", "additional_config": {"caption_input_images": False, "condense_k_mode": "multiprior"}, "k_max": 3},
                    {"k_prompt_id": "k_2p_expert", "fp_model": models_fp[1], "img_ann_types": "", "additional_config": {"caption_input_images": False, "condense_k_mode": "multiprior"}, "k_max": 3},
                    {"k_prompt_id": "k_2p_expert", "fp_model": models_fp[2], "img_ann_types": "", "additional_config": {"caption_input_images": False, "condense_k_mode": "multiprior"}, "k_max": 3},
                ]
            ],
            "trajectories_data": traj_data_vwa,
            "additional_config": {"caption_input_images": True, "multimodel_k":True},
        }
    },

    # OSW
    {
        "2p": {
            "eval_criterias": ["tri"],
            "sys_prompts": ["base"],
            "cot_parts": [
                "basic_cot",
            ],
            "rules": ["rule_2p_k"],
            "trace_infos": [{"type": "utt_actions", "idxs": []}],
            "img_ann_types": ["coord"],
            "k_configs": [
                [
                    {"k_prompt_id": "k_2p_expert", "fp_model": models_fp[0], "img_ann_types": "", "additional_config": {"caption_input_images": False, "condense_k_mode": "multiprior"}, "k_max": 3},
                    {"k_prompt_id": "k_2p_expert", "fp_model": models_fp[1], "img_ann_types": "", "additional_config": {"caption_input_images": False, "condense_k_mode": "multiprior"}, "k_max": 3},
                    {"k_prompt_id": "k_2p_expert", "fp_model": models_fp[2], "img_ann_types": "", "additional_config": {"caption_input_images": False, "condense_k_mode": "multiprior"}, "k_max": 3},
                ]
            ],
            "trajectories_data": traj_data_osw,
            "additional_config": {"caption_input_images": False, "multimodel_k":True},
        }
    },
]
sort_by_config = True
overwrite = False
batch_mode = True
dump_html = False

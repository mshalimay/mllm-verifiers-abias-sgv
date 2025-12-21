# fmt:off
# This template can be used for all non-thinking models. Simply change the `model_fp` and `model_v` lines to the desired model. All model/gen configs are in `llm_configs.py`.


# If task list is provided, only evaluate for tasks in the list
task_list_vwa = "offline_experiments/config/task_lists/vwa_lite.txt"
# task_list_vwa = "offline_experiments/config/task_lists/debug_vwa.txt"
task_list_osw = ""
task_list_agrb_vwa = ""

# Models for first and second step. See llm_configs.py for all options.
model_fp = "gemini-2.5-flash-zero-random"
# model_fp = "qwen3-vl-instruct-base" You can use a different model for the first and second step by changing this line.
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
# VWA or AGRB-VWA
    {
        "1p_no_k": {
            "eval_criterias": ["tri"],
            "sys_prompts": ["base"],
            "cot_parts": ["basic_cot"],
            "rules": ["rule_1p"],
            "trace_infos": [{"type": "actions", "idxs": [], "shuffle": False}],
            "img_ann_types": ["som"],
            "trajectories_data": traj_data_vwa + traj_data_agrb,
            # "trajectories_data":traj_data_agrb,
            # "trajectories_data": traj_data_vwa,
            "additional_config": {"caption_input_images": True}, # Uses the VisualWebArena default captioner to caption images included in the intents.
        }
    },
    {
        "2p": {
            "eval_criterias": ["tri"],
            "sys_prompts": ["base"],
            "cot_parts": [
                "basic_cot",
                # "desc_compare", # Adds a comparison to the first step generation. See prompts_vwa.py for all options.
            ],
            "rules": ["rule_2p_k"],
            "trace_infos": [{"type": "actions", "idxs": []}],
            "img_ann_types": ["som"],
            "k_configs": [
                [
                    {"k_prompt_id": "k_2p_expert", "fp_model": model_fp, "img_ann_types": "", "additional_config": {"caption_input_images": False}},
                ]
            ],
            "trajectories_data": traj_data_vwa + traj_data_agrb,
            # "trajectories_data": traj_data_vwa,
            "additional_config": {"caption_input_images": True},
        }
    },
    
## OSW
    {
        "1p_no_k": {
            "eval_criterias": ["tri"],
            "sys_prompts": ["base"],
            "cot_parts": [
                "basic_cot",
            ],
            "rules": ["rule_1p"],
            "trace_infos": [{"type": "utt_actions", "idxs": [], "shuffle": False}],
            "img_ann_types": ["coord"],
            "trajectories_data": traj_data_osw,
        }
    },
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
            "trajectories_data": traj_data_osw,
            "k_configs": [[{"k_prompt_id": "k_2p_expert", "fp_model": model_fp, "img_ann_types": ""}]],
        }
    },
]

results_dir_base = "offline_experiments/results"
out_dir_template = f"{results_dir_base}/{{env}}/{{agent_id}}/{{model_name}}/{{full_config_name}}/{{domain}}"

all_env_configs = {
    "vwa": {
        "domains": ["classifieds", "shopping", "reddit"],
        "trajectories_data": [
            {
                "agent_id": "gemini-2.5-30s",
                "source_file_list": "offline_experiments/gold_scores/vwa/gemini-2.5-30s.csv",
            },
            {
                "agent_id": "gemini-2.5-60s",
                "source_file_list": "offline_experiments/gold_scores/vwa/gemini-2.5-60s.csv",
            },
        ],
    },
    "osw": {
        "domains": [
            "chrome",
            "gimp",
            "libreoffice_calc",
            "libreoffice_impress",
            "libreoffice_writer",
            "multi_apps",
            "os",
            "thunderbird",
            "vlc",
            "vs_code",
        ],
        "trajectories_data": [
            {"agent_id": "uitars_50s_apr25", "source_file_list": "offline_experiments/gold_scores/osw/uitars_50s_apr25.csv"},
        ],
    },
    "agrb_vwa": {
        "domains": ["classifieds", "shopping", "reddit"],
        "trajectories_data": [
            {"agent_id": "agrb_claude-3.7", "source_file_list": "offline_experiments/gold_scores/agrb_vwa/agrb_claude-3.7.csv"},
            {"agent_id": "agrb_gpt4o", "source_file_list": "offline_experiments/gold_scores/agrb_vwa/agrb_gpt4o.csv"},
            {"agent_id": "agrb_qwen2.5-vl", "source_file_list": "offline_experiments/gold_scores/agrb_vwa/agrb_qwen2.5-vl.csv"},
        ],
    },
    "agrb_wa": {
        "domains": ["classifieds", "shopping", "gitlab", "shopping_admin", "map"],
        "trajectories_data": [],
    },
}
DOMAINS = [
    "shopping",
    "classifieds",
    "reddit",
    "chrome",
    "gimp",
    "libreoffice_calc",
    "libreoffice_impress",
    "libreoffice_writer",
    "multi_apps",
    "os",
    "thunderbird",
    "vlc",
    "vs_code",
]
VWA_DOCKER_INSTANCE_ID = 90

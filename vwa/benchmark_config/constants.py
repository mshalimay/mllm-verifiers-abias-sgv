from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.resolve().as_posix()


# ===============================================================================
# Constants for run.py
# ===============================================================================
# Paths for Agents and MLLM configs
AGENTS_CONFIG_DIR = f"{ROOT_DIR}/agent/config"

# Paths for outputs
DEFAULT_RESULTS_DIR = f"{ROOT_DIR}/results"
RESULT_DIR_TEMPLATE = "{results_dir}/{model}/{annotation}"
HTMLS_SUBDIR: str = "htmls"  # If '', saves HTML files directly to args.result_dir
TRAJECTORIES_SUBDIR = "trajectories"
TRAJ_DIR_TEMPLATE = "{result_dir}/" + TRAJECTORIES_SUBDIR + "/{task_id}"


# ===============================================================================
# (Visual)WebArena Eval Configuration - Edit as needed
# ===============================================================================
# Websites hosted at "http://<base_url>:<...>"
BASE_URL = "localhost"


# Maps the corresponding (Visual)WebArena domain to a real URL to be displayed to Agents
# e.g.: "http://localhost:9000" -> "http://reddit.com", etc.
URLS = {
    "homepage": ["http://homepage.com"],
    "shopping": ["http://onestopmarket.com"],
    "reddit": ["http://reddit.com"],
    "classifieds": ["http://classifieds.com"],
    "shopping_admin": ["http://luma.com/admin"],
    "gitlab": ["http://gitlab.com"],
    "map": ["http://openstreetmap.org"],
    "wikipedia": ["http://wikipedia.org"],
}
# Token used to reset classifieds website
CLASSIFIEDS_RESET_TOKEN = "4b61655535e7ed388f0d40a93600254c"

# Paths with input images to build intent for each test config file
LOCAL_INPUT_IMAGES_DIR_TEMPLATE = f"{ROOT_DIR}/config_files/vwa_input_images/{{domain}}/task_{{task_id}}"

# Models used by the environment oracle evaluators for tasks using Fuzzy Match evaluation
DEFAULT_FUZZY_MATCH_MODEL = "gpt-5-2025-08-07"

# Captioner configuration
CAPTIONER_HOST_SCRIPT = Path(ROOT_DIR, "utils_vwa/host_captioner.py").as_posix()
CAPTIONER_PORT = 9555
CAPTIONER_ENDPOINT = f"http://localhost:{CAPTIONER_PORT}/caption/"
CAPTIONER_SUPPORTED_MODELS = ["Salesforce/blip2-flan-t5-xl", "Salesforce/blip2-flan-t5-xxl", "Qwen/Qwen3-VL-8B-Instruct"]
CAPTIONER_DEFAULT_IMAGE_FORMAT = "PNG"

# ===============================================================================
# Paths for environment Docker images and related resources
# ===============================================================================
# The directory where the docker images and `.zim` file for the Wikipedia website are located
# Change this path if you want to keep the docker images and `.zim` file in a different directory.
DOCKER_IMGS_PATH = f"{ROOT_DIR}/environment_docker/docker_imgs"

# Paths to scripts used to start/reset/clean environments
RESET_ENV_SCRIPT = f"{ROOT_DIR}/scripts/environments/start_reset_envs.py"
# RESET_ENV_SCRIPT = f"{ROOT_DIR}/scripts/environments/start_reset_envs.sh" # if you prefer to use a bash script
CLEAN_ENV_SCRIPT = f"{ROOT_DIR}/scripts/environments/clean_containers.sh"

# Directory and file template to save URLs of existing websites
URLS_DIR = Path(ROOT_DIR, "browser_env", "docker_urls").as_posix()
URLS_FILE_TEMPLATE = f"{URLS_DIR}/urls-{{PROCESS_ID}}.txt"

# Directory to save cookies
AUTH_DIR = Path(ROOT_DIR, ".auth").as_posix()

# The directories where the files to render the homepage are saved
HOMEPAGE_PATH_WA = f"{ROOT_DIR}/environment_docker/webarena-homepage"
HOMEPAGE_PATH_VWA = f"{ROOT_DIR}/environment_docker/vwebarena-homepage"

# The directory where the classifieds website Docker Compose files are located
# (usually no need to change, change DOCKER_IMGS_PATH instead)
CLASSIFIEDS_DOCKER_COMPOSE_DIR = f"{DOCKER_IMGS_PATH}/classifieds_docker_compose"

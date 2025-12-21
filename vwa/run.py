import sys

DEBUG_ENABLED = sys.gettrace() is not None
if DEBUG_ENABLED:
    print("\n====================\n DEBUG MODE ACTIVATED\n====================\n")

import argparse
import copy
import json
import logging
import os
import random
import shutil
import time
import traceback
import warnings
from collections import deque
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

from agent.agent_utils import get_agent_config

# Agent
from agent.modular_agent import ModularAgent, construct_modular_agent

# Constants
from benchmark_config import AGENTS_CONFIG_DIR, AUTH_DIR, DEFAULT_RESULTS_DIR, HTMLS_SUBDIR, RESULT_DIR_TEMPLATE
from benchmark_config.constants import TRAJ_DIR_TEMPLATE

# Environment
from browser_env import ActionTypes, ScriptBrowserEnv, StateInfo, Trajectory, create_stop_action
from browser_env.actions import Action
from browser_env.env_utils import build_task_config
from browser_env.helper_functions import RenderHelper, get_action_description
from evaluation_harness import evaluator_router
from trajectory_utils.trajectory import VWA_Trajectory
from utils_vwa.captioner_utils import define_captioning_fn
from utils_vwa.utils_vwa import TrajectoryView, early_stop, maybe_reset_environments, reset_cookies_with_retry

from core_utils import timing_utils as timer

# Utilities
from core_utils.data_recorder import DataRecorder
from core_utils.eval_utils import set_seed
from core_utils.file_utils import make_json_serializable
from core_utils.logger_utils import logger, save_log_file, save_log_file_path
from core_utils.signal_utils import signal_manager
from core_utils.string_utils import safe_format
from core_utils.timing_utils import dump_timings, set_timings_global_id
from llms.setup_utils import restore_api_keys_to_file

# Filter warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# warnings.filterwarnings("ignore", category=BeartypeDecorHintPep585DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning, message="`resume_download` is deprecated and will be removed in version 1.0.0.")
warnings.filterwarnings("ignore", category=UserWarning, message='Field "model_id" has conflict with protected namespace "model_".')

# Handler for termination signals
signal_manager.add_cleanup_function(restore_api_keys_to_file)


def set_debug_inputs(args: argparse.Namespace) -> None:
    # @debug
    if not DEBUG_ENABLED:
        return

    args.skip_env_reset = True
    args.skip_cookie_reset = True
    args.agent_config_file = "noverifier.yaml"

    args.task_list = "./evaluation_harness/task_lists/all_vwa-lite.txt"
    args.fuzzy_match_model = "gemini-2.5-flash"

    args.manual_input = True
    domain = "reddit"
    args.test_config_json_file = f"config_files/vwa/test_{domain}.raw.json"
    args.test_start_idx = 10  # inclusive
    args.test_end_idx = 10  # inclusive
    args.sleep_after_execution = 0.0
    args.log_html = True
    args.repeating_action_failure_th = 30

    args.log_trajectory = True
    args.show_scroll_bar = True
    args.require_all_sites_up = 0
    args.docker_instance_id = 90
    args.force_reset = False
    args.max_steps = 30

    args.render = False

    args.agent_captioning_model_device = "server-cuda"  # 'cuda'
    args.eval_captioning_model_device = "server-cuda"  # 'cuda'

    args.viewport_width = 1280  # Default: 1280
    args.viewport_height = 2048  # Default: 720 for small context window models | 2048 for large context window models

    args.render_screenshot = True
    args.observation_type = "image_som"  # accessibility_tree, image_som, accessibility_tree_with_captioner, html, image

    # less used
    args.result_dir = "results/debug"
    args.slow_mo = 0

    if "debug" in args.result_dir and os.path.exists(os.path.abspath(args.result_dir)):
        shutil.rmtree(os.path.abspath(args.result_dir))
        os.makedirs(os.path.abspath(args.result_dir))


# ===============================================================================
# CMD arguments
# ===============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end evaluation on the benchmark")

    # --- Agent configuration ---
    parser.add_argument(
        "--agent_config_file",
        type=str,
        default="agent_config_base.yaml",
        help="Filename for the YAML config file.",
    )

    # --- Subdir to store outputs ---
    parser.add_argument(
        "--result_dir",
        type=str,
        nargs="?",
        const="",
        default="",
    )

    # --- Execution configuration ---
    parser.add_argument(
        "--manual_input",
        action="store_true",
        help="Low level action input comes from the user.",
    )

    parser.add_argument(
        "--sleep_after_execution",
        type=float,
        default=0.0,
        help="If >0, automatically wait for pages to stabilize then sleep up to this amount of seconds after environmenment step.",
    )

    # --- Task configuration ---
    parser.add_argument(
        "--test_config_json_file",
        nargs="?",
        type=str,
        default="",
    )
    parser.add_argument(
        "--test_start_idx",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--test_end_idx",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--task_list",
        nargs="?",
        const=None,
        type=str,
        default=None,
    )
    parser.add_argument(
        "--max_tasks",
        type=int,
        default=None,
        help="Maximum number of tasks to evaluate.",
    )
    parser.add_argument(
        "--shuffle_tasks",
        action="store_true",
        help="Shuffle the task list.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Set seed for testing.",
    )

    # --- Environment configuration ---
    parser.add_argument(
        "--env",
        type=str,
        default="vwa",
        help="Environment to run the test on.",
        choices=["vwa", "wa"],
    )
    parser.add_argument(
        "--force_reset",
        action="store_true",
        help="Force reset websites for each task before its execution.",
    )
    parser.add_argument(
        "--skip_env_reset",
        action="store_true",
        help="Skip resetting the environment for each task. Useful for debugging.",
    )
    parser.add_argument(
        "--skip_cookie_reset",
        action="store_true",
        help="Skip resetting the cookies for each task. Useful for debugging.",
    )
    parser.add_argument(
        "--require_all_sites_up", type=int, choices=[0, 1], default=1, help="Require all sites to be running for evaluation."
    )

    parser.add_argument(
        "--docker_instance_id",
        type=str,
        default="0",
        help="Instance ID for homepage websites and env containers.",
    )

    # --- Captioner configuration ---
    parser.add_argument(
        "--agent_captioning_model",
        type=str,
        default="",
        help="Captioning model to use for an agent.",
    )
    parser.add_argument(
        "--agent_captioning_model_device",
        type=str,
        default="server-cuda",
        help="Device to run captioning model on.",
    )

    parser.add_argument(
        "--eval_captioning_model",
        type=str,
        default="Salesforce/blip2-flan-t5-xl",
        choices=["Salesforce/blip2-flan-t5-xl", "llava-hf/llava-1.5-7b-hf"],
        help="Captioning backbone for VQA-type evals.",
    )

    parser.add_argument(
        "--eval_captioning_model_device",
        type=str,
        default="server-cuda",
        help="Device to run eval captioning model on.",
    )

    # --- Observation configuration ---
    parser.add_argument(
        "--observation_type",
        choices=[
            "accessibility_tree",  # text: actree of the webpages
            "accessibility_tree_with_captioner",  # text: actree of the webpages; caption model generates description to images and add to actree
            "image_som",  # image: SOM-marked screenshot of webpage; text: ID of items marked in the screenshot
            "html",  # NOTE: (V)WA codebase doesn't have action parser for this, even though it's allowed
            "image",  # NOTE: (V)WA codebase doesn't have action parser for this, even though it's allowed
        ],
        default="accessibility_tree",
        help="Observation type",
    )

    parser.add_argument(
        "--show_scroll_bar",
        action="store_true",
        help="Show the scroll bar in the observation",
    )

    parser.add_argument(
        "--current_viewport_only",
        action="store_true",
        help="Only use the current viewport for the observation",
    )
    parser.add_argument("--viewport_width", type=int, default=1280)
    parser.add_argument("--viewport_height", type=int, default=2048)

    # --- Evaluation configuration ---
    parser.add_argument(
        "--parsing_failure_th",
        type=int,
        default=3,
        help="When consecutive parsing failure exceeds this threshold, the agent will stop",
    )

    parser.add_argument(
        "--repeating_action_failure_th",
        type=int,
        default=30,
        help="When consecutive repeated actions exceed this threshold, the agent will terminate early.",
    )

    parser.add_argument(
        "--fuzzy_match_model",
        type=str,
        default="gemini-2.5-flash",
        help="LLM model for fuzzy matching evaluation.",
    )

    parser.add_argument("--max_steps", type=int, default=30, help="Max number of environment steps allowed. If exceeded, FAIL.")

    # --- Outputs/results configuration ---
    parser.add_argument("--render_screenshot", action="store_true")
    parser.add_argument("--log_html", action="store_true")

    # real-time browser rendering
    parser.add_argument("--render", action="store_true", help="Shows the browser in the screen.")
    parser.add_argument("--slow_mo", type=int, default=0, help="Slow down the browser by the specified amount")

    parser.add_argument("--log_trajectory", action="store_true", help="Log trajectory to a JSON file.")

    parser.add_argument("--save_trace_enabled", action="store_true", help="Save playwright traces in the result directory.")

    args = parser.parse_args()

    return args


def maybe_create_subdir(result_dir: str, domain: str) -> str:
    # Create the result directory, if not exists
    result_dir = os.path.join(result_dir, domain)
    os.makedirs(result_dir, exist_ok=True)

    return result_dir


def build_test_config_list(args: argparse.Namespace) -> list[str]:
    test_config_list = []

    if args.task_list:
        logger.info(f"\nExecuting tasks from: {args.task_list}")

        with open(args.task_list, "r") as f:
            # json is first line
            lines = f.readlines()
            json_file_path = lines[0].strip()
            args.test_config_json_file = json_file_path
            initial_task_ids = [line.strip() for line in lines[1:]]
    else:
        json_file_path = args.test_config_json_file
        if not json_file_path:
            raise ValueError("Either --task_list or --test_config_json_file must be provided.")
        initial_task_ids = [str(i) for i in range(args.test_start_idx, args.test_end_idx + 1)]

    if not Path(json_file_path).is_file():
        raise FileNotFoundError(f"File {json_file_path} not found")

    # Open json file and get the task_ids
    with open(json_file_path, "r") as f:
        all_task_data = json.load(f)
    domains = set([task["domain"] for task in all_task_data])

    test_config_list = []
    # NOTE: allowing both numeric lists of task_ids and domain_task_ids for backwards compatibility with (Visual)WebArena
    for task_id in initial_task_ids:
        if task_id.isdigit():
            key = "uid" if len(domains) > 1 else "task_id"
            task = next((task for task in all_task_data if str(task[key]) == str(task_id)), None)
            if task is None:
                logger.warning(f"Task {task_id} not found in {json_file_path}")
                continue
            test_config_list.append(task)
        else:
            key = "domain_task_id"
            if any(task[key] == task_id for task in all_task_data):
                test_config_list.append(next(task for task in all_task_data if task[key] == task_id))
            else:
                logger.warning(f"Task {task_id} not found in {json_file_path}")

    num_tasks = len(test_config_list)
    if num_tasks == 0:
        logger.info("No task left to run")
        sys.exit(0)

    # Print the number of tasks to run
    logger.info(f"Running {num_tasks} tasks")

    if args.max_tasks is not None and args.max_tasks > 0 and num_tasks > args.max_tasks:
        logger.info(f"`max_tasks` set to {args.max_tasks}. Evaluating the first {args.max_tasks} tasks of {num_tasks}.")
        test_config_list = test_config_list[: args.max_tasks]

    if args.shuffle_tasks:
        logger.info(f"Shuffling tasks. Seed: {args.seed}")
        random.shuffle(test_config_list)

    return test_config_list


# Pre-process args
def preprocess_args(args: argparse.Namespace) -> None:
    if args.agents_configs.get("executor", {}).get("action_set_tag"):
        args.action_set_tag = args.agents_configs["executor"]["action_set_tag"]

    # check the whether the action space is compatible with the observation space
    # REVIEW: removed `som_image`` because `id_accessibility_tree` with `som_image` gives wrong hint in get_action_description(..) (see the function for more details)
    if args.action_set_tag == "id_accessibility_tree" and args.observation_type not in [
        "image",
        "accessibility_tree",
        "accessibility_tree_with_captioner",
    ]:
        raise ValueError(f"Action type {args.action_set_tag} is incompatible with the observation type {args.observation_type}")

    if args.action_set_tag == "som" and args.observation_type not in ["image_som"]:  # image is not supported yet
        raise ValueError(f"Action type {args.action_set_tag} is incompatible with the observation type {args.observation_type}")

    if args.observation_type in ["accessibility_tree_with_captioner"] and args.captioning_model is None:
        args.captioning_model = "Salesforce/blip2-flan-t5-xl"
        logging.warning(
            f"Observation type `{args.observation_type}` requires captioner but none provided. Captioning model set to {args.captioning_model} (default)"
        )

    # ---------------------------------------------------------------------------
    # Results and log directory paths
    # ---------------------------------------------------------------------------
    # Define a default `args.result_dir` if not provided
    if not args.result_dir:
        top_dir = DEFAULT_RESULTS_DIR
        # Annotate model and current date and time
        date_annotate = datetime.now().strftime("%Y-%m-%d-%H%M")
        model = args.agents_configs["executor"]["lm_config"]["model"]  # NOTE: change if using multiple models
        args.result_dir = safe_format(RESULT_DIR_TEMPLATE, results_dir=top_dir, model=model, annotation=date_annotate)
        # Example: results/gpt-4o-mini-2024-07-18/2025-02-22-1430

    args.result_dir = os.path.abspath(args.result_dir)

    os.makedirs(args.result_dir, exist_ok=True)

    if args.task_list:
        # Save task list to results directory
        src_path = os.path.abspath(args.task_list)
        dest_path = os.path.abspath(os.path.join(args.result_dir, os.path.basename(args.task_list)))
        if src_path != dest_path:
            shutil.copyfile(src_path, dest_path)

    args.test_config_list = build_test_config_list(args)


# ===============================================================================
# Evaluation
# ===============================================================================


def error_handler(error: Exception, result_dir: str, data_recorder: DataRecorder, config_id: str) -> None:
    logger.error(repr(error), exc_info=True)
    with open(Path(result_dir) / "error.txt", "a") as f:
        # Dump config file content
        if config_id:
            f.write(f"[Config ID]: {config_id}\n")

        f.write(f"[Error trace] {repr(error)}\n")
        f.write(traceback.format_exc())  # write stack trace to file
    data_recorder.num_failed_executions += 1


def initialize_data_loggers(
    args: argparse.Namespace,
    data_recorder: DataRecorder,
    config_dict: dict[str, Any],
    result_dir: str,
    task_id: str,
    intent: str,
    intent_image_paths: list[str],
    domain: str,
    env_name: str,
    uid: str,
) -> tuple[RenderHelper | None, VWA_Trajectory | None]:
    """
    Initialize data loggers for HTML rendering and trajectory logging.

    Args:
        args: Command line arguments
        data_recorder: Data recorder instance
        config_dict: Task configuration dictionary
        htmls_dir: Directory for HTML files
        result_dir: Results directory
        task_id: Task identifier
        intent: Task intent/objective
        image_paths: List of image paths
        domain: Task domain
        env_name: Environment name
        uid: Unique identifier for the task

    Returns:
        tuple: (render_helper, traj_logger)
    """
    render_helper = None
    traj_logger = None

    # Data logging
    if args.log_html:
        htmls_dir = Path(result_dir) / HTMLS_SUBDIR
        htmls_dir.mkdir(parents=True, exist_ok=True)
        render_helper = RenderHelper(
            config_file=config_dict,
            result_dir=htmls_dir,  # type: ignore
            action_set_tag=args.action_set_tag,
            postfix=args.attempts_per_task[uid],
        )

    if args.log_trajectory:
        output_dir = Path(TRAJ_DIR_TEMPLATE.format(result_dir=result_dir, task_id=f"{task_id}_{args.attempts_per_task[uid]}"))
        filename = f"trajectory-{domain}-{task_id}.json"

        traj_logger = VWA_Trajectory(
            task_id=task_id,
            objective_text=intent,
            objective_imgs=intent_image_paths,
            domain=domain,
            config=config_dict,
            args=f"{args.result_dir}/args.json",
            file_path=str(output_dir / filename),
        )

    data_recorder.initialize_task(
        task_id=task_id,
        domain=domain,
        env=env_name,
        sites=config_dict["sites"],
        attempt_id=args.attempts_per_task[uid],
        traj_json_file=traj_logger.file_path if traj_logger is not None else "",
    )

    return render_helper, traj_logger


def init_environment(args: argparse.Namespace) -> ScriptBrowserEnv:
    return ScriptBrowserEnv(
        headless=not args.render,
        slow_mo=args.slow_mo,
        observation_type=args.observation_type,
        current_viewport_only=args.current_viewport_only,
        viewport_size={
            "width": args.viewport_width,
            "height": args.viewport_height,
        },
        save_trace_enabled=args.save_trace_enabled,
        sleep_after_execution=args.sleep_after_execution,
        captioning_fn=args.caption_img_fn,
        show_scroll_bar=args.show_scroll_bar,
    )


def maybe_generate_reflection(
    agent: ModularAgent,
    trajectory_copy: TrajectoryView | None,
    intent: str,
    intent_images: list[Any],
    meta_data: dict[str, Any],
    score: float,
    uid: str,
) -> tuple[bool | None, bool, str, str, dict[str, Any]]:
    if agent.reflexion is not None and trajectory_copy is not None:
        should_retry = False
        v_feedback: str = ""
        v_response: str = ""
        v_parsed_response: dict[str, Any] = {}
        if agent.reflexion.get_eval_mode() == "model":
            is_success, v_feedback, v_response, v_parsed_response = agent.get_verifier_response(
                trajectory_copy, intent, intent_images, meta_data
            )
        elif agent.reflexion.get_eval_mode() == "oracle":
            is_success = True if score == 1 else False
        else:
            raise ValueError(f"Unknown eval mode: {agent.reflexion.get_eval_mode()}")

        if is_success is None:
            raise ValueError(f"Error during reflexion: verifier {agent.reflexion.get_eval_mode()} didn't return evaluation. Config: {uid}")

        if not is_success:
            agent.generate_reflection(trajectory_copy, intent, intent_images, meta_data)
            should_retry = True
        return is_success, should_retry, v_feedback, v_response, v_parsed_response
    else:
        return None, False, "", "", {}


def generate_action(
    agent: ModularAgent,
    trajectory_view: TrajectoryView,
    intent: str,
    intent_images: list[Any],
    meta_data: dict[str, Any],
    terminated: bool,
    trajectory: Trajectory,
    early_stop_thresholds: dict[str, Any],
) -> tuple[Action, bool]:
    if terminated:
        action = create_stop_action("")
        action["stop_reason"] = "Execution terminated by environment."  # type: ignore
        logger.info("Execution terminated by environment.")
        return action, False

    # Else, generate action
    try:
        # If early stop threshold reached, issue STOP action
        early_stop_flag, stop_info = early_stop(trajectory, early_stop_thresholds)
        if early_stop_flag:
            action = create_stop_action(stop_info)
            action["stop_reason"] = stop_info  # type: ignore
            logger.info(f"Early stop: {stop_info}")

        # Else, Get action on environment
        else:
            action = agent.next_action(
                trajectory_view,
                intent,
                intent_images=intent_images,  # type: ignore
                meta_data=meta_data,
            )

            if "early_stop" in action:
                early_stop_reason = action["early_stop"]  # type: ignore
                action = create_stop_action(action["raw_prediction"])
                action["early_stop"] = early_stop_reason  # type: ignore
                action["stop_reason"] = early_stop_reason  # type: ignore
                early_stop_flag = True
                logger.info(f"Early stop: {early_stop_reason}")
    except Exception as e:
        raise e

    return action, early_stop_flag


def test_one(
    args: argparse.Namespace,
    agent: ModularAgent,
    data_recorder: DataRecorder,
    env: ScriptBrowserEnv,
    raw_config_dict: dict[str, Any],
) -> tuple[bool, bool]:
    env_name = raw_config_dict["env"]
    domain = raw_config_dict["domain"]
    task_id = raw_config_dict["task_id"]
    config_str = f"[Config]: env: {env_name}, domain_task_id: {domain}_{task_id}"
    uid = f"{domain}_{task_id}_{env_name}"
    test_execution_success = should_retry = False
    args.attempts_per_task[uid] = args.attempts_per_task.get(uid, -1) + 1

    logger.info(config_str)
    result_dir = maybe_create_subdir(args.result_dir, domain)
    traj_logger, render_helper = None, None
    try:
        # Reset for classifieds is lightweight if using classifieds reset token. Forcing it.
        if domain == "classifieds":
            force_reset = True
            force_reset_site_list = ["classifieds"]
        else:
            force_reset = args.force_reset
            force_reset_site_list = []
        reset_success, sites_failed_reset = maybe_reset_environments(
            instance_id=args.docker_instance_id,
            env=args.env,
            config_dict=raw_config_dict,
            require_all_sites_up=args.require_all_sites_up,
            force_reset=force_reset,
            force_reset_site_list=force_reset_site_list,
            skip_reset=args.skip_env_reset,
        )
        if not reset_success:
            logger.error(f"Failed to reset environments for {sites_failed_reset}, {config_str}.")
            raise Exception(f"Failed to reset environments for {sites_failed_reset}, {config_str}")

        # Build task config from raw config file
        config_dict, intent, task_id, intent_images, intent_img_paths = build_task_config(raw_config_dict, args.docker_instance_id)

        # Reset cookies for current task
        if not args.skip_cookie_reset:
            reset_cookies_success, msg = reset_cookies_with_retry(
                docker_instance_id=args.docker_instance_id,
                sites=config_dict["sites"],
                expired_only=False,
                exc_comb=False,
                wait_for_cookies_reset=True,
                auth_folder=AUTH_DIR,
            )
            if not reset_cookies_success:
                logger.error(f"Failed to reset cookies: {msg}. {config_str}", exc_info=True)
                raise Exception(f"Failed to reset cookies: {msg}. {config_str}")

        # Prepare agent
        agent.reset(config_dict, {**vars(args), "result_dir": result_dir, "uid": uid})

        # If reflexion enabled, populate reflection memory and update attempt count
        if agent.reflexion is not None:
            args.attempts_per_task[uid] = agent.reflexion.get_num_attempts_per_task(uid)
            if args.attempts_per_task[uid] > agent.reflexion.max_reflexion_attempts:
                logger.info(f"Max reflexion attempts reached for {config_str}: {args.max_reflexion_attempts}")
                return True, False

        # Log info, Data logging
        logger.info(f"[Config]: {config_str}, attempt: {args.attempts_per_task[uid]}\n[Intent]: {intent}")
        set_timings_global_id(f"{uid}_{args.attempts_per_task[uid]}")
        timer.start("RUN:test")
        task_start_time = time.time()
        render_helper, traj_logger = initialize_data_loggers(
            args, data_recorder, config_dict, result_dir, task_id, intent, intent_img_paths, domain, env_name, uid
        )

        # Evaluation setup
        early_stop_thresholds = {
            "max_steps": args.max_steps,
            "parsing_failure": args.parsing_failure_th,
            "repeating_action": args.repeating_action_failure_th,
        }
        evaluator = evaluator_router(config_dict, captioning_fn=args.eval_caption_img_fn, fuzzy_match_model=args.fuzzy_match_model)  # type: ignore

        trajectory: Trajectory = []
        trajectory_view = TrajectoryView(trajectory)
        obs, info = env.reset(options={"config_file": config_dict})  # type: ignore
        state_info: StateInfo = {"observation": obs, "info": info}
        trajectory.append(state_info)
        meta_data: dict[str, Any] = {
            "action_str_history": ["None"],
            "manual_input": args.manual_input,
            "trajectory": trajectory_view,
            "args": args,
            "evaluator": partial(evaluator, config_file=config_dict),  # type: ignore
            "uid": uid,
            "task_id": task_id,
            "domain": domain,
            "attempt_num": args.attempts_per_task[uid],
            "env": env,
        }
        terminated = False
        should_stop = False

        # LINK: Loop: execute current task
        while True:
            save_log_file(args.result_dir)

            # LINK: generate action
            action, _ = generate_action(
                agent, trajectory_view, intent, intent_images, meta_data, terminated, trajectory, early_stop_thresholds
            )

            # Append action to trajectory
            trajectory.append(action)
            # Copy state info for logging
            state_info_copy = copy.deepcopy(state_info)

            # LINK Act on environment
            if action["action_type"] == ActionTypes.STOP:
                should_stop = True
            else:
                obs, _, terminated, _, info = env.step(action)

            # Convert action to string  to serve as input in next generations
            action_str = get_action_description(
                action=action,
                observation_metadata=state_info_copy["info"]["observation_metadata"],
                action_set_tag=args.action_set_tag,
                action_splitter=agent.get_prompt_constructor().instruction["meta_data"]["action_splitter"],
            )
            meta_data["action_str_history"].append(action_str)
            action["action_str_env_parsed"] = action_str  # type: ignore

            # Log data
            dump_timings(args.result_dir)
            if render_helper is not None:
                render_helper.render(action, state_info_copy, meta_data, args.render_screenshot)

            if traj_logger is not None:
                traj_logger.add_step(state=state_info_copy, action=action, metadata=meta_data)

            if should_stop:
                break

            # Update state
            state_info = {"observation": obs, "info": info}
            trajectory.append(state_info)
        # END OF TASK LOOP

        # If reflexion, deep copy trajectory for verification and reflection generation (copy because oracle evaluator can change state)
        trajectory_copy = None
        if agent.reflexion is not None and agent.reflexion.should_retry(uid):
            trajectory_copy = TrajectoryView(copy.deepcopy(trajectory))

        # Evaluate and print score.
        score = evaluator(trajectory=trajectory, config_file=config_dict, page=env.page)  # type: ignore
        logger.info(f"[Result] ({'PASS' if score == 1 else 'FAIL'}) {config_str}, attempt: {args.attempts_per_task[uid]}")
        elapsed_time = time.time() - task_start_time

        # If reflexion enabled, evaluate, generate reflection, and add config_file for next attempt
        is_success, should_retry, v_feedback, v_response, v_parsed_response = maybe_generate_reflection(
            agent, trajectory_copy, intent, intent_images, meta_data, score, uid
        )

        # Save trace
        if args.save_trace_enabled:
            env.save_trace(Path(result_dir) / "traces" / f"{task_id}_{args.attempts_per_task[uid]}.zip")

        # Record and save task stats
        data_recorder.update_save_data(
            task_id, domain, env_name, score, elapsed_time, num_actions=len(trajectory[1::2]), attempt_id=args.attempts_per_task[uid]
        )
        test_execution_success = True

        # Log trajectory data
        if traj_logger is not None:
            truncated = True if "stop_reason" in action else False
            traj_logger.end_episode(reward=score, total_time_seconds=elapsed_time, episode_completed=True, truncated=truncated)
            if hasattr(agent, "request_refiner") and agent.request_refiner is not None:
                obj_img_captions = [caption for _, caption in agent.request_refiner.image_captions.items()]  # type: ignore
                traj_logger.objective.update({"obj_img_captions": obj_img_captions})  # type: ignore
            if agent.reflexion is not None:
                reflexion_data = {
                    "reflections": agent.reflexion.reflection_memory.get(uid, []),
                    "attempt_num": args.attempts_per_task[uid],
                    "verifier_response": v_response,
                    "verifier_feedback": v_feedback,
                    "verifier_score": is_success,
                    "eval_mode": agent.reflexion.get_eval_mode(),
                }
                traj_logger.metadata.update({"reflexion_data": reflexion_data})  # type: ignore
            if "stop_reason" in action:
                traj_logger.info.update({"truncated_reason": action["stop_reason"]})  # type: ignore
            traj_logger.dump_to_json()

    except Exception as e:
        error_handler(e, result_dir=args.result_dir, data_recorder=data_recorder, config_id=uid)
        try:
            if traj_logger is not None:
                traj_logger.info.update({"error": str(e)})
                traj_logger.episode_completed = False
                traj_logger.dump_to_json()
        except Exception as e:
            logger.error(f"Error dumping trajectory: {e}")
    finally:
        timer.end("RUN:test")
        data_recorder.update_unfinished_failed_tasks(task_id, test_execution_success, domain, env_name)
        if render_helper is not None:
            render_helper.close()

    return test_execution_success, should_retry


# LINK: Evaluation
def test(args: argparse.Namespace, agent: ModularAgent, config_file_list: list[dict[str, Any]]) -> None:
    test_start_time = time.time()

    # Initialize data recorder to record experiment data
    data_recorder = DataRecorder(args.result_dir, config_file_list, args.test_config_json_file)

    # Initialize environment
    env = init_environment(args)

    # LINK: Iterate each task
    config_file_queue = deque(config_file_list)
    args.attempts_per_task = {}
    iteration_count = 0
    while config_file_queue:
        iteration_count += 1
        raw_config_dict = config_file_queue.popleft()
        _, should_retry = test_one(args, agent, data_recorder, env, raw_config_dict)
        if should_retry:
            config_file_queue.appendleft(raw_config_dict)

    env.close()
    elapsed_time = time.time() - test_start_time
    scores = data_recorder.get_scores()
    logger.info(f"Average score: {sum(scores) / len(scores)}; {len(scores)} tasks")
    logger.info(f"NOTE: {len(config_file_list) - data_recorder.num_failed_executions} of {len(config_file_list)} completed without error.")
    logger.info(f"Total test time (min): {elapsed_time / 60}")

    # Save summary execution stats
    data_recorder.save_execution_summary(total_time=elapsed_time, provider=agent.get_provider("executor"))


# LINK: Main
if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    args = parse_args()

    # Set seed for numpy, random, torch
    set_seed(args.seed)

    # Set debugging inputs, if any
    if DEBUG_ENABLED:
        set_debug_inputs(args)

    # Set Agent configurations
    if AGENTS_CONFIG_DIR in args.agent_config_file:
        args.agent_config_file = Path(args.agent_config_file).name
    args.agent_config_path = str(Path(AGENTS_CONFIG_DIR) / args.agent_config_file)
    args.agents_configs = get_agent_config(args.agent_config_path)

    # Define captioning functions if provided
    if not args.agent_captioning_model and "captioner" in args.agents_configs:
        args.agent_captioning_model = args.agents_configs["captioner"]["lm_config"]["model"]
    args.caption_img_fn, args.eval_caption_img_fn = define_captioning_fn(
        agent_captioning_model=args.agent_captioning_model,
        agent_captioning_model_device=args.agent_captioning_model_device,
        eval_captioning_model=args.eval_captioning_model,
        eval_captioning_model_device=args.eval_captioning_model_device,
        observation_type=args.observation_type,
    )

    # Validate and preprocess args
    preprocess_args(args)

    # Dump test_file_list and args to results folder
    with open(os.path.join(args.result_dir, "attempted_tasks.txt"), "w") as f:
        for test_config in args.test_config_list:
            f.write(f"{test_config['domain']}_{test_config['task_id']}_{test_config['env']}\n")

    save_log_file_path(args.result_dir)
    with open(f"{args.result_dir}/args.json", "w") as f:
        json.dump(make_json_serializable(args), f, indent=4)

    # Build Agent
    agent = construct_modular_agent(args.agents_configs, args.caption_img_fn)

    # Evaluate
    logger.info(f"Starting evaluation. Docker instance ID: {args.docker_instance_id}. Environment: {args.env}.")
    logger.info(f"\nTotal {len(args.test_config_list)} tasks left.\n")
    logger.info(f"\nResults will be saved in {args.result_dir}\n")
    try:
        test(args, agent, args.test_config_list)
        save_log_file(args.result_dir)
        logger.info(f"Results in {os.path.abspath(args.result_dir)}")

    except Exception as e:
        logger.error(repr(e), exc_info=True)
        raise e

    finally:
        restore_api_keys_to_file()

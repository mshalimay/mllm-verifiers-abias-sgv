"""vwa/scripts/runs/parallel_run.py

Parallel runner/orchestrator for VWA/WA-style benchmark tasks.

At a high level, this script:

1) Loads a list of task IDs (usually from a ``tasks.txt``-style file where the
     first line points to a JSON config file and the remaining lines are task IDs).
2) Splits tasks into batches and dispatches them to a pool of workers.
3) Maintains an environment pool (Docker instances) that are proactively reset
     so batches can start quickly.
4) Runs each batch by launching ``./scripts/runs/run.sh`` inside a tmux pane,
     then monitors the pane until completion (or until a timeout/stop signal).
5) Periodically logs progress and writes bookkeeping artifacts (e.g.,
     ``tasks.txt``, ``unfinished_tasks.txt``, and ``stats.csv``) under the results
     directory.

Key characteristics / operational notes:

* Requires ``tmux``: each worker batch is executed in its own tmux pane.
* Uses multiprocessing + a process pool to coordinate environment resets and
    batch execution.
* Environment resets and cookie resets are performed via helpers in
    ``utils_vwa.utils_vwa``.
* Results are written under ``--results-dir`` (or a default results directory),
    with per-batch subdirectories.
* The orchestrator can stop batches that exceed the configured max runtime.

See `prun.py` for an example of how to launch multiple instances of this script.
"""

import argparse
import concurrent.futures
import json
import logging
import os
import random
import signal
import subprocess
import time
from concurrent.futures import Future
from datetime import datetime
from enum import Enum
from multiprocessing import Lock, Manager, active_children
from multiprocessing.managers import BaseManager
from pathlib import Path
from pprint import pformat
from queue import Empty, Queue
from typing import Any, cast

import numpy as np
import pandas as pd
from agent.agent_utils import get_agent_attribute
from benchmark_config import AGENTS_CONFIG_DIR, DEFAULT_RESULTS_DIR, RESET_ENV_SCRIPT
from utils_vwa.captioner_utils import start_captioner
from utils_vwa.utils_vwa import clean_envs, get_uids_from_csv, get_uids_from_txt, reset_cookies_with_retry, reset_envs_with_retry

from core_utils.concurrency_utils import get_file_lock
from core_utils.eval_utils import set_seed
from core_utils.file_utils import find_files, resolve_path_conflict
from core_utils.signal_utils import signal_manager
from llms.constants import API_KEYS_BACKUP, API_KEYS_PATH

SINGLE_RUN_SCRIPT = "./scripts/runs/run.sh"  # script to dispatch to parallel processes
TEMP_FILES_DIR = ".temp_files"
EXC_COOKIE_SITE_COMB = False
MAX_THREADPOOL_WORKERS = 32
MAX_RESET_TRIES = 10
CURRENT_TMUX_SESSION: str | None = None
ENV_MANAGERS = []
PROCESS_POOL_EXECUTOR: concurrent.futures.Executor | None = None
LOCK_TIMEOUT = 120
PRINT_EVERY = 60 * 1  # print every PRINT_EVERY seconds

env_lock = Lock()
ALL_ENV_IDS = []

MAX_CONCURRENT_ENV_RESETS = 3


class ErrorCodes(Enum):
    SUCCESS = 0
    TIMEOUT = 1
    ERROR = -1


# Custom print function that includes datetime
def dprint(*args, **kwargs):
    """Print with datetime prefix in format YYYY-MM-DD_HH-MM"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    if args:
        # Convert first argument to string and add timestamp prefix
        first_arg = f"[{timestamp}] {args[0]}"
        print(first_arg, *args[1:], **kwargs)
    else:
        print(f"[{timestamp}]", **kwargs)


# ===============================================================================
# Argument parsing
# ===============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tasks in parallel.")

    # Parallelization configs
    parser.add_argument(
        "-n",
        "--num-processes",
        type=int,
        default=4,
        help="Number of parallel worker processes to keep busy. Each receives a batch of tasks to run.",
    )

    parser.add_argument(
        "-b",
        "--tasks-per-process",
        type=int,
        default=10,
        help="Number of tasks per worker process (tasks batch size).",
    )

    parser.add_argument(
        "-nenvs",
        "--num-envs-per-process",
        type=float,
        default=2,
        help="Number of environments per worker process. Set this to >1 so environments are started in advance for the next batch of tasks.",
    )

    # Max running time / Max attempts per task
    parser.add_argument(
        "-mrt",
        "--max-running-time",
        type=int,
        default=0,
        help="Max time a batch of tasks can run in minutes. Set -1 for no limit. If set to 0 will be estimated as `avg-running-time-per-task` *  `tasks-per-process`.",
    )

    parser.add_argument(
        "-art",
        "--avg-running-time-per-task",
        type=float,
        default=180,
        help="Average running time per task in seconds. Used to compute the max running time if `max-running-time` is set to 0.",
    )

    parser.add_argument(
        "-ma",
        "--max-attempts-per-task",
        type=int,
        default=2,
        help="Maximum number of attempts per task.",
    )

    parser.add_argument(
        "-ck",
        "--copy_api_keys",
        action="store_true",
        help="Make a copy of the api keys file at initialization.",
    )

    # Tasks configs
    parser.add_argument(
        "-t",
        "--tasks-file",
        default="",
        help="Path to file with tasks (first line is json_config_file, rest are numeric IDs)",
    )
    parser.add_argument(
        "-c",
        "--json-config-file",
        default="",
        help="If no tasks file, use this directory to find numeric tasks (min..max).",
    )

    parser.add_argument(
        "-st",
        "--shuffle-tasks",
        action="store_true",
        help="Shuffle the task list.",
    )

    parser.add_argument(
        "-seed",
        "--seed",
        type=int,
        default=42,
        help="Random seed. Used to shuffle the task list and to set the seed for the captioner.",
    )

    # Agent configs
    parser.add_argument(
        "-a",
        "--agent-config",
        default="",
        help="YAML configuration for the Agent",
    )

    # Env / (V)WA Specific configs
    parser.add_argument(
        "--env",
        default="vwa",
        help="Environment to run the tasks in.",
    )

    parser.add_argument(
        "-d",
        "--domains-to-reset",
        nargs="*",
        default="",
        help="Domains to reset. If none, will reset all domains based on `--env`.",
    )

    parser.add_argument(
        "-cd",
        "--captioner-device",
        default="server-cuda",
        help="Device to host captioner on.",
    )

    parser.add_argument(
        "-r",
        "--results-dir",
        default="",
        help="Directory to save the results.",
    )

    parser.add_argument(
        "-id",
        "--experiment-id",
        default="",
        help="Experiment ID. Used to identify the experiment.",
    )

    parser.add_argument(
        "-at",
        "--attempt-num",
        type=int,
        default=-1,
        help="Attempt number. Used to identify the attempt for a given task id.",
    )

    parser.add_argument(
        "-sc",
        "--skip-completed",
        action="store_true",
        help="Skip completed tasks.",
    )

    parser.add_argument("-esi", "--env-start-idx", type=int, default=0, help="Start index for environment IDs.")

    parser.add_argument(
        "-x",
        "--max-steps",
        type=int,
        default=30,
        help="Max number of environment steps allowed. If exceeded, FAIL.",
    )

    return parser.parse_args()


# ===============================================================================
# LINK Classes
# ===============================================================================


class TaskBatch:
    def __init__(
        self,
        task_list: list[str],
        hard_reset: bool = False,
        start_run_time: float = -1,
        batch_id: int = -1,
        max_run_time: float = 0,
        avg_run_time_per_task: float | None = None,
    ):
        self.task_list = task_list  # List of task ids.
        self.hard_reset = hard_reset  # Marks batch suffered hard reset. Useful to skip updating attempts for tasks in the batch.
        self.start_run_time = start_run_time  # Time when the batch started processing.
        self.batch_id = batch_id  # ID of the batch.
        self.max_run_time = self._set_max_run_time(max_run_time, avg_run_time_per_task)
        self.worker_id: int | None = None  # ID of the worker that is processing the batch.

    def _set_max_run_time(self, max_run_time: float, avg_run_time_per_task: float | None) -> float:
        if max_run_time < 0:
            return np.inf
        elif max_run_time == 0:
            if avg_run_time_per_task is None:
                raise ValueError("avg_run_time_per_task must be provided if max_run_time is 0.")
            else:
                return avg_run_time_per_task * len(self.task_list)
        else:
            return max_run_time


class Status(Enum):
    READY = "ready"  # type: ignore
    IN_USE = "in_use"  # type: ignore
    NEEDS_RESET = "needs_reset"  # type: ignore
    RESETTING = "resetting"  # type: ignore
    FAILED = "failed"  # type: ignore


class EnvPool:
    def __init__(self, num_envs: int, start_idx: int = 0):
        self.envs: dict[int, dict[str, Any]] = {
            start_idx + i: {
                "status": Status.NEEDS_RESET,
                "try_count": 0,
                "worker_id": None,
                "batch_id": None,
                "env_id": start_idx + i,
            }
            for i in range(num_envs)
        }
        self.lock = Lock()
        self.free_envs = Queue()

    def get_env_ids(self) -> list[int]:
        with self.lock:
            return list(self.envs.keys())

    def update_status(self, env_id: int, new_status: Status):
        with self.lock:
            self.envs[env_id]["status"] = new_status

    def increment_try_count(self, env_id: int):
        with self.lock:
            self.envs[env_id]["try_count"] += 1

    def get_try_count(self, env_id: int) -> int:
        with self.lock:
            return self.envs[env_id]["try_count"]

    def compare_and_set_status(self, env_id: int, expected_status: Status, new_status: Status) -> bool:
        with self.lock:
            if self.envs[env_id]["status"] == expected_status:
                self.envs[env_id]["status"] = new_status
                return True
            return False

    def update_data(
        self,
        env_id: int,
        status: Status | None = None,
        inc_try_count: int = 0,
        batch_id: int | None = None,
        worker_id: int | None = None,
    ):
        with self.lock:
            if status is not None:
                self.envs[env_id]["status"] = status
            if batch_id is not None:
                self.envs[env_id]["batch_id"] = batch_id
            if worker_id is not None:
                self.envs[env_id]["worker_id"] = worker_id
            self.envs[env_id]["try_count"] += inc_try_count

    def get_env_data(self, env_id: int) -> dict[str, Any]:
        with self.lock:
            return self.envs[env_id]

    def get_all_env_data(self) -> list[dict[str, Any]]:
        with self.lock:
            return list(self.envs.values())


BaseManager.register("EnvPool", EnvPool)


# ===============================================================================
# LINK Logger helpers
# ===============================================================================


def compute_stats(
    tasks_finished: set[str],
    initial_task_list: set[str],
    tasks_to_run: set[str],
    running_time_seconds: float,
    env_data: list[dict[str, Any]],
    results_dir: str,
) -> dict[str, Any]:
    """
    Compute comprehensive stats for the current run, including timing and environment info.
    Only counts tasks from the current run when computing averages.
    """
    try:
        total_overall = len(initial_task_list)
        finished_overall = len(tasks_finished & initial_task_list)
        percent_finished_overall = (finished_overall / total_overall * 100) if total_overall else 0.0

        finished_this_run = len(tasks_finished & tasks_to_run)
        remaining_this_run = max(0, len(tasks_to_run) - finished_this_run)

        if finished_this_run > 0:
            avg_time_per_task = running_time_seconds / finished_this_run
        else:
            avg_time_per_task = None

        # Flatten env status counts
        status_counts: dict[str, int] = {}
        for env in env_data:
            status_val = env.get("status")
            if isinstance(status_val, Enum):
                key = str(status_val.value)
            elif status_val is None:
                key = "none"
            else:
                key = str(status_val)
            status_counts[key] = status_counts.get(key, 0) + 1

        success_stats = compute_success_rate(results_dir)

        stats: dict[str, Any] = {
            "percent_finished": f"{percent_finished_overall:.2f}",
            "num_finished": finished_overall,
            "num_total": total_overall,
            "num_finished_this_run": finished_this_run,
            "num_remaining_this_run": remaining_this_run,
            "avg_time_per_task_seconds": avg_time_per_task,
        }
        if success_stats:
            stats.update(success_stats)
        return stats
    except Exception as e:
        dprint(f"[ERROR] Failed to compute stats: {e}")
        return {}


# TODO: add more fine-grained timer to also print times ex env reset
class Printer:
    def __init__(self, print_every: int = PRINT_EVERY) -> None:
        self.running_time = 0
        self.print_every = print_every

    def print_performance_metrics(self, all_stats: dict[str, Any], json_config_file: str) -> None:
        if not all_stats:
            return
        template = "[INFO] [Orchestrator] [{time}]: {message}"
        formatted = pformat(all_stats, indent=2, width=120, compact=True, sort_dicts=False)
        print(
            template.format(
                time=datetime.now().strftime("%Y-%d-%m-%H:%M"),
                message=f"Performance Stats: \n{formatted} \nConfig Dir: {json_config_file}",
            )
        )

    def start_timer(self) -> None:
        self.start_time = time.time()
        self.last_print_time = self.start_time

    def update_running_time(self) -> None:
        self.running_time = time.time() - self.start_time

    def print_env_data(self, env_data: list[dict[str, Any]]) -> None:
        template = "[INFO] [Orchestrator] [{time}]: {message}"
        formatted = pformat(env_data, indent=2, width=120, compact=True)
        print(template.format(time=datetime.now().strftime("%Y-%d-%m-%H:%M"), message=f"Env data:\n{formatted}"))

    def print_stats(
        self,
        tasks_finished: set[str],
        initial_task_list: set[str],
        tasks_to_run: set[str],
        json_config_file: str,
        env_data: list[dict[str, Any]],
        results_dir: str,
        remaining_tasks: set[str],
        free_env_pool=None,
    ) -> None:
        template = "[INFO] [Orchestrator] [{time}]: {message}"
        if time.time() - self.last_print_time > self.print_every:
            self.last_print_time = time.time()
            self.update_running_time()
            overall_tasks_finished = tasks_finished & initial_task_list
            percent_finished = len(overall_tasks_finished) / len(initial_task_list) * 100

            tasks_finished_for_this_run = tasks_finished & tasks_to_run

            if len(tasks_finished_for_this_run) > 0:
                avg_time_per_task = self.running_time / len(tasks_finished_for_this_run)
            else:
                avg_time_per_task = -1

            msg = f"Finished for this run: {percent_finished:.2f}% ({len(tasks_finished_for_this_run)}/{len(tasks_to_run)}) tasks for {json_config_file}."
            print(template.format(time=datetime.now().strftime("%Y-%d-%m-%H:%M"), message=msg))
            dprint(f"Total running time: {int(self.running_time / 60)} minutes and {int(self.running_time % 60)} seconds.")
            dprint(f"Number of tasks remaining: {len(remaining_tasks)}")
            if avg_time_per_task != -1:
                dprint(f"Avg time per task: {int(avg_time_per_task / 60)} minutes and {int(avg_time_per_task % 60)} seconds.")
                if remaining_tasks:
                    estimated_time_remaining = avg_time_per_task * len(remaining_tasks)
                    dprint(f"Estimated time remaining: {int(estimated_time_remaining / 60)} minutes and {int(estimated_time_remaining % 60)} seconds.")

            performance_stats = compute_stats(
                tasks_finished=tasks_finished,
                initial_task_list=initial_task_list,
                tasks_to_run=tasks_to_run,
                running_time_seconds=self.running_time,
                env_data=env_data,
                results_dir=results_dir,
            )
            self.print_performance_metrics(performance_stats, json_config_file)
            self.print_env_data(env_data)

            # Log free env pool snapshot (best-effort)
            if free_env_pool is not None:
                try:
                    free_qsize = free_env_pool.qsize()
                except Exception:
                    free_qsize = -1
                ready_env_ids = [e.get("env_id") for e in env_data if e.get("status") == Status.READY]
                all_env_ids = [e.get("env_id") for e in env_data]
                self.print_message(f"Free env pool snapshot: qsize={free_qsize}, READY env_ids={ready_env_ids}, all env_ids={all_env_ids}")

    def print_batch_completion(self, batch_id: int, run_time: float, future: Future) -> None:
        """
        Prints the completion status for a batch.
        """
        error_msg = ""
        try:
            ret_code, msg = future.result()
            if ret_code == 0:
                error_msg = "finished with no errors."
            elif ret_code == 1:
                error_msg = "finished with a timeout."
            else:
                error_msg = f"finished with error code {ret_code}. Msg: {msg}"
        except Exception as e:
            error_msg = f"encountered an exception: {e}"

        dprint(f"[Orchestrator]: Batch {batch_id} finished after {int(run_time / 60)} minutes and {int(run_time % 60)} seconds.\n[Orchestrator]: Batch {batch_id} {error_msg}")

    def print_run_info(
        self,
        agent_config: str,
        json_config_file: str,
        total_tasks: int,
        total_batches: int,
        tasks_per_process: int,
        num_workers: int,
        max_attempts_per_task: int,
        shuffle_tasks: bool,
        max_running_time: float,
        first_batch_max_run_time: float,
        avg_running_time_per_task: float,
        domains_to_reset: str | list[str],
        num_envs_per_process: float,
    ) -> None:
        """
        Prints an initial overview of the parallel run settings.
        """
        dprint("\n-------------- Starting parallel run --------------")
        dprint(f"Agent config: [{agent_config}] | json_config_file: [{json_config_file}]")
        dprint(f"Total tasks: [{total_tasks}] | Total batches: [{total_batches}]")
        dprint(f"Tasks per process: [{tasks_per_process}] | Num processes: [{num_workers}] | Num envs per process: [{num_envs_per_process}] | Total envs: [{int(num_workers * num_envs_per_process)}]")
        dprint(f"Max attempts per task: [{max_attempts_per_task}] | Shuffle tasks: [{shuffle_tasks}]")
        max_time_str = "NO LIMIT" if max_running_time < 0 else (f"{first_batch_max_run_time} (auto)" if max_running_time == 0 else f"{max_running_time}")
        dprint(f"Max run time per process: [{max_time_str}] | Avg run time per task: [{avg_running_time_per_task}]")
        dprint(f"Domains to reset: [{', '.join(domains_to_reset)}]")
        dprint("-----------------------------------------------------")

    def print_message(self, message: str) -> None:
        dprint(f"[INFO][Orchestrator] {message}")

    def print_dispatched_batch(self, batch_id: int, task_list: list[str]) -> None:
        # Print tasks without forcing integer cast to support string IDs like 'reddit_170'
        dprint(f"[INFO] [Orchestrator] Dispatched batch {batch_id}: {list(map(str, task_list))}")

    def print_waiting_batches(self, waiting_count: int) -> None:
        dprint(f"[INFO] [Orchestrator] No batches to dispatch. Waiting {waiting_count} batches to finish...")

    def print_all_completed(self, json_config_file: str, agent_config: str, result_dir: str) -> None:
        dprint(f"[INFO] [Orchestrator] All tasks completed for {json_config_file}, agent config {agent_config}.")
        dprint(f"Results saved in {result_dir}")


# ===============================================================================
# LINK Task I/O helpers
# ===============================================================================
def get_tasks_success_failed(
    dir: str,
    return_failed: bool = False,
    attempt_num: int = -1,
) -> tuple[set[str], set[str]]:
    """
    Get the number of tasks completed in a given directory.
    Returns:
        dict of worker_id -> tuple(successful_tasks, failed_tasks).
        set of all successful tasks.
        set of all failed tasks.
    """
    args_files = find_files(dir, "args.json", upwards=False, downwards=True)
    all_successful_tasks = set()
    all_failed_tasks = set()
    for args_file in args_files:
        parent_dir = Path(args_file).parent
        summary_data_path = parent_dir / "summary_data.csv"
        if summary_data_path.exists():
            all_successful_tasks.update(get_uids_from_csv(summary_data_path))

        failed_tasks_path = parent_dir / "failed_tasks.txt"
        if failed_tasks_path.exists() and return_failed:
            all_failed_tasks.update(get_uids_from_txt(txt_path=str(failed_tasks_path)))

    return all_successful_tasks, all_failed_tasks


# @deprecated
# def get_range_from_config_dir(json_config_file: str) -> tuple[str, str]:
#     """
#     Looks in json_config_file for filenames starting with digits, extracts the min and max.
#     Returns (start_id, end_id).
#     """
#     if not os.path.isdir(json_config_file):
#         raise FileNotFoundError(f"Config dir not found: {json_config_file}")

#     numeric_ids = []
#     for fname in os.listdir(json_config_file):
#         match = re.match(r"^(\d+)", fname)
#         if match:
#             numeric_ids.append(int(match.group(1)))

#     if not numeric_ids:
#         raise ValueError(f"No numeric files found in {json_config_file}")

#     numeric_ids.sort()
#     return numeric_ids[0], numeric_ids[-1]


def get_task_list(
    tasks_file: str = "",
    json_config_file: str = "",
) -> tuple[str, list[str]]:
    if tasks_file:
        # Get test config dir and task ids from tasks file
        with open(tasks_file, "r") as f:
            lines = f.readlines()
            json_config_file = lines[0].strip()
            task_ids = [line.strip() for line in lines[1:]]

        # Check if test config dir is valid
        if not json_config_file or not os.path.isfile(json_config_file):
            raise ValueError(f"[ERROR] No config file found in file: {tasks_file}")

    else:
        raise ValueError(f"[ERROR] No tasks file given: {tasks_file}")
    # else:
    #     # numeric range mode from --config-dir
    #     if not json_config_file:
    #         raise ValueError("[ERROR] No config directory given.")

    #     start_id, end_id = get_range_from_config_dir(json_config_file)
    #     task_ids = list(range(start_id, end_id + 1))

    return json_config_file, task_ids


def create_task_txt_file(
    json_config_file: str,
    task_list: list[str] | set[str],
    out_dir: str = TEMP_FILES_DIR,
    filename: str = "tasks.txt",
    overwrite: bool = False,
) -> str:
    """
    Creates a text file with the task list.
    """
    if isinstance(task_list, set):
        task_list = list(task_list)

    # Create a name if path exists
    file_path = f"{out_dir}/{filename}"

    if not overwrite:
        final_file_path = resolve_path_conflict(file_path, int_suffix=True)
    else:
        final_file_path = file_path

    os.makedirs(out_dir, exist_ok=True)
    with open(final_file_path, "w") as f:
        f.write(json_config_file + "\n")
        for task_id in task_list:
            f.write(f"{task_id}\n")
    return str(final_file_path)


def write_unfinished_tasks(
    original_task_list: list[str] | set[str],
    json_config_file: str,
    out_dir: str,
    tasks_finished: set[str] | list[str] | None = None,
    result_dir: str = "",
    filename: str = "unfinished_tasks.txt",
    overwrite: bool = True,
    attempt_num: int = -1,
) -> str:
    if tasks_finished is None and result_dir:
        tasks_finished, _ = get_tasks_success_failed(result_dir, return_failed=False, attempt_num=attempt_num)

    if tasks_finished is None:
        raise ValueError("No tasks finished or not able to retrieve from 'result_dir'.")

    if isinstance(tasks_finished, list):
        tasks_finished = set(tasks_finished)

    unfinished_tasks = set(original_task_list) - tasks_finished

    final_file_path = create_task_txt_file(
        json_config_file=json_config_file,
        task_list=unfinished_tasks,
        out_dir=out_dir,
        filename=filename,
        overwrite=overwrite,
    )
    return final_file_path


def restore_api_keys_file(
    src_file: str = API_KEYS_BACKUP,
    dest_file: str = API_KEYS_PATH,
    min_keys: dict[str, int] = {"google": 1, "openai": 0},
    logger: logging.Logger | None = None,
    num_restores: int = 0,
) -> int:
    if not os.path.exists(src_file):
        return num_restores

    all_api_keys = json.load(open(src_file))
    # safe read the API_KEYS_PATH
    try:
        lock = get_file_lock(API_KEYS_PATH)
        with lock, open(dest_file, "r+") as f:
            api_keys = json.load(f)

            for provider in api_keys.keys():
                min_keys_for_provider = min_keys.get(provider, 0)
                if len(api_keys[provider]) <= min_keys_for_provider:
                    num_restores += 1
                    dprint(f"\n--------------WARNING--------------:\n{dest_file} has <= {min_keys_for_provider} keys for {provider}. Restoring from {src_file}.\n------------------------------------")
                    if provider in all_api_keys:
                        api_keys[provider] = all_api_keys[provider]
            f.seek(0)
            json.dump(api_keys, f, indent=2)
            f.truncate()
    except Exception as e:
        dprint(f"[ERROR] Failed to restore API keys file: {e}")
    finally:
        return num_restores


# ===============================================================================
# LINK Stats helpers
# ===============================================================================
def compute_execution_stats(
    tasks_finished: set[str],
    tasks_to_run: set[str],
    json_config_file: str,
    results_dir: str,
    initial_time: float,
    args: dict[str, Any],
) -> dict[str, Any]:
    """
    Compute basic execution stats for the current run without timing information.
    Does not include tasks from previous runs by intersecting with tasks_to_run.
    """
    try:
        total_this_run = len(tasks_to_run)
        finished_this_run = len(tasks_finished & tasks_to_run)

        success_stats = compute_success_rate(results_dir)

        avg_time_per_task_seconds = (time.time() - initial_time) / finished_this_run if finished_this_run else None

        stats: dict[str, Any] = {
            "total_tasks_for_this_run": total_this_run,
            "finished_tasks_for_this_run": finished_this_run,
            "avg_time_per_task_seconds": avg_time_per_task_seconds,
            "json_config_file": json_config_file,
        }
        stats.update(args)
        if success_stats:
            stats.update(success_stats)
        return stats
    except Exception as e:
        dprint(f"[ERROR] Failed to compute execution stats: {e}")
        return {}


def compute_success_rate(dir: str) -> dict[str, Any]:
    """
    Compute success rate and counts from summary_data.csv files under a results directory.
    """
    try:
        args_files = find_files(dir, "args.json", upwards=False, downwards=True)
        df = pd.DataFrame()
        for args_file in args_files:
            parent_dir = Path(args_file).parent
            summary_data_path = parent_dir / "summary_data.csv"
            if summary_data_path.exists():
                df = pd.concat([df, pd.read_csv(summary_data_path)])
        if df.empty:
            return {}

        # If duplicate task_ids, pick the max score
        if "domain_task_id" not in df.columns:
            df["domain_task_id"] = df["domain"].astype(str) + "_" + df["task_id"].astype(str)
        df = df.groupby("domain_task_id").max().reset_index()

        df["success"] = df["score"] >= 1
        success_rate = df["success"].mean() * 100
        num_tasks = len(df["success"])

        # Aggregate by domain
        if "domain" not in df.columns:
            try:
                df["domain"] = df["domain_task_id"].str.split("_").str[0]
            except Exception:
                df["domain"] = "unknown"
        per_domain = {}
        for domain, g in df.groupby("domain"):
            total = int(len(g))
            succ = int(g["success"].sum())
            per_domain[domain] = {
                "total": total,
                "success": succ,
                "fail": int(total - succ),
                "success_rate": f"{(succ / total * 100) if total else 0:.1f}%",
            }

        all_stats = {
            "success_rate": f"{success_rate:.1f}%",
            "total_tasks": num_tasks,
            "total_success": int(df["success"].sum()),
            "total_fail": int(num_tasks - df["success"].sum()),
            "per_domain": per_domain,
        }
    except Exception as e:
        dprint(f"[ERROR] Failed to compute success rate: {e}")
        return {}
    return all_stats


def write_stats_to_csv(
    dir: str,
    tasks_finished: set[str],
    tasks_to_run: set[str],
    json_config_file: str,
    results_dir: str,
    initial_time: float,
    args: dict[str, Any],
    exper_id: str,
) -> None:
    """
    Write the stats to a csv file.
    """
    execution_stats = compute_execution_stats(
        tasks_finished=tasks_finished,
        tasks_to_run=tasks_to_run,
        json_config_file=json_config_file,
        results_dir=results_dir,
        initial_time=initial_time,
        args=args,
    )
    file_path = f"{dir}/stats.csv"
    initial_df = pd.DataFrame()
    if os.path.exists(file_path):
        try:
            initial_df = pd.read_csv(file_path)
        except Exception:
            initial_df = pd.DataFrame()

    # Ensure exper_id is included in the row we write
    stats_with_ts = dict(execution_stats)
    stats_with_ts["exper_id"] = exper_id

    # Replace entries with the same timestamp if the column exists
    if not initial_df.empty and "exper_id" in initial_df.columns:
        initial_df = initial_df[initial_df["exper_id"] != exper_id]

    df = pd.concat([initial_df, pd.DataFrame([stats_with_ts])], ignore_index=True)
    df.to_csv(file_path, index=False, na_rep="NA")


# ===============================================================================
# ANCHOR: Tmux helpers
# ===============================================================================


def create_tmux_session() -> str:
    """
    Create a new tmux session in detached mode and return its session id.
    """
    try:
        result = subprocess.run(
            ["tmux", "new-session", "-d", "-P", "-F", "#{session_id}"],
            capture_output=True,
            text=True,
            check=True,
        )
        session_id = result.stdout.strip()
        dprint(f"Created tmux session with id: {session_id}")
        return session_id
    except subprocess.CalledProcessError as e:
        dprint(f"Failed to create tmux session: {e.stderr}")
        raise e


def is_pane_running(pane_id: str, tmux_session_id: str) -> tuple[bool, list[str]]:
    try:
        pane_list = subprocess.run(
            ["tmux", "list-panes", "-t", tmux_session_id, "-a", "-F", "#{pane_id}"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
        return pane_id in pane_list, pane_list
    except subprocess.CalledProcessError as e:
        raise e


def run_tmux_pane(tmux_session_id: str, cmd: str, new_window: bool = False) -> subprocess.CompletedProcess:
    tmux_cmd = [
        "tmux",
        "split-window" if not new_window else "new-window",
        "-t",
        tmux_session_id,
        "-P",  # Print pane information.
        "-F",
        "#{pane_id}",
        cmd,
    ]
    result = subprocess.run(tmux_cmd, check=False, capture_output=True, text=True)
    return result


def run_tmux_pane_fallback(tmux_session_id: str, cmd: str) -> subprocess.CompletedProcess:
    """
    Tries splitting the tmux session's active window first.
    If it fails with "no space for new pane", it creates a new window instead.
    Returns the final subprocess.CompletedProcess.
    """
    result = run_tmux_pane(tmux_session_id, cmd, new_window=False)
    if result.returncode != 0 and "no space for new pane" in result.stderr.lower():
        result = run_tmux_pane(tmux_session_id, cmd, new_window=True)
    return result


# ===============================================================================
# ANCHOR: Workers
# ===============================================================================


def run_sh_script_tmux(
    batch_id: int,
    json_config_file: str,
    agent_config: str,
    tmux_session_id: str,
    docker_instance_id: int,
    stop_dict: dict[int, bool],
    task_list: list[str] = [],
    start_id: int | None = None,
    end_id: int | None = None,
    result_dir: str = "",
    captioner_device: str = "server-cuda",
    max_steps: int = 30,
    wait_before_check: int = 30,
    max_retry_ls_tmux: int = 3,
    wait_until_dead: int = 30,
    worker_id: int | None = None,
) -> tuple[int, str]:
    cmd = [SINGLE_RUN_SCRIPT]

    if worker_id is None:
        worker_id = batch_id

    if task_list:
        # Create a temporary text file with the task list.
        task_list_file = create_task_txt_file(json_config_file, task_list, filename=f"tasks_{batch_id}.txt")
        # Append task_list to result_dir
        result_dir = f"{result_dir}/{Path(task_list_file).stem}"
        result_dir = str(resolve_path_conflict(result_dir, int_suffix=True))
        dprint(f"Worker {worker_id}, docker_instance_id {docker_instance_id}, batch {batch_id}, Result dir: {result_dir}")

        cmd += ["-t", task_list_file]
    elif start_id is not None and end_id is not None:
        cmd += ["-s", str(start_id), "-e", str(end_id)]

    cmd += ["-c", json_config_file]
    cmd += ["-a", agent_config]

    if result_dir:
        cmd += ["-d", result_dir]

    if captioner_device:
        cmd += ["-m", captioner_device]

    cmd += ["-i", str(docker_instance_id)]

    cmd += ["-x", str(max_steps)]
    # Build the command to be executed in the tmux pane.
    inner_cmd = f" ".join(cmd) + "; exit"
    # If required, use this to activate the env first
    # inner_cmd = f"conda activate {conda_env}; " + inner_cmd

    dprint(f"[Worker {worker_id}], batch {batch_id}, docker_instance_id {docker_instance_id}: TMUX CMD: {' '.join(cmd)}")
    # Create a new tmux pane and capture its pane id.
    result = run_tmux_pane_fallback(tmux_session_id, inner_cmd)
    if result.returncode != 0:
        dprint(f"[Batch {batch_id}], Worker {worker_id}, Instance {docker_instance_id}: TMUX ERROR {result.returncode}:\n{result.stderr}")
        return ErrorCodes.ERROR.value, str(result.stderr)

    # Get the pane id from stdout.
    pane_id = result.stdout.strip()
    dprint(f"[Batch {batch_id}], Worker {worker_id}, Instance {docker_instance_id}: Launched tmux pane with id: {pane_id}")

    # Block until the pane no longer exists
    retry_count = 0
    while True:
        # LOOP: block until task is finished or stop signal received.
        try:
            # If stop signal received, stop early.
            if stop_dict.get(batch_id, False):
                dprint(f"[Batch {batch_id}], Worker {worker_id}, Instance {docker_instance_id}: Stop signal received. Attempting graceful stop...")

                # Try 'ctrl + c' to stop `run.py` gracefully.
                ret_ctrl_c = subprocess.run(["tmux", "send-keys", "-t", pane_id, "C-c"], check=False)
                kill_pane = True

                # Wait for the pane to die.
                if ret_ctrl_c.returncode == 0:
                    start_time = time.time()
                    while time.time() - start_time < wait_until_dead:
                        if not is_pane_running(pane_id, tmux_session_id)[0]:
                            kill_pane = False
                            break
                        time.sleep(0.1)
                # If the pane is still running, forcefully kill it.
                if kill_pane:
                    # Forcefully kill the tmux pane (obs.: panes are unique within the same server, regardless of the session/window).
                    ret = subprocess.run(["tmux", "kill-pane", "-t", pane_id], check=False)
                    if ret.returncode != 0:
                        dprint(f"[Batch {batch_id}], Worker {worker_id}: Stopping early, but failed to kill tmux pane {pane_id}: {ret.stderr}")
                # Remove itself from the stop dict.
                del stop_dict[batch_id]
                return ErrorCodes.TIMEOUT.value, "Stop signal received."

            # If pane is not running -> batch finished -> return 0.
            if not is_pane_running(pane_id, tmux_session_id)[0]:
                dprint(f"[Batch {batch_id}], Worker {worker_id}: FINISHED: TMUX pane {pane_id} closed.")
                return ErrorCodes.SUCCESS.value, "Batch finished."
            time.sleep(wait_before_check)

        except subprocess.CalledProcessError as e:
            if retry_count < max_retry_ls_tmux:
                # If error listing panes, retry up to `max_retry_ls_tmux` times.
                dprint(f"[Batch {batch_id}], Worker {worker_id}: Error listing panes: {e.stderr}. Retrying...")
                retry_count += 1
                time.sleep(5)
            else:
                # If error listing panes more than `max_retry_ls_tmux` times, stop with return code -1.
                dprint(f"[Batch {batch_id}], Worker {worker_id}: Failed to check if tmux pane is running. Stopping.")
                return ErrorCodes.ERROR.value, str(e.stderr)


# ===============================================================================
# Dispatcher
# ===============================================================================


def kill_hanging_children(grace_period: float = 2.0) -> None:
    """
    Force-terminate any lingering multiprocessing children of this process.
    1) Send terminate()
    2) Wait briefly
    3) SIGKILL if still alive
    """
    children = active_children()
    if not children:
        return
    for proc in children:
        try:
            proc.terminate()
        except Exception:
            pass

    deadline = time.time() + grace_period
    while time.time() < deadline and any(p.is_alive() for p in children):
        time.sleep(0.1)

    for proc in children:
        if proc.is_alive():
            try:
                if proc.pid is not None:
                    os.kill(proc.pid, signal.SIGKILL)
            except Exception:
                pass


def create_task_batches(
    task_list: list[str] | set[str],
    tasks_per_worker: int,
    max_run_time: float = 0,
    avg_run_time_per_task: float | None = None,
    shuffle: bool = False,
) -> list[TaskBatch]:
    if len(task_list) == 0:
        return []

    if isinstance(task_list, set):
        task_list = list(task_list)

    if max_run_time == 0 and avg_run_time_per_task is None:
        raise ValueError("avg_run_time_per_task must be provided if `max_run_time` is 0.")

    if shuffle:
        task_list = shuffle_task_list(task_list)

    # Determine the number of batches required
    num_batches = (len(task_list) + tasks_per_worker - 1) // tasks_per_worker

    # Evenly split the tasks into num_batches arrays
    batches = [list(map(str, batch)) for batch in np.array_split(task_list, num_batches)]

    # Convert numpy arrays back to lists
    return [TaskBatch(list(batch), max_run_time=max_run_time, avg_run_time_per_task=avg_run_time_per_task) for batch in batches]


def shuffle_task_list(task_list: list[str] | set[str]) -> list[str]:
    if isinstance(task_list, set):
        task_list = list(task_list)
    random.shuffle(task_list)
    return task_list


def recompute_task_batches(
    waiting_batches: list[TaskBatch],
    submitted_batches: list[TaskBatch],
    attempts_per_task: dict[str, int],
    tasks_finished: set[str],
    tasks_failed: set[str],
    max_attempts: int = 2,
    tasks_per_worker: int = 10,
    max_run_time: float = 0,
    avg_run_time_per_task: float | None = None,
    shuffle: bool = False,
) -> list[TaskBatch]:
    """
    Recomputes the task batches for re-assignment from two sources:
    - waiting_batches: tasks that haven't yet been dispatched.
    - submitted_batches: tasks that were already dispatched but may have failed.

    The function updates the attempt counts only for tasks in submitted_batches
    (i.e. tasks already in process), and it only includes tasks that are not finished.

    Returns:
        New task batches created using the union of waiting tasks (filtered) and
        unfinished tasks from submitted batches.
    """
    # Get tasks from the waiting batches that haven't been finished yet (obs.: should give the same set, but just in case)
    waiting_tasks = set(task for batch in waiting_batches for task in batch.task_list if task not in tasks_finished)

    # For tasks that failed: (i) update attempts, (ii) add to unfinished tasks if attempts < max_attempts.
    unfinished_tasks = []
    for batch in submitted_batches:
        for task in batch.task_list:
            if task in tasks_finished:
                continue

            # If there is a failed task set, and the task is in it, update attempts.
            if task in tasks_failed:
                attempts_per_task[task] += 1

            # If not identified as failed, update attempts only if the batch did not suffer a hard reset.
            elif not batch.hard_reset:
                attempts_per_task[task] += 1

            # If the task has reached the maximum number of attempts, skip it.
            if attempts_per_task[task] < max_attempts:
                unfinished_tasks.append(task)
            else:
                dprint(f"[Info] Task {task} reached maximum attempts, skipping.")

    # Combine the waiting tasks with the unfinished tasks from submitted batches.
    new_task_set = waiting_tasks.union(unfinished_tasks)
    return create_task_batches(new_task_set, tasks_per_worker, max_run_time, avg_run_time_per_task, shuffle=shuffle)


# LINK Reset environments and cookies helpers
def wait_until_all_workers_dead(
    futures: list[Future],
    max_wait_time: int = 120,
) -> bool:
    start_time = time.time()
    while time.time() - start_time < max_wait_time:
        if all(future.done() for future in futures):
            return True
        time.sleep(0.5)
    return False


def send_stop_signals(
    future_to_batch: dict[Future, TaskBatch],
    stop_dict: dict[int, bool],
    max_wait_dead_workers: int = 120,
) -> None:
    cur_time = time.time()
    killed_futures = []
    for future in future_to_batch.keys():
        batch = future_to_batch[future]
        if cur_time - batch.start_run_time > batch.max_run_time:
            dprint(f"[Orchestrator] Stopping batch {batch.batch_id}, worker {batch.worker_id}. Running for more than {batch.max_run_time} seconds.")
            stop_dict[batch.batch_id] = True
            killed_futures.append(future)
    # Wait until workers are dead
    if killed_futures:
        wait_until_all_workers_dead(killed_futures, max_wait_dead_workers)


def preprocess_args(
    num_workers: int,
    task_list: list[str] | set[str],
    tasks_per_worker: int,
    max_running_time: float,
    avg_running_time_per_task: float,
) -> tuple[list[str] | set[str], int, float]:
    # Remove any duplicates in the task list.
    if isinstance(task_list, list):
        task_list = list(set(task_list))

    if tasks_per_worker == 0:
        tasks_per_worker = max(1, len(task_list) // num_workers)

    # Convert to seconds
    max_running_time = max_running_time * 60 if max_running_time > 0 else max_running_time

    return task_list, tasks_per_worker, max_running_time


def start_reset_env(
    env_pool: EnvPool,
    free_env_pool: Queue,
    instance_id: int,
    reset_env_domains: list[str] | str = "all_vwa",
    reset_cookies_sites: list[str] | str = "all_vwa",
    reset_cookies_exc_comb: bool = EXC_COOKIE_SITE_COMB,
    env: str = "vwa",
) -> None:
    try:
        # Update status to resetting
        env_pool.update_status(instance_id, Status.RESETTING)

        # Reset envs if requested
        if reset_env_domains:
            dprint(f"[INFO] Env {instance_id}: starting/resetting {reset_env_domains}.")
            code, failed_sites = reset_envs_with_retry(
                instance_id=instance_id,
                domains=reset_env_domains,
                wait_for_reset=True,
                env=env,
                force_homepage=True,
            )
            if code == 0:
                raise Exception(f"Failed to reset env {instance_id} for {failed_sites}.")
            dprint(f"[INFO] Env {instance_id}: successfully reset {reset_env_domains}.")

        # Reset cookies if requested
        if reset_cookies_sites:
            dprint(f"[INFO] Env {instance_id}: resetting cookies {reset_cookies_sites}.")
            code, failed_sites = reset_cookies_with_retry(
                docker_instance_id=str(instance_id),
                sites=reset_cookies_sites,
                exc_comb=reset_cookies_exc_comb,
                wait_for_cookies_reset=True,
                expired_only=False,
            )
            if code != 1:
                raise Exception(f"Failed to reset cookies {failed_sites} for env {instance_id}.")
            dprint(f"[INFO] Env {instance_id}: successfully reset cookies {reset_cookies_sites}.")

        # Update status to ready
        env_pool.update_status(instance_id, Status.READY)
        free_env_pool.put(instance_id)
    except Exception as e:
        if env_pool.get_try_count(instance_id) < MAX_RESET_TRIES:
            dprint(f"[ERROR] Failed to start reset env {instance_id}: {e}, number of tries: {env_pool.get_try_count(instance_id)}")
            env_pool.update_data(instance_id, Status.NEEDS_RESET, 1)
        else:
            dprint(f"[ERROR] Failed to start reset env {instance_id} after {MAX_RESET_TRIES} tries. Removing from pool.")
            env_pool.update_status(instance_id, Status.FAILED)


def job_wrapper(
    env_pool: EnvPool,
    free_env_pool: Queue,
    env_to_batch_id: dict[int, int | None],
    batch_id_counter: int,
    json_config_file: str,
    stop_dict: dict[int, bool],
    agent_config: str,
    tmux_session_id: str,
    batch: TaskBatch,
    result_dir: str,
    captioner_device: str,
    max_wait_env: int = 5 * 60,
    max_steps: int = 30,
) -> tuple[int, str]:
    try:
        deadline = time.time() + max_wait_env
        env_id: int | None = None
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                candidate_env_id = free_env_pool.get(timeout=max(0.1, remaining))
                # Atomically claim only if the env is READY
                if env_pool.compare_and_set_status(candidate_env_id, Status.READY, Status.IN_USE):
                    env_id = candidate_env_id
                    break
                else:
                    # Someone else already claimed or changed it; skip and try again
                    dprint(f"[WARN] Batch {batch_id_counter}, Worker {batch.worker_id}: dequeued env {candidate_env_id} but it was not READY. Retrying...")
                    continue
            except Empty:
                break
        if env_id is None:
            batch.hard_reset = True
            raise RuntimeError(f"Batch {batch_id_counter}, Worker {batch.worker_id}: failed to claim env after {max_wait_env} seconds.")
        dprint(f"[INFO] Batch {batch_id_counter}, Worker {batch.worker_id}, {time.time()}: claimed env {env_id}.")
    except Empty:
        batch.hard_reset = True
        raise RuntimeError(f"Batch {batch_id_counter}, Worker {batch.worker_id}: failed to claim env after {max_wait_env} seconds.")

    env_pool.update_data(env_id=env_id, status=Status.IN_USE, batch_id=batch_id_counter, worker_id=batch.worker_id)
    env_to_batch_id[env_id] = batch_id_counter
    try:
        code, msg = run_sh_script_tmux(
            batch_id=batch_id_counter,
            json_config_file=json_config_file,
            stop_dict=stop_dict,  # type: ignore
            agent_config=agent_config,
            tmux_session_id=tmux_session_id,
            task_list=batch.task_list,
            result_dir=result_dir,
            captioner_device=captioner_device,
            docker_instance_id=env_id,
            worker_id=batch.worker_id,
            max_steps=max_steps,
        )
        dprint(f"[INFO] Batch {batch_id_counter}, Worker {batch.worker_id}: finished running on env {env_id}. Code: {code}, Msg: {msg}")
        env_pool.update_data(env_id, Status.NEEDS_RESET, batch_id=None, worker_id=None)
        env_to_batch_id[env_id] = None
        return ErrorCodes.SUCCESS.value, msg
    except Exception as e:
        dprint(f"[ERROR] Batch {batch_id_counter} failed to run on env {env_id}: {e}")
        env_pool.update_data(env_id=env_id, status=Status.NEEDS_RESET, batch_id=None, worker_id=None)
        env_to_batch_id[env_id] = None
        return ErrorCodes.ERROR.value, str(e)


# LINK TaskDispatcher
def run_tasks_dispatcher(
    task_list: list[str] | set[str],
    num_workers: int,
    json_config_file: str,
    agent_config: str,
    result_dir: str,
    tasks_per_worker: int = 0,
    captioner_device: str = "server-cuda",
    domains_to_reset: list[str] | str = "",
    avg_running_time_per_task: float = 3 * 60,  # Average X minutes per task
    max_attempts_per_task: int = 2,
    shuffle_tasks: bool = True,
    max_running_time: float = 0,
    print_every: int = 10,
    num_envs_per_process: float = 2.0,
    args: dict[str, Any] | None = None,
    attempt_num: int = -1,
    skip_completed: bool = True,
    env_start_idx: int = 0,
    max_steps: int = 30,
) -> None:
    """
    Runs each integer task in 'task_list' with up to 'num_workers' processes,
    scheduling a new batch as soon as a worker finishes one.

    Args:
        task_list (list[int] | set[int]): List of task ids to run.
        num_workers (int): Number of concurent processes allowed at each time.
        json_config_file (str): path to the test configuration directory.
        agent_config (str): path to the agent configuration file.
        result_dir (str): path to the results directory.
        domains_to_reset (list[str]): List of domains to reset.
        captioner_device (str, optional): device to run the captioner on. Defaults to "server-cuda".

        tasks_per_worker (int, optional): number of tasks to assign to each process.
            If 0, divide `task_list` evenly among `num_workers`.
            Defaults to 0.

        reset_after (int, optional): Environments will reset after this many tasks.
            If -1, no reset.
            If 0, estimate based on how many `tasks_per_worker` and number of workers.
            If >0, reset after this many tasks.

        max_running_time (float, optional): Max minutes for each process to finish `tasks_per_worker` tasks.
            If -1, no limit.
            If 0, estimate as: `num of tasks in the batch` * `avg_running_time_per_task`.
            If >0, maximum running time in minutes.

        avg_running_time_per_task (float, optional): Average running time per task. Defaults to 180.

        num_envs_per_process (float, optional): Number of environments to keep running in parallel. Defaults to 2.
            Set this to >1 if wants some environments to be "ready" for the next batch of tasks.

    """
    global CURRENT_TMUX_SESSION, ALL_ENV_IDS, ENV_MANAGERS, PROCESS_POOL_EXECUTOR  # Globals used in cleanup functions.
    args = args or {}

    # Preprocess arguments
    initial_task_list, tasks_per_worker, max_running_time = preprocess_args(num_workers, task_list, tasks_per_worker, max_running_time, avg_running_time_per_task)

    if skip_completed:
        completed, _ = get_tasks_success_failed(result_dir, return_failed=True, attempt_num=attempt_num)
        tasks_to_run = initial_task_list.copy()
        if completed:
            tasks_to_run = set(initial_task_list) - completed
            if tasks_not_run := set(initial_task_list) - tasks_to_run:
                dprint(f"[WARNING] Found {len(tasks_not_run)} tasks completed in {result_dir}. Removing them from the task list.\nTasks ids removed: {tasks_not_run}")
    else:
        tasks_to_run = set(initial_task_list)

    # Shuffle the task list
    tasks_to_run = shuffle_task_list(tasks_to_run) if shuffle_tasks else tasks_to_run

    # Write task list to file
    create_task_txt_file(json_config_file, tasks_to_run, out_dir=result_dir, filename="tasks.txt", overwrite=True)

    # Create batches of tasks
    task_batches = create_task_batches(
        tasks_to_run,
        tasks_per_worker,
        max_run_time=max_running_time,
        avg_run_time_per_task=avg_running_time_per_task,
    )
    # Double check that there are tasks to run.
    if len(task_batches) == 0:
        raise ValueError("No tasks to run.")

    # Helper to print stats and run info.
    printer = Printer(print_every=print_every)

    # Print run configuration info.
    printer.print_run_info(
        agent_config=agent_config,
        json_config_file=json_config_file,
        total_tasks=len(tasks_to_run),
        total_batches=len(task_batches),
        tasks_per_process=tasks_per_worker,
        num_workers=num_workers,
        max_attempts_per_task=max_attempts_per_task,
        shuffle_tasks=shuffle_tasks,
        max_running_time=max_running_time,
        first_batch_max_run_time=task_batches[0].max_run_time,
        avg_running_time_per_task=avg_running_time_per_task,
        domains_to_reset=domains_to_reset,
        num_envs_per_process=num_envs_per_process,
    )

    # Create a tmux session that will be used for running tasks.
    tmux_session_id = create_tmux_session()
    CURRENT_TMUX_SESSION = tmux_session_id

    # Auxiliary variables
    num_envs = int(num_workers * num_envs_per_process)
    batch: TaskBatch
    future_to_batch: dict[Future, TaskBatch] = {}  # Maps futures to their associated job batch
    worker_id_pool: list[int] = list(reversed(range(1, num_workers + 1)))  # IDs to assign to workers. Debugging purposes.
    future_to_env: dict[Future, int] = {}  # Maps futures to their associated env id
    num_restores = 0

    # Keeps track of IDs to assign unique IDs to batches.
    batch_id_counter: int = 0
    # Maps tasks to num retries the task has suffered.
    attempts_per_task: dict[str, int] = {task_id: 0 for task_id in tasks_to_run}

    # Thread-safe dictionary to send stop signals.
    manager = Manager()
    stop_dict = manager.dict()

    # Pool of envs; IDs are used to deploy new websites / containers
    env_pool_manager = BaseManager()
    env_pool_manager.start()
    env_pool: EnvPool = cast(EnvPool, env_pool_manager.EnvPool(num_envs, env_start_idx))  # type: ignore

    free_env_manager = Manager()
    free_env_pool = free_env_manager.Queue()

    env_to_batch_id_manager = Manager()
    env_to_batch_id = env_to_batch_id_manager.dict()

    # Cleanup and logging control
    ALL_ENV_IDS = env_pool.get_env_ids()
    ENV_MANAGERS = [env_pool_manager, free_env_manager, env_to_batch_id_manager]

    initial_task_set = set(initial_task_list)  # avoid creation of sets in every logging iteration.
    tasks_to_run_set = set(tasks_to_run)

    signal_manager.add_cleanup_function(lambda: cleanup(CURRENT_TMUX_SESSION))

    global MAX_CONCURRENT_ENV_RESETS
    if MAX_CONCURRENT_ENV_RESETS <= 0:
        MAX_CONCURRENT_ENV_RESETS = float("inf")

    # Start parallel run
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max(num_workers * num_envs, MAX_THREADPOOL_WORKERS),
    ) as executor:
        PROCESS_POOL_EXECUTOR = executor
        start_time = time.time()
        printer.start_timer()

        # ----------------------------------------------------------------------
        # Submit initial tasks.
        # ----------------------------------------------------------------------
        printer.print_message(f"Initial env initialization and task submission.")
        env_ids = env_pool.get_env_ids()
        active_env_resets = 0  # Track active environment reset jobs

        for env_id in env_ids[: min(num_workers, len(task_batches))]:
            # Submit job to start/reset envs, atomically mark as resetting to avoid duplicate submissions.
            # Only submit if we haven't reached the limit of concurrent env resets
            if active_env_resets < MAX_CONCURRENT_ENV_RESETS and env_pool.compare_and_set_status(env_id, Status.NEEDS_RESET, Status.RESETTING):
                future_env = executor.submit(
                    start_reset_env,
                    env_pool,
                    free_env_pool,
                    env_id,
                    reset_env_domains=domains_to_reset,
                    reset_cookies_sites=[],
                    reset_cookies_exc_comb=EXC_COOKIE_SITE_COMB,
                )
                future_to_env[future_env] = env_id
                active_env_resets += 1

            # Get task batch and assign ID.
            batch = task_batches.pop()
            batch_id_counter += 1
            batch.batch_id = batch_id_counter
            batch.worker_id = worker_id_pool.pop()

            future_job = executor.submit(
                job_wrapper,
                env_pool=env_pool,
                free_env_pool=free_env_pool,
                env_to_batch_id=env_to_batch_id,  # type: ignore
                batch_id_counter=batch_id_counter,
                json_config_file=json_config_file,
                stop_dict=stop_dict,  # type: ignore
                agent_config=agent_config,
                tmux_session_id=CURRENT_TMUX_SESSION,
                batch=batch,
                result_dir=result_dir,
                captioner_device=captioner_device,
                max_wait_env=num_envs * 2 * 60,
                max_steps=max_steps,
            )
            batch.start_run_time = time.time()
            future_to_batch[future_job] = batch
            printer.print_dispatched_batch(batch_id_counter, batch.task_list)

        while future_to_batch:
            # Send stop signals if batches have been running for too long.
            send_stop_signals(future_to_batch, stop_dict)  # type: ignore

            done, _ = concurrent.futures.wait(future_to_batch, timeout=0.5, return_when=concurrent.futures.FIRST_COMPLETED)

            # -------------------------------------------------------------------
            # Orchestrator do work.
            # ------------------------------------------------------------------
            # Print stats
            tasks_finished, _ = get_tasks_success_failed(result_dir, return_failed=False, attempt_num=attempt_num)
            remaining_tasks = [t for batch in task_batches + list(future_to_batch.values()) for t in batch.task_list if attempts_per_task[t] < max_attempts_per_task]
            remaining_tasks = set(remaining_tasks) - tasks_finished
            printer.print_stats(
                tasks_finished,
                initial_task_set,
                tasks_to_run_set,
                json_config_file,
                env_pool.get_all_env_data(),
                result_dir,
                remaining_tasks,
                free_env_pool=free_env_pool,
            )

            # Fill env pool with remaining envs if there are still batches to run
            if future_to_batch and len(task_batches) > 0:
                # Count currently active env reset jobs
                active_env_resets = sum(1 for future in future_to_env.keys() if not future.done())

                for env_id in env_pool.get_env_ids():
                    # Only submit new env reset jobs if we haven't reached the limit
                    if active_env_resets < MAX_CONCURRENT_ENV_RESETS and env_pool.compare_and_set_status(env_id, Status.NEEDS_RESET, Status.RESETTING):
                        future_env = executor.submit(
                            start_reset_env,
                            env_pool,
                            free_env_pool,
                            env_id,
                            reset_env_domains=domains_to_reset,
                            reset_cookies_sites=[],
                            reset_cookies_exc_comb=EXC_COOKIE_SITE_COMB,
                        )
                        future_to_env[future_env] = env_id
                        active_env_resets += 1

            # Save unfinished tasks for future reference
            write_unfinished_tasks(
                original_task_list=initial_task_set,
                tasks_finished=tasks_finished,
                json_config_file=json_config_file,
                out_dir=result_dir,
            )
            # Save stats for future reference
            write_stats_to_csv(
                dir=result_dir,
                tasks_finished=tasks_finished,
                tasks_to_run=tasks_to_run_set,
                json_config_file=json_config_file,
                results_dir=result_dir,
                initial_time=start_time,
                args=args,
                exper_id=args.get("experiment_id", "") if isinstance(args, dict) else "",
            )

            # Restore API keys file if it gets close to empty.
            num_restores += restore_api_keys_file(min_keys={"google": 1, "openai": 0})
            if num_restores > 10:
                dprint(f"[WARNING] Too many API key restores. Stopping run.")
                break

            # Process finished workers.
            for future in done:
                # Get the worker's task batch.
                completed_batch = future_to_batch.pop(future)
                run_time = time.time() - completed_batch.start_run_time
                worker_id_pool.insert(0, completed_batch.worker_id)  # type: ignore

                # Print finish reason for debugging purposes.
                printer.print_batch_completion(completed_batch.batch_id, run_time, future)

                # Update task_batches removing finished tasks so far and adding failed tasks.
                finished_tasks, failed_tasks = get_tasks_success_failed(result_dir, return_failed=True, attempt_num=attempt_num)
                task_batches = recompute_task_batches(
                    waiting_batches=task_batches,
                    submitted_batches=[completed_batch],
                    tasks_finished=finished_tasks,
                    tasks_failed=failed_tasks,
                    attempts_per_task=attempts_per_task,
                    tasks_per_worker=tasks_per_worker,
                    max_attempts=max_attempts_per_task,
                    max_run_time=max_running_time,
                    avg_run_time_per_task=avg_running_time_per_task,
                    shuffle=shuffle_tasks,
                )

                if task_batches:
                    # Submit new batch if available.
                    # Get task batch and assign ID.
                    next_batch = task_batches.pop()
                    batch_id_counter += 1
                    next_batch.batch_id = batch_id_counter
                    next_batch.worker_id = worker_id_pool.pop()

                    # Submit new task batch.
                    future_new = executor.submit(
                        job_wrapper,
                        env_pool=env_pool,
                        free_env_pool=free_env_pool,
                        env_to_batch_id=env_to_batch_id,  # type: ignore
                        batch_id_counter=batch_id_counter,
                        json_config_file=json_config_file,
                        stop_dict=stop_dict,  # type: ignore
                        agent_config=agent_config,
                        tmux_session_id=tmux_session_id,
                        batch=next_batch,
                        result_dir=result_dir,
                        captioner_device=captioner_device,
                        max_steps=max_steps,
                    )
                    future_to_batch[future_new] = next_batch
                    next_batch.start_run_time = time.time()
                    printer.print_dispatched_batch(batch_id_counter, next_batch.task_list)
                else:
                    # If no new batch is available, print message and wait for workers to finish.
                    # Obs.: a new batch can be available in future iterations if tasks fail and are retried.
                    printer.print_waiting_batches(len(future_to_batch))

        # Clean any futures in future to env mapping.
        for future in list(future_to_env.keys()):
            if future.done() or future.cancelled():
                future_to_env.pop(future, None)  # Remove completed futures
            elif not future.done() and not future.cancelled():
                future.cancel()
        # Ensure the pool does not block on remaining env-reset tasks
        executor.shutdown(wait=False, cancel_futures=True)

    # Orchestration done.
    printer.print_all_completed(json_config_file, agent_config, result_dir)


def cleanup(tmux_session_id: str | None, temp_files_dir: str = TEMP_FILES_DIR) -> None:
    # Clean the process pool executor.
    if PROCESS_POOL_EXECUTOR:
        PROCESS_POOL_EXECUTOR.shutdown(wait=False, cancel_futures=True)

    # Clean up any remaining managers.
    for manager in ENV_MANAGERS:
        try:
            manager.shutdown()
        except Exception:
            pass

    # Clean up any hanging children.
    try:
        kill_hanging_children()
    except Exception:
        pass

    # Kill any lingering start_reset_envs.sh processes
    try:
        subprocess.run(["pkill", "-f", RESET_ENV_SCRIPT], check=False)
        for env_id in ALL_ENV_IDS:
            try:
                subprocess.run(["pkill", "-f", f"{RESET_ENV_SCRIPT} -p {env_id}"], check=False)
            except Exception:
                pass
    except Exception:
        pass

    if ALL_ENV_IDS:
        clean_envs(instance_ids=ALL_ENV_IDS)

    # Kill tmux session.
    if tmux_session_id:
        subprocess.run(["tmux", "kill-session", "-t", tmux_session_id], check=True)

    # Clean up any temp files.
    try:
        subprocess.run(["rm", "-rf", temp_files_dir], check=True)
    except Exception:
        pass


def main() -> None:
    args = parse_args()

    # Get config dir and task list
    json_config_file, task_ids = get_task_list(tasks_file=args.tasks_file, json_config_file=args.json_config_file)

    if not args.agent_config or not os.path.isfile(f"{AGENTS_CONFIG_DIR}/{args.agent_config}"):
        raise ValueError(f"Agent config {args.agent_config} not found in {AGENTS_CONFIG_DIR}.")

    # Force domain recognition for reset, cookie renovation, other purposes.
    if not args.domains_to_reset:
        args.domains_to_reset = "all_vwa" if args.env == "vwa" else "all_wa"
        dprint(f"[WARNING] No domain provided. Using `{args.domains_to_reset}` as default.")

    # Host captioner on tmux session, if not running
    dprint("[INFO] Checking / starting captioner...")
    start_captioner(
        model_name="Salesforce/blip2-flan-t5-xl",
        model_device=args.captioner_device,
        tmux_session_name="vwa_captioner",
    )

    # Build result dir
    if not args.results_dir:
        results_dir = DEFAULT_RESULTS_DIR
        model = get_agent_attribute(f"{AGENTS_CONFIG_DIR}/{args.agent_config}", "executor_agent:lm_config:model")
        if model:
            results_dir = f"{results_dir}/{model}"
        date_ann = datetime.now().strftime("%Y-%m-%d-%H%M")
        results_dir = f"{results_dir}/p_run-{date_ann}"
    else:
        results_dir = args.results_dir

    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/prun_params.json", "w") as f:
        json.dump(args.__dict__, f, indent=2)

    dprint(f"[INFO] Running tasks from {json_config_file} with {len(task_ids)} tasks.")

    dprint(f"\n[INFO] Results will be saved in {results_dir}")

    # Set seed
    set_seed(args.seed)

    # Add experiment ID to args
    args.experiment_id = args.experiment_id or datetime.now().strftime("%Y-%m-%d_%H-%M")
    # Run tasks dispatcher

    try:
        run_tasks_dispatcher(
            task_list=task_ids,
            num_workers=args.num_processes,
            json_config_file=json_config_file,
            agent_config=args.agent_config,
            result_dir=results_dir,
            domains_to_reset=args.domains_to_reset,
            tasks_per_worker=args.tasks_per_process,
            captioner_device=args.captioner_device,
            avg_running_time_per_task=args.avg_running_time_per_task,
            max_attempts_per_task=args.max_attempts_per_task,
            shuffle_tasks=args.shuffle_tasks,
            max_running_time=args.max_running_time,
            num_envs_per_process=args.num_envs_per_process,
            args=args.__dict__,
            attempt_num=args.attempt_num,
            skip_completed=args.skip_completed,
            env_start_idx=args.env_start_idx,
            max_steps=args.max_steps,
        )

    except Exception as e:
        dprint(f"An error occurred during parallel run: {e}")
    finally:
        # Save unfinished tasks for future reference
        write_unfinished_tasks(
            original_task_list=task_ids,
            json_config_file=json_config_file,
            out_dir=results_dir,
            result_dir=results_dir,
        )
        cleanup(CURRENT_TMUX_SESSION)


if __name__ == "__main__":
    main()

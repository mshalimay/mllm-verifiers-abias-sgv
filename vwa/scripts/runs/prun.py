# autopep8: off
# fmt: off
import datetime
import os
import shutil
import subprocess
import threading
import time
from typing import Any

from utils_vwa.captioner_utils import is_captioner_running

from core_utils.signal_utils import signal_manager

# Usage: 
# python -u -m scripts.runs.p_run.py
# nohup python -u -m scripts.runs.p_run.py > prun.log 2>&1 & disown  

#===============================================
# Configs
#===============================================

# Each dict corresponds to one instance of parallel_run.py
# - agent_config: path to the agent config yaml file
# - task_list: path to the task list file
# - results_dir: directory to save the results
# - block: whether to block/wait before starting next parallel.run.py instance
# - num_processes: split tasks into `num_processes` concurrent processes, each with its own docker envs
# - max_steps: max steps per task

task_list = "evaluation_harness/task_lists/all_vwa-lite.txt"
run_configs = [
    {"agent_config": "noverifier.yaml", "task_list": task_list, "results_dir": "experiments/gemini-2.5-flash/debug", "block":True, "num_processes":3, "max_steps": 30},
]

#---- Params for parallel_run.py instances -----
start_env_idx = 0                       # Creation of docker environment IDs start from this index
tasks_per_process = 5                   # Max num of tasks for each process
max_running_time = -1                   # Max runtime (mins) of each run.py execution: -1 => no limit; 0 => dynamically set based on `avg_run_time_per_task` * `# tasks in the batch`
avg_run_time_per_task = 15*60           # Avg runtime (secs) of each task. Used to compute the max running time if `max-running-time` is set to 0.
shuffle_tasks = True                    # Shuffle the task list before parallel execution
seed = 42                               # Random seed
max_retry_per_task = 2                  # If error on task execution, it will be retried up to `max_retry_per_task` times (e.g.: 2 => try at most 2 times).
attempt_num = -1                        # Attempt number. Used to identify the attempt for a given task id. Not implemented yet.
skip_completed = True                   # Tasks in `results_dir` that are already completed will be skipped

domains_to_reset = "all_vwa"                                # Domains to start/reset. If empty, infer from `test_config_dir`
captioner_device = "server-cuda:0"                          # Device to hold the captioner

# Number of environments to maintain ready
# e.g.: If num_processes=3, num_envs = 5 => keeps 2 additional envs ready to dispatch when a batch of tasks is finished by a process.
for run_config in run_configs:
    num_envs = run_config["num_processes"] + 1
    run_config["num_envs_per_process"] = num_envs/run_config["num_processes"]

#===============================================
# Helpers
#===============================================
def build_command(run_config: dict[str, Any]) -> list[str]:
    """
    Build the command list to be executed for the given agent configuration
    and task list.
    """
    agent_config = run_config["agent_config"]
    task_list = run_config["task_list"]
    results_dir = run_config.get("results_dir", "")
    max_steps = run_config.get("max_steps", 30)
    cmd = [
        "python",
        "-u",                
        "-m", "scripts.runs.parallel_run",
        "-a", agent_config,
        "-t", task_list,
        "-n", str(run_config["num_processes"]),
        "-b", str(tasks_per_process),
        "-mrt", str(max_running_time),
        "-art", str(avg_run_time_per_task),
        "-d", domains_to_reset,
        "-ma", str(max_retry_per_task),
        "-cd", captioner_device,
        "-seed", str(seed),
        "-nenvs", str(run_config["num_envs_per_process"]),
        "-r", results_dir,
        "-at", str(attempt_num),
        "-esi", str(run_config.get("start_env_idx", start_env_idx)),
        "-x", str(max_steps),
    ]
    if skip_completed:
        cmd.append("-sc")
    if shuffle_tasks:
        cmd.append("-st")
    return cmd

def get_header(run_config: dict[str, Any]) -> str:
    """
    Generate a header string for the per-process log file.
    """
    agent_config = run_config["agent_config"]
    task_list = run_config["task_list"]
    header = (
        f"\n{'-'*80}\n"
        f"Running agent config: {agent_config}\n"
        f"Task list: {task_list}\n"
        f"{'-'*80}\n\n"
    )
    return header

def stream_output(proc: subprocess.Popen[Any], log_file: str) -> None:
    """Stream subprocess output, writing to both global logger and process-specific log."""
    with open(log_file, "a") as lf:
        for line in proc.stdout:  # type: ignore
            print(line.strip())
            lf.write(line)
            lf.flush()

# Global variable to track child processes for cleanup
child_processes = []

# Cleanup function and signal handler
def cleanup_processes() -> None:
    print("Cleaning up processes and resources...", flush=True)
    
    # Send termination signal to all child processes
    print("Sending termination signal to child processes...", flush=True)
    for proc in child_processes:
        try:
            if proc.poll() is None:  # Process is still running
                proc.terminate()
        except Exception as e:
            print(f"Error terminating process {proc.pid}: {e}", flush=True)
    
    # Wait for processes to finish naturally (with timeout)
    print("Waiting for processes to finish gracefully...", flush=True)
    timeout = 10
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if all(proc.poll() is not None for proc in child_processes):
            print("All processes finished gracefully", flush=True)
            break
        time.sleep(1)
    else:
        print("Timeout reached, force killing remaining processes...", flush=True)
        for proc in child_processes:
            try:
                if proc.poll() is None:
                    proc.kill()
            except Exception as _:
                os.system(
                    r"ps aux | grep 'parallel_run' | grep -v grep | awk '{print $2}' | xargs -r kill"
                )
    
    # Clean up any temporary files
    from .parallel_run import TEMP_FILES_DIR
    if os.path.exists(TEMP_FILES_DIR):
        print("Removing temporary files...", flush=True)
        shutil.rmtree(TEMP_FILES_DIR, ignore_errors=True)
    
    print("Cleanup complete.", flush=True)

signal_manager.add_cleanup_function(cleanup_processes)

#===============================================
# Main
#===============================================
try:    
    os.makedirs("log_files", exist_ok=True)

    # Start captioner
    if not is_captioner_running():
        print("Starting captioner...", flush=True)    
        result = subprocess.run([
            "python", "-m", "utils_vwa.captioner_utils", 
            "--model_name", "Salesforce/blip2-flan-t5-xl", 
            "--model_device", captioner_device, 
            "--port", "9555", 
            "--endpoint", "http://localhost:9555/caption/", 
            "--tmux_session_name", "vwa_captioner"], check=True)
        print(result, flush=True)

    # Launch parallel runs
    print(f"Launching parallel runs", flush=True)
    print(f"Configs: {run_configs}", flush=True)
    threads = []
    env_id_start = start_env_idx
    for run_config in run_configs:
        unique_log_file = f"log_files/prun_many_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Write header to the per-process log file.
        with open(unique_log_file, "a") as lf:
            lf.write(get_header(run_config))
            lf.flush()

        run_config["start_env_idx"] = env_id_start
        # Ensure env ids for next run are non-overlapping with the current run.
        env_id_start += int(run_config["num_envs_per_process"] * run_config["num_processes"]) + 1
        cmd = build_command(run_config)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            cwd=os.getcwd(),
        )
        # Track the child process for cleanup
        child_processes.append(proc)
        if not run_config.get("block", False):
            t = threading.Thread(target=stream_output, args=(proc, unique_log_file))
            t.start()
            threads.append(t)
            time.sleep(5)
        else:
            t = threading.Thread(target=stream_output, args=(proc, unique_log_file))
            t.start()
            proc.wait()
            t.join()

    # Wait for all non-blocking process threads to finish.
    for t in threads:
        t.join()

    print("All processes finished.", flush=True)


except Exception as e:
    print(f"Error: {e}", flush=True)

finally:
    cleanup_processes()

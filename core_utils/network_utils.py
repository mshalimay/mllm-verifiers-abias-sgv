import socket
import subprocess
import time

import requests


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]  # type: ignore


def wait_for_server(
    endpoint: str,
    max_retries: int | float = 10,
    total_time: int = 10,
    timeout: int = 5,
    sleep_for: int = 1,
    verbose: bool = False,
) -> bool:
    retries = 0
    time_start = time.time()
    print(f"Waiting for server to start at {endpoint}. Total time: {total_time} seconds, Max retries: {max_retries}")
    while time.time() - time_start < total_time:
        if retries > max_retries:
            raise RuntimeError(f"Server not found. Failed to start server after {max_retries} retries")
        try:
            if is_server_running(endpoint=endpoint, timeout=timeout, verbose=verbose):
                print(f"[INFO] Server running on {endpoint}")
                return True
        except Exception as e:
            print(f"[ERROR] Error checking server {endpoint}: {e}")
            return False

        retries += 1
        time.sleep(sleep_for)
    return False


def is_server_running(endpoint: str, timeout: int = 10, verbose: bool = True) -> bool:
    is_running = False
    if verbose:
        print(f"Checking server endpoint: {endpoint}")
    try:
        response = requests.post(endpoint, timeout=timeout)
        if response.status_code:
            is_running = True
            print(f"[INFO] Server running on {endpoint}")
        else:
            is_running = False
    except requests.exceptions.RequestException as _:
        is_running = False
    except Exception as _:
        raise Exception(f"Error checking server: {endpoint}")
    return is_running


def start_server(
    model_name: str,
    model_device: str,
    script: str,
    port: int,
    endpoint: str,
    tmux_session_name: str,
    max_retries: int | float = 3,
) -> None:
    if is_server_running(endpoint=endpoint, timeout=1):
        print(f"[INFO] `{model_name}` running on {endpoint}")
        return

    retries, is_running = 0, False
    while True:
        if retries > max_retries:
            raise RuntimeError(f"Server not found. Failed to start server script {script} after {max_retries} retries")

        # Try to start a tmux session hosting the captioner
        try:
            tmux_command = (
                f'tmux new-session -d -s "{tmux_session_name}" '
                # f'"conda init; conda activate {conda_env}; python {script} --model_name {model_name} --device {model_device} --port {port}"'
                f'"python {script} --model_name {model_name} --device {model_device} --port {port}"'
            )
            print(f"Trying to start server on tmux with command: `{tmux_command}`")
            _ = subprocess.run(
                tmux_command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,  # Raise an error if the command fails
            )

        except subprocess.CalledProcessError as e:
            if "duplicate session" in e.stderr:
                is_running = True
                pass

            else:
                print("tmux command failed:")
                print("stdout:", e.stdout)
                print("stderr:", e.stderr)
                raise

        is_running = is_running or is_server_running(endpoint=endpoint)
        if is_running:
            print(f"[INFO] `{script}` running on {endpoint}")
            return
        retries += 1

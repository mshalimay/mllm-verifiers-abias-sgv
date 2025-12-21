import fcntl
import hashlib
import os
import subprocess
import time
from contextlib import contextmanager
from typing import Generator

import pandas as pd
from filelock import FileLock

from core_utils.signal_utils import signal_manager

LOCKS = {}


def cleanup():
    global LOCKS
    for lock in list(LOCKS.keys()):
        try:
            lock.release(force=True)
        except Exception as e:
            print(f"{__file__}: Error releasing lock: {e}")


signal_manager.add_cleanup_function(cleanup)


def get_file_lock(file_path: str, timeout: int = 60) -> FileLock:
    """
    Generate a unique lock file using a hidden file (starting with .)
    based on the provided file path.
    """
    directory, filename = os.path.split(file_path)
    lock_file_name = os.path.join(directory, f".{filename}.lock")
    lock = FileLock(lock_file_name, timeout=timeout)
    if lock not in LOCKS:
        LOCKS[lock] = lock
    return LOCKS[lock]


def get_lock_file(identifier: str, directory: str = "/tmp") -> str:
    """
    Generate a unique lock file name based on a unique identifier,
    typically the absolute path of the target script.
    """
    identifier_hash = hashlib.md5(identifier.encode("utf-8")).hexdigest()
    return os.path.join(directory, f"{identifier_hash}.lock")


@contextmanager
def single_instance_lock(
    lock_file: str = "",
    identifier: str = "",
    retry_interval: float = 1.0,
    max_wait: float = 60,
) -> Generator:
    """
    Context manager that acquires a file lock on the given file.
    If the lock is not acquired within `max_wait` seconds,
    a TimeoutError is raised.
    """
    if not lock_file and not identifier:
        raise ValueError("Either lock_file or identifier must be provided.")

    if not lock_file:
        lock_file = get_lock_file(identifier)

    fp = open(lock_file, "w")
    start_time = time.time()
    acquired = False  # Track whether we have acquired the lock

    try:
        while time.time() - start_time < max_wait:
            try:
                fcntl.flock(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                time.sleep(retry_interval)

        if not acquired:
            raise TimeoutError(f"[ERROR] Timeout reached ({max_wait} sec) while waiting for lock on {lock_file}.")

        yield  # Proceed to the critical section if lock was acquired.

    finally:
        if acquired:
            fcntl.flock(fp, fcntl.LOCK_UN)
        fp.close()


# TODO: finish this function
def grep_wait_for_process(pattern: str, max_wait: int = 60, retry_interval: float = 0.5) -> bool:
    """
    Wait until no process matching the given pattern is running.

    This function periodically checks (using "pgrep -f")
    if there is any process whose command line matches the provided pattern.
    If such processes exist, it waits until they are finished or until the
    maximum wait time is reached.

    NOTE: use a pattern like '[a]uto_login' so that the search
    does not inadvertently match the current process.

    Parameters:
        pattern (str): The regex pattern to search for in running processes.
        max_wait (int): Maximum number of seconds to wait. Defaults to 60.
        retry_interval (float): Seconds to wait between checks. Defaults to 0.5 seconds.

    Returns:
        bool: True if no matching process was found before timeout,
              False if the timeout was reached while matching processes still exist.

    Example:
        # Wait until no process containing "auto_login" is running.
        wait_for_process("[a]uto_login", max_wait=120)
    """
    start_time = time.time()
    while time.time() - start_time < max_wait:
        # Call pgrep to check for running processes that match the pattern
        result = subprocess.run(f"pgrep -f '{pattern}'", shell=True, capture_output=True, text=True)
        pids = result.stdout.strip()

        if not pids:
            # No matching process found.
            return True
        else:
            print(f"[INFO] Processes matching pattern '{pattern}' still running (PIDs: {pids}). Waiting...")
            time.sleep(retry_interval)

    print(f"[WARN] Timeout reached ({max_wait} sec) while waiting for processes matching '{pattern}' to finish.")
    return False


# TODO: finish this function
def grep_wait_run_process(pattern: str, cmd: str, max_wait: int = 60, retry_interval: float = 0.5) -> subprocess.CompletedProcess | None:
    """
    Wait until no process matching the given pattern is running.
    If such processes exist, it waits until they are finished or until the
    maximum wait time is reached.
    """
    try:
        if grep_wait_for_process(pattern, max_wait, retry_interval):
            return subprocess.run(cmd, shell=True)
        else:
            print(f"[ERROR] Timeout reached ({max_wait} sec) while waiting for processes matching '{pattern}' to finish.")
            return None
    except Exception as e:
        print(f"[ERROR] Error running command: {e}")
        return None


def atomic_save_file(file_path: str, data: str) -> None:
    """
    Save data to a file atomically.
    """
    tmp_file_path = file_path + ".tmp"
    with open(tmp_file_path, "w") as f:
        f.write(data)
    os.rename(tmp_file_path, file_path)


def atomic_save_df(df: pd.DataFrame, file_path: str) -> None:
    """
    Save a pandas DataFrame to a file atomically.
    """
    tmp_file_path = file_path + ".tmp"
    df.to_csv(tmp_file_path, index=False)
    os.rename(tmp_file_path, file_path)

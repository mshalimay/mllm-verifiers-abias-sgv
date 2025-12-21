import logging
import os
import random
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

from core_utils import LOG_FOLDER
from core_utils.signal_utils import signal_manager

# Creates a unique log file name with timestamp and random number to avoid collisions
Path(LOG_FOLDER).mkdir(parents=True, exist_ok=True)
now = datetime.now()
LOG_FILE_PATH = Path(f"{LOG_FOLDER}/log_{now.strftime('%Y-%m-%d_%H-%M-%S')}__{int(now.microsecond / 1000):03d}ms__{random.randint(0, 9999)}.log").resolve().as_posix()


def setup_logger(name: str = "logger", log_file_path: str = LOG_FILE_PATH) -> logging.Logger:
    logger = logging.getLogger(name)
    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # Console handler and logging format
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("[%(levelname)s] [%(filename)s:%(lineno)d]: %(message)s")
        console_handler.setFormatter(console_formatter)

        # File handler and logging format
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter("%(asctime)s - [%(levelname)s] [%(filename)s:%(lineno)d]: %(message)s")
        file_handler.setFormatter(file_formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        print(f"LOGGER INITIALIZED. Saving logs to: {log_file_path}")
    return logger


def save_log_file_path(save_dir: str) -> None:
    with open(os.path.join(save_dir, "log_files.txt"), "a+") as f:
        f.write(f"'{LOG_FILE_PATH}'\n")


def save_log_file(save_dir: str, log_filename: str = "log.txt") -> None:
    try:
        shutil.copy(LOG_FILE_PATH, os.path.join(save_dir, log_filename))
    except Exception as e:
        logger.error(f"Error saving log file: {e}")


def cleanup_logs(log_dir: str = LOG_FOLDER, cleanup_threshold_days: int = 5, cleanup_threshold_kb: int = 1) -> None:
    """
    Deletes log files in the specified directory that are empty or whitespace-only,
    Args:
        log_dir (str): Directory where log files are stored.
        cleanup_threshold_days (int): Number of days after which logs are considered for cleanup.
        cleanup_threshold_kb (int): Size threshold in KB below which logs are considered for cleanup
    """
    try:
        log_directory = Path(log_dir)
        cutoff_time = datetime.now() - timedelta(days=cleanup_threshold_days)
        for log_file in log_directory.glob("*.log"):
            if not log_file.is_file():
                continue

            # If the file is completely empty (0 bytes), delete it immediately.
            if log_file.stat().st_size <= 0:
                # logger.info(f"Deleting empty log file: {log_file.name}")
                log_file.unlink()
                continue

            # Check if contains only whitespace
            try:
                with log_file.open("r", encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError:
                # Fallback in case of decoding issues
                with log_file.open("r", encoding="latin-1") as f:
                    content = f.read()

            # Delete the file if it contains only whitespace.
            if content.strip() == "":
                # logger.info(f"Deleting whitespace-only log file: {log_file.name}")
                log_file.unlink()

            # Otherwise, delete files by checking date and size.

            # Only consider files older than the cutoff
            file_mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
            if file_mtime > cutoff_time:
                continue

            # If the file is completely empty (0 bytes), delete it immediately.
            if log_file.stat().st_size <= cleanup_threshold_kb * 1024:
                # logger.info(f"Deleting empty log file: {log_file.name}")
                log_file.unlink()
                continue

    except Exception as e:
        logger.error(f"Error cleaning up logs: {e}")


# Create a default logger when module is imported
signal_manager.add_cleanup_function(cleanup_logs)
logger = setup_logger()


def handle_uncaught_exception(exc_type, exc_value, exc_traceback):
    """
    This function logs all uncaught exceptions other than KeyboardInterrupt.
    By setting sys.excepthook, any uncaught exceptions in the script
    importing this logger will be logged.
    """
    # Do not log KeyboardInterrupt as an error.
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


# Set the custom exception handler to log unhandled exceptions.
sys.excepthook = handle_uncaught_exception


if __name__ == "__main__":
    cleanup_logs(log_dir=LOG_FOLDER, cleanup_threshold_days=0, cleanup_threshold_kb=10_0000)  # For testing purposes

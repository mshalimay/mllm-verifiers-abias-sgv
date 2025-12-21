import json
import os
import random
import time
from functools import partial
from typing import Any, Optional

from filelock import Timeout

from core_utils.concurrency_utils import get_file_lock
from core_utils.logger_utils import logger
from core_utils.signal_utils import signal_manager
from core_utils.timing_utils import timeit
from llms.constants import API_KEY_ENV_VARS
from llms.constants.constants import API_KEYS_PATH
from llms.setup_utils import safe_add_key_to_file, safe_remove_key_from_file

# Default value
MAX_API_KEY_RETRY = 2
MAX_KEY_PROCESS_COUNT = 1
LOCK_TIMEOUT = 60


class NoAPIKeyException(Exception):
    """
    Custom exception raised when no API key is available.
    """

    def __init__(self, message: str = "No API keys available.") -> None:
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {self.message}"


class ClientManager:
    def __init__(
        self,
        provider: str,
        max_api_key_retry: int | float = MAX_API_KEY_RETRY,
        max_key_process_count: int | float = MAX_KEY_PROCESS_COUNT,
        api_key_env_var: str | None = None,
        force_key_from_file: bool = False,
        api_key: str = "",
    ) -> None:
        self.api_key: str = api_key
        self.client: Optional[Any] = None
        self.aclient: Optional[Any] = None
        self.max_api_key_retry = max_api_key_retry  # Maximum number of times to retry an API key
        self.api_keys_retry_count: dict[str, int | float] = {}  # Track the number of times an API key has been used
        self.provider = provider
        self.api_key_env_var = api_key_env_var or API_KEY_ENV_VARS[provider]
        self.api_process_count: dict[str, int] = {}
        self.max_key_process_count = max_key_process_count
        self.force_key_from_file = force_key_from_file
        self.last_reset_time = 0
        self.api_keys_start_time = {}

    def fetch_api_key(self) -> None:
        try:
            if not self.force_key_from_file and os.environ.get(self.api_key_env_var):
                # Fetch API key from environment variables if available
                self.api_key = os.environ[self.api_key_env_var]
                self.api_keys_retry_count[self.api_key] = 0
                self.api_process_count[self.api_key] = 1
                # If API key in env variables, remove it from repo to prevent other processes from using it
                try:
                    if self.api_process_count[self.api_key] >= self.max_key_process_count:
                        safe_remove_key_from_file(self.api_key, provider=self.provider, logger=logger)
                except FileNotFoundError:
                    pass
            else:
                # Fetch API key from local file if available
                self.api_key = self._get_add_remove_key_from_file() or ""
                if self.api_key:
                    os.environ[self.api_key_env_var] = self.api_key
                else:
                    raise NoAPIKeyException(f"No API keys available during initialization for {self.provider} client")
            self.api_keys_start_time[self.api_key] = time.time()
        except Exception as e:
            raise Exception(f"Error while fetching {self.provider} API key: {e}")

    def _get_add_remove_key_from_file(self, previous_api_key: str = "") -> str | None:
        """Safely retrieve and remove an API key from the shared file."""
        try:
            lock = get_file_lock(API_KEYS_PATH, timeout=LOCK_TIMEOUT)
            with lock, open(API_KEYS_PATH, "r+") as file:
                # Read all API keys
                api_keys = json.load(file)
                provider_keys = api_keys.get(self.provider, [])

                # If previous API key provided (changing keys), decrease num process using it
                if previous_api_key:
                    self.api_process_count[previous_api_key] -= 1

                # If previous API key provided hasn't exceeded retry limit add it back to the end of the list
                retry_count = self.api_keys_retry_count.get(previous_api_key, 0)
                if previous_api_key and retry_count < self.max_api_key_retry:
                    logger.info(f"Adding back {previous_api_key[-5:] if previous_api_key else None} to {self.provider} API keys. Retry count: {retry_count}")
                    if provider_keys:
                        if previous_api_key not in provider_keys:
                            provider_keys.append(previous_api_key)
                    else:
                        provider_keys = [previous_api_key]
                elif previous_api_key:
                    logger.info(f"{previous_api_key[-5:] if previous_api_key else None} has exceeded the retry limit: {retry_count}")

                # If no API keys available, return None
                if not provider_keys:
                    return None

                # Try to get an API key that hasn't exceeded the process count limit
                api_key = None
                random.shuffle(provider_keys)
                for i, candidate_key in enumerate(provider_keys):
                    # Initialize retry count if not exists
                    if candidate_key not in self.api_keys_retry_count:
                        self.api_keys_retry_count[candidate_key] = 0

                    # Initialize process count if not exists
                    if candidate_key not in self.api_process_count:
                        self.api_process_count[candidate_key] = 0

                    proposed_process_count = self.api_process_count[candidate_key] + 1
                    if proposed_process_count > self.max_key_process_count:
                        api_key = None
                        continue
                    else:
                        # FIXME: clean this logic
                        # Update process count
                        self.api_process_count[candidate_key] = proposed_process_count
                        # Select this key
                        api_key = candidate_key
                        # Remove the key from the file if it's reached the process limit
                        if proposed_process_count >= self.max_key_process_count:
                            provider_keys.pop(i)
                        break

                if not api_key:
                    return None

                # Reset file position and write updated keys
                api_keys[self.provider] = provider_keys
                file.seek(0)
                json.dump(api_keys, file, indent=2)
                file.truncate()
                signal_manager.add_cleanup_function(partial(safe_add_key_to_file, api_key=api_key, provider=self.provider, logger=logger))

                return api_key  # type: ignore
        except Timeout:
            logger.info(f"Failed to acquire lock on API keys file after {LOCK_TIMEOUT} seconds")
            return None
        except Exception as e:
            logger.info(f"Error while fetching {self.provider} API key: {e}")
            return None

    @timeit(custom_name="LLM:reset_api_key")
    def reset_api_key(self) -> None:
        """
        Reset the API key by retrieving a new one and update the {provider} client.
        """
        while True:
            new_api_key = self._get_add_remove_key_from_file(previous_api_key=self.api_key)
            if not new_api_key:
                raise Exception("Resources exhausted and no other API keys available.")
            self.api_key = new_api_key
            break

        os.environ[self.api_key_env_var] = self.api_key
        self.set_client()
        logger.info(f"API key and {self.provider} client were redefined.")
        logger.info(f"New API key: {self.api_key[-5:] if self.api_key else None}")
        self.last_reset_time = time.time()
        self.api_keys_start_time[self.api_key] = time.time()

    def get_start_time(self, api_key: str | None) -> float:
        if not api_key:
            return self.last_reset_time
        if api_key not in self.api_keys_start_time:
            return self.last_reset_time
        return self.api_keys_start_time[api_key]

    def set_aclient(self) -> None:
        """Set the async client using the API key from client manager."""
        raise NotImplementedError("Subclasses must implement this method")

    def set_client(self) -> None:
        """Set the sync client using the API key from client manager."""
        raise NotImplementedError("Subclasses must implement this method")

    def get_client(self) -> Any:
        """Return the provider client instance."""
        if not self.client:
            try:
                self.set_client()
            except Exception as e:
                raise Exception(f"Error getting {self.provider} client: {e}")
        return self.client

    def get_aclient(self) -> Any:
        """Return the provider async client instance."""
        if not self.aclient:
            try:
                self.set_aclient()
            except Exception as e:
                raise Exception(f"Error getting {self.provider} async client: {e}")
        return self.aclient

    def close_client(self) -> None:
        # Add key back to the file
        if not self.api_key:
            return
        safe_add_key_to_file(self.api_key, provider=self.provider, logger=None)
        self.api_key = ""
        self.client = None
        self.aclient = None

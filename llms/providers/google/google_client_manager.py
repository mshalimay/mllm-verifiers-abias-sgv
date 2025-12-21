from typing import Dict

from google import genai
from google.genai import types as genai_types

from llms.providers.client_manager import ClientManager
from llms.providers.google.constants import (
    API_VERSION,
    DEFAULT_REQUEST_TIMEOUT,
    MAX_API_KEY_RETRY,
    MAX_KEY_PROCESS_COUNT,
)


class GoogleClientManager(ClientManager):
    def __init__(self, api_key: str = "", force_key_from_file: bool = False) -> None:
        super().__init__(
            provider="google",
            api_key=api_key,
            max_api_key_retry=MAX_API_KEY_RETRY,
            max_key_process_count=MAX_KEY_PROCESS_COUNT,
            force_key_from_file=force_key_from_file,
        )

    def set_client(self) -> None:
        """Set the client using the API key from client manager."""
        try:
            # If no API key, fetch it
            if not self.api_key:
                self.fetch_api_key()

            # Set Google client
            self.client = genai.Client(
                api_key=self.api_key,
                http_options=genai_types.HttpOptions(api_version=API_VERSION, timeout=DEFAULT_REQUEST_TIMEOUT),
            )
        except Exception as e:
            raise Exception(f"Error setting {self.provider} client: {e}")

    def set_aclient(self) -> None:
        # Google sync and async clients are the same
        self.set_client()

    def get_client(self) -> genai.Client:
        """Get the client using the API key from client manager.
        If client is not set, automatically try to set it by fetching the API keys."""
        return self.client if self.client else super().get_client()  # type: ignore

    def get_aclient(self) -> genai.Client:
        # Google sync and async clients are the same
        return self.get_client()


# Module-level instance (singleton)
_global_client_manager: Dict[str, GoogleClientManager] = {}


def get_client_manager(client_manager_id: int | str = 0, api_key: str | None = None) -> GoogleClientManager:
    global _global_client_manager

    client_manager_id = str(client_manager_id)

    if client_manager_id not in _global_client_manager:
        force_key_from_file = False if api_key else True

        if len(_global_client_manager) > 0:
            _global_client_manager[client_manager_id] = GoogleClientManager(force_key_from_file=force_key_from_file)
        else:
            _global_client_manager[client_manager_id] = GoogleClientManager()
    return _global_client_manager[client_manager_id]


def get_client_managers() -> Dict[str, GoogleClientManager]:
    global _global_client_manager
    return _global_client_manager

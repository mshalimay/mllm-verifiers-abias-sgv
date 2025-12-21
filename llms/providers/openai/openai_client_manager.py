from typing import Any, Dict

import openai

from core_utils.logger_utils import logger
from llms.providers.client_manager import ClientManager
from llms.providers.openai.constants import DEFAULT_TIMEOUT, MAX_API_KEY_RETRY, MAX_KEY_PROCESS_COUNT


class OpenAIClientManager(ClientManager):
    def __init__(
        self,
        api_key: str = "",
        base_url: str | None = None,
        model_id: str | None = None,
        provider: str = "openai",
        max_api_key_retry: int = MAX_API_KEY_RETRY,
        max_key_process_count: int = MAX_KEY_PROCESS_COUNT,
    ) -> None:
        super().__init__(
            provider=provider,
            max_api_key_retry=max_api_key_retry,
            max_key_process_count=max_key_process_count,
        )
        self.api_key = api_key
        self.base_url = base_url
        self.model_id = model_id

    def set_client(self) -> None:
        """Set the client using the API key from client manager."""
        try:
            # If no API key, fetch it
            if not self.api_key:
                self.fetch_api_key()

            # Set OpenAI client
            if self.base_url:
                self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=DEFAULT_TIMEOUT)
            else:
                self.client = openai.OpenAI(api_key=self.api_key, timeout=DEFAULT_TIMEOUT)
        except Exception as e:
            raise Exception(f"Error setting {self.provider} client: {e}")

    def set_aclient(self) -> None:
        """Set the async client using the API key from client manager."""
        try:
            # If no API key, fetch it
            if not self.api_key:
                self.fetch_api_key()

            # Set OpenAI async client
            if self.base_url:
                self.aclient = openai.AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, timeout=DEFAULT_TIMEOUT)
            else:
                self.aclient = openai.AsyncOpenAI(api_key=self.api_key, timeout=DEFAULT_TIMEOUT)
        except Exception as e:
            raise Exception(f"Error setting {self.provider} async client: {e}")

    def get_client(self) -> openai.OpenAI:
        """Get the client using the API key from client manager.
        If client is not set, automatically try to set it by fetching the API keys."""
        return self.client if self.client else super().get_client()  # type: ignore

    def get_aclient(self) -> openai.AsyncOpenAI:
        """Get the async client using the API key from client manager.
        If async client is not set, automatically try to set it by fetching the API keys."""
        return self.aclient if self.aclient else super().get_aclient()  # type: ignore


# ================================================================================
# Tools to get information on OpenAI models
# ================================================================================
def get_model_info(model_id: str) -> dict[str, Any]:
    client = get_client_manager().get_client()
    try:
        model_info = client.models.retrieve(model=model_id)
        return model_info.model_dump()
    except Exception as e:
        raise Exception(f"Failed to fetch model info for '{model_id}': {e}")


def get_openai_models() -> list[dict[str, Any]]:
    client = get_client_manager().get_client()
    all_models = []
    for model in client.models.list():
        all_models.append({"model_path": model.id, "data": model.model_dump()})
    # get_client_manager().close_client()
    return all_models


# ================================================================================
# LINK Client setter and getters
# ================================================================================
# Module-level instance (singleton)
_global_client_manager: Dict[str, OpenAIClientManager] = {}


def get_client_manager(
    model_id: str = "",
    api_key: str = "",
    base_url: str | None = None,
    provider: str = "openai",
    max_api_key_retry: int = MAX_API_KEY_RETRY,
    max_key_process_count: int = MAX_KEY_PROCESS_COUNT,
    timeout: int = DEFAULT_TIMEOUT,
) -> OpenAIClientManager:
    global _global_client_manager

    if not model_id:
        _global_client_manager["default"] = OpenAIClientManager(
            api_key=api_key,
            base_url=base_url,
            provider=provider,
            max_api_key_retry=max_api_key_retry,
            max_key_process_count=max_key_process_count,
        )
        return _global_client_manager["default"]

    else:
        if model_id not in _global_client_manager:
            logger.info(f"Creating new OpenAIClientManager for model_id: {model_id}")
            _global_client_manager[model_id] = OpenAIClientManager(
                model_id=model_id,
                base_url=base_url,
                api_key=api_key,
                provider=provider,
                max_api_key_retry=max_api_key_retry,
                max_key_process_count=max_key_process_count,
            )
        return _global_client_manager[model_id]

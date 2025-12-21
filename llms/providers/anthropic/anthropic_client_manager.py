from typing import Any, Dict

import anthropic

from llms.providers.client_manager import ClientManager
from llms.providers.openai.constants import MAX_API_KEY_RETRY, MAX_KEY_PROCESS_COUNT


class AnthropicClientManager(ClientManager):
    def __init__(
        self,
        api_key: str = "",
        base_url: str | None = None,
        model_id: str | None = None,
    ) -> None:
        super().__init__(
            provider="anthropic",
            max_api_key_retry=MAX_API_KEY_RETRY,
            max_key_process_count=MAX_KEY_PROCESS_COUNT,
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

            # Set Anthropic client
            if self.base_url:
                self.client = anthropic.Anthropic(api_key=self.api_key, base_url=self.base_url)
            else:
                self.client = anthropic.Anthropic(api_key=self.api_key)
        except Exception as e:
            raise Exception(f"Error setting {self.provider} client: {e}")

    def set_aclient(self) -> None:
        """Set the async client using the API key from client manager."""
        try:
            # If no API key, fetch it
            if not self.api_key:
                self.fetch_api_key()

            # Set Anthropic async client
            if self.base_url:
                self.aclient = anthropic.AsyncAnthropic(api_key=self.api_key, base_url=self.base_url)
            else:
                self.aclient = anthropic.AsyncAnthropic(api_key=self.api_key)
        except Exception as e:
            raise Exception(f"Error setting {self.provider} async client: {e}")

    def get_client(self) -> anthropic.Anthropic:
        """Get the client using the API key from client manager.
        If client is not set, automatically try to set it by fetching the API keys."""
        return self.client if self.client else super().get_client()  # type: ignore

    def get_aclient(self) -> anthropic.AsyncAnthropic:
        """Get the async client using the API key from client manager.
        If async client is not set, automatically try to set it by fetching the API keys."""
        return self.aclient if self.aclient else super().get_aclient()  # type: ignore


# ================================================================================
# Tools to get information on Anthropic models
# ================================================================================
def get_model_info(model_id: str) -> dict[str, Any]:
    client = get_client_manager().get_client()
    try:
        model_info = client.models.retrieve(model_id=model_id)
        return model_info.model_dump()
    except Exception as e:
        raise Exception(f"Failed to fetch model info for '{model_id}': {e}")


def get_anthropic_models() -> list[dict[str, Any]]:
    client = get_client_manager().get_client()
    all_models = []
    for model in client.models.list():
        all_models.append({"model_path": model.id, "data": model.model_dump()})
    return all_models


# ================================================================================
# LINK Client setter and getters
# ================================================================================
# Module-level instance (singleton)
_global_client_manager: Dict[str, AnthropicClientManager] = {}


def get_client_manager(model_id: str = "", api_key: str = "", base_url: str | None = None) -> AnthropicClientManager:
    global _global_client_manager

    if not model_id:
        _global_client_manager["default"] = AnthropicClientManager(api_key=api_key, base_url=base_url)
        return _global_client_manager["default"]

    else:
        if model_id not in _global_client_manager:
            _global_client_manager[model_id] = AnthropicClientManager(model_id=model_id, base_url=base_url, api_key=api_key)
        return _global_client_manager[model_id]

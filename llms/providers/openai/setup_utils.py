from llms.providers.openai.constants import DEFAULT_OPENAI_MODE


def get_default_mode(provider: str) -> str:
    """Get the default mode for a given provider."""
    if provider.lower() == "openai":
        return DEFAULT_OPENAI_MODE
    # Add other providers as needed
    return "chat_completion"

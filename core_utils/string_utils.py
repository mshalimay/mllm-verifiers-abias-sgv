import re
from string import Formatter
from typing import Any


def extract_urls(text: str) -> list[str]:
    """Extract URLs from a given text"""
    try:
        url_pattern = r"(?:https?://|www\.|ftp://)[^\s\[\],;()\"'<>]+"

        matches = re.findall(url_pattern, text, re.IGNORECASE)
        return matches
    except Exception as e:
        print(f"Error extracting URLs from {text}: {e}")
        return []


def safe_format(string_template: str, fill_with: str = "", **kwargs: Any) -> str:
    """
    Formats a given template using the provided keyword arguments.
    Missing keys in the template are replaced with an empty string.

    Args:
        template (str): The string template with placeholders.
        **kwargs: Key-value pairs for formatting.

    Returns:
        str: The formatted string with missing keys as empty strings.
    """

    class DefaultDict(dict[Any, Any]):
        def __missing__(self, key: Any) -> Any:
            return fill_with

    return string_template.format_map(DefaultDict(**kwargs))


def clean_spaces(text: str) -> str:
    """Replace multiple newlines with a single newline and trim excess whitespace."""
    text = re.sub(r"\n{2,}", "\n", text)
    return re.sub(r"[ \t\r]+", " ", text).strip()


def partial_format(string_template: str, **kwargs) -> str:
    """
    Partially formats a string by replacing only those placeholders
    for which corresponding keyword arguments are provided.
    Placeholders with missing keys remain in the output.

    Example:
        template = "Hello, {name}! Today is {day}."
        result = partial_format(template, name="Alice")
        # result will be "Hello, Alice! Today is {day}."

    Args:
        string_template (str): The string template with placeholders.
        **kwargs: Keyword arguments with values for placeholders.

    Returns:
        str: A partially formatted string.
    """
    formatter = Formatter()
    result_str = ""

    for literal_text, field_name, format_spec, conversion in formatter.parse(string_template):
        # Add the literal text between placeholders
        result_str += literal_text

        # If there is no field, then nothing to format here.
        if field_name is None:
            continue

        if field_name in kwargs:
            value = kwargs[field_name]
            if conversion:
                if conversion == "r":
                    value = repr(value)
                elif conversion == "s":
                    value = str(value)
                elif conversion == "a":
                    value = ascii(value)
            try:
                format_spec = format_spec or ""
                formatted_value = format(value, format_spec)
            except Exception:
                # Fallback: if formatting fails, just convert to string.
                formatted_value = str(value)
            result_str += formatted_value
        else:
            # Key not provided. Reconstruct the original placeholder including
            # any conversion flags and format specifiers.
            placeholder = "{" + field_name
            if conversion:
                placeholder += "!" + conversion
            if format_spec:
                placeholder += ":" + format_spec
            placeholder += "}"
            result_str += placeholder
    return result_str

import re

from google.genai.errors import APIError

from core_utils.logger_utils import logger


class TestAPIError(Exception):
    """
    Exception for debugging. Mimics attributes of Google API errors.
    """

    def __init__(self, message: str, status: str = ""):
        super().__init__(message)
        self.message = message
        self.status = status
        self.details = {}


class PromptFeedbackError(Exception):
    def __init__(self, message: str, status: str = ""):
        super().__init__(message)
        self.message = message
        self.status = status


def parse_retry_delay(retry_delay: str) -> float:
    """
    Parses a retry delay string like '35s', '2m', '1h' into seconds (float).
    """
    try:
        if not retry_delay:
            return 0.0
        match = re.match(r"(?P<value>\d+)(?P<unit>[smh])", retry_delay)
        if not match:
            return 0.0
        value = int(match.group("value"))
        unit = match.group("unit")
        if unit == "s":
            return value
        elif unit == "m":
            return value * 60
        elif unit == "h":
            return value * 3600
        return 0.0
    except Exception as _:
        return 0.0


def parse_quota_error(e: APIError | TestAPIError) -> tuple[str, float | None]:
    """
    Parses the quota error from the APIError and returns a tuple:
    (quota_limit_type, retry_delay)
    """
    try:
        if not hasattr(e, "details"):
            return "", None

        if not isinstance(e.details, dict):
            return "", None

        details = e.details.get("error", {}).get("details", [])
        quota_limit = ""
        retry_delay = None
        for detail in details:
            detail_type = detail.get("@type", "")
            if detail_type == "type.googleapis.com/google.rpc.QuotaFailure":
                for violation in detail.get("violations", []):
                    quota_id = violation.get("quotaId", "")
                    logger.info(f"Quota violation: details: {details}")
                    logger.info(f"Quota violation: violation: {violation}")
                    if re.match(".*PerDay*.", quota_id, re.IGNORECASE):
                        quota_limit = "day"
                    elif re.match(".*PerMinute*.", quota_id, re.IGNORECASE):
                        quota_limit = "minute"
            elif detail_type == "type.googleapis.com/google.rpc.RetryInfo":
                retry_delay_str = detail.get("retryDelay", "")
                retry_delay = parse_retry_delay(retry_delay_str)
        return quota_limit, retry_delay
    except Exception as parse_error:
        logger.error(f"Error parsing quota error: {parse_error}")
        return "", None

class TestAPIError(Exception):
    """
    Exception for API errors that mimics the attributes of OpenAI API errors.
    """

    def __init__(self, message: str, status: str = ""):
        super().__init__(message)
        self.message = message
        self.status = status


class EmptyResponseError(Exception):
    """
    Exception for empty responses from the API.
    """

    def __init__(self, message: str):
        super().__init__(message)

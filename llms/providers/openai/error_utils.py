class TestAPIError(Exception):
    """
    Exception for API errors that mimics the attributes of OpenAI API errors.
    """

    def __init__(self, message: str, status: str = ""):
        super().__init__(message)
        self.message = message
        self.status = status

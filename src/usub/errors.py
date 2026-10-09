class UsubError(Exception):
    """A user-facing error (bad input, config or submission failure)."""


class SubmitError(UsubError):
    """The scheduler rejected the submission or its output could not be parsed."""

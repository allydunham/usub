"""
Errors supporting usub
"""

class UsubError(Exception):
    """A user-facing error (bad input, config or submission failure)."""

class SubmitError(UsubError):
    """The scheduler rejected the submission or its output could not be parsed."""

class UsubWarning(UserWarning):
    """Something is probably wrong but the job can still be generated, e.g. an unknown option."""

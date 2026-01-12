"""
Dummy scorer for testing purposes.
"""

from ..base import Scorer


class DummyScorer(Scorer):
    """
    A dummy scorer that returns a configurable constant value.

    Useful for testing the pipeline without actual scoring logic.
    """

    def __init__(self, config: dict = None):
        super().__init__(config)
        self.return_value = self.config.get("return_value", 0.0)

    def score(self, text: str) -> float:
        """Return the configured constant value."""
        return float(self.return_value)


# This class is used in tests to verify error handling
class NotAScorer:
    """Not a scorer - used for testing error handling."""
    pass

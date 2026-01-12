"""
Custom exceptions for counterfactual probing.

Provides a hierarchy of exceptions for better error handling and reporting.
"""


class CounterfactualProbingError(Exception):
    """Base exception for all counterfactual probing errors."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({detail_str})"
        return self.message


class ConfigurationError(CounterfactualProbingError):
    """Invalid or missing configuration."""

    pass


class DatasetError(CounterfactualProbingError):
    """Error loading or processing dataset."""

    pass


class ModelError(CounterfactualProbingError):
    """Error with model loading or inference."""

    pass


class GenerationError(CounterfactualProbingError):
    """Error during text generation."""

    pass


class ScorerError(CounterfactualProbingError):
    """Error in scorer execution."""

    pass


class ExtractionError(CounterfactualProbingError):
    """Error during activation extraction."""

    pass


class CheckpointError(CounterfactualProbingError):
    """Error with checkpoint save/load."""

    pass


class ValidationError(CounterfactualProbingError):
    """Data validation failed."""

    pass


# Retry-able errors (for automatic retry logic)
class TransientError(CounterfactualProbingError):
    """Transient error that may succeed on retry."""

    def __init__(self, message: str, retry_after: float | None = None, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class GPUMemoryError(TransientError):
    """GPU out of memory - may succeed with smaller batch."""

    pass


class RateLimitError(TransientError):
    """Rate limited - retry after delay."""

    pass

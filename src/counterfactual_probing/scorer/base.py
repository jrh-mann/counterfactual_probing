"""
Base class for scorers and scorer loading utilities.
"""

import importlib
from abc import ABC, abstractmethod
from typing import List, Optional


class Scorer(ABC):
    """
    Abstract base class for scoring rollout completions.

    Users should subclass this and implement the score() method.
    The score_batch() method can optionally be overridden for efficiency.
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize the scorer.

        Args:
            config: Optional configuration dictionary passed from config file
        """
        self.config = config or {}

    @abstractmethod
    def score(self, text: str) -> float:
        """
        Score a completion text.

        Args:
            text: The generated completion text

        Returns:
            Float in [0, 1] indicating probability/confidence of target behavior.
            - 0.0 means the behavior is definitely not present
            - 1.0 means the behavior is definitely present
            - Values in between indicate uncertainty or partial presence
        """
        pass

    def score_batch(self, texts: List[str]) -> List[float]:
        """
        Score multiple texts.

        Override this method for more efficient batch processing.
        Default implementation calls score() for each text.

        Args:
            texts: List of completion texts to score

        Returns:
            List of scores, one per input text
        """
        return [self.score(t) for t in texts]


def load_scorer(scorer_config: dict) -> Scorer:
    """
    Dynamically load a scorer from a module path.

    Args:
        scorer_config: Dictionary containing:
            - module: Python module path (e.g., "mypackage.scorers")
            - class: Class name within the module
            - config: Optional config dict to pass to scorer __init__

    Returns:
        Instantiated Scorer object

    Raises:
        ImportError: If module cannot be imported
        AttributeError: If class doesn't exist in module
        TypeError: If class is not a Scorer subclass
    """
    module_path = scorer_config.get("module")
    class_name = scorer_config.get("class") or scorer_config.get("class_name")
    config = scorer_config.get("config", {})

    if not module_path:
        raise ValueError("Scorer config must include 'module' field")
    if not class_name:
        raise ValueError("Scorer config must include 'class' field")

    # Import the module
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"Could not import scorer module '{module_path}': {e}"
        ) from e

    # Get the class
    if not hasattr(module, class_name):
        raise AttributeError(
            f"Module '{module_path}' does not have class '{class_name}'"
        )

    scorer_class = getattr(module, class_name)

    # Verify it's a Scorer subclass
    if not isinstance(scorer_class, type) or not issubclass(scorer_class, Scorer):
        raise TypeError(
            f"Class '{class_name}' in module '{module_path}' is not a Scorer subclass"
        )

    # Instantiate and return
    return scorer_class(config=config)

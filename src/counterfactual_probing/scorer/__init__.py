"""
Scorer plugin system for evaluating generated completions.

Users can create custom scorers by subclassing the Scorer base class.
"""

from .base import Scorer, load_scorer


__all__ = ["Scorer", "load_scorer"]

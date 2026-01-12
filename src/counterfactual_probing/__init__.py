"""Counterfactual probing library for language models."""

__version__ = "0.1.0"

from .config import Config, create_default_config, load_config
from .dataset import Dataset
from .model_utils import ensure_experiment_dirs, get_experiment_paths, get_model_slug
from .run import run, run_from_config
from .sampler import TokenSampler
from .scorer import Scorer, load_scorer


__all__ = [
    "Config",
    "Dataset",
    "Scorer",
    "TokenSampler",
    "create_default_config",
    "ensure_experiment_dirs",
    "get_experiment_paths",
    "get_model_slug",
    "load_config",
    "load_scorer",
    "run",
    "run_from_config",
]

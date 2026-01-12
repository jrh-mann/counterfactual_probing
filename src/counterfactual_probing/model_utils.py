"""
Model utilities for handling model names and paths.

Provides consistent model slug derivation for organizing outputs by model.
"""

import re
from pathlib import Path


def get_model_slug(model_name: str) -> str:
    """
    Derive a filesystem-safe slug from a model name.

    Examples:
        "Qwen/Qwen3-0.6B" -> "qwen3-0.6b"
        "Qwen/Qwen3-4B" -> "qwen3-4b"
        "meta-llama/Llama-3.1-8B-Instruct" -> "llama-3.1-8b-instruct"
        "mistralai/Mistral-7B-v0.1" -> "mistral-7b-v0.1"
        "gpt2" -> "gpt2"

    Args:
        model_name: Full model name (e.g., "Qwen/Qwen3-4B")

    Returns:
        Lowercase slug suitable for directory names
    """
    # Take the part after the last "/" if present
    if "/" in model_name:
        slug = model_name.split("/")[-1]
    else:
        slug = model_name

    # Lowercase
    slug = slug.lower()

    # Replace spaces and underscores with hyphens
    slug = re.sub(r"[\s_]+", "-", slug)

    # Remove any non-alphanumeric characters except hyphens and dots
    slug = re.sub(r"[^a-z0-9\-\.]", "", slug)

    # Collapse multiple hyphens
    slug = re.sub(r"-+", "-", slug)

    # Strip leading/trailing hyphens
    slug = slug.strip("-")

    return slug


def get_model_output_dir(base_dir: str, model_name: str) -> Path:
    """
    Get the model-specific output directory.

    Args:
        base_dir: Base output directory (e.g., "outputs")
        model_name: Model name (e.g., "Qwen/Qwen3-4B")

    Returns:
        Path like "outputs/qwen3-4b/"
    """
    slug = get_model_slug(model_name)
    return Path(base_dir) / slug


def get_experiment_paths(
    model_name: str,
    experiment_name: str = "default",
    base_outputs: str = "outputs",
    base_activations: str = "activations",
    base_plots: str = "plots",
    base_probes: str = "probes",
) -> dict:
    """
    Get all experiment-related paths for a model.

    Args:
        model_name: Model name (e.g., "Qwen/Qwen3-4B")
        experiment_name: Experiment name (e.g., "math", "reward_hacking")
        base_*: Base directories

    Returns:
        Dict with all relevant paths:
        - outputs_dir: Where counterfactual outputs are saved
        - activations_dir: Where activations are saved
        - plots_dir: Where visualizations are saved
        - probes_dir: Where trained probes are saved
        - model_slug: The derived model slug
    """
    slug = get_model_slug(model_name)

    return {
        "model_slug": slug,
        "outputs_dir": Path(base_outputs) / slug / experiment_name,
        "activations_dir": Path(base_activations) / slug / experiment_name,
        "plots_dir": Path(base_plots) / slug / experiment_name,
        "probes_dir": Path(base_probes) / slug / experiment_name,
    }


def ensure_experiment_dirs(paths: dict) -> None:
    """
    Ensure all experiment directories exist.

    Args:
        paths: Dict from get_experiment_paths()
    """
    for key in ["outputs_dir", "activations_dir", "plots_dir", "probes_dir"]:
        if key in paths:
            paths[key].mkdir(parents=True, exist_ok=True)

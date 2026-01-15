"""
Configuration loading and validation using Pydantic.

Provides typed configuration with sensible defaults and clear error messages.
"""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .model_utils import get_experiment_paths, get_model_slug


class DatasetConfig(BaseModel):
    """Dataset configuration."""

    path: str = Field(..., description="Path to dataset file")
    prompt_field: str = Field(default="prompt", description="Field name containing the prompt")
    format: Literal["jsonl", "json", "csv"] = Field(
        default="jsonl",
        description="Dataset file format"
    )


class ModelConfig(BaseModel):
    """Model configuration."""

    name: str = Field(..., description="Model name or path")
    tensor_parallel_size: int = Field(default=1, ge=1, description="Tensor parallel size for vLLM")
    gpu_memory_utilization: float = Field(
        default=0.9,
        ge=0.1,
        le=1.0,
        description="GPU memory utilization fraction"
    )


class GenerationConfig(BaseModel):
    """Generation parameters."""

    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(default=4096, ge=1, description="Maximum tokens to generate")
    max_tokens_retry: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Max tokens for retry attempts on truncated rollouts. "
            "If None, uses max_tokens * 1.5. Set higher to handle long CoT."
        )
    )
    num_counterfactuals: int = Field(
        default=50,
        ge=1,
        description="Number of counterfactual continuations per sample point"
    )
    num_initial_rollouts: int = Field(
        default=3,
        ge=1,
        description=(
            "Number of initial rollouts to generate per prompt. "
            "The shortest valid rollout (has </think>, not truncated) is selected."
        )
    )

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v):
        if v < 0 or v > 2:
            raise ValueError("temperature must be between 0 and 2")
        return v

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v):
        if v < 1:
            raise ValueError("max_tokens must be positive")
        return v

    @field_validator("num_counterfactuals")
    @classmethod
    def validate_num_counterfactuals(cls, v):
        if v < 1:
            raise ValueError("num_counterfactuals must be positive")
        return v


class SamplingConfig(BaseModel):
    """Sampling configuration for selecting branch points."""

    method: Literal["random", "uniform_count", "density"] = Field(
        default="random",
        description=(
            "Sampling method: 'random' (pure random, recommended), "
            "'uniform_count' (deprecated, forces endpoints), "
            "'density' (deprecated, forces endpoints)"
        )
    )
    num_samples: int = Field(
        default=20,
        ge=1,
        description="Number of sample points for random/uniform_count methods"
    )
    density: float = Field(
        default=0.02,
        gt=0,
        le=1.0,
        description="Sampling density for density method"
    )
    seed: int | None = Field(
        default=None,
        description="Random seed for reproducibility"
    )
    cot_boundary_token: int | None = Field(
        default=None,
        description=(
            "Token ID marking end of chain-of-thought (e.g., </think> token). "
            "If set, sampling will only occur before this token. "
            "For Qwen3 models, use 151668 for the </think> token."
        )
    )

    @field_validator("num_samples")
    @classmethod
    def validate_num_samples(cls, v):
        if v < 1:
            raise ValueError("num_samples must be positive")
        return v

    @field_validator("density")
    @classmethod
    def validate_density(cls, v):
        if v <= 0 or v > 1:
            raise ValueError("density must be between 0 and 1 (exclusive of 0)")
        return v


class ScorerConfig(BaseModel):
    """Scorer plugin configuration."""

    module: str = Field(..., description="Python module path containing the scorer class")
    class_name: str = Field(..., alias="class", description="Name of the scorer class")
    config: dict = Field(default_factory=dict, description="Configuration passed to scorer __init__")

    class Config:
        populate_by_name = True


class OutputConfig(BaseModel):
    """Output configuration."""

    dir: str = Field(default="outputs/", description="Output directory for results")


class Config(BaseModel):
    """Main configuration container."""

    dataset: DatasetConfig
    model: ModelConfig
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    scorer: ScorerConfig | None = None
    output: OutputConfig = Field(default_factory=OutputConfig)
    skip_existing: bool = Field(
        default=True,
        description="Skip prompts with existing output files"
    )

    @property
    def model_slug(self) -> str:
        """Get the derived slug for the current model."""
        return get_model_slug(self.model.name)

    def get_experiment_paths(self, experiment_name: str = "default") -> dict:
        """
        Get all paths for this model/experiment combination.

        Args:
            experiment_name: Name of the experiment (e.g., "math")

        Returns:
            Dict with outputs_dir, activations_dir, plots_dir, probes_dir, model_slug
        """
        return get_experiment_paths(
            model_name=self.model.name,
            experiment_name=experiment_name,
        )


def load_config(config_path: str) -> Config:
    """
    Load and validate configuration from a JSON file.

    Args:
        config_path: Path to the configuration JSON file

    Returns:
        Validated Config object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid JSON or fails validation
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(path) as f:
            raw_config = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file: {e}") from e

    try:
        return Config(**raw_config)
    except Exception as e:
        # Re-raise with more context
        raise ValueError(f"Configuration validation error: {e}") from e


def create_default_config(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    experiment_name: str = "default",
) -> dict:
    """
    Create a default configuration dictionary.

    Args:
        model_name: Model to use (determines output paths)
        experiment_name: Experiment name for organizing outputs

    Returns:
        Dictionary with default configuration values
    """
    paths = get_experiment_paths(model_name, experiment_name)

    return {
        "dataset": {
            "path": "data/prompts.jsonl",
            "prompt_field": "prompt",
            "format": "jsonl",
        },
        "model": {
            "name": model_name,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.9,
        },
        "generation": {
            "temperature": 0.7,
            "max_tokens": 4096,
            "num_counterfactuals": 50,
            "num_initial_rollouts": 1,
        },
        "sampling": {
            "method": "random",
            "num_samples": 20,
            "density": 0.02,
            "cot_boundary_token": None,
        },
        "scorer": {
            "module": "counterfactual_probing.scorer.examples.dummy",
            "class": "DummyScorer",
            "config": {},
        },
        "output": {
            "dir": str(paths["outputs_dir"]),
        },
        "skip_existing": True,
    }

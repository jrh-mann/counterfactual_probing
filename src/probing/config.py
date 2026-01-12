"""
Configuration dataclasses for probe training.

Designed for extensibility - add new fields as needed,
old configs will use defaults for new fields.
"""

from dataclasses import dataclass, field
from typing import Optional, Union, List, Tuple, Any, Dict


@dataclass
class MetricConfig:
    """Configuration for which metrics to compute and how."""

    # Thresholds for binarizing continuous p_score labels
    # AUC and accuracy computed at each threshold
    binary_thresholds: Tuple[float, ...] = (0.3, 0.5, 0.7)

    # Primary threshold for single-number metrics
    primary_threshold: float = 0.5

    # Which metrics to compute
    include_accuracy: bool = True
    include_auc: bool = True
    include_loss: bool = True
    include_mse: bool = True      # Mean squared error (for regression)
    include_mae: bool = True      # Mean absolute error
    include_f1: bool = True
    include_precision_recall: bool = True

    # Future extensibility
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProbeConfig:
    """
    Full configuration for a probe training run.

    Designed for grid search - all fields that might be swept
    should be represented here.
    """

    # === Layer Selection ===
    # int: single layer index
    # List[int]: concatenate these layers
    # "all": concatenate all layers
    # "every_n:k": every k-th layer (e.g., "every_n:4" for layers 0,4,8,...)
    layer: Union[int, List[int], str] = -1  # -1 = last layer

    # === Label Configuration ===
    # 'sparse': use only measured branch points
    # 'interpolated': interpolate between branch points
    # 'sequence': use sequence-level label (final correctness)
    label_mode: str = 'sparse'

    # Interpolation method (if label_mode='interpolated')
    # 'linear', 'cubic', 'nearest', 'zero' (step function)
    interpolation_method: str = 'linear'

    # === Smoothing ===
    # None: no smoothing
    # 'swim': sliding window mean on logits
    # 'ema': exponential moving average
    smoothing: Optional[str] = None
    window_size: int = 5          # for SWiM
    ema_alpha: float = 0.1        # for EMA

    # === Probe Method ===
    # 'logistic': LogisticRegression (classification)
    # 'ridge': Ridge regression (continuous)
    # Extensible: add new methods to PROBE_METHODS registry
    method: str = 'logistic'

    # Regularization strength (C for logistic, alpha for ridge)
    regularization: float = 1.0

    # Max iterations for optimization
    max_iter: int = 1000

    # === Label Processing ===
    # Threshold for binarizing p_score (None = use continuous labels)
    label_threshold: Optional[float] = 0.5

    # === Preprocessing ===
    normalize: bool = True        # StandardScaler on activations

    # === Reproducibility ===
    seed: int = 42

    # === Metadata ===
    name: Optional[str] = None    # Optional name for this config
    tags: Tuple[str, ...] = ()    # Tags for filtering results

    def __post_init__(self):
        """Validate configuration."""
        valid_label_modes = ('sparse', 'interpolated', 'sequence')
        if self.label_mode not in valid_label_modes:
            raise ValueError(f"label_mode must be one of {valid_label_modes}")

        valid_smoothing = (None, 'swim', 'ema')
        if self.smoothing not in valid_smoothing:
            raise ValueError(f"smoothing must be one of {valid_smoothing}")

        valid_methods = ('logistic', 'ridge')  # Extend as needed
        if self.method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")

        valid_interp = ('linear', 'cubic', 'nearest', 'zero')
        if self.interpolation_method not in valid_interp:
            raise ValueError(f"interpolation_method must be one of {valid_interp}")

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'layer': self.layer,
            'label_mode': self.label_mode,
            'interpolation_method': self.interpolation_method,
            'smoothing': self.smoothing,
            'window_size': self.window_size,
            'ema_alpha': self.ema_alpha,
            'method': self.method,
            'regularization': self.regularization,
            'max_iter': self.max_iter,
            'label_threshold': self.label_threshold,
            'normalize': self.normalize,
            'seed': self.seed,
            'name': self.name,
            'tags': self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'ProbeConfig':
        """Create from dictionary."""
        return cls(**d)

    def short_description(self) -> str:
        """Short string describing this config for logging."""
        layer_str = str(self.layer) if isinstance(self.layer, int) else f"multi({len(self.layer)})" if isinstance(self.layer, list) else self.layer
        parts = [
            f"L{layer_str}",
            self.label_mode[:3],
            self.method[:3],
        ]
        if self.smoothing:
            parts.append(f"{self.smoothing}{self.window_size}")
        return "_".join(parts)

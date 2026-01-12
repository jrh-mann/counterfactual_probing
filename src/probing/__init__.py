"""
Probe training framework for counterfactual probing analysis.

Supports ablations across:
- Layer selection (single, multi-layer, all)
- Label modes (sparse, interpolated, sequence-level)
- Smoothing (none, SWiM, EMA)
- Probe methods (logistic, ridge, extensible)
- Metrics (accuracy, AUC at multiple thresholds, loss, MSE)
"""

from .config import ProbeConfig, MetricConfig
from .result import ProbeResult
from .trainer import ProbeTrainer
from .ablation import AblationRunner

__all__ = [
    'ProbeConfig',
    'MetricConfig',
    'ProbeResult',
    'ProbeTrainer',
    'AblationRunner',
]

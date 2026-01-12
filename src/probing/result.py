"""
ProbeResult dataclass for storing probe training results.

Stores everything needed to:
- Reproduce the experiment
- Analyze the results
- Use the probe for inference
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import numpy as np
import json
from pathlib import Path


@dataclass
class ProbeResult:
    """
    Complete result of a probe training run.

    Stores the trained probe, all metrics, and full configuration
    for reproducibility.
    """

    # === Probe Parameters ===
    direction: np.ndarray           # (hidden_dim,) or (concat_dim,) for multi-layer
    bias: float                     # Intercept term

    # === Configuration ===
    config: Dict[str, Any]          # Full ProbeConfig as dict

    # === Metrics at Primary Threshold ===
    train_accuracy: float
    test_accuracy: float
    train_loss: float
    test_loss: float

    # === Metrics at Multiple Thresholds ===
    # Dict[threshold, Dict[metric_name, value]]
    threshold_metrics: Dict[float, Dict[str, float]] = field(default_factory=dict)

    # === Regression Metrics (if applicable) ===
    train_mse: Optional[float] = None
    test_mse: Optional[float] = None
    train_mae: Optional[float] = None
    test_mae: Optional[float] = None

    # === AUC (threshold-independent for binary ground truth) ===
    # But we have continuous labels, so AUC computed per binarization threshold
    auc_by_threshold: Dict[float, float] = field(default_factory=dict)

    # === Additional Info ===
    n_train: int = 0
    n_test: int = 0
    layer_info: str = ""            # Description of layers used
    feature_dim: int = 0            # Dimension of input features

    # === Predictions (optional, for analysis) ===
    train_predictions: Optional[np.ndarray] = None
    test_predictions: Optional[np.ndarray] = None
    train_labels: Optional[np.ndarray] = None
    test_labels: Optional[np.ndarray] = None

    def predict(self, activations: np.ndarray) -> np.ndarray:
        """
        Apply probe to activations.

        Args:
            activations: (n_samples, hidden_dim) array

        Returns:
            (n_samples,) array of predictions (logits or probabilities depending on method)
        """
        return activations @ self.direction + self.bias

    def predict_proba(self, activations: np.ndarray) -> np.ndarray:
        """
        Get probability predictions (sigmoid of logits).

        Args:
            activations: (n_samples, hidden_dim) array

        Returns:
            (n_samples,) array of probabilities
        """
        logits = self.predict(activations)
        return 1 / (1 + np.exp(-logits))

    def summary(self) -> str:
        """Human-readable summary of results."""
        lines = [
            f"=== Probe Result ===",
            f"Layer: {self.layer_info}",
            f"Method: {self.config.get('method', 'unknown')}",
            f"Label mode: {self.config.get('label_mode', 'unknown')}",
            f"Smoothing: {self.config.get('smoothing', 'none')}",
            f"",
            f"Train samples: {self.n_train}, Test samples: {self.n_test}",
            f"Feature dim: {self.feature_dim}",
            f"",
            f"=== Metrics (threshold={self.config.get('label_threshold', 0.5)}) ===",
            f"Train accuracy: {self.train_accuracy:.4f}",
            f"Test accuracy:  {self.test_accuracy:.4f}",
            f"Train loss:     {self.train_loss:.4f}",
            f"Test loss:      {self.test_loss:.4f}",
        ]

        if self.train_mse is not None:
            lines.extend([
                f"",
                f"=== Regression Metrics ===",
                f"Train MSE: {self.train_mse:.4f}",
                f"Test MSE:  {self.test_mse:.4f}",
                f"Train MAE: {self.train_mae:.4f}",
                f"Test MAE:  {self.test_mae:.4f}",
            ])

        if self.auc_by_threshold:
            lines.append(f"")
            lines.append(f"=== AUC by Threshold ===")
            for thresh, auc in sorted(self.auc_by_threshold.items()):
                lines.append(f"  Threshold {thresh}: AUC = {auc:.4f}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            'direction': self.direction.tolist(),
            'bias': float(self.bias),
            'config': self.config,
            'train_accuracy': self.train_accuracy,
            'test_accuracy': self.test_accuracy,
            'train_loss': self.train_loss,
            'test_loss': self.test_loss,
            'threshold_metrics': self.threshold_metrics,
            'train_mse': self.train_mse,
            'test_mse': self.test_mse,
            'train_mae': self.train_mae,
            'test_mae': self.test_mae,
            'auc_by_threshold': self.auc_by_threshold,
            'n_train': self.n_train,
            'n_test': self.n_test,
            'layer_info': self.layer_info,
            'feature_dim': self.feature_dim,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ProbeResult':
        """Create from dictionary."""
        d = d.copy()
        d['direction'] = np.array(d['direction'])
        return cls(**d)

    def save(self, path: str) -> None:
        """Save to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'ProbeResult':
        """Load from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def metrics_row(self) -> Dict[str, Any]:
        """
        Flat dictionary of metrics for DataFrame row.

        Useful for aggregating results across experiments.
        """
        row = {
            'layer': self.layer_info,
            'method': self.config.get('method'),
            'label_mode': self.config.get('label_mode'),
            'smoothing': self.config.get('smoothing'),
            'window_size': self.config.get('window_size'),
            'label_threshold': self.config.get('label_threshold'),
            'regularization': self.config.get('regularization'),
            'train_acc': self.train_accuracy,
            'test_acc': self.test_accuracy,
            'train_loss': self.train_loss,
            'test_loss': self.test_loss,
            'train_mse': self.train_mse,
            'test_mse': self.test_mse,
            'n_train': self.n_train,
            'n_test': self.n_test,
            'feature_dim': self.feature_dim,
        }

        # Add AUC at each threshold
        for thresh, auc in self.auc_by_threshold.items():
            row[f'auc_thresh_{thresh}'] = auc

        return row

"""
Softmax-weighted linear probe from Constitutional Classifiers++ (Cunningham et al., 2026).

The key insight: when predicting sequence-level outcomes from token-level features,
weight the loss by softmax of predictions to focus on positions where the model is confident.

Loss = Σ w_t * BCE(z_t, y)
where w_t = exp(z_t / τ) / Σ exp(z_t' / τ)

This directs gradients toward samples where the probe makes confident predictions,
which are most informative for learning the classification boundary.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional
from dataclasses import dataclass


@dataclass
class SoftmaxWeightedConfig:
    """Configuration for softmax-weighted probe training."""
    temperature: float = 1.0  # τ for softmax weighting
    learning_rate: float = 0.01
    max_iter: int = 1000
    tol: float = 1e-6
    weight_decay: float = 0.0  # L2 regularization
    seed: int = 42


class SoftmaxWeightedProbe(nn.Module):
    """
    Linear probe trained with softmax-weighted BCE loss.

    For each group of samples (e.g., branch points from the same prompt),
    the loss weights are computed via softmax over the logits.
    """

    def __init__(self, input_dim: int, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits."""
        return self.linear(x).squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return probabilities."""
        return torch.sigmoid(self.forward(x))


def softmax_weighted_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    group_ids: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute softmax-weighted BCE loss.

    Args:
        logits: (n_samples,) model predictions (before sigmoid)
        targets: (n_samples,) binary labels
        group_ids: (n_samples,) integer group identifiers
        temperature: softmax temperature τ

    Returns:
        Scalar loss value
    """
    # Per-sample BCE (unreduced)
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction='none'
    )

    # Compute softmax weights within each group
    unique_groups = torch.unique(group_ids)
    weighted_loss = torch.tensor(0.0, device=logits.device)

    for group in unique_groups:
        mask = group_ids == group
        group_logits = logits[mask]
        group_bce = bce[mask]

        # Softmax weights based on logit magnitude
        # Higher logits (more confident positive) get more weight
        weights = torch.softmax(group_logits / temperature, dim=0)

        # Weighted sum of BCE within this group
        group_loss = (weights * group_bce).sum()
        weighted_loss = weighted_loss + group_loss

    # Average across groups
    return weighted_loss / len(unique_groups)


def train_softmax_weighted_probe(
    X: np.ndarray,
    y: np.ndarray,
    group_ids: np.ndarray,
    config: Optional[SoftmaxWeightedConfig] = None,
    verbose: bool = False,
) -> SoftmaxWeightedProbe:
    """
    Train a softmax-weighted linear probe.

    Args:
        X: (n_samples, n_features) feature matrix
        y: (n_samples,) binary labels
        group_ids: (n_samples,) group identifiers (e.g., prompt indices)
        config: Training configuration
        verbose: Print training progress

    Returns:
        Trained probe model
    """
    if config is None:
        config = SoftmaxWeightedConfig()

    # Convert to tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)
    groups_t = torch.tensor(group_ids, dtype=torch.long, device=device)

    # Initialize model
    model = SoftmaxWeightedProbe(X.shape[1], seed=config.seed).to(device)
    optimizer = optim.LBFGS(
        model.parameters(),
        lr=config.learning_rate,
        max_iter=20,
        tolerance_grad=config.tol,
        tolerance_change=config.tol,
    )

    prev_loss = float('inf')

    for iteration in range(config.max_iter // 20):  # LBFGS does 20 iters per step
        def closure():
            optimizer.zero_grad()
            logits = model(X_t)
            loss = softmax_weighted_bce_loss(
                logits, y_t, groups_t, config.temperature
            )
            if config.weight_decay > 0:
                l2_reg = sum(p.pow(2).sum() for p in model.parameters())
                loss = loss + config.weight_decay * l2_reg
            loss.backward()
            return loss

        loss = optimizer.step(closure)

        if verbose and iteration % 5 == 0:
            print(f"  Iter {iteration * 20}: loss = {loss.item():.6f}")

        # Check convergence
        if abs(prev_loss - loss.item()) < config.tol:
            if verbose:
                print(f"  Converged at iteration {iteration * 20}")
            break
        prev_loss = loss.item()

    model.eval()
    return model


class SoftmaxWeightedWrapper:
    """
    Sklearn-compatible wrapper for SoftmaxWeightedProbe.

    Implements fit(), predict(), predict_proba() for compatibility
    with the ProbeTrainer infrastructure.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        weight_decay: float = 0.01,
        seed: int = 42,
    ):
        self.config = SoftmaxWeightedConfig(
            temperature=temperature,
            learning_rate=learning_rate,
            max_iter=max_iter,
            weight_decay=weight_decay,
            seed=seed,
        )
        self.model: Optional[SoftmaxWeightedProbe] = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._group_ids: Optional[np.ndarray] = None

    def set_group_ids(self, group_ids: np.ndarray):
        """Set group IDs for training. Must be called before fit()."""
        self._group_ids = group_ids

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the probe.

        Note: group_ids must be set via set_group_ids() before calling fit().
        If not set, treats each sample as its own group (equivalent to standard BCE).
        """
        if self._group_ids is None:
            # Fall back to treating each sample as its own group
            self._group_ids = np.arange(len(y))

        self.model = train_softmax_weighted_probe(
            X, y, self._group_ids, self.config, verbose=False
        )

        # Store coef_ and intercept_ for compatibility
        with torch.no_grad():
            self.coef_ = self.model.linear.weight.cpu().numpy()
            self.intercept_ = self.model.linear.bias.cpu().numpy()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            proba = self.model.predict_proba(X_t).cpu().numpy()

        # Return in sklearn format: (n_samples, 2) for binary classification
        return np.column_stack([1 - proba, proba])

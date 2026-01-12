"""
Smoothing utilities for probe predictions.

Implements SWiM (Sliding Window Mean) and EMA (Exponential Moving Average)
following the Anthropic methodology.
"""

import numpy as np
from typing import Optional


def swim_smooth(
    values: np.ndarray,
    window_size: int,
    axis: int = -1,
) -> np.ndarray:
    """
    Sliding Window Mean smoothing.

    For each position t, computes the mean of values in [t-window_size+1, t].
    First window_size-1 positions use available values.

    Args:
        values: Array to smooth (any shape)
        window_size: Size of sliding window
        axis: Axis along which to smooth

    Returns:
        Smoothed array of same shape
    """
    if window_size <= 1:
        return values.copy()

    # Move target axis to end for easier processing
    values = np.moveaxis(values, axis, -1)
    original_shape = values.shape
    length = values.shape[-1]

    # Flatten all but last axis
    flat_values = values.reshape(-1, length)
    n_sequences = flat_values.shape[0]

    # Compute cumulative sum for efficient windowed mean
    smoothed = np.zeros_like(flat_values)

    for i in range(n_sequences):
        cumsum = np.cumsum(flat_values[i])
        # Insert 0 at beginning for window calculation
        cumsum = np.insert(cumsum, 0, 0)

        for t in range(length):
            start = max(0, t - window_size + 1)
            # Mean of values from start to t (inclusive)
            window_sum = cumsum[t + 1] - cumsum[start]
            window_len = t - start + 1
            smoothed[i, t] = window_sum / window_len

    # Restore original shape
    smoothed = smoothed.reshape(original_shape)
    smoothed = np.moveaxis(smoothed, -1, axis)

    return smoothed


def ema_smooth(
    values: np.ndarray,
    alpha: float = 0.1,
    axis: int = -1,
) -> np.ndarray:
    """
    Exponential Moving Average smoothing.

    EMA_t = alpha * value_t + (1 - alpha) * EMA_{t-1}

    More memory-efficient than SWiM for streaming inference.

    Args:
        values: Array to smooth
        alpha: Smoothing factor (0 < alpha <= 1). Higher = less smoothing.
        axis: Axis along which to smooth

    Returns:
        Smoothed array of same shape
    """
    if alpha >= 1.0:
        return values.copy()

    # Move target axis to end
    values = np.moveaxis(values, axis, -1)
    original_shape = values.shape
    length = values.shape[-1]

    # Flatten all but last axis
    flat_values = values.reshape(-1, length)
    smoothed = np.zeros_like(flat_values)

    # Initialize EMA with first value
    smoothed[:, 0] = flat_values[:, 0]

    # Apply EMA
    for t in range(1, length):
        smoothed[:, t] = alpha * flat_values[:, t] + (1 - alpha) * smoothed[:, t - 1]

    # Restore shape
    smoothed = smoothed.reshape(original_shape)
    smoothed = np.moveaxis(smoothed, -1, axis)

    return smoothed


def smooth_logits(
    logits: np.ndarray,
    method: Optional[str],
    window_size: int = 5,
    ema_alpha: float = 0.1,
) -> np.ndarray:
    """
    Apply smoothing to logits.

    Convenience function that dispatches to swim_smooth or ema_smooth.

    Args:
        logits: Logit values to smooth
        method: 'swim', 'ema', or None (no smoothing)
        window_size: Window size for SWiM
        ema_alpha: Alpha for EMA

    Returns:
        Smoothed logits
    """
    if method is None:
        return logits.copy()
    elif method == 'swim':
        return swim_smooth(logits, window_size)
    elif method == 'ema':
        return ema_smooth(logits, ema_alpha)
    else:
        raise ValueError(f"Unknown smoothing method: {method}")


def smooth_activations(
    activations: np.ndarray,
    method: Optional[str],
    window_size: int = 5,
    ema_alpha: float = 0.1,
    token_axis: int = 1,
) -> np.ndarray:
    """
    Apply smoothing to activations across token positions.

    This smooths the activation vectors themselves, creating
    "context-aware" representations that incorporate nearby tokens.

    Args:
        activations: (n_samples, n_tokens, hidden_dim) or similar
        method: 'swim', 'ema', or None
        window_size: Window size for SWiM
        ema_alpha: Alpha for EMA
        token_axis: Which axis represents token positions

    Returns:
        Smoothed activations of same shape
    """
    if method is None:
        return activations.copy()

    return smooth_logits(activations, method, window_size, ema_alpha)

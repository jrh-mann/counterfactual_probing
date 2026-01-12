"""
Label interpolation utilities for dense probe training.

Converts sparse branch-point labels to dense per-token labels
via interpolation.
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy import interpolate


def interpolate_labels(
    branch_points: List[int],
    p_scores: List[float],
    total_length: int,
    method: str = 'linear',
    fill_value: str = 'extrapolate',
) -> np.ndarray:
    """
    Interpolate p_scores from sparse branch points to all positions.

    Args:
        branch_points: Token indices where p_score was measured
        p_scores: P_score values at each branch point
        total_length: Total sequence length
        method: Interpolation method ('linear', 'cubic', 'nearest', 'zero')
        fill_value: How to handle extrapolation ('extrapolate' or float)

    Returns:
        Array of shape (total_length,) with interpolated p_scores
    """
    branch_points = np.array(branch_points)
    p_scores = np.array(p_scores)

    # Sort by position
    sort_idx = np.argsort(branch_points)
    branch_points = branch_points[sort_idx]
    p_scores = p_scores[sort_idx]

    # Handle edge cases
    if len(branch_points) == 0:
        return np.full(total_length, 0.5)  # Unknown, return neutral

    if len(branch_points) == 1:
        return np.full(total_length, p_scores[0])

    # Create interpolator
    if method == 'zero':
        # Step function: hold previous value until next point
        kind = 'previous'
    else:
        kind = method

    # Build interpolation function
    if fill_value == 'extrapolate':
        f = interpolate.interp1d(
            branch_points, p_scores,
            kind=kind,
            bounds_error=False,
            fill_value=(p_scores[0], p_scores[-1])  # Extend edges
        )
    else:
        f = interpolate.interp1d(
            branch_points, p_scores,
            kind=kind,
            bounds_error=False,
            fill_value=fill_value
        )

    # Interpolate at all positions
    all_positions = np.arange(total_length)
    interpolated = f(all_positions)

    # Clip to valid range
    interpolated = np.clip(interpolated, 0.0, 1.0)

    return interpolated


def interpolate_with_uncertainty(
    branch_points: List[int],
    p_scores: List[float],
    total_length: int,
    method: str = 'linear',
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate with uncertainty estimates.

    Uncertainty is higher further from measured points.

    Args:
        branch_points: Token indices where p_score was measured
        p_scores: P_score values at each branch point
        total_length: Total sequence length
        method: Interpolation method

    Returns:
        Tuple of (interpolated_values, uncertainty) arrays
    """
    interpolated = interpolate_labels(branch_points, p_scores, total_length, method)

    # Compute distance to nearest measured point
    all_positions = np.arange(total_length)
    branch_points = np.array(branch_points)

    distances = np.min(np.abs(all_positions[:, None] - branch_points[None, :]), axis=1)

    # Normalize uncertainty (0 at measured points, higher further away)
    max_distance = np.max(distances) if len(distances) > 0 else 1
    uncertainty = distances / max_distance if max_distance > 0 else distances

    return interpolated, uncertainty


def get_interpolation_weights(
    branch_points: List[int],
    total_length: int,
    decay: str = 'linear',
    decay_rate: float = 0.1,
) -> np.ndarray:
    """
    Compute weights for each position based on distance to measurements.

    Useful for weighted loss functions that trust measured points more.

    Args:
        branch_points: Token indices where p_score was measured
        total_length: Total sequence length
        decay: 'linear', 'exponential', or 'none'
        decay_rate: Rate of decay (interpretation depends on decay type)

    Returns:
        Array of weights, shape (total_length,)
    """
    all_positions = np.arange(total_length)
    branch_points = np.array(branch_points)

    if len(branch_points) == 0:
        return np.ones(total_length)

    # Distance to nearest measured point
    distances = np.min(np.abs(all_positions[:, None] - branch_points[None, :]), axis=1)

    if decay == 'none':
        weights = np.ones(total_length)
    elif decay == 'linear':
        max_dist = np.max(distances) if np.max(distances) > 0 else 1
        weights = 1.0 - decay_rate * (distances / max_dist)
        weights = np.maximum(weights, 0.1)  # Floor to avoid zero weights
    elif decay == 'exponential':
        weights = np.exp(-decay_rate * distances)
    else:
        raise ValueError(f"Unknown decay type: {decay}")

    return weights

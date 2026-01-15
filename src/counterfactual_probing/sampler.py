"""
Token-level sampling for selecting branch points.

The TokenSampler determines which token positions to use as branch points
for counterfactual generation.
"""

import random
import warnings


class TokenSampler:
    """
    Sample random token positions for counterfactual branching.

    Supports three sampling methods:
    - random: Pure random sampling without forced endpoints (recommended)
    - uniform_count: Sample exactly N positions with forced endpoints (deprecated)
    - density: Sample approximately (density * sequence_length) positions (deprecated)

    Can optionally limit sampling to before a boundary token (e.g., </think>).
    """

    VALID_METHODS = ("random", "uniform_count", "density")
    DEPRECATED_METHODS = ("uniform_count", "density")

    def __init__(
        self,
        method: str = "random",
        num_samples: int = 20,
        density: float = 0.02,
        seed: int | None = None,
        cot_boundary_token: int | None = None,
    ):
        """
        Initialize the sampler.

        Args:
            method: Sampling method ("random", "uniform_count", or "density")
                - "random": Pure random sampling, no forced endpoints (recommended)
                - "uniform_count": Forces first/last positions (deprecated)
                - "density": Forces first/last positions (deprecated)
            num_samples: Number of samples for random/uniform_count methods
            density: Sampling density for density method (0 < density <= 1)
            seed: Random seed for reproducibility
            cot_boundary_token: If set, only sample from positions before this token.
                Useful for limiting sampling to chain-of-thought tokens (before </think>).
                Set to None to sample from entire sequence.

        Raises:
            ValueError: If method is invalid or parameters are out of range
        """
        if method not in self.VALID_METHODS:
            raise ValueError(
                f"Unknown sampling method: {method}. "
                f"Must be one of {self.VALID_METHODS}"
            )

        # Warn about deprecated methods
        if method in self.DEPRECATED_METHODS:
            warnings.warn(
                f"Sampling method '{method}' is deprecated because it forces "
                f"inclusion of first and last positions. Use method='random' for "
                f"pure random sampling without forced endpoints.",
                DeprecationWarning,
                stacklevel=2,
            )

        if num_samples < 1:
            raise ValueError("num_samples must be positive")

        if density <= 0 or density > 1:
            raise ValueError("density must be between 0 and 1 (exclusive of 0)")

        self.method = method
        self.num_samples = num_samples
        self.density = density
        self.seed = seed
        self.cot_boundary_token = cot_boundary_token

        # Initialize random state
        self._rng = random.Random(seed)

    def _find_boundary_index(self, token_ids: list[int]) -> int | None:
        """
        Find the index of the boundary token in the sequence.

        Args:
            token_ids: Full sequence of token IDs

        Returns:
            Index of boundary token, or None if not found or not configured
        """
        if self.cot_boundary_token is None:
            return None

        try:
            return token_ids.index(self.cot_boundary_token)
        except ValueError:
            return None

    def sample(self, token_ids: list[int]) -> list[int]:
        """
        Return sorted list of token indices to branch from.

        Args:
            token_ids: Full sequence of token IDs

        Returns:
            Sorted list of indices into token_ids to use as branch points.
            If cot_boundary_token is set, only returns indices before that token.
        """
        # Find sampling boundary
        boundary_idx = self._find_boundary_index(token_ids)

        # Determine the range to sample from
        if boundary_idx is not None:
            # Sample only from tokens before the boundary (not including boundary)
            sample_range = boundary_idx
        else:
            # Sample from entire sequence
            sample_range = len(token_ids)

        # Handle edge cases
        if sample_range == 0:
            return []
        if sample_range == 1:
            return [0]
        if sample_range == 2:
            return [0, 1]

        if self.method == "random":
            return self._sample_random(sample_range)
        elif self.method == "uniform_count":
            return self._sample_uniform_count(sample_range)
        elif self.method == "density":
            return self._sample_density(sample_range)
        else:
            # Should not reach here due to __init__ validation
            raise ValueError(f"Unknown sampling method: {self.method}")

    def _sample_random(self, n: int) -> list[int]:
        """
        Pure random sampling without forced endpoints.

        Args:
            n: Number of positions available to sample from

        Returns:
            Sorted list of randomly sampled indices
        """
        # If we want more samples than available, return all
        if self.num_samples >= n:
            return list(range(n))

        # Pure random sample from all positions
        all_positions = list(range(n))
        sampled = self._rng.sample(all_positions, self.num_samples)

        return sorted(sampled)

    def _sample_uniform_count(self, n: int) -> list[int]:
        """
        Sample exactly num_samples positions with forced endpoints.

        DEPRECATED: Forces first (0) and last (n-1) positions.
        """
        # If we want more samples than available, return all
        if self.num_samples >= n:
            return list(range(n))

        # Start with endpoints (deprecated behavior)
        indices = {0, n - 1}

        # Fill remaining slots randomly from middle positions
        middle_positions = list(range(1, n - 1))
        remaining_needed = self.num_samples - len(indices)

        if remaining_needed > 0 and middle_positions:
            # Sample without replacement
            sampled = self._rng.sample(
                middle_positions,
                min(remaining_needed, len(middle_positions))
            )
            indices.update(sampled)

        return sorted(indices)

    def _sample_density(self, n: int) -> list[int]:
        """
        Sample approximately (density * n) positions with forced endpoints.

        DEPRECATED: Forces first (0) and last (n-1) positions.
        """
        # Calculate target number of samples
        target_count = max(2, int(n * self.density))

        # If target >= n, return all
        if target_count >= n:
            return list(range(n))

        # Start with endpoints (deprecated behavior)
        indices = {0, n - 1}

        # Fill remaining slots randomly from middle positions
        middle_positions = list(range(1, n - 1))
        remaining_needed = target_count - len(indices)

        if remaining_needed > 0 and middle_positions:
            sampled = self._rng.sample(
                middle_positions,
                min(remaining_needed, len(middle_positions))
            )
            indices.update(sampled)

        return sorted(indices)

    def reset(self, seed: int | None = None):
        """
        Reset the random state.

        Args:
            seed: New seed to use. If None, uses the original seed.
        """
        self._rng = random.Random(seed if seed is not None else self.seed)

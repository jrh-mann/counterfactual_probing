"""
Token-level sampling for selecting branch points.

The TokenSampler determines which token positions to use as branch points
for counterfactual generation.
"""

import random


class TokenSampler:
    """
    Sample random token positions for counterfactual branching.

    Supports two sampling methods:
    - uniform_count: Sample exactly N positions uniformly distributed
    - density: Sample approximately (density * sequence_length) positions
    """

    def __init__(
        self,
        method: str = "uniform_count",
        num_samples: int = 20,
        density: float = 0.02,
        seed: int | None = None,
    ):
        """
        Initialize the sampler.

        Args:
            method: Sampling method ("uniform_count" or "density")
            num_samples: Number of samples for uniform_count method
            density: Sampling density for density method (0 < density <= 1)
            seed: Random seed for reproducibility

        Raises:
            ValueError: If method is invalid or parameters are out of range
        """
        if method not in ("uniform_count", "density"):
            raise ValueError(
                f"Unknown sampling method: {method}. "
                f"Must be 'uniform_count' or 'density'"
            )

        if num_samples < 1:
            raise ValueError("num_samples must be positive")

        if density <= 0 or density > 1:
            raise ValueError("density must be between 0 and 1 (exclusive of 0)")

        self.method = method
        self.num_samples = num_samples
        self.density = density
        self.seed = seed

        # Initialize random state
        self._rng = random.Random(seed)

    def sample(self, token_ids: list[int]) -> list[int]:
        """
        Return sorted list of token indices to branch from.

        Args:
            token_ids: Full sequence of token IDs

        Returns:
            Sorted list of indices into token_ids to use as branch points.
            Always includes first (0) and last (len-1) positions when possible.
        """
        n = len(token_ids)

        # Handle edge cases
        if n == 0:
            return []
        if n == 1:
            return [0]
        if n == 2:
            return [0, 1]

        if self.method == "uniform_count":
            return self._sample_uniform_count(n)
        elif self.method == "density":
            return self._sample_density(n)
        else:
            # Should not reach here due to __init__ validation
            raise ValueError(f"Unknown sampling method: {self.method}")

    def _sample_uniform_count(self, n: int) -> list[int]:
        """
        Sample exactly num_samples positions uniformly.

        Always includes first (0) and last (n-1) positions.
        """
        # If we want more samples than available, return all
        if self.num_samples >= n:
            return list(range(n))

        # Start with endpoints
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
        Sample approximately (density * n) positions.

        Always includes first (0) and last (n-1) positions.
        """
        # Calculate target number of samples
        target_count = max(2, int(n * self.density))

        # If target >= n, return all
        if target_count >= n:
            return list(range(n))

        # Start with endpoints
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

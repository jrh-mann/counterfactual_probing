"""
Tests for token sampling correctness.

The TokenSampler determines which token positions to branch from
for counterfactual generation.
"""

import pytest
import random
from typing import List


class TestUniformCountSampling:
    """Test uniform_count sampling method."""

    def test_includes_endpoints(self):
        """First and last token should always be included."""
        from counterfactual_probing.sampler import TokenSampler

        sampler = TokenSampler(method="uniform_count", num_samples=5)
        token_ids = list(range(100))  # 100 tokens

        indices = sampler.sample(token_ids)

        assert 0 in indices, "First token (0) should be included"
        assert 99 in indices, "Last token (99) should be included"

    def test_exact_count(self):
        """Should return exactly num_samples indices."""
        from counterfactual_probing.sampler import TokenSampler

        for num_samples in [3, 5, 10, 20]:
            sampler = TokenSampler(method="uniform_count", num_samples=num_samples)
            token_ids = list(range(100))

            indices = sampler.sample(token_ids)

            assert len(indices) == num_samples, (
                f"Expected {num_samples} samples, got {len(indices)}"
            )

    def test_sorted_output(self):
        """Output indices should be sorted."""
        from counterfactual_probing.sampler import TokenSampler

        sampler = TokenSampler(method="uniform_count", num_samples=10)
        token_ids = list(range(50))

        indices = sampler.sample(token_ids)

        assert indices == sorted(indices), "Indices should be sorted"

    def test_unique_indices(self):
        """All indices should be unique."""
        from counterfactual_probing.sampler import TokenSampler

        sampler = TokenSampler(method="uniform_count", num_samples=15)
        token_ids = list(range(100))

        indices = sampler.sample(token_ids)

        assert len(indices) == len(set(indices)), "Indices should be unique"

    def test_indices_in_range(self):
        """All indices should be valid token positions."""
        from counterfactual_probing.sampler import TokenSampler

        sampler = TokenSampler(method="uniform_count", num_samples=10)
        token_ids = list(range(50))

        indices = sampler.sample(token_ids)

        for idx in indices:
            assert 0 <= idx < len(token_ids), (
                f"Index {idx} out of range [0, {len(token_ids)})"
            )


class TestDensitySampling:
    """Test density-based sampling method."""

    def test_includes_endpoints(self):
        """First and last token should always be included."""
        from counterfactual_probing.sampler import TokenSampler

        sampler = TokenSampler(method="density", density=0.1)
        token_ids = list(range(100))

        indices = sampler.sample(token_ids)

        assert 0 in indices, "First token (0) should be included"
        assert 99 in indices, "Last token (99) should be included"

    def test_approximate_density(self):
        """Should sample approximately density * len(tokens) positions."""
        from counterfactual_probing.sampler import TokenSampler

        density = 0.1
        sampler = TokenSampler(method="density", density=density)
        token_ids = list(range(100))

        # Run multiple times and check average
        counts = []
        for seed in range(50):
            sampler_with_seed = TokenSampler(method="density", density=density, seed=seed)
            indices = sampler_with_seed.sample(token_ids)
            counts.append(len(indices))

        avg_count = sum(counts) / len(counts)
        expected = density * len(token_ids)

        # Should be within 50% of expected (allowing for randomness + endpoints)
        assert expected * 0.5 <= avg_count <= expected * 2.0, (
            f"Average count {avg_count} too far from expected {expected}"
        )

    def test_minimum_samples(self):
        """Should always return at least 2 samples (endpoints)."""
        from counterfactual_probing.sampler import TokenSampler

        sampler = TokenSampler(method="density", density=0.001)  # Very low density
        token_ids = list(range(100))

        indices = sampler.sample(token_ids)

        assert len(indices) >= 2, "Should have at least 2 samples (endpoints)"


class TestSamplerDeterminism:
    """Test deterministic behavior with seeds."""

    def test_same_seed_same_result(self):
        """Same seed should produce same samples."""
        from counterfactual_probing.sampler import TokenSampler

        token_ids = list(range(100))

        sampler1 = TokenSampler(method="uniform_count", num_samples=10, seed=42)
        sampler2 = TokenSampler(method="uniform_count", num_samples=10, seed=42)

        indices1 = sampler1.sample(token_ids)
        indices2 = sampler2.sample(token_ids)

        assert indices1 == indices2, "Same seed should produce same result"

    def test_different_seed_different_result(self):
        """Different seeds should (usually) produce different samples."""
        from counterfactual_probing.sampler import TokenSampler

        token_ids = list(range(100))

        sampler1 = TokenSampler(method="uniform_count", num_samples=10, seed=42)
        sampler2 = TokenSampler(method="uniform_count", num_samples=10, seed=123)

        indices1 = sampler1.sample(token_ids)
        indices2 = sampler2.sample(token_ids)

        # Endpoints will be same, but middle should differ
        assert indices1 != indices2, "Different seeds should produce different results"

    def test_reproducible_across_calls(self):
        """Multiple calls with same sampler should be reproducible if re-seeded."""
        from counterfactual_probing.sampler import TokenSampler

        token_ids = list(range(100))

        sampler = TokenSampler(method="uniform_count", num_samples=10, seed=42)
        indices1 = sampler.sample(token_ids)

        # Create new sampler with same seed
        sampler2 = TokenSampler(method="uniform_count", num_samples=10, seed=42)
        indices2 = sampler2.sample(token_ids)

        assert indices1 == indices2


class TestShortSequences:
    """Test handling of short sequences."""

    def test_sequence_shorter_than_num_samples(self):
        """Sequences shorter than num_samples should return all indices."""
        from counterfactual_probing.sampler import TokenSampler

        sampler = TokenSampler(method="uniform_count", num_samples=20)
        token_ids = list(range(5))  # Only 5 tokens

        indices = sampler.sample(token_ids)

        # Should return all indices since we can't sample more than exist
        assert indices == [0, 1, 2, 3, 4]

    def test_single_token_sequence(self):
        """Single token sequence should return just that index."""
        from counterfactual_probing.sampler import TokenSampler

        sampler = TokenSampler(method="uniform_count", num_samples=10)
        token_ids = [42]  # Single token

        indices = sampler.sample(token_ids)

        assert indices == [0]

    def test_two_token_sequence(self):
        """Two token sequence should return both indices."""
        from counterfactual_probing.sampler import TokenSampler

        sampler = TokenSampler(method="uniform_count", num_samples=10)
        token_ids = [1, 2]

        indices = sampler.sample(token_ids)

        assert indices == [0, 1]

    def test_empty_sequence(self):
        """Empty sequence should return empty list."""
        from counterfactual_probing.sampler import TokenSampler

        sampler = TokenSampler(method="uniform_count", num_samples=10)
        token_ids = []

        indices = sampler.sample(token_ids)

        assert indices == []


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_num_samples_equals_length(self):
        """When num_samples equals sequence length, return all indices."""
        from counterfactual_probing.sampler import TokenSampler

        sampler = TokenSampler(method="uniform_count", num_samples=10)
        token_ids = list(range(10))

        indices = sampler.sample(token_ids)

        assert indices == list(range(10))

    def test_invalid_method_raises(self):
        """Invalid sampling method should raise error."""
        from counterfactual_probing.sampler import TokenSampler

        with pytest.raises(ValueError, match="Unknown sampling method"):
            TokenSampler(method="invalid_method")

    def test_negative_num_samples_raises(self):
        """Negative num_samples should raise error."""
        from counterfactual_probing.sampler import TokenSampler

        with pytest.raises(ValueError, match="num_samples must be positive"):
            TokenSampler(method="uniform_count", num_samples=-1)

    def test_zero_num_samples_raises(self):
        """Zero num_samples should raise error."""
        from counterfactual_probing.sampler import TokenSampler

        with pytest.raises(ValueError, match="num_samples must be positive"):
            TokenSampler(method="uniform_count", num_samples=0)

    def test_negative_density_raises(self):
        """Negative density should raise error."""
        from counterfactual_probing.sampler import TokenSampler

        with pytest.raises(ValueError, match="density must be between 0 and 1"):
            TokenSampler(method="density", density=-0.1)

    def test_density_greater_than_one_raises(self):
        """Density > 1 should raise error."""
        from counterfactual_probing.sampler import TokenSampler

        with pytest.raises(ValueError, match="density must be between 0 and 1"):
            TokenSampler(method="density", density=1.5)


class TestSamplerWithRealTokens:
    """Test sampler with actual tokenized text."""

    def test_with_tokenized_text(self, small_tokenizer):
        """Test sampler with real tokenized text."""
        from counterfactual_probing.sampler import TokenSampler

        text = "The quick brown fox jumps over the lazy dog."
        token_ids = small_tokenizer.encode(text, add_special_tokens=False)

        sampler = TokenSampler(method="uniform_count", num_samples=5, seed=42)
        indices = sampler.sample(token_ids)

        # Verify we can use these indices
        for idx in indices:
            prefix_ids = token_ids[:idx + 1]
            prefix_text = small_tokenizer.decode(prefix_ids)
            assert text.startswith(prefix_text) or len(prefix_text) <= len(text)

    def test_sampled_prefixes_are_valid(self, small_tokenizer):
        """Sampled indices should produce valid prefixes."""
        from counterfactual_probing.sampler import TokenSampler

        text = "This is a longer text with many tokens to sample from."
        token_ids = small_tokenizer.encode(text, add_special_tokens=False)

        sampler = TokenSampler(method="uniform_count", num_samples=10, seed=42)
        indices = sampler.sample(token_ids)

        for idx in indices:
            # Can extract prefix
            prefix_ids = token_ids[:idx + 1]
            assert len(prefix_ids) == idx + 1

            # Can decode prefix
            prefix_text = small_tokenizer.decode(prefix_ids)
            assert isinstance(prefix_text, str)

            # Prefix is substring of or equals decoded full sequence
            full_decoded = small_tokenizer.decode(token_ids)
            assert full_decoded.startswith(prefix_text) or prefix_text in full_decoded

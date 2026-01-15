"""
Tests for sampler fixes:
1. No forced endpoints - pure random sampling
2. CoT boundary detection - only sample before </think> token
3. Configurable sampling scope
"""

import pytest
import random
from typing import List


class TestNoForcedEndpoints:
    """Test that sampling is pure random without forced endpoints."""

    def test_endpoints_not_guaranteed(self):
        """First and last token should NOT always be included."""
        from counterfactual_probing.sampler import TokenSampler

        token_ids = list(range(100))

        # Run many times and check that endpoints aren't always present
        first_included_count = 0
        last_included_count = 0
        num_trials = 100

        for seed in range(num_trials):
            sampler = TokenSampler(method="random", num_samples=5, seed=seed)
            indices = sampler.sample(token_ids)

            if 0 in indices:
                first_included_count += 1
            if 99 in indices:
                last_included_count += 1

        # With pure random sampling of 5 from 100, probability of including
        # any specific index is 5/100 = 5%. After 100 trials, we'd expect ~5 hits.
        # Allow for some variance but definitely not 100%
        assert first_included_count < 50, (
            f"First token included {first_included_count}/100 times - "
            "should be random, not forced"
        )
        assert last_included_count < 50, (
            f"Last token included {last_included_count}/100 times - "
            "should be random, not forced"
        )

    def test_num_samples_is_exact(self):
        """num_samples should mean exactly that many samples, no forced additions."""
        from counterfactual_probing.sampler import TokenSampler

        sampler = TokenSampler(method="random", num_samples=10, seed=42)
        token_ids = list(range(100))

        indices = sampler.sample(token_ids)

        assert len(indices) == 10, (
            f"Asked for 10 samples, got {len(indices)}"
        )

    def test_pure_random_distribution(self):
        """Samples should be uniformly distributed across the sequence."""
        from counterfactual_probing.sampler import TokenSampler

        token_ids = list(range(100))

        # Collect all sampled indices across many trials
        all_indices = []
        for seed in range(500):
            sampler = TokenSampler(method="random", num_samples=10, seed=seed)
            indices = sampler.sample(token_ids)
            all_indices.extend(indices)

        # Check distribution is roughly uniform
        # Divide into 10 buckets of 10 indices each
        buckets = [0] * 10
        for idx in all_indices:
            buckets[idx // 10] += 1

        # Each bucket should have roughly equal counts
        avg = len(all_indices) / 10
        for i, count in enumerate(buckets):
            assert 0.5 * avg < count < 1.5 * avg, (
                f"Bucket {i} has {count} samples, expected ~{avg}. "
                f"Distribution not uniform: {buckets}"
            )


class TestCoTBoundaryDetection:
    """Test that sampling respects CoT boundary (</think> token)."""

    @pytest.fixture
    def think_token_id(self):
        """The </think> token ID for Qwen3."""
        return 151668

    def test_sample_only_before_think_token(self, think_token_id):
        """Samples should only come from tokens before </think>."""
        from counterfactual_probing.sampler import TokenSampler

        # Simulate: [CoT tokens...] </think> [answer tokens...]
        cot_tokens = list(range(100, 150))  # 50 CoT tokens
        answer_tokens = list(range(200, 220))  # 20 answer tokens
        token_ids = cot_tokens + [think_token_id] + answer_tokens

        sampler = TokenSampler(
            method="random",
            num_samples=10,
            seed=42,
            cot_boundary_token=think_token_id
        )
        indices = sampler.sample(token_ids)

        # Find where </think> is
        think_index = token_ids.index(think_token_id)

        # All samples should be before </think>
        for idx in indices:
            assert idx < think_index, (
                f"Sampled index {idx} is at or after </think> at {think_index}"
            )

    def test_no_think_token_samples_all(self, think_token_id):
        """If no </think> token, sample from entire sequence."""
        from counterfactual_probing.sampler import TokenSampler

        # No think token in sequence
        token_ids = list(range(100))

        sampler = TokenSampler(
            method="random",
            num_samples=10,
            seed=42,
            cot_boundary_token=think_token_id
        )
        indices = sampler.sample(token_ids)

        # Should be able to sample from anywhere
        assert len(indices) == 10
        # At least some samples should be in latter half
        assert any(idx >= 50 for idx in indices), (
            "Without boundary token, should sample from entire sequence"
        )

    def test_think_token_at_start(self, think_token_id):
        """Handle edge case where </think> is very early."""
        from counterfactual_probing.sampler import TokenSampler

        # </think> after only 3 tokens
        token_ids = [1, 2, 3, think_token_id] + list(range(100, 150))

        sampler = TokenSampler(
            method="random",
            num_samples=10,
            seed=42,
            cot_boundary_token=think_token_id
        )
        indices = sampler.sample(token_ids)

        # Should only sample from [0, 1, 2], not including think token position
        think_index = 3
        for idx in indices:
            assert idx < think_index

        # Can only get at most 3 samples
        assert len(indices) <= 3

    def test_boundary_token_none_disables_detection(self, think_token_id):
        """Setting cot_boundary_token=None should disable boundary detection."""
        from counterfactual_probing.sampler import TokenSampler

        # Has think token but boundary detection disabled
        cot_tokens = list(range(100, 150))
        answer_tokens = list(range(200, 220))
        token_ids = cot_tokens + [think_token_id] + answer_tokens

        sampler = TokenSampler(
            method="random",
            num_samples=20,
            seed=42,
            cot_boundary_token=None  # Disabled
        )
        indices = sampler.sample(token_ids)

        think_index = token_ids.index(think_token_id)

        # Should be able to sample from anywhere, including after </think>
        # With 20 samples from 71 tokens, very likely to get some after think
        has_post_think = any(idx > think_index for idx in indices)
        assert has_post_think, (
            "With boundary detection disabled, should sample from entire sequence"
        )


class TestBackwardsCompatibility:
    """Test that old sampling methods still work but with deprecation warnings."""

    def test_uniform_count_warns_deprecation(self):
        """uniform_count method should warn about forced endpoints behavior."""
        from counterfactual_probing.sampler import TokenSampler
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sampler = TokenSampler(method="uniform_count", num_samples=5)

            # Should have a deprecation warning
            assert any("deprecated" in str(warning.message).lower() or
                      "uniform_count" in str(warning.message).lower()
                      for warning in w), (
                "uniform_count should warn about deprecation"
            )

    def test_density_warns_deprecation(self):
        """density method should warn about forced endpoints behavior."""
        from counterfactual_probing.sampler import TokenSampler
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sampler = TokenSampler(method="density", density=0.1)

            assert any("deprecated" in str(warning.message).lower() or
                      "density" in str(warning.message).lower()
                      for warning in w), (
                "density should warn about deprecation"
            )


class TestSamplerConfig:
    """Test sampler configuration options."""

    def test_invalid_method_raises(self):
        """Invalid method should raise clear error."""
        from counterfactual_probing.sampler import TokenSampler

        with pytest.raises(ValueError, match="Unknown sampling method"):
            TokenSampler(method="invalid_method")

    def test_random_method_exists(self):
        """New 'random' method should be available."""
        from counterfactual_probing.sampler import TokenSampler

        # Should not raise
        sampler = TokenSampler(method="random", num_samples=10)
        assert sampler.method == "random"

    def test_cot_boundary_token_configurable(self):
        """cot_boundary_token should be configurable."""
        from counterfactual_probing.sampler import TokenSampler

        custom_token = 12345
        sampler = TokenSampler(
            method="random",
            num_samples=10,
            cot_boundary_token=custom_token
        )

        # Verify it's stored
        assert sampler.cot_boundary_token == custom_token


class TestSamplerWithRealTokenizer:
    """Test sampler with actual Qwen tokenizer and </think> token."""

    @pytest.fixture
    def qwen_tokenizer(self):
        """Load Qwen tokenizer if available."""
        pytest.importorskip("transformers")
        from transformers import AutoTokenizer
        try:
            return AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-0.5B",
                trust_remote_code=True
            )
        except Exception:
            pytest.skip("Qwen tokenizer not available")

    def test_with_simulated_cot_response(self, qwen_tokenizer):
        """Test with a simulated CoT response containing </think>."""
        from counterfactual_probing.sampler import TokenSampler

        # Simulate a response with thinking
        response = "<think>Let me work through this step by step. First, I need to consider the problem. Then I'll solve it.</think>The answer is 42."

        token_ids = qwen_tokenizer.encode(response, add_special_tokens=False)

        # Find </think> token - may be single token (Qwen3) or multiple (Qwen2.5)
        think_token_ids = qwen_tokenizer.encode("</think>", add_special_tokens=False)

        if len(think_token_ids) != 1:
            # Qwen2.5 and similar don't have </think> as single token
            # Use a synthetic boundary token for this test
            pytest.skip(
                f"</think> is {len(think_token_ids)} tokens in this tokenizer, "
                "test requires single-token boundary (like Qwen3)"
            )

        think_token_id = think_token_ids[0]

        # Find position in sequence
        try:
            think_pos = token_ids.index(think_token_id)
        except ValueError:
            pytest.skip("</think> token not found in tokenized response")

        sampler = TokenSampler(
            method="random",
            num_samples=10,
            seed=42,
            cot_boundary_token=think_token_id
        )

        indices = sampler.sample(token_ids)

        # All indices should be before </think>
        for idx in indices:
            assert idx < think_pos, (
                f"Index {idx} is at or after </think> at position {think_pos}"
            )

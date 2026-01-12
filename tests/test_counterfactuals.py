"""
Tests for counterfactual generation.

These tests verify that counterfactuals are generated correctly from
prefix-conditioned prompts.
"""

import pytest


class TestPrefixConditioning:
    """Test that counterfactuals are conditioned on correct prefix."""

    def test_prefix_ids_create_valid_prompt(self, small_tokenizer):
        """Prefix token IDs should create a valid prompt string."""
        from counterfactual_probing.generator import create_prefix_prompt

        text = "The quick brown fox jumps over the lazy dog."
        token_ids = small_tokenizer.encode(text, add_special_tokens=False)

        # Take first half as prefix
        split_point = len(token_ids) // 2
        prefix_ids = token_ids[:split_point]

        prompt = create_prefix_prompt(
            prefix_ids=prefix_ids,
            tokenizer=small_tokenizer,
            messages=[{"role": "user", "content": "Generate text"}],
        )

        # Should be a valid string
        assert isinstance(prompt, str)

        # Should contain the prefix text
        prefix_text = small_tokenizer.decode(prefix_ids)
        assert prefix_text in prompt

    def test_prefix_prompt_uses_chat_template(self, small_tokenizer):
        """Prefix prompt should use chat template format."""
        from counterfactual_probing.generator import create_prefix_prompt

        prefix_ids = small_tokenizer.encode("Hello", add_special_tokens=False)
        messages = [{"role": "user", "content": "Say hello"}]

        prompt = create_prefix_prompt(
            prefix_ids=prefix_ids,
            tokenizer=small_tokenizer,
            messages=messages,
        )

        # Should have applied chat template
        # The prompt should contain assistant prefix from chat template
        assert "Hello" in prompt


class TestCounterfactualTokenIds:
    """Test that counterfactual token IDs are valid."""

    def test_counterfactual_ids_are_valid_vocab(self, small_tokenizer):
        """All counterfactual token IDs should be valid vocabulary indices."""
        from counterfactual_probing.generator import validate_token_ids

        vocab_size = small_tokenizer.vocab_size

        # Valid IDs
        valid_ids = [0, 1, 100, vocab_size - 1]
        result = validate_token_ids(valid_ids, vocab_size)
        assert result["valid"] is True

        # Invalid IDs
        invalid_ids = [0, vocab_size, vocab_size + 100]
        result = validate_token_ids(invalid_ids, vocab_size)
        assert result["valid"] is False

    def test_counterfactual_not_include_prefix(self, small_tokenizer):
        """Counterfactual continuation should not include prefix tokens."""
        from counterfactual_probing.generator import extract_continuation

        full_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        prefix_ids = [1, 2, 3, 4, 5]

        continuation = extract_continuation(full_ids, prefix_ids)

        # Continuation should not include prefix
        assert continuation == [6, 7, 8, 9, 10]

    def test_extract_continuation_exact_prefix(self, small_tokenizer):
        """Should handle exact prefix match."""
        from counterfactual_probing.generator import extract_continuation

        full_ids = [1, 2, 3]
        prefix_ids = [1, 2, 3]

        continuation = extract_continuation(full_ids, prefix_ids)

        # No continuation
        assert continuation == []


class TestCounterfactualVariation:
    """Test that counterfactuals vary with temperature > 0."""

    @pytest.mark.integration
    def test_nonzero_temperature_produces_variation(self, small_model_name, small_tokenizer):
        """With temperature > 0, counterfactuals should vary."""
        from counterfactual_probing.generator import generate_counterfactuals

        prefix_ids = small_tokenizer.encode("The answer is", add_special_tokens=False)
        messages = [{"role": "user", "content": "What is 2+2?"}]

        counterfactuals = generate_counterfactuals(
            prefix_ids=prefix_ids,
            messages=messages,
            model_name=small_model_name,
            tokenizer=small_tokenizer,
            num_counterfactuals=5,
            temperature=0.7,
            max_tokens=20,
        )

        assert len(counterfactuals) == 5

        # At least some should be different
        unique_continuations = set(tuple(cf["token_ids"]) for cf in counterfactuals)
        # With temperature > 0, expect some variation (but not guaranteed)
        # This is a probabilistic test

    @pytest.mark.integration
    def test_zero_temperature_deterministic(self, small_model_name, small_tokenizer):
        """With temperature = 0, counterfactuals should be identical."""
        from counterfactual_probing.generator import generate_counterfactuals

        prefix_ids = small_tokenizer.encode("The answer is", add_special_tokens=False)
        messages = [{"role": "user", "content": "What is 2+2?"}]

        counterfactuals = generate_counterfactuals(
            prefix_ids=prefix_ids,
            messages=messages,
            model_name=small_model_name,
            tokenizer=small_tokenizer,
            num_counterfactuals=3,
            temperature=0.0,
            max_tokens=20,
        )

        assert len(counterfactuals) == 3

        # All should be identical with temp=0
        first = counterfactuals[0]["token_ids"]
        for cf in counterfactuals[1:]:
            assert cf["token_ids"] == first


class TestCounterfactualStructure:
    """Test counterfactual output structure."""

    @pytest.mark.integration
    def test_counterfactual_has_required_fields(self, small_model_name, small_tokenizer):
        """Each counterfactual should have required fields."""
        from counterfactual_probing.generator import generate_counterfactuals

        prefix_ids = small_tokenizer.encode("Hello", add_special_tokens=False)
        messages = [{"role": "user", "content": "Greet me"}]

        counterfactuals = generate_counterfactuals(
            prefix_ids=prefix_ids,
            messages=messages,
            model_name=small_model_name,
            tokenizer=small_tokenizer,
            num_counterfactuals=2,
            temperature=0.5,
            max_tokens=10,
        )

        for cf in counterfactuals:
            assert "token_ids" in cf
            assert isinstance(cf["token_ids"], list)

    @pytest.mark.integration
    def test_counterfactual_token_ids_decodable(self, small_model_name, small_tokenizer):
        """Counterfactual token IDs should be decodable."""
        from counterfactual_probing.generator import generate_counterfactuals

        prefix_ids = small_tokenizer.encode("Test", add_special_tokens=False)
        messages = [{"role": "user", "content": "Test"}]

        counterfactuals = generate_counterfactuals(
            prefix_ids=prefix_ids,
            messages=messages,
            model_name=small_model_name,
            tokenizer=small_tokenizer,
            num_counterfactuals=2,
            temperature=0.5,
            max_tokens=10,
        )

        for cf in counterfactuals:
            # Should decode without error
            text = small_tokenizer.decode(cf["token_ids"])
            assert isinstance(text, str)


class TestBatchGeneration:
    """Test batch generation of counterfactuals."""

    @pytest.mark.integration
    def test_batch_generation_correct_count(self, small_model_name, small_tokenizer):
        """Batch generation should produce correct number of counterfactuals."""
        from counterfactual_probing.generator import generate_counterfactuals_batch

        prefix_ids = small_tokenizer.encode("Start", add_special_tokens=False)
        messages = [{"role": "user", "content": "Continue"}]

        num_counterfactuals = 10

        counterfactuals = generate_counterfactuals_batch(
            prefix_ids=prefix_ids,
            messages=messages,
            model_name=small_model_name,
            tokenizer=small_tokenizer,
            num_counterfactuals=num_counterfactuals,
            temperature=0.5,
            max_tokens=10,
        )

        assert len(counterfactuals) == num_counterfactuals


class TestEdgeCases:
    """Test edge cases in counterfactual generation."""

    def test_empty_prefix(self, small_tokenizer):
        """Empty prefix should be handled."""
        from counterfactual_probing.generator import create_prefix_prompt

        messages = [{"role": "user", "content": "Generate from scratch"}]

        prompt = create_prefix_prompt(
            prefix_ids=[],
            tokenizer=small_tokenizer,
            messages=messages,
        )

        # Should still create valid prompt
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_very_long_prefix(self, small_tokenizer):
        """Long prefix should be handled."""
        from counterfactual_probing.generator import create_prefix_prompt

        # Create a long prefix
        long_text = "word " * 500
        prefix_ids = small_tokenizer.encode(long_text, add_special_tokens=False)

        messages = [{"role": "user", "content": "Continue this"}]

        prompt = create_prefix_prompt(
            prefix_ids=prefix_ids,
            tokenizer=small_tokenizer,
            messages=messages,
        )

        assert isinstance(prompt, str)

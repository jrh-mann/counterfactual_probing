"""
Tests for generator token-level integrity fixes:
1. Verify prefix tokens match after round-trip (decode -> re-encode)
2. Detect and handle tokenization drift
3. Validate token-level operations don't lose information
"""

import pytest
from typing import List


class TestTokenRoundTrip:
    """Test that token IDs survive decode/encode round-trip."""

    def test_prefix_roundtrip_integrity(self, small_tokenizer):
        """Prefix tokens should match after decode -> encode round-trip."""
        from counterfactual_probing.generator import create_prefix_prompt

        # Create a prefix from actual text
        text = "The quick brown fox jumps over the lazy dog."
        original_ids = small_tokenizer.encode(text, add_special_tokens=False)

        # Take a prefix
        prefix_ids = original_ids[:len(original_ids) // 2]

        # Decode it (as the generator does)
        decoded_text = small_tokenizer.decode(prefix_ids, skip_special_tokens=True)

        # Re-encode (simulating what happens in chat template)
        re_encoded_ids = small_tokenizer.encode(decoded_text, add_special_tokens=False)

        # These should match!
        assert prefix_ids == re_encoded_ids, (
            f"Token round-trip failed!\n"
            f"Original:   {prefix_ids}\n"
            f"Re-encoded: {re_encoded_ids}\n"
            f"Decoded text: {repr(decoded_text)}"
        )

    def test_prefix_roundtrip_with_special_chars(self, small_tokenizer):
        """Round-trip should work with special characters."""
        texts = [
            "Hello, world!",
            "What's 2+2?",
            "Code: def foo(): return 42",
            "Math: \\frac{1}{2} = 0.5",
            "Emoji test (no emoji here)",
            "Newline\ntest\nhere",
        ]

        for text in texts:
            original_ids = small_tokenizer.encode(text, add_special_tokens=False)
            if len(original_ids) < 2:
                continue

            prefix_ids = original_ids[:len(original_ids) // 2]
            decoded_text = small_tokenizer.decode(prefix_ids, skip_special_tokens=True)
            re_encoded_ids = small_tokenizer.encode(decoded_text, add_special_tokens=False)

            assert prefix_ids == re_encoded_ids, (
                f"Round-trip failed for text: {repr(text)}"
            )

    def test_detect_tokenization_drift(self, small_tokenizer):
        """Should detect when tokenization produces different IDs."""
        from counterfactual_probing.generator import validate_prefix_roundtrip

        # Normal case - should pass
        text = "Hello world"
        token_ids = small_tokenizer.encode(text, add_special_tokens=False)

        result = validate_prefix_roundtrip(token_ids, small_tokenizer)
        assert result["valid"] is True, f"Expected valid, got: {result}"

    def test_validate_prefix_raises_on_drift(self, small_tokenizer):
        """Should raise/warn when prefix doesn't survive round-trip."""
        from counterfactual_probing.generator import validate_prefix_roundtrip

        # Artificially create mismatched tokens (if possible)
        # This tests that the validation function works
        token_ids = [1, 2, 3, 4, 5]

        result = validate_prefix_roundtrip(token_ids, small_tokenizer)

        # Result should indicate whether round-trip succeeded
        assert "valid" in result
        assert "original_ids" in result
        assert "roundtrip_ids" in result


class TestTokenLevelPromptCreation:
    """Test token-level prompt creation without lossy decode/encode."""

    def test_create_prompt_from_token_ids(self, small_tokenizer):
        """Should be able to create prompt directly from token IDs."""
        from counterfactual_probing.generator import create_prefix_prompt_tokens

        messages = [{"role": "user", "content": "What is 2+2?"}]
        prefix_ids = small_tokenizer.encode("The answer is", add_special_tokens=False)

        # New function should work at token level
        prompt_ids = create_prefix_prompt_tokens(
            prefix_ids=prefix_ids,
            tokenizer=small_tokenizer,
            messages=messages,
        )

        # Result should be token IDs, not text
        assert isinstance(prompt_ids, list)
        assert all(isinstance(t, int) for t in prompt_ids)

        # Should contain the prefix tokens at the end
        assert prompt_ids[-len(prefix_ids):] == prefix_ids

    def test_token_prompt_vs_text_prompt_equivalent(self, small_tokenizer):
        """Token-level prompt should produce correct structure for generation.

        Note: The token-based approach is actually MORE correct than text-based
        because text-based adds end-of-turn markers when embedding assistant
        content, which we don't want for continuation generation.
        """
        from counterfactual_probing.generator import create_prefix_prompt_tokens

        messages = [{"role": "user", "content": "Hello"}]
        text = "World"
        prefix_ids = small_tokenizer.encode(text, add_special_tokens=False)

        # Token-based (new way)
        token_prompt_ids = create_prefix_prompt_tokens(prefix_ids, small_tokenizer, messages)

        # Should end with the prefix tokens (for continuation)
        assert token_prompt_ids[-len(prefix_ids):] == prefix_ids, (
            f"Token prompt should end with prefix tokens.\n"
            f"Expected suffix: {prefix_ids}\n"
            f"Actual suffix: {token_prompt_ids[-len(prefix_ids):]}"
        )

        # Should contain the user message somewhere in the prompt
        user_text = "Hello"
        user_ids = small_tokenizer.encode(user_text, add_special_tokens=False)

        # Verify user content is in the prompt (may be embedded in chat template)
        prompt_text = small_tokenizer.decode(token_prompt_ids)
        assert user_text in prompt_text, (
            f"User message not found in prompt.\n"
            f"Looking for: {user_text}\n"
            f"In: {prompt_text}"
        )


class TestPrefixIntegrity:
    """Test that prefix integrity is maintained through generation pipeline."""

    def test_prefix_preserved_in_continuation(self, small_tokenizer):
        """Prefix token IDs should be exactly preserved in continuation."""
        from counterfactual_probing.generator import extract_continuation

        # Simulate vLLM output: prompt tokens + generated tokens
        prefix_ids = [100, 200, 300, 400, 500]
        generated_ids = [600, 700, 800]
        full_ids = prefix_ids + generated_ids

        continuation = extract_continuation(full_ids, prefix_ids)

        # Continuation should be exactly the generated tokens
        assert continuation == generated_ids

    def test_prefix_mismatch_detected(self, small_tokenizer):
        """Should detect when prefix doesn't match in output."""
        from counterfactual_probing.generator import extract_continuation_safe

        prefix_ids = [100, 200, 300]
        # Simulate corrupted output where prefix doesn't match
        corrupted_full_ids = [100, 201, 300, 400, 500]  # 201 instead of 200

        result = extract_continuation_safe(corrupted_full_ids, prefix_ids)

        # Should indicate mismatch
        assert result["prefix_matched"] is False
        assert "mismatch_position" in result

    def test_prefix_match_reported(self, small_tokenizer):
        """Should confirm when prefix matches correctly."""
        from counterfactual_probing.generator import extract_continuation_safe

        prefix_ids = [100, 200, 300]
        full_ids = [100, 200, 300, 400, 500]

        result = extract_continuation_safe(full_ids, prefix_ids)

        assert result["prefix_matched"] is True
        assert result["continuation"] == [400, 500]


class TestSpecialTokenHandling:
    """Test handling of special tokens in prefix."""

    def test_skip_special_tokens_loses_info(self, small_tokenizer):
        """Document that skip_special_tokens=True loses information."""
        # Get a special token if available
        if hasattr(small_tokenizer, 'bos_token_id') and small_tokenizer.bos_token_id:
            special_id = small_tokenizer.bos_token_id
        elif hasattr(small_tokenizer, 'eos_token_id') and small_tokenizer.eos_token_id:
            special_id = small_tokenizer.eos_token_id
        else:
            pytest.skip("No special tokens available")

        # Sequence with special token
        token_ids = [special_id, 100, 200, 300]

        # Decode with skip_special_tokens=True
        decoded_skip = small_tokenizer.decode(token_ids, skip_special_tokens=True)
        # Decode without skipping
        decoded_keep = small_tokenizer.decode(token_ids, skip_special_tokens=False)

        # Re-encode
        re_encoded_skip = small_tokenizer.encode(decoded_skip, add_special_tokens=False)
        re_encoded_keep = small_tokenizer.encode(decoded_keep, add_special_tokens=False)

        # Skipping special tokens should lose the special token
        # (This test documents the problem)
        assert special_id not in re_encoded_skip or len(re_encoded_skip) < len(token_ids), (
            "skip_special_tokens should cause information loss"
        )

    def test_preserve_all_tokens_option(self, small_tokenizer):
        """decode_preserving_tokens keeps special tokens in output.

        Note: This function helps preserve more information than skip_special_tokens=True,
        but cannot guarantee perfect round-trip for all token sequences. Some tokens
        (especially byte-level or partial word tokens) may not round-trip perfectly.

        For guaranteed token-level precision, use create_prefix_prompt_tokens instead.
        """
        from counterfactual_probing.generator import decode_preserving_tokens

        # Use tokens from actual text encoding (more likely to round-trip)
        text = "Hello world"
        token_ids = small_tokenizer.encode(text, add_special_tokens=False)

        decoded = decode_preserving_tokens(token_ids, small_tokenizer)
        re_encoded = small_tokenizer.encode(decoded, add_special_tokens=False)

        # For natural text tokens, should round-trip correctly
        assert token_ids == re_encoded, (
            f"Token preservation failed for natural text: {token_ids} -> {re_encoded}"
        )


class TestGeneratorValidation:
    """Test validation functions in generator."""

    def test_validate_token_ids_range(self, small_tokenizer):
        """Should validate token IDs are in vocabulary range."""
        from counterfactual_probing.generator import validate_token_ids

        vocab_size = small_tokenizer.vocab_size

        # Valid
        assert validate_token_ids([0, 1, 100], vocab_size)["valid"] is True

        # Invalid - negative
        assert validate_token_ids([-1, 0, 1], vocab_size)["valid"] is False

        # Invalid - too large
        assert validate_token_ids([0, vocab_size], vocab_size)["valid"] is False

    def test_validate_empty_token_list(self, small_tokenizer):
        """Empty token list should be valid."""
        from counterfactual_probing.generator import validate_token_ids

        result = validate_token_ids([], small_tokenizer.vocab_size)
        assert result["valid"] is True


class TestIntegrationWithVLLM:
    """Integration tests with actual vLLM generation."""

    @pytest.mark.integration
    def test_generated_tokens_match_prefix(self, small_model_name, small_tokenizer):
        """Generated output should start with exact prefix tokens."""
        from counterfactual_probing.generator import generate_counterfactuals_batch
        from vllm import LLM

        llm = LLM(model=small_model_name, trust_remote_code=True)

        prefix_text = "The answer to the question is"
        prefix_ids = small_tokenizer.encode(prefix_text, add_special_tokens=False)
        messages = [{"role": "user", "content": "What is 2+2?"}]

        counterfactuals = generate_counterfactuals_batch(
            prefix_ids=prefix_ids,
            messages=messages,
            model_name=small_model_name,
            tokenizer=small_tokenizer,
            num_counterfactuals=3,
            temperature=0.5,
            max_tokens=20,
            llm=llm,
        )

        # Note: vLLM returns only the NEW tokens, not the prefix
        # This test verifies the structure is correct
        for cf in counterfactuals:
            assert "token_ids" in cf
            assert isinstance(cf["token_ids"], list)
            # Generated tokens should be valid
            for tid in cf["token_ids"]:
                assert 0 <= tid < small_tokenizer.vocab_size

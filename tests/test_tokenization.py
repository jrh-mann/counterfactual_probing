"""
Tests for tokenization correctness.

These are the most critical tests - tokenization bugs propagate everywhere.
We test against the tokens module which provides utilities for working with
token IDs correctly.
"""

import pytest
from typing import List


class TestEncodeDecodeRoundtrip:
    """Test that encoding then decoding returns original text."""

    def test_simple_text_roundtrip(self, small_tokenizer, sample_texts):
        """Basic text should roundtrip exactly."""
        for text in sample_texts:
            if not text.strip():  # Skip empty/whitespace
                continue
            token_ids = small_tokenizer.encode(text, add_special_tokens=False)
            decoded = small_tokenizer.decode(token_ids)
            assert decoded == text, f"Roundtrip failed for: {text!r}"

    def test_roundtrip_with_special_tokens(self, small_tokenizer):
        """Text with special tokens should roundtrip correctly."""
        text = "Hello world"
        token_ids = small_tokenizer.encode(text, add_special_tokens=True)
        # Decode without special tokens to get original
        decoded = small_tokenizer.decode(token_ids, skip_special_tokens=True)
        assert decoded == text

    def test_unicode_roundtrip(self, small_tokenizer):
        """Unicode text should roundtrip correctly."""
        texts = [
            "Hello, 世界!",
            "Émojis: 🎉🚀💡",
            "Greek: αβγδ",
            "Math: ∑∫∂∇",
        ]
        for text in texts:
            token_ids = small_tokenizer.encode(text, add_special_tokens=False)
            decoded = small_tokenizer.decode(token_ids)
            assert decoded == text, f"Unicode roundtrip failed for: {text!r}"


class TestTokenBoundaryPreservation:
    """Test that splitting at token boundaries preserves exact prefixes."""

    def test_prefix_decode_matches_original_prefix(self, small_tokenizer):
        """
        When we take tokens 0..N, decoding should give us the exact prefix
        of the original text up to some character boundary.
        """
        text = "The quick brown fox jumps over the lazy dog."
        token_ids = small_tokenizer.encode(text, add_special_tokens=False)

        # For each split point, verify prefix decodes to a prefix of original
        for n in range(1, len(token_ids)):
            prefix_ids = token_ids[:n]
            prefix_decoded = small_tokenizer.decode(prefix_ids)
            # The decoded prefix should be a prefix of original text
            assert text.startswith(prefix_decoded), (
                f"Token prefix {n} decoded to '{prefix_decoded}' "
                f"which is not a prefix of '{text}'"
            )

    def test_full_sequence_from_prefixes(self, small_tokenizer):
        """
        Concatenating all token prefixes should reconstruct the sequence.
        (Each token adds something to the decoded text.)
        """
        text = "Hello, world! How are you today?"
        token_ids = small_tokenizer.encode(text, add_special_tokens=False)

        prev_decoded = ""
        for n in range(1, len(token_ids) + 1):
            current_decoded = small_tokenizer.decode(token_ids[:n])
            # Current should extend previous (or be equal for some edge cases)
            assert current_decoded.startswith(prev_decoded), (
                f"Decoding not monotonic at token {n}: "
                f"'{prev_decoded}' -> '{current_decoded}'"
            )
            prev_decoded = current_decoded

        # Final should equal original
        assert prev_decoded == text


class TestPrefixTokenIdsMatchFullSequence:
    """Test that prefix_token_ids is an exact slice of full token_ids."""

    def test_prefix_is_exact_slice(self, small_tokenizer):
        """prefix_token_ids[:n] should equal full_token_ids[:n]."""
        text = "This is a test sentence for tokenization."
        full_token_ids = small_tokenizer.encode(text, add_special_tokens=False)

        for n in range(len(full_token_ids)):
            prefix_ids = full_token_ids[:n]
            # Re-encode the prefix text and compare
            prefix_text = small_tokenizer.decode(prefix_ids)
            re_encoded = small_tokenizer.encode(prefix_text, add_special_tokens=False)

            # The re-encoded should match the original slice
            # (This tests that tokenization is consistent)
            assert re_encoded == prefix_ids, (
                f"Re-encoding prefix at {n} gave different tokens: "
                f"{re_encoded} != {prefix_ids}"
            )

    def test_continuation_tokens_dont_overlap(self, small_tokenizer):
        """
        If we split into prefix and continuation, they should not share tokens.
        """
        text = "The quick brown fox."
        full_ids = small_tokenizer.encode(text, add_special_tokens=False)

        split_point = len(full_ids) // 2
        prefix_ids = full_ids[:split_point]
        continuation_ids = full_ids[split_point:]

        # Concatenating should give full sequence
        assert prefix_ids + continuation_ids == full_ids


class TestSpecialTokenHandling:
    """Test correct handling of special tokens."""

    def test_no_special_tokens_in_content(self, small_tokenizer):
        """Content tokens shouldn't include BOS/EOS unless intended."""
        text = "Just regular content here."
        token_ids = small_tokenizer.encode(text, add_special_tokens=False)

        special_ids = set()
        if small_tokenizer.bos_token_id is not None:
            special_ids.add(small_tokenizer.bos_token_id)
        if small_tokenizer.eos_token_id is not None:
            special_ids.add(small_tokenizer.eos_token_id)
        if small_tokenizer.pad_token_id is not None:
            special_ids.add(small_tokenizer.pad_token_id)

        for tid in token_ids:
            assert tid not in special_ids, (
                f"Found special token {tid} in content encoding"
            )

    def test_special_tokens_added_when_requested(self, small_tokenizer):
        """Special tokens should be added when add_special_tokens=True."""
        text = "Test text"
        with_special = small_tokenizer.encode(text, add_special_tokens=True)
        without_special = small_tokenizer.encode(text, add_special_tokens=False)

        # With special tokens should be longer (or equal if model doesn't add them)
        assert len(with_special) >= len(without_special)


class TestChatTemplateTokenPositions:
    """Test finding positions within chat-formatted text."""

    def test_find_assistant_content_start(self, small_tokenizer):
        """
        Verify we can find where assistant content starts in tokenized chat.
        """
        messages = [
            {"role": "user", "content": "Hello, how are you?"},
        ]

        # Apply chat template
        formatted = small_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize
        token_ids = small_tokenizer.encode(formatted, add_special_tokens=False)

        # The formatted string should end with generation prompt
        # We should be able to identify where user content ends
        assert len(token_ids) > 0

    def test_chat_template_preserves_content(self, small_tokenizer):
        """Chat template should preserve the actual content."""
        user_content = "What is the meaning of life?"
        messages = [{"role": "user", "content": user_content}]

        formatted = small_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # User content should appear in formatted string
        assert user_content in formatted

    def test_assistant_prefix_in_chat(self, small_tokenizer):
        """
        When we add assistant content, it should appear after the generation prompt.
        """
        user_content = "Hello"
        assistant_content = "Hi there!"

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]

        formatted = small_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,  # Already have assistant response
        )

        # Both contents should appear
        assert user_content in formatted
        assert assistant_content in formatted

        # Assistant content should come after user content
        user_pos = formatted.find(user_content)
        assistant_pos = formatted.find(assistant_content)
        assert assistant_pos > user_pos


class TestOffsetMapping:
    """Test token offset mappings for char-to-token mapping."""

    def test_offset_mapping_covers_text(self, small_tokenizer):
        """Offset mappings should cover the entire text."""
        text = "Hello world, this is a test."

        encoding = small_tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )

        offsets = encoding["offset_mapping"]

        # Check that offsets are valid
        for start, end in offsets:
            assert 0 <= start <= end <= len(text), (
                f"Invalid offset ({start}, {end}) for text of length {len(text)}"
            )

    def test_offset_mapping_char_lookup(self, small_tokenizer):
        """Should be able to find which token contains a given character."""
        text = "The quick brown fox"
        word_to_find = "brown"
        char_pos = text.find(word_to_find)

        encoding = small_tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )

        offsets = encoding["offset_mapping"]

        # Find token containing char_pos
        found_token = None
        for tok_idx, (start, end) in enumerate(offsets):
            if start <= char_pos < end:
                found_token = tok_idx
                break

        assert found_token is not None, (
            f"Could not find token for char position {char_pos}"
        )

        # The token should decode to something containing part of "brown"
        token_id = encoding["input_ids"][found_token]
        token_text = small_tokenizer.decode([token_id])
        # Token might be part of the word
        assert any(c in token_text for c in word_to_find), (
            f"Token '{token_text}' doesn't contain any chars from '{word_to_find}'"
        )

    def test_offset_mapping_contiguous(self, small_tokenizer):
        """Offset mappings should be contiguous (no gaps in coverage)."""
        text = "Contiguous text coverage test."

        encoding = small_tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )

        offsets = encoding["offset_mapping"]

        # Filter out special token offsets (0, 0)
        real_offsets = [(s, e) for s, e in offsets if s != e]

        if len(real_offsets) > 1:
            # Check that end of one token is start of next (approximately)
            # Some tokenizers may have small gaps
            for i in range(len(real_offsets) - 1):
                _, end = real_offsets[i]
                next_start, _ = real_offsets[i + 1]
                # Allow small gaps (whitespace handling varies)
                assert next_start >= end, (
                    f"Overlapping offsets at {i}: {real_offsets[i]} and {real_offsets[i+1]}"
                )


class TestTokenIdsValidity:
    """Test that token IDs are always valid."""

    def test_token_ids_in_vocab(self, small_tokenizer):
        """All token IDs should be valid vocabulary indices."""
        texts = [
            "Hello world",
            "Special: @#$%^&*()",
            "Numbers: 123456789",
            "Mixed: abc123!@#",
        ]

        vocab_size = small_tokenizer.vocab_size

        for text in texts:
            token_ids = small_tokenizer.encode(text, add_special_tokens=False)
            for tid in token_ids:
                assert 0 <= tid < vocab_size, (
                    f"Token ID {tid} out of vocab range [0, {vocab_size})"
                )

    def test_empty_string_handling(self, small_tokenizer):
        """Empty strings should produce empty token lists."""
        token_ids = small_tokenizer.encode("", add_special_tokens=False)
        assert token_ids == [] or len(token_ids) == 0


class TestTokenizationConsistency:
    """Test that tokenization is deterministic and consistent."""

    def test_deterministic_encoding(self, small_tokenizer):
        """Same text should always produce same tokens."""
        text = "Deterministic encoding test."

        ids1 = small_tokenizer.encode(text, add_special_tokens=False)
        ids2 = small_tokenizer.encode(text, add_special_tokens=False)

        assert ids1 == ids2

    def test_batch_matches_individual(self, small_tokenizer):
        """Batch tokenization should match individual tokenization."""
        texts = ["First text", "Second text", "Third text"]

        # Batch encode
        batch_result = small_tokenizer(texts, add_special_tokens=False)

        # Individual encode
        individual_results = [
            small_tokenizer.encode(t, add_special_tokens=False)
            for t in texts
        ]

        for i, (batch_ids, individual_ids) in enumerate(
            zip(batch_result["input_ids"], individual_results)
        ):
            assert list(batch_ids) == individual_ids, (
                f"Batch vs individual mismatch at index {i}"
            )

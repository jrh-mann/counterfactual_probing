"""
Tests for activation extraction.

These tests verify that activations are correctly extracted from models
and aligned with token positions.
"""

import pytest
import torch


@pytest.mark.integration
class TestActivationShape:
    """Test that activations have correct shape."""

    def test_activation_dimensions(self, small_tokenizer, small_model_name):
        """Activations should be (num_layers, num_tokens, hidden_dim)."""
        from counterfactual_probing.activations import extract_activations

        text = "Hello, world!"
        token_ids = small_tokenizer.encode(text, add_special_tokens=False)

        activations = extract_activations(
            text=text,
            model_name=small_model_name,
            tokenizer=small_tokenizer,
        )

        assert len(activations.shape) == 3
        num_layers, num_tokens, hidden_dim = activations.shape

        # Should have some layers
        assert num_layers > 0

        # Token count should match
        assert num_tokens == len(token_ids)

        # Hidden dim should be positive
        assert hidden_dim > 0

    def test_activation_at_positions(self, small_tokenizer, small_model_name):
        """Should extract activations only at requested positions."""
        from counterfactual_probing.activations import extract_activations

        text = "This is a test sentence for activation extraction."
        token_ids = small_tokenizer.encode(text, add_special_tokens=False)

        positions = [0, 3, len(token_ids) - 1]

        activations = extract_activations(
            text=text,
            model_name=small_model_name,
            tokenizer=small_tokenizer,
            positions=positions,
        )

        num_layers, num_tokens, hidden_dim = activations.shape

        # Should only have activations at requested positions
        assert num_tokens == len(positions)


@pytest.mark.integration
class TestActivationAlignment:
    """Test that activations are aligned with tokens."""

    def test_activation_position_correspondence(self, small_tokenizer, small_model_name):
        """Activation at position i should correspond to token i."""
        from counterfactual_probing.activations import extract_activations

        text = "Test alignment"
        token_ids = small_tokenizer.encode(text, add_special_tokens=False)

        activations = extract_activations(
            text=text,
            model_name=small_model_name,
            tokenizer=small_tokenizer,
        )

        # Each activation position should map to a token
        assert activations.shape[1] == len(token_ids)


@pytest.mark.integration
class TestActivationDeterminism:
    """Test that activation extraction is deterministic."""

    def test_same_input_same_output(self, small_tokenizer, small_model_name):
        """Same input should produce identical activations."""
        from counterfactual_probing.activations import extract_activations

        text = "Deterministic test"

        activations1 = extract_activations(
            text=text,
            model_name=small_model_name,
            tokenizer=small_tokenizer,
        )

        activations2 = extract_activations(
            text=text,
            model_name=small_model_name,
            tokenizer=small_tokenizer,
        )

        assert torch.allclose(activations1, activations2)


@pytest.mark.integration
class TestActivationTypes:
    """Test activation data types."""

    def test_activations_are_float(self, small_tokenizer, small_model_name):
        """Activations should be floating point."""
        from counterfactual_probing.activations import extract_activations

        text = "Float test"

        activations = extract_activations(
            text=text,
            model_name=small_model_name,
            tokenizer=small_tokenizer,
        )

        assert activations.dtype in [torch.float32, torch.float16, torch.bfloat16]

    def test_activations_finite(self, small_tokenizer, small_model_name):
        """Activations should not contain NaN or Inf."""
        from counterfactual_probing.activations import extract_activations

        text = "Finite values test"

        activations = extract_activations(
            text=text,
            model_name=small_model_name,
            tokenizer=small_tokenizer,
        )

        assert torch.isfinite(activations).all()


class TestActivationInputValidation:
    """Test input validation for activation extraction (no model needed)."""

    def test_empty_text_handling(self, small_tokenizer):
        """Empty text should be handled gracefully."""
        from counterfactual_probing.activations import validate_extraction_input

        # Should either raise clear error or return empty
        result = validate_extraction_input(
            text="",
            tokenizer=small_tokenizer,
        )

        assert result["valid"] is False
        assert "empty" in result["error"].lower()

    def test_positions_out_of_range(self, small_tokenizer):
        """Positions outside token range should raise error."""
        from counterfactual_probing.activations import validate_extraction_input

        text = "Short"
        token_ids = small_tokenizer.encode(text, add_special_tokens=False)

        result = validate_extraction_input(
            text=text,
            tokenizer=small_tokenizer,
            positions=[0, 100],  # 100 is out of range
        )

        assert result["valid"] is False
        assert "range" in result["error"].lower()

    def test_negative_positions(self, small_tokenizer):
        """Negative positions should raise error."""
        from counterfactual_probing.activations import validate_extraction_input

        result = validate_extraction_input(
            text="Test",
            tokenizer=small_tokenizer,
            positions=[-1, 0],
        )

        assert result["valid"] is False

    def test_valid_input(self, small_tokenizer):
        """Valid input should pass validation."""
        from counterfactual_probing.activations import validate_extraction_input

        text = "Valid input test"
        token_ids = small_tokenizer.encode(text, add_special_tokens=False)

        result = validate_extraction_input(
            text=text,
            tokenizer=small_tokenizer,
            positions=[0, len(token_ids) - 1],
        )

        assert result["valid"] is True


class TestActivationNnsightInput:
    """Test that the exact string passed to nnsight is correct."""

    def test_formatted_string_matches_expected(self, small_tokenizer):
        """Verify the formatted string that would be passed to nnsight."""
        from counterfactual_probing.activations import prepare_nnsight_input

        messages = [
            {"role": "user", "content": "Hello"},
        ]
        assistant_content = "Hi there!"

        formatted = prepare_nnsight_input(
            messages=messages,
            assistant_content=assistant_content,
            tokenizer=small_tokenizer,
        )

        # Should contain both user and assistant content
        assert "Hello" in formatted
        assert "Hi there!" in formatted

        # Should be tokenizable
        token_ids = small_tokenizer.encode(formatted, add_special_tokens=False)
        assert len(token_ids) > 0

    def test_chat_template_applied(self, small_tokenizer):
        """Chat template should be applied correctly."""
        from counterfactual_probing.activations import prepare_nnsight_input

        messages = [
            {"role": "user", "content": "Test message"},
        ]

        formatted = prepare_nnsight_input(
            messages=messages,
            assistant_content="Response",
            tokenizer=small_tokenizer,
        )

        # Should match applying chat template directly
        expected = small_tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": "Response"}],
            tokenize=False,
            add_generation_prompt=False,
        )

        assert formatted == expected


class TestActivationSaving:
    """Test saving and loading activations."""

    def test_save_and_load_roundtrip(self, tmp_path):
        """Saved activations should load correctly."""
        from counterfactual_probing.activations import save_activations, load_activations

        # Create dummy activations
        activations = torch.randn(24, 10, 768)  # 24 layers, 10 tokens, 768 hidden
        token_ids = list(range(10))
        metadata = {"text": "test", "positions": [0, 5, 9]}

        save_path = tmp_path / "activations.pt"

        save_activations(
            activations=activations,
            token_ids=token_ids,
            metadata=metadata,
            path=save_path,
        )

        loaded = load_activations(save_path)

        assert torch.allclose(loaded["activations"], activations)
        assert loaded["token_ids"] == token_ids
        assert loaded["metadata"] == metadata

    def test_load_nonexistent_raises(self, tmp_path):
        """Loading nonexistent file should raise error."""
        from counterfactual_probing.activations import load_activations

        with pytest.raises(FileNotFoundError):
            load_activations(tmp_path / "nonexistent.pt")


@pytest.mark.integration
class TestLayerSelection:
    """Test extracting specific layers."""

    def test_single_layer_extraction(self, small_tokenizer, small_model_name):
        """Should extract activations from specific layer."""
        from counterfactual_probing.activations import extract_activations

        text = "Layer selection test"

        # Extract single layer
        activations = extract_activations(
            text=text,
            model_name=small_model_name,
            tokenizer=small_tokenizer,
            layers=[5],  # Just layer 5
        )

        # Should have only 1 layer
        assert activations.shape[0] == 1

    def test_multiple_layer_extraction(self, small_tokenizer, small_model_name):
        """Should extract activations from multiple specific layers."""
        from counterfactual_probing.activations import extract_activations

        text = "Multiple layers test"

        # Extract specific layers
        layers = [0, 5, 10]
        activations = extract_activations(
            text=text,
            model_name=small_model_name,
            tokenizer=small_tokenizer,
            layers=layers,
        )

        # Should have only requested layers
        assert activations.shape[0] == len(layers)

    def test_all_layers_extraction(self, small_tokenizer, small_model_name):
        """Should extract all layers by default."""
        from counterfactual_probing.activations import extract_activations

        text = "All layers test"

        activations = extract_activations(
            text=text,
            model_name=small_model_name,
            tokenizer=small_tokenizer,
            # No layers specified - should get all
        )

        # Should have multiple layers
        assert activations.shape[0] > 1

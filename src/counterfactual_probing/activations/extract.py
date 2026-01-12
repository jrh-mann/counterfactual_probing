"""
Activation extraction using nnsight.

Provides utilities for extracting activations from transformer models
at specific token positions.
"""

from typing import Any

import torch
from transformers import PreTrainedTokenizer


def validate_extraction_input(
    text: str,
    tokenizer: PreTrainedTokenizer,
    positions: list[int] | None = None,
) -> dict[str, Any]:
    """
    Validate inputs before extraction.

    Args:
        text: Text to extract activations from
        tokenizer: Tokenizer for the model
        positions: Optional specific positions to extract

    Returns:
        Dict with 'valid' bool and 'error' message if invalid
    """
    if not text or not text.strip():
        return {"valid": False, "error": "Text is empty"}

    # Tokenize to check length
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    n_tokens = len(token_ids)

    if n_tokens == 0:
        return {"valid": False, "error": "Text produces no tokens"}

    if positions is not None:
        for pos in positions:
            if pos < 0:
                return {"valid": False, "error": f"Negative position: {pos}"}
            if pos >= n_tokens:
                return {
                    "valid": False,
                    "error": f"Position {pos} out of range [0, {n_tokens})"
                }

    return {"valid": True, "error": None, "n_tokens": n_tokens}


def prepare_nnsight_input(
    messages: list[dict[str, str]],
    assistant_content: str,
    tokenizer: PreTrainedTokenizer,
) -> str:
    """
    Prepare the formatted string that will be passed to nnsight.

    Args:
        messages: Chat messages (user/system)
        assistant_content: Content from the assistant response
        tokenizer: Tokenizer with chat template

    Returns:
        Formatted string ready for model input
    """
    full_messages = messages.copy()
    full_messages.append({"role": "assistant", "content": assistant_content})

    formatted = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return formatted


def extract_activations(
    text: str,
    model_name: str,
    tokenizer: PreTrainedTokenizer,
    positions: list[int] | None = None,
    layers: list[int] | None = None,
    device: str = "auto",
    model: Any | None = None,
) -> torch.Tensor:
    """
    Extract activations from a model for given text.

    Args:
        text: Text to extract activations from
        model_name: Name/path of the model
        tokenizer: Tokenizer for the model
        positions: Specific token positions to extract (None = all)
        layers: Specific layers to extract (None = all)
        device: Device to run on ("auto", "cuda", "cpu", "mps")
        model: Optional pre-loaded nnsight model

    Returns:
        Tensor of shape (num_layers, num_positions, hidden_dim)
    """
    from nnsight import LanguageModel

    # Load model if not provided
    if model is None:
        model = LanguageModel(
            model_name,
            device_map=device,
            trust_remote_code=True,
        )

    # Tokenize the text
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    n_tokens = len(token_ids)

    # Determine positions to extract
    if positions is None:
        positions = list(range(n_tokens))

    # Run forward pass with nnsight tracing
    with torch.no_grad(), model.trace(text):
        # Collect activations from each layer
        saved_outputs = []

        if layers is None:
            # Get all layers
            for layer in model.model.layers:
                saved_outputs.append(layer.output[0].save())
        else:
            # Get specific layers
            for layer_idx in layers:
                saved_outputs.append(
                    model.model.layers[layer_idx].output[0].save()
                )

    # Stack layer outputs
    # Each saved output is (batch=1, seq_len, hidden_dim)
    layer_tensors = [out.value.squeeze(0) for out in saved_outputs]
    all_activations = torch.stack(layer_tensors, dim=0)  # (num_layers, seq_len, hidden_dim)

    # Extract only requested positions
    if positions != list(range(n_tokens)):
        all_activations = all_activations[:, positions, :]

    return all_activations.cpu()


def extract_activations_at_token_indices(
    formatted_text: str,
    token_indices: list[int],
    model_name: str,
    tokenizer: PreTrainedTokenizer,
    layers: list[int] | None = None,
    model: Any | None = None,
) -> torch.Tensor:
    """
    Extract activations at specific token indices.

    This is a more precise version that works with exact token indices
    rather than inferring positions from text.

    Args:
        formatted_text: Full formatted text (with chat template)
        token_indices: Indices of tokens to extract activations for
        model_name: Name/path of the model
        tokenizer: Tokenizer for the model
        layers: Specific layers to extract (None = all)
        model: Optional pre-loaded nnsight model

    Returns:
        Tensor of shape (num_layers, len(token_indices), hidden_dim)
    """
    from nnsight import LanguageModel

    # Load model if not provided
    if model is None:
        model = LanguageModel(
            model_name,
            device_map="auto",
            trust_remote_code=True,
        )

    # Run forward pass with nnsight tracing
    with torch.no_grad(), model.trace(formatted_text):
        saved_outputs = []

        if layers is None:
            for layer in model.model.layers:
                saved_outputs.append(layer.output[0].save())
        else:
            for layer_idx in layers:
                saved_outputs.append(
                    model.model.layers[layer_idx].output[0].save()
                )

    # Stack and extract
    layer_tensors = [out.value.squeeze(0) for out in saved_outputs]
    all_activations = torch.stack(layer_tensors, dim=0)

    # Extract at specific indices
    all_activations = all_activations[:, token_indices, :]

    return all_activations.cpu()

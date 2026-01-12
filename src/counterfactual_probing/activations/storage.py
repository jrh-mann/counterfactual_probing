"""
Activation storage utilities.

Functions for saving and loading activations to/from disk.
"""

import torch
from pathlib import Path
from typing import Dict, Any, List, Union


def save_activations(
    activations: torch.Tensor,
    token_ids: List[int],
    metadata: Dict[str, Any],
    path: Union[str, Path],
) -> None:
    """
    Save activations to disk.

    Args:
        activations: Activation tensor (num_layers, num_tokens, hidden_dim)
        token_ids: Token IDs corresponding to activations
        metadata: Additional metadata to save
        path: Path to save to
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "activations": activations,
        "token_ids": token_ids,
        "metadata": metadata,
    }

    torch.save(data, path)


def load_activations(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load activations from disk.

    Args:
        path: Path to load from

    Returns:
        Dict with 'activations', 'token_ids', and 'metadata'

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Activation file not found: {path}")

    return torch.load(path, map_location="cpu")


def load_activations_batch(
    paths: List[Union[str, Path]],
    stack: bool = True,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Load multiple activation files.

    Args:
        paths: List of paths to load
        stack: If True, stack into single tensor

    Returns:
        Stacked tensor or list of tensors
    """
    activations = []

    for path in paths:
        data = load_activations(path)
        activations.append(data["activations"])

    if stack:
        return torch.stack(activations, dim=0)

    return activations

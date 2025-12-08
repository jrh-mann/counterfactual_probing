"""
Utilities for loading activations from saved .pt files for analysis.

Designed for use in Jupyter notebooks.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import torch
import numpy as np


def load_single_activation(activation_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load a single activation file.
    
    Args:
        activation_path: Path to .pt file
        
    Returns:
        Tuple of (token_ids, activations)
        - token_ids: 1D tensor of token IDs
        - activations: 2D tensor of shape (num_layers, hidden_dim)
    """
    data = torch.load(activation_path, map_location='cpu')
    if not isinstance(data, tuple) or len(data) != 2:
        raise ValueError(f"Expected tuple of (token_ids, activations), got {type(data)}")
    token_ids, activations = data
    return token_ids, activations


def load_metadata(metadata_path: Path) -> Dict[str, Any]:
    """Load metadata JSON file."""
    with open(metadata_path, 'r') as f:
        return json.load(f)


def load_prompt_activations(
    activations_dir: Path,
    include_token_ids: bool = True
) -> Dict[str, Any]:
    """
    Load all activations for a single prompt.
    
    Args:
        activations_dir: Directory containing sentence_*.pt files and metadata.json
        include_token_ids: Whether to include token_ids in the output
        
    Returns:
        Dictionary with keys:
        - 'metadata': Metadata dict from metadata.json
        - 'activations': Dict mapping sentence_index -> activations tensor (num_layers, hidden_dim)
        - 'token_ids': Dict mapping sentence_index -> token_ids tensor (if include_token_ids=True)
        - 'sentence_data': List of sentence data dicts from metadata
    """
    activations_dir = Path(activations_dir)
    
    # Load metadata
    metadata_path = activations_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    metadata = load_metadata(metadata_path)
    
    # Load all activation files
    activation_files = sorted(activations_dir.glob("sentence_*.pt"))
    
    result = {
        'metadata': metadata,
        'activations': {},
        'sentence_data': metadata.get('sentence_data', [])
    }
    
    if include_token_ids:
        result['token_ids'] = {}
    
    for activation_file in activation_files:
        # Extract sentence index from filename
        sentence_idx = int(activation_file.stem.split('_')[1])
        
        # Load activation
        token_ids, activations = load_single_activation(activation_file)
        
        result['activations'][sentence_idx] = activations
        if include_token_ids:
            result['token_ids'][sentence_idx] = token_ids
    
    return result


def load_all_activations(
    base_dir: Path,
    include_token_ids: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Load activations for all prompts in a directory.
    
    Args:
        base_dir: Base directory containing prompt subdirectories
        include_token_ids: Whether to include token_ids in the output
        
    Returns:
        Dictionary mapping prompt_name -> prompt_activations dict
        (see load_prompt_activations for structure)
    """
    base_dir = Path(base_dir)
    
    # Find all prompt directories (directories containing metadata.json)
    prompt_dirs = []
    for item in base_dir.iterdir():
        if item.is_dir():
            metadata_path = item / "metadata.json"
            if metadata_path.exists():
                prompt_dirs.append(item)
    
    result = {}
    for prompt_dir in sorted(prompt_dirs):
        prompt_name = prompt_dir.name
        result[prompt_name] = load_prompt_activations(prompt_dir, include_token_ids=include_token_ids)
    
    return result


def stack_activations_by_layer(
    prompt_activations: Dict[str, Any],
    layer_idx: Optional[int] = None
) -> np.ndarray:
    """
    Stack activations across sentences for a single prompt.
    
    Args:
        prompt_activations: Output from load_prompt_activations
        layer_idx: Specific layer to extract (None = all layers)
        
    Returns:
        If layer_idx is None: (num_sentences, num_layers, hidden_dim)
        If layer_idx is specified: (num_sentences, hidden_dim)
    """
    activations_dict = prompt_activations['activations']
    sentence_indices = sorted(activations_dict.keys())
    
    # Get first activation to determine shape
    first_activation = activations_dict[sentence_indices[0]]
    num_layers, hidden_dim = first_activation.shape
    
    if layer_idx is not None:
        if layer_idx < 0 or layer_idx >= num_layers:
            raise ValueError(f"layer_idx {layer_idx} out of range [0, {num_layers})")
        stacked = np.stack([
            activations_dict[idx][layer_idx].float().numpy() 
            for idx in sentence_indices
        ])
        return stacked  # (num_sentences, hidden_dim)
    else:
        stacked = np.stack([
            activations_dict[idx].float().numpy() 
            for idx in sentence_indices
        ])
        return stacked  # (num_sentences, num_layers, hidden_dim)


def get_activation_matrix(
    all_activations: Dict[str, Dict[str, Any]],
    layer_idx: Optional[int] = None,
    prompt_names: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Create a matrix of activations across all prompts and sentences.
    
    Args:
        all_activations: Output from load_all_activations
        layer_idx: Specific layer to extract (None = all layers)
        prompt_names: List of prompt names to include (None = all)
        
    Returns:
        Tuple of (activation_matrix, metadata_list)
        - activation_matrix: (total_sentences, num_layers, hidden_dim) or (total_sentences, hidden_dim)
        - metadata_list: List of dicts with keys: prompt_name, sentence_index, sentence, p_reward_hacks
    """
    if prompt_names is None:
        prompt_names = sorted(all_activations.keys())
    
    all_stacked = []
    all_metadata = []
    
    for prompt_name in prompt_names:
        if prompt_name not in all_activations:
            continue
        
        prompt_data = all_activations[prompt_name]
        stacked = stack_activations_by_layer(prompt_data, layer_idx=layer_idx)
        all_stacked.append(stacked)
        
        # Add metadata for each sentence
        for sentence_data in prompt_data['sentence_data']:
            all_metadata.append({
                'prompt_name': prompt_name,
                'sentence_index': sentence_data['sentence_index'],
                'sentence': sentence_data['sentence'],
                'p_reward_hacks': sentence_data['p_reward_hacks']
            })
    
    if not all_stacked:
        raise ValueError("No activations found")
    
    activation_matrix = np.concatenate(all_stacked, axis=0)
    return activation_matrix, all_metadata


def get_reward_hacking_labels(
    all_activations: Dict[str, Dict[str, Any]],
    prompt_names: Optional[List[str]] = None
) -> np.ndarray:
    """
    Extract reward hacking probability labels.
    
    Args:
        all_activations: Output from load_all_activations
        prompt_names: List of prompt names to include (None = all)
        
    Returns:
        Array of shape (total_sentences,) with p_reward_hacks values
    """
    if prompt_names is None:
        prompt_names = sorted(all_activations.keys())
    
    labels = []
    for prompt_name in prompt_names:
        if prompt_name not in all_activations:
            continue
        
        prompt_data = all_activations[prompt_name]
        for sentence_data in prompt_data['sentence_data']:
            labels.append(sentence_data['p_reward_hacks'])
    
    return np.array(labels)


# Convenience function for notebook use
def quick_load(
    base_dir: str = "/workspace",
    include_token_ids: bool = False,
    layer_idx: Optional[int] = None
) -> Tuple[np.ndarray, List[Dict[str, Any]], np.ndarray]:
    """
    Quick load function for notebook analysis.
    
    Args:
        base_dir: Base directory containing prompt subdirectories
        include_token_ids: Whether to include token_ids (not used in quick_load)
        layer_idx: Specific layer to extract (None = all layers)
        
    Returns:
        Tuple of (activations, metadata, labels)
        - activations: (total_sentences, num_layers, hidden_dim) or (total_sentences, hidden_dim)
        - metadata: List of metadata dicts
        - labels: (total_sentences,) array of p_reward_hacks values
    """
    base_path = Path(base_dir)
    all_activations = load_all_activations(base_path, include_token_ids=include_token_ids)
    
    activations, metadata = get_activation_matrix(all_activations, layer_idx=layer_idx)
    labels = get_reward_hacking_labels(all_activations)
    
    return activations, metadata, labels


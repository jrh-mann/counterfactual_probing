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
        - 'missing_activations': Set of sentence indices that are in metadata but don't have activation files
    """
    activations_dir = Path(activations_dir)
    
    # Load metadata
    metadata_path = activations_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    metadata = load_metadata(metadata_path)
    sentence_data = metadata.get('sentence_data', [])
    
    # Get expected sentence indices from metadata
    expected_indices = {s.get('sentence_index') for s in sentence_data if 'sentence_index' in s}
    
    # Load all activation files
    activation_files = sorted(activations_dir.glob("sentence_*.pt"))
    loaded_indices = set()
    
    result = {
        'metadata': metadata,
        'activations': {},
        'sentence_data': sentence_data
    }
    
    if include_token_ids:
        result['token_ids'] = {}
    
    for activation_file in activation_files:
        # Extract sentence index from filename
        sentence_idx = int(activation_file.stem.split('_')[1])
        loaded_indices.add(sentence_idx)
        
        # Load activation
        token_ids, activations = load_single_activation(activation_file)
        
        result['activations'][sentence_idx] = activations
        if include_token_ids:
            result['token_ids'][sentence_idx] = token_ids
    
    # Track missing activations
    missing_activations = expected_indices - loaded_indices
    result['missing_activations'] = missing_activations
    
    return result


def load_all_activations(
    base_dir: Path,
    include_token_ids: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Load activations for all prompts in a directory.
    
    Handles both flat structure (prompt_name/metadata.json) and nested structure
    (prompt_name/rollout_X/metadata.json). For nested structures, merges all rollouts
    for each prompt into a single entry.
    
    Args:
        base_dir: Base directory containing prompt subdirectories
        include_token_ids: Whether to include token_ids in the output
        
    Returns:
        Dictionary mapping prompt_name -> prompt_activations dict
        (see load_prompt_activations for structure)
    """
    base_dir = Path(base_dir)
    
    # Find all directories containing metadata.json
    # Check both flat structure (base_dir/prompt_name/metadata.json) 
    # and nested structure (base_dir/prompt_name/rollout_X/metadata.json)
    activation_dirs = []
    
    for item in base_dir.iterdir():
        if not item.is_dir():
            continue
            
        # Check for flat structure: base_dir/prompt_name/metadata.json
        metadata_path = item / "metadata.json"
        if metadata_path.exists():
            activation_dirs.append((item, item.name))  # (dir_path, prompt_name)
        else:
            # Check for nested structure: base_dir/prompt_name/rollout_X/metadata.json
            for subitem in item.iterdir():
                if subitem.is_dir():
                    metadata_path = subitem / "metadata.json"
                    if metadata_path.exists():
                        activation_dirs.append((subitem, item.name))  # (dir_path, prompt_name)
    
    result = {}
    for activation_dir, prompt_name in sorted(activation_dirs, key=lambda x: (x[1], str(x[0]))):
        # Extract rollout index from directory path if it exists
        # Path format: base_dir/prompt_name/rollout_X/ or base_dir/prompt_name/
        rollout_idx = None
        dir_name = activation_dir.name
        if dir_name.startswith('rollout_'):
            try:
                rollout_idx = int(dir_name.split('_')[1])
            except (ValueError, IndexError):
                pass
        
        # Load activations from this directory
        activations_data = load_prompt_activations(activation_dir, include_token_ids=include_token_ids)
        
        # Convert integer keys to tuple keys if we have a rollout_idx
        # This ensures consistent key format for merging
        if rollout_idx is not None:
            # Convert activation keys from integers to tuples
            converted_activations = {}
            for key, value in activations_data['activations'].items():
                if isinstance(key, int):
                    converted_activations[(key, rollout_idx)] = value
                else:
                    converted_activations[key] = value
            activations_data['activations'] = converted_activations
            
            # Convert token_ids keys if present
            if 'token_ids' in activations_data:
                converted_token_ids = {}
                for key, value in activations_data['token_ids'].items():
                    if isinstance(key, int):
                        converted_token_ids[(key, rollout_idx)] = value
                    else:
                        converted_token_ids[key] = value
                activations_data['token_ids'] = converted_token_ids
            
            # Update sentence_data to have correct _activation_key
            # Match sentence_data entries to their activation keys
            for s in activations_data['sentence_data']:
                sentence_idx = s.get('sentence_index')
                if sentence_idx is not None:
                    # Set the activation key to match the converted key
                    s['_activation_key'] = (sentence_idx, rollout_idx)
        
        # If we already have data for this prompt (multiple rollouts), merge it
        if prompt_name in result:
            # Merge activations dictionaries
            # Note: Different rollouts can have the same sentence_index (e.g., both rollout_0 
            # and rollout_1 might have sentence_index=0, 5, 11, etc.). We need to keep all of them.
            # We'll create unique keys using (sentence_index, rollout_idx) tuples.
            
            # Merge activations - keys in activations_data are already converted to tuples if rollout_idx is not None
            for key, activation in activations_data['activations'].items():
                # Key is already in the correct format (tuple if rollout_idx is not None, int otherwise)
                result[prompt_name]['activations'][key] = activation
            
            # Merge token_ids if included
            if include_token_ids:
                if 'token_ids' not in result[prompt_name]:
                    result[prompt_name]['token_ids'] = {}
                if 'token_ids' in activations_data:
                    for key, token_ids in activations_data['token_ids'].items():
                        # Key is already in the correct format
                        result[prompt_name]['token_ids'][key] = token_ids
            
            # Merge sentence_data - keep ALL sentences from all rollouts
            # The _activation_key is already set correctly in the conversion step above
            for s in activations_data['sentence_data']:
                result[prompt_name]['sentence_data'].append(s)
        else:
            # First time seeing this prompt
            # Keys are already converted to tuples in the step above if rollout_idx is not None
            # So we can just use activations_data directly
            result[prompt_name] = activations_data
    
    return result


def stack_activations_by_layer(
    prompt_activations: Dict[str, Any],
    layer_idx: Optional[int] = None
) -> np.ndarray:
    """
    Stack activations across sentences for a single prompt.
    
    Handles both integer keys (single rollout) and tuple keys (multiple rollouts).
    For tuple keys, sorts by (sentence_index, rollout_idx).
    
    Args:
        prompt_activations: Output from load_prompt_activations
        layer_idx: Specific layer to extract (None = all layers)
        
    Returns:
        If layer_idx is None: (num_sentences, num_layers, hidden_dim)
        If layer_idx is specified: (num_sentences, hidden_dim)
    """
    activations_dict = prompt_activations['activations']
    
    # Check if we have any activations
    if not activations_dict:
        # Return empty array with appropriate shape
        # We need to determine the shape from somewhere - try to get it from another prompt
        # For now, return empty array - this will be handled by the caller
        raise ValueError("No activations found in prompt_activations")
    
    # Sort keys - handle both integer and tuple keys
    def sort_key(k):
        if isinstance(k, tuple):
            return k  # (sentence_index, rollout_idx) - sort by sentence_index first, then rollout
        else:
            return (k, 0)  # Treat integer keys as (sentence_index, 0)
    
    sorted_keys = sorted(activations_dict.keys(), key=sort_key)
    
    if not sorted_keys:
        raise ValueError("No activation keys found after sorting - activations_dict may be empty")
    
    # Get first activation to determine shape
    first_activation = activations_dict[sorted_keys[0]]
    num_layers, hidden_dim = first_activation.shape
    
    if layer_idx is not None:
        if layer_idx < 0 or layer_idx >= num_layers:
            raise ValueError(f"layer_idx {layer_idx} out of range [0, {num_layers})")
        stacked = np.stack([
            activations_dict[key][layer_idx].float().numpy() 
            for key in sorted_keys
        ])
        return stacked  # (num_sentences, hidden_dim)
    else:
        stacked = np.stack([
            activations_dict[key].float().numpy() 
            for key in sorted_keys
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
        
        # Skip prompts with no activations
        if not prompt_data.get('activations'):
            import warnings
            warnings.warn(f"Skipping prompt {prompt_name}: no activations found")
            continue
        
        try:
            stacked = stack_activations_by_layer(prompt_data, layer_idx=layer_idx)
            all_stacked.append(stacked)
        except ValueError as e:
            # Skip prompts with no activations or other errors
            import warnings
            warnings.warn(f"Skipping prompt {prompt_name}: {e}")
            continue
        
        # Sort sentence_data to match the order of stacked activations
        # Create a mapping from activation_key to sentence_data
        activations_dict = prompt_data['activations']
        sentence_data_map = {}
        
        # Build mapping: for each activation key, find the matching sentence_data
        # We need to handle the case where multiple rollouts have the same sentence_index
        def sort_key(k):
            if isinstance(k, tuple):
                return k
            else:
                return (k, 0)
        sorted_activation_keys = sorted(activations_dict.keys(), key=sort_key)
        
        # Track which sentence_data entries we've already matched
        used_sentence_data = set()
        
        # First pass: match sentence_data entries that have _activation_key set
        for s in prompt_data['sentence_data']:
            key = s.get('_activation_key')
            if key is not None and key in activations_dict:
                if key not in sentence_data_map:
                    sentence_data_map[key] = s
                    used_sentence_data.add(id(s))  # Use id() to track which dict we've used
        
        # Second pass: match remaining sentence_data entries to activation keys
        # This handles cases where _activation_key wasn't set correctly
        for key in sorted_activation_keys:
            if key in sentence_data_map:
                continue  # Already matched
            
            sentence_idx = key[0] if isinstance(key, tuple) else key
            
            # Find an unmatched sentence_data with this sentence_index
            for s in prompt_data['sentence_data']:
                if id(s) in used_sentence_data:
                    continue  # Already used
                
                if s.get('sentence_index') == sentence_idx:
                    # Found a match
                    sentence_data_map[key] = s
                    s['_activation_key'] = key  # Set it for consistency
                    used_sentence_data.add(id(s))
                    break
        
        # Add metadata in the same order as stacked activations
        for key in sorted_activation_keys:
            sentence_data = sentence_data_map.get(key)
            if sentence_data:
                all_metadata.append({
                    'prompt_name': prompt_name,
                    'sentence_index': sentence_data['sentence_index'],
                    'sentence': sentence_data['sentence'],
                    'p_reward_hacks': sentence_data['p_reward_hacks']
                })
            else:
                # This should not happen if all activations have matching sentence_data
                # But if it does, we skip it (activation exists but no metadata)
                import warnings
                warnings.warn(f"Could not find sentence_data for activation key {key} in prompt {prompt_name}")
    
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
    
    Only includes labels for sentences that have corresponding activations,
    matching the behavior of get_activation_matrix.
    
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
        
        # Skip prompts with no activations (same check as in get_activation_matrix)
        if not prompt_data.get('activations'):
            continue
        
        activations_dict = prompt_data['activations']
        
        # Sort keys the same way as in stack_activations_by_layer
        def sort_key(k):
            if isinstance(k, tuple):
                return k
            else:
                return (k, 0)
        sorted_keys = sorted(activations_dict.keys(), key=sort_key)
        
        # Build mapping from activation_key to sentence_data (same logic as get_activation_matrix)
        sentence_data_map = {}
        used_sentence_data = set()
        
        # First pass: match sentence_data entries that have _activation_key set
        for s in prompt_data['sentence_data']:
            key = s.get('_activation_key')
            if key is not None and key in activations_dict:
                if key not in sentence_data_map:
                    sentence_data_map[key] = s
                    used_sentence_data.add(id(s))
        
        # Second pass: match remaining sentence_data entries to activation keys
        for key in sorted_keys:
            if key in sentence_data_map:
                continue  # Already matched
            
            sentence_idx = key[0] if isinstance(key, tuple) else key
            
            # Find an unmatched sentence_data with this sentence_index
            for s in prompt_data['sentence_data']:
                if id(s) in used_sentence_data:
                    continue
                
                if s.get('sentence_index') == sentence_idx:
                    sentence_data_map[key] = s
                    s['_activation_key'] = key
                    used_sentence_data.add(id(s))
                    break
        
        # Extract labels in the same order as activations
        for key in sorted_keys:
            sentence_data = sentence_data_map.get(key)
            if sentence_data:
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


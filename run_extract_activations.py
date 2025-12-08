#!/usr/bin/env python3
"""
Run activation extraction experiment.

This script extracts activations from rollout JSON files using nnsight.
Configure the experiment parameters at the top of this file, then run:

    python run_extract_activations.py
"""

import sys
import json
from pathlib import Path
from typing import Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import nnsight
from transformers import AutoTokenizer
from activations.extract import store_activations
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# EXPERIMENT CONFIGURATION - Modify these parameters as needed
# ============================================================================

# Data files
ROLLOUTS_DIR = "src/rollouts"  # Directory containing rollout JSON files
OUTPUT_DIR = "/workspace"  # Output directory for activations

# Model configuration
MODEL_NAME = "openai/gpt-oss-20b"
DEVICE_MAP = "auto"  # Device mapping strategy
DTYPE = "bfloat16"  # Data type: "bfloat16", "float16", or "float32"

# Extraction parameters
LAYER_IDX = None  # Specific layer index to extract (None = all layers)
TOKEN_POSITION = "final"  # Which token to extract: "final" (last content token) or integer index
START_IDX = 0  # Starting sentence index (0 = first sentence)
END_IDX = None  # Ending sentence index (None = all sentences)

# Processing options
VERBOSE = False  # Print detailed output for debugging
SKIP_IF_EXISTS = True  # Skip prompts that already have output directories

# ============================================================================


def extract_prompt_name(rollout_file: str) -> str:
    """Extract prompt name from rollout filename."""
    filename = Path(rollout_file).stem
    # Handle format: prompt_name_rollout_0_rollouts
    if '_rollout_' in filename and filename.endswith('_rollouts'):
        # Remove _rollout_X_rollouts
        parts = filename.rsplit('_rollout_', 1)
        return parts[0]
    elif filename.endswith('_rollouts'):
        return filename[:-9]
    return filename


def extract_rollout_idx(rollout_file: str) -> Optional[int]:
    """Extract rollout index from rollout filename."""
    filename = Path(rollout_file).stem
    # Handle format: prompt_name_rollout_0_rollouts
    if '_rollout_' in filename and filename.endswith('_rollouts'):
        try:
            # Extract the number between _rollout_ and _rollouts
            parts = filename.rsplit('_rollout_', 1)
            if len(parts) == 2:
                rollout_part = parts[1].rsplit('_rollouts', 1)[0]
                return int(rollout_part)
        except (ValueError, IndexError):
            pass
    return None


def load_rollout_json(rollout_file: str) -> list:
    """Load rollout JSON file."""
    with open(rollout_file, 'r') as f:
        return json.load(f)


def save_metadata(
    prompt_name: str,
    sentence_data_list: list,
    activations_dir: str,
    start_idx: int = 0,
    end_idx: int = None
) -> None:
    """Save metadata JSON file for cross-referencing."""
    end_idx = end_idx if end_idx is not None else len(sentence_data_list)
    sentences_to_save = sentence_data_list[start_idx:end_idx]
    
    metadata = {
        'prompt_name': prompt_name,
        'sentence_data': [
            {
                'sentence_index': s.get('sentence_index', i),
                'sentence': s.get('sentence', ''),
                'p_reward_hacks': s.get('p_reward_hacks', 0.0),
                'file': f"sentence_{s.get('sentence_index', i)}.pt"
            }
            for i, s in enumerate(sentences_to_save)
        ]
    }
    
    metadata_path = Path(activations_dir) / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved metadata to {metadata_path}")


def find_rollout_files(rollouts_dir: str) -> list:
    """Find all rollout JSON files in the directory."""
    rollouts_path = Path(rollouts_dir)
    if not rollouts_path.exists():
        raise FileNotFoundError(f"Rollouts directory not found: {rollouts_dir}")
    
    # Find both formats: *_rollout_*_rollouts.json and *_rollouts.json
    rollout_files = sorted(rollouts_path.glob("*_rollout_*_rollouts.json")) + sorted(rollouts_path.glob("*_rollouts.json"))
    # Remove duplicates (in case a file matches both patterns)
    rollout_files = list(dict.fromkeys(rollout_files))
    return [str(f) for f in rollout_files]


def process_rollout_file(
    rollout_file: str,
    tokenizer: AutoTokenizer,
    model: Any,
    output_base_dir: str,
    start_idx: int,
    end_idx: int,
    layer_idx: int,
    token_position: str,
    verbose: bool,
    skip_if_exists: bool
) -> bool:
    """Process a single rollout file. Returns True if successful."""
    try:
        # Load rollout JSON
        logger.info(f"\n{'=' * 70}")
        logger.info(f"Processing: {rollout_file}")
        logger.info(f"{'=' * 70}")
        sentence_data_list = load_rollout_json(rollout_file)
        logger.info(f"Loaded {len(sentence_data_list)} sentences")
        
        # Extract prompt name and rollout index
        prompt_name = extract_prompt_name(rollout_file)
        rollout_idx = extract_rollout_idx(rollout_file)
        logger.info(f"Prompt name: {prompt_name}, Rollout index: {rollout_idx}")
        
        # Create output directory - include rollout index in path
        if rollout_idx is not None:
            activations_dir = Path(output_base_dir) / prompt_name / f"rollout_{rollout_idx}"
        else:
            activations_dir = Path(output_base_dir) / prompt_name
        
        # Check if already exists
        if skip_if_exists and activations_dir.exists():
            # Check if it has sentence files
            sentence_files = list(activations_dir.glob("sentence_*.pt"))
            if sentence_files:
                rollout_str = f"rollout_{rollout_idx}" if rollout_idx is not None else "default"
                logger.info(f"  ⏭ Skipping {prompt_name}/{rollout_str} (output directory already exists with {len(sentence_files)} files)")
                return True
        
        activations_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {activations_dir}")
        
        # Extract activations
        store_activations(
            model_name=MODEL_NAME,
            sentence_data_list=sentence_data_list,
            activations_dir=str(activations_dir),
            tokenizer=tokenizer,
            model=model,
            start_idx=start_idx,
            end_idx=end_idx,
            layer_idx=layer_idx,
            token_position=token_position,
            verbose=verbose
        )
        
        # Save metadata
        save_metadata(
            prompt_name=prompt_name,
            sentence_data_list=sentence_data_list,
            activations_dir=str(activations_dir),
            start_idx=start_idx,
            end_idx=end_idx
        )
        
        logger.info(f"✓ Completed {prompt_name}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to process {rollout_file}: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False


def main():
    """Run the activation extraction experiment for all rollout files."""
    
    print("=" * 70)
    print("ACTIVATION EXTRACTION EXPERIMENT")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Rollouts directory: {ROLLOUTS_DIR}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Layer index: {LAYER_IDX if LAYER_IDX is not None else 'all'}")
    print(f"  Token position: {TOKEN_POSITION}")
    print(f"  Start index: {START_IDX}")
    print(f"  End index: {END_IDX if END_IDX is not None else 'all'}")
    print(f"  Verbose: {VERBOSE}")
    print(f"  Skip if exists: {SKIP_IF_EXISTS}")
    print()
    
    # Find all rollout files
    logger.info(f"Finding rollout files in: {ROLLOUTS_DIR}")
    rollout_files = find_rollout_files(ROLLOUTS_DIR)
    logger.info(f"Found {len(rollout_files)} rollout file(s)")
    
    if not rollout_files:
        logger.error(f"No rollout files found in {ROLLOUTS_DIR}")
        return
    
    # Load tokenizer (once for all files)
    logger.info(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    logger.info("✓ Tokenizer loaded")
    
    # Parse dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(DTYPE.lower(), torch.bfloat16)
    
    # Load model with nnsight (once for all files)
    logger.info(f"\nLoading model: {MODEL_NAME}")
    logger.info(f"  Device map: {DEVICE_MAP}")
    logger.info(f"  Dtype: {dtype}")
    logger.info("  This may take a minute to load the model...")
    model = nnsight.LanguageModel(
        MODEL_NAME,
        device_map=DEVICE_MAP,
        dtype=dtype
    )
    logger.info("✓ Model loaded\n")
    
    # Process each rollout file
    successful = 0
    failed = 0
    
    for rollout_file in rollout_files:
        success = process_rollout_file(
            rollout_file=rollout_file,
            tokenizer=tokenizer,
            model=model,
            output_base_dir=OUTPUT_DIR,
            start_idx=START_IDX,
            end_idx=END_IDX,
            layer_idx=LAYER_IDX,
            token_position=TOKEN_POSITION,
            verbose=VERBOSE,
            skip_if_exists=SKIP_IF_EXISTS
        )
        
        if success:
            successful += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"\nProcessed {len(rollout_files)} rollout file(s)")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"\nActivations saved to: {OUTPUT_DIR}/<prompt_name>/")
    print(f"\nTo load activations:")
    print(f"  token_ids, activations = torch.load('{OUTPUT_DIR}/<prompt_name>/sentence_0.pt')")
    print(f"\nActivations shape: (num_layers, hidden_dim)")
    print(f"  Example: (24, 2880) for 24 layers with 2880 hidden dimensions")
    print()


if __name__ == "__main__":
    main()


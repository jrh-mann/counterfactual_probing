#!/usr/bin/env python3
"""
Main script to extract activations from rollout JSON files using nnsight.

Usage:
    python extract_activations.py --rollout-file src/rollouts/sat_solver_rollouts.json
    python extract_activations.py --rollout-file src/rollouts/sat_solver_rollouts.json --test --verbose
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import nnsight
from transformers import AutoTokenizer
from activations.extract import store_activations
import logging

logger = logging.getLogger(__name__)


def extract_prompt_name(rollout_file: str) -> str:
    """
    Extract prompt name from rollout filename.
    
    Example: 'src/rollouts/sat_solver_rollouts.json' -> 'sat_solver'
    """
    filename = Path(rollout_file).stem
    if filename.endswith('_rollouts'):
        return filename[:-9]  # Remove '_rollouts' suffix
    return filename


def load_rollout_json(rollout_file: str) -> list:
    """Load rollout JSON file."""
    with open(rollout_file, 'r') as f:
        return json.load(f)


def save_metadata(
    prompt_name: str,
    sentence_data_list: list,
    activations_dir: str,
    start_idx: int = 0,
    end_idx: Optional[int] = None
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


def main():
    parser = argparse.ArgumentParser(
        description="Extract activations from rollout JSON files using nnsight"
    )
    parser.add_argument(
        "--rollout-file",
        type=str,
        required=True,
        help="Path to rollout JSON file (e.g., src/rollouts/sat_solver_rollouts.json)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/workspace",
        help="Output directory (default: /workspace)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="openai/gpt-oss-20b",
        help="Model name (default: openai/gpt-oss-20b)"
    )
    parser.add_argument(
        "--layer-idx",
        type=int,
        default=None,
        help="Specific layer index to extract (default: None, extracts all layers)"
    )
    parser.add_argument(
        "--token-position",
        type=str,
        default="final",
        help="Which token to extract (default: 'final', can be 'final' or integer index)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode - process only first sentence of first rollout"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output for review (chat template, tokenization, shapes)"
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Starting sentence index (default: 0)"
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=None,
        help="Ending sentence index (default: None, processes all)"
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device mapping strategy (default: auto)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Data type for model (default: bfloat16)"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("=" * 70)
    print("ACTIVATION EXTRACTION WITH NNSIGHT")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Rollout file: {args.rollout_file}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Model: {args.model_name}")
    print(f"  Layer index: {args.layer_idx if args.layer_idx is not None else 'all'}")
    print(f"  Token position: {args.token_position}")
    print(f"  Test mode: {args.test}")
    print(f"  Verbose: {args.verbose}")
    print(f"  Start index: {args.start_idx}")
    print(f"  End index: {args.end_idx if args.end_idx is not None else 'all'}")
    print()
    
    # Load rollout JSON
    logger.info(f"Loading rollout file: {args.rollout_file}")
    sentence_data_list = load_rollout_json(args.rollout_file)
    logger.info(f"Loaded {len(sentence_data_list)} sentences")
    
    # Extract prompt name
    prompt_name = extract_prompt_name(args.rollout_file)
    logger.info(f"Prompt name: {prompt_name}")
    
    # Create output directory
    activations_dir = Path(args.output_dir) / prompt_name
    activations_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {activations_dir}")
    
    # Adjust indices for test mode
    start_idx = args.start_idx
    end_idx = args.end_idx
    if args.test:
        start_idx = 0
        end_idx = 1
        logger.info("Test mode: processing only first sentence")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Parse dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(args.dtype.lower(), torch.bfloat16)
    
    # Load model with nnsight
    logger.info(f"Loading model: {args.model_name}")
    logger.info(f"  Device map: {args.device_map}")
    logger.info(f"  Dtype: {dtype}")
    logger.info("  This may take a minute to load the model...")
    model = nnsight.LanguageModel(
        args.model_name,
        device_map=args.device_map,
        dtype=dtype
    )
    logger.info("  ✓ Model loaded\n")
    
    # Extract activations
    store_activations(
        model_name=args.model_name,
        sentence_data_list=sentence_data_list,
        activations_dir=str(activations_dir),
        tokenizer=tokenizer,
        model=model,
        start_idx=start_idx,
        end_idx=end_idx,
        layer_idx=args.layer_idx,
        token_position=args.token_position,
        verbose=args.verbose
    )
    
    # Save metadata
    save_metadata(
        prompt_name=prompt_name,
        sentence_data_list=sentence_data_list,
        activations_dir=str(activations_dir),
        start_idx=start_idx,
        end_idx=end_idx
    )
    
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"\nActivations saved to: {activations_dir}")
    print(f"Metadata saved to: {activations_dir / 'metadata.json'}")
    print(f"\nTo load activations:")
    print(f"  token_ids, activations = torch.load('{activations_dir / 'sentence_0.pt'}')")
    print()


if __name__ == "__main__":
    main()


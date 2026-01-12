#!/usr/bin/env python3
"""
Run activation extraction on counterfactual outputs.

Usage:
    # Using model-specific auto paths:
    python scripts/run_extraction.py --model Qwen/Qwen3-4B --experiment math

    # Using explicit paths (legacy):
    python scripts/run_extraction.py --input-dir outputs/math/ --output-dir activations/math/
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from counterfactual_probing.activations import CounterfactualActivationExtractor
from counterfactual_probing.model_utils import get_model_slug, get_experiment_paths


def main():
    parser = argparse.ArgumentParser(
        description="Extract activations from counterfactual outputs"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., Qwen/Qwen3-4B)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="math",
        help="Experiment name (determines subdirectory, default: math)",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Override: directory containing counterfactual JSON outputs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override: directory to save activation files",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="examples/math/problems.jsonl",
        help="Path to dataset for prompt lookup",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of files to process (for testing)",
    )
    parser.add_argument(
        "--test-one",
        action="store_true",
        help="Test extraction on one file and print details",
    )

    args = parser.parse_args()

    # Resolve paths based on model slug
    paths = get_experiment_paths(args.model, args.experiment)
    input_dir = args.input_dir or str(paths["outputs_dir"])
    output_dir = args.output_dir or str(paths["activations_dir"])

    print(f"Model: {args.model}")
    print(f"Model slug: {paths['model_slug']}")
    print(f"Experiment: {args.experiment}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Dataset: {args.dataset_path}")

    # Initialize extractor
    extractor = CounterfactualActivationExtractor(model_name=args.model)

    if args.test_one:
        # Test on one file
        import json

        input_path = Path(input_dir)
        json_files = sorted(input_path.glob("*.json"))

        if not json_files:
            print("No JSON files found!")
            return

        test_file = json_files[0]
        print(f"\nTesting on: {test_file}")

        with open(test_file) as f:
            output_data = json.load(f)

        activation_data = extractor.extract_for_output(
            output_data,
            dataset_path=args.dataset_path,
        )

        print(f"\nResults:")
        print(f"  Prompt ID: {activation_data.prompt_id}")
        print(f"  Prompt tokens: {activation_data.prompt_token_count}")
        print(f"  Generation tokens: {len(activation_data.generation_token_ids)}")
        print(f"  Full sequence: {activation_data.full_sequence_length}")
        print(f"  Activations shape: {activation_data.activations.shape}")
        print(f"  Branch points: {len(activation_data.branch_points)}")

        # Show branch points
        print("\n  Branch points:")
        for bp in activation_data.branch_points[:5]:
            print(f"    token_index={bp.token_index}, p_score={bp.p_score:.2f}")
        if len(activation_data.branch_points) > 5:
            print(f"    ... and {len(activation_data.branch_points) - 5} more")

        # Test get_branch_point_activations
        X, y = activation_data.get_branch_point_activations(layer_idx=-1)
        print(f"\n  Branch point activations (last layer): {X.shape}")
        print(f"  Labels: {y.shape}, range [{y.min():.2f}, {y.max():.2f}]")

    else:
        # Full extraction
        if args.limit:
            # Manual limit - process only first N files
            import json

            input_path = Path(input_dir)
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            json_files = sorted(input_path.glob("*.json"))[:args.limit]

            from tqdm import tqdm
            import gc
            import torch

            for json_file in tqdm(json_files, desc="Extracting"):
                act_file = output_path / f"{json_file.stem}.pt"
                if act_file.exists():
                    continue

                try:
                    with open(json_file) as f:
                        output_data = json.load(f)

                    activation_data = extractor.extract_for_output(
                        output_data,
                        dataset_path=args.dataset_path,
                    )

                    extractor._save_activation_data(activation_data, act_file)

                    gc.collect()
                    torch.cuda.empty_cache()

                except Exception as e:
                    print(f"Error: {json_file.name}: {e}")
                    continue

        else:
            extractor.extract_all(
                output_dir=input_dir,
                dataset_path=args.dataset_path,
                output_path=output_dir,
            )

    print("\nDone!")


if __name__ == "__main__":
    main()

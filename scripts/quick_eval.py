#!/usr/bin/env python3
"""
Quick model evaluation for counterfactual probing.

Runs a fast end-to-end evaluation on a small subset to estimate model performance
before committing to a full run.

Usage:
    python scripts/quick_eval.py --model Qwen/Qwen3-4B
    python scripts/quick_eval.py --model Qwen/Qwen3-4B --num-prompts 10
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def run_quick_eval(
    model_name: str,
    num_prompts: int = 5,
    num_counterfactuals: int = 20,
    num_samples: int = 10,
    dataset_path: str = "examples/math/problems.jsonl",
    gpu_memory_utilization: float = 0.9,
    verbose: bool = True,
):
    """
    Run a quick evaluation of a model for counterfactual probing.

    Args:
        model_name: Model to evaluate (e.g., "Qwen/Qwen3-4B")
        num_prompts: Number of prompts to test on
        num_counterfactuals: Counterfactuals per branch point
        num_samples: Branch points per prompt
        dataset_path: Path to dataset
        gpu_memory_utilization: GPU memory fraction to use
        verbose: Print detailed output

    Returns:
        Dict with evaluation results
    """
    from counterfactual_probing import (
        run_from_config,
        get_model_slug,
        get_experiment_paths,
    )
    from counterfactual_probing.activations import (
        CounterfactualActivationExtractor,
        load_all_activation_data,
        prepare_probe_data,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

    results = {
        "model": model_name,
        "model_slug": get_model_slug(model_name),
        "num_prompts": num_prompts,
        "num_counterfactuals": num_counterfactuals,
        "num_samples": num_samples,
        "timings": {},
        "metrics": {},
        "error": None,
    }

    # Use a temp directory for quick eval outputs
    with tempfile.TemporaryDirectory(prefix="quick_eval_") as tmpdir:
        tmpdir = Path(tmpdir)
        outputs_dir = tmpdir / "outputs"
        activations_dir = tmpdir / "activations"

        if verbose:
            print(f"Model: {model_name}")
            print(f"Slug: {results['model_slug']}")
            print(f"Prompts: {num_prompts}, Counterfactuals: {num_counterfactuals}")
            print(f"Temp dir: {tmpdir}")
            print()

        # Step 1: Generate counterfactuals
        if verbose:
            print("=" * 60)
            print("Step 1: Generating counterfactuals...")
            print("=" * 60)

        config = {
            "dataset": {
                "path": dataset_path,
                "prompt_field": "prompt",
            },
            "model": {
                "name": model_name,
                "gpu_memory_utilization": gpu_memory_utilization,
            },
            "generation": {
                "temperature": 0.7,
                "max_tokens": 4096,
                "num_counterfactuals": num_counterfactuals,
            },
            "sampling": {
                "method": "uniform_count",
                "num_samples": num_samples,
                "seed": 42,
            },
            "scorer": {
                "module": "counterfactual_probing.scorer.examples.math",
                "class": "MathScorer",
                "config": {"answer_field": "answer"},
            },
            "output": {
                "dir": str(outputs_dir),
            },
            "skip_existing": False,
        }

        start_time = time.time()
        try:
            run_from_config(config, limit=num_prompts)
        except Exception as e:
            results["error"] = f"Generation failed: {e}"
            return results

        results["timings"]["generation"] = time.time() - start_time
        if verbose:
            print(f"Generation time: {results['timings']['generation']:.1f}s")
            print()

        # Count generated files
        generated_files = list(outputs_dir.glob("*.json"))
        if not generated_files:
            results["error"] = "No output files generated"
            return results

        results["generated_files"] = len(generated_files)
        if verbose:
            print(f"Generated {len(generated_files)} output files")

        # Step 2: Extract activations
        if verbose:
            print()
            print("=" * 60)
            print("Step 2: Extracting activations...")
            print("=" * 60)

        start_time = time.time()
        try:
            extractor = CounterfactualActivationExtractor(model_name=model_name)
            extractor.extract_all(
                output_dir=str(outputs_dir),
                dataset_path=dataset_path,
                output_path=str(activations_dir),
            )
        except Exception as e:
            results["error"] = f"Extraction failed: {e}"
            return results

        results["timings"]["extraction"] = time.time() - start_time
        if verbose:
            print(f"Extraction time: {results['timings']['extraction']:.1f}s")
            print()

        # Step 3: Train probe and evaluate
        if verbose:
            print()
            print("=" * 60)
            print("Step 3: Training probe...")
            print("=" * 60)

        start_time = time.time()
        try:
            activation_data_list = load_all_activation_data(str(activations_dir))
            if not activation_data_list:
                results["error"] = "No activation files loaded"
                return results

            num_layers = activation_data_list[0].activations.shape[0]
            results["num_layers"] = num_layers

            # Test a few layers (first, middle, last)
            test_layers = [0, num_layers // 2, num_layers - 1]
            layer_results = {}

            for layer_idx in test_layers:
                X, y, metadata = prepare_probe_data(activation_data_list, layer_idx=layer_idx)

                if len(X) < 10:
                    continue

                # Simple train/test split
                n_test = max(1, len(X) // 5)
                X_train, X_test = X[:-n_test], X[-n_test:]
                y_train, y_test = y[:-n_test], y[-n_test:]

                # Binarize
                y_train_bin = (y_train >= 0.5).astype(int)
                y_test_bin = (y_test >= 0.5).astype(int)

                # Skip if only one class
                if len(np.unique(y_train_bin)) < 2:
                    continue

                # Train
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
                model.fit(X_train_scaled, y_train_bin)

                # Evaluate
                test_pred = model.predict(X_test_scaled)
                test_proba = model.predict_proba(X_test_scaled)

                layer_metrics = {
                    "train_samples": len(X_train),
                    "test_samples": len(X_test),
                    "test_accuracy": float(accuracy_score(y_test_bin, test_pred)),
                }

                if len(np.unique(y_test_bin)) == 2:
                    layer_metrics["test_auc"] = float(roc_auc_score(y_test_bin, test_proba[:, 1]))
                    layer_metrics["test_f1"] = float(f1_score(y_test_bin, test_pred, zero_division=0))

                layer_results[layer_idx] = layer_metrics

            results["layer_results"] = layer_results
            results["timings"]["probing"] = time.time() - start_time

            # Find best layer
            if layer_results:
                best_layer = max(
                    layer_results.keys(),
                    key=lambda l: layer_results[l].get("test_auc", 0)
                )
                results["best_layer"] = best_layer
                results["metrics"] = layer_results[best_layer]

        except Exception as e:
            results["error"] = f"Probing failed: {e}"
            return results

    results["timings"]["total"] = sum(results["timings"].values())

    if verbose:
        print()
        print("=" * 60)
        print("QUICK EVAL RESULTS")
        print("=" * 60)
        print(f"Model: {results['model']}")
        print(f"Model slug: {results['model_slug']}")
        print(f"Layers: {results.get('num_layers', 'N/A')}")
        print(f"Best layer: {results.get('best_layer', 'N/A')}")
        print()
        if results.get("metrics"):
            print("Best layer metrics:")
            for k, v in results["metrics"].items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.3f}")
                else:
                    print(f"  {k}: {v}")
        print()
        print("Timings:")
        for k, v in results["timings"].items():
            print(f"  {k}: {v:.1f}s")

        if results.get("error"):
            print(f"\nError: {results['error']}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Quick model evaluation for counterfactual probing"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model to evaluate (e.g., Qwen/Qwen3-4B)",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=5,
        help="Number of prompts to test (default: 5)",
    )
    parser.add_argument(
        "--num-counterfactuals",
        type=int,
        default=20,
        help="Counterfactuals per branch point (default: 20)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Branch points per prompt (default: 10)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="examples/math/problems.jsonl",
        help="Path to dataset",
    )
    parser.add_argument(
        "--gpu-memory",
        type=float,
        default=0.9,
        help="GPU memory utilization (default: 0.9)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output",
    )

    args = parser.parse_args()

    results = run_quick_eval(
        model_name=args.model,
        num_prompts=args.num_prompts,
        num_counterfactuals=args.num_counterfactuals,
        num_samples=args.num_samples,
        dataset_path=args.dataset,
        gpu_memory_utilization=args.gpu_memory,
        verbose=not args.quiet,
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    # Exit with error if failed
    if results.get("error"):
        sys.exit(1)


if __name__ == "__main__":
    main()

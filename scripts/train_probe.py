#!/usr/bin/env python3
"""
Train a logistic regression probe on counterfactual activation data.

Usage:
    # Using model-specific auto paths:
    python scripts/train_probe.py --model Qwen/Qwen3-4B --experiment math

    # Using explicit paths (legacy):
    python scripts/train_probe.py --activation-dir activations/qwen3-4b/math/
"""

import argparse
import sys
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, classification_report
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from counterfactual_probing.activations import (
    load_all_activation_data,
    prepare_probe_data,
)
from counterfactual_probing.model_utils import get_model_slug, get_experiment_paths


def split_by_prompt(
    metadata: list,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple:
    """
    Split data by prompt to avoid leakage.

    All branch points from a prompt go to either train or test, not both.
    """
    rng = np.random.RandomState(seed)

    # Group by prompt_id
    prompt_to_indices = {}
    for idx, meta in enumerate(metadata):
        prompt_id = meta["prompt_id"]
        if prompt_id not in prompt_to_indices:
            prompt_to_indices[prompt_id] = []
        prompt_to_indices[prompt_id].append(idx)

    # Split prompts
    prompt_ids = list(prompt_to_indices.keys())
    rng.shuffle(prompt_ids)

    n_test = max(1, int(len(prompt_ids) * test_ratio))
    test_prompts = set(prompt_ids[:n_test])
    train_prompts = set(prompt_ids[n_test:])

    # Collect indices
    train_indices = []
    test_indices = []

    for prompt_id, indices in prompt_to_indices.items():
        if prompt_id in test_prompts:
            test_indices.extend(indices)
        else:
            train_indices.extend(indices)

    return np.array(train_indices), np.array(test_indices)


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    threshold: float = 0.5,
    regularization: float = 1.0,
) -> dict:
    """
    Train logistic regression and evaluate.

    Args:
        X: Features (n_samples, n_features)
        y: Labels (n_samples,) - continuous p_scores
        train_idx: Training indices
        test_idx: Test indices
        threshold: Threshold for binarizing labels
        regularization: C parameter for logistic regression

    Returns:
        Dict with metrics and model
    """
    # Binarize labels
    y_binary = (y >= threshold).astype(int)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_binary[train_idx], y_binary[test_idx]

    # Check class balance
    train_pos_ratio = y_train.mean()
    test_pos_ratio = y_test.mean()

    print(f"  Train: {len(X_train)} samples, {train_pos_ratio:.1%} positive")
    print(f"  Test:  {len(X_test)} samples, {test_pos_ratio:.1%} positive")

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = LogisticRegression(
        C=regularization,
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    # Predictions
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    train_proba = model.predict_proba(X_train_scaled)[:, 1]
    test_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Compute metrics
    metrics = {
        "train_accuracy": accuracy_score(y_train, train_pred),
        "test_accuracy": accuracy_score(y_test, test_pred),
        "train_f1": f1_score(y_train, train_pred, zero_division=0),
        "test_f1": f1_score(y_test, test_pred, zero_division=0),
        "test_precision": precision_score(y_test, test_pred, zero_division=0),
        "test_recall": recall_score(y_test, test_pred, zero_division=0),
    }

    # AUC (requires both classes)
    if len(np.unique(y_test)) == 2:
        metrics["test_auc"] = roc_auc_score(y_test, test_proba)
    else:
        metrics["test_auc"] = float("nan")

    if len(np.unique(y_train)) == 2:
        metrics["train_auc"] = roc_auc_score(y_train, train_proba)
    else:
        metrics["train_auc"] = float("nan")

    return {
        "metrics": metrics,
        "model": model,
        "scaler": scaler,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "threshold": threshold,
    }


def main():
    parser = argparse.ArgumentParser(description="Train probe on activations")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (e.g., Qwen/Qwen3-4B) - used to derive paths",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="math",
        help="Experiment name (default: math)",
    )
    parser.add_argument(
        "--activation-dir",
        type=str,
        default=None,
        help="Override: directory containing activation .pt files",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="Layer to use (-1 = last layer, None = all layers)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for binarizing p_score labels",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Ratio of prompts to use for testing",
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=1.0,
        help="Regularization parameter C",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON",
    )
    parser.add_argument(
        "--sweep-layers",
        action="store_true",
        help="Sweep across all layers",
    )

    args = parser.parse_args()

    # Resolve activation directory
    if args.activation_dir:
        activation_dir = args.activation_dir
    elif args.model:
        paths = get_experiment_paths(args.model, args.experiment)
        activation_dir = str(paths["activations_dir"])
        print(f"Model: {args.model}")
        print(f"Model slug: {paths['model_slug']}")
        print(f"Experiment: {args.experiment}")
    else:
        # Fallback to legacy default
        activation_dir = "activations/math/"
        print("Warning: No --model specified, using legacy default path")

    print(f"Activation dir: {activation_dir}")
    print("Loading activation data...")
    activation_data_list = load_all_activation_data(activation_dir)
    print(f"Loaded {len(activation_data_list)} files")

    if not activation_data_list:
        print("No activation files found!")
        return

    # Get number of layers
    num_layers = activation_data_list[0].activations.shape[0]
    print(f"Model has {num_layers} layers")

    if args.sweep_layers:
        # Sweep across all layers
        print("\n" + "=" * 60)
        print("Layer sweep")
        print("=" * 60)

        results = []
        for layer_idx in range(num_layers):
            print(f"\nLayer {layer_idx}:")
            X, y, metadata = prepare_probe_data(activation_data_list, layer_idx=layer_idx)
            train_idx, test_idx = split_by_prompt(metadata, test_ratio=args.test_ratio)

            result = train_and_evaluate(
                X, y, train_idx, test_idx,
                threshold=args.threshold,
                regularization=args.regularization,
            )

            metrics = result["metrics"]
            print(f"  Test Accuracy: {metrics['test_accuracy']:.3f}")
            print(f"  Test AUC: {metrics['test_auc']:.3f}")
            print(f"  Test F1: {metrics['test_f1']:.3f}")

            results.append({
                "layer": layer_idx,
                **metrics,
            })

        # Find best layer
        best = max(results, key=lambda x: x.get("test_auc", 0))
        print(f"\nBest layer: {best['layer']} (AUC={best['test_auc']:.3f})")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Saved results to {args.output}")

    else:
        # Single layer
        layer_idx = args.layer if args.layer >= 0 else num_layers + args.layer
        print(f"\nUsing layer {layer_idx}")

        print("\nPreparing data...")
        X, y, metadata = prepare_probe_data(activation_data_list, layer_idx=layer_idx)
        print(f"Features shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Label distribution: min={y.min():.2f}, max={y.max():.2f}, mean={y.mean():.2f}")

        print(f"\nSplitting by prompt (test_ratio={args.test_ratio})...")
        train_idx, test_idx = split_by_prompt(metadata, test_ratio=args.test_ratio)

        print(f"\nTraining probe (threshold={args.threshold}, C={args.regularization})...")
        result = train_and_evaluate(
            X, y, train_idx, test_idx,
            threshold=args.threshold,
            regularization=args.regularization,
        )

        metrics = result["metrics"]
        print("\n" + "=" * 40)
        print("Results")
        print("=" * 40)
        print(f"Train Accuracy: {metrics['train_accuracy']:.3f}")
        print(f"Test Accuracy:  {metrics['test_accuracy']:.3f}")
        print(f"Train AUC:      {metrics['train_auc']:.3f}")
        print(f"Test AUC:       {metrics['test_auc']:.3f}")
        print(f"Test F1:        {metrics['test_f1']:.3f}")
        print(f"Test Precision: {metrics['test_precision']:.3f}")
        print(f"Test Recall:    {metrics['test_recall']:.3f}")

        # Detailed classification report
        y_binary = (y >= args.threshold).astype(int)
        y_test = y_binary[test_idx]
        X_test = X[test_idx]
        X_test_scaled = result["scaler"].transform(X_test)
        test_pred = result["model"].predict(X_test_scaled)

        print("\nClassification Report:")
        print(classification_report(y_test, test_pred, target_names=["Low p_score", "High p_score"]))

        if args.output:
            with open(args.output, "w") as f:
                json.dump({
                    "layer": layer_idx,
                    "threshold": args.threshold,
                    "regularization": args.regularization,
                    **metrics,
                }, f, indent=2)
            print(f"\nSaved results to {args.output}")

    print("\nDone!")


if __name__ == "__main__":
    main()

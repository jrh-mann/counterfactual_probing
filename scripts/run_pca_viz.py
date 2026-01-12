#!/usr/bin/env python3
"""
Run PCA visualization on counterfactual activations.

Usage:
    python scripts/run_pca_viz.py --activation-dir activations/math/ --layer 10
    python scripts/run_pca_viz.py --activation-dir activations/math/ --layer 10 --interpolate
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from counterfactual_probing.activations import load_all_activation_data, prepare_probe_data
from visualization.pca_plot import plot_pca_with_score, sweep_layers
from probing.interpolation import interpolate_labels


def prepare_interpolated_data(
    activation_data_list,
    layer_idx: int,
    sample_every: int = 50,
    min_relative_pos: float = 0.02,
    max_relative_pos: float = 0.95,
    method: str = 'linear',
):
    """
    Prepare data with interpolated labels at many token positions.

    Instead of only using 20 branch points, interpolate p_scores and
    sample activations at many more positions.
    """
    all_X = []
    all_y = []
    all_meta = []

    for data in activation_data_list:
        gen_len = len(data.generation_token_ids)

        # Get branch points and scores
        branch_points = [bp.token_index for bp in data.branch_points]
        p_scores = [bp.p_score for bp in data.branch_points]

        # Interpolate to all positions
        interpolated = interpolate_labels(
            branch_points, p_scores, gen_len, method=method
        )

        # Sample positions (every N tokens, within bounds)
        min_idx = int(gen_len * min_relative_pos)
        max_idx = int(gen_len * max_relative_pos)

        positions = list(range(min_idx, max_idx, sample_every))

        for pos in positions:
            acts = data.activations[layer_idx, pos, :].float().numpy()
            all_X.append(acts)
            all_y.append(interpolated[pos])
            all_meta.append({
                "prompt_id": data.prompt_id,
                "token_index": pos,
                "relative_pos": pos / gen_len,
                "interpolated": pos not in branch_points,
            })

    return np.stack(all_X), np.array(all_y), all_meta


def prepare_filtered_data(
    activation_data_list,
    layer_idx: int,
    min_relative_pos: float = 0.0,
    max_relative_pos: float = 1.0,
    exclude_first_token: bool = False,
    exclude_last_token: bool = False,
):
    """Prepare probe data with position filtering."""
    all_X = []
    all_y = []
    all_meta = []

    for data in activation_data_list:
        gen_len = len(data.generation_token_ids)

        for bp in data.branch_points:
            rel_pos = bp.token_index / gen_len

            # Apply filters
            if exclude_first_token and bp.token_index == 0:
                continue
            if exclude_last_token and bp.token_index == gen_len - 1:
                continue
            if rel_pos < min_relative_pos or rel_pos > max_relative_pos:
                continue

            # Get activation at this position
            acts = data.activations[layer_idx, bp.token_index, :].float().numpy()
            all_X.append(acts)
            all_y.append(bp.p_score)
            all_meta.append({
                "prompt_id": data.prompt_id,
                "token_index": bp.token_index,
                "relative_pos": rel_pos,
            })

    return np.stack(all_X), np.array(all_y), all_meta


def main():
    parser = argparse.ArgumentParser(description="PCA visualization of activations")
    parser.add_argument(
        "--activation-dir",
        type=str,
        default="activations/math/",
        help="Directory containing activation .pt files",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=10,
        help="Layer to visualize (default: 10, the best layer)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="plots/pca_counterfactual.html",
        help="Output HTML file",
    )
    parser.add_argument(
        "--exclude-first",
        action="store_true",
        help="Exclude first token (index=0) from visualization",
    )
    parser.add_argument(
        "--exclude-last",
        action="store_true",
        help="Exclude last token from visualization",
    )
    parser.add_argument(
        "--min-pos",
        type=float,
        default=0.0,
        help="Minimum relative position (0-1) to include",
    )
    parser.add_argument(
        "--max-pos",
        type=float,
        default=1.0,
        help="Maximum relative position (0-1) to include",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Generate plots for all layers",
    )
    parser.add_argument(
        "--sweep-layers",
        type=str,
        default=None,
        help="Comma-separated list of layers to sweep (e.g., '0,5,10,15,20,27')",
    )
    parser.add_argument(
        "--interpolate",
        action="store_true",
        help="Use interpolated labels and sample many more token positions",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=50,
        help="Sample every N tokens when using interpolation (default: 50)",
    )
    parser.add_argument(
        "--interp-method",
        type=str,
        default="linear",
        choices=["linear", "cubic", "nearest", "zero"],
        help="Interpolation method (default: linear)",
    )

    args = parser.parse_args()

    print("Loading activation data...")
    activation_data_list = load_all_activation_data(args.activation_dir)
    print(f"Loaded {len(activation_data_list)} files")

    # Get number of layers
    num_layers = activation_data_list[0].activations.shape[0]
    print(f"Model has {num_layers} layers")

    # Filter settings
    filter_desc = []
    if args.exclude_first:
        filter_desc.append("excluding first token")
    if args.exclude_last:
        filter_desc.append("excluding last token")
    if args.min_pos > 0:
        filter_desc.append(f"min_pos={args.min_pos}")
    if args.max_pos < 1:
        filter_desc.append(f"max_pos={args.max_pos}")

    if filter_desc:
        print(f"Filters: {', '.join(filter_desc)}")

    if args.sweep or args.sweep_layers:
        # Sweep mode - generate for multiple layers
        if args.sweep_layers:
            layers = [int(l) for l in args.sweep_layers.split(",")]
        else:
            layers = list(range(num_layers))

        print(f"Generating plots for {len(layers)} layers...")
        output_dir = Path(args.output).parent if args.output else Path("plots")
        output_dir.mkdir(parents=True, exist_ok=True)

        for layer_idx in layers:
            X, y, metadata = prepare_filtered_data(
                activation_data_list,
                layer_idx=layer_idx,
                min_relative_pos=args.min_pos,
                max_relative_pos=args.max_pos,
                exclude_first_token=args.exclude_first,
                exclude_last_token=args.exclude_last,
            )

            meta_list = [{"prompt_name": m["prompt_id"], "token_index": m["token_index"]} for m in metadata]

            fig = plot_pca_with_score(
                activations=X,
                labels=y,
                layer=0,
                metadata=meta_list,
                title=f"Layer {layer_idx} - PCA (n={len(y)} samples)",
            )

            filepath = output_dir / f"pca_layer_{layer_idx:02d}.html"
            fig.write_html(str(filepath))
            print(f"Saved: {filepath}")

    else:
        # Single layer
        layer_idx = args.layer if args.layer >= 0 else num_layers + args.layer
        print(f"\nUsing layer {layer_idx}")

        if args.interpolate:
            print(f"Preparing interpolated data (sample_every={args.sample_every}, method={args.interp_method})...")
            X, y, metadata = prepare_interpolated_data(
                activation_data_list,
                layer_idx=layer_idx,
                sample_every=args.sample_every,
                min_relative_pos=args.min_pos,
                max_relative_pos=args.max_pos,
                method=args.interp_method,
            )
            title_suffix = f"interpolated, sample_every={args.sample_every}"
        else:
            print("Preparing data with filters...")
            X, y, metadata = prepare_filtered_data(
                activation_data_list,
                layer_idx=layer_idx,
                min_relative_pos=args.min_pos,
                max_relative_pos=args.max_pos,
                exclude_first_token=args.exclude_first,
                exclude_last_token=args.exclude_last,
            )
            title_suffix = "filtered"

        meta_list = [{"prompt_name": m["prompt_id"], "token_index": m["token_index"]} for m in metadata]

        print(f"Data shape: {X.shape}")
        print(f"Labels: min={y.min():.2f}, max={y.max():.2f}, mean={y.mean():.2f}")

        print(f"\nGenerating PCA plot for layer {layer_idx}...")
        fig = plot_pca_with_score(
            activations=X,
            labels=y,
            layer=0,  # Already extracted single layer
            metadata=meta_list,
            title=f"Layer {layer_idx} - PCA Projection (n={len(y)} samples, {title_suffix})",
        )

        # Save
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        print(f"Saved: {output_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Maximal PCA visualization: dense sampling with both predicted and interpolated labels.

Creates a dual view:
- Left: Z = probe predictions (smooth decision surface)
- Right: Z = interpolated actual labels (ground truth surface)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from counterfactual_probing.activations import load_all_activation_data
from probing.interpolation import interpolate_labels


def extract_maximal_data(
    activation_data_list,
    layer_idx: int,
    sample_every: int = 3,
    min_relative_pos: float = 0.02,
    max_relative_pos: float = 0.95,
    interp_method: str = 'linear',
):
    """
    Extract activations at maximum density with interpolated labels.
    """
    all_X = []
    all_interpolated = []
    all_meta = []

    for data in activation_data_list:
        gen_len = len(data.generation_token_ids)

        # Get branch points and interpolate
        branch_points = [bp.token_index for bp in data.branch_points]
        p_scores = [bp.p_score for bp in data.branch_points]
        interpolated = interpolate_labels(branch_points, p_scores, gen_len, method=interp_method)

        # Sample densely
        min_idx = int(gen_len * min_relative_pos)
        max_idx = int(gen_len * max_relative_pos)

        measured_set = set(branch_points)

        for pos in range(min_idx, max_idx, sample_every):
            acts = data.activations[layer_idx, pos, :].float().numpy()
            all_X.append(acts)
            all_interpolated.append(interpolated[pos])
            all_meta.append({
                "prompt_id": data.prompt_id,
                "token_index": pos,
                "relative_pos": pos / gen_len,
                "is_measured": pos in measured_set,
            })

    return np.stack(all_X), np.array(all_interpolated), all_meta


def train_probe(activation_data_list, layer_idx: int):
    """Train probe on measured points."""
    all_X = []
    all_y = []

    for data in activation_data_list:
        gen_len = len(data.generation_token_ids)
        for bp in data.branch_points:
            rel_pos = bp.token_index / gen_len
            if bp.token_index == 0 or rel_pos > 0.95:
                continue
            acts = data.activations[layer_idx, bp.token_index, :].float().numpy()
            all_X.append(acts)
            all_y.append(bp.p_score)

    X = np.stack(all_X)
    y = np.array(all_y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Binary classification
    y_binary = (y >= 0.5).astype(int)
    probe = LogisticRegression(C=1.0, max_iter=1000)
    probe.fit(X_scaled, y_binary)

    return probe, scaler


def create_maximal_dual_plot(
    X_pca: np.ndarray,
    predicted: np.ndarray,
    interpolated: np.ndarray,
    metadata: list,
    var_explained: tuple,
    title: str,
):
    """Create dual 3D scatter plot."""

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=(
            f'Probe Predictions (n={len(X_pca):,})',
            f'Interpolated Labels (n={len(X_pca):,})'
        ),
        horizontal_spacing=0.02,
    )

    # Common hover text
    hover_text = [
        f"prompt: {m['prompt_id']}<br>"
        f"token: {m['token_index']}<br>"
        f"rel_pos: {m['relative_pos']:.2f}<br>"
        f"pred: {predicted[i]:.3f}<br>"
        f"interp: {interpolated[i]:.3f}"
        for i, m in enumerate(metadata)
    ]

    # Left: Predicted
    fig.add_trace(go.Scatter3d(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        z=predicted,
        mode='markers',
        marker=dict(
            size=2,
            color=predicted,
            colorscale='RdYlGn',
            opacity=0.6,
            cmin=0, cmax=1,
        ),
        text=hover_text,
        hoverinfo='text',
        showlegend=False,
    ), row=1, col=1)

    # Right: Interpolated
    fig.add_trace(go.Scatter3d(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        z=interpolated,
        mode='markers',
        marker=dict(
            size=2,
            color=interpolated,
            colorscale='RdYlGn',
            opacity=0.6,
            cmin=0, cmax=1,
        ),
        text=hover_text,
        hoverinfo='text',
        showlegend=False,
    ), row=1, col=2)

    var_0, var_1 = var_explained

    # Update layout
    fig.update_layout(
        title=dict(text=title, x=0.5),
        width=1600,
        height=800,
    )

    # Update both scenes
    scene_settings = dict(
        xaxis_title=f'PC1 ({var_0:.1f}%)',
        yaxis_title=f'PC2 ({var_1:.1f}%)',
        zaxis_title='p_score',
        xaxis=dict(range=[X_pca[:, 0].min(), X_pca[:, 0].max()]),
        yaxis=dict(range=[X_pca[:, 1].min(), X_pca[:, 1].max()]),
        zaxis=dict(range=[0, 1]),
    )

    fig.update_layout(
        scene=scene_settings,
        scene2=scene_settings,
    )

    return fig


def main():
    parser = argparse.ArgumentParser(description="Maximal PCA with predictions and interpolation")
    parser.add_argument("--activation-dir", type=str, default="activations/math/")
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--sample-every", type=int, default=3,
                        help="Sample every N tokens (lower = more points)")
    parser.add_argument("--output", type=str, default="plots/pca_maximal.html")
    parser.add_argument("--interp-method", type=str, default="linear",
                        choices=["linear", "cubic", "nearest"])

    args = parser.parse_args()

    print("Loading activation data...")
    data_list = load_all_activation_data(args.activation_dir)
    print(f"Loaded {len(data_list)} files")

    num_layers = data_list[0].activations.shape[0]
    layer_idx = args.layer if args.layer >= 0 else num_layers + args.layer
    print(f"Using layer {layer_idx}")

    # Train probe
    print("\nTraining probe...")
    probe, scaler = train_probe(data_list, layer_idx)

    # Extract maximal data
    print(f"\nExtracting activations (sample_every={args.sample_every})...")
    X, interpolated_labels, metadata = extract_maximal_data(
        data_list,
        layer_idx=layer_idx,
        sample_every=args.sample_every,
        interp_method=args.interp_method,
    )
    print(f"Total points: {len(X):,}")

    # Scale and predict
    print("Scaling and predicting...")
    X_scaled = scaler.transform(X)
    predicted_proba = probe.predict_proba(X_scaled)[:, 1]

    # PCA
    print("Fitting PCA...")
    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X_scaled)

    var_explained = (
        pca.explained_variance_ratio_[0] * 100,
        pca.explained_variance_ratio_[1] * 100,
    )

    # Create plot
    print("\nGenerating visualization...")
    fig = create_maximal_dual_plot(
        X_pca=X_pca,
        predicted=predicted_proba,
        interpolated=interpolated_labels,
        metadata=metadata,
        var_explained=var_explained,
        title=f"Layer {layer_idx} PCA: Probe Predictions vs Interpolated Labels",
    )

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    print(f"Saved: {output_path}")

    # Stats
    print(f"\nStats:")
    print(f"  Predicted - mean: {predicted_proba.mean():.3f}, std: {predicted_proba.std():.3f}")
    print(f"  Interpolated - mean: {interpolated_labels.mean():.3f}, std: {interpolated_labels.std():.3f}")
    print(f"  Correlation: {np.corrcoef(predicted_proba, interpolated_labels)[0,1]:.3f}")

    print("\nDone!")


if __name__ == "__main__":
    main()

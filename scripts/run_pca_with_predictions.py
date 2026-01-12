#!/usr/bin/env python3
"""
PCA visualization with probe predictions.

Shows the full activation domain with predicted labels from the trained probe,
giving a smoother, denser view of the decision surface.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from counterfactual_probing.activations import load_all_activation_data


def extract_all_activations(
    activation_data_list,
    layer_idx: int,
    sample_every: int = 10,
    min_relative_pos: float = 0.02,
    max_relative_pos: float = 0.95,
):
    """
    Extract activations at ALL token positions (densely sampled).

    Returns activations and metadata, but NO labels (we'll predict those).
    """
    all_X = []
    all_meta = []

    for data in activation_data_list:
        gen_len = len(data.generation_token_ids)

        # Build lookup for actual measured p_scores
        measured = {bp.token_index: bp.p_score for bp in data.branch_points}

        min_idx = int(gen_len * min_relative_pos)
        max_idx = int(gen_len * max_relative_pos)

        for pos in range(min_idx, max_idx, sample_every):
            acts = data.activations[layer_idx, pos, :].float().numpy()
            all_X.append(acts)

            # Check if this position was actually measured
            actual_score = measured.get(pos, None)

            all_meta.append({
                "prompt_id": data.prompt_id,
                "token_index": pos,
                "relative_pos": pos / gen_len,
                "actual_score": actual_score,  # None if not measured
                "is_measured": actual_score is not None,
            })

    return np.stack(all_X), all_meta


def train_probe_for_viz(
    activation_data_list,
    layer_idx: int,
    threshold: float = 0.5,
):
    """Train a simple probe on measured branch points."""
    all_X = []
    all_y = []

    for data in activation_data_list:
        for bp in data.branch_points:
            # Skip boundary tokens
            gen_len = len(data.generation_token_ids)
            rel_pos = bp.token_index / gen_len
            if bp.token_index == 0 or rel_pos > 0.95:
                continue

            acts = data.activations[layer_idx, bp.token_index, :].float().numpy()
            all_X.append(acts)
            all_y.append(bp.p_score)

    X = np.stack(all_X)
    y = np.array(all_y)

    # Fit scaler and probe
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use regression-style output (predict continuous p_score)
    # But logistic regression gives us probabilities which work well
    y_binary = (y >= threshold).astype(int)

    probe = LogisticRegression(C=1.0, max_iter=1000)
    probe.fit(X_scaled, y_binary)

    return probe, scaler


def create_visualization(
    X: np.ndarray,
    metadata: list,
    predicted_scores: np.ndarray,
    pca: PCA,
    title: str = "PCA with Probe Predictions",
):
    """Create interactive 3D visualization."""

    # Project to PCA
    X_pca = pca.transform(X)

    # Separate measured and predicted points
    measured_mask = np.array([m["is_measured"] for m in metadata])

    # Get actual scores for measured points
    actual_scores = np.array([
        m["actual_score"] if m["actual_score"] is not None else 0
        for m in metadata
    ])

    fig = go.Figure()

    # Plot all points colored by PREDICTED score
    hover_text = [
        f"prompt: {m['prompt_id']}<br>"
        f"token: {m['token_index']}<br>"
        f"rel_pos: {m['relative_pos']:.2f}<br>"
        f"predicted: {predicted_scores[i]:.3f}<br>"
        f"actual: {m['actual_score']:.3f}" if m['actual_score'] is not None else
        f"prompt: {m['prompt_id']}<br>"
        f"token: {m['token_index']}<br>"
        f"rel_pos: {m['relative_pos']:.2f}<br>"
        f"predicted: {predicted_scores[i]:.3f}<br>"
        f"actual: N/A"
        for i, m in enumerate(metadata)
    ]

    fig.add_trace(go.Scatter3d(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        z=predicted_scores,
        mode='markers',
        marker=dict(
            size=3,
            color=predicted_scores,
            colorscale='RdYlGn',
            opacity=0.6,
            colorbar=dict(title='Predicted p_score', x=1.0),
        ),
        text=hover_text,
        hoverinfo='text',
        name='All tokens (predicted)',
    ))

    # Overlay measured points with actual scores (larger, different marker)
    if np.any(measured_mask):
        measured_indices = np.where(measured_mask)[0]
        fig.add_trace(go.Scatter3d(
            x=X_pca[measured_indices, 0],
            y=X_pca[measured_indices, 1],
            z=actual_scores[measured_indices],
            mode='markers',
            marker=dict(
                size=6,
                color=actual_scores[measured_indices],
                colorscale='RdYlGn',
                opacity=1.0,
                symbol='diamond',
                line=dict(width=1, color='black'),
            ),
            text=[hover_text[i] for i in measured_indices],
            hoverinfo='text',
            name='Measured points (actual)',
        ))

    # Variance explained
    var_0 = pca.explained_variance_ratio_[0] * 100
    var_1 = pca.explained_variance_ratio_[1] * 100

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=f'PC1 ({var_0:.1f}% var)',
            yaxis_title=f'PC2 ({var_1:.1f}% var)',
            zaxis_title='p_score (predicted)',
        ),
        width=1000,
        height=800,
        showlegend=True,
        legend=dict(x=0, y=1),
    )

    return fig


def create_dual_visualization(
    X: np.ndarray,
    metadata: list,
    predicted_scores: np.ndarray,
    pca: PCA,
    title: str = "PCA: Predicted vs Actual",
):
    """Create side-by-side visualization showing predicted and actual."""

    X_pca = pca.transform(X)
    measured_mask = np.array([m["is_measured"] for m in metadata])
    actual_scores = np.array([
        m["actual_score"] if m["actual_score"] is not None else predicted_scores[i]
        for i, m in enumerate(metadata)
    ])

    # Create figure with 2 subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=('Z = Predicted p_score', 'Z = Actual p_score (where measured)'),
        horizontal_spacing=0.05,
    )

    hover_text = [
        f"prompt: {m['prompt_id']}<br>"
        f"token: {m['token_index']}<br>"
        f"predicted: {predicted_scores[i]:.3f}<br>"
        f"actual: {m['actual_score']:.3f}" if m['actual_score'] is not None else
        f"prompt: {m['prompt_id']}<br>"
        f"token: {m['token_index']}<br>"
        f"predicted: {predicted_scores[i]:.3f}<br>"
        f"actual: N/A"
        for i, m in enumerate(metadata)
    ]

    # Left plot: Z = predicted
    fig.add_trace(go.Scatter3d(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        z=predicted_scores,
        mode='markers',
        marker=dict(
            size=3,
            color=predicted_scores,
            colorscale='RdYlGn',
            opacity=0.7,
        ),
        text=hover_text,
        hoverinfo='text',
        name='Predicted',
    ), row=1, col=1)

    # Right plot: Z = actual (only measured points)
    if np.any(measured_mask):
        measured_idx = np.where(measured_mask)[0]
        fig.add_trace(go.Scatter3d(
            x=X_pca[measured_idx, 0],
            y=X_pca[measured_idx, 1],
            z=actual_scores[measured_idx],
            mode='markers',
            marker=dict(
                size=5,
                color=actual_scores[measured_idx],
                colorscale='RdYlGn',
                opacity=0.9,
            ),
            text=[hover_text[i] for i in measured_idx],
            hoverinfo='text',
            name='Actual (measured)',
        ), row=1, col=2)

    var_0 = pca.explained_variance_ratio_[0] * 100
    var_1 = pca.explained_variance_ratio_[1] * 100

    fig.update_layout(
        title=title,
        width=1400,
        height=700,
        showlegend=True,
    )

    # Update both scenes
    for i in [1, 2]:
        fig.update_scenes(
            dict(
                xaxis_title=f'PC1 ({var_0:.1f}%)',
                yaxis_title=f'PC2 ({var_1:.1f}%)',
                zaxis_title='p_score',
            ),
            row=1, col=i
        )

    return fig


def main():
    parser = argparse.ArgumentParser(description="PCA with probe predictions")
    parser.add_argument("--activation-dir", type=str, default="activations/math/")
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--sample-every", type=int, default=10,
                        help="Sample every N tokens (lower = denser)")
    parser.add_argument("--output", type=str, default="plots/pca_predicted.html")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for probe training")
    parser.add_argument("--dual", action="store_true",
                        help="Create side-by-side predicted vs actual plot")
    parser.add_argument("--n-components", type=int, default=10,
                        help="Number of PCA components to compute")

    args = parser.parse_args()

    print("Loading activation data...")
    activation_data_list = load_all_activation_data(args.activation_dir)
    print(f"Loaded {len(activation_data_list)} files")

    num_layers = activation_data_list[0].activations.shape[0]
    layer_idx = args.layer if args.layer >= 0 else num_layers + args.layer
    print(f"Using layer {layer_idx}")

    # Train probe on measured points
    print("\nTraining probe on measured branch points...")
    probe, scaler = train_probe_for_viz(activation_data_list, layer_idx, args.threshold)

    # Extract ALL activations (dense sampling)
    print(f"\nExtracting activations (sample_every={args.sample_every})...")
    X, metadata = extract_all_activations(
        activation_data_list,
        layer_idx=layer_idx,
        sample_every=args.sample_every,
    )
    print(f"Total points: {len(X)}")

    # Predict scores for all points
    print("Predicting scores with probe...")
    X_scaled = scaler.transform(X)
    predicted_proba = probe.predict_proba(X_scaled)[:, 1]  # Probability of positive class

    # Fit PCA
    print(f"Fitting PCA (n_components={args.n_components})...")
    n_components = min(args.n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)

    # Count measured vs predicted
    n_measured = sum(1 for m in metadata if m["is_measured"])
    print(f"Measured points: {n_measured}, Predicted-only: {len(metadata) - n_measured}")

    # Create visualization
    print("\nGenerating visualization...")
    if args.dual:
        fig = create_dual_visualization(
            X_scaled, metadata, predicted_proba, pca,
            title=f"Layer {layer_idx} - PCA with Probe Predictions (n={len(X)})"
        )
    else:
        fig = create_visualization(
            X_scaled, metadata, predicted_proba, pca,
            title=f"Layer {layer_idx} - PCA with Probe Predictions (n={len(X)})"
        )

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    print(f"Saved: {output_path}")

    # Print some stats
    print(f"\nPrediction stats:")
    print(f"  Mean predicted: {predicted_proba.mean():.3f}")
    print(f"  Std predicted: {predicted_proba.std():.3f}")
    print(f"  Min/Max: {predicted_proba.min():.3f} / {predicted_proba.max():.3f}")

    print("\nDone!")


if __name__ == "__main__":
    main()

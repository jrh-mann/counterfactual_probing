"""
PCA visualization of activations with p_score as third dimension.
"""

import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from typing import Optional, Tuple, List, Dict, Any


def plot_pca_with_score(
    activations: np.ndarray,
    labels: np.ndarray,
    layer: int,
    metadata: Optional[List[Dict[str, Any]]] = None,
    n_components: int = 10,
    pc_x: int = 0,
    pc_y: int = 1,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Create 3D scatter plot with PCA components as X, Y and p_score as Z.

    Args:
        activations: Array of shape (n_samples, n_layers, hidden_dim) or (n_samples, hidden_dim)
        labels: Array of shape (n_samples,) with p_score values
        layer: Which layer to use (ignored if activations is 2D)
        metadata: Optional list of metadata dicts for hover info
        n_components: Number of PCA components to compute
        pc_x: Which PC to use for X axis (0-indexed, default 0 = PC1)
        pc_y: Which PC to use for Y axis (0-indexed, default 1 = PC2)
        title: Optional plot title

    Returns:
        Plotly Figure object
    """
    # Extract layer if 3D
    if len(activations.shape) == 3:
        # (n_samples, n_layers, hidden_dim)
        layer_activations = activations[:, layer, :]
    elif len(activations.shape) == 2:
        # (n_samples, hidden_dim)
        layer_activations = activations
    else:
        raise ValueError(f"Expected 2D or 3D activations, got shape {activations.shape}")

    n_samples, hidden_dim = layer_activations.shape

    # Fit PCA
    n_components = min(n_components, n_samples, hidden_dim)
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(layer_activations)

    # Extract coordinates
    x = projected[:, pc_x]
    y = projected[:, pc_y]
    z = labels

    # Build hover text
    if metadata:
        hover_text = [
            f"prompt: {m.get('prompt_name', 'unknown')}<br>"
            f"p_score: {labels[i]:.3f}<br>"
            f"PC{pc_x+1}: {x[i]:.3f}<br>"
            f"PC{pc_y+1}: {y[i]:.3f}"
            for i, m in enumerate(metadata)
        ]
    else:
        hover_text = [
            f"p_score: {labels[i]:.3f}<br>"
            f"PC{pc_x+1}: {x[i]:.3f}<br>"
            f"PC{pc_y+1}: {y[i]:.3f}"
            for i in range(n_samples)
        ]

    # Create figure
    fig = go.Figure(data=[
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=4,
                color=z,
                colorscale='RdYlGn',  # Red (low) -> Yellow -> Green (high)
                opacity=0.7,
                colorbar=dict(title='p_score'),
            ),
            text=hover_text,
            hoverinfo='text',
        )
    ])

    # Variance explained
    var_x = pca.explained_variance_ratio_[pc_x] * 100
    var_y = pca.explained_variance_ratio_[pc_y] * 100

    # Layout
    if title is None:
        title = f"Layer {layer} - PCA Projection"

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=f'PC{pc_x+1} ({var_x:.1f}% var)',
            yaxis_title=f'PC{pc_y+1} ({var_y:.1f}% var)',
            zaxis_title='p_score',
        ),
        width=900,
        height=700,
    )

    return fig


def sweep_layers(
    activations: np.ndarray,
    labels: np.ndarray,
    metadata: Optional[List[Dict[str, Any]]] = None,
    layers: Optional[List[int]] = None,
    output_dir: str = "plots",
    **kwargs,
) -> List[str]:
    """
    Generate PCA plots for multiple layers.

    Args:
        activations: Array of shape (n_samples, n_layers, hidden_dim)
        labels: Array of shape (n_samples,) with p_score values
        metadata: Optional list of metadata dicts
        layers: Which layers to plot (default: all)
        output_dir: Directory to save HTML files
        **kwargs: Additional args passed to plot_pca_with_score

    Returns:
        List of saved file paths
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    if len(activations.shape) != 3:
        raise ValueError("sweep_layers requires 3D activations (n_samples, n_layers, hidden_dim)")

    n_layers = activations.shape[1]

    if layers is None:
        layers = list(range(n_layers))

    saved_files = []
    for layer in layers:
        fig = plot_pca_with_score(
            activations=activations,
            labels=labels,
            layer=layer,
            metadata=metadata,
            **kwargs,
        )

        filepath = os.path.join(output_dir, f"pca_layer_{layer:02d}.html")
        fig.write_html(filepath)
        saved_files.append(filepath)
        print(f"Saved: {filepath}")

    return saved_files


# Convenience function for quick use
def quick_pca_plot(
    base_dir: str,
    layer: int,
    label_key: str = 'p_score',
    **kwargs,
) -> go.Figure:
    """
    Quick PCA plot from a directory of activations.

    Args:
        base_dir: Directory containing activation files
        layer: Which layer to visualize
        label_key: Key in metadata for the label (default 'p_score')
        **kwargs: Additional args passed to plot_pca_with_score

    Returns:
        Plotly Figure object
    """
    from src.activations.load import quick_load

    activations, metadata, _ = quick_load(base_dir)

    # Extract labels from metadata
    labels = np.array([m.get(label_key, 0.0) for m in metadata])

    return plot_pca_with_score(
        activations=activations,
        labels=labels,
        layer=layer,
        metadata=metadata,
        **kwargs,
    )

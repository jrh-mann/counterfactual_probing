"""
Ablation runner for systematic probe experiments.

Provides grid search, result aggregation, and comparison visualization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Iterator
from itertools import product
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
from datetime import datetime

from .config import ProbeConfig, MetricConfig
from .result import ProbeResult
from .trainer import ProbeTrainer


class AblationRunner:
    """
    Runs ablation studies across probe configurations.

    Handles:
    - Grid search over parameter combinations
    - Result aggregation into DataFrames
    - Comparison visualizations
    - Result persistence
    """

    def __init__(
        self,
        activations: np.ndarray,
        labels: np.ndarray,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None,
        metric_config: Optional[MetricConfig] = None,
    ):
        """
        Initialize ablation runner.

        Args:
            activations: (n_samples, n_layers, hidden_dim) activation array
            labels: (n_samples,) p_score labels
            train_idx: Indices for training set
            test_idx: Indices for test set
            metadata: Optional sample metadata
            metric_config: Metric configuration
        """
        self.activations = activations
        self.labels = labels
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.metadata = metadata
        self.metric_config = metric_config or MetricConfig()
        self.trainer = ProbeTrainer(self.metric_config)

        # Results storage
        self.results: List[ProbeResult] = []

    def run(self, config: ProbeConfig) -> ProbeResult:
        """
        Run single experiment.

        Args:
            config: Probe configuration

        Returns:
            ProbeResult
        """
        result = self.trainer.train(
            self.activations,
            self.labels,
            config,
            self.train_idx,
            self.test_idx,
            self.metadata,
        )
        self.results.append(result)
        return result

    def sweep(
        self,
        configs: List[ProbeConfig],
        progress: bool = True,
    ) -> pd.DataFrame:
        """
        Run multiple experiments.

        Args:
            configs: List of configurations to run
            progress: Show progress bar

        Returns:
            DataFrame with results
        """
        iterator = tqdm(configs, desc="Running ablations") if progress else configs

        for config in iterator:
            self.run(config)

        return self.results_dataframe()

    def grid(
        self,
        progress: bool = True,
        **param_ranges: Union[List, Any],
    ) -> pd.DataFrame:
        """
        Run grid search over parameter combinations.

        Args:
            progress: Show progress bar
            **param_ranges: Parameter names and their values to sweep.
                           Single values are treated as fixed.
                           Lists are swept over.

        Returns:
            DataFrame with results

        Example:
            runner.grid(
                layer=[5, 10, 15, 20],
                method=['logistic', 'ridge'],
                smoothing=[None, 'swim'],
                label_threshold=0.5,  # fixed
            )
        """
        configs = list(self._generate_configs(**param_ranges))

        if progress:
            print(f"Running {len(configs)} configurations...")

        return self.sweep(configs, progress=progress)

    def _generate_configs(self, **param_ranges) -> Iterator[ProbeConfig]:
        """Generate ProbeConfig instances from parameter grid."""
        # Separate fixed and variable parameters
        fixed = {}
        variable = {}

        for key, value in param_ranges.items():
            if isinstance(value, list):
                variable[key] = value
            else:
                fixed[key] = value

        # Generate all combinations of variable parameters
        if variable:
            keys = list(variable.keys())
            for combo in product(*[variable[k] for k in keys]):
                params = dict(fixed)
                params.update(zip(keys, combo))
                yield ProbeConfig(**params)
        else:
            # No variable parameters, just yield single config
            yield ProbeConfig(**fixed)

    def results_dataframe(self) -> pd.DataFrame:
        """
        Convert results to DataFrame.

        Returns:
            DataFrame with one row per experiment
        """
        rows = [r.metrics_row() for r in self.results]
        df = pd.DataFrame(rows)

        # Add index
        df.index.name = 'experiment_id'
        df = df.reset_index()

        return df

    def best_result(self, metric: str = 'test_acc', higher_is_better: bool = True) -> ProbeResult:
        """
        Get best result by specified metric.

        Args:
            metric: Metric name to optimize
            higher_is_better: Whether higher values are better

        Returns:
            Best ProbeResult
        """
        if not self.results:
            raise ValueError("No results yet. Run some experiments first.")

        df = self.results_dataframe()

        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found. Available: {list(df.columns)}")

        if higher_is_better:
            best_idx = df[metric].idxmax()
        else:
            best_idx = df[metric].idxmin()

        return self.results[best_idx]

    def summary(self) -> str:
        """Generate text summary of results."""
        if not self.results:
            return "No results yet."

        df = self.results_dataframe()

        lines = [
            f"=== Ablation Summary ===",
            f"Total experiments: {len(self.results)}",
            f"",
            f"=== Best by Test Accuracy ===",
        ]

        best = self.best_result('test_acc')
        lines.append(best.summary())

        lines.extend([
            f"",
            f"=== Metric Ranges ===",
            f"Test accuracy: {df['test_acc'].min():.4f} - {df['test_acc'].max():.4f}",
            f"Test loss: {df['test_loss'].min():.4f} - {df['test_loss'].max():.4f}",
        ])

        if 'test_mse' in df.columns:
            lines.append(f"Test MSE: {df['test_mse'].min():.4f} - {df['test_mse'].max():.4f}")

        return "\n".join(lines)

    # === Visualization Methods ===

    def plot_layer_comparison(
        self,
        metric: str = 'test_acc',
        color_by: Optional[str] = None,
    ) -> go.Figure:
        """
        Plot metric across layers.

        Args:
            metric: Which metric to plot
            color_by: Column to use for color grouping

        Returns:
            Plotly Figure
        """
        df = self.results_dataframe()

        if color_by:
            fig = px.line(df, x='layer', y=metric, color=color_by, markers=True)
        else:
            fig = px.line(df, x='layer', y=metric, markers=True)

        fig.update_layout(
            title=f'{metric} by Layer',
            xaxis_title='Layer',
            yaxis_title=metric,
        )

        return fig

    def plot_comparison(
        self,
        x: str,
        y: str = 'test_acc',
        color: Optional[str] = None,
        facet_col: Optional[str] = None,
    ) -> go.Figure:
        """
        General comparison plot.

        Args:
            x: Column for x-axis
            y: Column for y-axis
            color: Column for color grouping
            facet_col: Column for faceting

        Returns:
            Plotly Figure
        """
        df = self.results_dataframe()

        fig = px.scatter(
            df, x=x, y=y,
            color=color,
            facet_col=facet_col,
            hover_data=df.columns.tolist(),
        )

        fig.update_layout(title=f'{y} vs {x}')

        return fig

    def plot_heatmap(
        self,
        x: str,
        y: str,
        value: str = 'test_acc',
    ) -> go.Figure:
        """
        Heatmap of metric across two parameters.

        Args:
            x: Column for x-axis
            y: Column for y-axis
            value: Metric to show as color

        Returns:
            Plotly Figure
        """
        df = self.results_dataframe()

        pivot = df.pivot_table(index=y, columns=x, values=value, aggfunc='mean')

        fig = px.imshow(
            pivot,
            labels={'color': value},
            aspect='auto',
        )

        fig.update_layout(title=f'{value} by {x} and {y}')

        return fig

    # === Persistence ===

    def save_results(self, output_dir: str) -> None:
        """
        Save all results to directory.

        Saves:
        - results.csv: DataFrame of metrics
        - results.json: Full ProbeResult objects
        - config.json: Run configuration

        Args:
            output_dir: Directory to save to
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save DataFrame
        df = self.results_dataframe()
        df.to_csv(output_dir / 'results.csv', index=False)

        # Save full results
        results_data = [r.to_dict() for r in self.results]
        with open(output_dir / 'results.json', 'w') as f:
            json.dump(results_data, f, indent=2)

        # Save run info
        run_info = {
            'timestamp': datetime.now().isoformat(),
            'n_experiments': len(self.results),
            'n_train': len(self.train_idx),
            'n_test': len(self.test_idx),
            'activation_shape': list(self.activations.shape),
        }
        with open(output_dir / 'run_info.json', 'w') as f:
            json.dump(run_info, f, indent=2)

        print(f"Saved {len(self.results)} results to {output_dir}")

    def load_results(self, output_dir: str) -> None:
        """
        Load results from directory.

        Args:
            output_dir: Directory to load from
        """
        output_dir = Path(output_dir)

        with open(output_dir / 'results.json') as f:
            results_data = json.load(f)

        self.results = [ProbeResult.from_dict(r) for r in results_data]
        print(f"Loaded {len(self.results)} results")

    def clear_results(self) -> None:
        """Clear stored results."""
        self.results = []

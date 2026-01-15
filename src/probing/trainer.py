"""
Core probe training logic.

ProbeTrainer handles:
- Layer selection and concatenation
- Label processing (binarization, interpolation)
- Smoothing
- Model fitting
- Metric computation
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, log_loss, mean_squared_error, mean_absolute_error,
    roc_auc_score, precision_score, recall_score, f1_score
)

from .config import ProbeConfig, MetricConfig
from .result import ProbeResult
from .smoothing import smooth_logits
from .softmax_weighted import SoftmaxWeightedWrapper


class ProbeTrainer:
    """
    Trains linear probes on activations.

    Handles all preprocessing, training, and evaluation.
    Extensible via subclassing or method registration.
    """

    # Registry of probe methods - extend by adding here
    METHODS = {
        'logistic': lambda cfg: LogisticRegression(
            C=cfg.regularization,
            max_iter=cfg.max_iter,
            random_state=cfg.seed,
            solver='lbfgs',
        ),
        'ridge': lambda cfg: Ridge(
            alpha=1.0 / cfg.regularization,  # Ridge uses alpha, inverse of C
            random_state=cfg.seed,
        ),
        'softmax_weighted': lambda cfg: SoftmaxWeightedWrapper(
            temperature=getattr(cfg, 'softmax_temperature', 1.0),
            learning_rate=0.01,
            max_iter=cfg.max_iter,
            weight_decay=1.0 / cfg.regularization if cfg.regularization > 0 else 0.0,
            seed=cfg.seed,
        ),
    }

    def __init__(self, metric_config: Optional[MetricConfig] = None):
        """
        Initialize trainer.

        Args:
            metric_config: Configuration for metrics. Uses defaults if None.
        """
        self.metric_config = metric_config or MetricConfig()

    def train(
        self,
        activations: np.ndarray,
        labels: np.ndarray,
        config: ProbeConfig,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> ProbeResult:
        """
        Train a probe with given configuration.

        Args:
            activations: (n_samples, n_layers, hidden_dim) or (n_samples, hidden_dim)
            labels: (n_samples,) continuous p_score values
            config: Probe configuration
            train_idx: Indices for training set
            test_idx: Indices for test set
            metadata: Optional metadata for each sample

        Returns:
            ProbeResult with trained probe and metrics
        """
        # 1. Extract and prepare features
        X, layer_info = self._prepare_features(activations, config)
        feature_dim = X.shape[1]

        # 2. Prepare labels
        y, y_binary = self._prepare_labels(labels, config)

        # 3. Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        y_train_binary, y_test_binary = y_binary[train_idx], y_binary[test_idx]

        # 4. Normalize if requested
        if config.normalize:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # 5. Train model
        model = self._create_model(config)

        # Softmax-weighted probe needs group information
        if config.method == 'softmax_weighted':
            # Extract group IDs from metadata (indices into train set)
            if metadata is not None:
                # Build prompt_id -> group_idx mapping
                prompt_ids = [m.get('prompt_id', i) for i, m in enumerate(metadata)]
                unique_prompts = list(dict.fromkeys(prompt_ids))  # Preserve order
                prompt_to_idx = {p: i for i, p in enumerate(unique_prompts)}
                group_ids = np.array([prompt_to_idx[prompt_ids[i]] for i in train_idx])
            else:
                # Fall back to individual groups
                group_ids = np.arange(len(train_idx))
            model.set_group_ids(group_ids)

        model.fit(X_train, y_train_binary if config.method in ('logistic', 'softmax_weighted') else y_train)

        # 6. Get predictions
        if config.method in ('logistic', 'softmax_weighted'):
            train_proba = model.predict_proba(X_train)[:, 1]
            test_proba = model.predict_proba(X_test)[:, 1]
            eps = 1e-10
            train_logits = np.log((train_proba + eps) / (1 - train_proba + eps))
            test_logits = np.log((test_proba + eps) / (1 - test_proba + eps))
        else:
            # Ridge regression - predictions are continuous
            train_proba = model.predict(X_train)
            test_proba = model.predict(X_test)
            train_proba = np.clip(train_proba, 0, 1)
            test_proba = np.clip(test_proba, 0, 1)
            train_logits = train_proba  # Not really logits, but keep interface consistent
            test_logits = test_proba

        # 7. Apply smoothing if configured (for evaluation)
        if config.smoothing:
            train_logits_smoothed = smooth_logits(
                train_logits, config.smoothing,
                config.window_size, config.ema_alpha
            )
            test_logits_smoothed = smooth_logits(
                test_logits, config.smoothing,
                config.window_size, config.ema_alpha
            )
            # Convert back to probabilities
            if config.method == 'logistic':
                train_proba = 1 / (1 + np.exp(-train_logits_smoothed))
                test_proba = 1 / (1 + np.exp(-test_logits_smoothed))

        # 8. Compute metrics
        metrics = self._compute_metrics(
            y_train, y_test, y_train_binary, y_test_binary,
            train_proba, test_proba, config
        )

        # 9. Extract probe direction
        if hasattr(model, 'coef_'):
            direction = model.coef_.flatten()
        else:
            direction = np.zeros(feature_dim)

        bias = model.intercept_ if hasattr(model, 'intercept_') else 0.0
        if isinstance(bias, np.ndarray):
            bias = bias[0]

        # 10. Build result
        return ProbeResult(
            direction=direction,
            bias=float(bias),
            config=config.to_dict(),
            train_accuracy=metrics['train_accuracy'],
            test_accuracy=metrics['test_accuracy'],
            train_loss=metrics['train_loss'],
            test_loss=metrics['test_loss'],
            threshold_metrics=metrics.get('threshold_metrics', {}),
            train_mse=metrics.get('train_mse'),
            test_mse=metrics.get('test_mse'),
            train_mae=metrics.get('train_mae'),
            test_mae=metrics.get('test_mae'),
            auc_by_threshold=metrics.get('auc_by_threshold', {}),
            n_train=len(train_idx),
            n_test=len(test_idx),
            layer_info=layer_info,
            feature_dim=feature_dim,
            train_predictions=train_proba,
            test_predictions=test_proba,
            train_labels=y_train,
            test_labels=y_test,
        )

    def _prepare_features(
        self,
        activations: np.ndarray,
        config: ProbeConfig,
    ) -> Tuple[np.ndarray, str]:
        """
        Extract and prepare features based on layer config.

        Returns:
            Tuple of (features array, layer description string)
        """
        if len(activations.shape) == 2:
            # Already 2D, just use as-is
            return activations, "flat"

        n_samples, n_layers, hidden_dim = activations.shape

        if isinstance(config.layer, int):
            # Single layer
            layer_idx = config.layer if config.layer >= 0 else n_layers + config.layer
            X = activations[:, layer_idx, :]
            layer_info = f"layer_{layer_idx}"

        elif isinstance(config.layer, list):
            # Concatenate specified layers
            layers = [l if l >= 0 else n_layers + l for l in config.layer]
            X = np.concatenate([activations[:, l, :] for l in layers], axis=1)
            layer_info = f"layers_{layers}"

        elif config.layer == "all":
            # Concatenate all layers
            X = activations.reshape(n_samples, -1)
            layer_info = "all_layers"

        elif config.layer.startswith("every_n:"):
            # Every n-th layer
            n = int(config.layer.split(":")[1])
            layers = list(range(0, n_layers, n))
            X = np.concatenate([activations[:, l, :] for l in layers], axis=1)
            layer_info = f"every_{n}_layers"

        else:
            raise ValueError(f"Invalid layer config: {config.layer}")

        return X, layer_info

    def _prepare_labels(
        self,
        labels: np.ndarray,
        config: ProbeConfig,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare continuous and binary labels.

        Returns:
            Tuple of (continuous_labels, binary_labels)
        """
        y_continuous = labels.copy()

        if config.label_threshold is not None:
            y_binary = (labels >= config.label_threshold).astype(int)
        else:
            # For regression, binary is just for metric computation
            y_binary = (labels >= 0.5).astype(int)

        return y_continuous, y_binary

    def _create_model(self, config: ProbeConfig):
        """Create model instance from config."""
        if config.method not in self.METHODS:
            raise ValueError(f"Unknown method: {config.method}. Available: {list(self.METHODS.keys())}")
        return self.METHODS[config.method](config)

    def _compute_metrics(
        self,
        y_train: np.ndarray,
        y_test: np.ndarray,
        y_train_binary: np.ndarray,
        y_test_binary: np.ndarray,
        train_proba: np.ndarray,
        test_proba: np.ndarray,
        config: ProbeConfig,
    ) -> Dict[str, Any]:
        """Compute all configured metrics."""
        metrics = {}
        mc = self.metric_config

        # Binary predictions at primary threshold
        threshold = config.label_threshold or mc.primary_threshold
        train_pred = (train_proba >= threshold).astype(int)
        test_pred = (test_proba >= threshold).astype(int)

        # Accuracy
        if mc.include_accuracy:
            metrics['train_accuracy'] = accuracy_score(y_train_binary, train_pred)
            metrics['test_accuracy'] = accuracy_score(y_test_binary, test_pred)

        # Loss (binary cross-entropy)
        if mc.include_loss:
            eps = 1e-10
            train_proba_clipped = np.clip(train_proba, eps, 1 - eps)
            test_proba_clipped = np.clip(test_proba, eps, 1 - eps)
            metrics['train_loss'] = log_loss(y_train_binary, train_proba_clipped)
            metrics['test_loss'] = log_loss(y_test_binary, test_proba_clipped)

        # Regression metrics (against continuous labels)
        if mc.include_mse:
            metrics['train_mse'] = mean_squared_error(y_train, train_proba)
            metrics['test_mse'] = mean_squared_error(y_test, test_proba)

        if mc.include_mae:
            metrics['train_mae'] = mean_absolute_error(y_train, train_proba)
            metrics['test_mae'] = mean_absolute_error(y_test, test_proba)

        # AUC at multiple thresholds
        if mc.include_auc:
            auc_by_threshold = {}
            for thresh in mc.binary_thresholds:
                y_train_bin = (y_train >= thresh).astype(int)
                y_test_bin = (y_test >= thresh).astype(int)

                # AUC requires both classes present
                if len(np.unique(y_test_bin)) == 2:
                    auc_by_threshold[thresh] = roc_auc_score(y_test_bin, test_proba)
                else:
                    auc_by_threshold[thresh] = np.nan

            metrics['auc_by_threshold'] = auc_by_threshold

        # Precision, Recall, F1 at primary threshold
        if mc.include_precision_recall:
            if len(np.unique(y_test_binary)) == 2:
                metrics['test_precision'] = precision_score(y_test_binary, test_pred, zero_division=0)
                metrics['test_recall'] = recall_score(y_test_binary, test_pred, zero_division=0)
                metrics['test_f1'] = f1_score(y_test_binary, test_pred, zero_division=0)

        # Metrics at each threshold
        threshold_metrics = {}
        for thresh in mc.binary_thresholds:
            y_test_bin = (y_test >= thresh).astype(int)
            pred_bin = (test_proba >= thresh).astype(int)

            tm = {'threshold': thresh}
            if len(np.unique(y_test_bin)) >= 1:
                tm['accuracy'] = accuracy_score(y_test_bin, pred_bin)
                if len(np.unique(y_test_bin)) == 2:
                    tm['precision'] = precision_score(y_test_bin, pred_bin, zero_division=0)
                    tm['recall'] = recall_score(y_test_bin, pred_bin, zero_division=0)
                    tm['f1'] = f1_score(y_test_bin, pred_bin, zero_division=0)

            threshold_metrics[thresh] = tm

        metrics['threshold_metrics'] = threshold_metrics

        return metrics

    @classmethod
    def register_method(cls, name: str, factory):
        """
        Register a new probe method.

        Args:
            name: Name for the method
            factory: Callable that takes ProbeConfig and returns sklearn-like model
        """
        cls.METHODS[name] = factory

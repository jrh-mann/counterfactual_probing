#!/usr/bin/env python3
"""
Compare probe training methods across layers.

Optimized version: loads each file ONCE and extracts all layers.
"""

import argparse
import gc
import sys
import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, r2_score
from scipy.stats import pearsonr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from counterfactual_probing.model_utils import get_experiment_paths


@dataclass
class MethodResult:
    method: str
    layers: list
    train_accuracies: list
    test_accuracies: list
    train_aucs: list
    test_aucs: list
    # Correlation metrics (continuous)
    train_r2: list
    test_r2: list
    train_pearson: list
    test_pearson: list
    # Best layer info
    best_layer: int
    best_test_r2: float
    best_layer_test_proba: np.ndarray = None
    best_layer_test_labels: np.ndarray = None
    best_layer_train_proba: np.ndarray = None
    best_layer_train_labels: np.ndarray = None


def load_single_file(args):
    """Load a single file and extract branch point activations for all layers."""
    pt_file, num_layers = args
    data = torch.load(pt_file, map_location='cpu', weights_only=False)

    activations = data['activations']
    branch_points = data['branch_points']
    prompt_id = data['prompt_id']

    indices = [bp['token_index'] for bp in branch_points]
    p_scores = [bp['p_score'] for bp in branch_points]

    # Extract all layers at branch points only - much smaller!
    # Shape: (num_layers, num_branch_points, hidden_dim)
    bp_acts = activations[:, indices, :].float().numpy()

    return {
        'prompt_id': prompt_id,
        'bp_acts': bp_acts,  # (num_layers, num_bp, hidden_dim)
        'p_scores': p_scores,
        'bp_indices': indices,
    }


def load_all_layers_parallel(activation_dir: Path, num_layers: int, num_workers: int = 8):
    """
    Load ALL data with parallel file loading.

    Returns dict mapping layer_idx -> (X, y, metadata)
    """
    from multiprocessing import Pool

    pt_files = sorted(activation_dir.glob("*.pt"))
    args = [(f, num_layers) for f in pt_files]

    print(f"Loading {len(pt_files)} files with {num_workers} workers...")

    # Parallel load
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(load_single_file, args),
            total=len(pt_files),
            desc="Loading"
        ))

    # Reorganize by layer
    layer_data = {i: {'X': [], 'y': [], 'meta': []} for i in range(num_layers)}

    for res in results:
        prompt_id = res['prompt_id']
        bp_acts = res['bp_acts']  # (num_layers, num_bp, hidden_dim)
        p_scores = res['p_scores']
        bp_indices = res['bp_indices']

        for layer_idx in range(num_layers):
            layer_data[layer_idx]['X'].append(bp_acts[layer_idx])
            layer_data[layer_idx]['y'].extend(p_scores)
            for idx in bp_indices:
                layer_data[layer_idx]['meta'].append({
                    'prompt_id': prompt_id,
                    'token_index': idx,
                })

    del results
    gc.collect()

    # Convert to arrays
    result = {}
    for layer_idx in range(num_layers):
        X = np.vstack(layer_data[layer_idx]['X'])
        y = np.array(layer_data[layer_idx]['y'])
        meta = layer_data[layer_idx]['meta']
        result[layer_idx] = (X, y, meta)

    del layer_data
    gc.collect()

    return result


def split_by_prompt(metadata: list, test_ratio: float = 0.2, seed: int = 42):
    rng = np.random.RandomState(seed)
    prompt_to_indices = {}
    for idx, meta in enumerate(metadata):
        pid = meta["prompt_id"]
        if pid not in prompt_to_indices:
            prompt_to_indices[pid] = []
        prompt_to_indices[pid].append(idx)

    prompt_ids = list(prompt_to_indices.keys())
    rng.shuffle(prompt_ids)
    n_test = max(1, int(len(prompt_ids) * test_ratio))
    test_prompts = set(prompt_ids[:n_test])

    train_idx, test_idx = [], []
    for pid, indices in prompt_to_indices.items():
        if pid in test_prompts:
            test_idx.extend(indices)
        else:
            train_idx.extend(indices)

    return np.array(train_idx), np.array(test_idx)


def get_group_ids(metadata: list, indices: np.ndarray):
    prompt_ids = [metadata[i]['prompt_id'] for i in indices]
    unique = list(dict.fromkeys(prompt_ids))
    p2i = {p: i for i, p in enumerate(unique)}
    return np.array([p2i[p] for p in prompt_ids])


def train_logistic(X_train, y_train, y_train_cont, X_test, y_test, y_test_cont, reg=1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, roc_auc_score

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    model = LogisticRegression(C=reg, max_iter=1000, solver='lbfgs', random_state=42)
    model.fit(X_tr, y_train)

    tr_prob = model.predict_proba(X_tr)[:, 1]
    te_prob = model.predict_proba(X_te)[:, 1]

    # R² and Pearson against continuous labels
    train_r2 = r2_score(y_train_cont, tr_prob)
    test_r2 = r2_score(y_test_cont, te_prob)
    train_pearson = pearsonr(y_train_cont, tr_prob)[0]
    test_pearson = pearsonr(y_test_cont, te_prob)[0]

    return {
        'train_acc': accuracy_score(y_train, (tr_prob >= 0.5).astype(int)),
        'test_acc': accuracy_score(y_test, (te_prob >= 0.5).astype(int)),
        'train_auc': roc_auc_score(y_train, tr_prob) if len(np.unique(y_train)) == 2 else np.nan,
        'test_auc': roc_auc_score(y_test, te_prob) if len(np.unique(y_test)) == 2 else np.nan,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_pearson': train_pearson,
        'test_pearson': test_pearson,
        'train_proba': tr_prob,
        'test_proba': te_prob,
    }


def train_ridge(X_train, y_train, y_train_cont, X_test, y_test, y_test_cont, reg=1.0):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, roc_auc_score

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    model = Ridge(alpha=1.0/reg, random_state=42)
    model.fit(X_tr, y_train_cont)

    tr_prob = np.clip(model.predict(X_tr), 0, 1)
    te_prob = np.clip(model.predict(X_te), 0, 1)

    # R² and Pearson against continuous labels
    train_r2 = r2_score(y_train_cont, tr_prob)
    test_r2 = r2_score(y_test_cont, te_prob)
    train_pearson = pearsonr(y_train_cont, tr_prob)[0]
    test_pearson = pearsonr(y_test_cont, te_prob)[0]

    return {
        'train_acc': accuracy_score(y_train, (tr_prob >= 0.5).astype(int)),
        'test_acc': accuracy_score(y_test, (te_prob >= 0.5).astype(int)),
        'train_auc': roc_auc_score(y_train, tr_prob) if len(np.unique(y_train)) == 2 else np.nan,
        'test_auc': roc_auc_score(y_test, te_prob) if len(np.unique(y_test)) == 2 else np.nan,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_pearson': train_pearson,
        'test_pearson': test_pearson,
        'train_proba': tr_prob,
        'test_proba': te_prob,
    }


def train_softmax_weighted(X_train, y_train, y_train_cont, X_test, y_test, y_test_cont, group_ids, reg=1.0):
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, roc_auc_score
    from probing.softmax_weighted import SoftmaxWeightedWrapper

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    model = SoftmaxWeightedWrapper(
        temperature=1.0,
        weight_decay=1.0/reg if reg > 0 else 0.0,
        seed=42,
    )
    model.set_group_ids(group_ids)
    model.fit(X_tr, y_train)

    tr_prob = model.predict_proba(X_tr)[:, 1]
    te_prob = model.predict_proba(X_te)[:, 1]

    # R² and Pearson against continuous labels
    train_r2 = r2_score(y_train_cont, tr_prob)
    test_r2 = r2_score(y_test_cont, te_prob)
    train_pearson = pearsonr(y_train_cont, tr_prob)[0]
    test_pearson = pearsonr(y_test_cont, te_prob)[0]

    return {
        'train_acc': accuracy_score(y_train, (tr_prob >= 0.5).astype(int)),
        'test_acc': accuracy_score(y_test, (te_prob >= 0.5).astype(int)),
        'train_auc': roc_auc_score(y_train, tr_prob) if len(np.unique(y_train)) == 2 else np.nan,
        'test_auc': roc_auc_score(y_test, te_prob) if len(np.unique(y_test)) == 2 else np.nan,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_pearson': train_pearson,
        'test_pearson': test_pearson,
        'train_proba': tr_prob,
        'test_proba': te_prob,
    }


def train_method_all_layers(method, all_layer_data, num_layers, threshold=0.5, reg=1.0):
    print(f"\n{'='*60}")
    print(f"Method: {method.upper()}")
    print('='*60)

    layers = []
    tr_accs, te_accs = [], []
    tr_aucs, te_aucs = [], []
    tr_r2, te_r2 = [], []
    tr_pearson, te_pearson = [], []

    best_layer, best_r2 = 0, -999.0
    best_te_proba, best_te_labels = None, None
    best_tr_proba, best_tr_labels = None, None

    for layer_idx in tqdm(range(num_layers), desc=f"Training {method}"):
        X, y_cont, meta = all_layer_data[layer_idx]
        y_bin = (y_cont >= threshold).astype(int)

        train_idx, test_idx = split_by_prompt(meta, test_ratio=0.2)
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y_bin[train_idx], y_bin[test_idx]
        y_tr_cont, y_te_cont = y_cont[train_idx], y_cont[test_idx]

        if method == 'logistic':
            res = train_logistic(X_tr, y_tr, y_tr_cont, X_te, y_te, y_te_cont, reg)
        elif method == 'ridge':
            res = train_ridge(X_tr, y_tr, y_tr_cont, X_te, y_te, y_te_cont, reg)
        elif method == 'softmax_weighted':
            gids = get_group_ids(meta, train_idx)
            res = train_softmax_weighted(X_tr, y_tr, y_tr_cont, X_te, y_te, y_te_cont, gids, reg)

        layers.append(layer_idx)
        tr_accs.append(res['train_acc'])
        te_accs.append(res['test_acc'])
        tr_aucs.append(res['train_auc'])
        te_aucs.append(res['test_auc'])
        tr_r2.append(res['train_r2'])
        te_r2.append(res['test_r2'])
        tr_pearson.append(res['train_pearson'])
        te_pearson.append(res['test_pearson'])

        # Track best by R²
        if res['test_r2'] > best_r2:
            best_r2 = res['test_r2']
            best_layer = layer_idx
            best_te_proba = res['test_proba']
            best_te_labels = y_te_cont
            best_tr_proba = res['train_proba']
            best_tr_labels = y_tr_cont

        print(f"  L{layer_idx:2d}: R²={res['test_r2']:.3f} (tr={res['train_r2']:.3f}), r={res['test_pearson']:.3f}, AUC={res['test_auc']:.3f}")

    print(f"\n  Best: L{best_layer} (R²={best_r2:.3f})")

    return MethodResult(
        method=method, layers=layers,
        train_accuracies=tr_accs, test_accuracies=te_accs,
        train_aucs=tr_aucs, test_aucs=te_aucs,
        train_r2=tr_r2, test_r2=te_r2,
        train_pearson=tr_pearson, test_pearson=te_pearson,
        best_layer=best_layer, best_test_r2=best_r2,
        best_layer_test_proba=best_te_proba, best_layer_test_labels=best_te_labels,
        best_layer_train_proba=best_tr_proba, best_layer_train_labels=best_tr_labels,
    )


def plot_layer_comparison(results, output_dir, metric='auc'):
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {'logistic': 'blue', 'ridge': 'green', 'softmax_weighted': 'red'}
    labels = {'logistic': 'Logistic', 'ridge': 'Ridge', 'softmax_weighted': 'Softmax-Weighted (CC++)'}

    metric_map = {
        'auc': ('test_aucs', 'Test AUC'),
        'accuracy': ('test_accuracies', 'Test Accuracy'),
        'r2': ('test_r2', 'Test R²'),
        'pearson': ('test_pearson', 'Test Pearson r'),
        'train_r2': ('train_r2', 'Train R²'),
        'train_pearson': ('train_pearson', 'Train Pearson r'),
    }
    attr, ylabel = metric_map.get(metric, ('test_aucs', 'Test AUC'))

    for r in results:
        vals = getattr(r, attr)
        ax.plot(r.layers, vals, color=colors.get(r.method, 'black'),
                label=f"{labels.get(r.method, r.method)} (best: L{r.best_layer})", linewidth=2)
        ax.scatter([r.best_layer], [vals[r.best_layer]], color=colors.get(r.method, 'black'), s=100, zorder=5)

    ax.set_xlabel('Layer')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{ylabel} by Layer')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'layer_comparison_{metric}.png', dpi=150)
    plt.close()
    print(f"Saved: layer_comparison_{metric}.png")


def plot_roc_curves(results, output_dir):
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = {'logistic': 'blue', 'ridge': 'green', 'softmax_weighted': 'red'}
    labels = {'logistic': 'Logistic', 'ridge': 'Ridge', 'softmax_weighted': 'Softmax-Weighted (CC++)'}

    for r in results:
        if r.best_layer_test_proba is None:
            continue
        # Binarize continuous labels for ROC
        y_true_bin = (r.best_layer_test_labels >= 0.5).astype(int)
        if len(np.unique(y_true_bin)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true_bin, r.best_layer_test_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors.get(r.method, 'black'),
                label=f"{labels.get(r.method, r.method)} (L{r.best_layer}, AUC={roc_auc:.3f})", linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves at Best Layer')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_comparison.png', dpi=150)
    plt.close()
    print(f"Saved: roc_comparison.png")


def plot_scatter_comparison(results, output_dir):
    """Scatter plot of predictions vs actual p_score for best layers."""
    n_methods = len(results)
    fig, axes = plt.subplots(2, n_methods, figsize=(5*n_methods, 10))
    colors = {'logistic': 'blue', 'ridge': 'green', 'softmax_weighted': 'red'}
    labels = {'logistic': 'Logistic', 'ridge': 'Ridge', 'softmax_weighted': 'Softmax-Weighted (CC++)'}

    for i, r in enumerate(results):
        if r.best_layer_test_proba is None:
            continue

        # Test set
        ax = axes[0, i] if n_methods > 1 else axes[0]
        ax.scatter(r.best_layer_test_labels, r.best_layer_test_proba,
                   alpha=0.3, c=colors.get(r.method, 'black'), s=10)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
        ax.set_xlabel('Actual p_score')
        ax.set_ylabel('Predicted')
        ax.set_title(f'{labels.get(r.method, r.method)} L{r.best_layer} (Test)\nR²={r.test_r2[r.best_layer]:.3f}, r={r.test_pearson[r.best_layer]:.3f}')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

        # Train set
        ax = axes[1, i] if n_methods > 1 else axes[1]
        ax.scatter(r.best_layer_train_labels, r.best_layer_train_proba,
                   alpha=0.3, c=colors.get(r.method, 'black'), s=10)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
        ax.set_xlabel('Actual p_score')
        ax.set_ylabel('Predicted')
        ax.set_title(f'{labels.get(r.method, r.method)} L{r.best_layer} (Train)\nR²={r.train_r2[r.best_layer]:.3f}, r={r.train_pearson[r.best_layer]:.3f}')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(output_dir / 'scatter_comparison.png', dpi=150)
    plt.close()
    print(f"Saved: scatter_comparison.png")


def get_num_layers(activation_dir):
    pt_file = next(activation_dir.glob("*.pt"))
    data = torch.load(pt_file, map_location='cpu', weights_only=False)
    n = data['activations'].shape[0]
    del data
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--experiment", type=str, default="math_staged")
    parser.add_argument("--activation-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--regularization", type=float, default=1.0)
    parser.add_argument("--methods", nargs='+', default=['logistic', 'ridge', 'softmax_weighted'])
    args = parser.parse_args()

    if args.activation_dir:
        activation_dir = Path(args.activation_dir)
    else:
        paths = get_experiment_paths(args.model, args.experiment)
        activation_dir = Path(paths["activations_dir"])

    output_dir = Path(args.output_dir) if args.output_dir else activation_dir.parent / "probe_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Activation dir: {activation_dir}")
    print(f"Output dir: {output_dir}")

    num_layers = get_num_layers(activation_dir)
    num_files = len(list(activation_dir.glob('*.pt')))
    print(f"Model has {num_layers} layers, {num_files} files")

    # Load ALL data with parallel loading
    print("\n" + "="*60)
    print("Loading all data (parallel)...")
    print("="*60)
    all_layer_data = load_all_layers_parallel(activation_dir, num_layers, num_workers=8)
    print("Done loading!")

    # Train each method
    results = []
    for method in args.methods:
        res = train_method_all_layers(method, all_layer_data, num_layers, args.threshold, args.regularization)
        results.append(res)

    # Plots
    print("\n" + "="*60)
    print("Generating plots...")
    print("="*60)
    plot_layer_comparison(results, output_dir, 'r2')
    plot_layer_comparison(results, output_dir, 'pearson')
    plot_layer_comparison(results, output_dir, 'train_r2')
    plot_layer_comparison(results, output_dir, 'auc')
    plot_layer_comparison(results, output_dir, 'accuracy')
    plot_roc_curves(results, output_dir)
    plot_scatter_comparison(results, output_dir)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for r in results:
        bl = r.best_layer
        print(f"{r.method.upper()}: Best L{bl}")
        print(f"    Test:  R²={r.test_r2[bl]:.4f}, r={r.test_pearson[bl]:.4f}, AUC={r.test_aucs[bl]:.4f}, Acc={r.test_accuracies[bl]:.4f}")
        print(f"    Train: R²={r.train_r2[bl]:.4f}, r={r.train_pearson[bl]:.4f}, AUC={r.train_aucs[bl]:.4f}, Acc={r.train_accuracies[bl]:.4f}")

    # Save JSON
    with open(output_dir / 'comparison_results.json', 'w') as f:
        json.dump({
            'methods': args.methods,
            'num_layers': num_layers,
            'results': [{
                'method': r.method,
                'best_layer': r.best_layer,
                'best_test_r2': r.best_test_r2,
                # Test metrics
                'test_r2': r.test_r2,
                'test_pearson': r.test_pearson,
                'test_aucs': r.test_aucs,
                'test_accuracies': r.test_accuracies,
                # Train metrics
                'train_r2': r.train_r2,
                'train_pearson': r.train_pearson,
                'train_aucs': r.train_aucs,
                'train_accuracies': r.train_accuracies,
            } for r in results]
        }, f, indent=2)

    print(f"\nSaved to: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()

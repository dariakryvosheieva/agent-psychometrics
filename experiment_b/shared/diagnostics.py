"""Diagnostic plotting functions for Feature-IRT analysis.

This module handles:
- Grid search heatmaps
- Training loss curves
- Loss component breakdowns
- Diagnostic summary printing
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def plot_grid_search_heatmap(
    grid_diagnostics: List[Dict],
    source_name: str,
    output_path: Path,
) -> None:
    """Plot AUC heatmap for grid search results.

    Args:
        grid_diagnostics: List of diagnostic dicts from grid search.
        source_name: Feature source name (e.g., 'Embedding', 'LLM Judge').
        output_path: Path to save the heatmap image.
    """
    import matplotlib.pyplot as plt

    subset = [d for d in grid_diagnostics if d['source'] == source_name]
    if not subset:
        return

    # Create heatmap data
    l2_weights = sorted(set(d['l2_weight'] for d in subset))
    l2_residuals = sorted(set(d['l2_residual'] for d in subset))

    auc_matrix = np.full((len(l2_residuals), len(l2_weights)), np.nan)
    for d in subset:
        i = l2_residuals.index(d['l2_residual'])
        j = l2_weights.index(d['l2_weight'])
        auc_matrix[i, j] = d['auc']

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(auc_matrix, aspect='auto', cmap='viridis')
    ax.set_xticks(range(len(l2_weights)))
    ax.set_xticklabels([f"{w:.0e}" for w in l2_weights], rotation=45)
    ax.set_yticks(range(len(l2_residuals)))
    ax.set_yticklabels([f"{r:.0e}" for r in l2_residuals])
    ax.set_xlabel('l2_weight')
    ax.set_ylabel('l2_residual')
    ax.set_title(f'{source_name}: AUC vs Hyperparameters')
    plt.colorbar(im, ax=ax, label='AUC')

    # Add text annotations
    for i in range(len(l2_residuals)):
        for j in range(len(l2_weights)):
            if not np.isnan(auc_matrix[i, j]):
                ax.text(j, i, f'{auc_matrix[i, j]:.3f}',
                        ha='center', va='center', fontsize=8,
                        color='white' if auc_matrix[i, j] < 0.75 else 'black')

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_training_loss_curves(
    grid_diagnostics: List[Dict],
    output_path: Path,
) -> None:
    """Plot training loss curves for best config of each source.

    Args:
        grid_diagnostics: List of diagnostic dicts from grid search.
        output_path: Path to save the loss curves image.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, source_name in zip(axes, ['Embedding', 'LLM Judge']):
        subset = [d for d in grid_diagnostics if d['source'] == source_name]
        if not subset:
            ax.set_title(f'{source_name}: No data')
            continue

        # Find best config for this source
        best = max(subset, key=lambda x: x['auc'] or 0)
        loss_history = best['loss_history']

        ax.plot(loss_history, linewidth=1.5)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title(f'{source_name} Training Loss\n(l2_w={best["l2_weight"]}, l2_r={best["l2_residual"]}, AUC={best["auc"]:.4f})')
        ax.grid(True, alpha=0.3)

        # Mark final loss
        if loss_history:
            ax.axhline(y=loss_history[-1], color='r', linestyle='--', alpha=0.5,
                       label=f'Final: {loss_history[-1]:.4f}')
            ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_loss_components(
    grid_diagnostics: List[Dict],
    output_path: Path,
) -> None:
    """Plot stacked loss component breakdown for best config of each source.

    Args:
        grid_diagnostics: List of diagnostic dicts from grid search.
        output_path: Path to save the loss components image.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, source_name in zip(axes, ['Embedding', 'LLM Judge']):
        subset = [d for d in grid_diagnostics if d['source'] == source_name]
        if not subset:
            ax.set_title(f'{source_name}: No data')
            continue

        best = max(subset, key=lambda x: x['auc'] or 0)
        components = best['loss_components']
        if not components:
            ax.set_title(f'{source_name}: No loss components')
            continue

        iters = [c['iter'] for c in components]
        nll = [c['nll'] for c in components]
        weight_reg = [c['weight_reg'] for c in components]
        residual_reg = [c['residual_reg'] for c in components]
        ability_reg = [c['ability_reg'] for c in components]

        ax.stackplot(iters, nll, weight_reg, residual_reg, ability_reg,
                     labels=['NLL', 'Weight Reg', 'Residual Reg', 'Ability Reg'],
                     alpha=0.7)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss Component')
        ax.set_title(f'{source_name} Loss Components')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def print_diagnostic_summary(grid_diagnostics: List[Dict]) -> None:
    """Print diagnostic summary tables.

    Args:
        grid_diagnostics: List of diagnostic dicts from grid search.
    """
    # Print contribution analysis summary
    print("\n" + "=" * 90)
    print("FEATURE VS RESIDUAL CONTRIBUTION ANALYSIS (Best Config Per Source)")
    print("=" * 90)
    print(f"{'Source':<20} {'Feature Var %':>15} {'Residual Var %':>15} {'Covariance':>12} {'AUC':>10}")
    print("-" * 90)
    for source_name in ['Embedding', 'LLM Judge']:
        subset = [d for d in grid_diagnostics if d['source'] == source_name]
        if not subset:
            print(f"{source_name:<20} {'N/A':>15} {'N/A':>15} {'N/A':>12} {'N/A':>10}")
            continue

        best = max(subset, key=lambda x: x['auc'] or 0)
        contrib = best.get('contributions', {})
        feat_pct = contrib.get('feature_ratio', 0) * 100
        res_pct = contrib.get('residual_ratio', 0) * 100
        cov = contrib.get('covariance', 0)
        print(f"{source_name:<20} {feat_pct:>14.1f}% {res_pct:>14.1f}% {cov:>12.4f} {best['auc']:>10.4f}")

    # Print grid search summary table
    print("\n" + "=" * 90)
    print("GRID SEARCH RESULTS SUMMARY")
    print("=" * 90)
    for source_name in ['Embedding', 'LLM Judge']:
        subset = [d for d in grid_diagnostics if d['source'] == source_name]
        if not subset:
            continue

        print(f"\n{source_name}:")
        print(f"{'l2_weight':>12} {'l2_residual':>12} {'AUC':>10} {'Iters':>8} {'Final Loss':>12} {'Feat %':>10} {'Resid %':>10}")
        print("-" * 90)

        # Sort by AUC descending
        for d in sorted(subset, key=lambda x: x['auc'] or 0, reverse=True):
            contrib = d.get('contributions', {})
            feat_pct = contrib.get('feature_ratio', 0) * 100
            res_pct = contrib.get('residual_ratio', 0) * 100
            print(f"{d['l2_weight']:>12.4f} {d['l2_residual']:>12.1f} {d['auc']:>10.4f} "
                  f"{d['n_iterations']:>8} {d['final_loss']:>12.4f} {feat_pct:>9.1f}% {res_pct:>9.1f}%")

    print("\n" + "=" * 90)


def save_and_plot_diagnostics(
    grid_diagnostics: List[Dict],
    output_dir: Path,
) -> None:
    """Save diagnostic data and generate all diagnostic plots.

    Args:
        grid_diagnostics: List of diagnostic dicts from grid search.
        output_dir: Directory to save outputs.
    """
    print("\n" + "=" * 90)
    print("DIAGNOSTIC MODE: FEATURE-IRT ANALYSIS")
    print("=" * 90)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save diagnostics JSON
    diagnostics_path = output_dir / "feature_irt_diagnostics.json"
    with open(diagnostics_path, "w") as f:
        json.dump(grid_diagnostics, f, indent=2, default=str)
    print(f"\nSaved diagnostics to: {diagnostics_path}")

    # Plot grid search heatmaps
    for source_name in ['Embedding', 'LLM Judge']:
        heatmap_path = output_dir / f'grid_search_heatmap_{source_name.lower().replace(" ", "_")}.png'
        plot_grid_search_heatmap(grid_diagnostics, source_name, heatmap_path)
        if heatmap_path.exists():
            print(f"Saved heatmap: {heatmap_path}")

    # Plot training loss curves
    loss_curves_path = output_dir / 'training_loss_curves.png'
    plot_training_loss_curves(grid_diagnostics, loss_curves_path)
    print(f"Saved loss curves: {loss_curves_path}")

    # Plot loss components
    components_path = output_dir / 'loss_components.png'
    plot_loss_components(grid_diagnostics, components_path)
    print(f"Saved loss components: {components_path}")

    # Print summary tables
    print_diagnostic_summary(grid_diagnostics)

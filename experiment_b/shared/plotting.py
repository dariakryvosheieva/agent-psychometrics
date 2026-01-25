"""Plotting utilities for experiment_b threshold sweep.

Provides functions for generating threshold sweep plots showing
method performance across different pre-frontier thresholds.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Method display configuration
METHOD_STYLES: Dict[str, Dict[str, str]] = {
    "Oracle": {"color": "blue", "marker": "o"},
    "Baseline IRT": {"color": "orange", "marker": "s"},
    "Feature-IRT": {"color": "green", "marker": "^"},
}


def _get_method_style(method_name: str) -> Dict[str, str]:
    """Get plotting style for a method based on its name.

    Matches method names by prefix to handle variants like
    "Oracle (upper bound)" or "Baseline IRT (pre-frontier only)".
    """
    for key, style in METHOD_STYLES.items():
        if key in method_name:
            return style
    # Default style for unknown methods
    return {"color": "gray", "marker": "x"}


def plot_threshold_sweep_auc(
    df: pd.DataFrame,
    dataset_name: str,
    output_path: Path,
    y_limits: Optional[tuple] = None,
) -> None:
    """Generate threshold sweep AUC plot.

    Args:
        df: DataFrame with columns: threshold, method, mean_auc, sem
        dataset_name: Name of the dataset (for title)
        output_path: Path to save the plot
        y_limits: Optional (ymin, ymax) for y-axis
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get unique methods in their original order
    methods = df["method"].unique()

    for method_name in methods:
        method_df = df[df["method"] == method_name].sort_values("threshold")
        if method_df.empty:
            continue

        style = _get_method_style(method_name)

        # Use errorbar if SEM is available, otherwise just plot
        if "sem" in method_df.columns and not method_df["sem"].isna().all():
            ax.errorbar(
                method_df["threshold"] * 100,
                method_df["mean_auc"],
                yerr=method_df["sem"],
                label=method_name,
                color=style["color"],
                marker=style["marker"],
                capsize=3,
                linewidth=2,
                markersize=8,
            )
        else:
            ax.plot(
                method_df["threshold"] * 100,
                method_df["mean_auc"],
                label=method_name,
                color=style["color"],
                marker=style["marker"],
                linewidth=2,
                markersize=8,
            )

    # Configure plot
    ax.set_xlabel("Pre-frontier Threshold (%)", fontsize=12)
    ax.set_ylabel("Mean Per-Agent AUC", fontsize=12)
    ax.set_title(f"Threshold Sweep: {dataset_name}", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 32)

    if y_limits:
        ax.set_ylim(y_limits)
    else:
        ax.set_ylim(0.4, 1.0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved AUC plot to {output_path}")


def plot_threshold_sweep_mae(
    df: pd.DataFrame,
    dataset_name: str,
    output_path: Path,
    y_limits: Optional[tuple] = None,
) -> None:
    """Generate threshold sweep date forecast MAE plot.

    Args:
        df: DataFrame with columns: threshold, method, mae_days
        dataset_name: Name of the dataset (for title)
        output_path: Path to save the plot
        y_limits: Optional (ymin, ymax) for y-axis
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get unique methods in their original order
    methods = df["method"].unique()

    for method_name in methods:
        method_df = df[df["method"] == method_name].sort_values("threshold")
        if method_df.empty:
            continue

        # Skip if no MAE data
        if "mae_days" not in method_df.columns or method_df["mae_days"].isna().all():
            continue

        style = _get_method_style(method_name)

        ax.plot(
            method_df["threshold"] * 100,
            method_df["mae_days"],
            label=method_name,
            color=style["color"],
            marker=style["marker"],
            linewidth=2,
            markersize=8,
        )

    # Configure plot
    ax.set_xlabel("Pre-frontier Threshold (%)", fontsize=12)
    ax.set_ylabel("Date Forecast MAE (days)", fontsize=12)
    ax.set_title(f"Date Forecasting: {dataset_name}", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 32)

    if y_limits:
        ax.set_ylim(y_limits)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved MAE plot to {output_path}")


def plot_combined_threshold_sweep(
    df: pd.DataFrame,
    dataset_name: str,
    output_path: Path,
) -> None:
    """Generate combined threshold sweep plot with AUC and MAE side by side.

    Args:
        df: DataFrame with columns: threshold, method, mean_auc, sem, mae_days
        dataset_name: Name of the dataset (for title)
        output_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    methods = df["method"].unique()

    # Left plot: AUC
    for method_name in methods:
        method_df = df[df["method"] == method_name].sort_values("threshold")
        if method_df.empty:
            continue

        style = _get_method_style(method_name)

        if "sem" in method_df.columns and not method_df["sem"].isna().all():
            ax1.errorbar(
                method_df["threshold"] * 100,
                method_df["mean_auc"],
                yerr=method_df["sem"],
                label=method_name,
                color=style["color"],
                marker=style["marker"],
                capsize=3,
                linewidth=2,
                markersize=8,
            )
        else:
            ax1.plot(
                method_df["threshold"] * 100,
                method_df["mean_auc"],
                label=method_name,
                color=style["color"],
                marker=style["marker"],
                linewidth=2,
                markersize=8,
            )

    ax1.set_xlabel("Pre-frontier Threshold (%)", fontsize=12)
    ax1.set_ylabel("Mean Per-Agent AUC", fontsize=12)
    ax1.set_title("Mean Per-Agent AUC", fontsize=14)
    ax1.legend(loc="lower right", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-2, 32)
    ax1.set_ylim(0.4, 1.0)

    # Right plot: MAE
    for method_name in methods:
        method_df = df[df["method"] == method_name].sort_values("threshold")
        if method_df.empty:
            continue

        if "mae_days" not in method_df.columns or method_df["mae_days"].isna().all():
            continue

        style = _get_method_style(method_name)

        ax2.plot(
            method_df["threshold"] * 100,
            method_df["mae_days"],
            label=method_name,
            color=style["color"],
            marker=style["marker"],
            linewidth=2,
            markersize=8,
        )

    ax2.set_xlabel("Pre-frontier Threshold (%)", fontsize=12)
    ax2.set_ylabel("Date Forecast MAE (days)", fontsize=12)
    ax2.set_title("Date Forecast MAE", fontsize=14)
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-2, 32)

    fig.suptitle(f"Threshold Sweep: {dataset_name}", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved combined plot to {output_path}")


def plot_ability_vs_date(
    method_abilities: Dict[str, Dict[str, float]],
    agent_dates: Dict[str, str],
    fit_results: Dict[str, Dict[str, Any]],
    dataset_name: str,
    output_path: Path,
    reference_date: Optional[datetime] = None,
) -> None:
    """Generate ability-vs-date scatter plot with linear fit for each method.

    Args:
        method_abilities: Dict mapping method_name -> {agent_id: ability}
        agent_dates: Dict mapping agent_id -> date string (YYYYMMDD)
        fit_results: Dict mapping method_name -> fit stats (slope, intercept, r_squared, n_frontier_points)
                     Methods that failed to fit should have an empty dict or None.
        dataset_name: Name of the dataset (for title)
        output_path: Path to save the plot
        reference_date: Optional reference date for x-axis (days since). If None, uses earliest date.

    Raises:
        ValueError: If any agent in method_abilities is missing from agent_dates
    """
    from experiment_b.shared.date_forecasting import parse_date

    # Validate that all agents have dates
    for method_name, abilities in method_abilities.items():
        missing_dates = [a for a in abilities if a not in agent_dates]
        if missing_dates:
            raise ValueError(
                f"Method '{method_name}' has {len(missing_dates)} agents without dates: "
                f"{missing_dates[:5]}{'...' if len(missing_dates) > 5 else ''}"
            )

    n_methods = len(method_abilities)
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))

    if n_methods == 1:
        axes = [axes]

    # Determine global reference date if not provided
    if reference_date is None:
        all_dates = []
        for method_name, abilities in method_abilities.items():
            for agent_id in abilities:
                all_dates.append(parse_date(agent_dates[agent_id]))
        reference_date = min(all_dates) if all_dates else datetime(2025, 1, 1)

    for ax, (method_name, abilities) in zip(axes, method_abilities.items()):
        style = _get_method_style(method_name)

        # Build data for this method
        days = []
        thetas = []
        for agent in abilities:
            date = parse_date(agent_dates[agent])
            days.append((date - reference_date).days)
            thetas.append(abilities[agent])

        if not days:
            ax.set_title(f"{method_name}\n(no data)")
            continue

        # Scatter plot of all agents
        ax.scatter(days, thetas, alpha=0.6, color=style["color"], s=50, label="Agents")

        # Compute and plot frontier trajectory (cumulative max)
        df = pd.DataFrame({"days": days, "theta": thetas}).sort_values("days")
        df_grouped = df.groupby("days").agg({"theta": "max"}).reset_index()
        df_grouped = df_grouped.sort_values("days")
        df_grouped["frontier_theta"] = df_grouped["theta"].cummax()

        # Highlight frontier points
        frontier_changes = df_grouped[df_grouped["frontier_theta"].diff().fillna(1) > 0]
        ax.scatter(
            frontier_changes["days"],
            frontier_changes["frontier_theta"],
            color="red",
            s=100,
            marker="*",
            label=f"Frontier ({len(frontier_changes)} pts)",
            zorder=5,
        )

        # Draw the linear fit if available
        fit = fit_results.get(method_name)
        if fit and "slope" in fit and "r_squared" in fit:
            x_range = np.array([min(days), max(days)])
            y_fit = fit["slope"] * x_range + fit["intercept"]
            ax.plot(
                x_range,
                y_fit,
                "k--",
                linewidth=2,
                label=f"Fit (R²={fit['r_squared']:.3f})",
            )
            ax.set_title(
                f"{method_name}\nR²={fit['r_squared']:.3f}, slope={fit['slope']:.4f}/day",
                fontsize=11,
            )
        else:
            ax.set_title(f"{method_name}\n(fit failed: <2 frontier pts)", fontsize=11)

        ax.set_xlabel("Days since reference", fontsize=10)
        ax.set_ylabel("Ability (theta)", fontsize=10)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Ability vs Date: {dataset_name}", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved ability-vs-date plot to {output_path}")

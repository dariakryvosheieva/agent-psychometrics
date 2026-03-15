#!/usr/bin/env python3
"""Diagnose date prediction bottleneck in Experiment B.

Analyzes:
1. R² of ability-over-time fit for each method (using pre-frontier agents only)
2. Whether frontier (post-cutoff) agents are outliers from the pre-frontier trend

This helps identify whether poor date predictions come from:
- Bad linear fit of ability vs time
- Frontier agents not following the pre-frontier trend

Usage:
    python -m experiment_appendix_h_hard_tasks.diagnose_date_error
    python -m experiment_appendix_h_hard_tasks.diagnose_date_error --dataset terminalbench
"""

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from experiment_appendix_h_hard_tasks import get_dataset_config, list_datasets
from experiment_appendix_h_hard_tasks.shared import (
    load_and_prepare_data,
    split_agents_by_dates,
    parse_date,
)


@dataclass
class PrefrontierFitResult:
    """Result from fitting ability-over-time on pre-frontier agents only."""

    slope: float
    intercept: float
    r_squared: float
    reference_date: datetime
    n_pre_frontier: int
    n_frontier_points: int  # Number of cummax improvement points


@dataclass
class OutlierAnalysisResult:
    """Result from analyzing whether post-frontier agents are outliers."""

    mean_residual: float  # Average (actual - predicted) for post-frontier
    rmse_residual: float  # RMSE of residuals for post-frontier
    n_post_frontier: int
    pct_above_trend: float  # % of post-frontier agents above extrapolated trend
    max_positive_residual: float
    max_negative_residual: float


def fit_ability_prefrontier_only(
    abilities: Dict[str, float],
    agent_dates: Dict[str, str],
    cutoff_date: str,
) -> Tuple[PrefrontierFitResult, pd.DataFrame]:
    """Fit linear model on pre-frontier agents only.

    Uses the cumulative max (frontier trajectory) approach from Experiment D,
    but only on pre-frontier agents.

    Args:
        abilities: Dict mapping agent_id -> theta (ability)
        agent_dates: Dict mapping agent_id -> date string (YYYYMMDD)
        cutoff_date: Cutoff date string (YYYYMMDD)

    Returns:
        Tuple of (fit_result, all_agents_df):
            - fit_result: Linear fit parameters from pre-frontier agents
            - all_agents_df: DataFrame with all agents, their abilities, dates,
              and predicted abilities from the pre-frontier fit
    """
    # Build dataframe of all agents with dates
    agent_data = []
    for agent_id, theta in abilities.items():
        if agent_id not in agent_dates:
            continue
        date_str = agent_dates[agent_id]
        date = parse_date(date_str)
        is_pre_frontier = date_str < cutoff_date
        agent_data.append({
            "agent_id": agent_id,
            "theta": theta,
            "date": date,
            "date_str": date_str,
            "is_pre_frontier": is_pre_frontier,
        })

    if len(agent_data) < 3:
        raise ValueError(f"Insufficient agents with dates: {len(agent_data)}")

    df = pd.DataFrame(agent_data)
    df = df.sort_values("date")

    # Filter to pre-frontier only for fitting
    df_pre = df[df["is_pre_frontier"]].copy()
    if len(df_pre) < 3:
        raise ValueError(f"Insufficient pre-frontier agents: {len(df_pre)}")

    reference_date = df_pre["date"].min()

    # Group by date, take max ability per date
    df_pre_grouped = df_pre.groupby("date").agg({"theta": "max"}).reset_index()
    df_pre_grouped = df_pre_grouped.sort_values("date")

    # Compute cumulative max (frontier trajectory)
    df_pre_grouped["frontier_theta"] = df_pre_grouped["theta"].cummax()

    # Find points where frontier improved
    frontier_changes = df_pre_grouped[
        df_pre_grouped["frontier_theta"].diff().fillna(1) > 0
    ].copy()

    if len(frontier_changes) < 2:
        raise ValueError(f"Insufficient frontier improvement points: {len(frontier_changes)}")

    # Convert dates to days since reference
    frontier_x = np.array([(d - reference_date).days for d in frontier_changes["date"]])
    frontier_y = frontier_changes["frontier_theta"].values

    # Fit linear regression on frontier points
    slope, intercept, r_value, p_value, std_err = stats.linregress(frontier_x, frontier_y)

    fit_result = PrefrontierFitResult(
        slope=float(slope),
        intercept=float(intercept),
        r_squared=float(r_value**2),
        reference_date=reference_date,
        n_pre_frontier=len(df_pre),
        n_frontier_points=len(frontier_changes),
    )

    # Add predicted theta to all agents using pre-frontier fit
    df["days_since_ref"] = df["date"].apply(lambda d: (d - reference_date).days)
    df["theta_predicted"] = slope * df["days_since_ref"] + intercept
    df["residual"] = df["theta"] - df["theta_predicted"]

    return fit_result, df


def analyze_frontier_outliers(
    all_agents_df: pd.DataFrame,
) -> OutlierAnalysisResult:
    """Analyze if post-frontier agents are outliers from pre-frontier trend.

    Args:
        all_agents_df: DataFrame from fit_ability_prefrontier_only() with
            'is_pre_frontier', 'theta', 'theta_predicted', 'residual' columns

    Returns:
        OutlierAnalysisResult with statistics about post-frontier agents
    """
    df_post = all_agents_df[~all_agents_df["is_pre_frontier"]]

    if len(df_post) == 0:
        return OutlierAnalysisResult(
            mean_residual=float("nan"),
            rmse_residual=float("nan"),
            n_post_frontier=0,
            pct_above_trend=float("nan"),
            max_positive_residual=float("nan"),
            max_negative_residual=float("nan"),
        )

    residuals = df_post["residual"].values
    mean_residual = float(np.mean(residuals))
    rmse_residual = float(np.sqrt(np.mean(residuals**2)))
    pct_above = float(np.mean(residuals > 0) * 100)

    return OutlierAnalysisResult(
        mean_residual=mean_residual,
        rmse_residual=rmse_residual,
        n_post_frontier=len(df_post),
        pct_above_trend=pct_above,
        max_positive_residual=float(np.max(residuals)),
        max_negative_residual=float(np.min(residuals)),
    )


def plot_diagnostic(
    method_results: Dict[str, Tuple[PrefrontierFitResult, pd.DataFrame, OutlierAnalysisResult]],
    cutoff_date: str,
    output_path: Path,
):
    """Create diagnostic plot showing ability vs time for each method.

    Creates a 2x2 subplot with:
    - Top-left: Oracle
    - Top-right: Baseline IRT
    - Bottom-left: First other method (if any)
    - Bottom-right: Residual histogram for all methods
    """
    methods = list(method_results.keys())
    n_methods = len(methods)

    # Select up to 3 methods for individual plots
    plot_methods = methods[:3]
    cutoff_datetime = parse_date(cutoff_date)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot individual methods
    for idx, method_name in enumerate(plot_methods):
        row, col = divmod(idx, 2)
        if idx >= 3:
            break
        ax = axes[row, col]

        fit_result, df, outlier_result = method_results[method_name]

        # Pre-frontier agents (gray)
        df_pre = df[df["is_pre_frontier"]]
        ax.scatter(
            df_pre["date"], df_pre["theta"],
            alpha=0.5, s=30, color="#94a3b8", label="Pre-frontier agents", zorder=2
        )

        # Post-frontier agents (colored)
        df_post = df[~df["is_pre_frontier"]]
        ax.scatter(
            df_post["date"], df_post["theta"],
            alpha=0.7, s=40, color="#dc2626", label="Post-frontier agents", zorder=3
        )

        # Pre-frontier linear fit (solid)
        date_range = pd.date_range(df["date"].min(), cutoff_datetime, periods=50)
        days_range = np.array([(d - fit_result.reference_date).days for d in date_range])
        theta_fit = fit_result.slope * days_range + fit_result.intercept
        ax.plot(date_range, theta_fit, linewidth=2, color="#1d4ed8", label="Pre-frontier fit", zorder=4)

        # Extrapolated trend (dashed)
        date_range_post = pd.date_range(cutoff_datetime, df["date"].max(), periods=50)
        days_range_post = np.array([(d - fit_result.reference_date).days for d in date_range_post])
        theta_extrap = fit_result.slope * days_range_post + fit_result.intercept
        ax.plot(date_range_post, theta_extrap, linewidth=2, color="#1d4ed8", linestyle="--", label="Extrapolated", zorder=4)

        # Cutoff date vertical line
        ax.axvline(cutoff_datetime, color="#6b7280", linestyle=":", linewidth=1.5, label=f"Cutoff ({cutoff_date})")

        # Stats box
        stats_text = (
            f"R² (pre-frontier): {fit_result.r_squared:.3f}\n"
            f"Slope: {fit_result.slope:.4f} θ/day\n"
            f"Mean residual (post): {outlier_result.mean_residual:+.3f}\n"
            f"RMSE (post): {outlier_result.rmse_residual:.3f}\n"
            f"% above trend: {outlier_result.pct_above_trend:.0f}%"
        )
        ax.text(
            0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9)
        )

        ax.set_xlabel("Agent Submission Date", fontsize=10)
        ax.set_ylabel("IRT Ability (θ)", fontsize=10)
        ax.set_title(method_name, fontsize=11)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="lower right", fontsize=8)

    # Bottom-right: Residual histogram
    ax = axes[1, 1]
    colors = ["#1d4ed8", "#dc2626", "#059669", "#7c3aed"]
    for idx, method_name in enumerate(methods[:4]):
        _, df, _ = method_results[method_name]
        df_post = df[~df["is_pre_frontier"]]
        if len(df_post) > 0:
            ax.hist(
                df_post["residual"], bins=20, alpha=0.5,
                label=method_name, color=colors[idx % len(colors)]
            )
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Residual (actual - predicted θ)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Post-Frontier Agent Residuals", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.suptitle(
        f"Date Prediction Diagnostic: Are Frontier Agents Outliers?\n"
        f"(Cutoff: {cutoff_date})",
        fontsize=12, y=1.02
    )
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved diagnostic plot: {output_path}")
    plt.close()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Diagnose date prediction bottleneck in Experiment B"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="swebench",
        choices=list_datasets(),
        help="Dataset to analyze (default: swebench)",
    )
    parser.add_argument(
        "--cutoff_date",
        type=str,
        default=None,
        help="Override frontier cutoff date (YYYYMMDD format)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output/figures"),
        help="Directory for output plots",
    )
    parser.add_argument(
        "--frontier_definitions",
        type=str,
        nargs="+",
        default=["passrate", "irt"],
        help="Frontier definitions (for load_and_prepare_data compatibility)",
    )
    # Add stub args that load_and_prepare_data expects
    parser.add_argument("--oracle_irt_path", type=Path, default=None)
    parser.add_argument("--oracle_abilities_path", type=Path, default=None)
    parser.add_argument("--baseline_irt_path", type=Path, default=None)
    parser.add_argument("--pre_threshold", type=float, default=None)
    parser.add_argument("--post_threshold", type=float, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("DATE PREDICTION BOTTLENECK DIAGNOSTIC")
    print("=" * 80)

    # Load dataset configuration and data
    print(f"\nLoading dataset: {args.dataset}")
    config = get_dataset_config(args.dataset)
    data = load_and_prepare_data(args, config)

    cutoff_date = data.cutoff_date
    print(f"Cutoff date: {cutoff_date}")
    print(f"Pre-frontier agents: {len(data.pre_frontier_agents)}")
    print(f"Post-frontier agents: {len(data.post_frontier_agents)}")

    # Collect abilities from all methods
    method_abilities: Dict[str, Dict[str, float]] = {}

    # Oracle (always available)
    method_abilities["Oracle"] = data.oracle_abilities["theta"].to_dict()

    # Baseline IRT
    if data.baseline_abilities is not None:
        method_abilities["Baseline IRT"] = data.baseline_abilities["theta"].to_dict()

    print(f"\nMethods with abilities: {list(method_abilities.keys())}")

    # Analyze each method
    method_results: Dict[str, Tuple[PrefrontierFitResult, pd.DataFrame, OutlierAnalysisResult]] = {}

    for method_name, abilities in method_abilities.items():
        try:
            fit_result, all_agents_df = fit_ability_prefrontier_only(
                abilities, config.agent_dates, cutoff_date
            )
            outlier_result = analyze_frontier_outliers(all_agents_df)
            method_results[method_name] = (fit_result, all_agents_df, outlier_result)
        except Exception as e:
            print(f"  Warning: Could not analyze {method_name}: {e}")

    # Print results tables
    print("\n" + "=" * 80)
    print("ABILITY-OVER-TIME FIT QUALITY (Pre-frontier agents only)")
    print("=" * 80)
    print(f"{'Method':<35} {'R²':>8} {'Slope (θ/day)':>14} {'n_pre':>8} {'n_frontier':>12}")
    print("-" * 80)

    for method_name, (fit_result, _, _) in method_results.items():
        print(
            f"{method_name:<35} "
            f"{fit_result.r_squared:>8.4f} "
            f"{fit_result.slope:>14.6f} "
            f"{fit_result.n_pre_frontier:>8} "
            f"{fit_result.n_frontier_points:>12}"
        )

    print("\n" + "=" * 80)
    print("FRONTIER AGENT OUTLIER ANALYSIS")
    print("=" * 80)
    print(f"{'Method':<35} {'Mean Res':>10} {'RMSE':>8} {'% Above':>10} {'n_post':>8}")
    print("-" * 80)

    for method_name, (_, _, outlier_result) in method_results.items():
        print(
            f"{method_name:<35} "
            f"{outlier_result.mean_residual:>+10.4f} "
            f"{outlier_result.rmse_residual:>8.4f} "
            f"{outlier_result.pct_above_trend:>9.1f}% "
            f"{outlier_result.n_post_frontier:>8}"
        )

    # Print interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    # Check if R² is consistently high
    r2_values = [fit.r_squared for fit, _, _ in method_results.values()]
    mean_r2 = np.mean(r2_values)
    print(f"\nLinear fit quality (R²): mean = {mean_r2:.3f}, range = [{min(r2_values):.3f}, {max(r2_values):.3f}]")

    if mean_r2 > 0.9:
        print("  → R² is HIGH: Linear assumption holds well for pre-frontier agents")
    elif mean_r2 > 0.7:
        print("  → R² is MODERATE: Some non-linearity in pre-frontier ability growth")
    else:
        print("  → R² is LOW: Linear assumption may be problematic")

    # Check if frontier agents are outliers (skip methods with no post-frontier agents)
    valid_residuals = [
        outlier.mean_residual
        for _, _, outlier in method_results.values()
        if outlier.n_post_frontier > 0 and not np.isnan(outlier.mean_residual)
    ]
    if valid_residuals:
        avg_mean_residual = np.mean(valid_residuals)
        print(f"\nFrontier agent residuals: avg mean residual = {avg_mean_residual:+.3f}")
    else:
        avg_mean_residual = 0.0
        print("\nNo valid frontier agent residuals to analyze")

    if avg_mean_residual > 0.3:
        print("  → Frontier agents are ABOVE the extrapolated trend")
        print("  → Ability is growing FASTER than pre-frontier trend predicted")
        print("  → Date predictions will be LATE (underestimate ability)")
    elif avg_mean_residual < -0.3:
        print("  → Frontier agents are BELOW the extrapolated trend")
        print("  → Ability is growing SLOWER than pre-frontier trend predicted")
        print("  → Date predictions will be EARLY (overestimate ability)")
    else:
        print("  → Frontier agents roughly follow the extrapolated trend")
        print("  → Linear extrapolation is reasonable")

    # Generate diagnostic plot
    output_path = args.output_dir / "date_forecast_diagnostic.png"
    plot_diagnostic(method_results, cutoff_date, output_path)


if __name__ == "__main__":
    main()

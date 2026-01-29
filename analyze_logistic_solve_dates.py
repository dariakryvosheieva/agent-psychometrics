#!/usr/bin/env python3
"""Estimate task solve dates using per-task logistic curve fitting on Pareto agents.

This script:
1. Identifies Pareto frontier agents (highest ability at each date)
2. Fits per-task logistic curves: P(solve) = 1 / (1 + exp(-k*(x - x0)))
3. Extracts x0 as the "solve date" (50% solve probability)
4. Correlates solve dates with oracle IRT difficulty (beta)

Usage:
    python analyze_logistic_solve_dates.py
    python analyze_logistic_solve_dates.py --plot_individual
    python analyze_logistic_solve_dates.py --output_dir chris_output/logistic_solve_dates
"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class LogisticFitResult:
    """Result from fitting a logistic curve to one task."""

    task_id: str
    x0: float  # Midpoint in days since reference
    k: float  # Steepness parameter
    solve_date: Optional[datetime]  # x0 converted to datetime
    n_pareto_agents: int
    n_solves: int
    converged: bool
    error_msg: Optional[str] = None


@dataclass
class ParetoAgentInfo:
    """Information about Pareto frontier agents."""

    agent_ids: List[str]  # Pareto agents in date order
    dates: List[datetime]  # Corresponding dates
    days: np.ndarray  # Days since reference
    thetas: np.ndarray  # Abilities (cumulative max)
    reference_date: datetime


# =============================================================================
# Core Functions
# =============================================================================


def logistic_function(x: np.ndarray, k: float, x0: float) -> np.ndarray:
    """Logistic curve: P(solve) = 1 / (1 + exp(-k*(x - x0)))"""
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def compute_pareto_agents(
    abilities: Dict[str, float],
    agent_dates: Dict[str, str],
) -> ParetoAgentInfo:
    """Compute Pareto frontier agents (cumulative max ability over time).

    Reuses logic from fit_ability_over_time() but returns more detailed info.
    """
    # Build dataframe
    agent_data = []
    for agent_id, theta in abilities.items():
        if agent_id not in agent_dates:
            continue
        date = datetime.strptime(agent_dates[agent_id], "%Y%m%d")
        agent_data.append({"agent_id": agent_id, "theta": theta, "date": date})

    if len(agent_data) < 2:
        raise ValueError(f"Insufficient agents with dates: {len(agent_data)}")

    df = pd.DataFrame(agent_data).sort_values("date")
    reference_date = df["date"].min()

    # Group by date, take max ability per date, keeping best agent ID
    df_grouped = (
        df.groupby("date")
        .agg(
            {
                "theta": "max",
                "agent_id": lambda x: x.iloc[
                    np.argmax([abilities[a] for a in x])
                ],  # Best agent
            }
        )
        .reset_index()
        .sort_values("date")
    )

    # Compute cumulative max (frontier trajectory)
    df_grouped["frontier_theta"] = df_grouped["theta"].cummax()

    # Keep only points where frontier improved
    frontier_changes = df_grouped[
        df_grouped["frontier_theta"].diff().fillna(1) > 0
    ].copy()

    return ParetoAgentInfo(
        agent_ids=frontier_changes["agent_id"].tolist(),
        dates=frontier_changes["date"].tolist(),
        days=np.array([(d - reference_date).days for d in frontier_changes["date"]]),
        thetas=frontier_changes["frontier_theta"].values,
        reference_date=reference_date,
    )


def fit_logistic_for_task(
    task_id: str,
    pareto_info: ParetoAgentInfo,
    responses: Dict[str, Dict[str, int]],
    min_data_points: int = 3,
) -> LogisticFitResult:
    """Fit a logistic curve for a single task using Pareto agent responses.

    Args:
        task_id: The task to fit
        pareto_info: Pareto agent information
        responses: Full response matrix
        min_data_points: Minimum number of data points for fitting

    Returns:
        LogisticFitResult with fit parameters or error info
    """
    # Collect (day, solve) pairs from Pareto agents
    days = []
    solves = []

    for agent_id, day in zip(pareto_info.agent_ids, pareto_info.days):
        if agent_id not in responses:
            continue
        if task_id not in responses[agent_id]:
            continue
        days.append(day)
        solves.append(responses[agent_id][task_id])

    n_points = len(days)
    n_solves = sum(solves)

    # Edge case: no data or too few data points
    if n_points < min_data_points:
        return LogisticFitResult(
            task_id=task_id,
            x0=np.nan,
            k=np.nan,
            solve_date=None,
            n_pareto_agents=n_points,
            n_solves=n_solves,
            converged=False,
            error_msg=f"Insufficient data points: {n_points} < {min_data_points}",
        )

    # Edge case: all solves or no solves (can't fit sigmoid)
    if n_solves == 0:
        return LogisticFitResult(
            task_id=task_id,
            x0=np.nan,
            k=np.nan,
            solve_date=None,
            n_pareto_agents=n_points,
            n_solves=0,
            converged=False,
            error_msg="Never solved by any Pareto agent",
        )

    if n_solves == n_points:
        return LogisticFitResult(
            task_id=task_id,
            x0=np.nan,
            k=np.nan,
            solve_date=None,
            n_pareto_agents=n_points,
            n_solves=n_solves,
            converged=False,
            error_msg="Solved by all Pareto agents",
        )

    days_arr = np.array(days)
    solves_arr = np.array(solves, dtype=float)

    # Initial guess: x0 = median day, k = 0.01 (shallow slope)
    x0_init = float(np.median(days_arr))
    k_init = 0.01

    # Define bounds for curve fitting
    x0_lower_bound = days_arr.min() - 100
    x0_upper_bound = days_arr.max() + 500

    try:
        popt, pcov = curve_fit(
            logistic_function,
            days_arr,
            solves_arr,
            p0=[k_init, x0_init],
            bounds=([0, x0_lower_bound], [1.0, x0_upper_bound]),
            maxfev=5000,
        )
        k, x0 = popt

        # Check if x0 hit the bounds (extrapolation, not interpolation)
        bound_tolerance = 1.0  # within 1 day of bound
        if x0 <= x0_lower_bound + bound_tolerance:
            return LogisticFitResult(
                task_id=task_id,
                x0=x0,
                k=k,
                solve_date=None,
                n_pareto_agents=n_points,
                n_solves=n_solves,
                converged=False,
                error_msg="x0 hit lower bound (task solved too early to estimate)",
            )
        if x0 >= x0_upper_bound - bound_tolerance:
            return LogisticFitResult(
                task_id=task_id,
                x0=x0,
                k=k,
                solve_date=None,
                n_pareto_agents=n_points,
                n_solves=n_solves,
                converged=False,
                error_msg="x0 hit upper bound (task solved too late to estimate)",
            )

        solve_date = pareto_info.reference_date + timedelta(days=int(round(x0)))

        return LogisticFitResult(
            task_id=task_id,
            x0=x0,
            k=k,
            solve_date=solve_date,
            n_pareto_agents=n_points,
            n_solves=n_solves,
            converged=True,
        )
    except RuntimeError as e:
        return LogisticFitResult(
            task_id=task_id,
            x0=np.nan,
            k=np.nan,
            solve_date=None,
            n_pareto_agents=n_points,
            n_solves=n_solves,
            converged=False,
            error_msg=str(e),
        )


def fit_all_tasks(
    task_ids: List[str],
    pareto_info: ParetoAgentInfo,
    responses: Dict[str, Dict[str, int]],
) -> List[LogisticFitResult]:
    """Fit logistic curves for all tasks."""
    results = []
    for task_id in task_ids:
        result = fit_logistic_for_task(task_id, pareto_info, responses)
        results.append(result)
    return results


# =============================================================================
# Plotting Functions
# =============================================================================


def plot_beta_vs_solve_date(
    results: List[LogisticFitResult],
    oracle_items: pd.DataFrame,
    output_path: Path,
) -> Dict[str, float]:
    """Plot oracle IRT beta vs logistic solve date (x0)."""
    # Filter to converged results with valid x0
    valid_results = [r for r in results if r.converged and np.isfinite(r.x0)]

    x0_days = []
    betas = []

    for r in valid_results:
        if r.task_id in oracle_items.index:
            x0_days.append(r.x0)
            betas.append(oracle_items.loc[r.task_id, "b"])

    x0_arr = np.array(x0_days)
    beta_arr = np.array(betas)

    # Compute correlations
    pearson_r, pearson_p = pearsonr(x0_arr, beta_arr)
    spearman_r, spearman_p = spearmanr(x0_arr, beta_arr)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(x0_arr, beta_arr, alpha=0.6, s=30)

    # Add linear fit
    slope, intercept = np.polyfit(x0_arr, beta_arr, 1)
    x_line = np.linspace(x0_arr.min(), x0_arr.max(), 100)
    ax.plot(
        x_line,
        slope * x_line + intercept,
        "r--",
        label=f"Linear fit: beta = {slope:.4f}*x0 + {intercept:.2f}",
    )

    ax.set_xlabel("Logistic Solve Date x0 (days since reference)", fontsize=12)
    ax.set_ylabel("Oracle IRT Difficulty (beta)", fontsize=12)
    ax.set_title(
        f"Oracle IRT Difficulty vs Logistic Solve Date\n"
        f"N={len(x0_arr)} tasks, Pearson r={pearson_r:.3f} (p={pearson_p:.4f}), "
        f"Spearman rho={spearman_r:.3f} (p={spearman_p:.4f})",
        fontsize=11,
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()

    return {
        "n_tasks": len(x0_arr),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "slope": slope,
        "intercept": intercept,
    }


def plot_individual_logistic_fit(
    result: LogisticFitResult,
    pareto_info: ParetoAgentInfo,
    responses: Dict[str, Dict[str, int]],
    output_path: Path,
) -> None:
    """Plot the logistic fit for a single task."""
    # Collect data points
    days = []
    solves = []
    for agent_id, day in zip(pareto_info.agent_ids, pareto_info.days):
        if agent_id not in responses:
            continue
        if result.task_id not in responses[agent_id]:
            continue
        days.append(day)
        solves.append(responses[agent_id][result.task_id])

    days_arr = np.array(days)
    solves_arr = np.array(solves)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Scatter actual data
    ax.scatter(days_arr, solves_arr, s=50, alpha=0.7, label="Pareto agent solves")

    # Plot fitted curve if we have fit parameters (even if not "converged")
    if np.isfinite(result.x0) and np.isfinite(result.k):
        # Extend x-range to include x0 so we can see the full sigmoid
        x_min = min(days_arr.min() - 50, result.x0 - 200)
        x_max = max(days_arr.max() + 50, result.x0 + 200)
        x_smooth = np.linspace(x_min, x_max, 300)
        y_smooth = logistic_function(x_smooth, result.k, result.x0)
        ax.plot(
            x_smooth,
            y_smooth,
            "r-",
            linewidth=2,
            label=f"Logistic fit (x0={result.x0:.1f}, k={result.k:.4f})",
        )
        ax.axvline(
            result.x0,
            color="green",
            linestyle="--",
            alpha=0.7,
            label=f"x0 = {result.x0:.1f} days",
        )

    ax.set_xlabel("Days since reference")
    ax.set_ylabel("Solve (0/1)")
    ax.set_title(
        f"Task: {result.task_id}\nN={result.n_pareto_agents} agents, {result.n_solves} solves"
    )
    ax.legend()
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100)
    plt.close()


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Estimate task solve dates via logistic fitting"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("chris_output/logistic_solve_dates"),
        help="Directory for output files",
    )
    parser.add_argument(
        "--plot_individual",
        action="store_true",
        help="Generate individual logistic fit plots per task",
    )
    parser.add_argument(
        "--max_individual_plots",
        type=int,
        default=50,
        help="Max number of individual plots to generate",
    )
    args = parser.parse_args()

    # === Load data ===
    from experiment_b.shared.evaluation import load_responses_dict
    from experiment_b.swebench.config import SWEBenchConfig

    config = SWEBenchConfig()

    print("Loading data...")
    responses = load_responses_dict(config.responses_path)
    oracle_items = pd.read_csv(config.oracle_irt_path, index_col=0)
    oracle_abilities = pd.read_csv(config.oracle_abilities_path, index_col=0)

    print(f"  Responses: {len(responses)} agents")
    print(f"  Oracle items: {len(oracle_items)} tasks")
    print(f"  Oracle abilities: {len(oracle_abilities)} agents")

    # Get agent dates
    agent_dates = config.get_agent_dates(list(responses.keys()))
    print(f"  Agents with dates: {len(agent_dates)}")

    # === Compute Pareto agents ===
    print("\nComputing Pareto frontier agents...")
    abilities_dict = oracle_abilities["theta"].to_dict()
    pareto_info = compute_pareto_agents(abilities_dict, agent_dates)
    print(f"  Pareto agents: {len(pareto_info.agent_ids)}")
    print(
        f"  Date range: {pareto_info.reference_date.strftime('%Y-%m-%d')} to "
        f"{pareto_info.dates[-1].strftime('%Y-%m-%d')}"
    )
    print(f"  Ability range: {pareto_info.thetas[0]:.3f} to {pareto_info.thetas[-1]:.3f}")

    # === Fit logistic curves ===
    print("\nFitting logistic curves for all tasks...")
    all_task_ids = list(oracle_items.index)
    results = fit_all_tasks(all_task_ids, pareto_info, responses)

    converged = [r for r in results if r.converged]
    valid_x0 = [r for r in converged if np.isfinite(r.x0)]
    never_solved = [r for r in results if r.error_msg == "Never solved by any Pareto agent"]
    always_solved = [r for r in results if r.error_msg == "Solved by all Pareto agents"]
    hit_lower_bound = [r for r in results if r.error_msg and "lower bound" in r.error_msg]
    hit_upper_bound = [r for r in results if r.error_msg and "upper bound" in r.error_msg]

    print(f"  Total tasks: {len(results)}")
    print(f"  Converged (interpolated): {len(converged)}")
    print(f"  Valid x0: {len(valid_x0)}")
    print(f"  Excluded - never solved: {len(never_solved)}")
    print(f"  Excluded - always solved: {len(always_solved)}")
    print(f"  Excluded - hit lower bound (too easy): {len(hit_lower_bound)}")
    print(f"  Excluded - hit upper bound (too hard): {len(hit_upper_bound)}")

    # === Save results ===
    print("\nSaving results...")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(
        [
            {
                "task_id": r.task_id,
                "x0_days": r.x0,
                "k": r.k,
                "solve_date": r.solve_date.strftime("%Y-%m-%d") if r.solve_date else None,
                "n_pareto_agents": r.n_pareto_agents,
                "n_solves": r.n_solves,
                "converged": r.converged,
                "error_msg": r.error_msg,
                "oracle_beta": (
                    oracle_items.loc[r.task_id, "b"]
                    if r.task_id in oracle_items.index
                    else None
                ),
            }
            for r in results
        ]
    )
    results_df.to_csv(args.output_dir / "results.csv", index=False)
    print(f"  Saved: {args.output_dir / 'results.csv'}")

    # === Plot correlation ===
    print("\nPlotting beta vs solve date correlation...")
    corr_stats = plot_beta_vs_solve_date(
        results, oracle_items, args.output_dir / "beta_vs_solve_date.png"
    )
    print(f"  Saved: {args.output_dir / 'beta_vs_solve_date.png'}")
    print(f"  Pearson r: {corr_stats['pearson_r']:.4f} (p={corr_stats['pearson_p']:.4f})")
    print(
        f"  Spearman rho: {corr_stats['spearman_r']:.4f} (p={corr_stats['spearman_p']:.4f})"
    )

    # === Individual plots ===
    if args.plot_individual:
        print(
            f"\nGenerating individual logistic fit plots (max {args.max_individual_plots})..."
        )
        plots_dir = args.output_dir / "logistic_fits"
        plots_dir.mkdir(parents=True, exist_ok=True)

        plot_count = 0
        for r in valid_x0[: args.max_individual_plots]:
            # Sanitize task_id for filename
            safe_task_id = r.task_id.replace("/", "_").replace("\\", "_")
            plot_individual_logistic_fit(
                r, pareto_info, responses, plots_dir / f"{safe_task_id}.png"
            )
            plot_count += 1
        print(f"  Generated {plot_count} plots")

    # === Summary ===
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Tasks analyzed: {len(results)}")
    print(f"Tasks with valid logistic fits (interpolated): {len(valid_x0)}")
    print(f"Tasks excluded:")
    print(f"  - Never solved by Pareto agents: {len(never_solved)}")
    print(f"  - Always solved by Pareto agents: {len(always_solved)}")
    print(f"  - Hit lower bound (too easy): {len(hit_lower_bound)}")
    print(f"  - Hit upper bound (too hard): {len(hit_upper_bound)}")
    print(f"\nCorrelation (Oracle beta vs logistic solve date x0):")
    print(f"  Pearson r = {corr_stats['pearson_r']:.4f}")
    print(f"  Spearman rho = {corr_stats['spearman_r']:.4f}")
    print(f"  Linear fit: beta = {corr_stats['slope']:.4f} * x0 + {corr_stats['intercept']:.2f}")


if __name__ == "__main__":
    main()

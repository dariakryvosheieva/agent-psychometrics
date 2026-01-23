"""Output formatting functions for Experiment B.

This module handles:
- Printing comparison tables
- Saving results to CSV
- Formatting date forecasting results
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from experiment_b.shared.data_preparation import AgentFilteringStats


def print_comparison_table(
    results: Dict[str, Dict],
    frontier_task_count: int,
    pre_frontier_count: int,
    post_frontier_count: int,
    anchor_task_count: int = 0,
    alignment_method: str = "affine",
    cutoff_date: str = "20250401",
    frontier_definition: str = "passrate",
    irt_solve_prob: float = 0.5,
    date_results: Optional[Dict[str, Dict]] = None,
    oracle_date_results: Optional[Dict[str, Dict]] = None,
    last_agent_date: Optional[str] = None,
    verbose: bool = False,
    dataset_name: str = "",
    filtering_stats: Optional["AgentFilteringStats"] = None,
) -> None:
    """Print formatted comparison table."""
    print("=" * 90)
    print("EXPERIMENT B: FRONTIER TASK DIFFICULTY PREDICTION")
    print("=" * 90)
    print()
    if frontier_definition == "irt":
        print("Frontier Task Definition (IRT-based):")
        print(f"  - No pre-frontier agent has >={irt_solve_prob:.0%} solve probability")
    elif frontier_definition == "zero_pre":
        print("Frontier Task Definition (zero pre-frontier):")
        print("  - Pre-frontier pass rate == 0% (no pre-frontier agent solves)")
        print("  - Post-frontier pass rate > 0% (at least one post-frontier agent solves)")
    else:  # passrate
        print("Frontier Task Definition (pass-rate based):")
        print("  - Pre-frontier pass rate <= 10%")
        print("  - Post-frontier pass rate > 10%")
    # Format cutoff date as YYYY-MM-DD for readability
    cutoff_formatted = f"{cutoff_date[:4]}-{cutoff_date[4:6]}-{cutoff_date[6:]}"
    if last_agent_date:
        print(f"  - Date range: {cutoff_formatted} to {last_agent_date}")
    else:
        print(f"  - Cutoff date: {cutoff_formatted}")
    print()
    print("Data Summary:")
    print(f"  - Pre-frontier agents: {pre_frontier_count}")
    if filtering_stats is not None and filtering_stats.removed_count > 0:
        print(f"  - Post-frontier agents: {filtering_stats.total_post_frontier} "
              f"(filtered to {post_frontier_count})")
        print(f"    └─ Removed bottom {filtering_stats.filter_bottom_percentile * 100:.0f}% "
              f"(success rate < {filtering_stats.success_rate_threshold:.3f})")
    else:
        print(f"  - Post-frontier agents: {post_frontier_count}")
    print(f"  - Frontier tasks: {frontier_task_count}")
    print(f"  - Anchor tasks (for AUC alignment): {anchor_task_count}")
    print(f"  - Alignment method: {alignment_method}")
    print()

    # Print alignment parameters if verbose
    if verbose:
        print("=" * 90)
        print("ALIGNMENT PARAMETERS (fitted on anchor tasks)")
        print("=" * 90)
        print()
        if alignment_method == "affine":
            print(f"{'Method':<45} {'Slope':>10} {'Intercept':>12} {'R²':>10}")
        else:
            print(f"{'Method':<45} {'Offset':>10}")
        print("-" * 90)

        for method, metrics in results.items():
            params = metrics.get("alignment_params", {})
            if alignment_method == "affine":
                slope = params.get("slope", float("nan"))
                intercept = params.get("intercept", float("nan"))
                r2 = params.get("r_squared", float("nan"))
                print(f"{method:<45} {slope:>10.4f} {intercept:>12.4f} {r2:>10.4f}")
            else:
                offset = params.get("offset", float("nan"))
                print(f"{method:<45} {offset:>10.4f}")

        print()

    # Build table header with dataset and frontier definition
    if frontier_definition == "irt":
        frontier_label = "IRT"
    elif frontier_definition == "zero_pre":
        frontier_label = "Zero-pre"
    else:
        frontier_label = "Pass-rate"
    header_parts = []
    if dataset_name:
        header_parts.append(dataset_name)
    header_parts.append(f"{frontier_label} definition")
    header_parts.append(f"{frontier_task_count} frontier tasks")
    header_parts.append(f"{post_frontier_count} eval agents")
    header_line = " | ".join(header_parts)

    print("=" * 90)
    print(header_line)
    print("=" * 90)
    print()

    # Always show ROC-AUC and MAE metrics
    # Oracle MAE uses Oracle abilities for date lookup (isolates difficulty prediction error)
    has_oracle_mae = oracle_date_results is not None and len(oracle_date_results) > 0
    if has_oracle_mae:
        print(f"{'Method':<50} {'ROC-AUC':>10} {'MAE (days)':>12} {'Oracle MAE†':>12}")
        print("-" * 86)
    else:
        print(f"{'Method':<50} {'ROC-AUC':>10} {'MAE (days)':>12}")
        print("-" * 73)

    # Sort by AUC (descending)
    def sort_key(item):
        auc = item[1].get("auc")
        if auc is None or (isinstance(auc, float) and np.isnan(auc)):
            return float("-inf")
        return auc

    sorted_methods = sorted(results.items(), key=sort_key, reverse=True)

    for method, metrics in sorted_methods:
        auc = metrics.get("auc")

        # Format AUC
        if auc is None or (isinstance(auc, float) and np.isnan(auc)):
            auc_str = "N/A"
        else:
            auc_str = f"{auc:.4f}"

        # Get MAE from date_results if available
        if date_results:
            date_metrics = date_results.get(method, {})
            mae = date_metrics.get("mae_days", float("nan"))
            if isinstance(mae, float) and np.isnan(mae):
                mae_str = "N/A"
            else:
                mae_str = f"{mae:.1f}"
        else:
            mae_str = "N/A"

        # Get Oracle MAE if available
        if has_oracle_mae:
            oracle_metrics = oracle_date_results.get(method, {})
            oracle_mae = oracle_metrics.get("mae_days", float("nan"))
            if isinstance(oracle_mae, float) and np.isnan(oracle_mae):
                oracle_mae_str = "N/A"
            else:
                oracle_mae_str = f"{oracle_mae:.1f}"
            print(f"{method:<50} {auc_str:>10} {mae_str:>12} {oracle_mae_str:>12}")
        else:
            print(f"{method:<50} {auc_str:>10} {mae_str:>12}")

    # Print footnote for Oracle MAE if present
    if has_oracle_mae:
        print()
        print("† Oracle MAE: Earliest Oracle agent with θ ≥ predicted β (bypasses regression)")

    print()


def save_results_csv(results: Dict[str, Dict], output_path: Path) -> None:
    """Save results to CSV."""
    rows = []
    for method, metrics in results.items():
        rows.append({
            "method": method,
            "auc": metrics.get("auc"),
            "n_tasks": metrics.get("num_frontier_tasks"),
            "auc_n_pairs": metrics.get("auc_n_pairs"),
            "auc_n_positive": metrics.get("auc_n_positive"),
            "auc_n_negative": metrics.get("auc_n_negative"),
        })

    df = pd.DataFrame(rows)
    # Sort by AUC descending (with NaN at bottom)
    df = df.sort_values("auc", ascending=False, na_position="last")
    df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")


def print_date_forecast_table(
    date_results: Dict[str, Dict],
    n_frontier_total: int,
    n_frontier_with_gt: int,
    n_excluded: int,
    earliest_agent_date: str,
    latest_agent_date: str,
    cutoff_date: str,
    gt_date_min: str,
    gt_date_max: str,
) -> None:
    """Print formatted date forecasting results table."""
    print()
    print("=" * 90)
    print("DATE FORECASTING: PREDICT WHEN TASKS BECOME SOLVABLE")
    print("=" * 90)
    print()
    print("Data Summary:")
    print(f"  Post-cutoff tasks (eval set): {n_frontier_total}")
    print(f"  Tasks without any capable agent: {n_excluded}")
    print()
    print("Date Range:")
    print(f"  Earliest agent date: {earliest_agent_date}")
    print(f"  Latest agent date: {latest_agent_date}")
    print(f"  Frontier cutoff: {cutoff_date}")
    print(f"  Ground truth date range: {gt_date_min} to {gt_date_max}")
    print()
    print(f"{'Method':<45} {'MAE (days)':>12} {'Pearson r':>12} {'R²(fit)':>10} {'n':>6}")
    print("-" * 86)

    # Sort by MAE (ascending), NaN at bottom
    def sort_key(item):
        mae = item[1].get("mae_days", float("inf"))
        if isinstance(mae, float) and np.isnan(mae):
            return float("inf")
        return mae

    sorted_results = sorted(date_results.items(), key=sort_key)

    for method, metrics in sorted_results:
        mae = metrics.get("mae_days", float("nan"))
        pearson = metrics.get("pearson_r", float("nan"))
        r2_fit = metrics.get("r_squared_fit", float("nan"))
        n_tasks = metrics.get("n_tasks", 0)

        mae_str = f"{mae:.1f}" if not (isinstance(mae, float) and np.isnan(mae)) else "N/A"
        pearson_str = f"{pearson:.4f}" if not (isinstance(pearson, float) and np.isnan(pearson)) else "N/A"
        r2_str = f"{r2_fit:.4f}" if not (isinstance(r2_fit, float) and np.isnan(r2_fit)) else "N/A"

        print(f"{method:<45} {mae_str:>12} {pearson_str:>12} {r2_str:>10} {n_tasks:>6}")

    print()

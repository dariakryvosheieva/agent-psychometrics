"""Per-agent correlation analysis: assistant message length vs task difficulty.

Investigates whether the negative correlation between trajectory length and task
difficulty (found in tensor decomposition) holds at the per-agent level.

Key analysis:
1. Per-agent correlation on ALL 500 tasks
2. Per-agent correlation on FRONTIER tasks only (zero_pre definition)

If correlations are significant, generates a normalized feature matrix for FeatureIRT.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Import from experiment_b
from experiment_b.shared import (
    identify_frontier_tasks_zero_pre,
    load_responses_dict,
    split_agents_by_dates,
)
from experiment_b.swebench.config import SWEBenchConfig, extract_date_prefix

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
CHAR_COUNTS_PATH = PROJECT_ROOT / "chris_output" / "tensor_analysis" / "swebench_verified_char_counts.csv"
IRT_PATH = PROJECT_ROOT / "clean_data" / "swebench_verified_20251120_full" / "1d_1pl" / "items.csv"
OUTPUT_DIR = PROJECT_ROOT / "chris_output" / "tensor_analysis"

# Get config for paths and cutoff date
CONFIG = SWEBenchConfig()


def compute_per_agent_correlations(
    char_counts: pd.DataFrame,
    irt_difficulties: pd.DataFrame,
    task_subset: list[str] | None = None,
) -> pd.DataFrame:
    """Compute Pearson correlation for each agent.

    Args:
        char_counts: DataFrame with columns [agent, task_id, assistant_char_count]
        irt_difficulties: DataFrame indexed by task_id with column 'b'
        task_subset: Optional list of task_ids to filter to

    Returns:
        DataFrame with columns [agent, r, p_value, n_tasks]
    """
    results = []

    for agent in sorted(char_counts["agent"].unique()):
        agent_data = char_counts[char_counts["agent"] == agent].copy()

        # Filter to task subset if provided
        if task_subset is not None:
            agent_data = agent_data[agent_data["task_id"].isin(task_subset)]

        # Join with IRT difficulties
        agent_data = agent_data.set_index("task_id")
        merged = agent_data.join(irt_difficulties[["b"]], how="inner")

        if len(merged) < 3:
            # Not enough data for correlation
            results.append({
                "agent": agent,
                "r": np.nan,
                "p_value": np.nan,
                "n_tasks": len(merged),
            })
            continue

        # Compute Pearson correlation
        r, p_value = stats.pearsonr(merged["assistant_char_count"], merged["b"])

        results.append({
            "agent": agent,
            "r": r,
            "p_value": p_value,
            "n_tasks": len(merged),
        })

    return pd.DataFrame(results)


def plot_correlation_histograms(
    all_tasks_corr: pd.DataFrame,
    frontier_corr: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot side-by-side histograms of r values."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # All tasks
    ax = axes[0]
    valid_r = all_tasks_corr["r"].dropna()
    ax.hist(valid_r, bins=20, edgecolor="black", alpha=0.7)
    ax.axvline(x=0, color="red", linestyle="--", linewidth=2)
    ax.axvline(x=valid_r.mean(), color="blue", linestyle="-", linewidth=2, label=f"Mean: {valid_r.mean():.3f}")
    ax.set_xlabel("Pearson r")
    ax.set_ylabel("Number of Agents")
    ax.set_title(f"All Tasks (n=500)\n{len(valid_r)} agents")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add significance count
    n_sig = (all_tasks_corr["p_value"] < 0.05).sum()
    n_neg_sig = ((all_tasks_corr["p_value"] < 0.05) & (all_tasks_corr["r"] < 0)).sum()
    ax.text(0.05, 0.95, f"Significant (p<0.05): {n_sig}/{len(valid_r)}\nNeg & sig: {n_neg_sig}",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Frontier tasks
    ax = axes[1]
    valid_r = frontier_corr["r"].dropna()
    if len(valid_r) > 0:
        ax.hist(valid_r, bins=20, edgecolor="black", alpha=0.7, color="orange")
        ax.axvline(x=0, color="red", linestyle="--", linewidth=2)
        ax.axvline(x=valid_r.mean(), color="blue", linestyle="-", linewidth=2, label=f"Mean: {valid_r.mean():.3f}")
        ax.legend()

        n_sig = (frontier_corr["p_value"] < 0.05).sum()
        n_neg_sig = ((frontier_corr["p_value"] < 0.05) & (frontier_corr["r"] < 0)).sum()
        ax.text(0.05, 0.95, f"Significant (p<0.05): {n_sig}/{len(valid_r)}\nNeg & sig: {n_neg_sig}",
                transform=ax.transAxes, fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    else:
        ax.text(0.5, 0.5, "No valid correlations", ha="center", va="center", transform=ax.transAxes)

    n_frontier = frontier_corr["n_tasks"].iloc[0] if len(frontier_corr) > 0 else 0
    ax.set_xlabel("Pearson r")
    ax.set_ylabel("Number of Agents")
    ax.set_title(f"Frontier Tasks Only (n={n_frontier})\n{len(valid_r)} agents")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def generate_feature_matrix(
    char_counts: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Generate normalized feature matrix for CSVFeatureSource.

    For each agent, z-score normalize their char counts across all tasks.
    Output: CSV with columns [task_id, agent1_zscore, agent2_zscore, ...]
    """
    # Pivot to get tasks x agents matrix
    pivot = char_counts.pivot(index="task_id", columns="agent", values="assistant_char_count")

    # Z-score normalize each agent (column)
    normalized = (pivot - pivot.mean()) / pivot.std()

    # Reset index to make task_id a column
    normalized = normalized.reset_index()

    # Save
    normalized.to_csv(output_path, index=False)
    print(f"Saved feature matrix: {output_path}")
    print(f"  Shape: {normalized.shape[0]} tasks x {normalized.shape[1] - 1} agents")

    return normalized


def main():
    """Run per-agent correlation analysis."""
    print("=" * 60)
    print("Per-Agent Correlation Analysis")
    print("Assistant Message Length vs Task Difficulty")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    char_counts = pd.read_csv(CHAR_COUNTS_PATH)
    irt_difficulties = pd.read_csv(IRT_PATH, index_col=0)
    responses = load_responses_dict(CONFIG.responses_path)

    print(f"  Char counts: {len(char_counts)} rows")
    print(f"  IRT difficulties: {len(irt_difficulties)} tasks")
    print(f"  Response matrix: {len(responses)} agents")

    # Verify 44 agents claim
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)

    agents = sorted(char_counts["agent"].unique())
    print(f"\nAgents in char counts: {len(agents)}")

    # Check each agent has 500 tasks
    tasks_per_agent = char_counts.groupby("agent").size()
    all_500 = (tasks_per_agent == 500).all()
    print(f"All agents have 500 tasks: {all_500}")
    if not all_500:
        print("  Agents with != 500 tasks:")
        for agent, n in tasks_per_agent.items():
            if n != 500:
                print(f"    {agent}: {n}")

    # Check all char counts > 0
    zero_counts = (char_counts["assistant_char_count"] == 0).sum()
    print(f"Rows with zero char count: {zero_counts}")

    # Get frontier tasks
    print("\n" + "=" * 60)
    print("Frontier Task Identification")
    print("=" * 60)

    # Get agent dates and split by cutoff
    all_agents = list(responses.keys())
    agent_dates = {agent: extract_date_prefix(agent) for agent in all_agents}
    # Filter out agents without date prefixes
    agent_dates = {k: v for k, v in agent_dates.items() if v}

    pre_frontier, post_frontier = split_agents_by_dates(
        list(agent_dates.keys()),
        agent_dates,
        CONFIG.cutoff_date,
    )

    print(f"\nCutoff date: {CONFIG.cutoff_date}")
    print(f"Pre-frontier agents: {len(pre_frontier)}")
    print(f"Post-frontier agents: {len(post_frontier)}")

    frontier_tasks = identify_frontier_tasks_zero_pre(
        CONFIG.responses_path,
        pre_frontier,
        post_frontier,
    )
    print(f"Frontier tasks (zero_pre): {len(frontier_tasks)}")

    # Compute correlations - all tasks
    print("\n" + "=" * 60)
    print("Per-Agent Correlations: ALL TASKS")
    print("=" * 60)

    all_tasks_corr = compute_per_agent_correlations(char_counts, irt_difficulties)

    valid = all_tasks_corr["r"].notna()
    print(f"\nAgents with valid correlations: {valid.sum()}/{len(all_tasks_corr)}")
    print(f"\nCorrelation Summary (r):")
    print(f"  Mean:   {all_tasks_corr.loc[valid, 'r'].mean():.4f}")
    print(f"  Median: {all_tasks_corr.loc[valid, 'r'].median():.4f}")
    print(f"  Std:    {all_tasks_corr.loc[valid, 'r'].std():.4f}")
    print(f"  Min:    {all_tasks_corr.loc[valid, 'r'].min():.4f}")
    print(f"  Max:    {all_tasks_corr.loc[valid, 'r'].max():.4f}")

    n_sig = (all_tasks_corr["p_value"] < 0.05).sum()
    n_neg_sig = ((all_tasks_corr["p_value"] < 0.05) & (all_tasks_corr["r"] < 0)).sum()
    print(f"\nSignificant (p < 0.05): {n_sig}/{valid.sum()} ({100*n_sig/valid.sum():.1f}%)")
    print(f"Negative & significant: {n_neg_sig}/{valid.sum()} ({100*n_neg_sig/valid.sum():.1f}%)")

    # Compute correlations - frontier tasks only
    print("\n" + "=" * 60)
    print("Per-Agent Correlations: FRONTIER TASKS ONLY")
    print("=" * 60)

    frontier_corr = compute_per_agent_correlations(char_counts, irt_difficulties, task_subset=frontier_tasks)

    valid = frontier_corr["r"].notna()
    print(f"\nAgents with valid correlations: {valid.sum()}/{len(frontier_corr)}")

    if valid.sum() > 0:
        print(f"\nCorrelation Summary (r):")
        print(f"  Mean:   {frontier_corr.loc[valid, 'r'].mean():.4f}")
        print(f"  Median: {frontier_corr.loc[valid, 'r'].median():.4f}")
        print(f"  Std:    {frontier_corr.loc[valid, 'r'].std():.4f}")
        print(f"  Min:    {frontier_corr.loc[valid, 'r'].min():.4f}")
        print(f"  Max:    {frontier_corr.loc[valid, 'r'].max():.4f}")

        n_sig = (frontier_corr["p_value"] < 0.05).sum()
        n_neg_sig = ((frontier_corr["p_value"] < 0.05) & (frontier_corr["r"] < 0)).sum()
        print(f"\nSignificant (p < 0.05): {n_sig}/{valid.sum()} ({100*n_sig/valid.sum():.1f}%)")
        print(f"Negative & significant: {n_neg_sig}/{valid.sum()} ({100*n_neg_sig/valid.sum():.1f}%)")
    else:
        print("No valid correlations on frontier tasks!")

    # Save correlation results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_tasks_corr.to_csv(OUTPUT_DIR / "per_agent_correlations_all_tasks.csv", index=False)
    frontier_corr.to_csv(OUTPUT_DIR / "per_agent_correlations_frontier.csv", index=False)
    print(f"\nSaved correlation results to {OUTPUT_DIR}")

    # Plot histograms
    plot_correlation_histograms(
        all_tasks_corr,
        frontier_corr,
        OUTPUT_DIR / "per_agent_correlation_histograms.png",
    )

    # Generate feature matrix
    print("\n" + "=" * 60)
    print("Feature Matrix Generation")
    print("=" * 60)

    generate_feature_matrix(
        char_counts,
        OUTPUT_DIR / "agent_char_count_features.csv",
    )

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()

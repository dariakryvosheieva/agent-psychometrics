#!/usr/bin/env python3
"""
Analyze 2D IRT difficulty dimensions to understand what they represent.

Creates visualizations:
1. Scatter plot of b1 vs b2 with extreme points labeled
2. Distribution of b1-b2 (difficulty difference) by repo
3. Breakdown of extreme problems in each direction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def load_data(results_dir: Path):
    """Load items and abilities from 2D model results."""
    items = pd.read_csv(results_dir / "2d" / "items.csv", index_col=0)
    abilities = pd.read_csv(results_dir / "2d" / "abilities.csv", index_col=0)
    return items, abilities


def parse_task_id(task_id: str):
    """Parse a SWE-bench task ID into repo and issue number."""
    # Format: repo__repo-issue_number (e.g., django__django-12345)
    parts = task_id.rsplit("-", 1)
    if len(parts) == 2:
        repo_part = parts[0]
        issue_num = parts[1]
        # repo_part is like "django__django"
        repo = repo_part.replace("__", "/")
        return repo, issue_num
    return task_id, None


def analyze_extreme_tasks(items: pd.DataFrame, n_extreme: int = 15):
    """Identify tasks with extreme b1-b2 differences."""
    items = items.copy()
    items["b_diff"] = items["b1"] - items["b2"]
    items["repo"], items["issue"] = zip(*[parse_task_id(idx) for idx in items.index])

    print("\n" + "="*80)
    print("EXTREME TASKS ANALYSIS")
    print("="*80)

    # High b1, low b2 (hard on dim1, easier on dim2)
    print(f"\n📊 TOP {n_extreme} TASKS: High b1, Low b2 (Hard on Dim1, Easier on Dim2)")
    print("-"*80)
    high_b1 = items.nlargest(n_extreme, "b_diff")[["b1", "b2", "b_diff", "repo"]]
    for task_id, row in high_b1.iterrows():
        print(f"  {task_id:50s} b1={row['b1']:6.2f}  b2={row['b2']:6.2f}  diff={row['b_diff']:+6.2f}")

    # High b2, low b1 (hard on dim2, easier on dim1)
    print(f"\n📊 TOP {n_extreme} TASKS: High b2, Low b1 (Hard on Dim2, Easier on Dim1)")
    print("-"*80)
    high_b2 = items.nsmallest(n_extreme, "b_diff")[["b1", "b2", "b_diff", "repo"]]
    for task_id, row in high_b2.iterrows():
        print(f"  {task_id:50s} b1={row['b1']:6.2f}  b2={row['b2']:6.2f}  diff={row['b_diff']:+6.2f}")

    return items


def analyze_by_repo(items: pd.DataFrame):
    """Analyze difficulty dimensions by repository."""
    items = items.copy()
    if "b_diff" not in items.columns:
        items["b_diff"] = items["b1"] - items["b2"]
    if "repo" not in items.columns:
        items["repo"], items["issue"] = zip(*[parse_task_id(idx) for idx in items.index])

    # Calculate statistics per repo
    repo_stats = items.groupby("repo").agg({
        "b1": ["mean", "std", "count"],
        "b2": ["mean", "std"],
        "b_diff": ["mean", "std"]
    }).round(3)
    repo_stats.columns = ["b1_mean", "b1_std", "count", "b2_mean", "b2_std", "b_diff_mean", "b_diff_std"]
    repo_stats = repo_stats.sort_values("b_diff_mean", ascending=False)

    print("\n" + "="*80)
    print("REPOSITORY ANALYSIS (sorted by b_diff_mean)")
    print("="*80)
    print(f"{'Repo':<40} {'N':>4}  {'b1_mean':>8}  {'b2_mean':>8}  {'diff_mean':>10}")
    print("-"*80)
    for repo, row in repo_stats.iterrows():
        print(f"{repo:<40} {int(row['count']):>4}  {row['b1_mean']:>8.2f}  {row['b2_mean']:>8.2f}  {row['b_diff_mean']:>+10.2f}")

    return repo_stats


def plot_b1_b2_scatter(items: pd.DataFrame, output_path: Path, n_label: int = 10):
    """Create scatter plot of b1 vs b2 with extreme points labeled."""
    items = items.copy()
    items["b_diff"] = items["b1"] - items["b2"]

    fig, ax = plt.subplots(figsize=(12, 10))

    # Color by b_diff
    scatter = ax.scatter(items["b1"], items["b2"],
                         c=items["b_diff"], cmap="RdYlBu_r",
                         alpha=0.7, s=50, edgecolors='white', linewidth=0.5)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("b1 - b2 (difficulty difference)", fontsize=11)

    # Add diagonal line (b1 = b2)
    lims = [min(items["b1"].min(), items["b2"].min()) - 0.5,
            max(items["b1"].max(), items["b2"].max()) + 0.5]
    ax.plot(lims, lims, 'k--', alpha=0.3, label="b1 = b2")

    # Label extreme points
    extreme_high = items.nlargest(n_label, "b_diff")
    extreme_low = items.nsmallest(n_label, "b_diff")

    for task_id, row in extreme_high.iterrows():
        # Shorten task name for display
        short_name = task_id.split("__")[-1][:25]
        ax.annotate(short_name, (row["b1"], row["b2"]),
                    fontsize=7, alpha=0.8,
                    xytext=(5, 5), textcoords='offset points')

    for task_id, row in extreme_low.iterrows():
        short_name = task_id.split("__")[-1][:25]
        ax.annotate(short_name, (row["b1"], row["b2"]),
                    fontsize=7, alpha=0.8,
                    xytext=(5, -10), textcoords='offset points')

    ax.set_xlabel("b1 (Difficulty on Dimension 1)", fontsize=12)
    ax.set_ylabel("b2 (Difficulty on Dimension 2)", fontsize=12)
    ax.set_title("2D IRT: Task Difficulty Distribution\nExtreme points labeled", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add correlation annotation
    corr = items["b1"].corr(items["b2"])
    ax.text(0.05, 0.95, f"Correlation: {corr:.3f}",
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path / "b1_b2_scatter.png", dpi=150)
    plt.close()
    print(f"\n✓ Saved: {output_path / 'b1_b2_scatter.png'}")


def plot_repo_difficulty_bars(repo_stats: pd.DataFrame, output_path: Path):
    """Create bar chart showing mean b_diff by repository."""
    # Filter to repos with enough tasks
    filtered = repo_stats[repo_stats["count"] >= 5].copy()
    filtered = filtered.sort_values("b_diff_mean")

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['#d73027' if x > 0 else '#4575b4' for x in filtered["b_diff_mean"]]
    bars = ax.barh(range(len(filtered)), filtered["b_diff_mean"], color=colors, alpha=0.7)

    ax.set_yticks(range(len(filtered)))
    ax.set_yticklabels([r.split("/")[-1] for r in filtered.index], fontsize=9)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel("Mean (b1 - b2) per Repository", fontsize=11)
    ax.set_title("Repository Difficulty Profile\n(Red = harder on Dim1, Blue = harder on Dim2)", fontsize=12)
    ax.grid(True, axis='x', alpha=0.3)

    # Add count labels
    for i, (idx, row) in enumerate(filtered.iterrows()):
        ax.text(row["b_diff_mean"] + 0.02 if row["b_diff_mean"] >= 0 else row["b_diff_mean"] - 0.02,
                i, f"n={int(row['count'])}",
                va='center', ha='left' if row["b_diff_mean"] >= 0 else 'right',
                fontsize=8, alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path / "repo_difficulty_profile.png", dpi=150)
    plt.close()
    print(f"✓ Saved: {output_path / 'repo_difficulty_profile.png'}")


def plot_difficulty_distribution(items: pd.DataFrame, output_path: Path):
    """Plot distribution of b1, b2, and b_diff."""
    items = items.copy()
    items["b_diff"] = items["b1"] - items["b2"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # b1 distribution
    axes[0].hist(items["b1"], bins=30, color='steelblue', alpha=0.7, edgecolor='white')
    axes[0].axvline(items["b1"].mean(), color='red', linestyle='--', label=f'mean={items["b1"].mean():.2f}')
    axes[0].set_xlabel("b1 (Difficulty Dim 1)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of b1")
    axes[0].legend()

    # b2 distribution
    axes[1].hist(items["b2"], bins=30, color='darkorange', alpha=0.7, edgecolor='white')
    axes[1].axvline(items["b2"].mean(), color='red', linestyle='--', label=f'mean={items["b2"].mean():.2f}')
    axes[1].set_xlabel("b2 (Difficulty Dim 2)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Distribution of b2")
    axes[1].legend()

    # b_diff distribution
    axes[2].hist(items["b_diff"], bins=30, color='purple', alpha=0.7, edgecolor='white')
    axes[2].axvline(0, color='black', linestyle='-', linewidth=0.8)
    axes[2].axvline(items["b_diff"].mean(), color='red', linestyle='--', label=f'mean={items["b_diff"].mean():.2f}')
    axes[2].set_xlabel("b1 - b2 (Difficulty Difference)")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Distribution of b_diff")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_path / "difficulty_distributions.png", dpi=150)
    plt.close()
    print(f"✓ Saved: {output_path / 'difficulty_distributions.png'}")


def plot_agent_abilities(abilities: pd.DataFrame, output_path: Path, n_label: int = 15):
    """Plot agent abilities in 2D space."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Color by theta_avg
    scatter = ax.scatter(abilities["theta1"], abilities["theta2"],
                         c=abilities["theta_avg"], cmap="viridis",
                         alpha=0.7, s=60, edgecolors='white', linewidth=0.5)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Average ability (theta_avg)", fontsize=11)

    # Label top agents
    top_agents = abilities.nlargest(n_label, "theta_avg")
    for agent_id, row in top_agents.iterrows():
        # Shorten agent name
        short_name = agent_id.split("_")[0][:20] if "_" in agent_id else agent_id[:20]
        ax.annotate(short_name, (row["theta1"], row["theta2"]),
                    fontsize=7, alpha=0.8,
                    xytext=(5, 5), textcoords='offset points')

    ax.set_xlabel("θ1 (Ability on Dimension 1)", fontsize=12)
    ax.set_ylabel("θ2 (Ability on Dimension 2)", fontsize=12)
    ax.set_title("2D IRT: Agent Ability Distribution", fontsize=14)
    ax.grid(True, alpha=0.3)

    # Add correlation
    corr = abilities["theta1"].corr(abilities["theta2"])
    ax.text(0.05, 0.95, f"Correlation: {corr:.3f}",
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path / "agent_abilities_2d.png", dpi=150)
    plt.close()
    print(f"✓ Saved: {output_path / 'agent_abilities_2d.png'}")


def plot_dimension_specialization(abilities: pd.DataFrame, output_path: Path):
    """Show agents that specialize in one dimension vs the other."""
    abilities = abilities.copy()
    abilities["theta_diff"] = abilities["theta1"] - abilities["theta2"]

    # Sort by theta_diff
    sorted_abilities = abilities.sort_values("theta_diff")

    fig, ax = plt.subplots(figsize=(14, 8))

    colors = ['#d73027' if x > 0 else '#4575b4' for x in sorted_abilities["theta_diff"]]

    y_pos = range(len(sorted_abilities))
    ax.barh(y_pos, sorted_abilities["theta_diff"], color=colors, alpha=0.7)

    # Only label every nth agent for readability
    n_show = len(sorted_abilities)
    ax.set_yticks(y_pos[::max(1, n_show//30)])
    short_names = [idx.split("_")[0][:15] if "_" in idx else idx[:15]
                   for idx in sorted_abilities.index]
    ax.set_yticklabels([short_names[i] for i in range(0, n_show, max(1, n_show//30))], fontsize=7)

    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel("θ1 - θ2 (Ability Difference)", fontsize=11)
    ax.set_title("Agent Dimension Specialization\n(Red = stronger on Dim1, Blue = stronger on Dim2)", fontsize=12)
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "agent_specialization.png", dpi=150)
    plt.close()
    print(f"✓ Saved: {output_path / 'agent_specialization.png'}")


def main():
    parser = argparse.ArgumentParser(description="Analyze 2D IRT difficulty dimensions")
    parser.add_argument("--results_dir", type=str,
                        default="clean_data/swebench_verified_20250930_full",
                        help="Directory containing trained model results")
    parser.add_argument("--output_dir", type=str,
                        default="chris_output/figures/2d_analysis",
                        help="Output directory for figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {results_dir}")
    items, abilities = load_data(results_dir)

    print(f"Loaded {len(items)} items and {len(abilities)} agents")

    # Basic stats
    print("\n" + "="*80)
    print("BASIC STATISTICS")
    print("="*80)
    print(f"b1: mean={items['b1'].mean():.3f}, std={items['b1'].std():.3f}")
    print(f"b2: mean={items['b2'].mean():.3f}, std={items['b2'].std():.3f}")
    print(f"b1-b2 correlation: {items['b1'].corr(items['b2']):.3f}")
    print(f"\nθ1: mean={abilities['theta1'].mean():.3f}, std={abilities['theta1'].std():.3f}")
    print(f"θ2: mean={abilities['theta2'].mean():.3f}, std={abilities['theta2'].std():.3f}")
    print(f"θ1-θ2 correlation: {abilities['theta1'].corr(abilities['theta2']):.3f}")

    # Analyze extreme tasks
    items = analyze_extreme_tasks(items, n_extreme=15)

    # Analyze by repo
    repo_stats = analyze_by_repo(items)

    # Generate plots
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)

    plot_b1_b2_scatter(items, output_dir, n_label=12)
    plot_repo_difficulty_bars(repo_stats, output_dir)
    plot_difficulty_distribution(items, output_dir)
    plot_agent_abilities(abilities, output_dir, n_label=15)
    plot_dimension_specialization(abilities, output_dir)

    print(f"\n✓ All figures saved to: {output_dir}")


if __name__ == "__main__":
    main()

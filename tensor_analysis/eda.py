"""Exploratory data analysis for trajectory features.

Generates plots showing:
- Per-agent distributions of assistant character counts
- Cross-agent comparisons
- Success vs failure distributions
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_char_counts(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load extracted character counts."""
    verified_df = pd.read_csv(output_dir / "swebench_verified_char_counts.csv")
    pro_df = pd.read_csv(output_dir / "swebench_pro_char_counts.csv")
    return verified_df, pro_df


def plot_per_agent_distributions(
    df: pd.DataFrame,
    agents: list[str],
    output_path: Path,
    title_prefix: str = "",
) -> None:
    """Plot histograms of assistant char counts for selected agents."""
    n_agents = len(agents)
    fig, axes = plt.subplots(1, n_agents, figsize=(4 * n_agents, 4))
    if n_agents == 1:
        axes = [axes]

    for ax, agent in zip(axes, agents):
        agent_data = df[df["agent"] == agent]["assistant_char_count"]
        ax.hist(agent_data / 1000, bins=30, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Assistant chars (K)")
        ax.set_ylabel("Count")
        ax.set_title(f"{agent[:30]}...")

        # Add stats
        stats_text = f"n={len(agent_data)}\nmean={agent_data.mean()/1000:.1f}K\nmed={agent_data.median()/1000:.1f}K"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment="top", horizontalalignment="right", fontsize=8)

    plt.suptitle(f"{title_prefix}Per-Agent Assistant Character Distributions")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_cross_agent_boxplot(
    df: pd.DataFrame,
    output_path: Path,
    title: str = "Cross-Agent Character Count Distribution",
) -> None:
    """Plot box plot comparing all agents."""
    # Sort agents by median char count
    agent_medians = df.groupby("agent")["assistant_char_count"].median().sort_values()

    fig, ax = plt.subplots(figsize=(14, 6))

    # Create box plot
    df_sorted = df.copy()
    df_sorted["agent"] = pd.Categorical(df_sorted["agent"], categories=agent_medians.index, ordered=True)

    sns.boxplot(data=df_sorted, x="agent", y="assistant_char_count", ax=ax, showfliers=False)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=6)
    ax.set_ylabel("Assistant chars")
    ax.set_xlabel("Agent")
    ax.set_title(title)

    # Format y-axis in K
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x/1000:.0f}K"))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_overall_distribution(
    df: pd.DataFrame,
    output_path: Path,
    title: str = "Overall Assistant Character Distribution",
) -> None:
    """Plot overall histogram with KDE."""
    fig, ax = plt.subplots(figsize=(10, 6))

    chars = df["assistant_char_count"] / 1000  # Convert to K

    ax.hist(chars, bins=50, density=True, alpha=0.7, edgecolor="black")
    chars.plot.kde(ax=ax, color="red", linewidth=2)

    ax.set_xlabel("Assistant chars (K)")
    ax.set_ylabel("Density")
    ax.set_title(title)

    # Add stats
    stats_text = (
        f"n={len(chars)}\n"
        f"mean={chars.mean():.1f}K\n"
        f"median={chars.median():.1f}K\n"
        f"std={chars.std():.1f}K"
    )
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_by_outcome(
    df: pd.DataFrame,
    output_path: Path,
    title: str = "Character Distribution by Outcome",
) -> None:
    """Plot char count distributions split by success/failure."""
    if "resolved" not in df.columns:
        print(f"Skipping outcome plot - no 'resolved' column")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Split by resolved
    success = df[df["resolved"] == True]["assistant_char_count"] / 1000
    failure = df[df["resolved"] == False]["assistant_char_count"] / 1000

    # Histogram comparison
    ax = axes[0]
    ax.hist(success, bins=40, alpha=0.6, label=f"Success (n={len(success)})", density=True)
    ax.hist(failure, bins=40, alpha=0.6, label=f"Failure (n={len(failure)})", density=True)
    ax.set_xlabel("Assistant chars (K)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution by Outcome")
    ax.legend()

    # Box plot comparison
    ax = axes[1]
    df_plot = df.copy()
    df_plot["outcome"] = df_plot["resolved"].map({True: "Success", False: "Failure"})
    sns.boxplot(data=df_plot, x="outcome", y="assistant_char_count", ax=ax, showfliers=False)
    ax.set_ylabel("Assistant chars")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x/1000:.0f}K"))
    ax.set_title("Box Plot by Outcome")

    # Stats
    print(f"\n=== Outcome Statistics ===")
    print(f"Success: n={len(success)}, mean={success.mean():.1f}K, median={success.median():.1f}K")
    print(f"Failure: n={len(failure)}, mean={failure.mean():.1f}K, median={failure.median():.1f}K")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Run EDA on trajectory features."""
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "chris_output" / "tensor_analysis"
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    verified_df, pro_df = load_char_counts(output_dir)

    print(f"Loaded Verified: {len(verified_df)} rows, {verified_df['agent'].nunique()} agents")
    print(f"Loaded Pro: {len(pro_df)} rows, {pro_df['agent'].nunique()} agents")

    # === SWE-bench Verified ===
    print("\n" + "=" * 60)
    print("SWE-bench Verified EDA")
    print("=" * 60)

    # Select 5 agents at different performance levels (by median char count)
    agent_medians = verified_df.groupby("agent")["assistant_char_count"].median().sort_values()
    indices = np.linspace(0, len(agent_medians) - 1, 5).astype(int)
    sample_agents = agent_medians.index[indices].tolist()

    plot_per_agent_distributions(
        verified_df, sample_agents,
        plots_dir / "verified_per_agent_hist.png",
        title_prefix="SWE-bench Verified: "
    )

    plot_cross_agent_boxplot(
        verified_df,
        plots_dir / "verified_cross_agent_boxplot.png",
        title="SWE-bench Verified: Cross-Agent Character Distribution"
    )

    plot_overall_distribution(
        verified_df,
        plots_dir / "verified_overall_hist.png",
        title="SWE-bench Verified: Overall Assistant Character Distribution"
    )

    plot_by_outcome(
        verified_df,
        plots_dir / "verified_by_outcome.png",
        title="SWE-bench Verified: Character Distribution by Outcome"
    )

    # === SWE-bench Pro ===
    print("\n" + "=" * 60)
    print("SWE-bench Pro EDA")
    print("=" * 60)

    # Select up to 5 agents
    pro_agent_medians = pro_df.groupby("agent")["assistant_char_count"].median().sort_values()
    n_pro_agents = min(5, len(pro_agent_medians))
    pro_indices = np.linspace(0, len(pro_agent_medians) - 1, n_pro_agents).astype(int)
    pro_sample_agents = pro_agent_medians.index[pro_indices].tolist()

    plot_per_agent_distributions(
        pro_df, pro_sample_agents,
        plots_dir / "pro_per_agent_hist.png",
        title_prefix="SWE-bench Pro: "
    )

    plot_cross_agent_boxplot(
        pro_df,
        plots_dir / "pro_cross_agent_boxplot.png",
        title="SWE-bench Pro: Cross-Agent Character Distribution"
    )

    plot_overall_distribution(
        pro_df,
        plots_dir / "pro_overall_hist.png",
        title="SWE-bench Pro: Overall Assistant Character Distribution"
    )

    print("\n" + "=" * 60)
    print("EDA Complete")
    print("=" * 60)
    print(f"Plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()

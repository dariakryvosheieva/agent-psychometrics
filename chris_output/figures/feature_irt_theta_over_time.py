"""
Plot Feature-IRT learned theta (ability) values vs agent submission date.

X-axis: Date (agent submission date from YYYYMMDD prefix)
Y-axis: Feature-IRT learned ability (theta)

This is analogous to sad_irt_theta_over_time.py but uses learned theta
values from Feature-IRT training instead of SAD-IRT.
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy import stats

# Add project root to path for imports
import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiment_b import get_dataset_config
from experiment_b.shared.data_splits import (
    get_all_agents_from_responses,
    split_agents_by_dates,
)
from experiment_b.shared.evaluate import load_responses_dict
from experiment_b.shared.feature_irt_predictor import FeatureIRTPredictor
from shared.feature_source import EmbeddingFeatureSource


def extract_submission_date(agent_name: str) -> datetime | None:
    """Extract the submission date from agent name (YYYYMMDD prefix)."""
    try:
        date_str = agent_name[:8]
        return datetime.strptime(date_str, "%Y%m%d")
    except (ValueError, IndexError):
        return None


def get_short_name(agent_name: str) -> str:
    """Get a shortened display name for an agent."""
    name = agent_name[9:] if len(agent_name) > 9 else agent_name
    if len(name) > 35:
        name = name[:32] + "..."
    return name


def main():
    # Load dataset configuration (default: swebench)
    dataset_config = get_dataset_config("swebench")

    responses_path = dataset_config.responses_path
    embeddings_path = dataset_config.embeddings_path
    cutoff_date = dataset_config.cutoff_date

    print(f"Dataset: {dataset_config.name}")
    print(f"Responses: {responses_path}")
    print(f"Embeddings: {embeddings_path}")
    print(f"Cutoff date: {cutoff_date}")

    # Load responses
    print("\nLoading response matrix...")
    responses = load_responses_dict(responses_path)
    print(f"  Loaded responses for {len(responses)} agents")

    # Split agents by cutoff date
    print("\nSplitting agents by cutoff date...")
    all_agents = get_all_agents_from_responses(responses_path)
    agent_dates = dataset_config.get_agent_dates(all_agents)
    pre_frontier, post_frontier = split_agents_by_dates(all_agents, agent_dates, cutoff_date)
    print(f"  Pre-frontier agents (< {cutoff_date}): {len(pre_frontier)}")
    print(f"  Post-frontier agents (>= {cutoff_date}): {len(post_frontier)}")

    # Get all task IDs from responses
    all_task_ids = set()
    for agent_responses in responses.values():
        all_task_ids.update(agent_responses.keys())
    all_task_ids = list(all_task_ids)
    print(f"  Total tasks: {len(all_task_ids)}")

    # Filter responses to pre-frontier agents only
    train_responses = {
        agent_id: agent_responses
        for agent_id, agent_responses in responses.items()
        if agent_id in pre_frontier
    }
    print(f"\nTraining Feature-IRT with {len(train_responses)} pre-frontier agents...")

    # Initialize Feature-IRT predictor
    source = EmbeddingFeatureSource(embeddings_path)
    predictor = FeatureIRTPredictor(
        source,
        use_residuals=True,
        l2_weight=0.01,
        l2_residual=10.0,
        verbose=True,
    )

    # Fit the predictor
    # Note: ground_truth_b is unused by FeatureIRTPredictor (it learns from responses)
    dummy_ground_truth = np.zeros(len(all_task_ids))
    predictor.fit(
        task_ids=all_task_ids,
        ground_truth_b=dummy_ground_truth,
        responses=train_responses,
    )

    # Extract learned abilities
    learned_abilities = predictor.learned_abilities
    print(f"\nExtracted abilities for {len(learned_abilities)} agents")

    # Build dataframe with submission dates
    agent_data = []
    for agent_name, theta in learned_abilities.items():
        submission_date = extract_submission_date(agent_name)
        if submission_date:
            agent_data.append({
                "agent": agent_name,
                "short_name": get_short_name(agent_name),
                "theta": theta,
                "submission_date": submission_date,
            })
        else:
            print(f"  Could not extract date from: {agent_name}")

    df = pd.DataFrame(agent_data)
    print(f"\n{len(df)} agents with valid dates")

    # Sort by submission date
    df = df.sort_values("submission_date")

    # For each unique date, find the maximum ability
    df_grouped = df.groupby("submission_date").agg({
        "theta": "max",
        "short_name": lambda x: list(x),
        "agent": lambda x: list(x),
    }).reset_index()

    df_grouped = df_grouped.sort_values("submission_date")
    df_grouped["frontier_theta"] = df_grouped["theta"].cummax()

    # Compute linear regression on frontier
    first_date = df["submission_date"].min()
    frontier_changes = df_grouped[df_grouped["frontier_theta"].diff().fillna(1) > 0].copy()
    frontier_dates = frontier_changes["submission_date"]
    frontier_x = np.array([(d - first_date).days for d in frontier_dates])
    frontier_y = frontier_changes["frontier_theta"].values

    slope, intercept, r_value, p_value, std_err = stats.linregress(frontier_x, frontier_y)

    print(f"\nLinear regression on frontier:")
    print(f"  Slope: {slope:.6f} theta/day = {slope * 365:.3f} theta/year")
    print(f"  Intercept: {intercept:.3f}")
    print(f"  R-value: {r_value:.4f}")
    print(f"  R²: {r_value**2:.4f}")
    print(f"  p-value: {p_value:.2e}")

    # Also compute regression on all agents
    all_x_days = np.array([(d - first_date).days for d in df["submission_date"]])
    all_y_theta = df["theta"].values
    slope_all, _, r_value_all, _, _ = stats.linregress(all_x_days, all_y_theta)
    print(f"\nLinear regression on all agents:")
    print(f"  Slope: {slope_all:.6f} theta/day = {slope_all * 365:.3f} theta/year")
    print(f"  R²: {r_value_all**2:.4f}")

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot all individual agents as scatter points
    ax.scatter(df["submission_date"], df["theta"],
               alpha=0.5, s=50, color="#3b82f6", label="Individual agents", zorder=2)

    # Plot the frontier as a step function
    ax.step(df_grouped["submission_date"], df_grouped["frontier_theta"],
            where="post", linewidth=2.5, color="#1d4ed8", label="Frontier ability", zorder=4)

    # Mark each frontier improvement with a larger point
    ax.scatter(frontier_changes["submission_date"], frontier_changes["frontier_theta"],
               s=100, color="#1d4ed8", zorder=5, edgecolors="white", linewidths=2)

    # Plot the trendline (frontier)
    trendline_x = np.array([frontier_x.min(), frontier_x.max()])
    trendline_y = slope * trendline_x + intercept
    trendline_dates = [first_date + pd.Timedelta(days=int(x)) for x in trendline_x]
    ax.plot(trendline_dates, trendline_y,
            linewidth=2.5, color="#dc2626", linestyle="--",
            label=f"Frontier trendline (R²={r_value**2:.3f})", zorder=3)

    # Add annotation with regression stats
    stats_text = (
        f"Frontier: {slope * 365:.3f} θ/year, R² = {r_value**2:.3f}\n"
        f"All agents: {slope_all * 365:.3f} θ/year, R² = {r_value_all**2:.3f}"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))

    # Formatting
    ax.set_xlabel("Agent Submission Date", fontsize=12)
    ax.set_ylabel("Feature-IRT Learned θ (Ability)", fontsize=12)
    ax.set_title(
        "Feature-IRT Learned Agent Abilities Over Time\n"
        f"(Embedding features, {len(df)} pre-frontier agents)",
        fontsize=14
    )

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45, ha="right")

    # Add grid
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Set y-axis limits with some padding
    y_min, y_max = df["theta"].min(), df["theta"].max()
    ax.set_ylim(y_min - 0.3, y_max + 0.3)

    # Legend
    ax.legend(loc="lower right", fontsize=10)

    # Tight layout
    plt.tight_layout()

    # Save
    output_path = PROJECT_ROOT / "chris_output/figures/feature_irt_theta_over_time.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to: {output_path}")

    # Also save as PDF
    output_pdf = PROJECT_ROOT / "chris_output/figures/feature_irt_theta_over_time.pdf"
    plt.savefig(output_pdf, bbox_inches="tight")
    print(f"Saved PDF to: {output_pdf}")

    plt.close()


if __name__ == "__main__":
    main()

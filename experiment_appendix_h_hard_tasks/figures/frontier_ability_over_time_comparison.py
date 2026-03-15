"""
Plot the frontier of model abilities over time for both 1PL and 2PL IRT models.

X-axis: Date (agent submission date from YYYYMMDD prefix)
Y-axis: Highest IRT ability score of any agent submitted before that date

Compares:
- 1D 1PL (Rasch model) - no discrimination parameter
- 1D 2PL - with discrimination parameter
"""

import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats


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


def compute_frontier_data(abilities: pd.DataFrame):
    """Build dataframe with frontier progression."""
    agent_data = []
    for agent_name in abilities.index:
        theta = abilities.loc[agent_name, "theta"]
        submission_date = extract_submission_date(agent_name)

        if submission_date:
            agent_data.append({
                "agent": agent_name,
                "short_name": get_short_name(agent_name),
                "theta": theta,
                "submission_date": submission_date,
            })

    df = pd.DataFrame(agent_data).sort_values("submission_date")

    # Group by date and compute frontier
    df_grouped = df.groupby("submission_date").agg({
        "theta": "max",
        "short_name": lambda x: list(x),
        "agent": lambda x: list(x),
    }).reset_index().sort_values("submission_date")

    df_grouped["frontier_theta"] = df_grouped["theta"].cummax()

    # Track which agent achieved the frontier
    frontier_agents = []
    current_max = float("-inf")
    current_agent = None
    for _, row in df_grouped.iterrows():
        if row["theta"] > current_max:
            current_max = row["theta"]
            for agent, name in zip(row["agent"], row["short_name"]):
                if abilities.loc[agent, "theta"] == row["theta"]:
                    current_agent = name
                    break
        frontier_agents.append(current_agent)
    df_grouped["frontier_agent"] = frontier_agents

    return df, df_grouped


def compute_linear_regression(df_grouped):
    """Compute linear regression on frontier changes."""
    frontier_changes = df_grouped[df_grouped["frontier_theta"].diff().fillna(1) > 0].copy()

    frontier_dates = frontier_changes["submission_date"]
    first_date = frontier_dates.min()
    frontier_x = np.array([(d - first_date).days for d in frontier_dates])
    frontier_y = frontier_changes["frontier_theta"].values

    slope, intercept, r_value, p_value, std_err = stats.linregress(frontier_x, frontier_y)

    return {
        "slope": slope,
        "intercept": intercept,
        "r_value": r_value,
        "p_value": p_value,
        "slope_per_year": slope * 365,
        "frontier_changes": frontier_changes,
        "first_date": first_date,
    }


def plot_single_model(abilities_path: Path, title_suffix: str, output_prefix: str):
    """Plot frontier for a single model."""
    abilities = pd.read_csv(abilities_path, index_col=0)
    print(f"\nProcessing {abilities_path.name}...")
    print(f"  Loaded {len(abilities)} agents")

    df, df_grouped = compute_frontier_data(abilities)
    reg = compute_linear_regression(df_grouped)

    print(f"\n  Linear regression on frontier:")
    print(f"    Slope: {reg['slope']:.4f} θ/day = {reg['slope_per_year']:.2f} θ/year")
    print(f"    R²: {reg['r_value']**2:.4f}")
    print(f"    p-value: {reg['p_value']:.2e}")

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # All agents
    ax.scatter(df["submission_date"], df["theta"],
               alpha=0.35, s=40, color="#94a3b8", label="Individual agents", zorder=2)

    # Frontier step function
    ax.step(df_grouped["submission_date"], df_grouped["frontier_theta"],
            where="post", linewidth=3.5, color="#1d4ed8", label="Frontier ability", zorder=4)

    # Frontier markers
    frontier_changes = reg["frontier_changes"]
    ax.scatter(frontier_changes["submission_date"], frontier_changes["frontier_theta"],
               s=120, color="#1d4ed8", zorder=5, edgecolors="white", linewidths=2)

    # Trendline
    first_date = reg["first_date"]
    frontier_x = np.array([(d - first_date).days for d in frontier_changes["submission_date"]])
    trendline_x = np.array([frontier_x.min(), frontier_x.max()])
    trendline_y = reg["slope"] * trendline_x + reg["intercept"]
    trendline_dates = [first_date + pd.Timedelta(days=int(x)) for x in trendline_x]
    ax.plot(trendline_dates, trendline_y,
            linewidth=2.5, color="#dc2626", linestyle="--",
            label=f"Trendline (r={reg['r_value']:.3f})", zorder=3)

    # Stats box
    stats_text = f"Frontier trend: {reg['slope_per_year']:.2f} θ/year\nr = {reg['r_value']:.3f}, R² = {reg['r_value']**2:.3f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))

    # Formatting
    ax.set_xlabel("Agent Submission Date", fontsize=12)
    ax.set_ylabel("IRT Ability (θ)", fontsize=12)
    ax.set_title(f"Frontier of SWE-bench Agent Ability Over Time\n({title_suffix}, {len(abilities)} agents, 500 tasks)", fontsize=14)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45, ha="right")

    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    y_min, y_max = df["theta"].min(), df["theta"].max()
    ax.set_ylim(y_min - 0.5, y_max + 0.5)

    ax.legend(loc="lower right", fontsize=10)

    plt.tight_layout()

    # Save
    output_dir = Path(__file__).resolve().parents[2] / "output/figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{output_prefix}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_path}")

    output_pdf = output_dir / f"{output_prefix}.pdf"
    plt.savefig(output_pdf, bbox_inches="tight")
    print(f"  Saved: {output_pdf}")

    plt.close()

    return {
        "slope_per_year": reg["slope_per_year"],
        "r_value": reg["r_value"],
        "r_squared": reg["r_value"]**2,
        "p_value": reg["p_value"],
        "n_agents": len(abilities),
    }


def plot_comparison(abilities_1pl_path: Path, abilities_2pl_path: Path):
    """Plot both models side-by-side for comparison."""
    abilities_1pl = pd.read_csv(abilities_1pl_path, index_col=0)
    abilities_2pl = pd.read_csv(abilities_2pl_path, index_col=0)

    df_1pl, df_grouped_1pl = compute_frontier_data(abilities_1pl)
    df_2pl, df_grouped_2pl = compute_frontier_data(abilities_2pl)

    reg_1pl = compute_linear_regression(df_grouped_1pl)
    reg_2pl = compute_linear_regression(df_grouped_2pl)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, df, df_grouped, reg, title in [
        (axes[0], df_1pl, df_grouped_1pl, reg_1pl, "1D 1PL (Rasch)"),
        (axes[1], df_2pl, df_grouped_2pl, reg_2pl, "1D 2PL"),
    ]:
        ax.scatter(df["submission_date"], df["theta"],
                   alpha=0.35, s=30, color="#94a3b8", zorder=2)

        ax.step(df_grouped["submission_date"], df_grouped["frontier_theta"],
                where="post", linewidth=2.5, color="#1d4ed8", zorder=4)

        frontier_changes = reg["frontier_changes"]
        ax.scatter(frontier_changes["submission_date"], frontier_changes["frontier_theta"],
                   s=80, color="#1d4ed8", zorder=5, edgecolors="white", linewidths=1.5)

        # Trendline
        first_date = reg["first_date"]
        frontier_x = np.array([(d - first_date).days for d in frontier_changes["submission_date"]])
        trendline_x = np.array([frontier_x.min(), frontier_x.max()])
        trendline_y = reg["slope"] * trendline_x + reg["intercept"]
        trendline_dates = [first_date + pd.Timedelta(days=int(x)) for x in trendline_x]
        ax.plot(trendline_dates, trendline_y,
                linewidth=2, color="#dc2626", linestyle="--", zorder=3)

        # Stats box
        stats_text = f"Trend: {reg['slope_per_year']:.2f} θ/year\nR² = {reg['r_value']**2:.3f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))

        ax.set_xlabel("Agent Submission Date", fontsize=11)
        ax.set_ylabel("IRT Ability (θ)", fontsize=11)
        ax.set_title(title, fontsize=12)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        ax.tick_params(axis='x', rotation=45)

        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

    plt.suptitle("Frontier Ability: 1PL vs 2PL IRT Models (130 agents, 500 tasks)", fontsize=14, y=1.02)
    plt.tight_layout()

    output_dir = Path(__file__).resolve().parents[2] / "output/figures"
    output_path = output_dir / "frontier_ability_1pl_vs_2pl.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved comparison: {output_path}")

    output_pdf = output_dir / "frontier_ability_1pl_vs_2pl.pdf"
    plt.savefig(output_pdf, bbox_inches="tight")
    print(f"Saved comparison: {output_pdf}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot frontier ability over time")
    parser.add_argument("--model_dir", type=str,
                       default="data/swebench_verified/irt",
                       help="Directory containing 1d/ and 1d_1pl/ subdirectories")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[2] / args.model_dir

    abilities_2pl_path = base_dir / "1d" / "abilities.csv"
    abilities_1pl_path = base_dir / "1d_1pl" / "abilities.csv"

    if not abilities_2pl_path.exists():
        print(f"Error: {abilities_2pl_path} not found")
        return

    if not abilities_1pl_path.exists():
        print(f"Error: {abilities_1pl_path} not found")
        return

    print("=" * 60)
    print("EXPERIMENT D: FRONTIER ABILITY OVER TIME")
    print("=" * 60)

    # Plot individual models
    results_2pl = plot_single_model(
        abilities_2pl_path,
        "1D 2PL IRT",
        "frontier_ability_over_time_2pl"
    )

    results_1pl = plot_single_model(
        abilities_1pl_path,
        "1D 1PL (Rasch) IRT",
        "frontier_ability_over_time_1pl"
    )

    # Plot comparison
    plot_comparison(abilities_1pl_path, abilities_2pl_path)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\n  1D 2PL:")
    print(f"    Trend: {results_2pl['slope_per_year']:.3f} θ/year")
    print(f"    R²:    {results_2pl['r_squared']:.4f}")

    print("\n  1D 1PL (Rasch):")
    print(f"    Trend: {results_1pl['slope_per_year']:.3f} θ/year")
    print(f"    R²:    {results_1pl['r_squared']:.4f}")

    print("\n  Conclusion:", end=" ")
    if results_2pl['r_squared'] > results_1pl['r_squared']:
        print("2PL has better linear fit")
    elif results_1pl['r_squared'] > results_2pl['r_squared']:
        print("1PL has better linear fit")
    else:
        print("Both models have similar linear fit")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Analyze why frontier task difficulty prediction is hard.

This script demonstrates that features which correlate with IRT difficulty
on non-frontier tasks LOSE their predictive signal on frontier tasks.

For each feature type, generates side-by-side scatter plots:
- Left panel: Feature vs Oracle difficulty for non-frontier tasks
- Right panel: Feature vs Oracle difficulty for frontier tasks

Usage:
    python -m experiment_b.frontier_feature_analysis
    python -m experiment_b.frontier_feature_analysis --output_dir chris_output/experiment_b/frontier_analysis
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from experiment_b.swebench.config import SWEBenchConfig
from experiment_b.trajectory_features.utils import load_frontier_tasks_with_difficulties


# =============================================================================
# Feature Configuration
# =============================================================================


@dataclass
class FeatureConfig:
    """Configuration for a single feature to analyze."""

    name: str  # Column name in data source
    display_name: str  # Human-readable name for plots
    source: str  # "rubric", "llm_judge", "trajectory", "date"
    per_agent: bool = False  # If True, find best-case agent for trajectory features


@dataclass
class CorrelationResult:
    """Result from analyzing one feature."""

    feature_name: str
    display_name: str
    source: str
    non_frontier_n: int
    non_frontier_pearson_r: float
    non_frontier_spearman_rho: float
    frontier_n: int
    frontier_pearson_r: float
    frontier_spearman_rho: float
    r_drop: float  # non_frontier_r - frontier_r (positive = correlation degraded)
    best_agent: Optional[str] = None  # For per-agent features


# All features to analyze
FEATURE_CONFIGS = [
    # Rubric features (10 features, per-task aggregated means)
    FeatureConfig("trajectory_length_mean", "Rubric: Trajectory Length", "rubric"),
    FeatureConfig("loop_detection_mean", "Rubric: Loop Detection", "rubric"),
    FeatureConfig("localization_quality_mean", "Rubric: Localization Quality", "rubric"),
    FeatureConfig("debugging_cycles_mean", "Rubric: Debugging Cycles", "rubric"),
    FeatureConfig("error_recovery_mean", "Rubric: Error Recovery", "rubric"),
    FeatureConfig("exploration_breadth_mean", "Rubric: Exploration Breadth", "rubric"),
    FeatureConfig("focus_drift_mean", "Rubric: Focus Drift", "rubric"),
    FeatureConfig("solution_completeness_mean", "Rubric: Solution Completeness", "rubric"),
    FeatureConfig("edge_case_handling_mean", "Rubric: Edge Case Handling", "rubric"),
    FeatureConfig("test_verification_mean", "Rubric: Test Verification", "rubric"),
    # LLM Judge features are loaded dynamically from the CSV in main()
    # Per-agent trajectory features (find best-case agent)
    FeatureConfig(
        "assistant_char_count", "Trajectory: Assistant Char Count", "trajectory", per_agent=True
    ),
    FeatureConfig(
        "n_assistant_messages", "Trajectory: N Assistant Messages", "trajectory", per_agent=True
    ),
    # Date-based feature (logistic solve date)
    FeatureConfig("x0_days", "Logistic Solve Date", "date"),
]


# =============================================================================
# Data Loading
# =============================================================================


def load_rubric_features(config: SWEBenchConfig) -> pd.DataFrame:
    """Load pre-aggregated trajectory rubric features."""
    path = config.trajectory_features_path
    if path is None or not path.exists():
        raise FileNotFoundError(f"Rubric features not found at {path}")
    df = pd.read_csv(path)
    if "task_id" in df.columns:
        df = df.set_index("task_id")
    return df


def load_llm_judge_features(config: SWEBenchConfig) -> pd.DataFrame:
    """Load LLM judge features."""
    path = config.llm_judge_path
    if path is None or not path.exists():
        raise FileNotFoundError(f"LLM judge features not found at {path}")
    df = pd.read_csv(path)
    # Handle both old format (_instance_id) and new format (instance_id)
    if "_instance_id" in df.columns:
        df = df.set_index("_instance_id")
    elif "instance_id" in df.columns:
        df = df.set_index("instance_id")
    return df


def load_trajectory_char_counts() -> pd.DataFrame:
    """Load raw per-agent trajectory features (assistant char counts)."""
    path = Path("chris_output/tensor_analysis/swebench_verified_char_counts.csv")
    if not path.exists():
        raise FileNotFoundError(f"Trajectory char counts not found at {path}")
    return pd.read_csv(path)


def load_logistic_solve_dates() -> pd.DataFrame:
    """Load logistic solve date analysis results."""
    path = Path("chris_output/logistic_solve_dates/results.csv")
    if not path.exists():
        raise FileNotFoundError(f"Logistic solve dates not found at {path}")
    df = pd.read_csv(path)
    df = df.set_index("task_id")
    return df


# =============================================================================
# Correlation Analysis
# =============================================================================


def compute_correlation(
    feature_values: pd.Series,
    oracle_difficulties: pd.Series,
    task_ids: List[str],
) -> Tuple[float, float, int]:
    """Compute correlation between feature and oracle difficulty for given tasks.

    Args:
        feature_values: Series indexed by task_id with feature values
        oracle_difficulties: Series indexed by task_id with oracle IRT beta values
        task_ids: List of task IDs to include in correlation

    Returns:
        Tuple of (pearson_r, spearman_rho, n_tasks)

    Raises:
        ValueError: If NaN values are found in the data for the specified tasks
    """
    # Find common tasks
    common_tasks = list(
        set(feature_values.index) & set(oracle_difficulties.index) & set(task_ids)
    )

    if len(common_tasks) < 3:
        return np.nan, np.nan, len(common_tasks)

    x = feature_values.loc[common_tasks].values
    y = oracle_difficulties.loc[common_tasks].values

    # Check for NaN values - these should not exist for supported tasks
    if np.any(np.isnan(x)):
        nan_tasks = [t for t in common_tasks if np.isnan(feature_values.loc[t])]
        raise ValueError(f"NaN values found in feature data for tasks: {nan_tasks[:5]}")
    if np.any(np.isnan(y)):
        nan_tasks = [t for t in common_tasks if np.isnan(oracle_difficulties.loc[t])]
        raise ValueError(f"NaN values found in oracle difficulties for tasks: {nan_tasks[:5]}")

    pearson_r, _ = pearsonr(x, y)
    spearman_rho, _ = spearmanr(x, y)

    return pearson_r, spearman_rho, len(common_tasks)


# =============================================================================
# Plotting
# =============================================================================


def is_discrete_feature(feature_values: pd.Series, max_unique: int = 10) -> bool:
    """Check if a feature is discrete (small number of unique integer-like values)."""
    unique_vals = feature_values.dropna().unique()
    if len(unique_vals) > max_unique:
        return False
    # Check if all values are integers (or close to integers)
    return all(abs(v - round(v)) < 0.01 for v in unique_vals)


def plot_discrete_feature_comparison(
    feature_values: pd.Series,
    oracle_difficulties: pd.Series,
    frontier_tasks: List[str],
    non_frontier_tasks: List[str],
    display_name: str,
    output_path: Path,
) -> CorrelationResult:
    """Create side-by-side mean+SE plots for discrete features.

    For each unique feature value, shows mean difficulty ± standard error.
    Includes a trend line fitted through the means.
    """
    fig, (ax_non, ax_frontier) = plt.subplots(1, 2, figsize=(14, 6))

    # Compute correlations (on raw data, not means)
    r_non, rho_non, n_non = compute_correlation(
        feature_values, oracle_difficulties, non_frontier_tasks
    )
    r_frontier, rho_frontier, n_frontier = compute_correlation(
        feature_values, oracle_difficulties, frontier_tasks
    )

    def plot_mean_se_panel(ax, task_list, panel_title, r_val, n_val, color):
        """Helper to create mean+SE plot for one panel."""
        common_tasks = list(
            set(feature_values.index) & set(oracle_difficulties.index) & set(task_list)
        )

        if len(common_tasks) < 3:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{panel_title} (N={n_val})", fontsize=12)
            return

        x = feature_values.loc[common_tasks]
        y = oracle_difficulties.loc[common_tasks]

        # Group by feature value
        df = pd.DataFrame({"feature": x, "difficulty": y})
        grouped = df.groupby("feature")["difficulty"].agg(["mean", "std", "count"])
        grouped["se"] = grouped["std"] / np.sqrt(grouped["count"])

        # Only plot values that have data
        valid_levels = grouped.index.values
        means = grouped["mean"].values
        ses = grouped["se"].values
        counts = grouped["count"].values

        # Plot mean + SE bars
        ax.errorbar(
            valid_levels, means, yerr=ses,
            fmt="o", markersize=8, capsize=5, capthick=2,
            color=color, ecolor=color, markeredgecolor="black", markeredgewidth=1,
            label="Mean ± SE"
        )

        # Add count labels
        for level, mean, count in zip(valid_levels, means, counts):
            ax.annotate(f"n={int(count)}", (level, mean), textcoords="offset points",
                       xytext=(0, 12), ha="center", fontsize=9, color="gray")

        # Fit trend line through means (if at least 2 points)
        if len(valid_levels) >= 2:
            z = np.polyfit(valid_levels, means, 1)
            p_line = np.poly1d(z)
            x_range = np.linspace(min(valid_levels), max(valid_levels), 100)
            ax.plot(x_range, p_line(x_range), "r--", linewidth=2, alpha=0.7, label="Trend")

        ax.legend(loc="best", fontsize=9)

        title_color = "green" if abs(r_val) >= 0.15 else "gray"
        ax.set_title(
            f"{panel_title} (N={n_val})\nPearson r = {r_val:.3f}",
            fontsize=12,
            color=title_color,
        )

    # --- Non-frontier panel ---
    plot_mean_se_panel(ax_non, non_frontier_tasks, "Non-Frontier Tasks", r_non, n_non, "steelblue")
    ax_non.set_xlabel(display_name, fontsize=11)
    ax_non.set_ylabel("Oracle IRT Difficulty (β)", fontsize=11)
    ax_non.grid(True, alpha=0.3)

    # --- Frontier panel ---
    plot_mean_se_panel(ax_frontier, frontier_tasks, "Frontier Tasks", r_frontier, n_frontier, "darkorange")
    ax_frontier.set_xlabel(display_name, fontsize=11)
    ax_frontier.set_ylabel("Oracle IRT Difficulty (β)", fontsize=11)
    ax_frontier.grid(True, alpha=0.3)

    # Match y-axis limits across panels for comparison
    y_min = min(ax_non.get_ylim()[0], ax_frontier.get_ylim()[0])
    y_max = max(ax_non.get_ylim()[1], ax_frontier.get_ylim()[1])
    ax_non.set_ylim(y_min, y_max)
    ax_frontier.set_ylim(y_min, y_max)

    fig.suptitle(f"Mean Difficulty by Feature Value: {display_name}", fontsize=14, y=1.02)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    r_drop = r_non - r_frontier if not (np.isnan(r_non) or np.isnan(r_frontier)) else np.nan

    return CorrelationResult(
        feature_name=output_path.stem,
        display_name=display_name,
        source="unknown",
        non_frontier_n=n_non,
        non_frontier_pearson_r=r_non,
        non_frontier_spearman_rho=rho_non,
        frontier_n=n_frontier,
        frontier_pearson_r=r_frontier,
        frontier_spearman_rho=rho_frontier,
        r_drop=r_drop,
        best_agent=None,
    )


def plot_correlation_comparison(
    feature_values: pd.Series,
    oracle_difficulties: pd.Series,
    frontier_tasks: List[str],
    non_frontier_tasks: List[str],
    display_name: str,
    output_path: Path,
    best_agent: Optional[str] = None,
) -> CorrelationResult:
    """Create side-by-side scatter plots showing correlation breakdown.

    Left panel: Non-frontier tasks (correlation should be present)
    Right panel: Frontier tasks (correlation should be weaker/absent)
    """
    fig, (ax_non, ax_frontier) = plt.subplots(1, 2, figsize=(14, 6))

    # Compute correlations
    r_non, rho_non, n_non = compute_correlation(
        feature_values, oracle_difficulties, non_frontier_tasks
    )
    r_frontier, rho_frontier, n_frontier = compute_correlation(
        feature_values, oracle_difficulties, frontier_tasks
    )

    # --- Non-frontier panel ---
    common_non = list(
        set(feature_values.index) & set(oracle_difficulties.index) & set(non_frontier_tasks)
    )
    if len(common_non) >= 3:
        x_non = feature_values.loc[common_non].values
        y_non = oracle_difficulties.loc[common_non].values

        ax_non.scatter(x_non, y_non, alpha=0.5, s=30, c="steelblue", edgecolors="none")

        # Add trend line
        if not np.isnan(r_non):
            z = np.polyfit(x_non, y_non, 1)
            p_line = np.poly1d(z)
            x_range = np.linspace(np.min(x_non), np.max(x_non), 100)
            ax_non.plot(x_range, p_line(x_range), "r--", linewidth=2)

        title_color = "green" if abs(r_non) >= 0.15 else "gray"
        ax_non.set_title(
            f"Non-Frontier Tasks (N={n_non})\n"
            f"Pearson r = {r_non:.3f}, Spearman ρ = {rho_non:.3f}",
            fontsize=12,
            color=title_color,
        )
    else:
        ax_non.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax_non.transAxes)
        ax_non.set_title(f"Non-Frontier Tasks (N={n_non})", fontsize=12)

    ax_non.set_xlabel(display_name, fontsize=11)
    ax_non.set_ylabel("Oracle IRT Difficulty (β)", fontsize=11)
    ax_non.grid(True, alpha=0.3)

    # --- Frontier panel ---
    common_frontier = list(
        set(feature_values.index) & set(oracle_difficulties.index) & set(frontier_tasks)
    )
    if len(common_frontier) >= 3:
        x_frontier = feature_values.loc[common_frontier].values
        y_frontier = oracle_difficulties.loc[common_frontier].values

        ax_frontier.scatter(x_frontier, y_frontier, alpha=0.6, s=40, c="darkorange", edgecolors="none")

        # Add trend line
        if not np.isnan(r_frontier):
            z = np.polyfit(x_frontier, y_frontier, 1)
            p_line = np.poly1d(z)
            x_range = np.linspace(np.min(x_frontier), np.max(x_frontier), 100)
            ax_frontier.plot(x_range, p_line(x_range), "r--", linewidth=2)

        r_drop = r_non - r_frontier if not (np.isnan(r_non) or np.isnan(r_frontier)) else np.nan
        title_color = "red" if (not np.isnan(r_drop) and r_drop > 0.1) else "gray"
        ax_frontier.set_title(
            f"Frontier Tasks (N={n_frontier})\n"
            f"Pearson r = {r_frontier:.3f}, Spearman ρ = {rho_frontier:.3f}",
            fontsize=12,
            color=title_color,
        )
    else:
        ax_frontier.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax_frontier.transAxes)
        ax_frontier.set_title(f"Frontier Tasks (N={n_frontier})", fontsize=12)

    ax_frontier.set_xlabel(display_name, fontsize=11)
    ax_frontier.set_ylabel("Oracle IRT Difficulty (β)", fontsize=11)
    ax_frontier.grid(True, alpha=0.3)

    # Suptitle
    suptitle = f"Feature-Difficulty Correlation: {display_name}"
    if best_agent:
        suptitle += f"\n(Best-case agent: {best_agent})"
    fig.suptitle(suptitle, fontsize=14, y=1.02)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    r_drop = r_non - r_frontier if not (np.isnan(r_non) or np.isnan(r_frontier)) else np.nan

    return CorrelationResult(
        feature_name=output_path.stem,
        display_name=display_name,
        source="trajectory" if best_agent else "unknown",
        non_frontier_n=n_non,
        non_frontier_pearson_r=r_non,
        non_frontier_spearman_rho=rho_non,
        frontier_n=n_frontier,
        frontier_pearson_r=r_frontier,
        frontier_spearman_rho=rho_frontier,
        r_drop=r_drop,
        best_agent=best_agent,
    )


def analyze_per_agent_feature(
    config: FeatureConfig,
    char_counts_df: pd.DataFrame,
    oracle_difficulties: pd.Series,
    frontier_tasks: List[str],
    non_frontier_tasks: List[str],
    output_dir: Path,
) -> CorrelationResult:
    """Find agent with strongest frontier correlation, then plot scatter."""
    feature_col = config.name

    best_agent = None
    best_abs_frontier_r = -1.0  # Track absolute value separately

    # Evaluate all agents to find best-case
    for agent in char_counts_df["agent"].unique():
        agent_data = char_counts_df[char_counts_df["agent"] == agent].copy()
        agent_data = agent_data.set_index("task_id")

        if feature_col not in agent_data.columns:
            continue

        feature_values = agent_data[feature_col]

        # Compute frontier correlation for this agent
        try:
            r_frontier, _, n = compute_correlation(
                feature_values, oracle_difficulties, frontier_tasks
            )
        except ValueError:
            # NaN values - skip this agent
            continue

        # Track best by absolute correlation on frontier
        if not np.isnan(r_frontier) and abs(r_frontier) > best_abs_frontier_r:
            best_abs_frontier_r = abs(r_frontier)
            best_agent = agent

    if best_agent is None:
        raise ValueError(f"No valid agent found for {config.name}")

    # Use best agent's data for the plot
    best_agent_data = char_counts_df[char_counts_df["agent"] == best_agent].copy()
    best_agent_data = best_agent_data.set_index("task_id")
    feature_values = best_agent_data[feature_col]

    # Filter outliers: clip to 99th percentile of non-frontier tasks
    non_frontier_values = feature_values.loc[
        [t for t in non_frontier_tasks if t in feature_values.index]
    ]
    upper_clip = non_frontier_values.quantile(0.99)
    filtered_tasks = feature_values[feature_values <= upper_clip].index.tolist()
    filtered_frontier = [t for t in frontier_tasks if t in filtered_tasks]
    filtered_non_frontier = [t for t in non_frontier_tasks if t in filtered_tasks]

    n_removed = len(non_frontier_tasks) - len(filtered_non_frontier)
    if n_removed > 0:
        print(f"  Removed {n_removed} outliers (>{upper_clip:.0f}) from non-frontier")

    output_path = output_dir / "trajectory" / f"{config.name}.png"
    result = plot_correlation_comparison(
        feature_values,
        oracle_difficulties,
        filtered_frontier,
        filtered_non_frontier,
        config.display_name,
        output_path,
        best_agent=best_agent,
    )
    result.source = "trajectory"
    result.feature_name = config.name

    print(f"  Best agent: {best_agent} (frontier |r| = {best_abs_frontier_r:.3f})")

    return result


# =============================================================================
# Summary Generation
# =============================================================================


def save_correlation_table(results: List[CorrelationResult], output_path: Path) -> None:
    """Save correlation comparison table to CSV."""
    rows = []
    for r in results:
        rows.append({
            "feature": r.feature_name,
            "display_name": r.display_name,
            "source": r.source,
            "non_frontier_n": r.non_frontier_n,
            "non_frontier_pearson_r": r.non_frontier_pearson_r,
            "non_frontier_spearman_rho": r.non_frontier_spearman_rho,
            "frontier_n": r.frontier_n,
            "frontier_pearson_r": r.frontier_pearson_r,
            "frontier_spearman_rho": r.frontier_spearman_rho,
            "r_drop": r.r_drop,
            "best_agent": r.best_agent or "",
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("r_drop", ascending=False)  # Largest drop first
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved correlation table to {output_path}")


def generate_findings_markdown(results: List[CorrelationResult], output_path: Path) -> None:
    """Generate markdown summary document."""
    sorted_results = sorted(
        results, key=lambda x: x.r_drop if not np.isnan(x.r_drop) else -999, reverse=True
    )

    with open(output_path, "w") as f:
        f.write("# Frontier Feature Analysis: Why Difficulty Prediction is Hard\n\n")

        f.write("## Key Finding\n\n")
        f.write("Features that correlate with IRT difficulty on non-frontier tasks ")
        f.write("**lose their predictive signal** on frontier tasks.\n\n")

        f.write("## Correlation Comparison Table\n\n")
        f.write("Sorted by correlation drop (largest drop first).\n\n")
        f.write("| Feature | Source | Non-Frontier r | Frontier r | Drop |\n")
        f.write("|---------|--------|---------------|------------|------|\n")

        for r in sorted_results:
            non_r = f"{r.non_frontier_pearson_r:.3f}" if not np.isnan(r.non_frontier_pearson_r) else "N/A"
            front_r = f"{r.frontier_pearson_r:.3f}" if not np.isnan(r.frontier_pearson_r) else "N/A"
            drop = f"{r.r_drop:.3f}" if not np.isnan(r.r_drop) else "N/A"
            f.write(f"| {r.display_name} | {r.source} | {non_r} | {front_r} | {drop} |\n")

        f.write("\n## Summary Statistics\n\n")

        valid_drops = [r.r_drop for r in results if not np.isnan(r.r_drop)]
        valid_non_r = [r.non_frontier_pearson_r for r in results if not np.isnan(r.non_frontier_pearson_r)]
        valid_front_r = [r.frontier_pearson_r for r in results if not np.isnan(r.frontier_pearson_r)]

        if valid_drops:
            f.write(f"- **Average correlation drop**: {np.mean(valid_drops):.3f}\n")
        if valid_non_r:
            f.write(f"- **Average non-frontier r**: {np.mean(valid_non_r):.3f}\n")
        if valid_front_r:
            f.write(f"- **Average frontier r**: {np.mean(valid_front_r):.3f}\n")

        f.write("\n## Interpretation\n\n")
        f.write("Frontier tasks are defined as tasks with:\n")
        f.write("- **0% pre-frontier pass rate**: No pre-frontier agent solves them\n")
        f.write("- **>0% post-frontier pass rate**: At least one post-frontier agent solves them\n\n")

        f.write("The correlation breakdown occurs because:\n\n")
        f.write("1. **Pre-frontier agents fail on ALL frontier tasks** by definition\n")
        f.write("2. **Trajectory features are 'failure trajectories'** with no variance in outcome\n")
        f.write("3. **Features cannot distinguish** 'more impossible' from 'less impossible'\n")
        f.write("4. **Selection effect**: Frontier tasks are the hardest subset where feature ranges may saturate\n")

    print(f"Saved findings to {output_path}")


def create_summary_grids(output_dir: Path) -> None:
    """Create summary grid images combining individual plots by source.

    Creates:
    - rubric_grid.png: 5x2 grid of all rubric features
    - llm_judge_grid.png: 5x2 grid of all LLM judge features
    - other_grid.png: trajectory + date features
    """
    from PIL import Image

    def create_grid(image_paths: List[Path], output_path: Path, ncols: int = 2) -> None:
        """Create a grid image from a list of image paths."""
        # Filter to existing files
        existing = [p for p in image_paths if p.exists()]
        if not existing:
            print(f"  No images found for {output_path.name}")
            return

        # Load images
        images = [Image.open(p) for p in existing]

        # Get dimensions (assume all same size)
        img_width, img_height = images[0].size

        # Calculate grid dimensions
        nrows = (len(images) + ncols - 1) // ncols
        grid_width = ncols * img_width
        grid_height = nrows * img_height

        # Create grid
        grid = Image.new("RGB", (grid_width, grid_height), color="white")

        for idx, img in enumerate(images):
            row = idx // ncols
            col = idx % ncols
            x = col * img_width
            y = row * img_height
            grid.paste(img, (x, y))

        # Save
        grid.save(output_path, quality=95)
        print(f"  Saved: {output_path} ({len(existing)} images, {nrows}x{ncols} grid)")

    print("\nCreating summary grids...")

    # Rubric features (10 features -> 5x2 grid)
    rubric_dir = output_dir / "rubric"
    rubric_features = [
        "trajectory_length_mean", "loop_detection_mean", "localization_quality_mean",
        "debugging_cycles_mean", "error_recovery_mean", "exploration_breadth_mean",
        "focus_drift_mean", "solution_completeness_mean", "edge_case_handling_mean",
        "test_verification_mean"
    ]
    rubric_paths = [rubric_dir / f"{f}.png" for f in rubric_features]
    create_grid(rubric_paths, output_dir / "rubric_grid.png", ncols=2)

    # LLM Judge features (dynamically discover from output directory)
    llm_dir = output_dir / "llm_judge"
    llm_paths = sorted(llm_dir.glob("*.png")) if llm_dir.exists() else []
    create_grid(llm_paths, output_dir / "llm_judge_grid.png", ncols=2)

    # Trajectory features (2 features -> 1x2 grid)
    trajectory_paths = [
        output_dir / "trajectory" / "assistant_char_count.png",
        output_dir / "trajectory" / "n_assistant_messages.png",
    ]
    create_grid(trajectory_paths, output_dir / "trajectory_grid.png", ncols=2)


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Analyze feature-difficulty correlations across frontier/non-frontier tasks"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("chris_output/experiment_b/frontier_analysis"),
        help="Output directory for figures and summary",
    )
    parser.add_argument(
        "--frontier_def",
        type=str,
        default="zero_pre",
        choices=["zero_pre", "human_hard"],
        help="Frontier definition to use (default: zero_pre). "
             "'zero_pre' = 0%% pre, >0%% post; "
             "'human_hard' = human-labeled difficulty >= '1-4 hours'",
    )
    args = parser.parse_args()

    # Append frontier_def to output_dir to keep results separate
    output_dir = args.output_dir / args.frontier_def
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Frontier Feature Analysis (frontier_def={args.frontier_def})")
    print("=" * 70)

    # Load configuration and frontier tasks
    config = SWEBenchConfig()

    print("\nLoading frontier tasks and oracle difficulties...")
    frontier_tasks, oracle_items, pre_frontier, post_frontier = load_frontier_tasks_with_difficulties(
        config, frontier_def=args.frontier_def
    )
    oracle_diff = oracle_items["b"]

    # Non-frontier = all tasks minus frontier
    all_task_ids = set(config.all_task_ids)
    non_frontier_tasks = list(all_task_ids - set(frontier_tasks))

    print(f"  Frontier tasks: {len(frontier_tasks)}")
    print(f"  Non-frontier tasks: {len(non_frontier_tasks)}")
    print(f"  Pre-frontier agents: {len(pre_frontier)}")
    print(f"  Post-frontier agents: {len(post_frontier)}")

    # Load all feature sources
    print("\nLoading feature sources...")

    rubric_features = load_rubric_features(config)
    print(f"  Rubric features: {rubric_features.shape}")

    llm_judge_features = load_llm_judge_features(config)
    print(f"  LLM judge features: {llm_judge_features.shape}")

    # Generate LLM judge feature configs dynamically from loaded columns
    llm_judge_feature_configs = [
        FeatureConfig(col, f"LLM Judge: {col.replace('_', ' ').title()}", "llm_judge")
        for col in llm_judge_features.columns
    ]

    char_counts = load_trajectory_char_counts()
    print(f"  Trajectory char counts: {char_counts.shape}")

    logistic_dates = load_logistic_solve_dates()
    print(f"  Logistic solve dates: {logistic_dates.shape}")

    # Analyze features
    print("\n" + "=" * 70)
    print("Analyzing Features")
    print("=" * 70)

    results = []

    # Combine static configs with dynamically loaded LLM judge features
    all_feature_configs = FEATURE_CONFIGS + llm_judge_feature_configs

    for feature_config in all_feature_configs:
        print(f"\n{feature_config.display_name}...")

        if feature_config.source == "rubric":
            if feature_config.name not in rubric_features.columns:
                print(f"  Skipped ({feature_config.name} not found)")
                continue

            feature_values = rubric_features[feature_config.name]
            output_path = output_dir / "rubric" / f"{feature_config.name}.png"
            result = plot_correlation_comparison(
                feature_values,
                oracle_diff,
                frontier_tasks,
                non_frontier_tasks,
                feature_config.display_name,
                output_path,
            )
            result.source = "rubric"
            result.feature_name = feature_config.name

        elif feature_config.source == "llm_judge":
            if feature_config.name not in llm_judge_features.columns:
                print(f"  Skipped ({feature_config.name} not found)")
                continue

            feature_values = llm_judge_features[feature_config.name]
            output_path = output_dir / "llm_judge" / f"{feature_config.name}.png"

            # LLM judge features are discrete (ordinal scales) - use mean+SE plot
            if is_discrete_feature(feature_values):
                result = plot_discrete_feature_comparison(
                    feature_values,
                    oracle_diff,
                    frontier_tasks,
                    non_frontier_tasks,
                    feature_config.display_name,
                    output_path,
                )
            else:
                result = plot_correlation_comparison(
                    feature_values,
                    oracle_diff,
                    frontier_tasks,
                    non_frontier_tasks,
                    feature_config.display_name,
                    output_path,
                )
            result.source = "llm_judge"
            result.feature_name = feature_config.name

        elif feature_config.source == "trajectory" and feature_config.per_agent:
            result = analyze_per_agent_feature(
                feature_config,
                char_counts,
                oracle_diff,
                frontier_tasks,
                non_frontier_tasks,
                output_dir,
            )

        elif feature_config.source == "date":
            if feature_config.name not in logistic_dates.columns:
                print(f"  Skipped ({feature_config.name} not found)")
                continue

            # Date features have expected NaN values (tasks that were never solved)
            # Filter to tasks with valid values and reasonable range (0 to 1150)
            # Values at bounds (-100, 1219) are optimization artifacts
            feature_values = logistic_dates[feature_config.name].dropna()
            feature_values = feature_values[(feature_values >= 0) & (feature_values <= 1150)]
            valid_tasks = set(feature_values.index)
            valid_frontier = [t for t in frontier_tasks if t in valid_tasks]
            valid_non_frontier = [t for t in non_frontier_tasks if t in valid_tasks]

            print(f"  Valid tasks (0-1150 days): {len(valid_frontier)} frontier, {len(valid_non_frontier)} non-frontier")

            output_path = output_dir / f"{feature_config.name}.png"
            result = plot_correlation_comparison(
                feature_values,
                oracle_diff,
                valid_frontier,
                valid_non_frontier,
                feature_config.display_name,
                output_path,
            )
            result.source = "date"
            result.feature_name = feature_config.name

        else:
            print(f"  Unknown source: {feature_config.source}")
            continue

        results.append(result)
        print(
            f"  Non-frontier r={result.non_frontier_pearson_r:.3f}, "
            f"Frontier r={result.frontier_pearson_r:.3f}, "
            f"Drop={result.r_drop:.3f}"
        )

    # Generate outputs
    print("\n" + "=" * 70)
    print("Generating Summary")
    print("=" * 70)

    save_correlation_table(results, output_dir / "correlation_comparison.csv")
    generate_findings_markdown(results, output_dir / "findings.md")

    # Create summary grids
    create_summary_grids(output_dir)

    print(f"\nDone! Results saved to {output_dir}")
    print(f"  - {len(results)} feature plots generated")
    print(f"  - 3 summary grids created")


if __name__ == "__main__":
    main()

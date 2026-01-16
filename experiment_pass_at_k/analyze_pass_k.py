"""
Analyze pass@k results and compute correlation with IRT difficulty.

This script loads the results from run_pass_k.py and:
1. Computes pass@k rates for each (model, task) pair
2. Correlates pass@k with IRT difficulty scores
3. Analyzes how correlation varies with k
4. Generates visualization plots

Usage:
    python -m experiment_pass_at_k.analyze_pass_k

    # Analyze specific model only
    python -m experiment_pass_at_k.analyze_pass_k --model openai/o1-2024-12-05

    # Generate plots
    python -m experiment_pass_at_k.analyze_pass_k --plot
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from experiment_pass_at_k.config import ExperimentPassKConfig


def load_results_for_model(results_dir: Path) -> List[Dict]:
    """Load all task results from a model's results directory."""
    results = []
    for task_file in results_dir.glob("*.json"):
        if task_file.name == "summary.json":
            continue
        with open(task_file) as f:
            results.append(json.load(f))
    return results


def compute_pass_at_k_from_attempts(attempts: List[Dict], k: int) -> float:
    """Compute pass@k using only the first k attempts.

    pass@k = 1 if any of the first k attempts succeeded, else 0
    (For small samples, this is more appropriate than the probabilistic formula)
    """
    first_k = attempts[:k]
    return 1.0 if any(a.get("success", False) for a in first_k) else 0.0


def compute_empirical_pass_rate(attempts: List[Dict], k: int) -> float:
    """Compute empirical pass rate from first k attempts."""
    first_k = attempts[:k]
    if not first_k:
        return 0.0
    successes = sum(1 for a in first_k if a.get("success", False))
    return successes / len(first_k)


def analyze_single_model(
    results: List[Dict],
    k_values: List[int] = None,
) -> Dict:
    """Analyze results for a single model.

    Returns:
        Dict with correlation results and per-task data.
    """
    if not results:
        return {"error": "No results to analyze"}

    # Default k values
    max_k = max(len(r["attempts"]) for r in results)
    if k_values is None:
        k_values = [1, 2, 3, 5, 10, max_k] if max_k >= 10 else list(range(1, max_k + 1))
    k_values = [k for k in k_values if k <= max_k]

    model = results[0]["model"]

    # Build DataFrame
    data = []
    for r in results:
        task_id = r["task_id"]
        difficulty = r["irt_difficulty"]
        attempts = r["attempts"]

        row = {
            "task_id": task_id,
            "difficulty": difficulty,
            "total_attempts": len(attempts),
        }

        # Compute pass@k for each k value
        for k in k_values:
            row[f"pass_at_{k}"] = compute_pass_at_k_from_attempts(attempts, k)
            row[f"pass_rate_{k}"] = compute_empirical_pass_rate(attempts, k)

        # Overall stats
        row["pass_rate_all"] = r["summary"]["pass_rate"]
        row["first_success"] = r["summary"]["first_success_at"]

        data.append(row)

    df = pd.DataFrame(data)

    # Compute correlations for each k
    correlations = {}
    for k in k_values:
        pass_rate_col = f"pass_rate_{k}"
        if pass_rate_col in df.columns:
            # Pearson correlation: difficulty vs pass_rate (expect negative)
            r, p = stats.pearsonr(df["difficulty"], df[pass_rate_col])
            rho, rho_p = stats.spearmanr(df["difficulty"], df[pass_rate_col])

            correlations[k] = {
                "pearson_r": float(r),
                "pearson_p": float(p),
                "spearman_rho": float(rho),
                "spearman_p": float(rho_p),
                "n_tasks": len(df),
                "mean_pass_rate": float(df[pass_rate_col].mean()),
                "std_pass_rate": float(df[pass_rate_col].std()),
            }

    return {
        "model": model,
        "k_values": k_values,
        "max_k": max_k,
        "n_tasks": len(df),
        "correlations": correlations,
        "per_task": df.to_dict(orient="records"),
    }


def generate_plots(
    analysis_results: Dict,
    output_dir: Path,
):
    """Generate visualization plots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Get data
    per_task = pd.DataFrame(analysis_results["per_task"])
    correlations = analysis_results["correlations"]
    k_values = analysis_results["k_values"]

    # Plot 1: Scatter - Difficulty vs Pass Rate (at max k)
    ax1 = axes[0]
    max_k = analysis_results["max_k"]
    pass_rate_col = f"pass_rate_{max_k}"

    ax1.scatter(per_task["difficulty"], per_task[pass_rate_col], alpha=0.7, s=100)

    # Add regression line
    z = np.polyfit(per_task["difficulty"], per_task[pass_rate_col], 1)
    p = np.poly1d(z)
    x_line = np.linspace(per_task["difficulty"].min(), per_task["difficulty"].max(), 100)
    ax1.plot(x_line, p(x_line), "r--", alpha=0.8, label=f"r={correlations[max_k]['pearson_r']:.3f}")

    ax1.set_xlabel("IRT Difficulty (b)")
    ax1.set_ylabel(f"Pass Rate (k={max_k})")
    ax1.set_title(f"Difficulty vs Pass@{max_k}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Line - Correlation strength vs k
    ax2 = axes[1]
    k_vals = sorted(correlations.keys())
    pearson_rs = [correlations[k]["pearson_r"] for k in k_vals]
    spearman_rhos = [correlations[k]["spearman_rho"] for k in k_vals]

    ax2.plot(k_vals, pearson_rs, "b-o", label="Pearson r", markersize=8)
    ax2.plot(k_vals, spearman_rhos, "g-s", label="Spearman ρ", markersize=8)
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    ax2.set_xlabel("k (number of attempts)")
    ax2.set_ylabel("Correlation with Difficulty")
    ax2.set_title("Correlation Strength vs k")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Bar - Pass rate by difficulty quintile
    ax3 = axes[2]
    per_task["difficulty_quintile"] = pd.qcut(per_task["difficulty"], 5, labels=["Q1\n(Easy)", "Q2", "Q3", "Q4", "Q5\n(Hard)"])
    quintile_means = per_task.groupby("difficulty_quintile")[pass_rate_col].mean()

    bars = ax3.bar(range(5), quintile_means.values, color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, 5)))
    ax3.set_xticks(range(5))
    ax3.set_xticklabels(quintile_means.index)
    ax3.set_xlabel("Difficulty Quintile")
    ax3.set_ylabel(f"Mean Pass Rate (k={max_k})")
    ax3.set_title("Pass Rate by Difficulty")
    ax3.set_ylim(0, 1)

    # Add value labels on bars
    for bar, val in zip(bars, quintile_means.values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()

    # Save plot
    plot_file = output_dir / "pass_k_analysis.png"
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {plot_file}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze pass@k results")
    parser.add_argument(
        "--model",
        type=str,
        help="Analyze specific model only",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate visualization plots",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Override output directory",
    )
    args = parser.parse_args()

    config = ExperimentPassKConfig()
    output_dir = Path(args.output_dir) if args.output_dir else config.output_dir

    print("=" * 60)
    print("PASS@K ANALYSIS")
    print("=" * 60)

    # Find result directories
    result_dirs = list(output_dir.glob("results_*"))
    if not result_dirs:
        print(f"No results found in {output_dir}")
        return

    all_analyses = {}

    for result_dir in result_dirs:
        if not result_dir.is_dir():
            continue

        model_name = result_dir.name.replace("results_", "")

        if args.model and model_name not in args.model:
            continue

        print(f"\nAnalyzing {model_name}...")

        # Load results
        results = load_results_for_model(result_dir)
        if not results:
            print(f"  No results found")
            continue

        print(f"  Found {len(results)} tasks")

        # Analyze
        analysis = analyze_single_model(results)
        all_analyses[model_name] = analysis

        # Print results
        print(f"\n  Correlations (difficulty vs pass rate):")
        print(f"  {'k':<6} {'Pearson r':<12} {'p-value':<12} {'Spearman ρ':<12} {'Mean Pass%':<12}")
        print(f"  {'-'*54}")

        for k in analysis["k_values"]:
            corr = analysis["correlations"][k]
            print(f"  {k:<6} {corr['pearson_r']:<12.4f} {corr['pearson_p']:<12.4f} "
                  f"{corr['spearman_rho']:<12.4f} {corr['mean_pass_rate']*100:<12.1f}")

        # Generate plots if requested
        if args.plot:
            generate_plots(analysis, output_dir / "analysis")

    # Save analysis results
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    analysis_file = analysis_dir / "correlation_results.json"
    with open(analysis_file, "w") as f:
        # Convert any non-serializable types
        json.dump(all_analyses, f, indent=2, default=str)

    print(f"\nAnalysis saved to {analysis_file}")

    # Summary across models
    if len(all_analyses) > 1:
        print("\n" + "=" * 60)
        print("CROSS-MODEL SUMMARY")
        print("=" * 60)
        for model_name, analysis in all_analyses.items():
            max_k = analysis["max_k"]
            corr = analysis["correlations"].get(max_k, {})
            print(f"\n{model_name}:")
            print(f"  Tasks: {analysis['n_tasks']}")
            print(f"  Max k: {max_k}")
            print(f"  Correlation (k={max_k}): r={corr.get('pearson_r', 'N/A'):.4f}")


if __name__ == "__main__":
    main()

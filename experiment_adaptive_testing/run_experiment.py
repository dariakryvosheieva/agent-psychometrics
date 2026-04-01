"""Run the adaptive task selection experiment and generate plots.

Usage:
    python -m experiment_adaptive_testing.run_experiment --predictions_csv path/to/predictions.csv

To generate predictions first (train on Verified+TerminalBench+GSO, predict Pro):
    python -m experiment_agent_features.predict_question_difficulty_multi_benchmark \
        --split_by benchmark \
        --train_benchmarks verified,terminalbench,gso \
        --ood_benchmark pro \
        --out_dir output/experiment_adaptive_testing/ood_predictions \
        --method judge
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .cat_simulation import ExperimentConfig, run_experiment



def plot_spearman_curves(
    results: dict,
    output_path: Path,
) -> None:
    """Plot Spearman correlation vs. number of tasks administered."""
    # Start plotting from step 10 to skip noisy early steps
    start = 10
    steps = results["step"][start - 1:]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, results["fisher_oracle"][start - 1:], color="tab:blue", linewidth=2,
            label="Fisher (Oracle)")
    ax.plot(steps, results["fisher_predicted"][start - 1:], color="tab:orange", linewidth=2,
            label="Fisher (Predicted)")
    ax.plot(steps, results["random"][start - 1:], color="gray", linewidth=2, linestyle="--",
            label="Random")

    ax.set_xlabel("Number of Tasks")
    ax.set_ylabel("Spearman Correlation with Full Benchmark")
    ax.set_ylim(-0.2, 1.05)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {output_path}")


def plot_reliability_curves(
    results: dict,
    output_path: Path,
) -> None:
    """Plot empirical reliability vs. number of tasks administered."""
    # Start from the first step where all methods are defined (non-NaN)
    import math
    start = 1
    for i, s in enumerate(results["step"]):
        vals = [results[k][i] for k in [
            "fisher_predicted_reliability", "fisher_oracle_reliability", "random_reliability",
        ]]
        if all(not math.isnan(v) for v in vals):
            start = s
            break
    # Cut off at 100 tasks
    end = min(100, results["step"][-1])
    mask = [(s >= start and s <= end) for s in results["step"]]
    steps = [s for s, m in zip(results["step"], mask) if m]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, [v for v, m in zip(results["fisher_oracle_reliability"], mask) if m],
            color="tab:blue", linewidth=2, label="IRT (Oracle)")
    ax.plot(steps, [v for v, m in zip(results["fisher_predicted_reliability"], mask) if m],
            color="tab:orange", linewidth=2, label="IRT (Predicted)")
    ax.plot(steps, [v for v, m in zip(results["random_reliability"], mask) if m],
            color="gray", linewidth=2, linestyle="--", label="Random")

    ax.set_xlabel("Number of Tasks", fontsize=16)
    ax.set_ylabel("Empirical Reliability", fontsize=16)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower right", fontsize=14)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Adaptive task selection experiment")
    parser.add_argument(
        "--predictions_csv",
        type=str,
        default="output/experiment_adaptive_testing/ood_predictions/predictions.csv",
        help="Path to predicted difficulties CSV from multi-benchmark experiment.",
    )
    parser.add_argument(
        "--oracle_items",
        type=str,
        default="data/swebench_pro/irt/1d_1pl/items.csv",
        help="Path to ground truth IRT item difficulties.",
    )
    parser.add_argument(
        "--responses",
        type=str,
        default="data/swebench_pro/responses.jsonl",
        help="Path to response matrix JSONL.",
    )
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123, 314, 999])
    parser.add_argument("--prior_sigma", type=float, default=3.0)
    parser.add_argument("--output_dir", type=str, default="output/experiment_adaptive_testing")
    args = parser.parse_args()

    predictions_path = Path(args.predictions_csv)
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {predictions_path}\n"
            f"Generate it first with:\n"
            f"  python -m experiment_agent_features.predict_question_difficulty_multi_benchmark \\\n"
            f"      --split_by benchmark \\\n"
            f"      --train_benchmarks verified,terminalbench,gso \\\n"
            f"      --ood_benchmark pro \\\n"
            f"      --out_dir output/experiment_adaptive_testing/ood_predictions \\\n"
            f"      --method judge"
        )

    # Run experiment for each seed and collect results
    reliability_keys = [
        "fisher_predicted_reliability", "fisher_oracle_reliability", "random_reliability",
    ]
    all_runs: list[dict[str, list[float]]] = []
    for seed in args.seeds:
        print(f"\n=== Seed {seed} ===")
        config = ExperimentConfig(
            responses_path=Path(args.responses),
            oracle_items_path=Path(args.oracle_items),
            predictions_csv=predictions_path,
            max_steps=args.max_steps,
            seed=seed,
            prior_sigma=args.prior_sigma,
        )
        all_runs.append(run_experiment(config))

    # Average reliability across seeds
    steps = all_runs[0]["step"]
    avg_results: dict[str, list[float]] = {"step": steps}
    for k in reliability_keys:
        arr = np.array([run[k] for run in all_runs])
        avg_results[k] = np.nanmean(arr, axis=0).tolist()

    # Save config and averaged results
    run_dir = Path(args.output_dir) / "averaged"
    os.makedirs(run_dir, exist_ok=True)

    with open(run_dir / "config.json", "w") as f:
        json.dump({
            "predictions_csv": str(predictions_path),
            "oracle_items_path": args.oracle_items,
            "responses_path": args.responses,
            "max_steps": args.max_steps,
            "seeds": args.seeds,
            "prior_sigma": args.prior_sigma,
        }, f, indent=2)

    results_csv = run_dir / "results.csv"
    with open(results_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step"] + reliability_keys)
        for i in range(len(steps)):
            writer.writerow([steps[i]] + [avg_results[k][i] for k in reliability_keys])
    print(f"\nSaved averaged results: {results_csv}")

    # Plot
    plot_reliability_curves(avg_results, run_dir / "reliability_curves.pdf")

    print(f"All outputs in: {run_dir}")


if __name__ == "__main__":
    main()

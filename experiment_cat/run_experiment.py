"""Run the adaptive task selection experiment and generate plots.

Usage:
    python -m experiment_cat.run_experiment --predictions_csv path/to/predictions.csv

To generate predictions first (train on Verified+TerminalBench+GSO, predict Pro):
    python -m experiment_agent_features.predict_question_difficulty_multi_benchmark \
        --split_by benchmark \
        --train_benchmarks verified,terminalbench,gso \
        --ood_benchmark pro \
        --out_dir output/experiment_cat/ood_predictions \
        --method judge
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .cat_simulation import ExperimentConfig, run_experiment


def _config_hash(config: ExperimentConfig) -> str:
    """Short hash of config for creating unique result directories."""
    key = json.dumps({
        "predictions_csv": str(config.predictions_csv),
        "oracle_items_path": str(config.oracle_items_path),
        "responses_path": str(config.responses_path),
        "max_steps": config.max_steps,
        "seed": config.seed,
        "prior_sigma": config.prior_sigma,
    }, sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()[:8]


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


def main():
    parser = argparse.ArgumentParser(description="Adaptive task selection experiment")
    parser.add_argument(
        "--predictions_csv",
        type=str,
        default="output/experiment_cat/ood_predictions/predictions.csv",
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prior_sigma", type=float, default=3.0)
    parser.add_argument("--output_dir", type=str, default="output/experiment_cat")
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
            f"      --out_dir output/experiment_cat/ood_predictions \\\n"
            f"      --method judge"
        )

    config = ExperimentConfig(
        responses_path=Path(args.responses),
        oracle_items_path=Path(args.oracle_items),
        predictions_csv=predictions_path,
        max_steps=args.max_steps,
        seed=args.seed,
        prior_sigma=args.prior_sigma,
    )

    # Create a unique results subdirectory for this config
    run_dir = Path(args.output_dir) / _config_hash(config)
    os.makedirs(run_dir, exist_ok=True)

    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump({
            "predictions_csv": str(config.predictions_csv),
            "oracle_items_path": str(config.oracle_items_path),
            "responses_path": str(config.responses_path),
            "max_steps": config.max_steps,
            "seed": config.seed,
            "prior_sigma": config.prior_sigma,
        }, f, indent=2)

    results = run_experiment(config)

    # Save results CSV
    results_csv = run_dir / "results.csv"
    with open(results_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "fisher_predicted", "fisher_oracle", "random"])
        for i in range(len(results["step"])):
            writer.writerow([
                results["step"][i],
                results["fisher_predicted"][i],
                results["fisher_oracle"][i],
                results["random"][i],
            ])
    print(f"Saved results: {results_csv}")

    # Plot
    plot_spearman_curves(results, run_dir / "spearman_curves.png")

    print(f"All outputs in: {run_dir}")


if __name__ == "__main__":
    main()

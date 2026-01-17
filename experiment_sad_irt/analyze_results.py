#!/usr/bin/env python3
"""Analyze SAD-IRT experiment results.

Usage:
    python -m experiment_sad_irt.analyze_results chris_output/sad_irt_long/full
    python -m experiment_sad_irt.analyze_results chris_output/sad_irt_long/full chris_output/sad_irt_long/freeze_irt
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch


def load_results(output_dir: Path) -> dict:
    """Load results.json from output directory."""
    results_path = output_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"No results.json found in {output_dir}")

    with open(results_path, "r") as f:
        return json.load(f)


def load_checkpoint(output_dir: Path, name: str = "best") -> Optional[dict]:
    """Load a checkpoint from output directory."""
    checkpoint_path = output_dir / f"checkpoint_{name}.pt"
    if not checkpoint_path.exists():
        # Try to find any checkpoint
        checkpoints = list(output_dir.glob("checkpoint_*.pt"))
        if not checkpoints:
            return None
        checkpoint_path = sorted(checkpoints)[-1]  # Latest

    return torch.load(checkpoint_path, map_location="cpu")


def analyze_single_experiment(output_dir: Path) -> dict:
    """Analyze results from a single experiment."""
    output_dir = Path(output_dir)

    print(f"\n{'='*60}")
    print(f"Analyzing: {output_dir}")
    print(f"{'='*60}")

    # Load results
    try:
        results = load_results(output_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return {}

    # Key metrics
    sad_irt_rho = results.get("frontier_metrics", {}).get("frontier_spearman_rho", float("nan"))
    baseline_rho = results.get("baseline_frontier_metrics", {}).get("baseline_frontier_spearman_rho", float("nan"))
    improvement = results.get("improvement", float("nan"))

    print(f"\n--- Key Metrics ---")
    print(f"Frontier tasks: {results.get('num_frontier_tasks', 'N/A')}")
    print(f"Training samples: {results.get('num_training_samples', 'N/A')}")
    print(f"Pre-frontier agents: {results.get('num_pre_frontier_agents', 'N/A')}")
    print(f"Post-frontier agents: {results.get('num_post_frontier_agents', 'N/A')}")

    print(f"\n--- Spearman ρ (correlation with oracle difficulty) ---")
    print(f"Baseline IRT (no trajectories): {baseline_rho:.4f}")
    print(f"SAD-IRT (with trajectories):    {sad_irt_rho:.4f}")
    print(f"Improvement:                    {improvement:+.4f}")

    # Statistical significance
    sad_irt_p = results.get("frontier_metrics", {}).get("frontier_spearman_p", float("nan"))
    baseline_p = results.get("baseline_frontier_metrics", {}).get("baseline_frontier_spearman_p", float("nan"))
    print(f"\n--- Statistical Significance ---")
    print(f"Baseline p-value: {baseline_p:.4e}")
    print(f"SAD-IRT p-value:  {sad_irt_p:.4e}")

    # Pearson correlation (linear relationship)
    sad_irt_pearson = results.get("frontier_metrics", {}).get("frontier_pearson_r", float("nan"))
    baseline_pearson = results.get("baseline_frontier_metrics", {}).get("baseline_frontier_pearson_r", float("nan"))
    print(f"\n--- Pearson r (linear correlation) ---")
    print(f"Baseline: {baseline_pearson:.4f}")
    print(f"SAD-IRT:  {sad_irt_pearson:.4f}")

    # Load checkpoint for additional info
    checkpoint = load_checkpoint(output_dir, "best")
    if checkpoint:
        print(f"\n--- Best Checkpoint ---")
        print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"Global step: {checkpoint.get('global_step', 'N/A')}")
        if "best_spearman_rho" in checkpoint:
            print(f"Best Spearman ρ during training: {checkpoint['best_spearman_rho']:.4f}")

    # Config summary
    config = results.get("config", {})
    print(f"\n--- Configuration ---")
    print(f"Model: {config.get('model_name', 'N/A')}")
    print(f"LoRA rank: {config.get('lora_r', 'N/A')}")
    print(f"Epochs: {config.get('epochs', 'N/A')}")
    print(f"Batch size: {config.get('batch_size', 'N/A')}")
    print(f"Learning rate (encoder): {config.get('learning_rate_encoder', 'N/A')}")
    print(f"Learning rate (embeddings): {config.get('learning_rate_embeddings', 'N/A')}")
    print(f"ψ normalization: {config.get('psi_normalization', 'N/A')}")
    print(f"Freeze IRT: {config.get('freeze_irt', False)}")

    return results


def compare_experiments(dirs: list) -> None:
    """Compare results across multiple experiments."""
    results_list = []

    for d in dirs:
        try:
            results = load_results(Path(d))
            results["dir"] = str(d)
            results_list.append(results)
        except FileNotFoundError:
            print(f"Warning: No results in {d}")

    if len(results_list) < 2:
        print("Need at least 2 experiments to compare")
        return

    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")

    # Build comparison table
    rows = []
    for r in results_list:
        config = r.get("config", {})
        rows.append({
            "Experiment": Path(r["dir"]).name,
            "Freeze IRT": config.get("freeze_irt", False),
            "ψ norm": config.get("psi_normalization", "N/A"),
            "Baseline ρ": r.get("baseline_frontier_metrics", {}).get("baseline_frontier_spearman_rho", float("nan")),
            "SAD-IRT ρ": r.get("frontier_metrics", {}).get("frontier_spearman_rho", float("nan")),
            "Improvement": r.get("improvement", float("nan")),
        })

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))

    # Summary
    print(f"\n--- Summary ---")
    best_idx = df["SAD-IRT ρ"].idxmax()
    print(f"Best experiment: {df.loc[best_idx, 'Experiment']}")
    print(f"Best SAD-IRT ρ: {df.loc[best_idx, 'SAD-IRT ρ']:.4f}")
    print(f"Best improvement: {df.loc[best_idx, 'Improvement']:+.4f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze SAD-IRT experiment results")
    parser.add_argument("output_dirs", nargs="+", help="Output directories to analyze")
    parser.add_argument("--compare", action="store_true", help="Compare multiple experiments")
    args = parser.parse_args()

    if args.compare or len(args.output_dirs) > 1:
        # First analyze each individually
        for d in args.output_dirs:
            analyze_single_experiment(d)

        # Then compare
        if len(args.output_dirs) > 1:
            compare_experiments(args.output_dirs)
    else:
        analyze_single_experiment(args.output_dirs[0])


if __name__ == "__main__":
    main()

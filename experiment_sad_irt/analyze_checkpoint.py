#!/usr/bin/env python3
"""Analyze SAD-IRT experiment results with clean summary output.

Shows:
1. Spearman ρ comparison (SAD-IRT vs baseline)
2. Parameter change summary (LoRA, IRT, ψ head)
3. Loss curve plot

Usage:
    python -m experiment_sad_irt.analyze_checkpoint chris_output/sad_irt_long/full
    python -m experiment_sad_irt.analyze_checkpoint chris_output/sad_irt_long/full chris_output/sad_irt_long/freeze_irt
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def load_results(output_dir: Path) -> dict:
    """Load results.json from output directory."""
    results_path = output_dir / "results.json"
    if not results_path.exists():
        return {}
    with open(results_path, "r") as f:
        return json.load(f)


def parse_loss_from_logs(output_dir: Path) -> tuple:
    """Parse loss values from training logs.

    Returns:
        Tuple of (steps, losses) arrays
    """
    # Try to find log files
    log_patterns = [
        Path("logs") / "sad_irt_long_*.out",
        output_dir.parent.parent / "logs" / "sad_irt_long_*.out",
    ]

    steps = []
    losses = []

    for pattern in log_patterns:
        if not pattern.parent.exists():
            continue
        log_files = sorted(pattern.parent.glob(pattern.name))
        if not log_files:
            continue

        # Use the most recent log file
        log_file = log_files[-1]

        with open(log_file, "r") as f:
            content = f.read()

        # Pattern: "Step 123/456 | Loss: 0.6789"
        pattern_re = r"Step\s+(\d+)/\d+.*?Loss:\s*([\d.]+)"
        for match in re.finditer(pattern_re, content):
            steps.append(int(match.group(1)))
            losses.append(float(match.group(2)))

        if steps:
            break

    return np.array(steps), np.array(losses)


def compute_param_changes(checkpoint1: dict, checkpoint2: dict) -> dict:
    """Compute parameter changes between two checkpoints."""
    sd1 = checkpoint1.get("model_state_dict", checkpoint1)
    sd2 = checkpoint2.get("model_state_dict", checkpoint2)

    changes = {
        "lora": {"changed": False, "total_diff": 0.0, "num_params": 0},
        "irt": {"changed": False, "total_diff": 0.0, "num_params": 0},  # theta + beta
        "psi_head": {"changed": False, "total_diff": 0.0, "num_params": 0},
    }

    for key in sd1.keys():
        if key not in sd2:
            continue

        p1 = sd1[key].cpu().numpy()
        p2 = sd2[key].cpu().numpy()
        diff_norm = np.linalg.norm(p2 - p1)
        num_params = p1.size

        if "lora" in key.lower():
            category = "lora"
        elif "embedding" in key.lower() or "theta" in key.lower() or "beta" in key.lower():
            category = "irt"
        elif "psi" in key.lower():
            category = "psi_head"
        else:
            continue

        changes[category]["total_diff"] += diff_norm
        changes[category]["num_params"] += num_params
        if diff_norm > 1e-10:
            changes[category]["changed"] = True

    return changes


def analyze_experiment(output_dir: Path) -> dict:
    """Analyze a single experiment directory."""
    output_dir = Path(output_dir)

    # Load results
    results = load_results(output_dir)

    # Find checkpoints
    checkpoints = sorted(output_dir.glob("checkpoint_*.pt"))

    # Load first and last checkpoints for comparison
    param_changes = None
    if len(checkpoints) >= 2:
        first_cp = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
        last_cp = torch.load(checkpoints[-1], map_location="cpu", weights_only=False)
        param_changes = compute_param_changes(first_cp, last_cp)
    elif len(checkpoints) == 1:
        # Only one checkpoint - can't compare
        param_changes = {
            "lora": {"changed": "N/A (single checkpoint)", "total_diff": 0.0, "num_params": 0},
            "irt": {"changed": "N/A (single checkpoint)", "total_diff": 0.0, "num_params": 0},
            "psi_head": {"changed": "N/A (single checkpoint)", "total_diff": 0.0, "num_params": 0},
        }

    # Parse loss curve
    steps, losses = parse_loss_from_logs(output_dir)

    return {
        "dir": output_dir,
        "results": results,
        "param_changes": param_changes,
        "loss_steps": steps,
        "loss_values": losses,
        "num_checkpoints": len(checkpoints),
    }


def print_summary(experiments: list) -> None:
    """Print clean summary of experiments."""
    print("\n" + "=" * 70)
    print("SAD-IRT EXPERIMENT ANALYSIS")
    print("=" * 70)

    for exp in experiments:
        output_dir = exp["dir"]
        results = exp["results"]
        param_changes = exp["param_changes"]

        print(f"\n{'─' * 70}")
        print(f"Experiment: {output_dir.name}")
        print(f"{'─' * 70}")

        # Spearman ρ comparison
        if results:
            sad_irt_rho = results.get("frontier_metrics", {}).get("frontier_spearman_rho", float("nan"))
            baseline_rho = results.get("baseline_frontier_metrics", {}).get("baseline_frontier_spearman_rho", float("nan"))
            improvement = results.get("improvement", float("nan"))

            print(f"\n📊 SPEARMAN ρ (correlation with oracle difficulty)")
            print(f"   Baseline (IRT only):     {baseline_rho:+.4f}")
            print(f"   SAD-IRT (+ trajectories): {sad_irt_rho:+.4f}")
            print(f"   Improvement:              {improvement:+.4f}")

            # Statistical significance
            sad_irt_p = results.get("frontier_metrics", {}).get("frontier_spearman_p", float("nan"))
            if not np.isnan(sad_irt_p):
                print(f"   p-value:                  {sad_irt_p:.4e}")
        else:
            print("\n⚠️  No results.json found")

        # Parameter changes
        if param_changes:
            print(f"\n🔧 PARAMETER CHANGES (first → last checkpoint)")

            # LoRA
            lora = param_changes["lora"]
            if isinstance(lora["changed"], str):
                print(f"   LoRA:     {lora['changed']}")
            elif lora["changed"]:
                print(f"   LoRA:     ✅ CHANGED (Δ = {lora['total_diff']:.6f})")
            else:
                print(f"   LoRA:     ❌ NO CHANGE")

            # IRT (θ/β)
            irt = param_changes["irt"]
            if isinstance(irt["changed"], str):
                print(f"   IRT (θ/β): {irt['changed']}")
            elif irt["changed"]:
                print(f"   IRT (θ/β): ✅ CHANGED (Δ = {irt['total_diff']:.4f})")
            else:
                print(f"   IRT (θ/β): ❌ NO CHANGE")

            # ψ head
            psi = param_changes["psi_head"]
            if isinstance(psi["changed"], str):
                print(f"   ψ head:   {psi['changed']}")
            elif psi["changed"]:
                print(f"   ψ head:   ✅ CHANGED (Δ = {psi['total_diff']:.6f})")
            else:
                print(f"   ψ head:   ❌ NO CHANGE")

        # Config info
        if results:
            config = results.get("config", {})
            print(f"\n⚙️  CONFIG")
            print(f"   Freeze IRT: {config.get('freeze_irt', False)}")
            print(f"   ψ norm:     {config.get('psi_normalization', 'N/A')}")
            print(f"   Epochs:     {config.get('epochs', 'N/A')}")


def plot_loss_curves(experiments: list, output_path: Path) -> None:
    """Plot loss curves for all experiments."""
    fig, ax = plt.subplots(figsize=(10, 6))

    has_data = False
    for exp in experiments:
        steps = exp["loss_steps"]
        losses = exp["loss_values"]

        if len(steps) > 0:
            has_data = True
            label = exp["dir"].name
            ax.plot(steps, losses, label=label, alpha=0.8)

    if has_data:
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("SAD-IRT Training Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"\n📈 Loss curve saved to: {output_path}")
    else:
        print("\n⚠️  No loss data found in logs - skipping plot")


def main():
    parser = argparse.ArgumentParser(description="Analyze SAD-IRT experiment results")
    parser.add_argument("output_dirs", nargs="+", help="Output directories to analyze")
    parser.add_argument("--plot", type=str, default="loss_curve.png", help="Output path for loss curve plot")
    args = parser.parse_args()

    # Analyze all experiments
    experiments = []
    for d in args.output_dirs:
        exp = analyze_experiment(Path(d))
        experiments.append(exp)

    # Print summary
    print_summary(experiments)

    # Plot loss curves
    plot_loss_curves(experiments, Path(args.plot))

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

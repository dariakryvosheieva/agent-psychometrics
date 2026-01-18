#!/usr/bin/env python3
"""Diagnostic script for SAD-IRT training analysis.

Analyzes why training loss decreases but validation Spearman ρ stays flat.

Usage:
    python -m experiment_sad_irt.diagnose_training \
        --checkpoint chris_output/sad_irt_long/full_TIMESTAMP/checkpoint_best.pt \
        --output diagnose_output/

    # Quick analysis without running forward passes (no GPU needed)
    python -m experiment_sad_irt.diagnose_training \
        --checkpoint chris_output/sad_irt_long/full_TIMESTAMP/checkpoint_best.pt \
        --quick
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats


def load_checkpoint(checkpoint_path: str) -> Dict:
    """Load a training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return checkpoint


def load_oracle_beta(oracle_dir: str = "clean_data/swebench_verified_20251120_full/1d") -> Dict[str, float]:
    """Load oracle β values from IRT model."""
    items_path = Path(oracle_dir) / "items.csv"
    if not items_path.exists():
        raise FileNotFoundError(f"Oracle items not found at {items_path}")

    df = pd.read_csv(items_path, index_col=0)
    # Column 'b' is the difficulty parameter in IRT
    return dict(zip(df.index, df["b"]))


def load_training_history(output_dir: str) -> Dict[str, List]:
    """Load training history from JSON file."""
    history_path = Path(output_dir) / "training_history.json"
    if history_path.exists():
        with open(history_path) as f:
            return json.load(f)
    return {}


def analyze_beta(
    checkpoint: Dict,
    oracle_beta: Dict[str, float],
    task_ids: List[str],
) -> Dict:
    """Analyze learned β vs oracle β."""
    # Extract learned β from checkpoint
    beta_weight = checkpoint["model_state_dict"]["beta.weight"]  # (num_tasks, 1)
    learned_beta = beta_weight.squeeze(-1).numpy()  # (num_tasks,)

    # Build task_id -> learned_beta mapping
    learned_beta_dict = {task_id: float(learned_beta[i]) for i, task_id in enumerate(task_ids)}

    # Compare on all tasks that have oracle values
    common_tasks = [t for t in task_ids if t in oracle_beta]

    learned_values = [learned_beta_dict[t] for t in common_tasks]
    oracle_values = [oracle_beta[t] for t in common_tasks]

    # Compute correlations
    spearman_rho, spearman_p = stats.spearmanr(learned_values, oracle_values)
    pearson_r, pearson_p = stats.pearsonr(learned_values, oracle_values)

    # Compute statistics
    learned_std = np.std(learned_values)
    oracle_std = np.std(oracle_values)
    learned_range = np.max(learned_values) - np.min(learned_values)
    oracle_range = np.max(oracle_values) - np.min(oracle_values)

    return {
        "num_tasks": len(common_tasks),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "learned_beta_std": float(learned_std),
        "oracle_beta_std": float(oracle_std),
        "learned_beta_range": float(learned_range),
        "oracle_beta_range": float(oracle_range),
        "std_ratio": float(learned_std / oracle_std) if oracle_std > 0 else float("inf"),
        "learned_values": learned_values,
        "oracle_values": oracle_values,
        "task_ids": common_tasks,
    }


def analyze_theta(checkpoint: Dict, agent_ids: List[str]) -> Dict:
    """Analyze learned θ (agent abilities)."""
    theta_weight = checkpoint["model_state_dict"]["theta.weight"]  # (num_agents, 1)
    learned_theta = theta_weight.squeeze(-1).numpy()  # (num_agents,)

    return {
        "num_agents": len(agent_ids),
        "theta_mean": float(np.mean(learned_theta)),
        "theta_std": float(np.std(learned_theta)),
        "theta_min": float(np.min(learned_theta)),
        "theta_max": float(np.max(learned_theta)),
        "theta_range": float(np.max(learned_theta) - np.min(learned_theta)),
    }


def analyze_psi_head(checkpoint: Dict) -> Dict:
    """Analyze ψ head weights and BatchNorm stats."""
    state_dict = checkpoint["model_state_dict"]

    # ψ head weights
    psi_weight = state_dict.get("psi_head.weight")  # (1, encoder_dim)
    if psi_weight is not None:
        psi_weight_np = psi_weight.numpy()
        psi_weight_norm = float(np.linalg.norm(psi_weight_np))
        psi_weight_mean = float(np.mean(np.abs(psi_weight_np)))
    else:
        psi_weight_norm = None
        psi_weight_mean = None

    # BatchNorm stats
    bn_mean = state_dict.get("psi_bn.running_mean")
    bn_var = state_dict.get("psi_bn.running_var")

    return {
        "psi_weight_norm": psi_weight_norm,
        "psi_weight_mean_abs": psi_weight_mean,
        "bn_running_mean": float(bn_mean.item()) if bn_mean is not None else None,
        "bn_running_var": float(bn_var.item()) if bn_var is not None else None,
        "bn_running_std": float(np.sqrt(bn_var.item())) if bn_var is not None else None,
    }


def analyze_training_dynamics(history: Dict[str, List]) -> Dict:
    """Analyze training dynamics from history."""
    if not history:
        return {"error": "No training history found"}

    losses = history.get("loss", [])
    spearman_rhos = history.get("frontier_spearman_rho", [])
    epochs = history.get("epoch", [])

    analysis = {
        "num_steps": len(losses),
        "num_evals": len([r for r in spearman_rhos if r is not None]),
    }

    if losses:
        analysis["initial_loss"] = losses[0]
        analysis["final_loss"] = losses[-1]
        analysis["min_loss"] = min(losses)
        analysis["loss_reduction"] = losses[0] - losses[-1]
        analysis["loss_reduction_pct"] = (losses[0] - losses[-1]) / losses[0] * 100 if losses[0] > 0 else 0

    valid_rhos = [r for r in spearman_rhos if r is not None]
    if valid_rhos:
        analysis["initial_spearman"] = valid_rhos[0]
        analysis["final_spearman"] = valid_rhos[-1]
        analysis["best_spearman"] = max(valid_rhos)
        analysis["spearman_change"] = valid_rhos[-1] - valid_rhos[0]

    # Check for divergence (loss decreases but spearman flat/decreasing)
    if losses and valid_rhos:
        loss_improved = analysis.get("loss_reduction_pct", 0) > 5  # >5% improvement
        spearman_improved = analysis.get("spearman_change", 0) > 0.01  # >0.01 improvement
        analysis["divergence_detected"] = loss_improved and not spearman_improved

    return analysis


def generate_plots(
    beta_analysis: Dict,
    history: Dict[str, List],
    output_dir: Path,
):
    """Generate diagnostic plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. β scatter plot: learned vs oracle
    if beta_analysis.get("learned_values") and beta_analysis.get("oracle_values"):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(beta_analysis["oracle_values"], beta_analysis["learned_values"], alpha=0.5, s=20)

        # Add diagonal line
        min_val = min(min(beta_analysis["oracle_values"]), min(beta_analysis["learned_values"]))
        max_val = max(max(beta_analysis["oracle_values"]), max(beta_analysis["learned_values"]))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')

        # Add regression line
        z = np.polyfit(beta_analysis["oracle_values"], beta_analysis["learned_values"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(min_val, max_val, 100)
        ax.plot(x_line, p(x_line), 'g-', label=f'Linear fit (r={beta_analysis["pearson_r"]:.3f})')

        ax.set_xlabel("Oracle β (from full IRT)")
        ax.set_ylabel("Learned β (from SAD-IRT)")
        ax.set_title(f"β Comparison (Spearman ρ = {beta_analysis['spearman_rho']:.4f})")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "beta_analysis.png", dpi=150)
        plt.close()
        print(f"Saved beta_analysis.png")

    # 2. Training dynamics: loss vs spearman
    if history:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Loss curve
        losses = history.get("loss", [])
        if losses:
            axes[0].plot(losses, alpha=0.5, linewidth=0.5)
            # Smoothed
            if len(losses) > 20:
                window = min(50, len(losses) // 10)
                smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
                axes[0].plot(range(window-1, len(losses)), smoothed, 'r-', linewidth=2)
            axes[0].set_ylabel("Training Loss")
            axes[0].set_title("Training Loss")
            axes[0].grid(True, alpha=0.3)

        # Spearman curve
        spearman_rhos = history.get("frontier_spearman_rho", [])
        steps = history.get("step", list(range(len(spearman_rhos))))
        if spearman_rhos:
            valid_points = [(s, r) for s, r in zip(steps, spearman_rhos) if r is not None]
            if valid_points:
                s, r = zip(*valid_points)
                axes[1].plot(s, r, 'bo-', markersize=4)
                axes[1].set_ylabel("Spearman ρ (frontier)")
                axes[1].set_xlabel("Step")
                axes[1].set_title("Validation Spearman ρ")
                axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "loss_vs_spearman.png", dpi=150)
        plt.close()
        print(f"Saved loss_vs_spearman.png")

    # 3. β distribution comparison
    if beta_analysis.get("learned_values") and beta_analysis.get("oracle_values"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].hist(beta_analysis["oracle_values"], bins=30, alpha=0.7, label="Oracle β")
        axes[0].set_xlabel("β value")
        axes[0].set_ylabel("Count")
        axes[0].set_title(f"Oracle β distribution (std={beta_analysis['oracle_beta_std']:.3f})")
        axes[0].grid(True, alpha=0.3)

        axes[1].hist(beta_analysis["learned_values"], bins=30, alpha=0.7, label="Learned β", color="orange")
        axes[1].set_xlabel("β value")
        axes[1].set_ylabel("Count")
        axes[1].set_title(f"Learned β distribution (std={beta_analysis['learned_beta_std']:.3f})")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "beta_distributions.png", dpi=150)
        plt.close()
        print(f"Saved beta_distributions.png")


def generate_report(
    checkpoint_path: str,
    beta_analysis: Dict,
    theta_analysis: Dict,
    psi_analysis: Dict,
    dynamics_analysis: Dict,
    output_dir: Path,
):
    """Generate diagnostic report."""
    lines = []
    lines.append("=" * 70)
    lines.append("SAD-IRT TRAINING DIAGNOSTIC REPORT")
    lines.append("=" * 70)
    lines.append(f"\nCheckpoint: {checkpoint_path}")
    lines.append(f"Output: {output_dir}")

    # Training dynamics
    lines.append("\n" + "-" * 70)
    lines.append("TRAINING DYNAMICS")
    lines.append("-" * 70)
    if "error" not in dynamics_analysis:
        lines.append(f"  Steps trained: {dynamics_analysis.get('num_steps', 'N/A')}")
        lines.append(f"  Loss: {dynamics_analysis.get('initial_loss', 0):.4f} -> {dynamics_analysis.get('final_loss', 0):.4f} ({dynamics_analysis.get('loss_reduction_pct', 0):.1f}% reduction)")
        lines.append(f"  Spearman ρ: {dynamics_analysis.get('initial_spearman', 0):.4f} -> {dynamics_analysis.get('final_spearman', 0):.4f} (best: {dynamics_analysis.get('best_spearman', 0):.4f})")
        if dynamics_analysis.get("divergence_detected"):
            lines.append("  ⚠️  DIVERGENCE DETECTED: Loss decreased but Spearman stayed flat")
    else:
        lines.append(f"  {dynamics_analysis['error']}")

    # β analysis
    lines.append("\n" + "-" * 70)
    lines.append("β (DIFFICULTY) ANALYSIS")
    lines.append("-" * 70)
    lines.append(f"  Tasks analyzed: {beta_analysis['num_tasks']}")
    lines.append(f"  Spearman ρ (learned vs oracle): {beta_analysis['spearman_rho']:.4f} (p={beta_analysis['spearman_p']:.2e})")
    lines.append(f"  Pearson r (learned vs oracle): {beta_analysis['pearson_r']:.4f} (p={beta_analysis['pearson_p']:.2e})")
    lines.append(f"  Learned β std: {beta_analysis['learned_beta_std']:.4f} (range: {beta_analysis['learned_beta_range']:.4f})")
    lines.append(f"  Oracle β std: {beta_analysis['oracle_beta_std']:.4f} (range: {beta_analysis['oracle_beta_range']:.4f})")
    lines.append(f"  Std ratio (learned/oracle): {beta_analysis['std_ratio']:.4f}")

    # Interpretation
    if beta_analysis['std_ratio'] < 0.5:
        lines.append("  ⚠️  Learned β has much lower variance than oracle - β may not be learning!")
    elif beta_analysis['std_ratio'] > 2.0:
        lines.append("  ⚠️  Learned β has much higher variance than oracle - possible overfitting")

    # θ analysis
    lines.append("\n" + "-" * 70)
    lines.append("θ (ABILITY) ANALYSIS")
    lines.append("-" * 70)
    lines.append(f"  Agents: {theta_analysis['num_agents']}")
    lines.append(f"  θ mean: {theta_analysis['theta_mean']:.4f}")
    lines.append(f"  θ std: {theta_analysis['theta_std']:.4f}")
    lines.append(f"  θ range: [{theta_analysis['theta_min']:.4f}, {theta_analysis['theta_max']:.4f}]")

    # ψ analysis
    lines.append("\n" + "-" * 70)
    lines.append("ψ (TRAJECTORY) ANALYSIS")
    lines.append("-" * 70)
    if psi_analysis.get("psi_weight_norm") is not None:
        lines.append(f"  ψ head weight norm: {psi_analysis['psi_weight_norm']:.4f}")
        lines.append(f"  ψ head weight mean |w|: {psi_analysis['psi_weight_mean_abs']:.6f}")
    if psi_analysis.get("bn_running_mean") is not None:
        lines.append(f"  BatchNorm running mean: {psi_analysis['bn_running_mean']:.6f}")
        lines.append(f"  BatchNorm running std: {psi_analysis['bn_running_std']:.4f}")

    # Diagnosis
    lines.append("\n" + "-" * 70)
    lines.append("DIAGNOSIS")
    lines.append("-" * 70)

    issues = []
    recommendations = []

    # Check for key issues
    if dynamics_analysis.get("divergence_detected"):
        issues.append("Loss/Spearman divergence: model is fitting training data but not learning useful β")
        recommendations.append("Try freezing encoder (--freeze_encoder) to force β learning")
        recommendations.append("Reduce ψ head capacity or add regularization")

    if beta_analysis['std_ratio'] < 0.5:
        issues.append("β variance collapse: learned β has much less variance than oracle")
        recommendations.append("Increase learning_rate_embeddings to encourage β movement")
        recommendations.append("Try larger batch size for more stable β gradients")

    if beta_analysis['spearman_rho'] < 0.3:
        issues.append("Low β-oracle correlation: learned β doesn't match oracle ranking")
        recommendations.append("Consider longer training or different architecture")

    if psi_analysis.get("bn_running_std", 0) > 2.0:
        issues.append("High ψ variance: trajectory encoder producing large ψ values")
        recommendations.append("ψ may be dominating predictions over β")

    if not issues:
        issues.append("No major issues detected")
        recommendations.append("Consider running ψ=0 ablation to check ψ contribution")

    lines.append("  Issues found:")
    for issue in issues:
        lines.append(f"    - {issue}")

    lines.append("\n  Recommendations:")
    for rec in recommendations:
        lines.append(f"    - {rec}")

    lines.append("\n" + "=" * 70)

    report = "\n".join(lines)

    # Print to console
    print(report)

    # Save to file
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "diagnosis_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nSaved report to {report_path}")

    return report


def get_task_ids_from_checkpoint_dir(checkpoint_path: str) -> List[str]:
    """Try to get task_ids from the checkpoint directory."""
    checkpoint_dir = Path(checkpoint_path).parent

    # Check for training_history.json which might have task info
    history_path = checkpoint_dir / "training_history.json"
    if history_path.exists():
        # History doesn't have task_ids, but we can check for dataset info
        pass

    # Fallback: load from default response matrix
    response_matrix_path = "clean_data/swebench_verified/swebench_verified_20251120_full.jsonl"
    if Path(response_matrix_path).exists():
        task_ids = set()
        with open(response_matrix_path) as f:
            for line in f:
                data = json.loads(line)
                task_ids.update(data.get("responses", {}).keys())
        return sorted(list(task_ids))

    raise FileNotFoundError("Could not determine task_ids - no response matrix found")


def get_agent_ids_from_checkpoint_dir(checkpoint_path: str) -> List[str]:
    """Try to get agent_ids from the checkpoint directory."""
    response_matrix_path = "clean_data/swebench_verified/swebench_verified_20251120_full.jsonl"
    if Path(response_matrix_path).exists():
        agent_ids = []
        with open(response_matrix_path) as f:
            for line in f:
                data = json.loads(line)
                agent_ids.append(data["agent"])
        return agent_ids

    raise FileNotFoundError("Could not determine agent_ids - no response matrix found")


def main():
    parser = argparse.ArgumentParser(description="Diagnose SAD-IRT training")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file")
    parser.add_argument("--output", default=None, help="Output directory for plots/report")
    parser.add_argument("--oracle_dir", default="clean_data/swebench_verified_20251120_full/1d",
                        help="Path to oracle IRT directory")
    parser.add_argument("--quick", action="store_true",
                        help="Quick analysis without forward passes (no GPU needed)")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = checkpoint_path.parent / "diagnosis"

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = load_checkpoint(str(checkpoint_path))

    print(f"Loading oracle β from {args.oracle_dir}...")
    oracle_beta = load_oracle_beta(args.oracle_dir)

    print("Loading task/agent IDs...")
    task_ids = get_task_ids_from_checkpoint_dir(str(checkpoint_path))
    agent_ids = get_agent_ids_from_checkpoint_dir(str(checkpoint_path))

    print(f"Found {len(task_ids)} tasks, {len(agent_ids)} agents")

    # Load training history
    history = load_training_history(str(checkpoint_path.parent))

    # Run analyses
    print("\nAnalyzing β...")
    beta_analysis = analyze_beta(checkpoint, oracle_beta, task_ids)

    print("Analyzing θ...")
    theta_analysis = analyze_theta(checkpoint, agent_ids)

    print("Analyzing ψ head...")
    psi_analysis = analyze_psi_head(checkpoint)

    print("Analyzing training dynamics...")
    dynamics_analysis = analyze_training_dynamics(history)

    # Generate report
    generate_report(
        str(checkpoint_path),
        beta_analysis,
        theta_analysis,
        psi_analysis,
        dynamics_analysis,
        output_dir,
    )

    # Generate plots
    if not args.quick:
        print("\nGenerating plots...")
        generate_plots(beta_analysis, history, output_dir)


if __name__ == "__main__":
    main()

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


def analyze_logits(
    checkpoint: Dict,
    checkpoint_path: str,
    trajectory_dir: str,
    response_matrix_path: str,
    num_samples: int = 500,
    model_name: str = "Qwen/Qwen3-0.6B",
) -> Dict:
    """Analyze logits and their components (θ, β, ψ) for overfitting detection.

    This runs forward passes on training samples to extract per-sample logits
    and diagnose if ψ is causing overfitting through extreme predictions.

    Args:
        checkpoint: Loaded checkpoint dict
        checkpoint_path: Path to checkpoint file
        trajectory_dir: Path to trajectory summaries directory
        response_matrix_path: Path to response matrix JSONL
        num_samples: Number of samples to analyze
        model_name: HuggingFace model name for encoder

    Returns:
        Dict with logit analysis statistics
    """
    import torch
    from torch.utils.data import DataLoader, Subset
    from transformers import AutoTokenizer

    from .dataset import TrajectoryIRTDataset
    from .model import SADIRT

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create dataset
    print(f"Loading dataset from {response_matrix_path}...")
    dataset = TrajectoryIRTDataset(
        response_matrix_path=response_matrix_path,
        trajectory_dir=trajectory_dir,
        tokenizer=tokenizer,
        max_length=1024,
        use_summaries=True,
    )

    # Get num_agents/num_tasks from checkpoint weights (model may have been trained on subset)
    num_agents_ckpt = checkpoint["model_state_dict"]["theta.weight"].shape[0]
    num_tasks_ckpt = checkpoint["model_state_dict"]["beta.weight"].shape[0]

    # Filter dataset samples to only include those with valid agent/task indices
    # The checkpoint may have been trained on a subset of agents/tasks
    valid_indices = []
    for i, sample in enumerate(dataset.samples):
        agent_idx, task_idx, _ = sample
        if agent_idx < num_agents_ckpt and task_idx < num_tasks_ckpt:
            valid_indices.append(i)

    print(f"  Checkpoint has {num_agents_ckpt} agents, {num_tasks_ckpt} tasks")
    print(f"  Dataset has {len(dataset.samples)} samples, {len(valid_indices)} have valid indices")

    if len(valid_indices) == 0:
        raise ValueError("No valid samples found - checkpoint indices don't match dataset")

    # Subsample from valid indices
    rng = np.random.RandomState(42)
    if len(valid_indices) > num_samples:
        selected_indices = rng.choice(valid_indices, size=num_samples, replace=False)
    else:
        selected_indices = valid_indices
    dataset = Subset(dataset, selected_indices)
    print(f"  Selected {len(dataset)} samples for analysis")

    # Create model
    print("Creating model...")
    config = checkpoint.get("config", {})
    # Handle None values in config (use defaults)
    psi_norm = config.get("psi_normalization")
    if psi_norm is None:
        psi_norm = "batchnorm"
    freeze_enc = config.get("freeze_encoder")
    if freeze_enc is None:
        freeze_enc = False
    # Get LoRA config from checkpoint
    lora_r = config.get("lora_r", 16)
    lora_alpha = config.get("lora_alpha", 32)
    lora_dropout = config.get("lora_dropout", 0.1)
    print(f"  psi_normalization: {psi_norm}, freeze_encoder: {freeze_enc}, lora_r: {lora_r}")
    model = SADIRT(
        num_agents=num_agents_ckpt,
        num_tasks=num_tasks_ckpt,
        model_name=model_name,
        psi_normalization=psi_norm,
        freeze_encoder=freeze_enc,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )

    # Load weights
    print("Loading checkpoint weights...")
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model = model.to(device)
    model.eval()

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    # Collect components
    all_logits = []
    all_theta = []
    all_beta = []
    all_psi = []
    all_responses = []

    print(f"Running forward passes on {len(dataset)} samples...")
    with torch.no_grad():
        for batch in dataloader:
            agent_idx = batch["agent_idx"].to(device)
            task_idx = batch["task_idx"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            responses = batch["response"]

            logits, theta, beta, psi = model.forward_with_components(
                agent_idx, task_idx, input_ids, attention_mask
            )

            all_logits.append(logits.cpu().numpy())
            all_theta.append(theta.cpu().numpy())
            all_beta.append(beta.cpu().numpy())
            all_psi.append(psi.cpu().numpy())
            all_responses.append(responses.numpy())

    # Concatenate
    logits = np.concatenate(all_logits)
    theta = np.concatenate(all_theta)
    beta = np.concatenate(all_beta)
    psi = np.concatenate(all_psi)
    responses = np.concatenate(all_responses)

    # Compute probabilities
    probs = 1 / (1 + np.exp(-logits))

    # Compute IRT-only logits (without ψ)
    irt_logits = theta - beta

    # Compute statistics
    def compute_stats(arr, name):
        return {
            f"{name}_mean": float(np.mean(arr)),
            f"{name}_std": float(np.std(arr)),
            f"{name}_min": float(np.min(arr)),
            f"{name}_p5": float(np.percentile(arr, 5)),
            f"{name}_p50": float(np.percentile(arr, 50)),
            f"{name}_p95": float(np.percentile(arr, 95)),
            f"{name}_max": float(np.max(arr)),
        }

    analysis = {
        "num_samples": len(logits),
        **compute_stats(theta, "theta"),
        **compute_stats(beta, "beta"),
        **compute_stats(psi, "psi"),
        **compute_stats(logits, "logit"),
        **compute_stats(probs, "prob"),
        **compute_stats(irt_logits, "irt_logit"),
    }

    # Overfitting indicators
    psi_var = np.var(psi)
    irt_var = np.var(irt_logits)
    total_var = np.var(logits)

    analysis["psi_variance"] = float(psi_var)
    analysis["irt_variance"] = float(irt_var)
    analysis["logit_variance"] = float(total_var)

    # ψ contribution to variance (approximate, ignoring covariance)
    if total_var > 0:
        analysis["psi_variance_contribution"] = float(psi_var / total_var)
        analysis["irt_variance_contribution"] = float(irt_var / total_var)
    else:
        analysis["psi_variance_contribution"] = 0.0
        analysis["irt_variance_contribution"] = 0.0

    # Count extreme predictions
    extreme_threshold = 5.0  # |logit| > 5 means prob > 0.993 or < 0.007
    n_extreme = np.sum(np.abs(logits) > extreme_threshold)
    analysis["n_extreme_logits"] = int(n_extreme)
    analysis["pct_extreme_logits"] = float(n_extreme / len(logits) * 100)

    # Store raw data for plotting
    analysis["_raw_data"] = {
        "logits": logits.tolist(),
        "theta": theta.tolist(),
        "beta": beta.tolist(),
        "psi": psi.tolist(),
        "probs": probs.tolist(),
        "responses": responses.tolist(),
        "irt_logits": irt_logits.tolist(),
    }

    return analysis


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


def generate_logit_plots(
    logit_analysis: Dict,
    output_dir: Path,
):
    """Generate logit analysis plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping logit plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    raw_data = logit_analysis.get("_raw_data", {})
    if not raw_data:
        print("No raw data available for logit plots")
        return

    logits = np.array(raw_data["logits"])
    psi = np.array(raw_data["psi"])
    irt_logits = np.array(raw_data["irt_logits"])
    probs = np.array(raw_data["probs"])
    responses = np.array(raw_data["responses"])

    # 1. Logit distribution histogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(logits, bins=50, alpha=0.7, edgecolor='black')
    axes[0].axvline(x=-5, color='r', linestyle='--', label='|logit|=5 threshold')
    axes[0].axvline(x=5, color='r', linestyle='--')
    axes[0].set_xlabel("Logit value")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Logit Distribution (N={len(logits)})\nMean={np.mean(logits):.2f}, Std={np.std(logits):.2f}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(psi, bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[1].set_xlabel("ψ value")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"ψ Distribution\nMean={np.mean(psi):.2f}, Std={np.std(psi):.2f}")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "logit_distributions.png", dpi=150)
    plt.close()
    print("Saved logit_distributions.png")

    # 2. ψ vs IRT logit scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Color by response
    colors = ['red' if r == 0 else 'green' for r in responses]
    ax.scatter(irt_logits, psi, c=colors, alpha=0.5, s=20)

    # Add reference line (perfect compensation would be ψ = -IRT logit)
    xlim = ax.get_xlim()
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    # Compute correlation
    corr = np.corrcoef(irt_logits, psi)[0, 1]

    ax.set_xlabel("IRT logit (θ - β)")
    ax.set_ylabel("ψ (trajectory term)")
    ax.set_title(f"ψ vs IRT Logit\nCorrelation: {corr:.3f}\nRed=failure, Green=success")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "psi_vs_irt.png", dpi=150)
    plt.close()
    print("Saved psi_vs_irt.png")

    # 3. Probability vs response (calibration-like plot)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Split by response
    probs_success = probs[responses == 1]
    probs_failure = probs[responses == 0]

    ax.hist(probs_failure, bins=30, alpha=0.6, label=f'Failures (N={len(probs_failure)})', color='red')
    ax.hist(probs_success, bins=30, alpha=0.6, label=f'Successes (N={len(probs_success)})', color='green')

    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Count")
    ax.set_title("Predicted Probability by Actual Outcome")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "prob_calibration.png", dpi=150)
    plt.close()
    print("Saved prob_calibration.png")


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
    logit_analysis: Optional[Dict] = None,
    checkpoint: Optional[Dict] = None,
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

    # Logit analysis (if available)
    if logit_analysis is not None:
        lines.append("\n" + "-" * 70)
        lines.append(f"LOGIT ANALYSIS (N={logit_analysis['num_samples']} training samples)")
        lines.append("-" * 70)

        # Table header
        lines.append(f"  {'Component':<10} {'Mean':>8} {'Std':>8} {'Min':>8} {'P5':>8} {'P50':>8} {'P95':>8} {'Max':>8}")
        lines.append("  " + "-" * 74)

        # Table rows
        for name in ["theta", "beta", "psi", "logit", "prob"]:
            row = f"  {name:<10}"
            row += f" {logit_analysis[f'{name}_mean']:>8.2f}"
            row += f" {logit_analysis[f'{name}_std']:>8.2f}"
            row += f" {logit_analysis[f'{name}_min']:>8.2f}"
            row += f" {logit_analysis[f'{name}_p5']:>8.2f}"
            row += f" {logit_analysis[f'{name}_p50']:>8.2f}"
            row += f" {logit_analysis[f'{name}_p95']:>8.2f}"
            row += f" {logit_analysis[f'{name}_max']:>8.2f}"

            # Add warning flag for extreme values
            if name == "psi" and logit_analysis["psi_std"] > 2.0:
                row += " ⚠️"
            if name == "logit" and logit_analysis["pct_extreme_logits"] > 10:
                row += " ⚠️"

            lines.append(row)

        lines.append("")
        lines.append("  Overfitting indicators:")

        # Check for high ψ variance
        psi_std = logit_analysis["psi_std"]
        beta_std = logit_analysis["beta_std"]
        if beta_std > 0:
            psi_beta_ratio = psi_std / beta_std
            if psi_beta_ratio > 2.0:
                lines.append(f"    ⚠️  High ψ variance ({psi_std:.2f}) vs β variance ({beta_std:.2f}) - ratio: {psi_beta_ratio:.1f}x")

        # Check for extreme logits
        pct_extreme = logit_analysis["pct_extreme_logits"]
        if pct_extreme > 5:
            lines.append(f"    ⚠️  {pct_extreme:.1f}% of samples have |logit| > 5 (extreme predictions)")

        # Check for ψ dominance
        psi_contrib = logit_analysis.get("psi_variance_contribution", 0)
        irt_contrib = logit_analysis.get("irt_variance_contribution", 0)
        if psi_contrib > 0.5:
            lines.append(f"    ⚠️  ψ contributes {psi_contrib*100:.0f}% of logit variance vs {irt_contrib*100:.0f}% from (θ - β)")

        if psi_beta_ratio <= 2.0 and pct_extreme <= 5 and psi_contrib <= 0.5:
            lines.append("    ✓  No major overfitting indicators detected in logit analysis")

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

    # Only flag high raw ψ variance if NOT using batchnorm (batchnorm normalizes to unit variance)
    psi_norm_mode = checkpoint.get("config", {}).get("psi_normalization", "batchnorm")
    if psi_norm_mode != "batchnorm" and (psi_analysis.get("bn_running_std") or 0) > 2.0:
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
                agent_ids.append(data["subject_id"])
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
    # Logit analysis arguments
    parser.add_argument("--analyze_logits", action="store_true",
                        help="Run logit analysis (requires GPU and trajectory data)")
    parser.add_argument("--num_samples", type=int, default=500,
                        help="Number of samples for logit analysis (default: 500)")
    parser.add_argument("--trajectory_dir", default="chris_output/trajectory_summaries_api",
                        help="Path to trajectory summaries directory")
    parser.add_argument("--response_matrix", default="clean_data/swebench_verified/swebench_verified_20251120_full.jsonl",
                        help="Path to response matrix JSONL")
    parser.add_argument("--model_name", default="Qwen/Qwen3-0.6B",
                        help="HuggingFace model name for encoder")
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

    # Logit analysis (optional, requires GPU)
    logit_analysis = None
    if args.analyze_logits:
        print(f"\nAnalyzing logits on {args.num_samples} samples...")
        try:
            logit_analysis = analyze_logits(
                checkpoint=checkpoint,
                checkpoint_path=str(checkpoint_path),
                trajectory_dir=args.trajectory_dir,
                response_matrix_path=args.response_matrix,
                num_samples=args.num_samples,
                model_name=args.model_name,
            )
        except Exception as e:
            print(f"Error during logit analysis: {e}")
            import traceback
            traceback.print_exc()

    # Generate report
    generate_report(
        str(checkpoint_path),
        beta_analysis,
        theta_analysis,
        psi_analysis,
        dynamics_analysis,
        output_dir,
        logit_analysis=logit_analysis,
        checkpoint=checkpoint,
    )

    # Generate plots
    if not args.quick:
        print("\nGenerating plots...")
        generate_plots(beta_analysis, history, output_dir)

        # Generate logit plots if analysis was run
        if logit_analysis is not None:
            print("\nGenerating logit plots...")
            generate_logit_plots(logit_analysis, output_dir)


if __name__ == "__main__":
    main()

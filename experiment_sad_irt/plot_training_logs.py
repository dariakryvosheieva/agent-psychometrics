#!/usr/bin/env python3
"""Parse training logs and plot loss curves.

Usage:
    python -m experiment_sad_irt.plot_training_logs logs/sad_irt_long_8175794.out
    python -m experiment_sad_irt.plot_training_logs logs/sad_irt_long_8175794.out --output loss_curve.png
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_log_file(log_path: str) -> dict:
    """Parse training log file to extract metrics.

    Returns dict with:
        - steps: list of step numbers
        - losses: list of loss values (from progress bar)
        - grad_norms: dict of gradient norm lists by component
        - lora_grads: list of top LoRA gradient values
    """
    steps = []
    losses = []
    lrs = []

    # Gradient norms by component
    grad_norms = {
        "total": [],
        "embedding": [],
        "encoder": [],
        "head": [],
    }
    grad_steps = []

    # Individual param gradients
    psi_head_grads = []
    theta_grads = []
    beta_grads = []
    lora_top_grads = []

    # Patterns
    # Explicit log: Step 10: loss=0.123456, lr=1.00e-04
    step_loss_pattern = re.compile(r'Step (\d+): loss=([\d.]+), lr=([\d.e+-]+)')

    # Progress bar: loss=0.255, lr=6.94e-05 (may be truncated in logs)
    # Handles both full format and truncated like "lr=5.91e-" or "lr=6.02"
    pbar_pattern = re.compile(r'loss=([\d.]+),\s*lr=([\d.e+-]+)')

    # Step gradients: Step 10 gradients (pre-clip): total=2.246549, embedding=0.062246, encoder=1.121982, head=1.945319
    grad_pattern = re.compile(
        r'Step (\d+) gradients \(pre-clip\): total=([\d.]+), embedding=([\d.]+), encoder=([\d.]+), head=([\d.]+)'
    )

    # Individual gradients: psi_head.weight: grad_norm=1.94531894
    psi_grad_pattern = re.compile(r'psi_head\.weight: grad_norm=([\d.]+)')
    theta_grad_pattern = re.compile(r'theta\.weight: grad_norm=([\d.]+)')
    beta_grad_pattern = re.compile(r'beta\.weight: grad_norm=([\d.]+)')

    # Top LoRA grads
    lora_pattern = re.compile(r"Top 3 LoRA grads: \[\('([^']+)', ([\d.]+)\)")

    # Epoch loss pattern from tqdm: Epoch 1: 100%|...| 944/944 [13:14<00:00, loss=0.123, lr=1.00e-06]
    epoch_pattern = re.compile(r'Epoch (\d+).*loss=([\d.]+)')

    epoch_losses = []

    with open(log_path, 'r') as f:
        for line in f:
            # Explicit step loss log (preferred)
            step_loss_match = step_loss_pattern.search(line)
            if step_loss_match:
                step = int(step_loss_match.group(1))
                loss = float(step_loss_match.group(2))
                lr = float(step_loss_match.group(3))
                steps.append(step)
                losses.append(loss)
                lrs.append(lr)
                continue  # Don't double-count

            # Progress bar loss (fallback)
            pbar_match = pbar_pattern.search(line)
            if pbar_match:
                loss = float(pbar_match.group(1))
                lr = float(pbar_match.group(2))
                losses.append(loss)
                lrs.append(lr)

            # Gradient norms
            grad_match = grad_pattern.search(line)
            if grad_match:
                step = int(grad_match.group(1))
                grad_steps.append(step)
                grad_norms["total"].append(float(grad_match.group(2)))
                grad_norms["embedding"].append(float(grad_match.group(3)))
                grad_norms["encoder"].append(float(grad_match.group(4)))
                grad_norms["head"].append(float(grad_match.group(5)))

            # Individual param gradients
            psi_match = psi_grad_pattern.search(line)
            if psi_match:
                psi_head_grads.append(float(psi_match.group(1)))

            theta_match = theta_grad_pattern.search(line)
            if theta_match:
                theta_grads.append(float(theta_match.group(1)))

            beta_match = beta_grad_pattern.search(line)
            if beta_match:
                beta_grads.append(float(beta_match.group(1)))

            # Top LoRA grad
            lora_match = lora_pattern.search(line)
            if lora_match:
                lora_top_grads.append(float(lora_match.group(2)))

            # Epoch losses
            epoch_match = epoch_pattern.search(line)
            if epoch_match and '100%' in line:
                epoch = int(epoch_match.group(1))
                loss = float(epoch_match.group(2))
                epoch_losses.append((epoch, loss))

    return {
        "losses": losses,
        "lrs": lrs,
        "grad_steps": grad_steps,
        "grad_norms": grad_norms,
        "psi_head_grads": psi_head_grads,
        "theta_grads": theta_grads,
        "beta_grads": beta_grads,
        "lora_top_grads": lora_top_grads,
        "epoch_losses": epoch_losses,
    }


def plot_training_curves(data: dict, output_path: str = None, title: str = None):
    """Plot training curves from parsed log data."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Title
    if title:
        fig.suptitle(title, fontsize=14)

    # 1. Loss curve
    ax = axes[0, 0]
    if data["losses"]:
        ax.plot(data["losses"], alpha=0.7, linewidth=0.5)
        # Add smoothed version
        if len(data["losses"]) > 20:
            window = min(50, len(data["losses"]) // 10)
            smoothed = np.convolve(data["losses"], np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(data["losses"])), smoothed, 'r-', linewidth=2, label=f'Smoothed (window={window})')
            ax.legend()
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.3)

    # 2. Learning rate
    ax = axes[0, 1]
    if data["lrs"]:
        ax.plot(data["lrs"])
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # 3. Gradient norms by component
    ax = axes[1, 0]
    if data["grad_steps"]:
        steps = data["grad_steps"]
        ax.plot(steps, data["grad_norms"]["encoder"], label="Encoder (LoRA)", alpha=0.8)
        ax.plot(steps, data["grad_norms"]["head"], label="Head (psi_head)", alpha=0.8)
        ax.plot(steps, data["grad_norms"]["embedding"], label="Embedding (θ/β)", alpha=0.8)
        ax.legend()
    ax.set_xlabel("Step")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Gradient Norms by Component")
    ax.grid(True, alpha=0.3)

    # 4. Top LoRA gradient
    ax = axes[1, 1]
    if data["lora_top_grads"]:
        ax.plot(data["grad_steps"][:len(data["lora_top_grads"])], data["lora_top_grads"], alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Top LoRA Gradient")
    ax.set_title("Top LoRA Layer Gradient")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()

    plt.close()


def print_summary(data: dict):
    """Print summary statistics from parsed data."""
    print("\n" + "="*60)
    print("TRAINING LOG SUMMARY")
    print("="*60)

    if data["losses"]:
        print(f"\nLoss:")
        print(f"  Initial: {data['losses'][0]:.4f}")
        print(f"  Final:   {data['losses'][-1]:.4f}")
        print(f"  Min:     {min(data['losses']):.4f}")
        print(f"  Steps:   {len(data['losses'])}")

    if data["epoch_losses"]:
        print(f"\nEpoch losses:")
        for epoch, loss in data["epoch_losses"]:
            print(f"  Epoch {epoch}: {loss:.4f}")

    if data["grad_norms"]["encoder"]:
        print(f"\nGradient norms (final):")
        print(f"  Encoder (LoRA): {data['grad_norms']['encoder'][-1]:.6f}")
        print(f"  Head (psi):     {data['grad_norms']['head'][-1]:.6f}")
        print(f"  Embedding:      {data['grad_norms']['embedding'][-1]:.6f}")

    if data["lora_top_grads"]:
        print(f"\nTop LoRA gradient:")
        print(f"  Initial: {data['lora_top_grads'][0]:.6f}")
        print(f"  Final:   {data['lora_top_grads'][-1]:.6f}")


def merge_data(data1: dict, data2: dict) -> dict:
    """Merge two parsed data dicts."""
    merged = {}
    for key in data1:
        if isinstance(data1[key], list):
            merged[key] = data1[key] + data2.get(key, [])
        elif isinstance(data1[key], dict):
            merged[key] = {k: data1[key].get(k, []) + data2.get(key, {}).get(k, [])
                          for k in set(data1[key]) | set(data2.get(key, {}))}
        else:
            merged[key] = data1[key]
    return merged


def main():
    parser = argparse.ArgumentParser(description="Parse and plot SAD-IRT training logs")
    parser.add_argument("log_file", help="Path to log file (e.g., logs/sad_irt_long_8175794.out)")
    parser.add_argument("--output", "-o", help="Output path for plot (default: show interactively)")
    parser.add_argument("--title", "-t", help="Plot title")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting, just print summary")
    args = parser.parse_args()

    log_path = Path(args.log_file)

    # Parse main log file
    print(f"Parsing {log_path}...")
    data = parse_log_file(str(log_path))

    # Also try to parse .err file if .out file was given (tqdm goes to stderr)
    if log_path.suffix == '.out':
        err_path = log_path.with_suffix('.err')
        if err_path.exists():
            print(f"Also parsing {err_path} (tqdm output)...")
            err_data = parse_log_file(str(err_path))
            data = merge_data(data, err_data)

    # Print summary
    print_summary(data)

    # Plot
    if not args.no_plot:
        title = args.title or log_path.stem
        output = args.output or str(log_path.with_suffix('.png'))
        plot_training_curves(data, output_path=output, title=title)


if __name__ == "__main__":
    main()

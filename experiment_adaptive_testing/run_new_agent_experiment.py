"""Run the New-Agent adaptive testing experiment and plot the error curves.

Usage:
    python -m experiment_adaptive_testing.run_new_agent_experiment \
        --dataset swebench_verified \
        --max_tasks 100

The experiment uses K-fold cross-validation over (LLM, scaffold) pairs to
simulate the appearance of a "new agent" on an established benchmark, then
adaptively administers tasks to estimate each held-out agent's ability.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from experiment_adaptive_testing.new_agent_simulation import (
    NewAgentExperimentConfig,
    run_new_agent_experiment,
    save_results,
)
from experiment_new_agents.config import DATASET_DEFAULTS


def plot_error_curves(results: dict, output_path: Path) -> None:
    """Plot per-K mean |theta_hat - theta_true| with percentile bands."""
    steps = results["step"]
    fig, ax = plt.subplots(figsize=(8, 5))
    methods = [
        ("fisher_informed_mae", "tab:blue", "-", "Fisher + IRT-Agent prior"),
        ("fisher_weak_mae", "tab:orange", "-", "Fisher + weak prior"),
        ("random_mae", "gray", "--", "Random + weak prior"),
    ]
    for key, color, linestyle, label in methods:
        mean = results[key]
        lo = results[key + "_lo"]
        hi = results[key + "_hi"]
        ax.fill_between(steps, lo, hi, color=color, alpha=0.2, linewidth=0)
        ax.plot(steps, mean, color=color, linewidth=2, linestyle=linestyle, label=label)

    ax.set_xlabel("Number of Tasks Administered", fontsize=16)
    ax.set_ylabel(r"$|\hat{\theta} - \theta_{\mathrm{true}}|$", fontsize=16)
    ax.legend(loc="upper right", fontsize=12)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="New-Agent adaptive testing experiment")
    parser.add_argument(
        "--dataset", type=str, default="swebench_verified",
        choices=list(DATASET_DEFAULTS.keys()),
    )
    parser.add_argument("--k_folds", type=int, default=5)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--max_tasks", type=int, default=100)
    parser.add_argument("--weak_prior_sigma", type=float, default=3.0)
    parser.add_argument("--n_random_subsets", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=0)
    parser.add_argument("--irt_epochs", type=int, default=2000)
    parser.add_argument("--irt_device", type=str, default="cpu")
    parser.add_argument("--irt_lr", type=float, default=0.01)
    parser.add_argument("--irt_model", type=str, default="1d_1pl", choices=["1d_1pl", "2d_1pl"])
    parser.add_argument("--theta_combine", type=str, default="sum")
    parser.add_argument(
        "--output_dir", type=str,
        default="output/experiment_adaptive_testing/new_agent",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    new_agents_cfg = DATASET_DEFAULTS[args.dataset]
    responses_path = Path(new_agents_cfg["responses_path"])
    new_agents_dir = Path(new_agents_cfg["output_dir"])

    config = NewAgentExperimentConfig(
        dataset=args.dataset,
        responses_path=responses_path,
        output_dir=Path(args.output_dir),
        irt_cache_dir=new_agents_dir / "irt_splits",
        oracle_cache_dir=new_agents_dir / "irt_oracle",
        k_folds=args.k_folds,
        split_seed=args.split_seed,
        irt_epochs=args.irt_epochs,
        irt_device=args.irt_device,
        irt_lr=args.irt_lr,
        irt_model=args.irt_model,
        theta_combine=args.theta_combine,
        max_tasks=args.max_tasks,
        weak_prior_sigma=args.weak_prior_sigma,
        n_random_subsets=args.n_random_subsets,
        seed=args.seed,
        n_bootstrap=args.n_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
    )

    output_dir = Path(args.output_dir)
    results = run_new_agent_experiment(config, root)
    save_results(results, output_dir)
    plot_error_curves(results, output_dir / "error_curves.pdf")
    print(f"\nAll outputs in: {output_dir}")


if __name__ == "__main__":
    main()

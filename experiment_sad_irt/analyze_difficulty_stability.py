"""Analyze stability of relative task difficulties when adding post-frontier agents.

This script measures whether pairwise relative task difficulties remain stable when
high-performing (post-frontier) agents are added to IRT training.

Key question: "If problem A was one difficulty point above problem B using pre-frontier
IRT, does this relationship persist in full IRT (with post-frontier agents)?"

Usage:
    python -m experiment_sad_irt.analyze_difficulty_stability
    python -m experiment_sad_irt.analyze_difficulty_stability --min_pass_rate 0.15
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyro
import torch
from scipy import stats

from py_irt.config import IrtConfig
from py_irt.dataset import Dataset
from py_irt.models import OneParamLog
from py_irt.training import IrtModelTrainer

from .data_splits import (
    compute_pass_rates,
    get_all_agents_from_responses,
    split_agents_by_cutoff,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def identify_nontrivial_tasks(
    responses_path: Path,
    pre_frontier_agents: List[str],
    post_frontier_agents: List[str],
    min_pass_rate: float = 0.10,
    max_pass_rate: float = 0.90,
) -> Tuple[List[str], Dict[str, float], Dict[str, float]]:
    """Identify tasks with non-trivial pass rates in BOTH agent groups.

    Non-trivial tasks have meaningful variation - neither too easy nor too hard
    for both pre-frontier and post-frontier agents.

    Args:
        responses_path: Path to JSONL response matrix
        pre_frontier_agents: List of pre-frontier agent names
        post_frontier_agents: List of post-frontier agent names
        min_pass_rate: Minimum pass rate threshold (default 0.10 = 10%)
        max_pass_rate: Maximum pass rate threshold (default 0.90 = 90%)

    Returns:
        Tuple of (nontrivial_task_ids, pre_pass_rates, post_pass_rates)
    """
    pre_pass_rates = compute_pass_rates(responses_path, pre_frontier_agents)
    post_pass_rates = compute_pass_rates(responses_path, post_frontier_agents)

    nontrivial_tasks = []
    for task_id in pre_pass_rates:
        pre_rate = pre_pass_rates.get(task_id, 0.0)
        post_rate = post_pass_rates.get(task_id, 0.0)

        # Both groups must have meaningful variation
        pre_nontrivial = min_pass_rate <= pre_rate <= max_pass_rate
        post_nontrivial = min_pass_rate <= post_rate <= max_pass_rate

        if pre_nontrivial and post_nontrivial:
            nontrivial_tasks.append(task_id)

    return nontrivial_tasks, pre_pass_rates, post_pass_rates


def train_irt_on_agents(
    responses_path: Path,
    agents: List[str],
    seed: int = 42,
    epochs: int = 2000,
) -> Dict[str, float]:
    """Train 1PL IRT model on a subset of agents.

    Args:
        responses_path: Path to response matrix JSONL
        agents: List of agent IDs to include
        seed: Random seed for reproducibility
        epochs: Number of training epochs

    Returns:
        Dict mapping task_id -> difficulty (β)
    """
    set_seed(seed)

    # Load response matrix and filter to specified agents
    agent_set = set(agents)
    data_list = []
    with open(responses_path, "r") as f:
        for line in f:
            record = json.loads(line)
            if record["subject_id"] in agent_set:
                row = {"subject_id": record["subject_id"]}
                row.update(record["responses"])
                data_list.append(row)

    df = pd.DataFrame(data_list)
    item_columns = [col for col in df.columns if col != "subject_id"]
    dataset = Dataset.from_pandas(df, subject_column="subject_id", item_columns=item_columns)

    # Train 1PL IRT (Rasch model)
    config = IrtConfig(
        model_type=OneParamLog,
        priors="hierarchical",
        initializers=[{"name": "difficulty_from_accuracy", "eps": 1e-3}],
        epochs=epochs,
    )

    # Clear pyro param store to avoid conflicts
    pyro.clear_param_store()

    trainer = IrtModelTrainer(config=config, data_path=None, dataset=dataset, verbose=False)
    trainer.train(epochs=epochs)

    # Extract difficulty parameters
    difficulties = list(trainer.best_params["diff"])
    item_id_map = trainer.best_params["item_ids"]
    item_ids = [item_id_map[i] for i in range(len(difficulties))]

    return {task_id: diff for task_id, diff in zip(item_ids, difficulties)}


def compute_pairwise_stability(
    beta_pre: Dict[str, float],
    beta_all: Dict[str, float],
    nontrivial_tasks: List[str],
) -> Dict[str, Any]:
    """Compute pairwise stability metrics comparing pre-frontier to all-agent IRT.

    For each pair (i, j) of non-trivial tasks:
        delta_pre = beta_pre[i] - beta_pre[j]
        delta_all = beta_all[i] - beta_all[j]
        change = delta_all - delta_pre

    Args:
        beta_pre: Dict of task difficulties from pre-frontier IRT
        beta_all: Dict of task difficulties from all-agent IRT
        nontrivial_tasks: List of task IDs to analyze

    Returns:
        Dict with metrics and per-pair DataFrame
    """
    # Filter to tasks present in both models
    tasks = [t for t in nontrivial_tasks if t in beta_pre and t in beta_all]
    n = len(tasks)
    n_pairs = n * (n - 1) // 2

    logger.info(f"Computing pairwise stability for {n} tasks ({n_pairs:,} pairs)")

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            task_i = tasks[i]
            task_j = tasks[j]

            delta_pre = beta_pre[task_i] - beta_pre[task_j]
            delta_all = beta_all[task_i] - beta_all[task_j]
            change = delta_all - delta_pre

            # Order preserved if signs match (or both are zero)
            sign_preserved = np.sign(delta_pre) == np.sign(delta_all)

            pairs.append({
                "task_i": task_i,
                "task_j": task_j,
                "delta_pre": delta_pre,
                "delta_all": delta_all,
                "change": change,
                "abs_change": abs(change),
                "sign_preserved": sign_preserved,
            })

    df = pd.DataFrame(pairs)

    # Compute aggregate metrics
    spearman_result = stats.spearmanr(df["delta_pre"], df["delta_all"])
    pearson_result = stats.pearsonr(df["delta_pre"], df["delta_all"])

    results = {
        "n_tasks": n,
        "n_pairs": n_pairs,
        # Correlation metrics
        "spearman_pairwise": float(spearman_result.statistic),
        "spearman_pvalue": float(spearman_result.pvalue),
        "pearson_pairwise": float(pearson_result.statistic),
        "pearson_pvalue": float(pearson_result.pvalue),
        # Order preservation
        "order_preservation_rate": float(df["sign_preserved"].mean()),
        "n_order_flips": int((~df["sign_preserved"]).sum()),
        # Change magnitude
        "mean_abs_change": float(df["abs_change"].mean()),
        "median_abs_change": float(df["abs_change"].median()),
        "std_abs_change": float(df["abs_change"].std()),
        "max_abs_change": float(df["abs_change"].max()),
        # Mean change (should be ~0 if no systematic shift)
        "mean_change": float(df["change"].mean()),
        # Raw data
        "per_pair_df": df,
    }

    return results


def plot_pairwise_stability(results: Dict[str, Any], output_path: Path):
    """Create scatter plot of delta_pre vs delta_all.

    Perfect stability: points on y=x line
    Order preservation: points in quadrants 1 and 3
    Order flips: points in quadrants 2 and 4
    """
    df = results["per_pair_df"]

    fig, ax = plt.subplots(figsize=(10, 10))

    # Separate preserved vs flipped
    preserved = df[df["sign_preserved"]]
    flipped = df[~df["sign_preserved"]]

    ax.scatter(
        preserved["delta_pre"],
        preserved["delta_all"],
        alpha=0.2,
        c="blue",
        label=f"Order preserved ({len(preserved):,})",
        s=5,
    )
    ax.scatter(
        flipped["delta_pre"],
        flipped["delta_all"],
        alpha=0.5,
        c="red",
        label=f"Order flipped ({len(flipped):,})",
        s=10,
    )

    # Reference line y=x
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "k--", alpha=0.5, label="y=x (perfect stability)")

    # Zero lines
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5)

    ax.set_xlabel("Pairwise difference (pre-frontier IRT): β_pre[i] - β_pre[j]", fontsize=12)
    ax.set_ylabel("Pairwise difference (all-agent IRT): β_all[i] - β_all[j]", fontsize=12)
    ax.set_title(
        f"Stability of Pairwise Task Difficulties\n"
        f"Spearman ρ = {results['spearman_pairwise']:.4f}, "
        f"Order preservation = {results['order_preservation_rate']:.1%}",
        fontsize=14,
    )
    ax.legend(loc="upper left")
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved pairwise scatter plot to {output_path}")


def plot_change_distribution(results: Dict[str, Any], output_path: Path):
    """Create histogram of pairwise changes (delta_all - delta_pre)."""
    df = results["per_pair_df"]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(df["change"], bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(0, color="red", linestyle="--", linewidth=2, label="No change")
    ax.axvline(
        df["change"].mean(),
        color="green",
        linestyle="-",
        linewidth=2,
        label=f"Mean = {df['change'].mean():.4f}",
    )

    ax.set_xlabel("Change in pairwise difference: (β_all[i] - β_all[j]) - (β_pre[i] - β_pre[j])", fontsize=11)
    ax.set_ylabel("Number of pairs", fontsize=12)
    ax.set_title(
        f"Distribution of Pairwise Difficulty Changes\n"
        f"Mean |change| = {results['mean_abs_change']:.4f}",
        fontsize=14,
    )
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved change distribution to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze stability of relative task difficulties")
    parser.add_argument(
        "--response_matrix_path",
        type=str,
        default="clean_data/swebench_verified/swebench_verified_20251120_full.jsonl",
        help="Path to response matrix JSONL",
    )
    parser.add_argument(
        "--frontier_cutoff_date",
        type=str,
        default="20250807",
        help="Date cutoff for pre/post frontier (YYYYMMDD)",
    )
    parser.add_argument(
        "--min_pass_rate",
        type=float,
        default=0.10,
        help="Minimum pass rate for non-trivial tasks (default: 0.10)",
    )
    parser.add_argument(
        "--max_pass_rate",
        type=float,
        default=0.90,
        help="Maximum pass rate for non-trivial tasks (default: 0.90)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for IRT training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2000,
        help="Number of IRT training epochs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="chris_output/difficulty_stability",
        help="Output directory for results",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Pairwise Difficulty Stability Analysis")
    logger.info("=" * 60)
    logger.info(f"Response matrix: {args.response_matrix_path}")
    logger.info(f"Cutoff date: {args.frontier_cutoff_date}")
    logger.info(f"Pass rate range: [{args.min_pass_rate}, {args.max_pass_rate}]")
    logger.info(f"Seed: {args.seed}")

    responses_path = Path(args.response_matrix_path)

    # ===== Step 1: Split agents by cutoff date =====
    logger.info("\n" + "-" * 40)
    logger.info("Step 1: Splitting agents by cutoff date")
    logger.info("-" * 40)

    all_agents = get_all_agents_from_responses(responses_path)
    pre_frontier_agents, post_frontier_agents = split_agents_by_cutoff(
        all_agents, cutoff_date=args.frontier_cutoff_date
    )

    logger.info(f"Total agents: {len(all_agents)}")
    logger.info(f"Pre-frontier (< {args.frontier_cutoff_date}): {len(pre_frontier_agents)}")
    logger.info(f"Post-frontier (>= {args.frontier_cutoff_date}): {len(post_frontier_agents)}")

    all_agents_combined = pre_frontier_agents + post_frontier_agents

    # ===== Step 2: Identify non-trivial tasks =====
    logger.info("\n" + "-" * 40)
    logger.info("Step 2: Identifying non-trivial tasks")
    logger.info("-" * 40)

    nontrivial_tasks, pre_pass_rates, post_pass_rates = identify_nontrivial_tasks(
        responses_path,
        pre_frontier_agents,
        post_frontier_agents,
        min_pass_rate=args.min_pass_rate,
        max_pass_rate=args.max_pass_rate,
    )

    logger.info(f"Non-trivial tasks ({args.min_pass_rate:.0%}-{args.max_pass_rate:.0%} in both groups): {len(nontrivial_tasks)}")

    # ===== Step 3: Train IRT on pre-frontier agents =====
    logger.info("\n" + "-" * 40)
    logger.info("Step 3: Training IRT on pre-frontier agents only")
    logger.info("-" * 40)

    beta_pre = train_irt_on_agents(
        responses_path,
        pre_frontier_agents,
        seed=args.seed,
        epochs=args.epochs,
    )
    logger.info(f"Trained pre-frontier IRT: {len(beta_pre)} tasks")

    # ===== Step 4: Train IRT on all agents =====
    logger.info("\n" + "-" * 40)
    logger.info("Step 4: Training IRT on all agents")
    logger.info("-" * 40)

    beta_all = train_irt_on_agents(
        responses_path,
        all_agents_combined,
        seed=args.seed,
        epochs=args.epochs,
    )
    logger.info(f"Trained all-agent IRT: {len(beta_all)} tasks")

    # ===== Step 5: Compute pairwise stability metrics =====
    logger.info("\n" + "-" * 40)
    logger.info("Step 5: Computing pairwise stability metrics")
    logger.info("-" * 40)

    results = compute_pairwise_stability(beta_pre, beta_all, nontrivial_tasks)

    # ===== Step 6: Generate visualizations =====
    logger.info("\n" + "-" * 40)
    logger.info("Step 6: Generating visualizations")
    logger.info("-" * 40)

    plot_pairwise_stability(results, output_dir / "pairwise_scatter.png")
    plot_change_distribution(results, output_dir / "change_distribution.png")

    # ===== Step 7: Save results =====
    logger.info("\n" + "-" * 40)
    logger.info("Step 7: Saving results")
    logger.info("-" * 40)

    # Save per-pair details
    per_pair_df = results.pop("per_pair_df")
    per_pair_df.to_csv(output_dir / "per_pair_details.csv", index=False)
    logger.info(f"Saved per-pair details to {output_dir / 'per_pair_details.csv'}")

    # Save summary
    summary = {
        "config": vars(args),
        "n_pre_frontier_agents": len(pre_frontier_agents),
        "n_post_frontier_agents": len(post_frontier_agents),
        **results,
    }
    with open(output_dir / "stability_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {output_dir / 'stability_summary.json'}")

    # ===== Print summary =====
    logger.info("\n" + "=" * 60)
    logger.info("PAIRWISE DIFFICULTY STABILITY RESULTS")
    logger.info("=" * 60)
    logger.info(f"Non-trivial tasks analyzed: {results['n_tasks']}")
    logger.info(f"Total pairs analyzed: {results['n_pairs']:,}")
    logger.info("")
    logger.info(f"Spearman ρ (pairwise): {results['spearman_pairwise']:.4f} (p={results['spearman_pvalue']:.2e})")
    logger.info(f"Pearson r (pairwise): {results['pearson_pairwise']:.4f} (p={results['pearson_pvalue']:.2e})")
    logger.info(f"Order preservation rate: {results['order_preservation_rate']:.1%}")
    logger.info(f"Number of order flips: {results['n_order_flips']:,}")
    logger.info("")
    logger.info(f"Mean |change|: {results['mean_abs_change']:.4f}")
    logger.info(f"Median |change|: {results['median_abs_change']:.4f}")
    logger.info(f"Max |change|: {results['max_abs_change']:.4f}")
    logger.info(f"Mean change (should be ~0): {results['mean_change']:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

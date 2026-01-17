"""Evaluate variance in baseline IRT Spearman rho across different random seeds.

This script trains the pre-frontier baseline IRT model multiple times with different
seeds to understand how much noise there is in the Spearman rho metric. This helps
establish confidence intervals for whether SAD-IRT improvements are meaningful.

Usage:
    python -m experiment_sad_irt.baseline_variance --num_seeds 20 --output_dir chris_output/baseline_variance
"""

import argparse
import json
import logging
import sys
from pathlib import Path

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
    get_all_agents_from_responses,
    get_agents_with_trajectories,
    split_agents_by_cutoff,
    identify_frontier_tasks,
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


def train_baseline_irt_with_seed(
    responses_path: Path,
    pre_frontier_agents: list,
    seed: int,
    epochs: int = 2000,
) -> dict:
    """Train standard IRT on pre-frontier agents with a specific seed.

    Args:
        responses_path: Path to response matrix JSONL
        pre_frontier_agents: List of pre-frontier agent IDs
        seed: Random seed for this run
        epochs: Number of training epochs

    Returns:
        Dict mapping task_id -> difficulty (β)
    """
    # Set seed before training
    set_seed(seed)

    # Load response matrix and filter to pre-frontier agents
    pre_frontier_set = set(pre_frontier_agents)
    data_list = []
    with open(responses_path, 'r') as f:
        for line in f:
            record = json.loads(line)
            if record['subject_id'] in pre_frontier_set:
                row = {'subject_id': record['subject_id']}
                row.update(record['responses'])
                data_list.append(row)

    df = pd.DataFrame(data_list)
    item_columns = [col for col in df.columns if col != 'subject_id']
    dataset = Dataset.from_pandas(df, subject_column="subject_id", item_columns=item_columns)

    # Train 1PL IRT
    config = IrtConfig(
        model_type=OneParamLog,
        priors="hierarchical",
        initializers=[
            {"name": "difficulty_from_accuracy", "eps": 1e-3},
        ],
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


def compute_spearman_rho(
    predicted_beta: dict,
    oracle_beta: dict,
    frontier_task_ids: list,
) -> float:
    """Compute Spearman rho for frontier task difficulties."""
    predicted_values = []
    oracle_values = []

    for task_id in frontier_task_ids:
        if task_id in predicted_beta and task_id in oracle_beta:
            predicted_values.append(predicted_beta[task_id])
            oracle_values.append(oracle_beta[task_id])

    if len(predicted_values) < 3:
        return float("nan")

    spearman_rho, _ = stats.spearmanr(predicted_values, oracle_values)
    return float(spearman_rho)


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline IRT variance across seeds")
    parser.add_argument("--num_seeds", type=int, default=20,
                        help="Number of random seeds to try")
    parser.add_argument("--start_seed", type=int, default=0,
                        help="Starting seed value")
    parser.add_argument("--epochs", type=int, default=2000,
                        help="Training epochs per run")
    parser.add_argument("--response_matrix_path", type=str,
                        default="clean_data/swebench_verified/swebench_verified_20251120_full.jsonl")
    parser.add_argument("--trajectory_dir", type=str,
                        default="trajectory_data/unified_trajs")
    parser.add_argument("--oracle_irt_dir", type=str,
                        default="clean_data/swebench_verified_20251120_full/1d")
    parser.add_argument("--frontier_cutoff_date", type=str, default="20250807")
    parser.add_argument("--pre_frontier_threshold", type=float, default=0.1)
    parser.add_argument("--post_frontier_threshold", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="chris_output/baseline_variance")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Baseline IRT Variance Evaluation")
    logger.info("=" * 60)
    logger.info(f"Number of seeds: {args.num_seeds}")
    logger.info(f"Starting seed: {args.start_seed}")
    logger.info(f"Epochs per run: {args.epochs}")

    # ===== Setup: Load agents and identify frontier tasks =====
    responses_path = Path(args.response_matrix_path)
    trajectory_dir = Path(args.trajectory_dir)

    # Get agents with trajectories
    all_agents = get_all_agents_from_responses(responses_path)
    traj_agents = get_agents_with_trajectories(trajectory_dir)
    agents_with_both = [a for a in all_agents if a in traj_agents]

    # Split by cutoff date
    pre_frontier_agents, post_frontier_agents = split_agents_by_cutoff(
        agents_with_both, cutoff_date=args.frontier_cutoff_date
    )
    logger.info(f"Pre-frontier agents: {len(pre_frontier_agents)}")
    logger.info(f"Post-frontier agents: {len(post_frontier_agents)}")

    # Identify frontier tasks
    frontier_task_ids = identify_frontier_tasks(
        responses_path,
        pre_frontier_agents,
        post_frontier_agents,
        pre_threshold=args.pre_frontier_threshold,
        post_threshold=args.post_frontier_threshold,
    )
    logger.info(f"Frontier tasks: {len(frontier_task_ids)}")

    # Load oracle IRT
    oracle_irt_dir = Path(args.oracle_irt_dir)
    oracle_items_df = pd.read_csv(oracle_irt_dir / "items.csv", index_col=0)
    oracle_beta = {
        task_id: oracle_items_df.loc[task_id, "b"]
        for task_id in frontier_task_ids
        if task_id in oracle_items_df.index
    }
    logger.info(f"Oracle β available for {len(oracle_beta)} frontier tasks")

    # ===== Run multiple seeds =====
    logger.info("\n" + "=" * 60)
    logger.info("Running baseline IRT with multiple seeds")
    logger.info("=" * 60)

    results = []
    for i, seed in enumerate(range(args.start_seed, args.start_seed + args.num_seeds)):
        logger.info(f"\n[{i+1}/{args.num_seeds}] Training with seed={seed}")

        try:
            baseline_beta = train_baseline_irt_with_seed(
                responses_path=responses_path,
                pre_frontier_agents=pre_frontier_agents,
                seed=seed,
                epochs=args.epochs,
            )

            spearman_rho = compute_spearman_rho(baseline_beta, oracle_beta, frontier_task_ids)
            logger.info(f"  Spearman ρ = {spearman_rho:.4f}")

            results.append({
                "seed": seed,
                "spearman_rho": spearman_rho,
            })
        except Exception as e:
            logger.error(f"  Failed: {e}")
            results.append({
                "seed": seed,
                "spearman_rho": float("nan"),
                "error": str(e),
            })

    # ===== Analyze results =====
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)

    rhos = [r["spearman_rho"] for r in results if not np.isnan(r["spearman_rho"])]

    if len(rhos) > 0:
        mean_rho = np.mean(rhos)
        std_rho = np.std(rhos)
        min_rho = np.min(rhos)
        max_rho = np.max(rhos)

        # Percentiles
        p5 = np.percentile(rhos, 5)
        p25 = np.percentile(rhos, 25)
        p50 = np.percentile(rhos, 50)
        p75 = np.percentile(rhos, 75)
        p95 = np.percentile(rhos, 95)

        logger.info(f"\nSpearman ρ statistics (n={len(rhos)} successful runs):")
        logger.info(f"  Mean:   {mean_rho:.4f}")
        logger.info(f"  Std:    {std_rho:.4f}")
        logger.info(f"  Min:    {min_rho:.4f}")
        logger.info(f"  Max:    {max_rho:.4f}")
        logger.info(f"  Range:  {max_rho - min_rho:.4f}")
        logger.info(f"\n  5th percentile:  {p5:.4f}")
        logger.info(f"  25th percentile: {p25:.4f}")
        logger.info(f"  Median (50th):   {p50:.4f}")
        logger.info(f"  75th percentile: {p75:.4f}")
        logger.info(f"  95th percentile: {p95:.4f}")

        logger.info(f"\n  95% CI: [{mean_rho - 1.96*std_rho:.4f}, {mean_rho + 1.96*std_rho:.4f}]")

        summary = {
            "num_seeds": args.num_seeds,
            "num_successful": len(rhos),
            "mean": mean_rho,
            "std": std_rho,
            "min": min_rho,
            "max": max_rho,
            "range": max_rho - min_rho,
            "p5": p5,
            "p25": p25,
            "p50": p50,
            "p75": p75,
            "p95": p95,
            "ci_95_lower": mean_rho - 1.96 * std_rho,
            "ci_95_upper": mean_rho + 1.96 * std_rho,
        }
    else:
        logger.error("No successful runs!")
        summary = {"num_seeds": args.num_seeds, "num_successful": 0}

    # ===== Save results =====
    # Save individual results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "per_seed_results.csv", index=False)
    logger.info(f"\nPer-seed results saved to: {output_dir / 'per_seed_results.csv'}")

    # Save summary
    summary["config"] = vars(args)
    summary["all_rhos"] = rhos
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

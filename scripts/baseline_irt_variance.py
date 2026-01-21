#!/usr/bin/env python3
"""Measure variance in Baseline IRT AUC due to random initialization.

This script trains the baseline IRT model multiple times with different
random initializations and measures the variance in evaluation AUC.

Usage:
    python scripts/baseline_irt_variance.py --n_iterations 30
"""

import argparse
import numpy as np
import pandas as pd
import sys
import tempfile
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiment_b import get_dataset_config
from experiment_b.shared.data_splits import (
    get_all_agents_from_responses,
    split_agents_by_dates,
    identify_frontier_tasks,
    identify_frontier_tasks_irt,
    identify_nontrivial_tasks,
)
from experiment_b.shared.evaluate import (
    compute_scale_offset,
    shift_to_oracle_scale,
    compute_frontier_auc,
    load_responses_dict,
)


def main():
    parser = argparse.ArgumentParser(description="Measure Baseline IRT variance")
    parser.add_argument("--n_iterations", type=int, default=30, help="Number of IRT training runs")
    parser.add_argument("--dataset", type=str, default="swebench", choices=["swebench", "terminalbench"])
    parser.add_argument("--epochs", type=int, default=2000, help="Training epochs per run")
    args = parser.parse_args()

    # Load configuration
    config = get_dataset_config(args.dataset)
    all_agents = get_all_agents_from_responses(config.responses_path)
    agent_dates = config.get_agent_dates(all_agents)
    pre_frontier, post_frontier = split_agents_by_dates(all_agents, agent_dates, config.cutoff_date)

    # Load oracle IRT
    oracle_items = pd.read_csv(config.oracle_irt_path, index_col=0)
    oracle_abilities = pd.read_csv(config.oracle_abilities_path, index_col=0)
    oracle_beta = oracle_items['b'].to_dict()

    # Identify frontier tasks (both definitions)
    frontier_passrate = identify_frontier_tasks(
        config.responses_path, pre_frontier, post_frontier, 0.1, 0.1
    )
    frontier_irt = identify_frontier_tasks_irt(
        oracle_items, oracle_abilities, agent_dates, config.cutoff_date, 0.3
    )

    # Identify anchor tasks
    anchor_task_ids, _, _ = identify_nontrivial_tasks(
        config.responses_path, pre_frontier, post_frontier, 0.1, 0.9
    )

    # Load responses
    responses = load_responses_dict(config.responses_path)

    print(f"Dataset: {args.dataset}")
    print(f"Pass-rate frontier: {len(frontier_passrate)} tasks")
    print(f"IRT frontier: {len(frontier_irt)} tasks")
    print(f"Anchor tasks: {len(anchor_task_ids)}")
    print(f"Running {args.n_iterations} iterations...")
    print()

    def compute_auc(baseline_beta, frontier_task_ids, anchor_task_ids):
        alignment_params = compute_scale_offset(
            baseline_beta, oracle_beta, anchor_task_ids, method='affine'
        )
        shifted_beta = shift_to_oracle_scale(baseline_beta, alignment_params)
        # Note: compute_frontier_auc signature is (oracle_abilities, shifted_beta, responses, ...)
        result = compute_frontier_auc(
            oracle_abilities, shifted_beta, responses, frontier_task_ids, post_frontier
        )
        return result['auc']

    # Suppress logging
    import logging
    logging.getLogger('experiment_sad_irt.train_evaluate').setLevel(logging.WARNING)
    logging.getLogger('py_irt').setLevel(logging.WARNING)
    logging.getLogger('experiment_b.shared.evaluate').setLevel(logging.WARNING)

    passrate_aucs = []
    irt_aucs = []

    # Import once before the loop
    import pyro
    from experiment_sad_irt.train_evaluate import train_baseline_irt_on_prefrontier

    for i in range(args.n_iterations):
        try:
            # Clear pyro param store for fresh random init
            pyro.clear_param_store()

            # Train baseline IRT
            temp_dir = Path(tempfile.mkdtemp())

            train_baseline_irt_on_prefrontier(
                config.responses_path, pre_frontier, temp_dir, epochs=args.epochs
            )

            baseline_items = pd.read_csv(temp_dir / 'items.csv', index_col=0)
            baseline_beta = baseline_items['b'].to_dict()

            # Evaluate on both frontier definitions
            auc_passrate = compute_auc(baseline_beta, frontier_passrate, anchor_task_ids)
            auc_irt = compute_auc(baseline_beta, frontier_irt, anchor_task_ids)

            passrate_aucs.append(auc_passrate)
            irt_aucs.append(auc_irt)

            print(f"Iteration {i+1:2d}: Pass-rate AUC = {auc_passrate:.4f}, IRT AUC = {auc_irt:.4f}")

        except Exception as e:
            import traceback
            print(f"Iteration {i+1:2d}: FAILED - {str(e)[:80]}")
            traceback.print_exc()
            continue

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Successful iterations: {len(passrate_aucs)} / {args.n_iterations}")

    if len(passrate_aucs) >= 3:
        print()
        print(f"Pass-rate definition ({len(frontier_passrate)} tasks):")
        print(f"  Mean AUC: {np.mean(passrate_aucs):.4f}")
        print(f"  Std AUC:  {np.std(passrate_aucs):.4f}")
        print(f"  Range:    {np.min(passrate_aucs):.4f} - {np.max(passrate_aucs):.4f}")
        print(f"  1-sigma:  {np.mean(passrate_aucs):.4f} +/- {np.std(passrate_aucs):.4f}")
        print()
        print(f"IRT definition ({len(frontier_irt)} tasks):")
        print(f"  Mean AUC: {np.mean(irt_aucs):.4f}")
        print(f"  Std AUC:  {np.std(irt_aucs):.4f}")
        print(f"  Range:    {np.min(irt_aucs):.4f} - {np.max(irt_aucs):.4f}")
        print(f"  1-sigma:  {np.mean(irt_aucs):.4f} +/- {np.std(irt_aucs):.4f}")
    else:
        print("\nNot enough successful iterations for statistics.")


if __name__ == "__main__":
    main()

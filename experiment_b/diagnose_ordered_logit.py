#!/usr/bin/env python3
"""Diagnostic script to understand why Ordered Logit IRT is underperforming.

Run after compare_methods.py to analyze the predictions.

Usage:
    python -m experiment_b.diagnose_ordered_logit
"""

from pathlib import Path

import numpy as np
import pandas as pd

from experiment_b.shared.rubric_preprocessing import RubricDataSource, RubricPreprocessor
from experiment_b.shared.ordered_logit_predictor import OrderedLogitIRTPredictor


def main():
    print("=" * 80)
    print("ORDERED LOGIT IRT DIAGNOSTICS")
    print("=" * 80)

    # Load rubric data
    rubric_path = Path("chris_output/trajectory_features/raw_features_500tasks_6agents.csv")
    rubric_source = RubricDataSource(rubric_path, RubricPreprocessor())

    # Load baseline IRT
    baseline_dir = Path("chris_output/experiment_b/swebench/baseline_irt")
    cache_dirs = list(baseline_dir.glob("cache_*"))
    if not cache_dirs:
        print("ERROR: No baseline IRT cache found")
        return

    cache_dir = sorted(cache_dirs)[-1]  # Use most recent
    print(f"Using baseline IRT cache: {cache_dir}")

    items_df = pd.read_csv(cache_dir / "items.csv", index_col=0)
    abilities_df = pd.read_csv(cache_dir / "abilities.csv", index_col=0)

    print(f"\nBaseline IRT: {len(items_df)} tasks, {len(abilities_df)} agents")

    # Check rubric agent overlap
    rubric_agents = set(rubric_source.agent_ids)
    baseline_agents = set(abilities_df.index)
    overlap = rubric_agents & baseline_agents

    print(f"\nRubric agents: {rubric_agents}")
    print(f"Overlap with baseline IRT: {len(overlap)}/{len(rubric_agents)}")

    # Check rubric agent abilities
    print("\nRubric agent abilities (from baseline IRT):")
    for agent in rubric_source.agent_ids:
        if agent in abilities_df.index:
            theta = abilities_df.loc[agent, "theta"]
            print(f"  {agent}: theta = {theta:.3f}")

    # Compute eta distribution for rubric data
    print("\n" + "=" * 80)
    print("ETA DISTRIBUTION ANALYSIS")
    print("=" * 80)

    obs_task_ids, obs_agent_ids, _ = rubric_source.get_all_observations()
    eta_values = []
    for t, a in zip(obs_task_ids, obs_agent_ids):
        if t in items_df.index and a in abilities_df.index:
            theta = abilities_df.loc[a, "theta"]
            beta = items_df.loc[t, "b"]
            eta_values.append(theta - beta)

    eta_values = np.array(eta_values)
    print(f"Eta (theta - beta) distribution:")
    print(f"  Mean: {eta_values.mean():.3f}")
    print(f"  Std:  {eta_values.std():.3f}")
    print(f"  Min:  {eta_values.min():.3f}")
    print(f"  Max:  {eta_values.max():.3f}")

    # Now train the model and analyze
    print("\n" + "=" * 80)
    print("TRAINING ORDERED LOGIT MODEL")
    print("=" * 80)

    predictor = OrderedLogitIRTPredictor(
        rubric_path=rubric_path,
        l2_discriminativeness=0.1,
        prior_strength=1.0,
        verbose=True,
    )

    # Get frontier task IDs (0% pre-frontier pass rate)
    # For now, use tasks with highest baseline difficulty as proxy
    sorted_tasks = items_df.sort_values("b", ascending=False)
    frontier_proxy = sorted_tasks.head(34).index.tolist()

    train_task_ids = list(items_df.index)
    ground_truth_b = items_df["b"].values

    predictor.fit(
        task_ids=train_task_ids,
        ground_truth_b=ground_truth_b,
        responses={},  # Not used
        baseline_abilities=abilities_df["theta"].values,
        baseline_agent_ids=list(abilities_df.index),
        frontier_task_ids=frontier_proxy,
    )

    # Predict for all tasks with rubric data
    rubric_task_ids = rubric_source.task_ids
    tasks_to_predict = [t for t in rubric_task_ids if t in items_df.index]

    print(f"\nPredicting for {len(tasks_to_predict)} tasks...")
    predictions = predictor.predict(tasks_to_predict)

    # Analyze predictions
    print("\n" + "=" * 80)
    print("PREDICTION ANALYSIS")
    print("=" * 80)

    pred_betas = np.array([predictions[t] for t in tasks_to_predict])
    baseline_betas = np.array([items_df.loc[t, "b"] for t in tasks_to_predict])

    print(f"\nPredicted beta distribution:")
    print(f"  Mean: {pred_betas.mean():.3f}")
    print(f"  Std:  {pred_betas.std():.3f}")
    print(f"  Min:  {pred_betas.min():.3f}")
    print(f"  Max:  {pred_betas.max():.3f}")

    print(f"\nBaseline beta distribution (same tasks):")
    print(f"  Mean: {baseline_betas.mean():.3f}")
    print(f"  Std:  {baseline_betas.std():.3f}")
    print(f"  Min:  {baseline_betas.min():.3f}")
    print(f"  Max:  {baseline_betas.max():.3f}")

    # Check if predictions are clustering around prior
    diag = predictor.get_training_diagnostics()
    prior_mean = diag["prior_mean"]
    prior_std = diag["prior_std"]

    print(f"\nPrior: N({prior_mean:.3f}, {prior_std:.3f}^2)")
    print(f"Predictions within 1 std of prior mean: {np.mean(np.abs(pred_betas - prior_mean) < prior_std):.1%}")
    print(f"Predictions within 0.5 std of prior mean: {np.mean(np.abs(pred_betas - prior_mean) < 0.5*prior_std):.1%}")

    # Correlation analysis
    from scipy.stats import pearsonr, spearmanr

    r_pearson, p_pearson = pearsonr(pred_betas, baseline_betas)
    r_spearman, p_spearman = spearmanr(pred_betas, baseline_betas)

    print(f"\nCorrelation (Predicted vs Baseline):")
    print(f"  Pearson:  r={r_pearson:.3f}, p={p_pearson:.4f}")
    print(f"  Spearman: r={r_spearman:.3f}, p={p_spearman:.4f}")

    # Check variance explained
    ss_tot = np.sum((baseline_betas - baseline_betas.mean())**2)
    ss_res = np.sum((baseline_betas - pred_betas)**2)
    r2 = 1 - ss_res / ss_tot
    print(f"  R² (variance explained): {r2:.3f}")

    # Diagnose: is the problem small lambda or strong prior?
    print("\n" + "=" * 80)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 80)

    params = diag["ordered_logit_params"]
    avg_lambda = np.mean([p["discriminativeness"] for p in params.values()])
    print(f"\nAverage discriminativeness (lambda): {avg_lambda:.3f}")

    # With 6 agents and 7 rubric items, each task has 42 observations
    # The effective likelihood strength per task is roughly:
    # n_obs * avg_lambda^2 (Fisher information)
    n_obs_per_task = 6 * 7
    fisher_approx = n_obs_per_task * avg_lambda**2
    print(f"Observations per task: {n_obs_per_task}")
    print(f"Approximate Fisher information per task: {fisher_approx:.3f}")
    print(f"Prior precision (1/sigma^2): {1/prior_std**2:.3f}")

    print(f"\nLikelihood/Prior ratio: {fisher_approx / (1/prior_std**2):.3f}")
    print("(Values < 1 mean prior dominates, > 1 means likelihood dominates)")

    # Suggestion
    print("\n" + "=" * 80)
    print("SUGGESTIONS")
    print("=" * 80)

    if fisher_approx < 1 / prior_std**2:
        print("DIAGNOSIS: Prior is dominating the likelihood.")
        print("\nPossible fixes:")
        print("  1. Reduce prior_strength (try 0.1 or 0.01)")
        print("  2. Increase discriminativeness by reducing l2_discriminativeness")
        print("  3. Use more rubric items or agents per task")
    else:
        print("DIAGNOSIS: Likelihood should have enough signal.")
        print("Issue may be with the model structure or scale alignment.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create scatter plot of predicted vs actual dates for Feature-IRT on frontier tasks.

Uses the pass-rate frontier definition (<=10% pre, >10% post) on SWE-bench Verified.
Reuses data loading and model training code from compare_methods.py.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiment_appendix_h_hard_tasks import get_dataset_config
from experiment_appendix_h_hard_tasks.shared.data_splits import (
    get_all_agents_from_responses,
    identify_frontier_tasks,
    split_agents_by_dates,
)
from experiment_appendix_h_hard_tasks.shared.evaluate import load_responses_dict
from experiment_appendix_h_hard_tasks.shared.date_forecasting import (
    compute_first_capable_dates,
    compute_ground_truth_days,
    DateForecastModel,
)
from experiment_appendix_h_hard_tasks.shared.baseline_irt import get_or_train_baseline_irt
from experiment_appendix_h_hard_tasks.shared.feature_irt_predictor import FeatureIRTPredictor
from experiment_new_tasks.feature_source import EmbeddingFeatureSource


def main():
    # Load dataset config
    dataset_config = get_dataset_config("swebench")

    # Load oracle IRT
    oracle_items = pd.read_csv(dataset_config.oracle_irt_path, index_col=0)
    oracle_abilities = pd.read_csv(dataset_config.oracle_abilities_path, index_col=0)
    print(f"Loaded oracle IRT: {len(oracle_items)} tasks, {len(oracle_abilities)} agents")

    # Load response matrix
    responses = load_responses_dict(dataset_config.responses_path)
    print(f"Loaded responses for {len(responses)} agents")

    # Get agent dates and split by cutoff
    all_agents = get_all_agents_from_responses(dataset_config.responses_path)
    agent_dates = dataset_config.get_agent_dates(all_agents)
    cutoff_date = dataset_config.cutoff_date

    pre_frontier, post_frontier = split_agents_by_dates(all_agents, agent_dates, cutoff_date)
    print(f"Pre-frontier agents: {len(pre_frontier)}, Post-frontier agents: {len(post_frontier)}")

    # Identify frontier tasks using pass-rate definition (<=10% pre, >10% post)
    pre_threshold = dataset_config.pre_threshold
    post_threshold = dataset_config.post_threshold

    frontier_task_ids = identify_frontier_tasks(
        dataset_config.responses_path,
        pre_frontier,
        post_frontier,
        pre_threshold,
        post_threshold,
    )
    print(f"Frontier tasks (pass-rate, <={pre_threshold:.0%} pre, >{post_threshold:.0%} post): {len(frontier_task_ids)}")

    # Compute ground truth dates
    gt_result = compute_first_capable_dates(oracle_items, oracle_abilities, agent_dates)
    first_capable_dates = gt_result.first_capable_dates
    earliest_agent_date = gt_result.earliest_agent_date

    ground_truth_days = compute_ground_truth_days(
        list(oracle_items.index), first_capable_dates, earliest_agent_date
    )
    print(f"Tasks with ground truth: {len(first_capable_dates)}")

    # Load or train baseline IRT (needed for Feature-IRT ground truth)
    print("\nLoading/training baseline IRT...")
    baseline_items, baseline_abilities = get_or_train_baseline_irt(
        responses_path=dataset_config.responses_path,
        pre_frontier_agents=pre_frontier,
        cutoff_date=cutoff_date,
        output_dir=dataset_config.output_dir,
    )
    print(f"Baseline IRT: {len(baseline_items)} tasks, {len(baseline_abilities)} agents")

    # Train Feature-IRT (Embedding) - same as compare_methods.py
    print("\nTraining Feature-IRT (Embedding)...")
    embeddings_path = dataset_config.embeddings_path
    feature_source = EmbeddingFeatureSource(embeddings_path)

    predictor = FeatureIRTPredictor(
        feature_source,
        use_residuals=True,
        l2_weight=0.01,
        l2_residual=10.0,
    )

    # Fit with pre-frontier agents only (no data leakage)
    train_tasks_available = [t for t in baseline_items.index if t in baseline_items.index]
    ground_truth_b = baseline_items.loc[train_tasks_available, "b"].values
    train_responses = {
        agent_id: agent_responses
        for agent_id, agent_responses in responses.items()
        if agent_id in pre_frontier
    }
    print(f"  Training on {len(train_tasks_available)} tasks with {len(train_responses)} pre-frontier agents")

    predictor.fit(
        task_ids=train_tasks_available,
        ground_truth_b=ground_truth_b,
        responses=train_responses,
    )

    # Get predictions and learned abilities
    feature_irt_beta = predictor.predict(train_tasks_available)
    feature_irt_theta = predictor.learned_abilities
    print(f"Feature-IRT predictions: {len(feature_irt_beta)} tasks, {len(feature_irt_theta)} agents")

    # Fit date model using Feature-IRT abilities
    date_model = DateForecastModel()
    fit_stats = date_model.fit(feature_irt_theta, agent_dates)
    print(f"Date model fit: R² = {fit_stats['r_squared']:.4f}, slope = {fit_stats['slope']:.6f}")

    # Get frontier tasks that have ground truth
    eval_tasks = [t for t in frontier_task_ids if t in first_capable_dates and t in feature_irt_beta]
    print(f"Frontier tasks with ground truth and Feature-IRT predictions: {len(eval_tasks)}")

    # Make predictions
    predictions = date_model.predict(feature_irt_beta, eval_tasks)

    # Extract predicted and actual days
    predicted_days = []
    actual_days = []
    task_ids_plotted = []

    for task_id in eval_tasks:
        if task_id in predictions and task_id in ground_truth_days:
            pred_day, pred_date = predictions[task_id]
            actual_day = ground_truth_days[task_id]
            predicted_days.append(pred_day)
            actual_days.append(actual_day)
            task_ids_plotted.append(task_id)

    predicted_days = np.array(predicted_days)
    actual_days = np.array(actual_days)

    print(f"Points in scatter plot: {len(predicted_days)}")

    # Compute metrics
    mae = np.mean(np.abs(predicted_days - actual_days))
    rmse = np.sqrt(np.mean((predicted_days - actual_days) ** 2))
    correlation = np.corrcoef(predicted_days, actual_days)[0, 1]

    print(f"MAE: {mae:.1f} days")
    print(f"RMSE: {rmse:.1f} days")
    print(f"Pearson r: {correlation:.4f}")

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(actual_days, predicted_days, alpha=0.7, s=50, c='steelblue', edgecolors='white', linewidth=0.5)

    # Add diagonal line (perfect prediction)
    min_val = min(actual_days.min(), predicted_days.min())
    max_val = max(actual_days.max(), predicted_days.max())
    margin = (max_val - min_val) * 0.05
    ax.plot([min_val - margin, max_val + margin], [min_val - margin, max_val + margin],
            'k--', alpha=0.5, label='Perfect prediction')

    # Labels and title
    ax.set_xlabel('Actual Days (since earliest agent)', fontsize=12)
    ax.set_ylabel('Predicted Days (Feature-IRT)', fontsize=12)
    ax.set_title(f'Feature-IRT Date Forecasting on Pass-rate Frontier Tasks\n'
                 f'SWE-bench Verified (n={len(predicted_days)}, MAE={mae:.1f} days, r={correlation:.3f})',
                 fontsize=12)

    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(min_val - margin, max_val + margin)
    ax.set_ylim(min_val - margin, max_val + margin)

    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = Path("output/figures/date_forecast_scatter_feature_irt.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved figure to: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()

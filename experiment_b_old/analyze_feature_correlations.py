"""Quick analysis of feature correlations with difficulty residual.

This script computes mechanical and cross-agent features for a sample of tasks
and correlates them with the difficulty residual (ground_truth_b - prior_predicted_b).

Usage:
    python -m experiment_b.analyze_feature_correlations --n_tasks 20 --n_agents 10
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Add parent to path for imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_b.config import ExperimentConfig
from experiment_b.data_splits import create_experiment_split
from experiment_b.trajectory_features_v2 import (
    MECHANICAL_FEATURE_NAMES,
    CROSS_AGENT_FEATURE_NAMES,
    extract_mechanical_features,
    aggregate_mechanical_features,
    compute_cross_agent_features,
    MechanicalFeatures,
)
from experiment_b.prior_model import PriorModel


UNIFIED_TRAJS_DIR = ROOT / "trajectory_data" / "unified_trajs"


def load_trajectory(agent: str, task_id: str) -> dict | None:
    """Load a unified trajectory JSON file."""
    traj_path = UNIFIED_TRAJS_DIR / agent / f"{task_id}.json"
    if not traj_path.exists():
        return None
    try:
        with open(traj_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def compute_all_features_for_task(
    task_id: str,
    agents: List[str],
) -> Tuple[Dict[str, float], Dict[str, MechanicalFeatures], Dict[str, bool]]:
    """Compute all features for a task across agents.

    Returns:
        Tuple of (aggregated_features, per_agent_features, resolved_status)
    """
    agent_features = {}
    agent_resolved = {}

    for agent in agents:
        traj = load_trajectory(agent, task_id)
        if traj is None:
            continue

        features = extract_mechanical_features(traj)
        agent_features[agent] = features
        agent_resolved[agent] = traj.get("resolved", False)

    if not agent_features:
        return {}, {}, {}

    # Compute aggregated mechanical features (means)
    agg_mechanical = aggregate_mechanical_features(agent_features)

    # Compute cross-agent features
    cross_agent = compute_cross_agent_features(agent_features, agent_resolved)

    # Also compute mean and std of each mechanical feature
    all_features = {}

    # Mechanical feature means (normalized)
    for i, name in enumerate(MECHANICAL_FEATURE_NAMES):
        all_features[f"{name}_mean"] = agg_mechanical[i]

    # Mechanical feature raw means (unnormalized for interpretability)
    all_features["syntax_error_mean_raw"] = np.mean([f.syntax_error_count for f in agent_features.values()])
    all_features["test_run_mean_raw"] = np.mean([f.test_run_count for f in agent_features.values()])
    all_features["traceback_mean_raw"] = np.mean([f.traceback_count for f in agent_features.values()])
    all_features["files_edited_mean_raw"] = np.mean([len(f.unique_files_edited) for f in agent_features.values()])
    all_features["commands_mean_raw"] = np.mean([f.total_commands for f in agent_features.values()])
    all_features["edit_attempts_mean_raw"] = np.mean([f.edit_attempts for f in agent_features.values()])

    # Cross-agent features
    for name, value in cross_agent.items():
        all_features[name] = value

    return all_features, agent_features, agent_resolved


def main():
    parser = argparse.ArgumentParser(description="Analyze feature correlations")
    parser.add_argument("--n_tasks", type=int, default=20, help="Number of tasks to analyze")
    parser.add_argument("--n_agents", type=int, default=10, help="Number of agents per task")
    args = parser.parse_args()

    # Load config and create splits
    config = ExperimentConfig()
    print("Loading data...")

    # Load IRT difficulties
    items_path = ROOT / config.items_path
    items_df = pd.read_csv(items_path, index_col=0)
    print(f"Loaded {len(items_df)} task difficulties")

    # Create splits to get agent lists
    split = create_experiment_split(
        responses_path=ROOT / config.responses_path,
        trajectories_dir=ROOT / config.trajectories_dir,
        weak_threshold=config.weak_threshold,
        strong_min_improvement=config.strong_min_improvement,
        m1_fraction=config.m1_fraction,
        m2_fraction=config.m2_fraction,
    )

    # Train simple prior model to get residuals
    print("\nTraining prior model...")
    all_task_ids = list(items_df.index)
    all_difficulties = items_df["b"].values
    prior_model = PriorModel(alpha=config.prior_alpha)
    prior_model.fit(all_task_ids, all_difficulties)

    # Select tasks (use D_train which has clear difficulty signal)
    task_ids = split.d_train_tasks[:args.n_tasks]
    agents = split.m1_agents[:args.n_agents]

    print(f"\nAnalyzing {len(task_ids)} tasks with {len(agents)} agents each")
    print(f"Tasks: {task_ids[:5]}...")
    print(f"Agents: {agents[:3]}...")

    # Compute features for each task
    results = []

    for task_id in task_ids:
        # Get ground truth and prior prediction
        if task_id not in items_df.index:
            continue

        ground_truth_b = items_df.loc[task_id, "b"]
        prior_preds = prior_model.get_prior_predictions([task_id])
        if task_id not in prior_preds:
            continue
        prior_b = prior_preds[task_id]
        residual = ground_truth_b - prior_b

        # Compute features
        features, _, _ = compute_all_features_for_task(task_id, agents)
        if not features:
            print(f"  Skipping {task_id}: no features")
            continue

        features["task_id"] = task_id
        features["ground_truth_b"] = ground_truth_b
        features["prior_b"] = prior_b
        features["residual"] = residual

        results.append(features)

    if not results:
        print("No results to analyze!")
        return

    df = pd.DataFrame(results)
    print(f"\nComputed features for {len(df)} tasks")

    # Compute correlations with residual
    print("\n" + "=" * 70)
    print("FEATURE CORRELATIONS WITH DIFFICULTY RESIDUAL")
    print("(residual = ground_truth_b - prior_b, positive = harder than expected)")
    print("=" * 70)

    feature_cols = [c for c in df.columns if c not in ["task_id", "ground_truth_b", "prior_b", "residual"]]

    correlations = []
    for col in feature_cols:
        if df[col].std() < 1e-10:  # Skip constant features
            continue
        r, p = stats.pearsonr(df[col], df["residual"])
        correlations.append({
            "feature": col,
            "pearson_r": r,
            "p_value": p,
            "significant": p < 0.05,
        })

    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values("pearson_r", key=abs, ascending=False)

    print("\nTop correlations (sorted by |r|):")
    print("-" * 70)
    for _, row in corr_df.head(15).iterrows():
        sig = "*" if row["significant"] else " "
        print(f"  {row['feature']:35s}  r={row['pearson_r']:+.3f}  p={row['p_value']:.3f} {sig}")

    print("\n* = significant at p < 0.05")

    # Show feature statistics
    print("\n" + "=" * 70)
    print("FEATURE STATISTICS")
    print("=" * 70)

    print("\nCross-agent features:")
    for name in CROSS_AGENT_FEATURE_NAMES:
        if name in df.columns:
            print(f"  {name:30s}  mean={df[name].mean():.3f}  std={df[name].std():.3f}")

    print("\nMechanical feature means (raw):")
    for col in df.columns:
        if col.endswith("_raw"):
            print(f"  {col:30s}  mean={df[col].mean():.2f}  std={df[col].std():.2f}")

    # Show some example tasks
    print("\n" + "=" * 70)
    print("EXAMPLE TASKS")
    print("=" * 70)

    # Highest positive residual (harder than expected)
    hard_task = df.loc[df["residual"].idxmax()]
    print(f"\nHardest (vs prior): {hard_task['task_id']}")
    print(f"  residual: {hard_task['residual']:.2f} (ground_truth={hard_task['ground_truth_b']:.2f}, prior={hard_task['prior_b']:.2f})")
    print(f"  edit_location_entropy: {hard_task.get('edit_location_entropy', 'N/A'):.3f}")
    print(f"  resolved_rate: {hard_task.get('resolved_rate', 'N/A'):.3f}")

    # Lowest residual (easier than expected)
    easy_task = df.loc[df["residual"].idxmin()]
    print(f"\nEasiest (vs prior): {easy_task['task_id']}")
    print(f"  residual: {easy_task['residual']:.2f} (ground_truth={easy_task['ground_truth_b']:.2f}, prior={easy_task['prior_b']:.2f})")
    print(f"  edit_location_entropy: {easy_task.get('edit_location_entropy', 'N/A'):.3f}")
    print(f"  resolved_rate: {easy_task.get('resolved_rate', 'N/A'):.3f}")


if __name__ == "__main__":
    main()

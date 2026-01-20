"""Analyze correlation of test progression features with task difficulty.

This script examines whether test progression features correlate with:
1. Ground truth difficulty (β) - IRT difficulty parameter
2. Residual (β - prior) - What the prior model can't explain

Usage:
    python -m experiment_b.test_progression.analyze_correlation

    # Only use trajectories with multiple test runs
    python -m experiment_b.test_progression.analyze_correlation --min_runs 2

    # Only use trajectories with granular per-test data
    python -m experiment_b.test_progression.analyze_correlation --require_granular
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Add parent to path for imports
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_b.test_progression.features import (
    TEST_PROGRESSION_FEATURE_NAMES,
    features_from_dict,
    to_raw_vector,
)
from experiment_b.config import ExperimentConfig
from experiment_b.prior_model import EmbeddingPriorModel


# Directories
FEATURES_DIR = ROOT / "chris_output" / "experiment_b" / "test_progression_features"
IRT_ITEMS_PATH = ROOT / "clean_data" / "swebench_verified_20251120_full" / "1d" / "items.csv"


def load_irt_difficulties() -> Dict[str, float]:
    """Load IRT difficulty (b) parameters for all tasks.

    Returns:
        Dict mapping task_id -> difficulty (b value)
    """
    if not IRT_ITEMS_PATH.exists():
        print(f"Error: IRT items file not found: {IRT_ITEMS_PATH}")
        return {}

    df = pd.read_csv(IRT_ITEMS_PATH, index_col=0)
    return df["b"].to_dict()


def load_test_progression_features(
    min_runs: int = 0,
    require_granular: bool = False,
    require_test_output: bool = False,
) -> Dict[str, Dict[str, dict]]:
    """Load test progression features from computed files.

    Args:
        min_runs: Minimum number of test runs required
        require_granular: If True, only load features with granular data
        require_test_output: If True, only load features with test output

    Returns:
        Dict mapping task_id -> agent -> feature dict
    """
    if not FEATURES_DIR.exists():
        print(f"Error: Features directory not found: {FEATURES_DIR}")
        print("Run: python -m experiment_b.test_progression.compute_features")
        return {}

    features_by_task: Dict[str, Dict[str, dict]] = {}
    total_loaded = 0
    total_filtered = 0

    for agent_dir in FEATURES_DIR.iterdir():
        if not agent_dir.is_dir():
            continue

        agent = agent_dir.name

        for task_file in agent_dir.glob("*.json"):
            try:
                with open(task_file) as f:
                    data = json.load(f)

                total_loaded += 1

                # Apply filters
                if require_test_output and not data.get("has_test_output", False):
                    total_filtered += 1
                    continue
                if require_granular and not data.get("has_granular_data", False):
                    total_filtered += 1
                    continue
                if data.get("num_test_runs", 0) < min_runs:
                    total_filtered += 1
                    continue

                task_id = data.get("task_id", task_file.stem)

                if task_id not in features_by_task:
                    features_by_task[task_id] = {}
                features_by_task[task_id][agent] = data

            except (json.JSONDecodeError, IOError):
                continue

    print(f"Loaded {total_loaded} feature files, filtered {total_filtered}")
    print(f"Tasks with features: {len(features_by_task)}")

    return features_by_task


def compute_prior_predictions(task_ids: List[str]) -> Dict[str, float]:
    """Compute prior model predictions for tasks.

    Args:
        task_ids: List of task IDs

    Returns:
        Dict mapping task_id -> prior prediction
    """
    # Load IRT difficulties for fitting
    difficulties = load_irt_difficulties()

    # Get all task IDs with difficulties
    all_task_ids = list(difficulties.keys())

    # Fit prior model on all data
    config = ExperimentConfig()

    # Default embeddings path
    embeddings_path = config.embeddings_path
    if embeddings_path is None:
        # Try the actual location
        embeddings_path = ROOT / "out" / "prior_qwen3vl8b" / "embeddings__Qwen__Qwen3-VL-8B-Instruct__pool-lasttoken__qs-sol-instr__qs_sol_instr_b7008f2d__idnorm_instance-v1__princeton-nlp_SWE-bench_Verified__test__n500__maxlen8192__seed0.npz"
    else:
        embeddings_path = ROOT / embeddings_path

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    prior_model = EmbeddingPriorModel(
        embeddings_path=embeddings_path,
        alpha=config.prior_alpha,
    )

    # Fit on all tasks
    task_difficulties = np.array([difficulties[t] for t in all_task_ids if t in difficulties])
    valid_task_ids = [t for t in all_task_ids if t in difficulties]

    prior_model.fit(valid_task_ids, task_difficulties)

    # Predict for requested tasks
    predictions = {}
    for task_id in task_ids:
        if task_id in valid_task_ids:
            pred = prior_model.predict([task_id])[0]
            predictions[task_id] = pred

    return predictions


def aggregate_features_for_task(
    agent_features: Dict[str, dict],
) -> np.ndarray:
    """Aggregate features across agents for a task.

    Args:
        agent_features: Dict mapping agent -> feature dict

    Returns:
        Aggregated feature vector
    """
    if not agent_features:
        return np.zeros(len(TEST_PROGRESSION_FEATURE_NAMES))

    vectors = []
    for agent, data in agent_features.items():
        features = features_from_dict(data)
        vectors.append(to_raw_vector(features))

    return np.mean(vectors, axis=0)


def analyze_correlations(
    features_by_task: Dict[str, Dict[str, dict]],
    difficulties: Dict[str, float],
    prior_predictions: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Analyze correlations between features and difficulty.

    Args:
        features_by_task: Dict mapping task_id -> agent -> feature dict
        difficulties: Dict mapping task_id -> difficulty (b)
        prior_predictions: Optional dict mapping task_id -> prior prediction

    Returns:
        DataFrame with correlation results
    """
    # Build aligned arrays
    task_ids = []
    feature_matrix = []
    difficulty_values = []
    residual_values = []

    for task_id, agent_features in features_by_task.items():
        if task_id not in difficulties:
            continue

        agg_features = aggregate_features_for_task(agent_features)
        b = difficulties[task_id]

        task_ids.append(task_id)
        feature_matrix.append(agg_features)
        difficulty_values.append(b)

        if prior_predictions and task_id in prior_predictions:
            residual = b - prior_predictions[task_id]
            residual_values.append(residual)
        else:
            residual_values.append(np.nan)

    if not task_ids:
        print("No tasks with both features and difficulties")
        return pd.DataFrame()

    X = np.array(feature_matrix)
    y_difficulty = np.array(difficulty_values)
    y_residual = np.array(residual_values)

    print(f"\nAnalyzing {len(task_ids)} tasks with features and difficulties")
    print(f"Difficulty stats: mean={y_difficulty.mean():.2f}, std={y_difficulty.std():.2f}")
    if not np.all(np.isnan(y_residual)):
        valid_residual = y_residual[~np.isnan(y_residual)]
        print(f"Residual stats: mean={valid_residual.mean():.2f}, std={valid_residual.std():.2f}")

    # Compute correlations for each feature
    results = []
    for i, name in enumerate(TEST_PROGRESSION_FEATURE_NAMES):
        feature_values = X[:, i]

        # Skip if no variance
        if np.std(feature_values) < 1e-10:
            results.append({
                "feature": name,
                "r_difficulty": np.nan,
                "p_difficulty": np.nan,
                "r_residual": np.nan,
                "p_residual": np.nan,
            })
            continue

        # Correlation with difficulty
        r_diff, p_diff = stats.pearsonr(feature_values, y_difficulty)

        # Correlation with residual
        if not np.all(np.isnan(y_residual)):
            valid_mask = ~np.isnan(y_residual)
            if np.sum(valid_mask) > 2:
                r_res, p_res = stats.pearsonr(
                    feature_values[valid_mask],
                    y_residual[valid_mask]
                )
            else:
                r_res, p_res = np.nan, np.nan
        else:
            r_res, p_res = np.nan, np.nan

        results.append({
            "feature": name,
            "r_difficulty": r_diff,
            "p_difficulty": p_diff,
            "r_residual": r_res,
            "p_residual": p_res,
        })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Analyze test progression feature correlations")
    parser.add_argument("--min_runs", type=int, default=0,
                        help="Minimum test runs required per trajectory")
    parser.add_argument("--require_granular", action="store_true",
                        help="Only use trajectories with granular test data")
    parser.add_argument("--require_test_output", action="store_true",
                        help="Only use trajectories with any test output")
    parser.add_argument("--skip_prior", action="store_true",
                        help="Skip prior model computation (faster)")
    args = parser.parse_args()

    print("=== Test Progression Feature Correlation Analysis ===\n")

    # Load IRT difficulties
    print("Loading IRT difficulties...")
    difficulties = load_irt_difficulties()
    print(f"Loaded {len(difficulties)} task difficulties")

    # Load test progression features
    print("\nLoading test progression features...")
    features_by_task = load_test_progression_features(
        min_runs=args.min_runs,
        require_granular=args.require_granular,
        require_test_output=args.require_test_output,
    )

    if not features_by_task:
        print("No features loaded. Run compute_features.py first.")
        return

    # Compute prior predictions (optional)
    prior_predictions = None
    if not args.skip_prior:
        print("\nComputing prior predictions...")
        task_ids = list(features_by_task.keys())
        try:
            prior_predictions = compute_prior_predictions(task_ids)
            print(f"Computed prior predictions for {len(prior_predictions)} tasks")
        except Exception as e:
            print(f"Warning: Could not compute prior predictions: {e}")
            print("Continuing without residual analysis...")

    # Analyze correlations
    print("\nComputing correlations...")
    results_df = analyze_correlations(
        features_by_task,
        difficulties,
        prior_predictions,
    )

    if results_df.empty:
        print("No results to display")
        return

    # Display results
    print("\n" + "=" * 80)
    print("CORRELATION RESULTS")
    print("=" * 80)

    print("\nCorrelation with Ground Truth Difficulty (β):")
    print("-" * 60)
    for _, row in results_df.sort_values("p_difficulty").iterrows():
        sig = "***" if row["p_difficulty"] < 0.001 else "**" if row["p_difficulty"] < 0.01 else "*" if row["p_difficulty"] < 0.05 else ""
        print(f"  {row['feature']:30s}  r={row['r_difficulty']:+.3f}  p={row['p_difficulty']:.4f} {sig}")

    if prior_predictions:
        print("\nCorrelation with Residual (β - prior):")
        print("-" * 60)
        for _, row in results_df.sort_values("p_residual").iterrows():
            if np.isnan(row["r_residual"]):
                continue
            sig = "***" if row["p_residual"] < 0.001 else "**" if row["p_residual"] < 0.01 else "*" if row["p_residual"] < 0.05 else ""
            print(f"  {row['feature']:30s}  r={row['r_residual']:+.3f}  p={row['p_residual']:.4f} {sig}")

    print("\n" + "=" * 80)
    print("Significance: * p<0.05, ** p<0.01, *** p<0.001")

    # Summary
    significant_diff = results_df[results_df["p_difficulty"] < 0.05]
    significant_res = results_df[results_df["p_residual"] < 0.05]

    print(f"\nFeatures significantly correlated with difficulty: {len(significant_diff)}")
    print(f"Features significantly correlated with residual: {len(significant_res)}")


if __name__ == "__main__":
    main()

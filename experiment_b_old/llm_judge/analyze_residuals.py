"""Analyze where LLM judge predictions help vs hurt on residuals.

This script examines:
1. What are the actual embedding prior residuals (ground_truth_b - prior_pred)?
2. What residuals is the LLM judge predicting?
3. On which tasks is the LLM judge helping vs hurting?
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Add parent to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_b.config import ExperimentConfig
from experiment_b.prior_model import EmbeddingPriorModel
from experiment_b.llm_judge.features_v1 import (
    LLM_JUDGE_FEATURE_NAMES,
    load_llm_judge_features_for_task,
    aggregate_llm_judge_features,
)


def load_llm_judge_features_detailed(
    task_id: str,
    agents: List[str],
    features_dir: Path,
) -> Tuple[np.ndarray, Dict[str, Dict]]:
    """Load LLM judge features and return both aggregated and per-agent details."""
    features_dict = load_llm_judge_features_for_task(task_id, agents, features_dir)
    if not features_dict:
        return None, {}

    aggregated = aggregate_llm_judge_features(features_dict)

    # Convert to detailed dict
    detailed = {}
    for agent, feat in features_dict.items():
        detailed[agent] = {
            "llm_judge_difficulty_score": feat.llm_judge_difficulty_score,
            "backtracking_exploration": feat.backtracking_exploration,
            "task_decomposition": feat.task_decomposition,
            "localization_failure": feat.localization_failure,
            "strategy_defect": feat.strategy_defect,
            "implementation_defect": feat.implementation_defect,
            "agent_looping": feat.agent_looping,
            "agent_wrong_focus": feat.agent_wrong_focus,
        }

    return aggregated, detailed


def main():
    # Load config and paths
    embeddings_path = ROOT / "out/prior_qwen3vl8b/embeddings__Qwen__Qwen3-VL-8B-Instruct__pool-lasttoken__qs-sol-instr__qs_sol_instr_b7008f2d__idnorm_instance-v1__princeton-nlp_SWE-bench_Verified__test__n500__maxlen8192__seed0.npz"
    items_path = ROOT / "clean_data/swebench_verified_20251120_full/1d/items.csv"
    llm_judge_dir = ROOT / "chris_output/experiment_b/llm_judge_features"
    results_path = ROOT / "chris_output/experiment_b/embedding_llm_judge_fixed/experiment_b_results.json"

    # Load IRT difficulties
    items_df = pd.read_csv(items_path, index_col=0)
    print(f"Loaded {len(items_df)} tasks")

    # Load experiment results for task splits and psi coefficients
    with open(results_path) as f:
        results = json.load(f)

    d_train_tasks = results["split"]["d_train_tasks"]
    d_valid_tasks = results["split"]["d_valid_tasks"]
    m1_agents = results["split"]["m1_agents"]
    m2_agents = results["split"]["m2_agents"]
    psi_coefficients = results["psi_coefficients"]

    print(f"\nD_train: {len(d_train_tasks)} tasks")
    print(f"D_valid: {len(d_valid_tasks)} tasks")

    # Train embedding prior on ALL tasks (same as in train_evaluate.py)
    print("\nTraining embedding prior...")
    prior_model = EmbeddingPriorModel(embeddings_path, alpha=10000.0)
    all_task_ids = list(items_df.index)
    all_difficulties = items_df["b"].values
    prior_model.fit(all_task_ids, all_difficulties)

    # Get prior predictions for train and valid tasks
    prior_train_preds = prior_model.get_prior_predictions(d_train_tasks)
    prior_valid_preds = prior_model.get_prior_predictions(d_valid_tasks)

    # Compute residuals (ground_truth - prior)
    def compute_residuals(task_ids, prior_preds):
        residuals = {}
        for task_id in task_ids:
            if task_id in prior_preds and task_id in items_df.index:
                gt = items_df.loc[task_id, "b"]
                pred = prior_preds[task_id]
                residuals[task_id] = gt - pred
        return residuals

    train_residuals = compute_residuals(d_train_tasks, prior_train_preds)
    valid_residuals = compute_residuals(d_valid_tasks, prior_valid_preds)

    print(f"\nResidual statistics (ground_truth - prior):")
    train_res_arr = np.array(list(train_residuals.values()))
    valid_res_arr = np.array(list(valid_residuals.values()))
    print(f"  D_train: mean={train_res_arr.mean():.3f}, std={train_res_arr.std():.3f}, range=[{train_res_arr.min():.3f}, {train_res_arr.max():.3f}]")
    print(f"  D_valid: mean={valid_res_arr.mean():.3f}, std={valid_res_arr.std():.3f}, range=[{valid_res_arr.min():.3f}, {valid_res_arr.max():.3f}]")

    # Load LLM judge features and compute predicted residuals
    print("\nLoading LLM judge features...")
    psi = np.array([psi_coefficients[name] for name in LLM_JUDGE_FEATURE_NAMES])

    def get_predicted_residual(task_id, agents, features_dir):
        """Get the residual predicted by LLM judge features."""
        features, detailed = load_llm_judge_features_detailed(task_id, agents, features_dir)
        if features is None:
            return None, None
        predicted_residual = float(np.dot(psi, features))
        return predicted_residual, detailed

    # Analyze train set
    print("\n" + "="*80)
    print("ANALYSIS: D_train tasks")
    print("="*80)

    train_analysis = []
    for task_id in d_train_tasks:
        if task_id not in train_residuals:
            continue
        actual_residual = train_residuals[task_id]
        pred_residual, details = get_predicted_residual(task_id, m1_agents, llm_judge_dir)

        if pred_residual is None:
            continue

        gt = items_df.loc[task_id, "b"]
        prior_pred = prior_train_preds[task_id]

        # Error without LLM judge (just prior)
        prior_error = abs(actual_residual)  # |gt - prior|

        # Error with LLM judge
        posterior_pred = prior_pred + pred_residual
        posterior_error = abs(gt - posterior_pred)

        # Did LLM judge help or hurt?
        improvement = prior_error - posterior_error

        train_analysis.append({
            "task_id": task_id,
            "ground_truth_b": gt,
            "prior_pred": prior_pred,
            "actual_residual": actual_residual,
            "predicted_residual": pred_residual,
            "posterior_pred": posterior_pred,
            "prior_error": prior_error,
            "posterior_error": posterior_error,
            "improvement": improvement,
            "helped": improvement > 0,
        })

    train_df = pd.DataFrame(train_analysis)

    # Summary statistics
    n_helped = train_df["helped"].sum()
    n_hurt = (~train_df["helped"]).sum()
    avg_improvement = train_df["improvement"].mean()

    print(f"\nTasks where LLM judge HELPED: {n_helped}/{len(train_df)} ({100*n_helped/len(train_df):.1f}%)")
    print(f"Tasks where LLM judge HURT:   {n_hurt}/{len(train_df)} ({100*n_hurt/len(train_df):.1f}%)")
    print(f"Average improvement: {avg_improvement:.4f}")

    # Correlation between actual and predicted residuals
    r, p = stats.pearsonr(train_df["actual_residual"], train_df["predicted_residual"])
    print(f"\nCorrelation between actual and predicted residuals: r={r:.4f}, p={p:.4f}")

    # Look at extreme cases
    print("\n" + "-"*80)
    print("TASKS WITH LARGE POSITIVE RESIDUALS (harder than prior expected)")
    print("-"*80)
    large_pos = train_df[train_df["actual_residual"] > 1.0].sort_values("actual_residual", ascending=False)
    for _, row in large_pos.head(10).iterrows():
        print(f"\n{row['task_id']}:")
        print(f"  Ground truth b: {row['ground_truth_b']:.3f}")
        print(f"  Prior prediction: {row['prior_pred']:.3f}")
        print(f"  Actual residual: {row['actual_residual']:.3f} (prior underestimated difficulty)")
        print(f"  Predicted residual: {row['predicted_residual']:.3f}")
        print(f"  {'HELPED' if row['helped'] else 'HURT'} by {abs(row['improvement']):.3f}")

    print("\n" + "-"*80)
    print("TASKS WITH LARGE NEGATIVE RESIDUALS (easier than prior expected)")
    print("-"*80)
    large_neg = train_df[train_df["actual_residual"] < -1.0].sort_values("actual_residual")
    for _, row in large_neg.head(10).iterrows():
        print(f"\n{row['task_id']}:")
        print(f"  Ground truth b: {row['ground_truth_b']:.3f}")
        print(f"  Prior prediction: {row['prior_pred']:.3f}")
        print(f"  Actual residual: {row['actual_residual']:.3f} (prior overestimated difficulty)")
        print(f"  Predicted residual: {row['predicted_residual']:.3f}")
        print(f"  {'HELPED' if row['helped'] else 'HURT'} by {abs(row['improvement']):.3f}")

    # Analyze the validation set
    print("\n" + "="*80)
    print("ANALYSIS: D_valid tasks (held-out)")
    print("="*80)

    valid_analysis = []
    for task_id in d_valid_tasks:
        if task_id not in valid_residuals:
            continue
        actual_residual = valid_residuals[task_id]
        pred_residual, details = get_predicted_residual(task_id, m2_agents, llm_judge_dir)

        if pred_residual is None:
            continue

        gt = items_df.loc[task_id, "b"]
        prior_pred = prior_valid_preds[task_id]

        prior_error = abs(actual_residual)
        posterior_pred = prior_pred + pred_residual
        posterior_error = abs(gt - posterior_pred)
        improvement = prior_error - posterior_error

        valid_analysis.append({
            "task_id": task_id,
            "ground_truth_b": gt,
            "prior_pred": prior_pred,
            "actual_residual": actual_residual,
            "predicted_residual": pred_residual,
            "posterior_pred": posterior_pred,
            "prior_error": prior_error,
            "posterior_error": posterior_error,
            "improvement": improvement,
            "helped": improvement > 0,
        })

    valid_df = pd.DataFrame(valid_analysis)

    if len(valid_df) > 0:
        n_helped = valid_df["helped"].sum()
        n_hurt = (~valid_df["helped"]).sum()
        avg_improvement = valid_df["improvement"].mean()

        print(f"\nTasks where LLM judge HELPED: {n_helped}/{len(valid_df)} ({100*n_helped/len(valid_df):.1f}%)")
        print(f"Tasks where LLM judge HURT:   {n_hurt}/{len(valid_df)} ({100*n_hurt/len(valid_df):.1f}%)")
        print(f"Average improvement: {avg_improvement:.4f}")

        r, p = stats.pearsonr(valid_df["actual_residual"], valid_df["predicted_residual"])
        print(f"\nCorrelation between actual and predicted residuals: r={r:.4f}, p={p:.4f}")

    # Analyze feature contributions
    print("\n" + "="*80)
    print("PSI COEFFICIENTS ANALYSIS")
    print("="*80)

    print("\nLearned psi coefficients (weights for residual prediction):")
    sorted_psi = sorted(psi_coefficients.items(), key=lambda x: abs(x[1]), reverse=True)
    for name, coef in sorted_psi:
        sign = "+" if coef > 0 else ""
        print(f"  {name}: {sign}{coef:.4f}")

    # Check what features look like for tasks where we're wrong
    print("\n" + "="*80)
    print("FEATURE ANALYSIS FOR TASKS WHERE WE'RE MOST WRONG")
    print("="*80)

    # Tasks where we hurt the most
    worst_tasks = train_df.sort_values("improvement").head(5)
    print("\nTasks where LLM judge HURT the most:")
    for _, row in worst_tasks.iterrows():
        print(f"\n{row['task_id']}:")
        print(f"  Actual residual: {row['actual_residual']:.3f}")
        print(f"  Predicted residual: {row['predicted_residual']:.3f}")
        print(f"  Error introduced: {-row['improvement']:.3f}")

        # Load features for this task
        features, detailed = load_llm_judge_features_detailed(row['task_id'], m1_agents, llm_judge_dir)
        if detailed:
            print(f"  Features from {len(detailed)} agents:")
            # Average features
            avg_features = {}
            for agent, feat in detailed.items():
                for k, v in feat.items():
                    if k not in avg_features:
                        avg_features[k] = []
                    avg_features[k].append(v)
            for k, vals in avg_features.items():
                print(f"    {k}: mean={np.mean(vals):.2f}, std={np.std(vals):.2f}")

    # Save detailed analysis
    output_path = ROOT / "chris_output/experiment_b/llm_judge_residual_analysis.json"
    analysis_output = {
        "train_summary": {
            "n_tasks": len(train_df),
            "n_helped": int(n_helped),
            "n_hurt": int(n_hurt),
            "avg_improvement": float(avg_improvement),
            "residual_correlation": float(r),
        },
        "train_tasks": train_df.to_dict(orient="records"),
        "valid_tasks": valid_df.to_dict(orient="records") if len(valid_df) > 0 else [],
        "psi_coefficients": psi_coefficients,
    }

    with open(output_path, "w") as f:
        json.dump(analysis_output, f, indent=2)
    print(f"\n\nDetailed analysis saved to: {output_path}")


if __name__ == "__main__":
    main()

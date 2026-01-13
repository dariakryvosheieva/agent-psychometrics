"""Main training and evaluation pipeline for Experiment B."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit  # sigmoid
from sklearn.metrics import roc_auc_score

# Add parent to path for imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_b.config import ExperimentConfig
from experiment_b.data_splits import create_experiment_split
from experiment_b.prior_model import PriorModel, EmbeddingPriorModel
from experiment_b.posterior_model import PosteriorModel


def load_responses(responses_path: Path) -> Dict[str, Dict[str, int]]:
    """Load response matrix from JSONL file.

    Handles two formats:
    1. Per-row format: {"subject_id": "agent", "item_id": "task", "response": 0|1}
    2. Nested format: {"subject_id": "agent", "responses": {"task1": 0, "task2": 1, ...}}

    Returns:
        Dict mapping agent_id -> {task_id -> 0|1}
    """
    responses = {}
    with open(responses_path) as f:
        for line in f:
            row = json.loads(line)
            agent_id = row["subject_id"]

            # Check format
            if "responses" in row:
                # Nested format
                responses[agent_id] = row["responses"]
            else:
                # Per-row format
                task_id = row["item_id"]
                response = row["response"]
                if agent_id not in responses:
                    responses[agent_id] = {}
                responses[agent_id][task_id] = response

    return responses


def compute_auc(
    predicted_difficulties: Dict[str, float],
    abilities: pd.DataFrame,
    responses: Dict[str, Dict[str, int]],
    task_ids: List[str],
    agent_ids: List[str],
) -> Dict:
    """Compute AUC for predicted difficulties.

    Uses IRT formula: P(success) = sigmoid(theta - beta)

    Args:
        predicted_difficulties: Dict mapping task_id to predicted difficulty
        abilities: DataFrame with index=agent_id, column 'theta'
        responses: Dict mapping agent_id -> {task_id -> 0|1}
        task_ids: Tasks to evaluate on
        agent_ids: Agents to use for evaluation

    Returns:
        Dict with 'auc', 'n_pairs', 'n_positive', 'n_negative'
    """
    y_true = []
    y_scores = []

    for task_id in task_ids:
        if task_id not in predicted_difficulties:
            continue

        beta_pred = predicted_difficulties[task_id]

        for agent_id in agent_ids:
            if agent_id not in responses:
                continue
            if task_id not in responses[agent_id]:
                continue
            if agent_id not in abilities.index:
                continue

            theta = float(abilities.loc[agent_id, "theta"])
            actual = responses[agent_id][task_id]
            prob = float(expit(theta - beta_pred))

            y_true.append(int(actual))
            y_scores.append(prob)

    if len(y_true) < 2:
        return {"error": "Insufficient data", "n_pairs": len(y_true)}

    if len(set(y_true)) < 2:
        return {"error": "Only one class in y_true", "n_pairs": len(y_true)}

    auc = roc_auc_score(y_true, y_scores)
    n_positive = sum(y_true)
    n_negative = len(y_true) - n_positive

    return {
        "auc": float(auc),
        "n_pairs": len(y_true),
        "n_positive": n_positive,
        "n_negative": n_negative,
    }


def evaluate_predictions(
    predictions: Dict[str, float],
    ground_truth: pd.Series,
) -> Dict[str, float]:
    """Evaluate prediction quality.

    Returns:
        Dict with pearson_r, p_value, mse, rmse, n
    """
    # Align predictions with ground truth
    common_tasks = set(predictions.keys()) & set(ground_truth.index)
    if len(common_tasks) < 3:
        return {"error": "Too few common tasks", "n": len(common_tasks)}

    pred_arr = np.array([predictions[t] for t in common_tasks])
    gt_arr = np.array([ground_truth[t] for t in common_tasks])

    r, p = stats.pearsonr(pred_arr, gt_arr)
    mse = np.mean((pred_arr - gt_arr) ** 2)

    return {
        "n": len(common_tasks),
        "pearson_r": float(r),
        "p_value": float(p),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
    }


def run_experiment(config: ExperimentConfig) -> Dict:
    """Run the full Experiment B pipeline.

    Args:
        config: Experiment configuration

    Returns:
        Dict with all results
    """
    print("=" * 60)
    print("EXPERIMENT B: POSTERIOR DIFFICULTY PREDICTION")
    print("=" * 60)
    print(f"Feature source: {config.feature_source}")
    print(f"Prior only: {config.prior_only}")

    # Resolve paths relative to ROOT
    items_path = ROOT / config.items_path
    responses_path = ROOT / config.responses_path
    trajectories_dir = ROOT / config.trajectories_dir
    lunette_features_dir = ROOT / config.lunette_features_dir
    llm_judge_features_dir = ROOT / config.llm_judge_features_dir
    llm_judge_v4_features_dir = ROOT / config.llm_judge_v4_features_dir
    llm_judge_v5_features_dir = ROOT / config.llm_judge_v5_features_dir
    llm_judge_v5_single_features_dir = ROOT / config.llm_judge_v5_single_features_dir
    output_dir = ROOT / config.output_dir

    # Load IRT difficulties and abilities
    print("\n1. Loading IRT parameters...")
    items_df = pd.read_csv(items_path, index_col=0)
    print(f"   Loaded {len(items_df)} tasks")
    print(f"   Difficulty range: [{items_df['b'].min():.2f}, {items_df['b'].max():.2f}]")

    # Load abilities from same directory as items
    abilities_path = items_path.parent / "abilities.csv"
    abilities_df = pd.read_csv(abilities_path, index_col=0)
    print(f"   Loaded {len(abilities_df)} agent abilities")

    # Load responses for AUC computation
    responses = load_responses(responses_path)
    print(f"   Loaded responses for {len(responses)} agents")

    # Create data splits
    print("\n2. Creating agent/task splits...")
    split = create_experiment_split(
        responses_path=responses_path,
        trajectories_dir=trajectories_dir,
        weak_threshold=config.weak_threshold,
        strong_min_improvement=config.strong_min_improvement,
        m1_fraction=config.m1_fraction,
        m2_fraction=config.m2_fraction,
    )
    print(f"   M1 agents: {len(split.m1_agents)}")
    print(f"   M2 agents: {len(split.m2_agents)}")
    print(f"   M3 agents: {len(split.m3_agents)}")
    print(f"   D_train tasks: {len(split.d_train_tasks)}")
    print(f"   D_valid tasks: {len(split.d_valid_tasks)}")

    if len(split.d_train_tasks) == 0:
        print("\n   WARNING: No training tasks found! Try adjusting weak_threshold or strong_min_improvement.")
        return {"error": "No training tasks"}

    # Train prior model (on ALL tasks)
    print("\n3. Training prior model...")
    print(f"   Prior source: {config.prior_source}")
    all_task_ids = list(items_df.index)
    all_difficulties = items_df["b"].values

    if config.prior_source == "embedding":
        if config.embeddings_path is None:
            raise ValueError("embeddings_path required when prior_source='embedding'")
        embeddings_path = ROOT / config.embeddings_path
        prior_model = EmbeddingPriorModel(embeddings_path, alpha=config.prior_alpha)
    else:
        prior_model = PriorModel(alpha=config.prior_alpha)
    prior_model.fit(all_task_ids, all_difficulties)

    # Evaluate prior on D_train
    prior_train_preds = prior_model.get_prior_predictions(split.d_train_tasks)
    train_gt = items_df.loc[split.d_train_tasks, "b"]
    prior_train_eval = evaluate_predictions(prior_train_preds, train_gt)
    print(f"   Prior on D_train: r={prior_train_eval.get('pearson_r', 'N/A'):.3f}, RMSE={prior_train_eval.get('rmse', 'N/A'):.3f}")

    # Compute AUC for prior on D_train (using M1 agents)
    prior_train_auc = compute_auc(prior_train_preds, abilities_df, responses, split.d_train_tasks, split.m1_agents)
    print(f"   Prior AUC on D_train (M1 agents): {prior_train_auc.get('auc', 'N/A'):.4f}" if 'auc' in prior_train_auc else f"   Prior AUC: {prior_train_auc.get('error', 'N/A')}")

    # Train posterior model on D_train using M1 trajectories
    print("\n4. Training posterior model...")
    train_difficulties = items_df.loc[split.d_train_tasks, "b"].values

    if config.prior_only:
        print("   Skipping posterior (prior_only mode)")
        posterior_model = None
    else:
        posterior_model = PosteriorModel(
            prior_model,
            alpha=config.posterior_alpha,
            feature_source=config.feature_source,
            lunette_features_dir=lunette_features_dir,
            llm_judge_features_dir=llm_judge_features_dir,
            llm_judge_v4_features_dir=llm_judge_v4_features_dir,
            llm_judge_v5_features_dir=llm_judge_v5_features_dir,
            llm_judge_v5_single_features_dir=llm_judge_v5_single_features_dir,
        )
        posterior_model.fit(
            task_ids=split.d_train_tasks,
            ground_truth_difficulties=train_difficulties,
            weak_agents=split.m1_agents,
            trajectories_dir=trajectories_dir,
        )

    # Evaluate posterior on D_train
    if posterior_model is not None:
        posterior_train_preds = posterior_model.predict(
            split.d_train_tasks, split.m1_agents, trajectories_dir
        )
        posterior_train_eval = evaluate_predictions(posterior_train_preds, train_gt)
        print(f"   Posterior on D_train: r={posterior_train_eval.get('pearson_r', 'N/A'):.3f}, RMSE={posterior_train_eval.get('rmse', 'N/A'):.3f}")

        # Compute AUC for posterior on D_train
        posterior_train_auc = compute_auc(posterior_train_preds, abilities_df, responses, split.d_train_tasks, split.m1_agents)
        print(f"   Posterior AUC on D_train: {posterior_train_auc.get('auc', 'N/A'):.4f}" if 'auc' in posterior_train_auc else f"   Posterior AUC: {posterior_train_auc.get('error', 'N/A')}")
    else:
        posterior_train_eval = {"skipped": True, "reason": "prior_only mode"}
        posterior_train_auc = {"skipped": True, "reason": "prior_only mode"}

    # Evaluate on D_valid
    print("\n5. Evaluating on D_valid...")
    if len(split.d_valid_tasks) == 0:
        print("   WARNING: No validation tasks found!")
        prior_valid_eval = {"error": "No validation tasks", "n": 0}
        prior_valid_auc = {"error": "No validation tasks"}
        posterior_valid_eval = {"error": "No validation tasks", "n": 0}
        posterior_valid_auc = {"error": "No validation tasks"}
    else:
        valid_gt = items_df.loc[split.d_valid_tasks, "b"]

        # Prior baseline on D_valid
        prior_valid_preds = prior_model.get_prior_predictions(split.d_valid_tasks)
        prior_valid_eval = evaluate_predictions(prior_valid_preds, valid_gt)
        print(f"   Prior on D_valid: r={prior_valid_eval.get('pearson_r', 'N/A'):.3f}, RMSE={prior_valid_eval.get('rmse', 'N/A'):.3f}")

        # Compute AUC for prior on D_valid (using M2 agents)
        prior_valid_auc = compute_auc(prior_valid_preds, abilities_df, responses, split.d_valid_tasks, split.m2_agents)
        print(f"   Prior AUC on D_valid (M2 agents): {prior_valid_auc.get('auc', 'N/A'):.4f}" if 'auc' in prior_valid_auc else f"   Prior AUC: {prior_valid_auc.get('error', 'N/A')}")

        # Posterior on D_valid (using M2 trajectories)
        if posterior_model is not None:
            posterior_valid_preds = posterior_model.predict(
                split.d_valid_tasks, split.m2_agents, trajectories_dir
            )
            posterior_valid_eval = evaluate_predictions(posterior_valid_preds, valid_gt)
            print(f"   Posterior on D_valid: r={posterior_valid_eval.get('pearson_r', 'N/A'):.3f}, RMSE={posterior_valid_eval.get('rmse', 'N/A'):.3f}")

            # Compute AUC for posterior on D_valid
            posterior_valid_auc = compute_auc(posterior_valid_preds, abilities_df, responses, split.d_valid_tasks, split.m2_agents)
            print(f"   Posterior AUC on D_valid: {posterior_valid_auc.get('auc', 'N/A'):.4f}" if 'auc' in posterior_valid_auc else f"   Posterior AUC: {posterior_valid_auc.get('error', 'N/A')}")
        else:
            posterior_valid_eval = {"skipped": True, "reason": "prior_only mode"}
            posterior_valid_auc = {"skipped": True, "reason": "prior_only mode"}

    # Compile results
    results = {
        "split": {
            "m1_agents": split.m1_agents,
            "m2_agents": split.m2_agents,
            "m3_agents": split.m3_agents,
            "d_train_tasks": split.d_train_tasks,
            "d_valid_tasks": split.d_valid_tasks,
        },
        "prior_train": prior_train_eval,
        "prior_train_auc": prior_train_auc,
        "posterior_train": posterior_train_eval,
        "posterior_train_auc": posterior_train_auc,
        "prior_valid": prior_valid_eval,
        "prior_valid_auc": prior_valid_auc,
        "posterior_valid": posterior_valid_eval,
        "posterior_valid_auc": posterior_valid_auc,
        "psi_coefficients": posterior_model.get_feature_importance() if posterior_model else {},
        "posterior_training_stats": posterior_model.get_training_stats() if posterior_model else {},
        "prior_coefficients": prior_model.get_feature_coefficients(),
        "config": config.to_dict(),
    }

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY (Primary metric: AUC)")
    print("=" * 60)

    def fmt_auc(auc_dict):
        if "auc" in auc_dict:
            return f"{auc_dict['auc']:.4f}"
        return auc_dict.get("error", "N/A")

    def fmt_rmse(eval_dict):
        if "rmse" in eval_dict:
            return f"{eval_dict['rmse']:.3f}"
        return eval_dict.get("error", "N/A")

    print(f"\nD_train ({prior_train_eval.get('n', 0)} tasks, {prior_train_auc.get('n_pairs', 0)} agent-task pairs):")
    print(f"  Prior:     AUC = {fmt_auc(prior_train_auc)}, RMSE = {fmt_rmse(prior_train_eval)}")
    if not posterior_train_eval.get("skipped"):
        print(f"  Posterior: AUC = {fmt_auc(posterior_train_auc)}, RMSE = {fmt_rmse(posterior_train_eval)}")
    else:
        print(f"  Posterior: (skipped - prior_only mode)")

    print(f"\nD_valid ({prior_valid_eval.get('n', 0)} tasks, {prior_valid_auc.get('n_pairs', 0)} agent-task pairs):")
    print(f"  Prior:     AUC = {fmt_auc(prior_valid_auc)}, RMSE = {fmt_rmse(prior_valid_eval)}")
    if not posterior_valid_eval.get("skipped"):
        print(f"  Posterior: AUC = {fmt_auc(posterior_valid_auc)}, RMSE = {fmt_rmse(posterior_valid_eval)}")
        if "auc" in posterior_valid_auc and "auc" in prior_valid_auc:
            auc_improvement = posterior_valid_auc["auc"] - prior_valid_auc["auc"]
            print(f"\n  AUC Improvement: {'+' if auc_improvement >= 0 else ''}{auc_improvement:.4f}")
    else:
        print(f"  Posterior: (skipped - prior_only mode)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run Experiment B")
    parser.add_argument(
        "--weak_threshold",
        type=float,
        default=0.2,
        help="Max pass rate for weak model group (default: 0.2)",
    )
    parser.add_argument(
        "--strong_min_improvement",
        type=float,
        default=0.1,
        help="Min improvement for strong model group (default: 0.1)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="chris_output/experiment_b",
        help="Output directory",
    )
    parser.add_argument(
        "--feature_source",
        type=str,
        choices=["simple", "lunette", "llm_judge", "llm_judge_v4", "llm_judge_v5", "llm_judge_v5_single"],
        default="simple",
        help="Feature source: 'simple', 'lunette', 'llm_judge', 'llm_judge_v4', 'llm_judge_v5', or 'llm_judge_v5_single'",
    )
    parser.add_argument(
        "--prior_source",
        type=str,
        choices=["heuristic", "embedding"],
        default="heuristic",
        help="Prior source: 'heuristic' (repo, text length) or 'embedding' (Daria's embeddings)",
    )
    parser.add_argument(
        "--embeddings_path",
        type=str,
        default=None,
        help="Path to embeddings .npz file (required if prior_source='embedding')",
    )
    parser.add_argument(
        "--prior_only",
        action="store_true",
        help="Run prior-only baseline (no trajectory correction)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show configuration without running",
    )
    args = parser.parse_args()

    config = ExperimentConfig(
        weak_threshold=args.weak_threshold,
        strong_min_improvement=args.strong_min_improvement,
        output_dir=Path(args.output_dir),
        feature_source=args.feature_source,
        prior_source=args.prior_source,
        embeddings_path=Path(args.embeddings_path) if args.embeddings_path else None,
        prior_only=args.prior_only,
    )

    if args.dry_run:
        print("DRY RUN - Configuration:")
        print(json.dumps(config.to_dict(), indent=2))
        return

    results = run_experiment(config)

    # Save results
    output_dir = ROOT / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "experiment_b_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

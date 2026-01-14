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

from experiment_b.config import ExperimentConfig, RegressionMode
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
    print(f"Regression mode: {config.regression_mode}")
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
    execution_features_dir = ROOT / config.execution_features_dir
    llm_judge_v6_features_dir = ROOT / config.llm_judge_v6_features_dir
    llm_judge_v7_features_dir = ROOT / config.llm_judge_v7_features_dir
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
        test_progression_features_dir = ROOT / config.test_progression_features_dir
        posterior_model = PosteriorModel(
            prior_model,
            alpha=config.posterior_alpha,
            feature_source=config.feature_source,
            regression_mode=config.regression_mode,
            lunette_features_dir=lunette_features_dir,
            llm_judge_features_dir=llm_judge_features_dir,
            llm_judge_v4_features_dir=llm_judge_v4_features_dir,
            llm_judge_v5_features_dir=llm_judge_v5_features_dir,
            llm_judge_v5_single_features_dir=llm_judge_v5_single_features_dir,
            execution_features_dir=execution_features_dir,
            llm_judge_v6_features_dir=llm_judge_v6_features_dir,
            llm_judge_v7_features_dir=llm_judge_v7_features_dir,
            test_progression_features_dir=test_progression_features_dir,
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

        # Posterior on D_valid
        # For LLM judge features, we use M1 agents since those are what we computed
        # For simple/lunette features, we use M2 agents which matches the original design
        if posterior_model is not None:
            if config.feature_source.startswith("llm_judge"):
                valid_agents_for_features = split.m1_agents
            else:
                valid_agents_for_features = split.m2_agents
            posterior_valid_preds = posterior_model.predict(
                split.d_valid_tasks, valid_agents_for_features, trajectories_dir
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


def run_all_regression_modes(config: ExperimentConfig) -> Dict:
    """Run experiment with all regression modes and compare results.

    Args:
        config: Base experiment configuration (regression_mode will be overridden)

    Returns:
        Dict with results for each mode and comparison summary
    """
    from dataclasses import replace

    modes: List[RegressionMode] = ["residual", "direct_with_prior", "direct_with_prior_features"]
    all_results = {}

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"RUNNING REGRESSION MODE: {mode}")
        print(f"{'='*60}")

        # Create config copy with this mode
        mode_config = replace(config, regression_mode=mode)
        try:
            results = run_experiment(mode_config)
            all_results[mode] = results
        except Exception as e:
            print(f"Error running mode {mode}: {e}")
            all_results[mode] = {"error": str(e)}

    # Generate comparison summary
    comparison = {
        "valid_auc": {},
        "valid_rmse": {},
        "train_auc": {},
        "train_rmse": {},
    }

    for mode, results in all_results.items():
        if "error" in results:
            continue
        comparison["valid_auc"][mode] = results.get("posterior_valid_auc", {}).get("auc")
        comparison["valid_rmse"][mode] = results.get("posterior_valid", {}).get("rmse")
        comparison["train_auc"][mode] = results.get("posterior_train_auc", {}).get("auc")
        comparison["train_rmse"][mode] = results.get("posterior_train", {}).get("rmse")

    # Find best mode by validation AUC
    valid_aucs = {k: v for k, v in comparison["valid_auc"].items() if v is not None}
    if valid_aucs:
        comparison["best_mode_by_auc"] = max(valid_aucs, key=valid_aucs.get)
        comparison["best_auc"] = valid_aucs[comparison["best_mode_by_auc"]]

        # Compute delta vs residual baseline
        residual_auc = comparison["valid_auc"].get("residual")
        if residual_auc is not None:
            comparison["delta_vs_residual"] = {
                mode: auc - residual_auc
                for mode, auc in valid_aucs.items()
            }

    # Print comparison table
    print("\n" + "=" * 80)
    print("REGRESSION MODE COMPARISON")
    print("=" * 80)
    print(f"{'Mode':<30} {'Train AUC':>12} {'Valid AUC':>12} {'ΔAUC vs residual':>18}")
    print("-" * 80)

    for mode in modes:
        if mode not in all_results or "error" in all_results[mode]:
            print(f"{mode:<30} {'ERROR':>12}")
            continue

        train_auc = comparison["train_auc"].get(mode)
        valid_auc = comparison["valid_auc"].get(mode)
        delta = comparison.get("delta_vs_residual", {}).get(mode)

        train_str = f"{train_auc:.4f}" if train_auc else "N/A"
        valid_str = f"{valid_auc:.4f}" if valid_auc else "N/A"
        delta_str = f"{delta:+.4f}" if delta is not None else "N/A"

        print(f"{mode:<30} {train_str:>12} {valid_str:>12} {delta_str:>18}")

    if "best_mode_by_auc" in comparison:
        print(f"\nBest mode: {comparison['best_mode_by_auc']} (AUC = {comparison['best_auc']:.4f})")

    return {
        "mode_results": all_results,
        "comparison": comparison,
        "config": config.to_dict(),
    }


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
        choices=["simple", "lunette", "llm_judge", "llm_judge_v4", "llm_judge_v5", "llm_judge_v5_single", "execution", "discoverability", "combined_v2", "llm_judge_v7", "mechanical_v7", "test_progression"],
        default="simple",
        help="Feature source: 'simple', 'execution' (mechanical), 'llm_judge_v7' (semantic), 'mechanical_v7' (both), 'test_progression', or legacy options",
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
        "--items_path",
        type=str,
        default=None,
        help="Path to IRT items.csv (overrides config default)",
    )
    parser.add_argument(
        "--regression_mode",
        type=str,
        choices=["residual", "direct_with_prior", "direct_with_prior_features"],
        default="residual",
        help="Regression mode: 'residual' (prior + correction), 'direct_with_prior' (prior as feature), 'direct_with_prior_features' (prior input features)",
    )
    parser.add_argument(
        "--compare_modes",
        action="store_true",
        help="Run all regression modes and compare results",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show configuration without running",
    )
    args = parser.parse_args()

    config_kwargs = dict(
        weak_threshold=args.weak_threshold,
        strong_min_improvement=args.strong_min_improvement,
        output_dir=Path(args.output_dir),
        feature_source=args.feature_source,
        prior_source=args.prior_source,
        embeddings_path=Path(args.embeddings_path) if args.embeddings_path else None,
        prior_only=args.prior_only,
        regression_mode=args.regression_mode,
    )
    if args.items_path:
        config_kwargs["items_path"] = Path(args.items_path)
    config = ExperimentConfig(**config_kwargs)

    if args.dry_run:
        print("DRY RUN - Configuration:")
        print(json.dumps(config.to_dict(), indent=2))
        return

    # Run experiment(s)
    output_dir = ROOT / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.compare_modes:
        # Run all regression modes and compare
        results = run_all_regression_modes(config)
        output_path = output_dir / f"experiment_b_mode_comparison_{config.feature_source}.json"
    else:
        # Single mode run
        results = run_experiment(config)
        output_path = output_dir / "experiment_b_results.json"

    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

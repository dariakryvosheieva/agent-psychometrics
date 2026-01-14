"""Main training and evaluation pipeline for Experiment A."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# Add parent to path for imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_a.config import ExperimentAConfig
from experiment_a.data_loader import load_experiment_data, ExperimentAData
from experiment_a.difficulty_predictor import (
    EmbeddingPredictor,
    EmbeddingSimilarityPredictor,
    MLEEmbeddingPredictor,
    ConstantPredictor,
    GroundTruthPredictor,
    LunettePredictor,
    LLMJudgePredictor,
)
from experiment_a.irt_evaluation import compute_auc, compute_difficulty_prediction_metrics
from experiment_a.baselines import agent_only_baseline, task_only_baseline


def run_experiment_a(config: ExperimentAConfig) -> Dict[str, Any]:
    """Run the full Experiment A pipeline.

    Args:
        config: Experiment configuration

    Returns:
        Dict with all results
    """
    print("=" * 60)
    print("EXPERIMENT A: PRIOR VALIDATION (IRT AUC)")
    print("=" * 60)

    # Resolve paths relative to ROOT
    abilities_path = ROOT / config.abilities_path
    items_path = ROOT / config.items_path
    responses_path = ROOT / config.responses_path
    output_dir = ROOT / config.output_dir

    # 1. Load data
    print("\n1. Loading data...")
    data = load_experiment_data(
        abilities_path=abilities_path,
        items_path=items_path,
        responses_path=responses_path,
        test_fraction=config.test_fraction,
        split_seed=config.split_seed,
    )
    print(f"   Agents: {data.n_agents}")
    print(f"   Tasks: {data.n_tasks}")
    print(f"   Train tasks: {data.n_train_tasks}")
    print(f"   Test tasks: {data.n_test_tasks}")

    # 2. Get ground truth difficulties for training
    train_b = data.items.loc[data.train_tasks, "b"].values

    # 3. Initialize results dict
    results: Dict[str, Any] = {
        "config": config.to_dict(),
        "data_summary": {
            "n_agents": data.n_agents,
            "n_tasks_total": data.n_tasks,
            "n_train_tasks": data.n_train_tasks,
            "n_test_tasks": data.n_test_tasks,
        },
    }

    # 4. Oracle baseline (ground truth difficulties)
    print("\n2. Computing oracle baseline (ground truth b)...")
    oracle_predictor = GroundTruthPredictor(data.items)
    oracle_preds = oracle_predictor.predict(data.test_tasks)
    oracle_result = compute_auc(
        oracle_preds, data.abilities, data.responses, data.test_tasks
    )
    print(f"   Oracle AUC: {oracle_result.get('auc', 'N/A'):.4f}")
    results["oracle"] = oracle_result

    # 5. Constant baseline (mean difficulty)
    print("\n3. Computing constant baseline (mean b)...")
    const_predictor = ConstantPredictor()
    const_predictor.fit(data.train_tasks, train_b)
    const_preds = const_predictor.predict(data.test_tasks)
    const_result = compute_auc(
        const_preds, data.abilities, data.responses, data.test_tasks
    )
    print(f"   Constant AUC: {const_result.get('auc', 'N/A'):.4f}")
    results["constant_baseline"] = const_result

    # 6. Agent-only baseline
    print("\n4. Computing agent-only baseline...")
    agent_baseline = agent_only_baseline(
        data.abilities, data.responses, data.test_tasks
    )
    print(f"   Agent-only AUC: {agent_baseline.get('auc', 'N/A'):.4f}")
    results["agent_only_baseline"] = agent_baseline

    # 7. Task-only baseline
    print("\n5. Computing task-only baseline...")
    task_baseline = task_only_baseline(
        data.responses, data.train_tasks, data.test_tasks
    )
    print(f"   Task-only AUC: {task_baseline.get('auc', 'N/A'):.4f}")
    results["task_only_baseline"] = task_baseline

    # 8. Embedding predictor (if embeddings provided)
    if config.embeddings_path is not None:
        embeddings_path = ROOT / config.embeddings_path
        if embeddings_path.exists():
            print(f"\n6. Training embedding predictor...")
            print(f"   Loading embeddings from: {embeddings_path}")

            try:
                embedding_predictor = EmbeddingPredictor(
                    embeddings_path=embeddings_path,
                    ridge_alpha=config.ridge_alpha,
                )
                print(f"   Embeddings loaded: {embedding_predictor.n_embeddings} tasks")
                print(f"   Embedding dim: {embedding_predictor.embedding_dim}")

                embedding_predictor.fit(data.train_tasks, train_b)
                embedding_preds = embedding_predictor.predict(data.test_tasks)

                # IRT-based AUC
                embedding_result = compute_auc(
                    embedding_preds, data.abilities, data.responses, data.test_tasks
                )
                print(f"   Embedding AUC: {embedding_result.get('auc', 'N/A'):.4f}")

                # Difficulty prediction metrics
                diff_metrics = compute_difficulty_prediction_metrics(
                    embedding_preds, data.items, data.test_tasks
                )
                print(f"   Difficulty prediction Pearson r: {diff_metrics.get('pearson_r', 'N/A'):.4f}")

                results["embedding_predictor"] = {
                    "auc_result": embedding_result,
                    "difficulty_metrics": diff_metrics,
                    "embeddings_path": str(embeddings_path),
                    "ridge_alpha": config.ridge_alpha,
                    "n_embeddings": embedding_predictor.n_embeddings,
                    "embedding_dim": embedding_predictor.embedding_dim,
                }

                # Store predictions for analysis
                results["difficulty_predictions"] = {
                    "embedding": embedding_preds,
                    "ground_truth": {t: float(data.items.loc[t, "b"]) for t in data.test_tasks},
                }

            except Exception as e:
                print(f"   Error loading embeddings: {e}")
                results["embedding_predictor"] = {"error": str(e)}
        else:
            print(f"\n6. Embeddings file not found: {embeddings_path}")
            results["embedding_predictor"] = {"error": f"File not found: {embeddings_path}"}
    else:
        print("\n6. No embeddings path provided, skipping embedding predictor")
        results["embedding_predictor"] = {"error": "No embeddings_path provided"}

    # 8b. Embedding Similarity predictor (if embeddings provided)
    if config.embeddings_path is not None:
        embeddings_path = ROOT / config.embeddings_path
        if embeddings_path.exists():
            print(f"\n6b. Training embedding similarity predictor...")

            try:
                sim_predictor = EmbeddingSimilarityPredictor(
                    embeddings_path=embeddings_path,
                    ridge_alpha=config.embedding_similarity_ridge_alpha,
                )
                print(f"   Embeddings loaded: {sim_predictor.n_embeddings} tasks")
                print(f"   Embedding dim: {sim_predictor.embedding_dim}")
                print(f"   Features: {len(sim_predictor.feature_names)} similarity statistics")

                sim_predictor.fit(data.train_tasks, train_b)
                sim_preds = sim_predictor.predict(data.test_tasks)

                # IRT-based AUC
                sim_result = compute_auc(
                    sim_preds, data.abilities, data.responses, data.test_tasks
                )
                print(f"   Embedding Similarity AUC: {sim_result.get('auc', 'N/A'):.4f}")

                # Difficulty prediction metrics
                diff_metrics = compute_difficulty_prediction_metrics(
                    sim_preds, data.items, data.test_tasks
                )
                print(f"   Difficulty prediction Pearson r: {diff_metrics.get('pearson_r', 'N/A'):.4f}")

                # Print feature coefficients for interpretability
                sim_predictor.print_feature_coefficients()

                results["embedding_similarity_predictor"] = {
                    "auc_result": sim_result,
                    "difficulty_metrics": diff_metrics,
                    "embeddings_path": str(embeddings_path),
                    "ridge_alpha": config.embedding_similarity_ridge_alpha,
                    "n_embeddings": sim_predictor.n_embeddings,
                    "embedding_dim": sim_predictor.embedding_dim,
                    "n_train_tasks": sim_predictor.n_train_tasks,
                    "feature_names": sim_predictor.feature_names,
                    "feature_coefficients": sim_predictor.feature_coefficients,
                }

                # Store predictions for analysis
                if "difficulty_predictions" not in results:
                    results["difficulty_predictions"] = {
                        "ground_truth": {t: float(data.items.loc[t, "b"]) for t in data.test_tasks},
                    }
                results["difficulty_predictions"]["embedding_similarity"] = sim_preds

            except Exception as e:
                print(f"   Error with embedding similarity predictor: {e}")
                import traceback
                traceback.print_exc()
                results["embedding_similarity_predictor"] = {"error": str(e)}
        else:
            results["embedding_similarity_predictor"] = {"error": f"File not found: {embeddings_path}"}
    else:
        results["embedding_similarity_predictor"] = {"error": "No embeddings_path provided"}

    # 8c. MLE Embedding predictor (Truong et al. 2025 approach)
    if config.use_mle_embedding and config.embeddings_path is not None:
        embeddings_path = ROOT / config.embeddings_path
        if embeddings_path.exists():
            print(f"\n6c. Training MLE embedding predictor (Truong et al. 2025)...")

            try:
                mle_predictor = MLEEmbeddingPredictor(
                    embeddings_path=embeddings_path,
                    lr=config.mle_lr,
                    max_iter=config.mle_max_iter,
                    l2_lambda=config.mle_l2_lambda,
                    use_mc_abilities=config.mle_use_mc_abilities,
                    n_mc_samples=config.mle_n_mc_samples,
                    verbose=True,
                )
                print(f"   Embeddings loaded: {mle_predictor.n_embeddings} tasks")
                print(f"   Embedding dim: {mle_predictor.embedding_dim}")

                # MLE fit requires abilities and responses
                mle_predictor.fit(
                    data.train_tasks,
                    train_b,
                    abilities=data.abilities,
                    responses=data.responses,
                )
                mle_preds = mle_predictor.predict(data.test_tasks)

                # IRT-based AUC
                mle_result = compute_auc(
                    mle_preds, data.abilities, data.responses, data.test_tasks
                )
                print(f"   MLE Embedding AUC: {mle_result.get('auc', 'N/A'):.4f}")

                # Difficulty prediction metrics
                diff_metrics = compute_difficulty_prediction_metrics(
                    mle_preds, data.items, data.test_tasks
                )
                print(f"   Difficulty prediction Pearson r: {diff_metrics.get('pearson_r', 'N/A'):.4f}")

                results["mle_embedding_predictor"] = {
                    "auc_result": mle_result,
                    "difficulty_metrics": diff_metrics,
                    "embeddings_path": str(embeddings_path),
                    "mle_lr": config.mle_lr,
                    "mle_max_iter": config.mle_max_iter,
                    "mle_l2_lambda": config.mle_l2_lambda,
                    "mle_use_mc_abilities": config.mle_use_mc_abilities,
                    "mle_n_mc_samples": config.mle_n_mc_samples if config.mle_use_mc_abilities else None,
                    "n_embeddings": mle_predictor.n_embeddings,
                    "embedding_dim": mle_predictor.embedding_dim,
                    "final_loss": mle_predictor.training_loss_history[-1] if mle_predictor.training_loss_history else None,
                }

                # Store predictions for analysis
                if "difficulty_predictions" not in results:
                    results["difficulty_predictions"] = {
                        "ground_truth": {t: float(data.items.loc[t, "b"]) for t in data.test_tasks},
                    }
                results["difficulty_predictions"]["mle_embedding"] = mle_preds

            except Exception as e:
                print(f"   Error with MLE embedding predictor: {e}")
                import traceback
                traceback.print_exc()
                results["mle_embedding_predictor"] = {"error": str(e)}
        else:
            results["mle_embedding_predictor"] = {"error": f"File not found: {embeddings_path}"}
    elif config.use_mle_embedding:
        results["mle_embedding_predictor"] = {"error": "No embeddings_path provided"}
    else:
        results["mle_embedding_predictor"] = {"skipped": "use_mle_embedding is False"}

    # 9. Lunette predictor (if features provided)
    if config.lunette_features_path is not None:
        lunette_path = ROOT / config.lunette_features_path
        if lunette_path.exists():
            print(f"\n7. Training Lunette predictor...")
            print(f"   Loading features from: {lunette_path}")

            try:
                lunette_predictor = LunettePredictor(
                    features_path=lunette_path,
                    ridge_alpha=config.lunette_ridge_alpha,
                    feature_selection=config.lunette_feature_selection,
                    max_features=config.lunette_max_features,
                )
                print(f"   Features loaded: {lunette_predictor.n_tasks} tasks")
                print(f"   Feature count: {lunette_predictor.n_features}")

                lunette_predictor.fit(data.train_tasks, train_b)

                # Print selected features
                lunette_predictor.print_selected_features()

                lunette_preds = lunette_predictor.predict(data.test_tasks)

                # IRT-based AUC
                lunette_result = compute_auc(
                    lunette_preds, data.abilities, data.responses, data.test_tasks
                )
                print(f"\n   Lunette AUC: {lunette_result.get('auc', 'N/A'):.4f}")

                # Difficulty prediction metrics
                diff_metrics = compute_difficulty_prediction_metrics(
                    lunette_preds, data.items, data.test_tasks
                )
                print(f"   Difficulty prediction Pearson r: {diff_metrics.get('pearson_r', 'N/A'):.4f}")

                results["lunette_predictor"] = {
                    "auc_result": lunette_result,
                    "difficulty_metrics": diff_metrics,
                    "features_path": str(lunette_path),
                    "ridge_alpha": config.lunette_ridge_alpha,
                    "feature_selection": config.lunette_feature_selection,
                    "max_features": config.lunette_max_features,
                    "n_tasks": lunette_predictor.n_tasks,
                    "n_features": lunette_predictor.n_features,
                    "selected_features": lunette_predictor.selected_features,
                    "feature_coefficients": lunette_predictor.feature_coefficients,
                }

                # Store predictions for analysis
                if "difficulty_predictions" not in results:
                    results["difficulty_predictions"] = {
                        "ground_truth": {t: float(data.items.loc[t, "b"]) for t in data.test_tasks},
                    }
                results["difficulty_predictions"]["lunette"] = lunette_preds

            except Exception as e:
                print(f"   Error loading Lunette features: {e}")
                import traceback
                traceback.print_exc()
                results["lunette_predictor"] = {"error": str(e)}
        else:
            print(f"\n7. Lunette features file not found: {lunette_path}")
            results["lunette_predictor"] = {"error": f"File not found: {lunette_path}"}
    else:
        print("\n7. No Lunette features path provided, skipping Lunette predictor")
        results["lunette_predictor"] = {"error": "No lunette_features_path provided"}

    # 10. LLM Judge predictor (if features provided)
    if config.llm_judge_features_path is not None:
        llm_judge_path = ROOT / config.llm_judge_features_path
        if llm_judge_path.exists():
            print(f"\n8. Training LLM Judge predictor...")
            print(f"   Loading features from: {llm_judge_path}")

            try:
                llm_judge_predictor = LLMJudgePredictor(
                    features_path=llm_judge_path,
                    ridge_alpha=config.llm_judge_ridge_alpha,
                    max_features=config.llm_judge_max_features,
                )
                print(f"   Features loaded: {llm_judge_predictor.n_tasks} tasks")
                print(f"   Feature count: {llm_judge_predictor.n_features}")

                llm_judge_predictor.fit(data.train_tasks, train_b)

                # Print selected features
                llm_judge_predictor.print_selected_features()

                llm_judge_preds = llm_judge_predictor.predict(data.test_tasks)

                # IRT-based AUC
                llm_judge_result = compute_auc(
                    llm_judge_preds, data.abilities, data.responses, data.test_tasks
                )
                print(f"\n   LLM Judge AUC: {llm_judge_result.get('auc', 'N/A'):.4f}")

                # Difficulty prediction metrics
                diff_metrics = compute_difficulty_prediction_metrics(
                    llm_judge_preds, data.items, data.test_tasks
                )
                print(f"   Difficulty prediction Pearson r: {diff_metrics.get('pearson_r', 'N/A'):.4f}")

                results["llm_judge_predictor"] = {
                    "auc_result": llm_judge_result,
                    "difficulty_metrics": diff_metrics,
                    "features_path": str(llm_judge_path),
                    "ridge_alpha": config.llm_judge_ridge_alpha,
                    "max_features": config.llm_judge_max_features,
                    "n_tasks": llm_judge_predictor.n_tasks,
                    "n_features": llm_judge_predictor.n_features,
                    "selected_features": llm_judge_predictor.selected_features,
                    "feature_coefficients": llm_judge_predictor.feature_coefficients,
                }

                # Store predictions for analysis
                if "difficulty_predictions" not in results:
                    results["difficulty_predictions"] = {
                        "ground_truth": {t: float(data.items.loc[t, "b"]) for t in data.test_tasks},
                    }
                results["difficulty_predictions"]["llm_judge"] = llm_judge_preds

            except Exception as e:
                print(f"   Error loading LLM Judge features: {e}")
                import traceback
                traceback.print_exc()
                results["llm_judge_predictor"] = {"error": str(e)}
        else:
            print(f"\n8. LLM Judge features file not found: {llm_judge_path}")
            results["llm_judge_predictor"] = {"error": f"File not found: {llm_judge_path}"}
    else:
        print("\n8. No LLM Judge features path provided, skipping LLM Judge predictor")
        results["llm_judge_predictor"] = {"error": "No llm_judge_features_path provided"}

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nTest set: {data.n_test_tasks} tasks")
    print(f"\n{'Method':<30} {'AUC':>10}")
    print("-" * 42)

    predictors_with_auc_result = {
        "embedding_predictor", "embedding_similarity_predictor",
        "mle_embedding_predictor", "lunette_predictor", "llm_judge_predictor"
    }

    for name, key in [
        ("Oracle (true b)", "oracle"),
        ("Embedding predictor", "embedding_predictor"),
        ("Embedding (MLE)", "mle_embedding_predictor"),
        ("Embedding Similarity", "embedding_similarity_predictor"),
        ("LLM Judge predictor", "llm_judge_predictor"),
        ("Lunette predictor", "lunette_predictor"),
        ("Constant (mean b)", "constant_baseline"),
        ("Agent-only", "agent_only_baseline"),
        ("Task-only", "task_only_baseline"),
    ]:
        result = results.get(key, {})
        if key in predictors_with_auc_result and "auc_result" in result:
            auc = result["auc_result"].get("auc")
        else:
            auc = result.get("auc")

        if auc is not None:
            print(f"{name:<30} {auc:>10.4f}")
        elif "error" in result:
            print(f"{name:<30} {'ERROR':>10}")
        else:
            print(f"{name:<30} {'N/A':>10}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run Experiment A: Prior Validation (IRT AUC)")
    parser.add_argument(
        "--test_fraction",
        type=float,
        default=0.2,
        help="Fraction of tasks to hold out for testing (default: 0.2)",
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=0,
        help="Random seed for train/test split (default: 0)",
    )
    parser.add_argument(
        "--embeddings_path",
        type=str,
        default=None,
        help="Path to pre-computed embeddings .npz file",
    )
    parser.add_argument(
        "--ridge_alpha",
        type=float,
        default=10000.0,
        help="Ridge regression alpha (default: 10000.0)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="chris_output/experiment_a",
        help="Output directory (default: chris_output/experiment_a)",
    )
    # Lunette predictor arguments
    parser.add_argument(
        "--lunette_features_path",
        type=str,
        default=None,
        help="Path to Lunette features CSV file",
    )
    parser.add_argument(
        "--lunette_ridge_alpha",
        type=float,
        default=1.0,
        help="Ridge alpha for Lunette predictor (default: 1.0)",
    )
    parser.add_argument(
        "--lunette_feature_selection",
        type=str,
        default="lasso_cv",
        choices=["lasso_cv", "select_k_best"],
        help="Feature selection method for Lunette (default: lasso_cv)",
    )
    parser.add_argument(
        "--lunette_max_features",
        type=int,
        default=10,
        help="Max features to select for Lunette (default: 10)",
    )
    # LLM Judge predictor arguments
    parser.add_argument(
        "--llm_judge_features_path",
        type=str,
        default=None,
        help="Path to LLM judge features CSV file",
    )
    parser.add_argument(
        "--llm_judge_ridge_alpha",
        type=float,
        default=1.0,
        help="Ridge alpha for LLM Judge predictor (default: 1.0)",
    )
    parser.add_argument(
        "--llm_judge_max_features",
        type=int,
        default=None,
        help="Max features to select for LLM Judge (default: None = all 9)",
    )
    parser.add_argument(
        "--embedding_similarity_ridge_alpha",
        type=float,
        default=1.0,
        help="Ridge alpha for embedding similarity predictor (default: 1.0)",
    )
    # MLE Embedding predictor arguments (Truong et al. 2025)
    parser.add_argument(
        "--use_mle_embedding",
        action="store_true",
        help="Enable MLE embedding predictor (Truong et al. 2025 approach)",
    )
    parser.add_argument(
        "--mle_lr",
        type=float,
        default=0.1,
        help="Learning rate for MLE L-BFGS optimizer (default: 0.1)",
    )
    parser.add_argument(
        "--mle_max_iter",
        type=int,
        default=100,
        help="Max iterations for MLE training (default: 100)",
    )
    parser.add_argument(
        "--mle_l2_lambda",
        type=float,
        default=0.15,
        help="L2 regularization for MLE weights (default: 0.15, tuned)",
    )
    parser.add_argument(
        "--mle_use_mc_abilities",
        action="store_true",
        help="Use MC marginalization over abilities instead of fixed θ values",
    )
    parser.add_argument(
        "--mle_n_mc_samples",
        type=int,
        default=100,
        help="Number of MC samples for ability marginalization (default: 100)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show configuration without running",
    )
    args = parser.parse_args()

    config = ExperimentAConfig(
        test_fraction=args.test_fraction,
        split_seed=args.split_seed,
        embeddings_path=Path(args.embeddings_path) if args.embeddings_path else None,
        ridge_alpha=args.ridge_alpha,
        output_dir=Path(args.output_dir),
        # Lunette config
        lunette_features_path=Path(args.lunette_features_path) if args.lunette_features_path else None,
        lunette_ridge_alpha=args.lunette_ridge_alpha,
        lunette_feature_selection=args.lunette_feature_selection,
        lunette_max_features=args.lunette_max_features,
        # LLM Judge config
        llm_judge_features_path=Path(args.llm_judge_features_path) if args.llm_judge_features_path else None,
        llm_judge_ridge_alpha=args.llm_judge_ridge_alpha,
        llm_judge_max_features=args.llm_judge_max_features,
        # Embedding Similarity config
        embedding_similarity_ridge_alpha=args.embedding_similarity_ridge_alpha,
        # MLE Embedding config (Truong et al. 2025)
        use_mle_embedding=args.use_mle_embedding,
        mle_lr=args.mle_lr,
        mle_max_iter=args.mle_max_iter,
        mle_l2_lambda=args.mle_l2_lambda,
        mle_use_mc_abilities=args.mle_use_mc_abilities,
        mle_n_mc_samples=args.mle_n_mc_samples,
    )

    if args.dry_run:
        print("DRY RUN - Configuration:")
        print(json.dumps(config.to_dict(), indent=2))
        return

    results = run_experiment_a(config)

    # Save results
    output_dir = ROOT / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "experiment_a_results.json"

    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    results = convert_numpy(results)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

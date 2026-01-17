"""Main training and evaluation pipeline for Experiment A on TerminalBench."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Add parent to path for imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_a_terminalbench.baselines import (
    agent_only_baseline_binomial,
    constant_baseline_binomial,
    random_baseline_binomial,
    task_only_baseline_binomial,
    verify_random_baseline_sanity,
)
from experiment_a_terminalbench.config import TerminalBenchConfig
from experiment_a_terminalbench.data_loader import load_terminalbench_data
from experiment_a_terminalbench.irt_evaluation import (
    compute_binomial_auc,
    compute_difficulty_prediction_metrics,
)

# Reuse difficulty predictors from experiment_a
from experiment_a.difficulty_predictor import (
    ConstantPredictor,
    EmbeddingPredictor,
    GroundTruthPredictor,
    LLMJudgePredictor,
)


def run_experiment_a_terminalbench(config: TerminalBenchConfig) -> Dict[str, Any]:
    """Run the full Experiment A pipeline on TerminalBench.

    Args:
        config: Experiment configuration

    Returns:
        Dict with all results
    """
    print("=" * 60)
    print("EXPERIMENT A: PRIOR VALIDATION (IRT AUC) - TERMINALBENCH")
    print("=" * 60)

    # Resolve paths relative to ROOT
    abilities_path = ROOT / config.abilities_path
    items_path = ROOT / config.items_path
    responses_path = ROOT / config.responses_path
    repo_path = ROOT / config.repo_path
    output_dir = ROOT / config.output_dir

    # 1. Load data
    print("\n1. Loading data...")
    data = load_terminalbench_data(
        abilities_path=abilities_path,
        items_path=items_path,
        responses_path=responses_path,
        repo_path=repo_path,
        test_fraction=config.test_fraction,
        split_seed=config.split_seed,
    )
    print(f"   Agents: {data.n_agents}")
    print(f"   Tasks: {data.n_tasks}")
    print(f"   Train tasks: {data.n_train_tasks}")
    print(f"   Test tasks: {data.n_test_tasks}")
    print(f"   Tasks with data loaded: {len(data.task_data)}")

    # 2. Get ground truth difficulties for training (from train-only IRT, no leakage)
    train_b = data.train_items.loc[data.train_tasks, "b"].values

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
    oracle_result = compute_binomial_auc(
        oracle_preds, data.abilities, data.responses, data.test_tasks
    )
    print(f"   Oracle AUC: {oracle_result.get('auc', 'N/A'):.4f}")
    results["oracle"] = oracle_result

    # 5. Constant baseline (mean difficulty from training)
    print("\n3. Computing constant baseline (mean b from training)...")
    const_result = constant_baseline_binomial(
        data.items, data.abilities, data.responses,
        data.train_tasks, data.test_tasks
    )
    print(f"   Constant AUC: {const_result.get('auc', 'N/A'):.4f}")
    print(f"   Mean train difficulty: {const_result.get('mean_train_difficulty', 'N/A'):.4f}")
    results["constant_baseline"] = const_result

    # 6. Agent-only baseline
    print("\n4. Computing agent-only baseline...")
    agent_result = agent_only_baseline_binomial(
        data.abilities, data.responses, data.train_tasks, data.test_tasks
    )
    print(f"   Agent-only AUC: {agent_result.get('auc', 'N/A'):.4f}")
    results["agent_only_baseline"] = agent_result

    # 7. Task-only baseline
    print("\n5. Computing task-only baseline...")
    task_result = task_only_baseline_binomial(
        data.responses, data.train_tasks, data.test_tasks
    )
    print(f"   Task-only AUC: {task_result.get('auc', 'N/A'):.4f}")
    print(f"   Mean train rate: {task_result.get('mean_train_rate', 'N/A'):.4f}")
    results["task_only_baseline"] = task_result

    # 8. Random baseline sanity check
    print("\n6. Verifying random baseline (sanity check)...")
    random_sanity = verify_random_baseline_sanity(
        data.responses, data.test_tasks, n_trials=100
    )
    print(f"   Random AUC (mean over 100 trials): {random_sanity.get('mean_auc', 'N/A'):.4f}")
    print(f"   Sanity check passed: {random_sanity.get('passed', False)}")
    results["random_baseline_sanity"] = random_sanity

    # 9. Embedding predictor (if embeddings provided)
    if config.embeddings_path is not None:
        embeddings_path = ROOT / config.embeddings_path
        if embeddings_path.exists():
            print(f"\n7. Training embedding predictor...")
            print(f"   Loading embeddings from: {embeddings_path}")
            print(f"   Ridge alphas for CV: {config.ridge_alphas}")

            try:
                from sklearn.linear_model import RidgeCV
                from sklearn.preprocessing import StandardScaler
                from sklearn.pipeline import Pipeline

                # Load embeddings
                emb_data = np.load(embeddings_path, allow_pickle=True)
                task_ids_emb = [str(x) for x in emb_data["task_ids"].tolist()]
                X_emb = emb_data["X"].astype(np.float32)
                embeddings = {tid: X_emb[i] for i, tid in enumerate(task_ids_emb)}

                print(f"   Embeddings loaded: {len(embeddings)} tasks")
                print(f"   Embedding dim: {X_emb.shape[1]}")

                # Get train embeddings and targets
                train_available = [t for t in data.train_tasks if t in embeddings]
                X_train = np.stack([embeddings[t] for t in train_available])
                y_train = np.array([train_b[data.train_tasks.index(t)] for t in train_available])

                # Fit RidgeCV to find best alpha
                model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("ridge", RidgeCV(alphas=config.ridge_alphas, cv=5)),
                ])
                model.fit(X_train, y_train)
                best_alpha = model.named_steps["ridge"].alpha_
                print(f"   Best alpha (CV): {best_alpha}")

                # Predict on test tasks
                test_available = [t for t in data.test_tasks if t in embeddings]
                X_test = np.stack([embeddings[t] for t in test_available])
                preds = model.predict(X_test)
                embedding_preds = dict(zip(test_available, preds.tolist()))

                # IRT-based AUC
                embedding_result = compute_binomial_auc(
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
                    "best_alpha": float(best_alpha),
                    "n_embeddings": len(embeddings),
                    "embedding_dim": int(X_emb.shape[1]),
                    "n_train_available": len(train_available),
                    "n_test_available": len(test_available),
                }

                # Store predictions for analysis
                results["difficulty_predictions"] = {
                    "embedding": embedding_preds,
                    "ground_truth": {t: float(data.items.loc[t, "b"]) for t in data.test_tasks},
                }

            except Exception as e:
                print(f"   Error with embedding predictor: {e}")
                import traceback
                traceback.print_exc()
                results["embedding_predictor"] = {"error": str(e)}
        else:
            print(f"\n7. Embeddings file not found: {embeddings_path}")
            results["embedding_predictor"] = {"error": f"File not found: {embeddings_path}"}
    else:
        print("\n7. No embeddings path provided, skipping embedding predictor")
        results["embedding_predictor"] = {"error": "No embeddings_path provided"}

    # 10. LLM Judge predictor (if features provided)
    if config.llm_judge_features_path is not None:
        llm_judge_path = ROOT / config.llm_judge_features_path
        if llm_judge_path.exists():
            print(f"\n8. Training LLM Judge predictor...")
            print(f"   Loading features from: {llm_judge_path}")
            print(f"   Ridge alphas for CV: {config.llm_judge_ridge_alphas}")

            try:
                from sklearn.linear_model import RidgeCV, LassoCV
                from sklearn.preprocessing import StandardScaler
                import pandas as pd

                # Load features
                features_df = pd.read_csv(llm_judge_path)
                if "task_id" in features_df.columns:
                    features_df = features_df.set_index("task_id")
                elif "instance_id" in features_df.columns:
                    features_df = features_df.set_index("instance_id")

                # Get feature columns (only numeric semantic features)
                from experiment_a_terminalbench.llm_judge_prompt import LLM_JUDGE_SEMANTIC_FEATURES
                feature_cols = [c for c in LLM_JUDGE_SEMANTIC_FEATURES if c in features_df.columns]
                print(f"   Features loaded: {len(features_df)} tasks")
                print(f"   Feature count: {len(feature_cols)}")

                # Get train features and targets
                train_available = [t for t in data.train_tasks if t in features_df.index]
                X_train = features_df.loc[train_available, feature_cols].values.astype(np.float32)
                X_train = np.nan_to_num(X_train, nan=0.0)
                y_train = np.array([train_b[data.train_tasks.index(t)] for t in train_available])

                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)

                # Lasso for feature selection
                lasso = LassoCV(cv=5, max_iter=10000, random_state=42)
                lasso.fit(X_train_scaled, y_train)

                # Get selected features
                coef_abs = np.abs(lasso.coef_)
                if config.llm_judge_max_features:
                    top_k_idx = np.argsort(coef_abs)[-config.llm_judge_max_features:]
                    selected_mask = np.zeros(len(feature_cols), dtype=bool)
                    selected_mask[top_k_idx] = True
                else:
                    selected_mask = coef_abs > 1e-6
                    if selected_mask.sum() == 0:
                        selected_mask = np.ones(len(feature_cols), dtype=bool)

                selected_features = [feature_cols[i] for i in range(len(feature_cols)) if selected_mask[i]]
                print(f"   Selected features: {selected_features}")

                # Fit RidgeCV on selected features
                X_train_selected = X_train_scaled[:, selected_mask]
                ridge = RidgeCV(alphas=config.llm_judge_ridge_alphas, cv=5)
                ridge.fit(X_train_selected, y_train)
                best_alpha = ridge.alpha_
                print(f"   Best alpha (CV): {best_alpha}")

                # Predict on test tasks
                test_available = [t for t in data.test_tasks if t in features_df.index]
                X_test = features_df.loc[test_available, feature_cols].values.astype(np.float32)
                X_test = np.nan_to_num(X_test, nan=0.0)
                X_test_scaled = scaler.transform(X_test)
                X_test_selected = X_test_scaled[:, selected_mask]
                preds = ridge.predict(X_test_selected)
                llm_judge_preds = dict(zip(test_available, preds.tolist()))

                # IRT-based AUC
                llm_judge_result = compute_binomial_auc(
                    llm_judge_preds, data.abilities, data.responses, data.test_tasks
                )
                print(f"   LLM Judge AUC: {llm_judge_result.get('auc', 'N/A'):.4f}")

                # Difficulty prediction metrics
                diff_metrics = compute_difficulty_prediction_metrics(
                    llm_judge_preds, data.items, data.test_tasks
                )
                print(f"   Difficulty prediction Pearson r: {diff_metrics.get('pearson_r', 'N/A'):.4f}")

                # Feature coefficients
                feature_coefficients = dict(zip(selected_features, ridge.coef_.tolist()))
                print(f"\n   Feature coefficients:")
                for name, coef in sorted(feature_coefficients.items(), key=lambda x: abs(x[1]), reverse=True):
                    sign = "+" if coef >= 0 else ""
                    print(f"     {name:30s}: {sign}{coef:.4f}")

                results["llm_judge_predictor"] = {
                    "auc_result": llm_judge_result,
                    "difficulty_metrics": diff_metrics,
                    "features_path": str(llm_judge_path),
                    "best_alpha": float(best_alpha),
                    "n_tasks": len(features_df),
                    "n_features": len(feature_cols),
                    "selected_features": selected_features,
                    "feature_coefficients": feature_coefficients,
                    "n_train_available": len(train_available),
                    "n_test_available": len(test_available),
                }

                # Store predictions for analysis
                if "difficulty_predictions" not in results:
                    results["difficulty_predictions"] = {
                        "ground_truth": {t: float(data.items.loc[t, "b"]) for t in data.test_tasks},
                    }
                results["difficulty_predictions"]["llm_judge"] = llm_judge_preds

            except Exception as e:
                print(f"   Error with LLM Judge predictor: {e}")
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

    predictors_with_auc_result = {"embedding_predictor", "llm_judge_predictor"}

    for name, key in [
        ("Oracle (true b)", "oracle"),
        ("Embedding predictor", "embedding_predictor"),
        ("LLM Judge predictor", "llm_judge_predictor"),
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
    parser = argparse.ArgumentParser(
        description="Run Experiment A: Prior Validation (IRT AUC) on TerminalBench"
    )
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
        "--llm_judge_features_path",
        type=str,
        default=None,
        help="Path to LLM judge features CSV file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="chris_output/experiment_a_terminalbench",
        help="Output directory (default: chris_output/experiment_a_terminalbench)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show configuration without running",
    )
    args = parser.parse_args()

    config = TerminalBenchConfig(
        test_fraction=args.test_fraction,
        split_seed=args.split_seed,
        embeddings_path=Path(args.embeddings_path) if args.embeddings_path else None,
        llm_judge_features_path=Path(args.llm_judge_features_path) if args.llm_judge_features_path else None,
        output_dir=Path(args.output_dir),
    )

    if args.dry_run:
        print("DRY RUN - Configuration:")
        print(json.dumps(config.to_dict(), indent=2))
        return

    results = run_experiment_a_terminalbench(config)

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

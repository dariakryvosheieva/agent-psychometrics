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
    ConstantPredictor,
    GroundTruthPredictor,
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

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nTest set: {data.n_test_tasks} tasks")
    print(f"\n{'Method':<30} {'AUC':>10}")
    print("-" * 42)

    for name, key in [
        ("Oracle (true b)", "oracle"),
        ("Embedding predictor", "embedding_predictor"),
        ("Constant (mean b)", "constant_baseline"),
        ("Agent-only", "agent_only_baseline"),
        ("Task-only", "task_only_baseline"),
    ]:
        result = results.get(key, {})
        if key == "embedding_predictor" and "auc_result" in result:
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

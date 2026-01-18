"""Main training and evaluation pipeline for Experiment A on TerminalBench."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Add parent to path for imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_a_terminalbench.config import TerminalBenchConfig
from experiment_a_terminalbench.data_loader import load_task_data_from_repo
from experiment_a.difficulty_predictor import (
    EmbeddingPredictor,
    ConstantPredictor,
    GroundTruthPredictor,
    LLMJudgePredictor,
)
from experiment_a_common import (
    load_dataset,
    compute_auc,
    run_evaluation_pipeline,
    PredictorConfig,
    agent_only_baseline,
    convert_numpy,
)


def build_predictor_configs(config: TerminalBenchConfig) -> List[PredictorConfig]:
    """Build list of predictor configurations from experiment config."""
    configs = []

    # Constant baseline (mean difficulty)
    configs.append(PredictorConfig(
        predictor_class=ConstantPredictor,
        name="constant_baseline",
        display_name="Constant (mean b)",
    ))

    # Embedding predictor
    if config.embeddings_path is not None:
        embeddings_path = ROOT / config.embeddings_path
        if embeddings_path.exists():
            configs.append(PredictorConfig(
                predictor_class=EmbeddingPredictor,
                name="embedding_predictor",
                display_name="Embedding",
                kwargs={
                    "embeddings_path": embeddings_path,
                    "ridge_alphas": config.ridge_alphas,  # Use CV to find best alpha
                },
            ))

    # LLM Judge predictor
    # TerminalBench uses different feature columns than SWE-bench
    TERMINALBENCH_LLM_JUDGE_FEATURES = [
        "solution_in_instruction",
        "task_clarity",
        "solution_size",
        "domain_knowledge_required",
        "task_complexity",
        "logical_reasoning_required",
        "atypicality",
        "tooling_complexity",
    ]
    if config.llm_judge_features_path is not None:
        llm_judge_path = ROOT / config.llm_judge_features_path
        if llm_judge_path.exists():
            configs.append(PredictorConfig(
                predictor_class=LLMJudgePredictor,
                name="llm_judge_predictor",
                display_name="LLM Judge",
                kwargs={
                    "features_path": llm_judge_path,
                    "ridge_alphas": config.llm_judge_ridge_alphas,  # Use CV to find best alpha
                    "max_features": config.llm_judge_max_features,
                    "feature_cols": TERMINALBENCH_LLM_JUDGE_FEATURES,
                },
            ))

    return configs


def run_experiment_a_terminalbench(config: TerminalBenchConfig) -> Dict[str, Any]:
    """Run the full Experiment A pipeline on TerminalBench.

    Args:
        config: Experiment configuration

    Returns:
        Dict with all results
    """
    print("=" * 60)
    print("EXPERIMENT A: PRIOR VALIDATION (IRT AUC) - TerminalBench")
    print("=" * 60)

    # Resolve paths relative to ROOT
    abilities_path = ROOT / config.abilities_path
    items_path = ROOT / config.items_path
    responses_path = ROOT / config.responses_path
    repo_path = ROOT / config.repo_path
    output_dir = ROOT / config.output_dir
    irt_cache_dir = ROOT / "chris_output" / "experiment_a_terminalbench" / "irt_splits"

    # Define metadata loader for task data
    def metadata_loader(task_ids: List[str]) -> Dict[str, Any]:
        return {"task_data": load_task_data_from_repo(task_ids, repo_path)}

    # 1. Load data using common loader
    print("\n1. Loading data...")
    data = load_dataset(
        abilities_path=abilities_path,
        items_path=items_path,
        responses_path=responses_path,
        test_fraction=config.test_fraction,
        split_seed=config.split_seed,
        is_binomial=True,  # TerminalBench uses binomial responses
        irt_cache_dir=irt_cache_dir,
        metadata_loader=metadata_loader,
    )
    print(f"   Agents: {data.n_agents}")
    print(f"   Tasks: {data.n_tasks}")
    print(f"   Train tasks: {data.n_train_tasks}")
    print(f"   Test tasks: {data.n_test_tasks}")
    print(f"   Tasks with metadata: {len(data.metadata.get('task_data', {}))}")

    # 2. Initialize results dict
    results: Dict[str, Any] = {
        "config": config.to_dict(),
        "data_summary": {
            "n_agents": data.n_agents,
            "n_tasks_total": data.n_tasks,
            "n_train_tasks": data.n_train_tasks,
            "n_test_tasks": data.n_test_tasks,
        },
    }

    # 3. Oracle baseline (handled specially - needs full_items)
    print("\n2. Computing oracle baseline (ground truth b from full IRT)...")
    oracle_predictor = GroundTruthPredictor(data.full_items)
    oracle_preds = oracle_predictor.predict(data.test_tasks)
    oracle_result = compute_auc(data, oracle_preds, use_full_abilities=True)
    print(f"   Oracle AUC: {oracle_result.get('auc', 'N/A'):.4f}")
    results["oracle"] = oracle_result

    # 4. Build predictor configs and run evaluation pipeline
    predictor_configs = build_predictor_configs(config)
    pipeline_results = run_evaluation_pipeline(data, predictor_configs, verbose=True)
    results.update(pipeline_results)

    # 5. Agent-only baseline (uses common implementation)
    print("\nComputing agent-only baseline...")
    agent_result = agent_only_baseline(data)
    print(f"   Agent-only AUC: {agent_result.get('auc', 'N/A'):.4f}")
    results["agent_only_baseline"] = agent_result

    # 6. Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nTest set: {data.n_test_tasks} tasks")
    print(f"\n{'Method':<30} {'AUC':>10}")
    print("-" * 42)

    # Define display order
    display_order = [
        ("Oracle (true b)", "oracle"),
        ("Embedding", "embedding_predictor"),
        ("LLM Judge", "llm_judge_predictor"),
        ("Constant (mean b)", "constant_baseline"),
        ("Agent-only", "agent_only_baseline"),
    ]

    for name, key in display_order:
        result = results.get(key, {})
        # Handle nested auc_result structure
        if "auc_result" in result:
            auc = result["auc_result"].get("auc")
        else:
            auc = result.get("auc")

        if auc is not None:
            print(f"{name:<30} {auc:>10.4f}")
        elif "error" in result:
            print(f"{name:<30} {'ERROR':>10}")
        elif "skipped" in result:
            continue
        elif key not in results:
            continue
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

    results = convert_numpy(results)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

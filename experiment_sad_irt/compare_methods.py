#!/usr/bin/env python3
"""Compare all SAD-IRT methods for frontier task difficulty prediction.

This script compares:
1. Baseline IRT (pre-frontier agents only, no trajectories)
2. SAD-IRT runs (from experiment tracker)
3. Embedding + Ridge (Experiment A style predictor)
4. LLM Judge + Ridge (Experiment A style predictor)

All methods are evaluated by Spearman correlation with oracle IRT
difficulties on frontier tasks only.

Usage:
    python -m experiment_sad_irt.compare_methods
    python -m experiment_sad_irt.compare_methods --sad_irt_csv chris_output/sad_irt_experiments.csv
    python -m experiment_sad_irt.compare_methods --output_csv chris_output/method_comparison.csv
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Import from experiment_sad_irt
from experiment_sad_irt.data_splits import (
    get_all_agents_from_responses,
    identify_frontier_tasks,
    split_agents_by_cutoff,
)
from experiment_sad_irt.evaluate import compute_frontier_difficulty_metrics

# Import from experiment_a
from experiment_a.difficulty_predictor import (
    DifficultyPredictorBase,
    EmbeddingPredictor,
    LLMJudgePredictor,
)


def compute_baseline_spearman(
    baseline_items: pd.DataFrame,
    oracle_items: pd.DataFrame,
    frontier_task_ids: List[str],
) -> Dict[str, float]:
    """Compute Spearman rho between baseline IRT and oracle IRT for frontier tasks."""
    baseline_dict = baseline_items["b"].to_dict()
    oracle_dict = oracle_items["b"].to_dict()
    return compute_frontier_difficulty_metrics(baseline_dict, oracle_dict, frontier_task_ids)


def evaluate_predictor(
    predictor: DifficultyPredictorBase,
    baseline_items: pd.DataFrame,
    oracle_items: pd.DataFrame,
    frontier_task_ids: List[str],
    train_task_ids: List[str],
    sanity_check: bool = True,
) -> Dict[str, float]:
    """Train a difficulty predictor on non-frontier tasks, evaluate on frontier tasks.

    Args:
        predictor: DifficultyPredictorBase instance (already initialized with data path)
        baseline_items: DataFrame with 'b' column (training targets)
        oracle_items: DataFrame with 'b' column (evaluation targets)
        frontier_task_ids: List of frontier task IDs (evaluation)
        train_task_ids: List of training task IDs
        sanity_check: If True, also compute correlation on training tasks

    Returns:
        Dict with correlation metrics
    """
    # Get training data
    train_tasks_available = [t for t in train_task_ids if t in baseline_items.index]
    ground_truth_b = baseline_items.loc[train_tasks_available, "b"].values

    # Fit predictor
    predictor.fit(train_tasks_available, ground_truth_b)

    # Sanity check: evaluate on training tasks (should be high)
    if sanity_check:
        train_predictions = predictor.predict(train_tasks_available)
        baseline_dict = baseline_items["b"].to_dict()
        train_metrics = compute_frontier_difficulty_metrics(
            train_predictions, baseline_dict, train_tasks_available
        )
        print(f"    [Sanity check] Train set Spearman rho: {train_metrics['frontier_spearman_rho']:.4f}")

    # Predict for frontier tasks
    predictions = predictor.predict(frontier_task_ids)

    # Compute correlation with oracle
    oracle_dict = oracle_items["b"].to_dict()
    return compute_frontier_difficulty_metrics(predictions, oracle_dict, frontier_task_ids)


def load_sad_irt_experiments(csv_path: Path) -> pd.DataFrame:
    """Load SAD-IRT experiment tracker results."""
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def print_comparison_table(
    results: Dict[str, Dict],
    frontier_task_count: int,
    pre_frontier_count: int,
    post_frontier_count: int,
) -> None:
    """Print formatted comparison table."""
    print("=" * 80)
    print("SAD-IRT METHOD COMPARISON")
    print("=" * 80)
    print()
    print("Frontier Task Definition:")
    print("  - Pre-frontier pass rate <= 10%")
    print("  - Post-frontier pass rate > 10%")
    print("  - Cutoff date: 20250807 (gpt-5-mini release)")
    print()
    print("Data Summary:")
    print(f"  - Pre-frontier agents: {pre_frontier_count}")
    print(f"  - Post-frontier agents: {post_frontier_count}")
    print(f"  - Frontier tasks: {frontier_task_count}")
    print()
    print("=" * 80)
    print(f"COMPARISON TABLE (Spearman rho with Oracle IRT on {frontier_task_count} Frontier Tasks)")
    print("=" * 80)
    print()
    print(f"{'Method':<50} {'Spearman rho':>12} {'p-value':>10} {'n':>5}")
    print("-" * 80)

    # Sort by Spearman rho (descending), handling NaN
    def sort_key(item):
        rho = item[1].get("frontier_spearman_rho", float("-inf"))
        return float("-inf") if np.isnan(rho) else rho

    sorted_methods = sorted(results.items(), key=sort_key, reverse=True)

    for method, metrics in sorted_methods:
        rho = metrics.get("frontier_spearman_rho", float("nan"))
        p = metrics.get("frontier_spearman_p", float("nan"))
        n = metrics.get("num_frontier_tasks", 0)

        if np.isnan(rho):
            print(f"{method:<50} {'N/A':>12} {'N/A':>10} {n:>5}")
        else:
            p_str = f"{p:.4f}" if p >= 0.0001 else "<0.0001"
            print(f"{method:<50} {rho:>12.4f} {p_str:>10} {n:>5}")

    print()


def save_results_csv(results: Dict[str, Dict], output_path: Path) -> None:
    """Save results to CSV."""
    rows = []
    for method, metrics in results.items():
        rows.append({
            "method": method,
            "spearman_rho": metrics.get("frontier_spearman_rho"),
            "spearman_p": metrics.get("frontier_spearman_p"),
            "pearson_r": metrics.get("frontier_pearson_r"),
            "pearson_p": metrics.get("frontier_pearson_p"),
            "n_tasks": metrics.get("num_frontier_tasks"),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("spearman_rho", ascending=False)
    df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare SAD-IRT methods for frontier task difficulty prediction"
    )
    parser.add_argument(
        "--responses_path",
        type=Path,
        default=Path("clean_data/swebench_verified/swebench_verified_20251120_full.jsonl"),
        help="Path to response matrix JSONL",
    )
    parser.add_argument(
        "--baseline_irt_path",
        type=Path,
        default=Path("chris_output/sad_irt/baseline_irt/items.csv"),
        help="Path to baseline IRT items CSV (pre-frontier only)",
    )
    parser.add_argument(
        "--oracle_irt_path",
        type=Path,
        default=Path("clean_data/swebench_verified_20251120_full/1d/items.csv"),
        help="Path to oracle IRT items CSV (all agents)",
    )
    parser.add_argument(
        "--embeddings_path",
        type=Path,
        default=Path(
            "out/prior_qwen3vl8b/embeddings__Qwen__Qwen3-VL-8B-Instruct__pool-lasttoken__"
            "qs-sol-instr__qs_sol_instr_b7008f2d__idnorm_instance-v1__"
            "princeton-nlp_SWE-bench_Verified__test__n500__maxlen8192__seed0.npz"
        ),
        help="Path to embeddings .npz file",
    )
    parser.add_argument(
        "--llm_judge_path",
        type=Path,
        default=Path("chris_output/experiment_a/llm_judge_features/llm_judge_features.csv"),
        help="Path to LLM judge features CSV",
    )
    parser.add_argument(
        "--sad_irt_csv",
        type=Path,
        default=Path("chris_output/sad_irt_experiments.csv"),
        help="Path to SAD-IRT experiments CSV",
    )
    parser.add_argument(
        "--cutoff_date",
        type=str,
        default="20250807",
        help="Frontier cutoff date (YYYYMMDD)",
    )
    parser.add_argument(
        "--pre_threshold",
        type=float,
        default=0.1,
        help="Max pre-frontier pass rate for frontier tasks",
    )
    parser.add_argument(
        "--post_threshold",
        type=float,
        default=0.1,
        help="Min post-frontier pass rate for frontier tasks",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Optional path to save results CSV",
    )
    args = parser.parse_args()

    # Validate required files exist
    required_files = [
        (args.responses_path, "Response matrix"),
        (args.baseline_irt_path, "Baseline IRT"),
        (args.oracle_irt_path, "Oracle IRT"),
    ]
    for path, name in required_files:
        if not path.exists():
            print(f"Error: {name} not found: {path}")
            sys.exit(1)

    # Load IRT models
    print("Loading IRT models...")
    baseline_items = pd.read_csv(args.baseline_irt_path, index_col=0)
    oracle_items = pd.read_csv(args.oracle_irt_path, index_col=0)
    print(f"  Baseline IRT: {len(baseline_items)} tasks")
    print(f"  Oracle IRT: {len(oracle_items)} tasks")

    # Identify frontier tasks
    print("\nIdentifying frontier tasks...")
    all_agents = get_all_agents_from_responses(args.responses_path)
    pre_frontier, post_frontier = split_agents_by_cutoff(all_agents, args.cutoff_date)
    print(f"  Pre-frontier agents (< {args.cutoff_date}): {len(pre_frontier)}")
    print(f"  Post-frontier agents (>= {args.cutoff_date}): {len(post_frontier)}")

    frontier_task_ids = identify_frontier_tasks(
        args.responses_path,
        pre_frontier,
        post_frontier,
        args.pre_threshold,
        args.post_threshold,
    )
    print(f"  Frontier tasks: {len(frontier_task_ids)}")

    # Non-frontier tasks for training predictors
    all_task_ids = list(baseline_items.index)
    train_task_ids = [t for t in all_task_ids if t not in frontier_task_ids]
    print(f"  Training tasks (non-frontier): {len(train_task_ids)}")

    # Collect results
    results = {}

    # 1. Baseline IRT
    print("\nEvaluating Baseline IRT...")
    results["Baseline IRT (pre-frontier only)"] = compute_baseline_spearman(
        baseline_items, oracle_items, frontier_task_ids
    )
    print(f"  Spearman rho: {results['Baseline IRT (pre-frontier only)']['frontier_spearman_rho']:.4f}")

    # 2. SAD-IRT runs (if CSV exists)
    if args.sad_irt_csv.exists():
        print(f"\nLoading SAD-IRT experiments from {args.sad_irt_csv}...")
        sad_irt_df = load_sad_irt_experiments(args.sad_irt_csv)
        print(f"  Found {len(sad_irt_df)} SAD-IRT runs")

        for _, row in sad_irt_df.iterrows():
            output_dir = row.get("output_dir", "unknown")
            dir_name = Path(output_dir).name if output_dir else "unknown"
            method_name = f"SAD-IRT ({dir_name[:30]}...)" if len(dir_name) > 30 else f"SAD-IRT ({dir_name})"

            results[method_name] = {
                "frontier_spearman_rho": row.get("best_spearman_rho", float("nan")),
                "frontier_spearman_p": row.get("final_spearman_p", float("nan")),
                "frontier_pearson_r": float("nan"),
                "frontier_pearson_p": float("nan"),
                "num_frontier_tasks": row.get("num_frontier_tasks", len(frontier_task_ids)),
            }
    else:
        print(f"\nSAD-IRT experiments CSV not found: {args.sad_irt_csv}")
        print("  To include SAD-IRT results, copy from cluster:")
        print(f"  scp <user>@engaging-submit.mit.edu:~/model_irt/{args.sad_irt_csv} {args.sad_irt_csv}")

    # 3. Embedding predictor
    if args.embeddings_path.exists():
        print("\nEvaluating Embedding + Ridge predictor...")
        try:
            predictor = EmbeddingPredictor(embeddings_path=args.embeddings_path)
            results["Embedding + Ridge"] = evaluate_predictor(
                predictor, baseline_items, oracle_items, frontier_task_ids, train_task_ids
            )
            print(f"  Spearman rho: {results['Embedding + Ridge']['frontier_spearman_rho']:.4f}")
        except Exception as e:
            print(f"  Error: {e}")
            results["Embedding + Ridge"] = {
                "frontier_spearman_rho": float("nan"),
                "frontier_spearman_p": float("nan"),
                "num_frontier_tasks": 0,
            }
    else:
        print(f"\nEmbeddings not found: {args.embeddings_path}")

    # 4. LLM Judge predictor
    if args.llm_judge_path.exists():
        print("\nEvaluating LLM Judge + Lasso/Ridge predictor...")
        try:
            predictor = LLMJudgePredictor(features_path=args.llm_judge_path)
            results["LLM Judge + Lasso/Ridge"] = evaluate_predictor(
                predictor, baseline_items, oracle_items, frontier_task_ids, train_task_ids
            )
            print(f"  Spearman rho: {results['LLM Judge + Lasso/Ridge']['frontier_spearman_rho']:.4f}")
        except Exception as e:
            print(f"  Error: {e}")
            results["LLM Judge + Lasso/Ridge"] = {
                "frontier_spearman_rho": float("nan"),
                "frontier_spearman_p": float("nan"),
                "num_frontier_tasks": 0,
            }
    else:
        print(f"\nLLM Judge features not found: {args.llm_judge_path}")

    # Print comparison table
    print()
    print_comparison_table(
        results,
        len(frontier_task_ids),
        len(pre_frontier),
        len(post_frontier),
    )

    # Save to CSV if requested
    if args.output_csv:
        save_results_csv(results, args.output_csv)


if __name__ == "__main__":
    main()

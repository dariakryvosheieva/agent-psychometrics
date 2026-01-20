#!/usr/bin/env python3
"""Compare methods for frontier task difficulty prediction.

This script compares:
1. Oracle (upper bound): True IRT difficulties
2. Baseline IRT: Train IRT on pre-frontier agents only
3. Embedding + Ridge: Task embeddings with Ridge regression
4. LLM Judge + Ridge: LLM-extracted semantic features with Ridge
5. SAD-IRT (optional): From experiment_sad_irt extracted beta values

Methods are evaluated by:
- Spearman correlation with oracle IRT difficulties on frontier tasks
- ROC-AUC on frontier tasks using oracle abilities and aligned difficulties

The AUC metric requires aligning predicted difficulties to the oracle scale using
an affine transformation fitted on "nontrivial" anchor tasks (10-90% pass rate in
both agent groups). This alignment uses oracle information and is ONLY for evaluation.

Usage:
    python -m experiment_b.compare_methods
    python -m experiment_b.compare_methods --embeddings_path path/to/embeddings.npz
    python -m experiment_b.compare_methods --output_csv chris_output/experiment_b_results.csv
    python -m experiment_b.compare_methods --alignment_method affine
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from experiment_b.data_splits import (
    get_all_agents_from_responses,
    identify_frontier_tasks,
    identify_nontrivial_tasks,
    split_agents_by_cutoff,
)
from experiment_b.evaluate import (
    compute_frontier_difficulty_metrics,
    compute_scale_offset,
    shift_to_oracle_scale,
    compute_frontier_auc,
    load_responses_dict,
)

# Import predictors from experiment_a
from experiment_a.difficulty_predictor import (
    DifficultyPredictorBase,
    EmbeddingPredictor,
    LLMJudgePredictor,
)


def compute_method_metrics(
    predicted_beta: Dict[str, float],
    oracle_items: pd.DataFrame,
    oracle_abilities: pd.DataFrame,
    responses: Dict[str, Dict[str, int]],
    frontier_task_ids: List[str],
    anchor_task_ids: List[str],
    eval_agents: List[str],
    alignment_method: str = "affine",
) -> Dict[str, Any]:
    """Compute all metrics for a difficulty prediction method.

    Args:
        predicted_beta: Dict mapping task_id -> predicted difficulty
        oracle_items: DataFrame with 'b' column (oracle difficulties)
        oracle_abilities: DataFrame with 'theta' column (oracle abilities)
        responses: Response matrix as nested dict
        frontier_task_ids: Tasks to evaluate Spearman/AUC on
        anchor_task_ids: Tasks for fitting the alignment transformation
        eval_agents: Agents to use for AUC computation (post-frontier)
        alignment_method: "constant" or "affine"

    Returns:
        Dict with spearman, pearson, auc, and alignment info
    """
    oracle_beta = oracle_items["b"].to_dict()

    # 1. Spearman correlation (no alignment needed - rank-based)
    spearman_metrics = compute_frontier_difficulty_metrics(
        predicted_beta, oracle_beta, frontier_task_ids
    )

    # 2. Compute alignment parameters using anchor tasks
    alignment_params = compute_scale_offset(
        predicted_beta, oracle_beta, anchor_task_ids, method=alignment_method
    )

    # 3. Shift predictions to oracle scale
    shifted_beta = shift_to_oracle_scale(predicted_beta, alignment_params)

    # 4. Compute AUC on frontier tasks using post-frontier agents
    auc_metrics = compute_frontier_auc(
        oracle_abilities, shifted_beta, responses, frontier_task_ids, eval_agents
    )

    return {
        **spearman_metrics,
        "auc": auc_metrics.get("auc"),
        "auc_n_pairs": auc_metrics.get("n_pairs"),
        "auc_n_positive": auc_metrics.get("n_positive"),
        "auc_n_negative": auc_metrics.get("n_negative"),
        "alignment_method": alignment_method,
        "alignment_params": alignment_params,
    }


def evaluate_predictor(
    predictor: DifficultyPredictorBase,
    baseline_items: pd.DataFrame,
    oracle_items: pd.DataFrame,
    oracle_abilities: pd.DataFrame,
    responses: Dict[str, Dict[str, int]],
    frontier_task_ids: List[str],
    train_task_ids: List[str],
    anchor_task_ids: List[str],
    eval_agents: List[str],
    alignment_method: str = "affine",
    sanity_check: bool = True,
) -> Dict[str, Any]:
    """Train a difficulty predictor on non-frontier tasks, evaluate with full metrics.

    Args:
        predictor: DifficultyPredictorBase instance (already initialized with data path)
        baseline_items: DataFrame with 'b' column (training targets)
        oracle_items: DataFrame with 'b' column (oracle difficulties)
        oracle_abilities: DataFrame with 'theta' column (oracle abilities)
        responses: Response matrix as nested dict
        frontier_task_ids: List of frontier task IDs (evaluation)
        train_task_ids: List of training task IDs
        anchor_task_ids: List of anchor task IDs (for scale alignment)
        eval_agents: Agents to use for AUC (post-frontier)
        alignment_method: "constant" or "affine"
        sanity_check: If True, print train set correlation

    Returns:
        Dict with spearman, pearson, auc metrics
    """
    # Get training data
    train_tasks_available = [t for t in train_task_ids if t in baseline_items.index]
    ground_truth_b = baseline_items.loc[train_tasks_available, "b"].values

    # Fit predictor
    predictor.fit(train_tasks_available, ground_truth_b)

    # Sanity check: evaluate on training tasks
    if sanity_check:
        train_predictions = predictor.predict(train_tasks_available)
        baseline_dict = baseline_items["b"].to_dict()
        train_metrics = compute_frontier_difficulty_metrics(
            train_predictions, baseline_dict, train_tasks_available
        )
        print(f"    [Sanity check] Train set Spearman rho: {train_metrics['frontier_spearman_rho']:.4f}")

    # Predict for all relevant tasks (frontier + anchor for alignment)
    all_tasks = list(set(frontier_task_ids + anchor_task_ids))
    predictions = predictor.predict(all_tasks)

    # Compute full metrics
    return compute_method_metrics(
        predicted_beta=predictions,
        oracle_items=oracle_items,
        oracle_abilities=oracle_abilities,
        responses=responses,
        frontier_task_ids=frontier_task_ids,
        anchor_task_ids=anchor_task_ids,
        eval_agents=eval_agents,
        alignment_method=alignment_method,
    )


def print_comparison_table(
    results: Dict[str, Dict],
    frontier_task_count: int,
    pre_frontier_count: int,
    post_frontier_count: int,
    anchor_task_count: int = 0,
    alignment_method: str = "affine",
    verbose: bool = False,
) -> None:
    """Print formatted comparison table."""
    print("=" * 90)
    print("EXPERIMENT B: FRONTIER TASK DIFFICULTY PREDICTION")
    print("=" * 90)
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
    print(f"  - Anchor tasks (for AUC alignment): {anchor_task_count}")
    print(f"  - Alignment method: {alignment_method}")
    print()

    # Print alignment parameters if verbose
    if verbose:
        print("=" * 90)
        print("ALIGNMENT PARAMETERS (fitted on anchor tasks)")
        print("=" * 90)
        print()
        if alignment_method == "affine":
            print(f"{'Method':<45} {'Slope':>10} {'Intercept':>12} {'R²':>10}")
        else:
            print(f"{'Method':<45} {'Offset':>10}")
        print("-" * 90)

        for method, metrics in results.items():
            params = metrics.get("alignment_params", {})
            if alignment_method == "affine":
                slope = params.get("slope", float("nan"))
                intercept = params.get("intercept", float("nan"))
                r2 = params.get("r_squared", float("nan"))
                print(f"{method:<45} {slope:>10.4f} {intercept:>12.4f} {r2:>10.4f}")
            else:
                offset = params.get("offset", float("nan"))
                print(f"{method:<45} {offset:>10.4f}")

        print()

    print("=" * 90)
    print(f"COMPARISON TABLE (Frontier Tasks: {frontier_task_count}, Eval Agents: {post_frontier_count})")
    print("=" * 90)
    print()
    print(f"{'Method':<45} {'ROC-AUC':>10} {'Spearman ρ':>12} {'p-value':>10}")
    print("-" * 90)

    # Sort by AUC (descending), with Spearman as tiebreaker
    def sort_key(item):
        auc = item[1].get("auc")
        rho = item[1].get("frontier_spearman_rho", float("-inf"))
        if auc is None or (isinstance(auc, float) and np.isnan(auc)):
            auc_val = float("-inf")
        else:
            auc_val = auc
        if isinstance(rho, float) and np.isnan(rho):
            rho = float("-inf")
        return (auc_val, rho)

    sorted_methods = sorted(results.items(), key=sort_key, reverse=True)

    for method, metrics in sorted_methods:
        auc = metrics.get("auc")
        rho = metrics.get("frontier_spearman_rho", float("nan"))
        p = metrics.get("frontier_spearman_p", float("nan"))

        # Format AUC
        if auc is None or (isinstance(auc, float) and np.isnan(auc)):
            auc_str = "N/A"
        else:
            auc_str = f"{auc:.4f}"

        # Format Spearman
        if isinstance(rho, float) and np.isnan(rho):
            rho_str = "N/A"
            p_str = "N/A"
        else:
            rho_str = f"{rho:.4f}"
            p_str = f"{p:.4f}" if p >= 0.0001 else "<0.0001"

        print(f"{method:<45} {auc_str:>10} {rho_str:>12} {p_str:>10}")

    print()


def save_results_csv(results: Dict[str, Dict], output_path: Path) -> None:
    """Save results to CSV."""
    rows = []
    for method, metrics in results.items():
        rows.append({
            "method": method,
            "auc": metrics.get("auc"),
            "spearman_rho": metrics.get("frontier_spearman_rho"),
            "spearman_p": metrics.get("frontier_spearman_p"),
            "pearson_r": metrics.get("frontier_pearson_r"),
            "pearson_p": metrics.get("frontier_pearson_p"),
            "n_tasks": metrics.get("num_frontier_tasks"),
            "auc_n_pairs": metrics.get("auc_n_pairs"),
            "auc_n_positive": metrics.get("auc_n_positive"),
            "auc_n_negative": metrics.get("auc_n_negative"),
        })

    df = pd.DataFrame(rows)
    # Sort by AUC descending (with NaN at bottom)
    df = df.sort_values("auc", ascending=False, na_position="last")
    df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare methods for frontier task difficulty prediction"
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
        "--oracle_abilities_path",
        type=Path,
        default=Path("clean_data/swebench_verified_20251120_full/1d/abilities.csv"),
        help="Path to oracle IRT abilities CSV (all agents)",
    )
    parser.add_argument(
        "--embeddings_path",
        type=Path,
        default=Path(
            "out/prior_qwen3vl8b/embeddings__Qwen__Qwen3-VL-8B-Instruct__pool-lasttoken__"
            "qs-sol-instr__qs_sol_instr_b7008f2d__idnorm_instance-v1__"
            "princeton-nlp_SWE-bench_Verified__test__n500__maxlen8192__seed0.npz"
        ),
        help="Path to embeddings .npz file (any backbone)",
    )
    parser.add_argument(
        "--llm_judge_path",
        type=Path,
        default=Path("chris_output/experiment_a/llm_judge_features/llm_judge_features.csv"),
        help="Path to LLM judge features CSV",
    )
    parser.add_argument(
        "--sad_irt_beta_dir",
        type=Path,
        default=Path("chris_output/sad_irt_beta_values"),
        help="Directory containing extracted SAD-IRT beta CSV files",
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
        "--alignment_method",
        type=str,
        default="affine",
        choices=["constant", "affine"],
        help="Method for aligning predicted difficulties to oracle scale (default: affine)",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Optional path to save results CSV",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print alignment parameters for each method",
    )
    args = parser.parse_args()

    # Validate required files exist
    required_files = [
        (args.responses_path, "Response matrix"),
        (args.baseline_irt_path, "Baseline IRT"),
        (args.oracle_irt_path, "Oracle IRT"),
        (args.oracle_abilities_path, "Oracle abilities"),
    ]
    for path, name in required_files:
        if not path.exists():
            print(f"Error: {name} not found: {path}")
            sys.exit(1)

    # Load IRT models and abilities
    print("Loading IRT models...")
    baseline_items = pd.read_csv(args.baseline_irt_path, index_col=0)
    oracle_items = pd.read_csv(args.oracle_irt_path, index_col=0)
    oracle_abilities = pd.read_csv(args.oracle_abilities_path, index_col=0)
    print(f"  Baseline IRT: {len(baseline_items)} tasks")
    print(f"  Oracle IRT: {len(oracle_items)} tasks")
    print(f"  Oracle abilities: {len(oracle_abilities)} agents")

    # Load response matrix for AUC computation
    print("\nLoading response matrix...")
    responses = load_responses_dict(args.responses_path)
    print(f"  Loaded responses for {len(responses)} agents")

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

    # Identify nontrivial anchor tasks for scale alignment
    print("\nIdentifying nontrivial anchor tasks...")
    anchor_task_ids, _, _ = identify_nontrivial_tasks(
        args.responses_path,
        pre_frontier,
        post_frontier,
        min_pass_rate=0.10,
        max_pass_rate=0.90,
    )
    print(f"  Anchor tasks (10-90% pass rate in both groups): {len(anchor_task_ids)}")

    # Non-frontier tasks for training predictors
    all_task_ids = list(baseline_items.index)
    train_task_ids = [t for t in all_task_ids if t not in frontier_task_ids]
    print(f"  Training tasks (non-frontier): {len(train_task_ids)}")

    # Collect results
    results = {}

    # 0. Oracle upper bound (uses true oracle beta - perfect alignment)
    print("\nEvaluating Oracle (upper bound)...")
    oracle_beta = oracle_items["b"].to_dict()
    oracle_metrics = compute_method_metrics(
        predicted_beta=oracle_beta,
        oracle_items=oracle_items,
        oracle_abilities=oracle_abilities,
        responses=responses,
        frontier_task_ids=frontier_task_ids,
        anchor_task_ids=anchor_task_ids,
        eval_agents=post_frontier,
        alignment_method=args.alignment_method,
    )
    results["Oracle (upper bound)"] = oracle_metrics
    print(f"  AUC: {oracle_metrics['auc']:.4f}")
    print(f"  Spearman rho: {oracle_metrics['frontier_spearman_rho']:.4f}")

    # 1. Baseline IRT
    print("\nEvaluating Baseline IRT...")
    baseline_beta = baseline_items["b"].to_dict()
    baseline_metrics = compute_method_metrics(
        predicted_beta=baseline_beta,
        oracle_items=oracle_items,
        oracle_abilities=oracle_abilities,
        responses=responses,
        frontier_task_ids=frontier_task_ids,
        anchor_task_ids=anchor_task_ids,
        eval_agents=post_frontier,
        alignment_method=args.alignment_method,
    )
    results["Baseline IRT (pre-frontier only)"] = baseline_metrics
    print(f"  AUC: {baseline_metrics['auc']:.4f}")
    print(f"  Spearman rho: {baseline_metrics['frontier_spearman_rho']:.4f}")

    # 2. SAD-IRT runs (from extracted beta CSV files)
    if args.sad_irt_beta_dir.exists():
        beta_files = list(args.sad_irt_beta_dir.glob("*.csv"))
        print(f"\nLoading SAD-IRT beta values from {args.sad_irt_beta_dir}...")
        print(f"  Found {len(beta_files)} beta CSV files")

        for beta_file in beta_files:
            # Load beta values from CSV
            beta_df = pd.read_csv(beta_file, index_col=0)
            if "beta" not in beta_df.columns:
                print(f"  Skipping {beta_file.name}: no 'beta' column")
                continue

            sad_irt_beta = beta_df["beta"].to_dict()

            # Create method name from filename
            method_name = f"SAD-IRT ({beta_file.stem})"
            if len(method_name) > 45:
                method_name = f"SAD-IRT ({beta_file.stem[:30]}...)"

            print(f"\n  Evaluating {method_name}...")
            sad_irt_metrics = compute_method_metrics(
                predicted_beta=sad_irt_beta,
                oracle_items=oracle_items,
                oracle_abilities=oracle_abilities,
                responses=responses,
                frontier_task_ids=frontier_task_ids,
                anchor_task_ids=anchor_task_ids,
                eval_agents=post_frontier,
                alignment_method=args.alignment_method,
            )
            results[method_name] = sad_irt_metrics
            auc = sad_irt_metrics.get('auc')
            rho = sad_irt_metrics.get('frontier_spearman_rho')
            print(f"    AUC: {auc:.4f}" if auc else "    AUC: N/A")
            print(f"    Spearman rho: {rho:.4f}" if rho else "    Spearman rho: N/A")
    else:
        print(f"\nSAD-IRT beta directory not found: {args.sad_irt_beta_dir}")
        print("  To include SAD-IRT results, run experiment_sad_irt and extract beta values")

    # 3. Embedding predictor
    if args.embeddings_path.exists():
        print("\nEvaluating Embedding + Ridge predictor...")
        try:
            predictor = EmbeddingPredictor(embeddings_path=args.embeddings_path)
            embedding_metrics = evaluate_predictor(
                predictor=predictor,
                baseline_items=baseline_items,
                oracle_items=oracle_items,
                oracle_abilities=oracle_abilities,
                responses=responses,
                frontier_task_ids=frontier_task_ids,
                train_task_ids=train_task_ids,
                anchor_task_ids=anchor_task_ids,
                eval_agents=post_frontier,
                alignment_method=args.alignment_method,
            )
            results["Embedding + Ridge"] = embedding_metrics
            auc = embedding_metrics.get('auc')
            rho = embedding_metrics.get('frontier_spearman_rho')
            print(f"  AUC: {auc:.4f}" if auc else "  AUC: N/A")
            print(f"  Spearman rho: {rho:.4f}" if rho else "  Spearman rho: N/A")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            results["Embedding + Ridge"] = {
                "frontier_spearman_rho": float("nan"),
                "frontier_spearman_p": float("nan"),
                "auc": None,
                "num_frontier_tasks": 0,
            }
    else:
        print(f"\nEmbeddings not found: {args.embeddings_path}")

    # 4. LLM Judge predictor
    if args.llm_judge_path.exists():
        print("\nEvaluating LLM Judge + Lasso/Ridge predictor...")
        try:
            predictor = LLMJudgePredictor(features_path=args.llm_judge_path)
            llm_judge_metrics = evaluate_predictor(
                predictor=predictor,
                baseline_items=baseline_items,
                oracle_items=oracle_items,
                oracle_abilities=oracle_abilities,
                responses=responses,
                frontier_task_ids=frontier_task_ids,
                train_task_ids=train_task_ids,
                anchor_task_ids=anchor_task_ids,
                eval_agents=post_frontier,
                alignment_method=args.alignment_method,
            )
            results["LLM Judge + Lasso/Ridge"] = llm_judge_metrics
            auc = llm_judge_metrics.get('auc')
            rho = llm_judge_metrics.get('frontier_spearman_rho')
            print(f"  AUC: {auc:.4f}" if auc else "  AUC: N/A")
            print(f"  Spearman rho: {rho:.4f}" if rho else "  Spearman rho: N/A")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            results["LLM Judge + Lasso/Ridge"] = {
                "frontier_spearman_rho": float("nan"),
                "frontier_spearman_p": float("nan"),
                "auc": None,
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
        anchor_task_count=len(anchor_task_ids),
        alignment_method=args.alignment_method,
        verbose=args.verbose,
    )

    # Save to CSV if requested
    if args.output_csv:
        save_results_csv(results, args.output_csv)


if __name__ == "__main__":
    main()

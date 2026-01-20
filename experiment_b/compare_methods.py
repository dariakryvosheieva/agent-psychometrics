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
    split_agents_by_dates,
)
from experiment_b.datasets import get_dataset_config, list_datasets
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
    cutoff_date: str = "20250807",
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
    print(f"  - Cutoff date: {cutoff_date}")
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
        "--dataset",
        type=str,
        default="swebench",
        choices=list_datasets(),
        help="Dataset to run experiment on (default: swebench)",
    )
    parser.add_argument(
        "--responses_path",
        type=Path,
        default=None,
        help="Path to response matrix JSONL (overrides dataset default)",
    )
    parser.add_argument(
        "--baseline_irt_path",
        type=Path,
        default=None,
        help="Path to baseline IRT items CSV (overrides dataset default)",
    )
    parser.add_argument(
        "--oracle_irt_path",
        type=Path,
        default=None,
        help="Path to oracle IRT items CSV (overrides dataset default)",
    )
    parser.add_argument(
        "--oracle_abilities_path",
        type=Path,
        default=None,
        help="Path to oracle IRT abilities CSV (overrides dataset default)",
    )
    parser.add_argument(
        "--embeddings_path",
        type=Path,
        default=None,
        help="Path to embeddings .npz file (overrides dataset default)",
    )
    parser.add_argument(
        "--llm_judge_path",
        type=Path,
        default=None,
        help="Path to LLM judge features CSV (overrides dataset default)",
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
        default=None,
        help="Frontier cutoff date YYYYMMDD (overrides dataset default)",
    )
    parser.add_argument(
        "--pre_threshold",
        type=float,
        default=None,
        help="Max pre-frontier pass rate for frontier tasks (overrides dataset default)",
    )
    parser.add_argument(
        "--post_threshold",
        type=float,
        default=None,
        help="Min post-frontier pass rate for frontier tasks (overrides dataset default)",
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
    parser.add_argument(
        "--train_on_all_tasks",
        action="store_true",
        help="Also train predictors on all tasks (still using baseline IRT difficulties) and compare",
    )
    args = parser.parse_args()

    # Load dataset configuration
    print(f"Loading dataset configuration: {args.dataset}")
    dataset_config = get_dataset_config(args.dataset)

    # Override paths from CLI args if provided
    responses_path = args.responses_path or dataset_config.responses_path
    oracle_irt_path = args.oracle_irt_path or dataset_config.oracle_irt_path
    oracle_abilities_path = args.oracle_abilities_path or dataset_config.oracle_abilities_path
    baseline_irt_path = args.baseline_irt_path or dataset_config.baseline_irt_path
    embeddings_path = args.embeddings_path or dataset_config.embeddings_path
    llm_judge_path = args.llm_judge_path or dataset_config.llm_judge_path
    cutoff_date = args.cutoff_date or dataset_config.cutoff_date
    pre_threshold = args.pre_threshold if args.pre_threshold is not None else dataset_config.pre_threshold
    post_threshold = args.post_threshold if args.post_threshold is not None else dataset_config.post_threshold
    output_dir = dataset_config.output_dir

    print(f"  Dataset: {dataset_config.name}")
    print(f"  Cutoff date: {cutoff_date}")

    # Validate required files exist
    required_files = [
        (responses_path, "Response matrix"),
        (oracle_irt_path, "Oracle IRT"),
        (oracle_abilities_path, "Oracle abilities"),
    ]
    for path, name in required_files:
        if not path.exists():
            print(f"Error: {name} not found: {path}")
            sys.exit(1)

    # Load IRT models and abilities
    print("\nLoading IRT models...")
    oracle_items = pd.read_csv(oracle_irt_path, index_col=0)
    oracle_abilities = pd.read_csv(oracle_abilities_path, index_col=0)
    print(f"  Oracle IRT: {len(oracle_items)} tasks")
    print(f"  Oracle abilities: {len(oracle_abilities)} agents")

    # Load response matrix for AUC computation
    print("\nLoading response matrix...")
    responses = load_responses_dict(responses_path)
    print(f"  Loaded responses for {len(responses)} agents")

    # Get agent dates from dataset config and split by cutoff
    print("\nIdentifying frontier tasks...")
    all_agents = get_all_agents_from_responses(responses_path)
    agent_dates = dataset_config.get_agent_dates(all_agents)
    print(f"  Agents with dates: {len(agent_dates)} / {len(all_agents)}")

    pre_frontier, post_frontier = split_agents_by_dates(all_agents, agent_dates, cutoff_date)
    print(f"  Pre-frontier agents (< {cutoff_date}): {len(pre_frontier)}")
    print(f"  Post-frontier agents (>= {cutoff_date}): {len(post_frontier)}")

    # Load or train baseline IRT (pre-frontier agents only)
    if baseline_irt_path and baseline_irt_path.exists():
        baseline_items = pd.read_csv(baseline_irt_path, index_col=0)
        print(f"  Baseline IRT: {len(baseline_items)} tasks (loaded from cache)")
    else:
        # Train baseline IRT on pre-frontier agents
        print("\n  Training baseline IRT on pre-frontier agents...")
        from experiment_sad_irt.train_evaluate import train_baseline_irt_on_prefrontier
        baseline_irt_output_dir = output_dir / "baseline_irt"
        baseline_beta = train_baseline_irt_on_prefrontier(
            responses_path=responses_path,
            pre_frontier_agents=pre_frontier,
            output_dir=baseline_irt_output_dir,
        )
        baseline_items = pd.read_csv(baseline_irt_output_dir / "items.csv", index_col=0)
        print(f"  Baseline IRT: {len(baseline_items)} tasks (newly trained)")

    frontier_task_ids = identify_frontier_tasks(
        responses_path,
        pre_frontier,
        post_frontier,
        pre_threshold,
        post_threshold,
    )
    print(f"  Frontier tasks: {len(frontier_task_ids)}")

    # Identify nontrivial anchor tasks for scale alignment
    print("\nIdentifying nontrivial anchor tasks...")
    anchor_task_ids, _, _ = identify_nontrivial_tasks(
        responses_path,
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

    # Define training configurations
    # Note: ground truth is ALWAYS baseline_items (pre-frontier IRT)
    # We only vary which tasks are included in training
    training_configs = [
        {
            "suffix": "",
            "train_task_ids": train_task_ids,  # non-frontier only
        }
    ]

    if args.train_on_all_tasks:
        # Include frontier tasks in training, but still use baseline_items difficulties
        # (baseline difficulties for frontier tasks are poorly calibrated but that's expected)
        all_task_ids_baseline = list(baseline_items.index)
        training_configs.append({
            "suffix": " (all tasks)",
            "train_task_ids": all_task_ids_baseline,
        })

    # 3. Embedding predictor
    if embeddings_path.exists():
        for config in training_configs:
            method_name = f"Embedding + Ridge{config['suffix']}"
            print(f"\nEvaluating {method_name}...")
            print(f"  Training on {len(config['train_task_ids'])} tasks")
            try:
                predictor = EmbeddingPredictor(embeddings_path=embeddings_path)
                embedding_metrics = evaluate_predictor(
                    predictor=predictor,
                    baseline_items=baseline_items,
                    oracle_items=oracle_items,
                    oracle_abilities=oracle_abilities,
                    responses=responses,
                    frontier_task_ids=frontier_task_ids,
                    train_task_ids=config['train_task_ids'],
                    anchor_task_ids=anchor_task_ids,
                    eval_agents=post_frontier,
                    alignment_method=args.alignment_method,
                )
                results[method_name] = embedding_metrics
                auc = embedding_metrics.get('auc')
                rho = embedding_metrics.get('frontier_spearman_rho')
                print(f"  AUC: {auc:.4f}" if auc else "  AUC: N/A")
                print(f"  Spearman rho: {rho:.4f}" if rho else "  Spearman rho: N/A")
            except Exception as e:
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()
                results[method_name] = {
                    "frontier_spearman_rho": float("nan"),
                    "frontier_spearman_p": float("nan"),
                    "auc": None,
                    "num_frontier_tasks": 0,
                }
    else:
        print(f"\nEmbeddings not found: {embeddings_path}")

    # 4. LLM Judge predictor
    if llm_judge_path.exists():
        for config in training_configs:
            method_name = f"LLM Judge + Lasso/Ridge{config['suffix']}"
            print(f"\nEvaluating {method_name}...")
            print(f"  Training on {len(config['train_task_ids'])} tasks")
            try:
                predictor = LLMJudgePredictor(
                    features_path=llm_judge_path,
                    feature_cols=dataset_config.llm_judge_feature_cols,
                )
                llm_judge_metrics = evaluate_predictor(
                    predictor=predictor,
                    baseline_items=baseline_items,
                    oracle_items=oracle_items,
                    oracle_abilities=oracle_abilities,
                    responses=responses,
                    frontier_task_ids=frontier_task_ids,
                    train_task_ids=config['train_task_ids'],
                    anchor_task_ids=anchor_task_ids,
                    eval_agents=post_frontier,
                    alignment_method=args.alignment_method,
                )
                results[method_name] = llm_judge_metrics
                auc = llm_judge_metrics.get('auc')
                rho = llm_judge_metrics.get('frontier_spearman_rho')
                print(f"  AUC: {auc:.4f}" if auc else "  AUC: N/A")
                print(f"  Spearman rho: {rho:.4f}" if rho else "  Spearman rho: N/A")
            except Exception as e:
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()
                results[method_name] = {
                    "frontier_spearman_rho": float("nan"),
                    "frontier_spearman_p": float("nan"),
                    "auc": None,
                    "num_frontier_tasks": 0,
                }
    else:
        print(f"\nLLM Judge features not found: {llm_judge_path}")

    # Print comparison table
    print()
    print_comparison_table(
        results,
        len(frontier_task_ids),
        len(pre_frontier),
        len(post_frontier),
        anchor_task_count=len(anchor_task_ids),
        alignment_method=args.alignment_method,
        cutoff_date=cutoff_date,
        verbose=args.verbose,
    )

    # Save to CSV if requested
    if args.output_csv:
        save_results_csv(results, args.output_csv)


if __name__ == "__main__":
    main()

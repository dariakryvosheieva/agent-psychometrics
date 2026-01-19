"""Main entry point for iterative prompt refinement.

Runs the full iterative refinement loop:
1. Quick eval current prompt on n=30 stratified tasks
2. Analyze high-residual tasks
3. Generate refined prompt
4. Repeat for max_iterations
5. Report best performing version

Usage:
    python -m llm_judge.iterative_refinement.run_iteration --dry_run
    python -m llm_judge.iterative_refinement.run_iteration
    python -m llm_judge.iterative_refinement.run_iteration --max_iterations 10
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd

# Add parent to path for imports
# ROOT is model_irt/ (2 levels up from llm_judge/iterative_refinement/)
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_judge.iterative_refinement.config import IterativeRefinementConfig
from llm_judge.iterative_refinement.prompt_store import (
    PromptStore,
    PromptVersion,
    FeatureDefinition,
    create_initial_feature_schema,
    generate_prompt_from_schema,
)
from llm_judge.iterative_refinement.quick_evaluator import (
    run_quick_evaluation_sync,
    stratified_sample_tasks,
    load_swebench_tasks,
)
from llm_judge.iterative_refinement.residual_analyzer import (
    analyze_residuals,
    format_residual_analysis_for_llm,
)
from llm_judge.iterative_refinement.prompt_refiner import (
    propose_refinement,
    apply_refinement_constraints,
)


def run_iteration_loop(config: IterativeRefinementConfig, dry_run: bool = False) -> Dict[str, Any]:
    """Run the full iterative refinement loop.

    Args:
        config: Configuration for refinement
        dry_run: If True, just print what would be done

    Returns:
        Dict with results from all iterations
    """
    print("=" * 70)
    print("ITERATIVE LLM JUDGE PROMPT REFINEMENT")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model: {config.model}")
    print(f"  Max iterations: {config.max_iterations}")
    print(f"  Quick eval tasks: {config.quick_eval_tasks}")
    print(f"  Tasks per tercile: {config.tasks_per_tercile}")
    print(f"  Output dir: {config.output_dir}")

    # Load ground truth items
    items_path = ROOT / config.items_path
    if not items_path.exists():
        print(f"\nError: Items file not found: {items_path}")
        return {"error": f"Items file not found: {items_path}"}

    items_df = pd.read_csv(items_path, index_col=0)
    print(f"\nLoaded {len(items_df)} tasks with ground truth difficulties")

    # Select stratified sample (same for all iterations for comparability)
    sample_tasks = stratified_sample_tasks(
        items_df, n_per_tercile=config.tasks_per_tercile, seed=42
    )
    print(f"Selected {len(sample_tasks)} stratified tasks for quick evaluation")

    if dry_run:
        print("\n[DRY RUN] Would run the following:")
        print(f"  1. Initialize with {len(config.initial_features)} features")
        print(f"  2. Quick eval on {len(sample_tasks)} tasks")
        print(f"  3. Analyze residuals, identify top-{config.high_residual_tasks} failures")
        print(f"  4. Propose refined features using {config.model}")
        print(f"  5. Repeat for {config.max_iterations} iterations")
        print("\nWould create output in:")
        print(f"  - {config.output_dir}/prompt_versions/")
        print(f"  - {config.output_dir}/evaluations/")
        return {"dry_run": True}

    # Initialize prompt store
    store = PromptStore(config.output_dir)

    # Initialize with baseline features if no versions exist
    if not store.list_versions():
        print("\n1. Initializing with baseline feature schema...")
        initial_features = create_initial_feature_schema()
        initial_prompt = generate_prompt_from_schema(initial_features)
        version = store.add_version(
            prompt_text=initial_prompt,
            feature_schema=initial_features,
            changes_from_parent="Initial 9-feature schema from experiment_a",
        )
        print(f"   Created version {version.version_id}")
    else:
        version = store.get_latest()
        print(f"\nResuming from version {version.version_id}")

    # Track results
    results = {
        "config": {
            "model": config.model,
            "max_iterations": config.max_iterations,
            "quick_eval_tasks": config.quick_eval_tasks,
        },
        "iterations": [],
        "best_version": None,
        "best_r": None,
    }

    # Main iteration loop
    for iteration in range(config.max_iterations):
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration + 1} / {config.max_iterations}")
        print(f"{'='*70}")

        current_version = store.get_latest()
        print(f"\nEvaluating version {current_version.version_id}")
        print(f"Features: {', '.join(current_version.feature_names)}")

        # Quick evaluation
        print(f"\n2. Running quick evaluation on {len(sample_tasks)} tasks...")
        try:
            eval_result = run_quick_evaluation_sync(
                feature_definitions=current_version.feature_schema,
                config=config,
                task_ids=sample_tasks,
                ground_truth_items=items_df,
            )
        except Exception as e:
            print(f"   Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
            results["iterations"].append({
                "iteration": iteration + 1,
                "version_id": current_version.version_id,
                "error": str(e),
            })
            continue

        print(f"   Successful extractions: {eval_result.n_successful}/{eval_result.n_tasks}")
        print(f"   Pearson r with difficulty: {eval_result.pearson_r:.3f}" if eval_result.pearson_r else "   Pearson r: N/A")
        print(f"   Estimated cost: ${eval_result.estimated_cost:.2f}")

        # Report feature quality
        if eval_result.low_entropy_features:
            print(f"   Low entropy features: {', '.join(eval_result.low_entropy_features)}")
        if eval_result.redundant_pairs:
            print(f"   Redundant pairs: {len(eval_result.redundant_pairs)}")

        # Update version metrics
        store.update_version(
            current_version.version_id,
            quick_eval_r=eval_result.pearson_r,
            quick_eval_entropy=eval_result.feature_entropies,
            quick_eval_redundant_pairs=eval_result.redundant_pairs,
        )

        # Track best
        if eval_result.pearson_r is not None:
            if results["best_r"] is None or abs(eval_result.pearson_r) > abs(results["best_r"]):
                results["best_version"] = current_version.version_id
                results["best_r"] = eval_result.pearson_r

        # Record iteration results
        iter_result = {
            "iteration": iteration + 1,
            "version_id": current_version.version_id,
            "pearson_r": eval_result.pearson_r,
            "n_successful": eval_result.n_successful,
            "feature_entropies": eval_result.feature_entropies,
            "low_entropy_features": eval_result.low_entropy_features,
            "redundant_pairs": [
                {"f1": p[0], "f2": p[1], "r": p[2]} for p in eval_result.redundant_pairs
            ],
            "cost": eval_result.estimated_cost,
        }

        # Skip refinement on last iteration
        if iteration == config.max_iterations - 1:
            print("\n   Last iteration - skipping refinement")
            results["iterations"].append(iter_result)
            break

        # Residual analysis
        if eval_result.features_df is not None and eval_result.n_successful >= 10:
            print(f"\n3. Analyzing residuals...")
            try:
                # Load task data for summaries
                task_data = load_swebench_tasks(sample_tasks)

                analysis = analyze_residuals(
                    features_df=eval_result.features_df,
                    ground_truth=items_df["b"],
                    feature_names=current_version.feature_names,
                    n_top=config.high_residual_tasks // 2,
                    task_data=task_data,
                )
                print(f"   RMSE: {analysis.rmse:.3f}")
                print(f"   Tasks harder than predicted: {len(analysis.harder_than_predicted)}")
                print(f"   Tasks easier than predicted: {len(analysis.easier_than_predicted)}")

                iter_result["residual_rmse"] = analysis.rmse
                iter_result["feature_coefficients"] = analysis.feature_coefficients

            except Exception as e:
                print(f"   Error during residual analysis: {e}")
                analysis = None
        else:
            print("\n3. Skipping residual analysis (not enough data)")
            analysis = None

        # Propose refinement
        if analysis is not None:
            print(f"\n4. Proposing refined features using {config.model}...")
            try:
                proposal = propose_refinement(
                    current_features=current_version.feature_schema,
                    analysis=analysis,
                    quick_eval_metrics=eval_result.to_dict(),
                    model=config.model,
                )
                proposal = apply_refinement_constraints(proposal)

                print(f"   Features added: {proposal.features_added or 'none'}")
                print(f"   Features removed: {proposal.features_removed or 'none'}")
                print(f"   Features modified: {proposal.features_modified or 'none'}")

                # Create new version
                new_prompt = generate_prompt_from_schema(proposal.new_features)
                changes = []
                if proposal.features_added:
                    changes.append(f"Added: {', '.join(proposal.features_added)}")
                if proposal.features_removed:
                    changes.append(f"Removed: {', '.join(proposal.features_removed)}")
                if proposal.features_modified:
                    changes.append(f"Modified: {', '.join(proposal.features_modified)}")

                new_version = store.add_version(
                    prompt_text=new_prompt,
                    feature_schema=proposal.new_features,
                    parent_version=current_version.version_id,
                    changes_from_parent="; ".join(changes) if changes else "No changes",
                    archived_features=proposal.archived_features,
                )
                print(f"   Created version {new_version.version_id}")

                iter_result["refinement"] = proposal.to_dict()
                iter_result["new_version_id"] = new_version.version_id

            except Exception as e:
                print(f"   Error during refinement: {e}")
                import traceback
                traceback.print_exc()
                iter_result["refinement_error"] = str(e)
        else:
            print("\n4. Skipping refinement (no residual analysis)")

        results["iterations"].append(iter_result)

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\nAll versions:")
    print(store.format_summary())
    print(f"\nBest version: {results['best_version']}")
    print(f"Best Pearson r: {results['best_r']:.3f}" if results['best_r'] else "Best Pearson r: N/A")

    # Save results
    results_path = config.output_dir / f"refinement_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run iterative LLM judge prompt refinement"
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=5,
        help="Maximum number of refinement iterations (default: 5)",
    )
    parser.add_argument(
        "--quick_eval_tasks",
        type=int,
        default=30,
        help="Number of tasks for quick evaluation (default: 30)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.2",
        help="Model for feature extraction and refinement (default: gpt-5.2)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="chris_output/iterative_refinement",
        help="Output directory (default: chris_output/iterative_refinement)",
    )
    parser.add_argument(
        "--items_path",
        type=str,
        default="clean_data/swebench_verified_20251120_full/1d/items.csv",
        help="Path to IRT items.csv with ground truth difficulties",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be done without running",
    )

    args = parser.parse_args()

    # Build config
    config = IterativeRefinementConfig(
        model=args.model,
        max_iterations=args.max_iterations,
        quick_eval_tasks=args.quick_eval_tasks,
        tasks_per_tercile=args.quick_eval_tasks // 3,
        output_dir=Path(args.output_dir),
        items_path=Path(args.items_path),
    )

    run_iteration_loop(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
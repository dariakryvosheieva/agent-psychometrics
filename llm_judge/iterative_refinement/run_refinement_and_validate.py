"""Run iterative refinement with n=100 and validate best prompts via CV.

This script:
1. Runs iterative refinement with n=100 tasks for better statistical power
2. Identifies the best performing prompt versions
3. Extracts features for all tasks using the best prompts
4. Runs full experiment_a k-fold CV on SWE-bench dataset
5. Reports comparative AUC results

Usage:
    # Dry run - show what would happen
    python -m llm_judge.iterative_refinement.run_refinement_and_validate --dry_run

    # Run refinement only
    python -m llm_judge.iterative_refinement.run_refinement_and_validate --refinement_only

    # Run validation only (using existing prompt versions)
    python -m llm_judge.iterative_refinement.run_refinement_and_validate --validation_only

    # Full pipeline
    python -m llm_judge.iterative_refinement.run_refinement_and_validate
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
from openai import AsyncOpenAI

# Add parent to path for imports
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_judge.iterative_refinement.config import IterativeRefinementConfig
from llm_judge.iterative_refinement.run_iteration import run_iteration_loop
from llm_judge.iterative_refinement.prompt_store import (
    PromptStore,
    PromptVersion,
    FeatureDefinition,
    generate_prompt_from_schema,
)
from llm_judge.iterative_refinement.quick_evaluator import (
    load_swebench_tasks,
    extract_features_batch,
)
from llm_judge.iterative_refinement.prompt_format import format_cacheable_prompt


def get_top_versions(
    store: PromptStore,
    n_top: int = 3,
    metric: str = "quick_eval_r",
) -> List[PromptVersion]:
    """Get the top N versions by performance metric.

    Args:
        store: The prompt store
        n_top: Number of top versions to return
        metric: Metric to sort by

    Returns:
        List of top performing PromptVersions
    """
    versions = store.list_versions()

    # Filter to versions with the metric
    valid = [v for v in versions if getattr(v, metric) is not None]

    # Sort by absolute value of metric (correlation can be negative)
    sorted_versions = sorted(valid, key=lambda v: abs(getattr(v, metric)), reverse=True)

    return sorted_versions[:n_top]


async def extract_features_for_all_tasks(
    feature_definitions: List[FeatureDefinition],
    model: str = "gpt-5.2",
) -> pd.DataFrame:
    """Extract features for all SWE-bench Verified tasks.

    Args:
        feature_definitions: Feature definitions to use
        model: Model for extraction

    Returns:
        DataFrame with features for all tasks
    """
    from datasets import load_dataset

    # Load all SWE-bench Verified tasks
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    all_task_ids = [row["instance_id"] for row in ds]

    print(f"Extracting features for {len(all_task_ids)} tasks...")

    # Load task data
    tasks = load_swebench_tasks(all_task_ids)

    # Build feature instructions
    feature_instructions = generate_prompt_from_schema(feature_definitions)
    feature_names = [f.name for f in feature_definitions]

    # Extract features
    async with AsyncOpenAI() as client:
        results, input_tokens, output_tokens = await extract_features_batch(
            client=client,
            tasks=tasks,
            feature_instructions=feature_instructions,
            feature_names=feature_names,
            model=model,
            max_concurrent=20,  # Higher concurrency for full extraction
        )

    # Build DataFrame
    rows = []
    for task_id, result in results.items():
        if result["success"] and result["features"]:
            row = {"_instance_id": task_id}
            row.update(result["features"])
            # Also save reasoning if available
            if result.get("reasoning"):
                row["reasoning"] = result["reasoning"]
            rows.append(row)

    df = pd.DataFrame(rows)
    print(f"Successfully extracted features for {len(df)} / {len(all_task_ids)} tasks")
    print(f"Tokens used: {input_tokens:,} input, {output_tokens:,} output")

    return df


def run_cv_validation(
    features_path: Path,
    feature_names: List[str],
    version_id: str,
    k_folds: int = 5,
) -> Dict[str, Any]:
    """Run k-fold cross-validation with specific LLM judge features.

    Args:
        features_path: Path to features CSV
        feature_names: List of feature column names to use
        version_id: Version ID for logging
        k_folds: Number of CV folds

    Returns:
        Dict with CV results
    """
    from experiment_a.config import ExperimentAConfig
    from experiment_a_common.pipeline import ExperimentSpec, run_cross_validation

    print(f"\n{'='*60}")
    print(f"Validating version {version_id}")
    print(f"Features: {', '.join(feature_names)}")
    print(f"Features path: {features_path}")
    print(f"{'='*60}")

    # Create experiment spec with custom features
    spec = ExperimentSpec(
        name=f"SWE-bench ({version_id})",
        is_binomial=False,
        irt_cache_dir=ROOT / "chris_output" / "experiment_a" / "irt_splits",
        llm_judge_features=feature_names,
    )

    # Create config with the features path
    config = ExperimentAConfig()
    config.llm_judge_features_path = features_path

    # Run CV
    results = run_cross_validation(config, spec, ROOT, k=k_folds)

    return results


def run_refinement_phase(
    config: IterativeRefinementConfig,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run the iterative refinement phase.

    Args:
        config: Configuration for refinement
        dry_run: If True, just print what would be done

    Returns:
        Results from refinement
    """
    print("=" * 70)
    print("PHASE 1: ITERATIVE REFINEMENT (n=100)")
    print("=" * 70)

    results = run_iteration_loop(config, dry_run=dry_run)
    return results


def run_validation_phase(
    output_dir: Path,
    n_top: int = 3,
    k_folds: int = 5,
    model: str = "gpt-5.2",
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run the validation phase on best prompts.

    Args:
        output_dir: Directory with prompt versions
        n_top: Number of top versions to validate
        k_folds: Number of CV folds
        model: Model for feature extraction
        dry_run: If True, just print what would be done

    Returns:
        Dict with validation results
    """
    print("\n" + "=" * 70)
    print("PHASE 2: CROSS-VALIDATION ON BEST PROMPTS")
    print("=" * 70)

    # Load prompt store
    store = PromptStore(output_dir)
    versions = store.list_versions()

    if not versions:
        print("\nNo prompt versions found. Run refinement first.")
        return {"error": "No prompt versions found"}

    print(f"\nLoaded {len(versions)} prompt versions")

    # Get top versions
    top_versions = get_top_versions(store, n_top=n_top)

    print(f"\nTop {len(top_versions)} versions by quick_eval_r:")
    for v in top_versions:
        print(f"  {v.version_id}: r={v.quick_eval_r:.3f}, features={', '.join(v.feature_names)}")

    if dry_run:
        print("\n[DRY RUN] Would:")
        print("  1. Run baseline CV with pre-computed features")
        print("  2. Extract features for all 500 tasks with each top version")
        print("  3. Run k-fold CV for each version")
        return {"dry_run": True, "top_versions": [v.version_id for v in top_versions]}

    validation_results = {}

    # Baseline features (use pre-computed)
    baseline_features_path = ROOT / "chris_output" / "experiment_a" / "llm_judge_features" / "llm_judge_features.csv"
    baseline_features = [
        "fix_in_description",
        "problem_clarity",
        "error_message_provided",
        "reproduction_steps",
        "fix_locality",
        "domain_knowledge_required",
        "fix_complexity",
        "logical_reasoning_required",
        "atypicality",
    ]

    # Run CV for baseline (using pre-computed features)
    print("\n" + "-" * 60)
    print("Baseline features (original 9, pre-computed)")
    print("-" * 60)
    try:
        baseline_result = run_cv_validation(
            features_path=baseline_features_path,
            feature_names=baseline_features,
            version_id="baseline",
            k_folds=k_folds,
        )
        validation_results["baseline"] = {
            "version_id": "baseline",
            "features": baseline_features,
            "cv_results": baseline_result,
        }
        llm_judge_result = baseline_result.get("cv_results", {}).get("llm_judge_predictor", {})
        print(f"\nBaseline LLM Judge: AUC = {llm_judge_result.get('mean_auc', 'N/A'):.4f} ± {llm_judge_result.get('std_auc', 'N/A'):.4f}")
    except Exception as e:
        print(f"Error running baseline CV: {e}")
        import traceback
        traceback.print_exc()
        validation_results["baseline"] = {"error": str(e)}

    # Run CV for each top version
    for version in top_versions:
        print("\n" + "-" * 60)
        print(f"Version {version.version_id}")
        print("-" * 60)

        try:
            # Check if features already extracted
            features_path = output_dir / "features" / f"{version.version_id}_features.csv"

            if not features_path.exists():
                print(f"Extracting features for all tasks...")
                features_path.parent.mkdir(parents=True, exist_ok=True)

                # Extract features for all tasks
                features_df = asyncio.run(
                    extract_features_for_all_tasks(
                        feature_definitions=version.feature_schema,
                        model=model,
                    )
                )

                # Save features
                features_df.to_csv(features_path, index=False)
                print(f"Saved features to {features_path}")
            else:
                print(f"Using cached features from {features_path}")

            # Run CV
            result = run_cv_validation(
                features_path=features_path,
                feature_names=version.feature_names,
                version_id=version.version_id,
                k_folds=k_folds,
            )
            validation_results[version.version_id] = {
                "version_id": version.version_id,
                "features": version.feature_names,
                "quick_eval_r": version.quick_eval_r,
                "cv_results": result,
            }

            # Update prompt store with full eval AUC
            llm_judge_result = result.get("cv_results", {}).get("llm_judge_predictor", {})
            if llm_judge_result and llm_judge_result.get("mean_auc"):
                store.update_version(
                    version.version_id,
                    full_eval_auc=llm_judge_result["mean_auc"],
                )
                print(f"\nVersion {version.version_id} LLM Judge: AUC = {llm_judge_result['mean_auc']:.4f} ± {llm_judge_result.get('std_auc', 0):.4f}")
        except Exception as e:
            print(f"Error running CV for {version.version_id}: {e}")
            import traceback
            traceback.print_exc()
            validation_results[version.version_id] = {"error": str(e)}

    return validation_results


def print_final_summary(
    refinement_results: Optional[Dict[str, Any]],
    validation_results: Optional[Dict[str, Any]],
):
    """Print final summary of all results."""
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    if refinement_results and not refinement_results.get("dry_run"):
        print(f"\nRefinement Phase:")
        print(f"  Total API cost: ${refinement_results.get('total_cost', 0):.4f}")
        print(f"  Best quick_eval version: {refinement_results.get('best_version')}")
        print(f"  Best quick_eval r: {refinement_results.get('best_r', 'N/A')}")

    if validation_results and not validation_results.get("dry_run") and not validation_results.get("error"):
        print(f"\nValidation Phase (5-fold CV AUC):")
        print(f"\n{'Version':<15} {'Quick r':>10} {'CV AUC':>12} {'CV Std':>10}")
        print("-" * 50)

        # Sort by CV AUC
        sorted_results = []
        for name, result in validation_results.items():
            if "error" not in result:
                cv_results = result.get("cv_results", {}).get("cv_results", {})
                llm_judge = cv_results.get("llm_judge_predictor", {})
                sorted_results.append({
                    "name": name,
                    "quick_r": result.get("quick_eval_r"),
                    "cv_auc": llm_judge.get("mean_auc"),
                    "cv_std": llm_judge.get("std_auc"),
                })

        sorted_results = sorted(
            sorted_results,
            key=lambda x: x["cv_auc"] if x["cv_auc"] else 0,
            reverse=True,
        )

        for r in sorted_results:
            quick_r = f"{r['quick_r']:.3f}" if r['quick_r'] else "N/A"
            cv_auc = f"{r['cv_auc']:.4f}" if r['cv_auc'] else "N/A"
            cv_std = f"± {r['cv_std']:.4f}" if r['cv_std'] else ""
            print(f"{r['name']:<15} {quick_r:>10} {cv_auc:>12} {cv_std:>10}")

        # Report best
        if sorted_results:
            best = sorted_results[0]
            print(f"\nBest version: {best['name']} with CV AUC = {best['cv_auc']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Run iterative refinement and validate best prompts via CV"
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=5,
        help="Maximum refinement iterations (default: 5)",
    )
    parser.add_argument(
        "--quick_eval_tasks",
        type=int,
        default=100,
        help="Tasks for quick evaluation (default: 100 for better power)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.2",
        help="Model for feature extraction (default: gpt-5.2)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="chris_output/iterative_refinement_n100",
        help="Output directory (default: chris_output/iterative_refinement_n100)",
    )
    parser.add_argument(
        "--n_top",
        type=int,
        default=3,
        help="Number of top versions to validate (default: 3)",
    )
    parser.add_argument(
        "--k_folds",
        type=int,
        default=5,
        help="Number of CV folds (default: 5)",
    )
    parser.add_argument(
        "--refinement_only",
        action="store_true",
        help="Only run refinement phase",
    )
    parser.add_argument(
        "--validation_only",
        action="store_true",
        help="Only run validation phase (uses existing versions)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be done without running",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Track overall results
    refinement_results = None
    validation_results = None

    # Phase 1: Refinement
    if not args.validation_only:
        config = IterativeRefinementConfig(
            model=args.model,
            max_iterations=args.max_iterations,
            quick_eval_tasks=args.quick_eval_tasks,
            tasks_per_tercile=args.quick_eval_tasks // 3,
            output_dir=output_dir,
        )

        refinement_results = run_refinement_phase(config, dry_run=args.dry_run)

    # Phase 2: Validation
    if not args.refinement_only:
        validation_results = run_validation_phase(
            output_dir=output_dir,
            n_top=args.n_top,
            k_folds=args.k_folds,
            model=args.model,
            dry_run=args.dry_run,
        )

    # Print final summary
    print_final_summary(refinement_results, validation_results)

    # Save combined results
    if not args.dry_run:
        results_path = output_dir / f"full_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        combined_results = {
            "refinement": refinement_results,
            "validation": validation_results,
        }

        # Convert to JSON-serializable
        def convert(obj):
            if isinstance(obj, Path):
                return str(obj)
            if hasattr(obj, 'to_dict'):
                return obj.to_dict()
            if hasattr(obj, '__dict__'):
                return obj.__dict__
            return str(obj)

        try:
            with open(results_path, "w") as f:
                json.dump(combined_results, f, indent=2, default=convert)
            print(f"\nResults saved to: {results_path}")
        except Exception as e:
            print(f"Warning: Could not save results: {e}")


if __name__ == "__main__":
    main()
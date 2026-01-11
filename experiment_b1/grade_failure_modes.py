"""
Experiment B.1: Grade failure modes using Lunette.

Uses the pre-uploaded annotated trajectories to classify failure modes,
then evaluates against manual annotations from IssueSolvingEmpirical.

Usage:
    # Dry run to see what would be graded
    python experiment_b1/grade_failure_modes.py --dry_run

    # Grade all tasks
    python experiment_b1/grade_failure_modes.py

    # Grade with limit
    python experiment_b1/grade_failure_modes.py --limit 20
"""

import argparse
import asyncio
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from lunette import LunetteClient
from lunette.analysis import GradingPlan

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "chris_output" / "experiment_b1"
UPLOAD_TRACKING = OUTPUT_DIR / "annotated_trajectories_upload.json"

# All 25 failure categories
FAILURE_CATEGORIES = [
    "issue_misleading", "keywords_location", "referred_code_location", "error_stack_trace",
    "specific_case_overfitting", "evasive_repair", "redundant_erroneous_implementation",
    "algorithmic_implementation", "control_flow", "boundary_handling", "data_processing_errors",
    "insufficient_domain_knowledge", "inheritance", "interface_contract", "component_coordination",
    "recurring_pattern", "issue_interference", "reproduction_script_failure",
    "reproduction_output_misreading", "insufficient_verification_capability",
    "non_progressive_iteration", "blindly_strategy_switching", "verification_abandonment",
    "verification_weakening", "context_forgetting",
]

# Category aliases and variations for parsing
CATEGORY_ALIASES = {
    "keyword": "keywords_location",
    "keywords": "keywords_location",
    "keyword_location": "keywords_location",
    "stack_trace": "error_stack_trace",
    "stacktrace": "error_stack_trace",
    "overfitting": "specific_case_overfitting",
    "case_overfitting": "specific_case_overfitting",
    "domain_knowledge": "insufficient_domain_knowledge",
    "verification_failure": "insufficient_verification_capability",
    "reproduction_failure": "reproduction_script_failure",
    "output_misreading": "reproduction_output_misreading",
    "misreading": "reproduction_output_misreading",
    "non_progressive": "non_progressive_iteration",
    "progressive_iteration": "non_progressive_iteration",
    "strategy_switching": "blindly_strategy_switching",
    "blind_switching": "blindly_strategy_switching",
    "abandonment": "verification_abandonment",
    "weakening": "verification_weakening",
    "forgetting": "context_forgetting",
    "amnesia": "context_forgetting",
}

def parse_failure_modes_from_explanation(explanation: str) -> list[str]:
    """Extract failure mode categories mentioned in the explanation text."""
    if not explanation:
        return []

    explanation_lower = explanation.lower()
    found = set()

    # Check for exact category matches
    for category in FAILURE_CATEGORIES:
        # Check both underscore and space versions
        if category in explanation_lower or category.replace("_", " ") in explanation_lower:
            found.add(category)

    # Check for aliases
    for alias, category in CATEGORY_ALIASES.items():
        if alias in explanation_lower:
            found.add(category)

    return list(found)


FAILURE_MODE_GRADING_PROMPT = """Analyze this FAILED SWE-bench trajectory and identify failure modes.

CATEGORIES (use exact names):
A. Localization: issue_misleading, keywords_location, referred_code_location, error_stack_trace
B1. Strategy: specific_case_overfitting, evasive_repair, redundant_erroneous_implementation
B2. Implementation: algorithmic_implementation, control_flow, boundary_handling, data_processing_errors, insufficient_domain_knowledge
B3. Incomplete: inheritance, interface_contract, component_coordination, recurring_pattern, issue_interference
C1. Verification: reproduction_script_failure, reproduction_output_misreading, insufficient_verification_capability
C2. Iteration: non_progressive_iteration, blindly_strategy_switching
C3. Retreat: verification_abandonment, verification_weakening
C4. Context: context_forgetting

List ALL applicable categories with brief justification. Use the exact category names above.
"""


def load_upload_tracking() -> dict:
    """Load the upload tracking file."""
    with open(UPLOAD_TRACKING) as f:
        return json.load(f)


async def grade_single_trajectory_run(
    client: LunetteClient,
    run_id: str,
    task_id: str,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Grade a single-trajectory run with retry logic."""
    last_error = None

    for attempt in range(max_retries):
        try:
            results = await client.investigate(
                run_id=run_id,
                plan=GradingPlan(
                    name="failure-mode-classification",
                    prompt=FAILURE_MODE_GRADING_PROMPT,
                ),
                limit=1,
            )

            if not results.results:
                return {"task_id": task_id, "run_id": run_id, "error": "No results"}

            result = results.results[0]
            explanation = result.data.get("explanation", "") if result.data else ""
            failure_modes = parse_failure_modes_from_explanation(explanation)

            return {
                "task_id": task_id,
                "trajectory_id": result.original_trajectory_id,
                "run_id": run_id,
                "failure_modes": failure_modes,
                "explanation": explanation,
                "score": result.data.get("score") if result.data else None,
            }

        except Exception as e:
            last_error = str(e)
            if "504" in last_error or "timeout" in last_error.lower():
                # Exponential backoff for timeouts
                wait_time = (2 ** attempt) * 5  # 5s, 10s, 20s
                await asyncio.sleep(wait_time)
            else:
                # Non-timeout error, don't retry
                break

    return {"task_id": task_id, "run_id": run_id, "error": last_error}


def compute_metrics(predictions: list[dict]) -> dict:
    """Compute multi-label classification metrics."""
    valid = [p for p in predictions if "error" not in p and p.get("predicted")]

    if not valid:
        return {"error": "No valid predictions"}

    jaccard_scores = []
    exact_matches = 0
    any_overlaps = 0

    category_tp = defaultdict(int)
    category_fp = defaultdict(int)
    category_fn = defaultdict(int)

    for pred in valid:
        gt_set = set(pred["ground_truth"])
        pred_set = set(pred["predicted"])

        # Jaccard (IoU)
        if gt_set or pred_set:
            jaccard = len(gt_set & pred_set) / len(gt_set | pred_set)
        else:
            jaccard = 1.0
        jaccard_scores.append(jaccard)

        # Exact match
        if gt_set == pred_set:
            exact_matches += 1

        # Any overlap
        if gt_set & pred_set:
            any_overlaps += 1

        # Per-category
        for cat in gt_set & pred_set:
            category_tp[cat] += 1
        for cat in pred_set - gt_set:
            category_fp[cat] += 1
        for cat in gt_set - pred_set:
            category_fn[cat] += 1

    # Per-category precision/recall/F1
    category_metrics = {}
    for cat in set(category_tp.keys()) | set(category_fp.keys()) | set(category_fn.keys()):
        tp = category_tp[cat]
        fp = category_fp[cat]
        fn = category_fn[cat]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        category_metrics[cat] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
        }

    # Micro-averaged
    total_tp = sum(category_tp.values())
    total_fp = sum(category_fp.values())
    total_fn = sum(category_fn.values())

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

    return {
        "n_samples": len(valid),
        "jaccard_mean": float(np.mean(jaccard_scores)),
        "jaccard_std": float(np.std(jaccard_scores)),
        "exact_match_rate": exact_matches / len(valid),
        "any_overlap_rate": any_overlaps / len(valid),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "category_metrics": category_metrics,
    }


async def main():
    parser = argparse.ArgumentParser(description="Grade failure modes using Lunette")
    parser.add_argument("--dry_run", action="store_true", help="Show what would be graded")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of runs to grade")
    args = parser.parse_args()

    # Load tracking data
    print("Loading upload tracking...")
    tracking = load_upload_tracking()
    run_ids = tracking["run_ids"]
    tasks = tracking["tasks"]

    # Build task_id -> ground_truth mapping
    task_to_info = {t["task_id"]: t for t in tasks}

    print(f"Found {len(tasks)} annotated trajectories across {len(run_ids)} runs")

    if args.dry_run:
        print("\n=== DRY RUN ===")
        print(f"Would grade {len(run_ids)} runs containing {len(tasks)} trajectories")
        print("\nSample tasks:")
        for t in tasks[:5]:
            print(f"  {t['task_id']} ({t['paper_agent']})")
            print(f"    Ground truth: {t['ground_truth']}")
        return

    # Apply limit to tasks (each task has its own run)
    tasks_to_grade = tasks
    if args.limit:
        tasks_to_grade = tasks[:args.limit]
        print(f"\nLimiting to {len(tasks_to_grade)} tasks")

    # Grade each task (1 trajectory per run)
    print("\nGrading tasks...")
    predictions = []

    async with LunetteClient() as client:
        for i, task_info in enumerate(tasks_to_grade):
            task_id = task_info["task_id"]
            run_id = task_info["run_id"]

            print(f"\n[{i+1}/{len(tasks_to_grade)}] Grading {task_id}... ", end="", flush=True)

            result = await grade_single_trajectory_run(client, run_id, task_id)

            prediction = {
                "task_id": task_id,
                "agent": task_info.get("paper_agent"),
                "ground_truth": task_info.get("ground_truth", []),
                "run_id": run_id,
            }

            if "error" in result:
                prediction["error"] = result["error"]
                print(f"ERROR: {result['error'][:50]}")
            else:
                prediction["predicted"] = result.get("failure_modes", [])
                prediction["explanation"] = result.get("explanation", "")
                print(f"OK - {len(prediction['predicted'])} modes")

            predictions.append(prediction)

            # Save incrementally
            with open(OUTPUT_DIR / "predictions_latest.json", "w") as f:
                json.dump(predictions, f, indent=2)

    print(f"\n\nGraded {len(predictions)} trajectories total")

    # Compute metrics
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    metrics = compute_metrics(predictions)

    if "error" in metrics:
        print(f"Error: {metrics['error']}")
        return

    print(f"\nSamples evaluated: {metrics['n_samples']}")
    print(f"\nOverall Metrics:")
    print(f"  Jaccard Similarity (IoU): {metrics['jaccard_mean']:.3f} ± {metrics['jaccard_std']:.3f}")
    print(f"  Exact Match Rate: {metrics['exact_match_rate']:.3f}")
    print(f"  Any Overlap Rate: {metrics['any_overlap_rate']:.3f}")
    print(f"\nMicro-averaged:")
    print(f"  Precision: {metrics['micro_precision']:.3f}")
    print(f"  Recall: {metrics['micro_recall']:.3f}")
    print(f"  F1: {metrics['micro_f1']:.3f}")

    print(f"\nPer-Category Performance (sorted by support):")
    cat_metrics = metrics["category_metrics"]
    sorted_cats = sorted(cat_metrics.items(), key=lambda x: x[1]["support"], reverse=True)
    for cat, m in sorted_cats[:15]:
        print(f"  {cat:40s} P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f} (n={m['support']})")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(OUTPUT_DIR / f"predictions_{timestamp}.json", "w") as f:
        json.dump(predictions, f, indent=2)

    with open(OUTPUT_DIR / f"metrics_{timestamp}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())

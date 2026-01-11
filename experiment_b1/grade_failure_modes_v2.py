"""
Experiment B.1: Grade failure modes using Lunette (V2 - Batch approach).

Uploads trajectories in batches and uses trajectory_id mapping to match results.

Usage:
    # Dry run to see what would be graded
    python experiment_b1/grade_failure_modes_v2.py --dry_run

    # Grade all tasks
    python experiment_b1/grade_failure_modes_v2.py

    # Grade with limit
    python experiment_b1/grade_failure_modes_v2.py --limit 20
"""

import argparse
import asyncio
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from lunette import LunetteClient
from lunette.analysis import GradingPlan
from lunette.models.run import Run
from lunette.models.trajectory import Trajectory, ScalarScore
from lunette.models.messages import UserMessage, AssistantMessage

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ISSUE_SOLVING_DIR = PROJECT_ROOT / "IssueSolvingEmpirical" / "dataset"
UNIFIED_TRAJS_DIR = PROJECT_ROOT / "trajectory_data" / "unified_trajs"
OUTPUT_DIR = PROJECT_ROOT / "chris_output" / "experiment_b1"

# Agent mapping
AGENT_MAPPING = {
    "openhands": "20241029_OpenHands-CodeAct-2.1-sonnet-20241022",
    "agentless": "20241028_agentless-1.5_gpt4o",
}

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

# Category aliases for parsing
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


def parse_failure_modes_from_explanation(explanation: str) -> list[str]:
    """Extract failure mode categories mentioned in the explanation text."""
    if not explanation:
        return []

    explanation_lower = explanation.lower()
    found = set()

    # Check for exact category matches
    for category in FAILURE_CATEGORIES:
        if category in explanation_lower or category.replace("_", " ") in explanation_lower:
            found.add(category)

    # Check for aliases
    for alias, category in CATEGORY_ALIASES.items():
        if alias in explanation_lower:
            found.add(category)

    return list(found)


def load_annotations(agent_name: str) -> dict[str, list[str]]:
    """Load annotations for an agent."""
    annotation_file = ISSUE_SOLVING_DIR / f"annotations_{agent_name}.json"

    try:
        with open(annotation_file) as f:
            raw = json.load(f)
    except json.JSONDecodeError:
        with open(annotation_file) as f:
            content = f.read()
        task_pattern = r'"([a-z_]+__[a-z_]+-\d+)":\s*\{([^}]+)\}'
        category_pattern = r'"category":\s*"([^"]+)"'
        raw = {}
        for match in re.finditer(task_pattern, content, re.DOTALL):
            task_id = match.group(1)
            task_content = match.group(2)
            categories = re.findall(category_pattern, task_content)
            if categories:
                raw[task_id] = {"extracted": [{"category": c} for c in categories]}

    result = {}
    for task_id, action_annotations in raw.items():
        categories = set()
        for action_data in action_annotations.values():
            if isinstance(action_data, list):
                for ann in action_data:
                    if "category" in ann:
                        categories.add(ann["category"])
        result[task_id] = list(categories)

    return result


def load_unified_trajectory(agent_dir: str, task_id: str) -> dict | None:
    """Load a unified trajectory JSON file."""
    traj_file = UNIFIED_TRAJS_DIR / agent_dir / f"{task_id}.json"
    if not traj_file.exists():
        return None
    with open(traj_file) as f:
        return json.load(f)


def convert_to_lunette_trajectory(task_id: str, unified_traj: dict) -> Trajectory:
    """Convert unified trajectory format to Lunette Trajectory."""
    messages = []

    for i, msg in enumerate(unified_traj.get("messages", [])):
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "user":
            messages.append(UserMessage(position=i, content=content))
        elif role == "assistant":
            messages.append(AssistantMessage(position=i, content=content))

    resolved = unified_traj.get("resolved", False)
    scores = {"resolved": ScalarScore(value=1.0 if resolved else 0.0)}

    return Trajectory(
        sample=task_id,
        messages=messages,
        scores=scores,
        metadata={"task_id": task_id},
    )


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


async def upload_and_grade_batch(
    client: LunetteClient,
    tasks_batch: list[dict],
    batch_idx: int,
    total_batches: int,
) -> list[dict]:
    """Upload a batch of trajectories and grade them."""
    # Convert to Lunette trajectories
    trajectories = []
    task_id_by_idx = {}  # Index -> task_id mapping

    for idx, task in enumerate(tasks_batch):
        traj = convert_to_lunette_trajectory(task["task_id"], task["unified_traj"])
        trajectories.append(traj)
        task_id_by_idx[idx] = task["task_id"]

    # Upload
    run = Run(
        task="failure-mode-classification",
        model="annotated-trajectories",
        trajectories=trajectories,
    )
    run_meta = await client.save_run(run)
    run_id = run_meta["run_id"]

    print(f"  Batch {batch_idx+1}/{total_batches}: Uploaded {len(trajectories)} trajectories -> run {run_id[:16]}...")

    # Grade
    try:
        results = await client.investigate(
            run_id=run_id,
            plan=GradingPlan(
                name="failure-mode-classification",
                prompt=FAILURE_MODE_GRADING_PROMPT,
            ),
        )
    except Exception as e:
        return [{"task_id": t["task_id"], "error": str(e)} for t in tasks_batch]

    # Match results to tasks by order (trajectories are returned in order)
    predictions = []
    for idx, result in enumerate(results.results):
        if idx >= len(tasks_batch):
            break

        task = tasks_batch[idx]
        explanation = result.data.get("explanation", "") if result.data else ""
        failure_modes = parse_failure_modes_from_explanation(explanation)

        predictions.append({
            "task_id": task["task_id"],
            "trajectory_id": result.original_trajectory_id,
            "run_id": run_id,
            "predicted": failure_modes,
            "explanation": explanation,
        })

    return predictions


async def main():
    parser = argparse.ArgumentParser(description="Grade failure modes using Lunette (V2)")
    parser.add_argument("--dry_run", action="store_true", help="Show what would be graded")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tasks to grade")
    parser.add_argument("--batch_size", type=int, default=20, help="Trajectories per batch")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all annotated task IDs
    print("Collecting annotated tasks...")
    tasks_to_grade = []

    for paper_agent, our_agent in AGENT_MAPPING.items():
        annotations = load_annotations(paper_agent)
        print(f"  {paper_agent}: {len(annotations)} annotated tasks")

        for task_id, ground_truth in annotations.items():
            unified_traj = load_unified_trajectory(our_agent, task_id)
            if unified_traj:
                tasks_to_grade.append({
                    "task_id": task_id,
                    "paper_agent": paper_agent,
                    "our_agent": our_agent,
                    "ground_truth": ground_truth,
                    "unified_traj": unified_traj,
                })

    print(f"\nTotal trajectories to grade: {len(tasks_to_grade)}")

    if args.dry_run:
        print("\n=== DRY RUN ===")
        print(f"Would grade {len(tasks_to_grade)} trajectories in batches of {args.batch_size}")
        print("\nSample tasks:")
        for t in tasks_to_grade[:5]:
            print(f"  {t['task_id']} ({t['paper_agent']})")
            print(f"    Ground truth: {t['ground_truth']}")
        return

    # Apply limit
    if args.limit:
        tasks_to_grade = tasks_to_grade[:args.limit]
        print(f"\nLimiting to {len(tasks_to_grade)} tasks")

    # Build task_id -> ground_truth mapping
    task_to_info = {t["task_id"]: t for t in tasks_to_grade}

    # Grade in batches
    print(f"\nGrading in batches of {args.batch_size}...")
    all_predictions = []

    async with LunetteClient() as client:
        num_batches = (len(tasks_to_grade) + args.batch_size - 1) // args.batch_size

        for batch_idx in range(num_batches):
            start = batch_idx * args.batch_size
            end = min(start + args.batch_size, len(tasks_to_grade))
            batch = tasks_to_grade[start:end]

            batch_predictions = await upload_and_grade_batch(
                client, batch, batch_idx, num_batches
            )

            # Add ground truth and agent info
            for pred in batch_predictions:
                task_info = task_to_info.get(pred["task_id"], {})
                pred["agent"] = task_info.get("paper_agent")
                pred["ground_truth"] = task_info.get("ground_truth", [])

            all_predictions.extend(batch_predictions)

            # Save incrementally
            with open(OUTPUT_DIR / "predictions_v2_latest.json", "w") as f:
                json.dump(all_predictions, f, indent=2)

            print(f"    Graded {len(batch)} tasks, {len(all_predictions)} total")

    print(f"\n\nGraded {len(all_predictions)} trajectories total")

    # Compute metrics
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    metrics = compute_metrics(all_predictions)

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

    with open(OUTPUT_DIR / f"predictions_v2_{timestamp}.json", "w") as f:
        json.dump(all_predictions, f, indent=2)

    with open(OUTPUT_DIR / f"metrics_v2_{timestamp}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())

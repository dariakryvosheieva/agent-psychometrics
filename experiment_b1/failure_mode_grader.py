"""
Experiment B.1: Lunette-based failure mode classification.

This script uses Lunette to classify failure modes in SWE-bench agent trajectories,
then evaluates against manual annotations from the IssueSolvingEmpirical paper.

Usage:
    # Dry run to see task matching
    python experiment_b1/failure_mode_grader.py --dry_run

    # Grade a small sample
    python experiment_b1/failure_mode_grader.py --limit 10

    # Full run on all annotated tasks
    python experiment_b1/failure_mode_grader.py
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
import pandas as pd

# Lunette imports
from lunette import LunetteClient
from lunette.analysis import GradingPlan

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ISSUE_SOLVING_DIR = PROJECT_ROOT / "IssueSolvingEmpirical" / "dataset"
UNIFIED_TRAJS_DIR = PROJECT_ROOT / "trajectory_data" / "unified_trajs"
OUTPUT_DIR = PROJECT_ROOT / "chris_output" / "experiment_b1"

# Agent mapping: IssueSolvingEmpirical name -> our unified_trajs agent name
AGENT_MAPPING = {
    "openhands": "20241029_OpenHands-CodeAct-2.1-sonnet-20241022",
    "agentless": "20241028_agentless-1.5_gpt4o",
    # "tools": "20241022_tools_claude-3-5-haiku",  # Not uploaded to Lunette yet
}

# All 25 failure categories from the paper's taxonomy
FAILURE_CATEGORIES = {
    # A. Localization Failures
    "issue_misleading": "A.1 Issue Misleading - Problem description contains misleading information",
    "keywords_location": "A.2.1 Keywords - Agent matches superficial keywords without understanding",
    "referred_code_location": "A.2.2 Referred Code - Agent fixates on code mentioned in issue but irrelevant to fix",
    "error_stack_trace": "A.2.3 Error Stack Trace - Agent focuses on stack trace location but root cause is elsewhere",

    # B. Repair Failures - B.1 Fix Strategy Defects
    "specific_case_overfitting": "B.1.1 Specific Case Overfitting - Fix only handles the example case, not general problem",
    "evasive_repair": "B.1.2 Evasive Repair - Fix suppresses error/symptom without addressing root cause",
    "redundant_erroneous_implementation": "B.1.3 Redundant Erroneous Implementation - Added code conflicts with existing logic",

    # B. Repair Failures - B.2 Implementation Details Defects
    "algorithmic_implementation": "B.2.1.1 Algorithmic Implementation - Algorithm logic is incorrect",
    "control_flow": "B.2.1.2 Control Flow - Incorrect conditions, loops, or branching",
    "boundary_handling": "B.2.1.3 Boundary Handling - Edge cases not properly handled",
    "data_processing_errors": "B.2.2 Data Processing Errors - Wrong data types, transformations, or parsing",
    "insufficient_domain_knowledge": "B.2.3 Insufficient Domain Knowledge - Lacks framework/library expertise",

    # B. Repair Failures - B.3 Incomplete Repair
    "inheritance": "B.3.1.1 Inheritance - Failed to update subclasses or parent classes",
    "interface_contract": "B.3.1.2 Interface Contract - Violated API contracts or interface requirements",
    "component_coordination": "B.3.1.3 Component Coordination - Multiple components need coordinated changes",
    "recurring_pattern": "B.3.2 Recurring Pattern - Same bug exists in multiple places, only fixed some",
    "issue_interference": "B.3.3 Issue Interference - Fix for one aspect breaks another aspect",

    # C. Iterative Verification Failures
    "reproduction_script_failure": "C.1.1 Reproduction Script Failure - Failed to create working reproduction",
    "reproduction_output_misreading": "C.1.2 Reproduction Output Misreading - Misinterpreted test/reproduction output",
    "insufficient_verification_capability": "C.1.3 Insufficient Verification - Cannot properly verify fix correctness",
    "non_progressive_iteration": "C.2.1 Non-Progressive Iteration - Agent loops without making progress",
    "blindly_strategy_switching": "C.2.2 Blindly Strategy Switching - Randomly tries different approaches",
    "verification_abandonment": "C.3.1 Verification Abandonment - Stops verifying and declares success prematurely",
    "verification_weakening": "C.3.2 Verification Weakening - Modifies test to pass instead of fixing code",
    "context_forgetting": "C.4 Context Forgetting - Loses track of previous findings or attempts",
}

# Build the grading prompt
FAILURE_MODE_GRADING_PROMPT = """You are analyzing a FAILED SWE-bench agent trajectory to identify WHY the agent failed.

## Task
Classify the failure modes exhibited in this trajectory. The agent attempted to solve a GitHub issue but failed.
You must identify ALL applicable failure categories from the taxonomy below.

## Failure Mode Taxonomy (25 categories)

### A. Localization Failures (finding the wrong code to modify)
- **issue_misleading**: Problem description contains misleading information that sent agent in wrong direction
- **keywords_location**: Agent matched superficial keywords without understanding the actual problem
- **referred_code_location**: Agent fixated on code mentioned in the issue that's irrelevant to the fix
- **error_stack_trace**: Agent focused on stack trace location but the root cause is elsewhere

### B. Repair Failures

#### B.1 Fix Strategy Defects (wrong approach to fixing)
- **specific_case_overfitting**: Fix only handles the specific example case, doesn't generalize
- **evasive_repair**: Fix suppresses the error/symptom without addressing the root cause
- **redundant_erroneous_implementation**: Added code that conflicts with or duplicates existing logic

#### B.2 Implementation Details Defects (correct approach, wrong execution)
- **algorithmic_implementation**: The algorithm logic itself is incorrect
- **control_flow**: Incorrect conditions, loops, or branching logic
- **boundary_handling**: Edge cases or boundary conditions not properly handled
- **data_processing_errors**: Wrong data types, transformations, or parsing
- **insufficient_domain_knowledge**: Agent lacks necessary framework/library/domain expertise

#### B.3 Incomplete Repair (partial fix)
- **inheritance**: Failed to update related subclasses or parent classes
- **interface_contract**: Violated API contracts or interface requirements
- **component_coordination**: Multiple components need coordinated changes, only some were made
- **recurring_pattern**: Same bug pattern exists in multiple places, only fixed some occurrences
- **issue_interference**: Fixing one aspect of the issue broke another aspect

### C. Iterative Verification Failures (problems in testing/validation loop)

#### C.1 Reproduction/Verification Failure
- **reproduction_script_failure**: Failed to create a working reproduction script
- **reproduction_output_misreading**: Misinterpreted test output or reproduction results
- **insufficient_verification_capability**: Cannot properly verify if fix is correct

#### C.2 Iteration Anomalies
- **non_progressive_iteration**: Agent loops through attempts without making meaningful progress
- **blindly_strategy_switching**: Randomly tries different approaches without learning from failures

#### C.3 Validation Retreat
- **verification_abandonment**: Stops verifying and declares success prematurely
- **verification_weakening**: Modifies the test/reproduction to pass instead of fixing the actual code

#### C.4 Context Amnesia
- **context_forgetting**: Agent loses track of previous findings, attempts, or important context

## Instructions

1. Carefully read the agent trajectory (all actions, thoughts, and observations)
2. Identify the PRIMARY failure modes - the main reasons why this agent failed
3. You may select MULTIPLE categories if multiple failure modes are present
4. Only select categories where there is clear evidence in the trajectory

## Response Format

Return a JSON object with:
- "failure_modes": List of category names (from the 25 categories above)
- "reasoning": Brief explanation (2-4 sentences) of why you selected each category
- "primary_failure": The single most important failure mode

Example:
{
    "failure_modes": ["keywords_location", "verification_weakening", "control_flow"],
    "reasoning": "The agent initially located the wrong file by keyword matching ('values_select'). After multiple failed attempts, it modified the test script to pass instead of fixing the actual bug. The condition logic in the fix was also incorrect.",
    "primary_failure": "keywords_location"
}
"""


def load_annotations(agent_name: str) -> dict[str, list[str]]:
    """Load annotations for an agent from IssueSolvingEmpirical.

    Returns: dict mapping task_id -> list of failure category names
    """
    annotation_file = ISSUE_SOLVING_DIR / f"annotations_{agent_name}.json"

    try:
        with open(annotation_file) as f:
            raw_annotations = json.load(f)
    except json.JSONDecodeError:
        # Handle malformed JSON (agentless has issues)
        with open(annotation_file) as f:
            content = f.read()
        # Extract task IDs and categories via regex
        task_pattern = r'"([a-z_]+__[a-z_]+-\d+)":\s*\{([^}]+)\}'
        category_pattern = r'"category":\s*"([^"]+)"'

        raw_annotations = {}
        for match in re.finditer(task_pattern, content, re.DOTALL):
            task_id = match.group(1)
            task_content = match.group(2)
            categories = re.findall(category_pattern, task_content)
            if categories:
                raw_annotations[task_id] = {"extracted": [{"category": c} for c in categories]}

    # Convert to task_id -> list of categories
    result = {}
    for task_id, action_annotations in raw_annotations.items():
        categories = set()
        for action_data in action_annotations.values():
            if isinstance(action_data, list):
                for ann in action_data:
                    if "category" in ann:
                        categories.add(ann["category"])
        result[task_id] = list(categories)

    return result


def load_lunette_upload_info(agent_dir: str) -> dict:
    """Load Lunette uploads info for an agent."""
    upload_file = UNIFIED_TRAJS_DIR / agent_dir / "_lunette_uploads.json"

    with open(upload_file) as f:
        return json.load(f)


async def build_task_to_run_mapping(
    client: LunetteClient,
    run_ids: list[str],
) -> dict[str, str]:
    """Build mapping from task_id to run_id by querying Lunette API.

    Args:
        client: LunetteClient instance.
        run_ids: List of run IDs to check.

    Returns:
        Dict mapping task_id -> run_id.
    """
    task_to_run = {}

    print(f"  Building task-to-run mapping from {len(run_ids)} runs...")
    for i, run_id in enumerate(run_ids):
        try:
            run = await client.get_run(run_id)
            # run.trajectories is a list of Trajectory objects
            # Each trajectory has a 'sample' field which is the task_id
            for traj in run.trajectories:
                task_to_run[traj.sample] = run_id
            print(f"    Run {i+1}/{len(run_ids)}: {len(run.trajectories)} tasks")
        except Exception as e:
            print(f"    Warning: Failed to fetch run {run_id[:16]}...: {e}")
            continue

    return task_to_run


async def prepare_grading_tasks(client: LunetteClient) -> list[dict]:
    """Prepare list of tasks to grade with their ground truth labels."""
    tasks = []

    for paper_agent, our_agent in AGENT_MAPPING.items():
        print(f"\nProcessing {paper_agent} ({our_agent})...")

        # Load ground truth annotations
        annotations = load_annotations(paper_agent)
        print(f"  Loaded {len(annotations)} annotated tasks")

        # Load Lunette upload info
        try:
            upload_info = load_lunette_upload_info(our_agent)
        except FileNotFoundError:
            print(f"  Warning: No Lunette uploads for {our_agent}")
            continue

        # Check if augmented mappings exist
        if "task_to_run_map" in upload_info:
            task_to_run = upload_info["task_to_run_map"]
            print(f"  Using cached task-to-run mapping ({len(task_to_run)} tasks)")
        else:
            # Build mapping by querying Lunette API
            run_ids = upload_info.get("run_ids", [])
            if not run_ids:
                print(f"  Warning: No run_ids found for {our_agent}")
                continue
            task_to_run = await build_task_to_run_mapping(client, run_ids)
            print(f"  Built task-to-run mapping ({len(task_to_run)} tasks)")

        # Match tasks
        matched = 0
        for task_id, ground_truth_categories in annotations.items():
            if task_id in task_to_run:
                tasks.append({
                    "task_id": task_id,
                    "agent": paper_agent,
                    "our_agent": our_agent,
                    "run_id": task_to_run[task_id],
                    "ground_truth": ground_truth_categories,
                })
                matched += 1

        print(f"  Matched {matched}/{len(annotations)} annotated tasks to Lunette runs")

    return tasks


def build_trajectory_id_to_task_mapping(agent_dir: str) -> dict[str, str]:
    """Build mapping from trajectory_id (UUID) to task_id using upload tracking file."""
    upload_file = UNIFIED_TRAJS_DIR / agent_dir / "_lunette_uploads.json"

    with open(upload_file) as f:
        data = json.load(f)

    # Build trajectory_id -> task_id mapping
    mapping = {}
    for traj in data.get("trajectories", []):
        traj_id = traj.get("trajectory_id")
        task_id = traj.get("task_id")
        if traj_id and task_id:
            mapping[traj_id] = task_id

    return mapping


async def grade_run(
    client: LunetteClient,
    run_id: str,
    target_tasks: set[str],
    trajectory_id_to_task: dict[str, str],
    max_trajectories: int = 200,
) -> list[dict[str, Any]]:
    """Grade all trajectories in a run and filter to target tasks.

    Args:
        client: LunetteClient instance.
        run_id: Lunette run ID.
        target_tasks: Set of task_ids we want to grade.
        trajectory_id_to_task: Mapping from trajectory UUID to task_id.
        max_trajectories: Max trajectories to grade in this run.

    Returns:
        List of grading results for target tasks only.
    """
    try:
        # First get the run to count trajectories
        run = await client.get_run(run_id)

        # Count how many target tasks are in this run
        target_in_run = sum(1 for traj in run.trajectories if traj.sample in target_tasks)
        if target_in_run == 0:
            return []

        print(f"    Run has {target_in_run} target tasks out of {len(run.trajectories)} total")

        # Grade the run
        results = await client.investigate(
            run_id=run_id,
            plan=GradingPlan(
                name="failure-mode-classification",
                prompt=FAILURE_MODE_GRADING_PROMPT,
            ),
            limit=min(len(run.trajectories), max_trajectories),
        )

        if not results.results:
            return [{"error": "No results returned", "run_id": run_id}]

        # Match results to task_ids using original_trajectory_id
        graded = []
        for result in results.results:
            traj_id = result.original_trajectory_id
            task_id = trajectory_id_to_task.get(traj_id)

            if task_id and task_id in target_tasks:
                result_data = result.data.copy()
                result_data["task_id"] = task_id
                result_data["run_id"] = run_id
                result_data["trajectory_id"] = traj_id
                graded.append(result_data)

        return graded

    except Exception as e:
        return [{"error": str(e), "run_id": run_id}]


def compute_metrics(predictions: list[dict]) -> dict:
    """Compute multi-label classification metrics."""

    # Filter to successful predictions
    valid = [p for p in predictions if "error" not in p and p.get("predicted")]

    if not valid:
        return {"error": "No valid predictions"}

    # Compute metrics
    jaccard_scores = []
    exact_matches = 0
    any_overlaps = 0

    # Per-category stats
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

    # Compute per-category precision/recall/F1
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
            "support": tp + fn,  # Ground truth count
        }

    # Micro-averaged metrics
    total_tp = sum(category_tp.values())
    total_fp = sum(category_fp.values())
    total_fn = sum(category_fn.values())

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

    return {
        "n_samples": len(valid),
        "jaccard_mean": np.mean(jaccard_scores),
        "jaccard_std": np.std(jaccard_scores),
        "exact_match_rate": exact_matches / len(valid),
        "any_overlap_rate": any_overlaps / len(valid),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "category_metrics": category_metrics,
    }


async def main():
    parser = argparse.ArgumentParser(description="Experiment B.1: Failure mode classification")
    parser.add_argument("--dry_run", action="store_true", help="Show task matching without grading")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tasks to grade")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare tasks (requires Lunette client for mapping)
    print("Preparing grading tasks...")
    async with LunetteClient() as client:
        tasks = await prepare_grading_tasks(client)

    print(f"\nFound {len(tasks)} tasks with ground truth annotations")

    # Show breakdown by agent
    agent_counts = defaultdict(int)
    for t in tasks:
        agent_counts[t["agent"]] += 1
    for agent, count in agent_counts.items():
        print(f"  {agent}: {count} tasks")

    if args.dry_run:
        print("\n=== DRY RUN ===")
        print(f"Would grade {len(tasks)} tasks")
        print("\nSample tasks:")
        for t in tasks[:5]:
            print(f"  {t['task_id']} ({t['agent']})")
            print(f"    Ground truth: {t['ground_truth']}")
            print(f"    Run ID: {t['run_id']}")
        return

    # Apply limit
    if args.limit:
        tasks = tasks[:args.limit]
        print(f"\nLimiting to {len(tasks)} tasks")

    # Group tasks by run_id and agent for efficient grading
    run_to_tasks = defaultdict(list)
    for task in tasks:
        run_to_tasks[task["run_id"]].append(task)

    print(f"\nTasks distributed across {len(run_to_tasks)} runs")

    # Build trajectory_id -> task_id mappings for each agent
    print("\nBuilding trajectory ID mappings...")
    agent_traj_mappings = {}
    for paper_agent, our_agent in AGENT_MAPPING.items():
        try:
            mapping = build_trajectory_id_to_task_mapping(our_agent)
            agent_traj_mappings[our_agent] = mapping
            print(f"  {our_agent}: {len(mapping)} trajectory IDs")
        except FileNotFoundError:
            print(f"  {our_agent}: No upload file found")

    # Grade by run
    print("\nGrading trajectories by run...")
    predictions = []
    task_to_ground_truth = {t["task_id"]: t for t in tasks}

    async with LunetteClient() as client:
        for run_idx, (run_id, run_tasks) in enumerate(run_to_tasks.items()):
            target_tasks = {t["task_id"] for t in run_tasks}
            agent = run_tasks[0]["agent"]
            our_agent = run_tasks[0]["our_agent"]

            # Get the trajectory mapping for this agent
            traj_mapping = agent_traj_mappings.get(our_agent, {})

            print(f"\n[Run {run_idx+1}/{len(run_to_tasks)}] {agent} - {len(target_tasks)} target tasks")
            print(f"  Run ID: {run_id[:16]}...")

            results = await grade_run(client, run_id, target_tasks, traj_mapping)

            for result in results:
                task_id = result.get("task_id")
                if not task_id:
                    continue

                task_info = task_to_ground_truth.get(task_id, {})
                prediction = {
                    "task_id": task_id,
                    "agent": task_info.get("agent"),
                    "run_id": run_id,
                    "ground_truth": task_info.get("ground_truth", []),
                }

                if "error" in result:
                    prediction["error"] = result["error"]
                    print(f"    {task_id}: ERROR - {result['error'][:50]}")
                else:
                    prediction["predicted"] = result.get("failure_modes", [])
                    prediction["primary_failure"] = result.get("primary_failure")
                    prediction["reasoning"] = result.get("reasoning")
                    print(f"    {task_id}: predicted {len(prediction['predicted'])} modes")

                predictions.append(prediction)

            # Save incrementally
            with open(output_dir / "predictions.json", "w") as f:
                json.dump(predictions, f, indent=2)

    print(f"\nGraded {len(predictions)} tasks total")

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

    print(f"\nPer-Category Performance (sorted by F1):")
    cat_metrics = metrics["category_metrics"]
    sorted_cats = sorted(cat_metrics.items(), key=lambda x: x[1]["f1"], reverse=True)
    for cat, m in sorted_cats[:15]:
        print(f"  {cat:40s} P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f} (n={m['support']})")

    # Save full results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(output_dir / f"predictions_{timestamp}.json", "w") as f:
        json.dump(predictions, f, indent=2)

    with open(output_dir / f"metrics_{timestamp}.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())

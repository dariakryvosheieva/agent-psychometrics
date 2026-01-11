"""
Experiment B.1: Grade failure modes with gold patch context.

The trajectory already contains the problem statement in the messages.
We add the gold patch to the grading prompt so the judge can compare.

Usage:
    python experiment_b1/grade_with_gold_patch.py --dry_run
    python experiment_b1/grade_with_gold_patch.py --limit 10
    python experiment_b1/grade_with_gold_patch.py --llm_only --limit 5  # Skip Lunette
"""

import argparse
import asyncio
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import anthropic
import httpx
import numpy as np
from datasets import load_dataset

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


def get_grading_prompt(gold_patch: str) -> str:
    """Generate grading prompt with gold patch included."""
    return f"""You are analyzing a FAILED SWE-bench agent trajectory to identify ROOT CAUSES of failure.

## GOLD PATCH (The Correct Solution)
The agent should have produced something equivalent to this fix:
```diff
{gold_patch}
```

## Failure Mode Categories

### A. LOCALIZATION FAILURES (agent looked in wrong place)
- **issue_misleading**: Problem description contained misleading info that sent agent wrong direction
- **keywords_location**: Agent matched superficial keywords without understanding actual problem
- **referred_code_location**: Agent fixated on code mentioned in issue that's irrelevant to fix
- **error_stack_trace**: Agent focused on stack trace location but root cause is elsewhere

### B. REPAIR FAILURES

#### B.1 Strategy Defects
- **specific_case_overfitting**: Fix only handles example case, doesn't generalize
- **evasive_repair**: Fix suppresses error without addressing root cause
- **redundant_erroneous_implementation**: Added conflicting or duplicate logic

#### B.2 Implementation Defects
- **algorithmic_implementation**: Algorithm logic itself is incorrect
- **control_flow**: Incorrect conditions, loops, or branching
- **boundary_handling**: Edge cases not handled
- **data_processing_errors**: Wrong data types or transformations
- **insufficient_domain_knowledge**: Lacks framework/library expertise

#### B.3 Incomplete Repair
- **inheritance**: Failed to update subclasses/parent classes
- **interface_contract**: Violated API contracts
- **component_coordination**: Multiple components need changes, only some made
- **recurring_pattern**: Same bug in multiple places, only fixed some
- **issue_interference**: Fixing one aspect broke another

### C. VERIFICATION FAILURES

#### C.1 Reproduction/Verification
- **reproduction_script_failure**: Failed to create working reproduction
- **reproduction_output_misreading**: Misinterpreted test output
- **insufficient_verification_capability**: Cannot verify if fix is correct

#### C.2 Iteration Anomalies
- **non_progressive_iteration**: Loops without meaningful progress
- **blindly_strategy_switching**: Randomly tries approaches without learning

#### C.3 Validation Retreat
- **verification_abandonment**: Stops verifying, declares success prematurely
- **verification_weakening**: Modifies test to pass instead of fixing code

#### C.4 Context
- **context_forgetting**: Loses track of previous findings

## Instructions

1. The trajectory already contains the PROBLEM STATEMENT in the first user message
2. Compare what the agent did vs the GOLD PATCH above
3. Identify ROOT CAUSES - why did the agent fail? Not just what went wrong.
4. Check if the problem description itself was misleading (compare to gold patch)
5. Select ALL applicable categories using exact names from above

List the failure mode categories that apply to this trajectory.
"""


def parse_failure_modes(text: str) -> list[str]:
    """Extract failure mode categories from response text."""
    if not text:
        return []
    text_lower = text.lower()
    found = set()
    for category in FAILURE_CATEGORIES:
        if category in text_lower or category.replace("_", " ") in text_lower:
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


def load_swebench_data() -> dict[str, dict]:
    """Load SWE-bench Verified dataset."""
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    return {ex["instance_id"]: ex for ex in ds}


def format_trajectory_text(unified_traj: dict, max_messages: int = 30) -> str:
    """Format trajectory as text for direct LLM."""
    messages = unified_traj.get("messages", [])[:max_messages]
    formatted = []
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    text_parts.append(item["text"])
                elif isinstance(item, str):
                    text_parts.append(item)
            content = "\n".join(text_parts)
        # Truncate very long messages
        if len(content) > 4000:
            content = content[:4000] + "\n... [truncated]"
        formatted.append(f"[{role} {i+1}]:\n{content}")
    return "\n\n---\n\n".join(formatted)


def convert_to_lunette_trajectory(
    task_id: str,
    unified_traj: dict,
    swebench_task: dict | None = None,
) -> Trajectory:
    """Convert unified trajectory to Lunette format with SWE-bench metadata.

    Mimics the real SWE-bench Lunette runs by including patch, test_patch,
    FAIL_TO_PASS, etc. in the metadata.
    """
    messages = []
    for i, msg in enumerate(unified_traj.get("messages", [])):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    text_parts.append(item["text"])
                elif isinstance(item, str):
                    text_parts.append(item)
            content = "\n".join(text_parts)

        if role == "user":
            messages.append(UserMessage(position=i, content=content))
        elif role == "assistant":
            messages.append(AssistantMessage(position=i, content=content))
        elif role == "system":
            messages.append(UserMessage(position=i, content=f"[SYSTEM]: {content}"))

    resolved = unified_traj.get("resolved", False)
    scores = {"resolved": ScalarScore(value=1.0 if resolved else 0.0)}

    # Build metadata mimicking real SWE-bench Lunette runs
    metadata = {"task_id": task_id}
    if swebench_task:
        metadata.update({
            "repo": swebench_task.get("repo", ""),
            "patch": swebench_task.get("patch", ""),  # Gold patch
            "test_patch": swebench_task.get("test_patch", ""),
            "version": swebench_task.get("version", ""),
            "hints_text": swebench_task.get("hints_text", ""),
            "base_commit": swebench_task.get("base_commit", ""),
            "FAIL_TO_PASS": swebench_task.get("FAIL_TO_PASS", []),
            "PASS_TO_PASS": swebench_task.get("PASS_TO_PASS", []),
        })

    return Trajectory(
        sample=task_id,
        messages=messages,
        scores=scores,
        metadata=metadata,
        solution=swebench_task.get("patch", "") if swebench_task else None,
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
        if gt_set or pred_set:
            jaccard = len(gt_set & pred_set) / len(gt_set | pred_set)
        else:
            jaccard = 1.0
        jaccard_scores.append(jaccard)
        if gt_set == pred_set:
            exact_matches += 1
        if gt_set & pred_set:
            any_overlaps += 1
        for cat in gt_set & pred_set:
            category_tp[cat] += 1
        for cat in pred_set - gt_set:
            category_fp[cat] += 1
        for cat in gt_set - pred_set:
            category_fn[cat] += 1

    category_metrics = {}
    for cat in set(category_tp.keys()) | set(category_fp.keys()) | set(category_fn.keys()):
        tp, fp, fn = category_tp[cat], category_fp[cat], category_fn[cat]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        category_metrics[cat] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}

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


def get_api_key() -> str:
    """Get Lunette API key."""
    config_path = Path.home() / ".lunette" / "config.json"
    with open(config_path) as f:
        return json.load(f)["api_key"]


async def delete_lunette_run(http_client: httpx.AsyncClient, run_id: str) -> bool:
    """Delete a Lunette run to maintain hygiene."""
    try:
        r = await http_client.delete(f"/runs/{run_id}")
        return r.status_code == 200
    except Exception:
        return False


async def grade_with_direct_llm(
    task_id: str,
    gold_patch: str,
    trajectory_text: str,
) -> dict:
    """Grade using direct Anthropic API call."""
    client = anthropic.Anthropic()
    prompt = get_grading_prompt(gold_patch)

    full_prompt = f"""{prompt}

## AGENT TRAJECTORY
{trajectory_text}

Identify the failure modes. List all applicable category names from the taxonomy above."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": full_prompt}],
        )
        explanation = response.content[0].text
        failure_modes = parse_failure_modes(explanation)
        return {
            "task_id": task_id,
            "predicted": failure_modes,
            "explanation": explanation,
            "method": "direct_llm",
        }
    except Exception as e:
        return {"task_id": task_id, "error": str(e), "method": "direct_llm"}


async def grade_with_lunette(
    lunette_client: LunetteClient,
    http_client: httpx.AsyncClient,
    task_id: str,
    unified_traj: dict,
    swebench_task: dict,
    timeout: int = 600,
) -> dict:
    """Grade using Lunette with non-blocking API."""
    run_id = None
    gold_patch = swebench_task.get("patch", "")
    try:
        # Upload trajectory with full SWE-bench metadata
        traj = convert_to_lunette_trajectory(task_id, unified_traj, swebench_task)
        run = Run(
            task="failure-mode-grading",
            model="annotated",
            trajectories=[traj],
        )
        run_meta = await lunette_client.save_run(run)
        run_id = run_meta["run_id"]

        # Launch non-blocking investigation
        prompt = get_grading_prompt(gold_patch)
        plan = GradingPlan(name="failure-mode", prompt=prompt)

        r = await http_client.post(
            "/investigations/run",
            json={
                "plan": plan.model_dump(mode="python"),
                "run_id": run_id,
                "blocking": False,
            },
            timeout=60.0,
        )
        r.raise_for_status()
        inv_id = r.json()["run_id"]

        # Poll for results with progress updates
        elapsed = 0
        poll_interval = 15
        while elapsed < timeout:
            try:
                r = await http_client.get(f"/investigations/{inv_id}/results", timeout=60.0)
                if r.status_code == 200:
                    data = r.json()
                    results = data.get("results", [])
                    if results:
                        result = results[0]
                        explanation = result.get("data", {}).get("explanation", "")
                        failure_modes = parse_failure_modes(explanation)

                        # Clean up the run
                        await delete_lunette_run(http_client, run_id)

                        return {
                            "task_id": task_id,
                            "predicted": failure_modes,
                            "explanation": explanation,
                            "method": "lunette",
                        }
                    if elapsed > 0 and elapsed % 60 == 0:
                        print(f"      Still waiting... {elapsed}s elapsed")
            except Exception as e:
                print(f"      Poll error: {e}")

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        # Clean up on timeout
        if run_id:
            await delete_lunette_run(http_client, run_id)
        return {"task_id": task_id, "error": f"Timeout after {timeout}s", "method": "lunette"}

    except Exception as e:
        if run_id:
            await delete_lunette_run(http_client, run_id)
        return {"task_id": task_id, "error": str(e), "method": "lunette"}


def print_comparison(task_id: str, ground_truth: list, llm_pred: dict | None, lunette_pred: dict | None):
    """Print comparison of predictions."""
    print(f"\n  Ground truth: {ground_truth}")

    if llm_pred and "error" not in llm_pred:
        overlap = set(llm_pred["predicted"]) & set(ground_truth)
        print(f"  LLM predicted: {llm_pred['predicted']}")
        print(f"  LLM overlap: {list(overlap) if overlap else 'NONE'}")
    elif llm_pred:
        print(f"  LLM error: {llm_pred.get('error', 'unknown')[:50]}")

    if lunette_pred and "error" not in lunette_pred:
        overlap = set(lunette_pred["predicted"]) & set(ground_truth)
        print(f"  Lunette predicted: {lunette_pred['predicted']}")
        print(f"  Lunette overlap: {list(overlap) if overlap else 'NONE'}")
    elif lunette_pred:
        print(f"  Lunette error: {lunette_pred.get('error', 'unknown')[:50]}")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--llm_only", action="store_true", help="Skip Lunette")
    parser.add_argument("--lunette_only", action="store_true", help="Skip direct LLM")
    parser.add_argument("--timeout", type=int, default=600, help="Lunette timeout per task")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load SWE-bench data
    print("Loading SWE-bench Verified data...")
    swebench_data = load_swebench_data()
    print(f"  Loaded {len(swebench_data)} tasks")

    # Collect annotated tasks
    print("\nCollecting annotated tasks...")
    tasks_to_grade = []

    for paper_agent, our_agent in AGENT_MAPPING.items():
        annotations = load_annotations(paper_agent)
        print(f"  {paper_agent}: {len(annotations)} annotated tasks")

        for task_id, ground_truth in annotations.items():
            unified_traj = load_unified_trajectory(our_agent, task_id)
            if unified_traj and task_id in swebench_data:
                tasks_to_grade.append({
                    "task_id": task_id,
                    "paper_agent": paper_agent,
                    "ground_truth": ground_truth,
                    "unified_traj": unified_traj,
                    "swebench_task": swebench_data[task_id],
                    "gold_patch": swebench_data[task_id]["patch"],
                })

    print(f"\nTotal tasks with all data: {len(tasks_to_grade)}")

    if args.dry_run:
        print("\n=== DRY RUN ===")
        print(f"Would grade {len(tasks_to_grade)} tasks")
        for t in tasks_to_grade[:3]:
            print(f"\n  {t['task_id']}:")
            print(f"    Ground truth: {t['ground_truth']}")
            print(f"    Gold patch (first 200 chars): {t['gold_patch'][:200]}...")
        return

    if args.limit:
        tasks_to_grade = tasks_to_grade[:args.limit]
        print(f"\nLimiting to {len(tasks_to_grade)} tasks")

    # Results
    llm_predictions = []
    lunette_predictions = []

    # Process
    api_key = get_api_key()

    async with LunetteClient() as lunette_client:
        async with httpx.AsyncClient(
            base_url="https://lunette.dev/api",
            headers={"X-API-Key": api_key},
            timeout=120.0,
        ) as http_client:

            for i, task in enumerate(tasks_to_grade):
                task_id = task["task_id"]
                print(f"\n[{i+1}/{len(tasks_to_grade)}] {task_id}")

                llm_result = None
                lunette_result = None

                # Direct LLM
                if not args.lunette_only:
                    print("  Running direct LLM...")
                    traj_text = format_trajectory_text(task["unified_traj"])
                    llm_result = await grade_with_direct_llm(
                        task_id, task["gold_patch"], traj_text
                    )
                    llm_result["ground_truth"] = task["ground_truth"]
                    llm_result["agent"] = task["paper_agent"]
                    llm_predictions.append(llm_result)

                # Lunette
                if not args.llm_only:
                    print("  Running Lunette...")
                    lunette_result = await grade_with_lunette(
                        lunette_client, http_client,
                        task_id, task["unified_traj"], task["swebench_task"],
                        timeout=args.timeout,
                    )
                    lunette_result["ground_truth"] = task["ground_truth"]
                    lunette_result["agent"] = task["paper_agent"]
                    lunette_predictions.append(lunette_result)

                # Print comparison
                print_comparison(task_id, task["ground_truth"], llm_result, lunette_result)

                # Save incrementally
                with open(OUTPUT_DIR / "predictions_gold_patch_latest.json", "w") as f:
                    json.dump({"llm": llm_predictions, "lunette": lunette_predictions}, f, indent=2)

    # Final metrics
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    if llm_predictions:
        print("\n--- DIRECT LLM ---")
        metrics = compute_metrics(llm_predictions)
        if "error" not in metrics:
            print(f"  Samples: {metrics['n_samples']}")
            print(f"  Jaccard: {metrics['jaccard_mean']:.3f} ± {metrics['jaccard_std']:.3f}")
            print(f"  Any Overlap: {metrics['any_overlap_rate']:.3f}")
            print(f"  Micro F1: {metrics['micro_f1']:.3f}")

    if lunette_predictions:
        print("\n--- LUNETTE ---")
        metrics = compute_metrics(lunette_predictions)
        if "error" not in metrics:
            print(f"  Samples: {metrics['n_samples']}")
            print(f"  Jaccard: {metrics['jaccard_mean']:.3f} ± {metrics['jaccard_std']:.3f}")
            print(f"  Any Overlap: {metrics['any_overlap_rate']:.3f}")
            print(f"  Micro F1: {metrics['micro_f1']:.3f}")

    # Save final
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(OUTPUT_DIR / f"results_gold_patch_{timestamp}.json", "w") as f:
        json.dump({
            "llm_predictions": llm_predictions,
            "lunette_predictions": lunette_predictions,
            "llm_metrics": compute_metrics(llm_predictions) if llm_predictions else None,
            "lunette_metrics": compute_metrics(lunette_predictions) if lunette_predictions else None,
        }, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())

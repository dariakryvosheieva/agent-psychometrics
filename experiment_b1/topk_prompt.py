"""
Top-K classification: Predict the TOP 3 most important failure modes.
Check if ANY of the predictions are in the ground truth set.
"""

import asyncio
import json
import re
from pathlib import Path

import anthropic
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UNIFIED_TRAJS_DIR = PROJECT_ROOT / "trajectory_data" / "unified_trajs"
OUTPUT_DIR = PROJECT_ROOT / "chris_output" / "experiment_b1"

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

TOPK_PROMPT = """You are analyzing a FAILED SWE-bench agent trajectory.

## GOLD PATCH (The Correct Solution)
```diff
{gold_patch}
```

## Your Task
Identify the TOP 3 MOST IMPORTANT failure modes - the primary root causes of why this agent failed.
Rank them from most important (#1) to least important (#3).

## Failure Mode Categories

### A. LOCALIZATION FAILURES (agent looked in wrong place)
- **issue_misleading**: Problem description ITSELF was confusing/misleading
- **keywords_location**: Agent matched keywords superficially
- **referred_code_location**: Agent fixated on irrelevant mentioned code
- **error_stack_trace**: Agent focused on stack trace not root cause

### B. REPAIR FAILURES

#### Strategy
- **specific_case_overfitting**: Fix only works for example case
- **evasive_repair**: Suppresses error without fixing cause
- **redundant_erroneous_implementation**: Added conflicting/duplicate logic

#### Implementation
- **algorithmic_implementation**: Wrong algorithm/math/data structure
- **control_flow**: Wrong if/else, loops, returns, exception handling
- **boundary_handling**: Edge cases not handled
- **data_processing_errors**: Wrong types/transformations
- **insufficient_domain_knowledge**: Lacks framework expertise

#### Incomplete
- **inheritance**: Didn't update subclasses
- **interface_contract**: Violated API contracts
- **component_coordination**: Multiple components need changes, only some made
- **recurring_pattern**: Same bug in multiple places, only fixed some
- **issue_interference**: Fix for X BROKE Y (previously working)

### C. VERIFICATION FAILURES
- **reproduction_script_failure**: Couldn't create working test
- **reproduction_output_misreading**: Ran tests but MISREAD the output
- **insufficient_verification_capability**: Cannot verify if fix works
- **non_progressive_iteration**: Loops without progress
- **blindly_strategy_switching**: Random approach changes
- **verification_abandonment**: Stopped testing early
- **verification_weakening**: MODIFIED tests to make them pass
- **context_forgetting**: Lost track of findings

## Instructions
1. Analyze the trajectory and compare to the gold patch
2. Identify the TOP 3 root causes - what were the MAIN reasons for failure?
3. Rank them from most to least important

Output format (exactly 3 categories, one per line):
#1: <category_name>
#2: <category_name>
#3: <category_name>
"""


def load_swebench_data() -> dict[str, dict]:
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    return {ex["instance_id"]: ex for ex in ds}


def load_unified_trajectory(agent_dir: str, task_id: str) -> dict | None:
    traj_file = UNIFIED_TRAJS_DIR / agent_dir / f"{task_id}.json"
    if not traj_file.exists():
        return None
    with open(traj_file) as f:
        return json.load(f)


def format_trajectory(traj: dict, max_messages: int = 30) -> str:
    messages = traj.get("messages", [])[:max_messages]
    formatted = []
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        if isinstance(content, list):
            parts = [item.get("text", str(item)) if isinstance(item, dict) else str(item) for item in content]
            content = "\n".join(parts)
        if len(content) > 4000:
            content = content[:4000] + "..."
        formatted.append(f"[{role} {i+1}]:\n{content}")
    return "\n\n---\n\n".join(formatted)


def parse_topk_predictions(text: str) -> list[str]:
    """Extract top-3 categories from numbered list."""
    if not text:
        return []

    predictions = []
    text_lower = text.lower()

    # Look for #1:, #2:, #3: patterns
    for pattern in [r"#1[:\s]+(\w+)", r"#2[:\s]+(\w+)", r"#3[:\s]+(\w+)"]:
        match = re.search(pattern, text_lower)
        if match:
            candidate = match.group(1).strip()
            # Match against known categories
            for cat in FAILURE_CATEGORIES:
                if cat == candidate or cat.startswith(candidate) or candidate in cat:
                    predictions.append(cat)
                    break

    # Fallback: look for category names after numbers
    if len(predictions) < 3:
        for line in text.split("\n"):
            line_lower = line.lower().strip()
            if any(line_lower.startswith(f"#{i}") for i in [1, 2, 3]):
                for cat in FAILURE_CATEGORIES:
                    if cat in line_lower or cat.replace("_", " ") in line_lower:
                        if cat not in predictions:
                            predictions.append(cat)
                            break

    return predictions[:3]  # Return at most 3


async def grade_topk(task_id: str, gold_patch: str, traj_text: str) -> dict:
    client = anthropic.Anthropic()
    prompt = TOPK_PROMPT.format(gold_patch=gold_patch)

    full_prompt = f"""{prompt}

## AGENT TRAJECTORY
{traj_text}

What are the TOP 3 failure modes? Rank them #1, #2, #3."""

    try:
        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": full_prompt}],
        )
        text = resp.content[0].text
        predictions = parse_topk_predictions(text)
        return {
            "task_id": task_id,
            "predictions": predictions,
            "explanation": text,
        }
    except Exception as e:
        return {"task_id": task_id, "error": str(e)}


TEST_TASKS = [
    {"task_id": "astropy__astropy-13236", "agent": "20241029_OpenHands-CodeAct-2.1-sonnet-20241022",
     "ground_truth": ["issue_interference"]},
    {"task_id": "django__django-10999", "agent": "20241029_OpenHands-CodeAct-2.1-sonnet-20241022",
     "ground_truth": ["issue_interference"]},
    {"task_id": "astropy__astropy-13398", "agent": "20241029_OpenHands-CodeAct-2.1-sonnet-20241022",
     "ground_truth": ["issue_misleading"]},
    {"task_id": "astropy__astropy-13453", "agent": "20241029_OpenHands-CodeAct-2.1-sonnet-20241022",
     "ground_truth": ["redundant_erroneous_implementation", "reproduction_output_misreading"]},
    {"task_id": "astropy__astropy-8872", "agent": "20241029_OpenHands-CodeAct-2.1-sonnet-20241022",
     "ground_truth": ["reproduction_output_misreading", "recurring_pattern", "control_flow"]},
    {"task_id": "django__django-11087", "agent": "20241029_OpenHands-CodeAct-2.1-sonnet-20241022",
     "ground_truth": ["algorithmic_implementation", "verification_weakening", "insufficient_domain_knowledge", "control_flow"]},
]


async def main():
    print("Loading data...")
    swebench = load_swebench_data()

    print("\nTesting TOP-K (K=3) classification...\n")

    results = []
    any_hit = 0
    total = 0

    for task in TEST_TASKS:
        task_id = task["task_id"]
        agent = task["agent"]
        ground_truth = set(task["ground_truth"])

        print(f"\n{'='*60}")
        print(f"Task: {task_id}")
        print(f"Ground truth: {sorted(ground_truth)}")
        print("="*60)

        traj = load_unified_trajectory(agent, task_id)
        if not traj or task_id not in swebench:
            print("  Skip - missing data")
            continue

        result = await grade_topk(task_id, swebench[task_id]["patch"], format_trajectory(traj))
        preds = set(result.get("predictions", []))

        print(f"Predictions: {sorted(preds)}")

        # Metrics
        intersection = preds & ground_truth
        hit_any = len(intersection) > 0
        hit_count = len(intersection)

        print(f"Overlap: {sorted(intersection)} ({hit_count}/{len(ground_truth)} ground truth matched)")
        status = "HIT" if hit_any else "MISS"
        print(f"Result: {status}")

        total += 1
        if hit_any:
            any_hit += 1

        results.append({
            "task_id": task_id,
            "ground_truth": sorted(ground_truth),
            "predictions": sorted(preds),
            "overlap": sorted(intersection),
            "hit_any": hit_any,
        })

    # Summary
    print("\n" + "="*60)
    print("SUMMARY - Top-3 Classification")
    print("="*60)

    accuracy_any = any_hit / total if total > 0 else 0
    print(f"\nAny prediction in ground truth: {any_hit}/{total} = {accuracy_any:.1%}")

    # Compare to approaches
    print("\n--- Comparison with other approaches ---")
    print(f"Single-label (Top-1): 0/6 = 0.0%")
    print(f"Top-3: {any_hit}/{total} = {accuracy_any:.1%}")
    print(f"Hierarchical multi-label: Jaccard 0.276 (best so far)")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "topk_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR / 'topk_results.json'}")


if __name__ == "__main__":
    asyncio.run(main())

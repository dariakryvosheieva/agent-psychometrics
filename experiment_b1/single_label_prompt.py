"""
Single-label classification: Predict the ONE most important failure mode.

Simpler task than multi-label. Check if prediction is in ground truth set.
"""

import asyncio
import json
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

SINGLE_LABEL_PROMPT = """You are analyzing a FAILED SWE-bench agent trajectory.

## GOLD PATCH (The Correct Solution)
```diff
{gold_patch}
```

## Your Task
Identify the SINGLE MOST IMPORTANT failure mode - the primary root cause of why this agent failed.

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
2. Identify the PRIMARY root cause - what was the MAIN reason for failure?
3. Choose exactly ONE category from the list above

Your answer must be a single category name from the list. Output format:
PRIMARY FAILURE MODE: <category_name>
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


def parse_single_prediction(text: str) -> str | None:
    """Extract single category from PRIMARY FAILURE MODE line."""
    if not text:
        return None

    text_lower = text.lower()

    # Look for PRIMARY FAILURE MODE line
    for line in text.split("\n"):
        if "primary failure mode" in line.lower():
            # Extract what comes after the colon
            if ":" in line:
                after = line.split(":", 1)[-1].strip().lower()
                after = after.replace(" ", "_").replace("-", "_")
                # Match against known categories
                for cat in FAILURE_CATEGORIES:
                    if cat in after or after in cat:
                        return cat

    # Fallback: find any category mentioned prominently
    for cat in FAILURE_CATEGORIES:
        if cat in text_lower or cat.replace("_", " ") in text_lower:
            return cat

    return None


async def grade_single_label(task_id: str, gold_patch: str, traj_text: str) -> dict:
    client = anthropic.Anthropic()
    prompt = SINGLE_LABEL_PROMPT.format(gold_patch=gold_patch)

    full_prompt = f"""{prompt}

## AGENT TRAJECTORY
{traj_text}

What is the PRIMARY FAILURE MODE? Choose exactly one category."""

    try:
        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": full_prompt}],
        )
        text = resp.content[0].text
        prediction = parse_single_prediction(text)
        return {
            "task_id": task_id,
            "predicted": prediction,
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

    print("\nTesting SINGLE-LABEL classification...\n")

    results = []
    hits = 0
    total = 0

    for task in TEST_TASKS:
        task_id = task["task_id"]
        agent = task["agent"]
        ground_truth = task["ground_truth"]

        print(f"\n{'='*60}")
        print(f"Task: {task_id}")
        print(f"Ground truth: {ground_truth}")
        print("="*60)

        traj = load_unified_trajectory(agent, task_id)
        if not traj or task_id not in swebench:
            print("  Skip - missing data")
            continue

        result = await grade_single_label(task_id, swebench[task_id]["patch"], format_trajectory(traj))
        pred = result.get("predicted")

        print(f"Predicted: {pred}")

        # Show brief explanation
        explanation = result.get("explanation", "")
        if explanation:
            # Extract just the reasoning, skip the long category list
            lines = explanation.split("\n")
            reasoning_lines = [l for l in lines if l.strip() and not l.startswith("###") and not l.startswith("**")]
            print(f"Reasoning excerpt: {' '.join(reasoning_lines[:3])[:200]}...")

        hit = pred in ground_truth if pred else False
        status = "HIT" if hit else "MISS"
        print(f"Result: {status}")

        if pred:
            total += 1
            if hit:
                hits += 1

        results.append({
            "task_id": task_id,
            "ground_truth": ground_truth,
            "predicted": pred,
            "hit": hit,
        })

    # Summary
    print("\n" + "="*60)
    print("SUMMARY - Single Label Classification")
    print("="*60)

    accuracy = hits / total if total > 0 else 0
    print(f"\nAccuracy (prediction in ground truth): {hits}/{total} = {accuracy:.1%}")

    # Per-category analysis
    print("\nPer-category hits:")
    for cat in ["issue_interference", "reproduction_output_misreading", "control_flow",
                "verification_weakening", "issue_misleading", "algorithmic_implementation"]:
        tasks_with_cat = [r for r in results if cat in r["ground_truth"]]
        hits_for_cat = sum(1 for r in tasks_with_cat if r["predicted"] == cat)
        if tasks_with_cat:
            print(f"  {cat}: {hits_for_cat}/{len(tasks_with_cat)}")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "single_label_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR / 'single_label_results.json'}")


if __name__ == "__main__":
    asyncio.run(main())

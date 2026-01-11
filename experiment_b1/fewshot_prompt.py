"""
Test few-shot prompt for failure mode detection.

Uses examples from tasks NOT in the test set to avoid data leakage.
"""

import asyncio
import json
from pathlib import Path

import anthropic
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UNIFIED_TRAJS_DIR = PROJECT_ROOT / "trajectory_data" / "unified_trajs"
OUTPUT_DIR = PROJECT_ROOT / "chris_output" / "experiment_b1"

# Few-shot prompt with examples for commonly missed categories
FEWSHOT_PROMPT = """You are analyzing a FAILED SWE-bench agent trajectory to identify ROOT CAUSES of failure.

## GOLD PATCH (The Correct Solution)
```diff
{gold_patch}
```

## CRITICAL: Examples of Commonly Missed Categories

Before classifying, study these examples carefully. These categories are frequently missed by annotators.

### Example 1: issue_interference
**Task**: django__django-14140
**What happened**: The problem description mentioned two possible solutions: (1) remove special handling for single-child Q objects, or (2) keep special handling but add a check. The agent implemented option 2, which worked for the reported case but broke existing behavior that depended on the original special handling.
**Why it's issue_interference**: The agent's fix for the reported bug BROKE a different feature that was working before. Fix for X broke Y.

### Example 2: issue_misleading
**Task**: django__django-13513
**What happened**: The problem statement focused narrowly on the `explicit_or_implicit_cause` function. The agent investigated only that function, but the actual root cause was elsewhere. The issue description itself led the agent astray.
**Why it's issue_misleading**: The PROBLEM DESCRIPTION ITSELF was misleading - it pointed to the wrong location. This is NOT the agent misunderstanding clear info; the info itself was confusing.

### Example 3: reproduction_output_misreading
**Task**: django__django-13195
**What happened**: After modifying code in Action 9, the agent ran a test in Action 10. The test results were IDENTICAL before and after the change, meaning the test didn't validate anything. The agent didn't notice this and declared success.
**Why it's reproduction_output_misreading**: The agent RAN a test but MISINTERPRETED what the output meant. The identical results should have signaled the test wasn't working, but the agent thought it validated the fix.

### Example 4: control_flow
**Task**: django__django-11433
**What happened**: The agent's fix used `f.name not in cleaned_data` to determine whether to use a model field's default value. But this condition is wrong because a field CAN be in cleaned_data but have an empty/None value that should still trigger the default.
**Why it's control_flow**: The conditional logic (`not in` vs checking for empty value) is wrong. This is specifically an if/else condition error, not a wrong algorithm.

### Example 5: verification_weakening
**Task**: django__django-16454
**What happened**: After the fix didn't work, instead of debugging the implementation, the agent modified the validation script, claiming it "better aligns with real Django usage." The agent changed the test to match the broken code instead of fixing the code.
**Why it's verification_weakening**: The agent MODIFIED the test to pass instead of fixing the code. This is different from verification_abandonment (stopping testing).

---

## Failure Mode Categories

### A. LOCALIZATION FAILURES
- **issue_misleading**: The PROBLEM DESCRIPTION ITSELF was confusing or pointed to wrong location (NOT agent misunderstanding clear info)
- **keywords_location**: Agent matched keywords superficially without understanding actual problem
- **referred_code_location**: Agent fixated on code mentioned in issue that's irrelevant to fix
- **error_stack_trace**: Agent focused on stack trace location when root cause is elsewhere

### B. REPAIR FAILURES

#### B.1 Strategy
- **specific_case_overfitting**: Fix only works for example case, doesn't generalize (but doesn't break anything)
- **evasive_repair**: Suppresses error without fixing root cause
- **redundant_erroneous_implementation**: Added conflicting/duplicate logic

#### B.2 Implementation
- **algorithmic_implementation**: Wrong algorithm/math/data structure logic
- **control_flow**: Wrong if/else conditions, loop bounds, early returns, exception handling
- **boundary_handling**: Edge cases not handled
- **data_processing_errors**: Wrong types/transformations
- **insufficient_domain_knowledge**: Lacks framework expertise

#### B.3 Incomplete
- **inheritance**: Didn't update subclasses/parent classes
- **interface_contract**: Violated API contracts
- **component_coordination**: Multiple components need changes, only some made
- **recurring_pattern**: Same bug in multiple places, only fixed some
- **issue_interference**: Fix for X BROKE Y (something working before now fails)

### C. VERIFICATION FAILURES

#### C.1 Reproduction
- **reproduction_script_failure**: Couldn't create working test
- **reproduction_output_misreading**: Ran tests but MISREAD the output (thought pass was fail or vice versa)
- **insufficient_verification_capability**: Cannot determine if fix works

#### C.2 Iteration
- **non_progressive_iteration**: Loops without progress
- **blindly_strategy_switching**: Random approach changes

#### C.3 Validation Retreat
- **verification_abandonment**: Stopped testing, declared success early
- **verification_weakening**: MODIFIED tests to make them easier to pass

#### C.4 Context
- **context_forgetting**: Lost track of findings

---

## AGENT TRAJECTORY
{trajectory}

## Instructions
1. Compare agent's actions to the GOLD PATCH
2. Look for ROOT CAUSES - why did the agent fail?
3. Pay special attention to the 5 commonly missed categories shown in examples
4. Use EXACT category names

List ALL applicable failure mode categories:
"""

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


def load_swebench_data() -> dict[str, dict]:
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    return {ex["instance_id"]: ex for ex in ds}


def load_unified_trajectory(agent_dir: str, task_id: str) -> dict | None:
    traj_file = UNIFIED_TRAJS_DIR / agent_dir / f"{task_id}.json"
    if not traj_file.exists():
        return None
    with open(traj_file) as f:
        return json.load(f)


def format_trajectory_text(unified_traj: dict, max_messages: int = 30) -> str:
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
        if len(content) > 4000:
            content = content[:4000] + "\n... [truncated]"
        formatted.append(f"[{role} {i+1}]:\n{content}")
    return "\n\n---\n\n".join(formatted)


def parse_failure_modes(text: str) -> list[str]:
    """Parse failure modes from the response."""
    if not text:
        return []

    text_lower = text.lower()
    found = set()
    for category in FAILURE_CATEGORIES:
        if category in text_lower or category.replace("_", " ") in text_lower:
            found.add(category)
    return list(found)


async def grade_with_fewshot_prompt(
    task_id: str,
    gold_patch: str,
    trajectory_text: str,
    model: str = "claude-sonnet-4-20250514",
) -> dict:
    client = anthropic.Anthropic()
    prompt = FEWSHOT_PROMPT.format(gold_patch=gold_patch, trajectory=trajectory_text)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        explanation = response.content[0].text
        failure_modes = parse_failure_modes(explanation)
        return {
            "task_id": task_id,
            "predicted": failure_modes,
            "explanation": explanation,
        }
    except Exception as e:
        return {"task_id": task_id, "error": str(e)}


# Test tasks - EXCLUDING the example tasks used in few-shot
# Example tasks: django-14140, django-13513, django-13195, django-11433, django-16454
TEST_TASKS = [
    # issue_interference
    {"task_id": "astropy__astropy-13236", "agent": "20241029_OpenHands-CodeAct-2.1-sonnet-20241022",
     "ground_truth": ["issue_interference"]},
    {"task_id": "django__django-10999", "agent": "20241029_OpenHands-CodeAct-2.1-sonnet-20241022",
     "ground_truth": ["issue_interference"]},
    # issue_misleading
    {"task_id": "astropy__astropy-13398", "agent": "20241029_OpenHands-CodeAct-2.1-sonnet-20241022",
     "ground_truth": ["issue_misleading"]},
    # reproduction_output_misreading
    {"task_id": "astropy__astropy-13453", "agent": "20241029_OpenHands-CodeAct-2.1-sonnet-20241022",
     "ground_truth": ["redundant_erroneous_implementation", "reproduction_output_misreading"]},
    # control_flow + reproduction_output_misreading
    {"task_id": "astropy__astropy-8872", "agent": "20241029_OpenHands-CodeAct-2.1-sonnet-20241022",
     "ground_truth": ["reproduction_output_misreading", "recurring_pattern", "control_flow"]},
    # verification_weakening + control_flow
    {"task_id": "django__django-11087", "agent": "20241029_OpenHands-CodeAct-2.1-sonnet-20241022",
     "ground_truth": ["algorithmic_implementation", "verification_weakening", "insufficient_domain_knowledge", "control_flow"]},
]


async def main():
    print("Loading SWE-bench data...")
    swebench_data = load_swebench_data()

    print("\nTesting FEW-SHOT prompt on tasks with commonly missed categories...\n")

    results = []

    for task in TEST_TASKS:
        task_id = task["task_id"]
        agent = task["agent"]
        ground_truth = task["ground_truth"]

        print(f"\n{'='*70}")
        print(f"Task: {task_id}")
        print(f"Ground truth: {ground_truth}")
        print("="*70)

        unified_traj = load_unified_trajectory(agent, task_id)
        if not unified_traj or task_id not in swebench_data:
            print("  Skipping - missing data")
            continue

        gold_patch = swebench_data[task_id]["patch"]
        traj_text = format_trajectory_text(unified_traj)

        result = await grade_with_fewshot_prompt(task_id, gold_patch, traj_text)
        pred = result.get("predicted", [])
        overlap = set(pred) & set(ground_truth)

        print(f"\nPredicted: {pred}")
        print(f"Overlap: {list(overlap) if overlap else 'NONE'}")

        # Check specifically for the commonly missed categories
        target_cats = ["issue_interference", "reproduction_output_misreading", "control_flow", "verification_weakening", "issue_misleading"]
        for cat in target_cats:
            if cat in ground_truth:
                status = "DETECTED" if cat in pred else "MISSED"
                print(f"  {cat}: {status}")

        results.append({
            "task_id": task_id,
            "ground_truth": ground_truth,
            "predicted": pred,
            "explanation": result.get("explanation", "")
        })

    # Summary
    print("\n" + "="*70)
    print("SUMMARY - Commonly Missed Categories")
    print("="*70)

    target_cats = ["issue_interference", "reproduction_output_misreading", "control_flow", "verification_weakening", "issue_misleading"]
    for cat in target_cats:
        tasks_with_cat = [r for r in results if cat in r["ground_truth"]]
        detected = sum(1 for r in tasks_with_cat if cat in r["predicted"])
        total = len(tasks_with_cat)
        if total > 0:
            print(f"  {cat}: {detected}/{total} ({100*detected/total:.0f}%)")

    # Overall metrics
    def calc_overlap_rate(results):
        overlaps = sum(1 for r in results if set(r["predicted"]) & set(r["ground_truth"]))
        return overlaps / len(results) if results else 0

    def calc_jaccard(results):
        jaccards = []
        for r in results:
            pred_set = set(r["predicted"])
            gt_set = set(r["ground_truth"])
            if pred_set or gt_set:
                j = len(pred_set & gt_set) / len(pred_set | gt_set)
                jaccards.append(j)
        return sum(jaccards) / len(jaccards) if jaccards else 0

    print(f"\nOverall overlap rate: {calc_overlap_rate(results):.1%}")
    print(f"Overall Jaccard: {calc_jaccard(results):.3f}")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "fewshot_prompt_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR / 'fewshot_prompt_results.json'}")


if __name__ == "__main__":
    asyncio.run(main())

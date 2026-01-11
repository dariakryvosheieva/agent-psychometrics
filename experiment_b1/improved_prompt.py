"""
Test improved prompts for failure mode detection.

Focus on the commonly missed categories:
- issue_misleading, issue_interference
- reproduction_output_misreading
- control_flow
- verification_weakening
"""

import asyncio
import json
from pathlib import Path
from collections import defaultdict

import anthropic
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UNIFIED_TRAJS_DIR = PROJECT_ROOT / "trajectory_data" / "unified_trajs"
OUTPUT_DIR = PROJECT_ROOT / "chris_output" / "experiment_b1"

# The original prompt (baseline)
ORIGINAL_PROMPT = """You are analyzing a FAILED SWE-bench agent trajectory to identify ROOT CAUSES of failure.

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

List the failure mode categories that apply to this trajectory.
"""

# Improved prompt with clearer distinctions for commonly missed categories
IMPROVED_PROMPT = """You are analyzing a FAILED SWE-bench agent trajectory to identify ROOT CAUSES of failure.

## GOLD PATCH (The Correct Solution)
```diff
{gold_patch}
```

## CRITICAL: Common Labeling Mistakes to Avoid

Before labeling, consider these commonly confused categories:

1. **issue_misleading** vs other localization failures:
   - Use `issue_misleading` when the PROBLEM DESCRIPTION ITSELF contains confusing or misleading information
   - Example: PR says "add warning in v5.1, change behavior in v5.2" but gold patch shows only the final change
   - Example: Issue describes a workaround that looks like the solution but isn't
   - NOT the same as: agent misunderstanding correct information

2. **issue_interference** vs specific_case_overfitting:
   - Use `issue_interference` when the agent's fix BREAKS something that was working before
   - Example: Fix for feature A causes feature B to fail
   - Use `specific_case_overfitting` when fix is too narrow but doesn't break other things

3. **reproduction_output_misreading** vs verification_abandonment:
   - Use `reproduction_output_misreading` when agent RUNS tests but MISINTERPRETS the results
   - Example: Test says "FAILED" but agent thinks it passed
   - Use `verification_abandonment` when agent STOPS testing entirely

4. **control_flow** vs algorithmic_implementation:
   - Use `control_flow` for wrong if/else, loop bounds, early returns, exception handling
   - Use `algorithmic_implementation` for wrong algorithm logic (math, data structures)

5. **verification_weakening** vs verification_abandonment:
   - Use `verification_weakening` when agent MODIFIES tests to make them pass
   - Use `verification_abandonment` when agent STOPS running tests

## Failure Mode Categories

### A. LOCALIZATION FAILURES
- **issue_misleading**: The problem description/PR itself was confusing or misleading (NOT agent misunderstanding clear info)
- **keywords_location**: Agent matched keywords superficially
- **referred_code_location**: Agent fixated on mentioned but irrelevant code
- **error_stack_trace**: Agent focused on stack trace not root cause

### B. REPAIR FAILURES

#### B.1 Strategy
- **specific_case_overfitting**: Fix too narrow, doesn't generalize (but doesn't break anything)
- **evasive_repair**: Suppresses error without fixing root cause
- **redundant_erroneous_implementation**: Added conflicting/duplicate logic

#### B.2 Implementation
- **algorithmic_implementation**: Wrong algorithm/math/data structure logic
- **control_flow**: Wrong conditionals, loops, early returns, exception handling
- **boundary_handling**: Edge cases not handled
- **data_processing_errors**: Wrong types/transformations
- **insufficient_domain_knowledge**: Lacks framework expertise

#### B.3 Incomplete
- **inheritance**: Didn't update subclasses
- **interface_contract**: Violated API contracts
- **component_coordination**: Multiple components need changes, only some made
- **recurring_pattern**: Same bug in multiple places, only fixed some
- **issue_interference**: Fix for X broke Y (something that worked before now fails)

### C. VERIFICATION FAILURES

#### C.1 Reproduction
- **reproduction_script_failure**: Couldn't create working test
- **reproduction_output_misreading**: Ran tests but MISREAD the output
- **insufficient_verification_capability**: Cannot determine if fix works

#### C.2 Iteration
- **non_progressive_iteration**: Loops without progress
- **blindly_strategy_switching**: Random approach changes

#### C.3 Validation Retreat
- **verification_abandonment**: Stopped testing, declared success early
- **verification_weakening**: Modified tests to pass instead of fixing code

#### C.4 Context
- **context_forgetting**: Lost track of findings

## Instructions
1. Compare agent's actions to the GOLD PATCH
2. Look for ROOT CAUSES - why did the agent fail?
3. Use the EXACT category names above
4. Pay special attention to the commonly confused categories

List ALL applicable failure mode categories.
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


def parse_failure_modes(text: str) -> list[str]:
    if not text:
        return []
    text_lower = text.lower()
    found = set()
    for category in FAILURE_CATEGORIES:
        if category in text_lower or category.replace("_", " ") in text_lower:
            found.add(category)
    return list(found)


async def grade_with_prompt(
    task_id: str,
    gold_patch: str,
    trajectory_text: str,
    prompt_template: str,
    model: str = "claude-sonnet-4-20250514",
) -> dict:
    client = anthropic.Anthropic()
    prompt = prompt_template.format(gold_patch=gold_patch)

    full_prompt = f"""{prompt}

## AGENT TRAJECTORY
{trajectory_text}

Identify the failure modes. List all applicable category names."""

    try:
        response = client.messages.create(
            model=model,
            max_tokens=2000,
            messages=[{"role": "user", "content": full_prompt}],
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


# Test tasks with commonly missed categories
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
    # control_flow
    {"task_id": "astropy__astropy-8872", "agent": "20241029_OpenHands-CodeAct-2.1-sonnet-20241022",
     "ground_truth": ["reproduction_output_misreading", "recurring_pattern", "control_flow"]},
]


async def main():
    print("Loading SWE-bench data...")
    swebench_data = load_swebench_data()

    print("\nTesting prompts on tasks with commonly missed categories...\n")

    original_results = []
    improved_results = []

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

        # Test original prompt
        print("\n--- ORIGINAL PROMPT ---")
        orig_result = await grade_with_prompt(task_id, gold_patch, traj_text, ORIGINAL_PROMPT)
        orig_pred = orig_result.get("predicted", [])
        orig_overlap = set(orig_pred) & set(ground_truth)
        print(f"Predicted: {orig_pred}")
        print(f"Overlap: {list(orig_overlap) if orig_overlap else 'NONE'}")
        original_results.append({"task_id": task_id, "ground_truth": ground_truth, "predicted": orig_pred})

        # Test improved prompt
        print("\n--- IMPROVED PROMPT ---")
        impr_result = await grade_with_prompt(task_id, gold_patch, traj_text, IMPROVED_PROMPT)
        impr_pred = impr_result.get("predicted", [])
        impr_overlap = set(impr_pred) & set(ground_truth)
        print(f"Predicted: {impr_pred}")
        print(f"Overlap: {list(impr_overlap) if impr_overlap else 'NONE'}")
        improved_results.append({"task_id": task_id, "ground_truth": ground_truth, "predicted": impr_pred})

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    def calc_overlap_rate(results):
        overlaps = sum(1 for r in results if set(r["predicted"]) & set(r["ground_truth"]))
        return overlaps / len(results) if results else 0

    print(f"\nOriginal prompt overlap rate: {calc_overlap_rate(original_results):.1%}")
    print(f"Improved prompt overlap rate: {calc_overlap_rate(improved_results):.1%}")

    # Per-category analysis
    print("\nPer-category detection (improved prompt):")
    target_cats = ["issue_interference", "issue_misleading", "reproduction_output_misreading", "control_flow"]
    for cat in target_cats:
        tasks_with_cat = [r for r in improved_results if cat in r["ground_truth"]]
        detected = sum(1 for r in tasks_with_cat if cat in r["predicted"])
        print(f"  {cat}: {detected}/{len(tasks_with_cat)}")


if __name__ == "__main__":
    asyncio.run(main())

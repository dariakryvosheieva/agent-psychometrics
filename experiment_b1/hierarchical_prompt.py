"""
Test hierarchical classification prompt for failure mode detection.

Uses explicit decision trees for commonly missed categories.
"""

import asyncio
import json
from pathlib import Path

import anthropic
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UNIFIED_TRAJS_DIR = PROJECT_ROOT / "trajectory_data" / "unified_trajs"
OUTPUT_DIR = PROJECT_ROOT / "chris_output" / "experiment_b1"

# Hierarchical prompt with explicit decision trees
HIERARCHICAL_PROMPT = """You are analyzing a FAILED SWE-bench agent trajectory to identify ROOT CAUSES of failure.

## GOLD PATCH (The Correct Solution)
```diff
{gold_patch}
```

## CLASSIFICATION PROCESS

Answer each question below IN ORDER. Your final answer will be the set of categories where you answered YES.

---

### PHASE 1: LOCALIZATION FAILURES

**Q1: issue_misleading**
Was the PROBLEM DESCRIPTION ITSELF confusing or misleading (NOT the agent misunderstanding clear info)?
- Example: Issue says "do X then Y" but gold patch only does Y
- Example: Issue contains a workaround that looks like the solution but isn't
Answer YES or NO, then explain briefly.

**Q2: keywords_location**
Did the agent search/navigate based on superficial keyword matching without understanding the actual problem?
Answer YES or NO, then explain briefly.

**Q3: referred_code_location**
Did the agent fixate on code explicitly mentioned in the issue that's irrelevant to the actual fix?
Answer YES or NO, then explain briefly.

**Q4: error_stack_trace**
Did the agent focus on the stack trace location when the root cause is elsewhere?
Answer YES or NO, then explain briefly.

---

### PHASE 2: REPAIR FAILURES

#### 2A: Strategy Defects

**Q5: specific_case_overfitting**
Does the agent's fix ONLY work for the example case but fail to generalize? (Fix is too narrow but doesn't break anything)
Answer YES or NO, then explain briefly.

**Q6: evasive_repair**
Does the fix suppress/hide the error without addressing the root cause?
Answer YES or NO, then explain briefly.

**Q7: redundant_erroneous_implementation**
Did the agent add conflicting or duplicate logic that shouldn't exist?
Answer YES or NO, then explain briefly.

#### 2B: Implementation Defects

**Q8: algorithmic_implementation**
Is the core algorithm/math/data structure logic wrong? (NOT control flow issues)
Answer YES or NO, then explain briefly.

**Q9: control_flow** [COMMONLY MISSED]
Are there errors specifically in:
- if/else conditions (wrong comparisons, missing cases)
- loop bounds (off-by-one, wrong termination)
- early returns (returning at wrong point)
- exception handling (wrong try/except, missing catches)
Answer YES or NO, then explain briefly.

**Q10: boundary_handling**
Are edge cases not handled properly?
Answer YES or NO, then explain briefly.

**Q11: data_processing_errors**
Are there wrong data types or transformations?
Answer YES or NO, then explain briefly.

**Q12: insufficient_domain_knowledge**
Does the agent lack framework/library expertise needed for the fix?
Answer YES or NO, then explain briefly.

#### 2C: Incomplete Repair

**Q13: inheritance**
Did the agent fail to update subclasses or parent classes that also need changes?
Answer YES or NO, then explain briefly.

**Q14: interface_contract**
Did the agent violate API contracts?
Answer YES or NO, then explain briefly.

**Q15: component_coordination**
Do multiple components need changes but the agent only modified some?
Answer YES or NO, then explain briefly.

**Q16: recurring_pattern**
Is the same bug in multiple places but the agent only fixed some?
Answer YES or NO, then explain briefly.

**Q17: issue_interference** [COMMONLY MISSED]
CRITICAL: Does the agent's fix cause ANY functionality that was WORKING BEFORE to now FAIL?
- This is different from "fix is wrong" - this specifically means "fix for X broke Y"
- Example: Fixed login page but now logout is broken
- Example: Fixed edge case handling but normal cases now fail
Answer YES or NO, then explain briefly.

---

### PHASE 3: VERIFICATION FAILURES

#### 3A: Reproduction/Verification

**Q18: reproduction_script_failure**
Did the agent fail to create a working reproduction script?
Answer YES or NO, then explain briefly.

**Q19: reproduction_output_misreading** [COMMONLY MISSED]
CRITICAL: Did the agent RUN tests/scripts and MISINTERPRET what the output means?
- NOT about skipping tests (that's verification_abandonment)
- Specifically: agent sees output, draws WRONG conclusion
- Example: Test says "1 failed" but agent thinks it passed
- Example: Error message is misunderstood as success
Answer YES or NO, then explain briefly.

**Q20: insufficient_verification_capability**
Is the agent unable to properly verify if the fix works?
Answer YES or NO, then explain briefly.

#### 3B: Iteration Anomalies

**Q21: non_progressive_iteration**
Does the agent loop without making meaningful progress?
Answer YES or NO, then explain briefly.

**Q22: blindly_strategy_switching**
Does the agent randomly try approaches without learning from failures?
Answer YES or NO, then explain briefly.

#### 3C: Validation Retreat

**Q23: verification_abandonment**
Did the agent STOP running tests and declare success prematurely?
Answer YES or NO, then explain briefly.

**Q24: verification_weakening** [COMMONLY MISSED]
CRITICAL: Did the agent MODIFY test files to make tests easier to pass?
- NOT about skipping tests (that's verification_abandonment)
- Specifically: agent changes test assertions, removes test cases, weakens validation
- Example: Changed "assert x == 5" to "assert x > 0"
- Example: Removed a failing test case instead of fixing the code
Answer YES or NO, then explain briefly.

#### 3D: Context

**Q25: context_forgetting**
Did the agent lose track of previous findings?
Answer YES or NO, then explain briefly.

---

## FINAL OUTPUT

After answering all questions, list ONLY the categories where you answered YES:

FAILURE MODES: [list category names separated by commas]
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
    """Parse failure modes from the FAILURE MODES: line or YES answers."""
    import re

    if not text:
        return []

    # Method 1: Look for the FAILURE MODES: line at the end
    lines = text.strip().split("\n")
    for line in reversed(lines):
        if "FAILURE MODES:" in line.upper():
            # Extract everything after the colon
            after_colon = line.split(":", 1)[-1].strip()
            # Remove brackets if present
            after_colon = after_colon.strip("[]")
            # Skip if empty or just whitespace
            if not after_colon or after_colon == "":
                continue
            # Split by comma and clean up
            found = set()
            for item in after_colon.split(","):
                item = item.strip().lower().replace(" ", "_").replace("-", "_")
                if not item:
                    continue
                # Match against known categories
                for cat in FAILURE_CATEGORIES:
                    if cat in item or item in cat:
                        found.add(cat)
            if found:  # Only return if we found something
                return list(found)

    # Method 2: Parse YES/NO answers for each Q#: category pattern
    # More strict matching: look for "Q#: category_name" pattern followed by YES on same or next line
    found = set()

    for cat in FAILURE_CATEGORIES:
        cat_pattern = cat.replace("_", "[_ ]")

        # Pattern: **Q17: issue_interference** [optional text]\nYES
        # The YES must be at the start of the next line (possibly with whitespace)
        pattern = rf"\*\*Q\d+[:\s]+{cat_pattern}\*\*[^\n]*\n\s*YES\b"
        if re.search(pattern, text, re.IGNORECASE):
            found.add(cat)
            continue

        # Pattern: Q17: issue_interference\nYES
        pattern2 = rf"Q\d+[:\s]+{cat_pattern}[^\n]*\n\s*YES\b"
        if re.search(pattern2, text, re.IGNORECASE):
            found.add(cat)

    return list(found)


async def grade_with_hierarchical_prompt(
    task_id: str,
    gold_patch: str,
    trajectory_text: str,
    model: str = "claude-sonnet-4-20250514",
) -> dict:
    client = anthropic.Anthropic()
    prompt = HIERARCHICAL_PROMPT.format(gold_patch=gold_patch)

    full_prompt = f"""{prompt}

## AGENT TRAJECTORY
{trajectory_text}

Now answer each question Q1-Q25 with YES or NO and a brief explanation, then provide your FAILURE MODES summary."""

    try:
        response = client.messages.create(
            model=model,
            max_tokens=4000,
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

    print("\nTesting HIERARCHICAL prompt on tasks with commonly missed categories...\n")

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

        result = await grade_with_hierarchical_prompt(task_id, gold_patch, traj_text)
        pred = result.get("predicted", [])
        overlap = set(pred) & set(ground_truth)

        print(f"\nPredicted: {pred}")
        print(f"Overlap: {list(overlap) if overlap else 'NONE'}")

        # Check specifically for the commonly missed categories
        target_cats = ["issue_interference", "reproduction_output_misreading", "control_flow", "verification_weakening"]
        for cat in target_cats:
            if cat in ground_truth:
                status = "DETECTED" if cat in pred else "MISSED"
                print(f"  {cat}: {status}")

        results.append({
            "task_id": task_id,
            "ground_truth": ground_truth,
            "predicted": pred,
            "explanation": result.get("explanation", "")  # Full explanation
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
    with open(OUTPUT_DIR / "hierarchical_prompt_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR / 'hierarchical_prompt_results.json'}")


if __name__ == "__main__":
    asyncio.run(main())

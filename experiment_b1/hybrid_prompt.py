"""
Hybrid prompt: Hierarchical YES/NO questions + few-shot examples for hard categories.
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

# Hybrid prompt: hierarchical questions with few-shot examples for hard categories
HYBRID_PROMPT = """You are analyzing a FAILED SWE-bench agent trajectory to identify ROOT CAUSES of failure.

## GOLD PATCH (The Correct Solution)
```diff
{gold_patch}
```

## CLASSIFICATION QUESTIONS

Answer each question with YES or NO, then explain briefly.

---

### PHASE 1: LOCALIZATION FAILURES

**Q1: issue_misleading**
EXAMPLE: In django-13513, the problem statement focused narrowly on `explicit_or_implicit_cause` function, but the root cause was elsewhere. The issue description ITSELF led the agent astray.
Was the PROBLEM DESCRIPTION ITSELF misleading (pointing to wrong location, suggesting wrong approach)? NOT the agent misunderstanding clear info.
Answer YES or NO:

**Q2: keywords_location**
Did the agent navigate based on superficial keyword matching without understanding the actual problem?
Answer YES or NO:

**Q3: referred_code_location**
Did the agent fixate on code explicitly mentioned in the issue that's irrelevant to the actual fix?
Answer YES or NO:

**Q4: error_stack_trace**
Did the agent focus on stack trace location when root cause is elsewhere?
Answer YES or NO:

---

### PHASE 2: REPAIR FAILURES

#### 2A: Strategy Defects

**Q5: specific_case_overfitting**
Does the fix ONLY work for the example case but fail to generalize? (But doesn't break other things)
Answer YES or NO:

**Q6: evasive_repair**
Does the fix suppress/hide the error without addressing root cause?
Answer YES or NO:

**Q7: redundant_erroneous_implementation**
Did the agent add conflicting or duplicate logic that shouldn't exist?
Answer YES or NO:

#### 2B: Implementation Defects

**Q8: algorithmic_implementation**
Is the core algorithm/math/data structure logic wrong? (NOT control flow)
Answer YES or NO:

**Q9: control_flow**
EXAMPLE: In django-11433, the agent used `f.name not in cleaned_data` but should have checked if the value is empty/None. This is a conditional logic error.
Are there errors in if/else conditions, loop bounds, early returns, or exception handling?
Answer YES or NO:

**Q10: boundary_handling**
Are edge cases not handled properly?
Answer YES or NO:

**Q11: data_processing_errors**
Are there wrong data types or transformations?
Answer YES or NO:

**Q12: insufficient_domain_knowledge**
Does the agent lack framework/library expertise needed for the fix?
Answer YES or NO:

#### 2C: Incomplete Repair

**Q13: inheritance**
Did agent fail to update subclasses or parent classes that also need changes?
Answer YES or NO:

**Q14: interface_contract**
Did the agent violate API contracts?
Answer YES or NO:

**Q15: component_coordination**
Do multiple components need changes but agent only modified some?
Answer YES or NO:

**Q16: recurring_pattern**
Is the same bug in multiple places but agent only fixed some?
Answer YES or NO:

**Q17: issue_interference**
EXAMPLE: In django-14140, agent implemented a fix for a reported bug but it BROKE existing behavior that other code depended on.
Does the agent's fix cause ANY functionality that was WORKING BEFORE to now FAIL?
Answer YES or NO:

---

### PHASE 3: VERIFICATION FAILURES

**Q18: reproduction_script_failure**
Did agent fail to create a working reproduction script?
Answer YES or NO:

**Q19: reproduction_output_misreading**
EXAMPLE: In django-13195, agent ran a test after making changes. The results were IDENTICAL before and after, meaning the test didn't validate anything. Agent thought success when test was ineffective.
Did agent RUN tests/scripts and MISINTERPRET what the output means?
Answer YES or NO:

**Q20: insufficient_verification_capability**
Is agent unable to properly verify if fix works?
Answer YES or NO:

**Q21: non_progressive_iteration**
Does agent loop without making meaningful progress?
Answer YES or NO:

**Q22: blindly_strategy_switching**
Does agent randomly try approaches without learning from failures?
Answer YES or NO:

**Q23: verification_abandonment**
Did agent STOP running tests and declare success prematurely?
Answer YES or NO:

**Q24: verification_weakening**
EXAMPLE: In django-16454, after the fix didn't work, agent modified the validation script to "better align with real Django usage" instead of fixing the code.
Did agent MODIFY test files to make tests easier to pass?
Answer YES or NO:

**Q25: context_forgetting**
Did agent lose track of previous findings?
Answer YES or NO:

---

## FINAL OUTPUT
List ONLY the categories where you answered YES:
FAILURE MODES: [comma-separated list]
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
    if not text:
        return []

    # Method 1: Look for the FAILURE MODES: line at the end
    lines = text.strip().split("\n")
    for line in reversed(lines):
        if "FAILURE MODES:" in line.upper():
            after_colon = line.split(":", 1)[-1].strip()
            after_colon = after_colon.strip("[]")
            if not after_colon or after_colon == "":
                continue
            found = set()
            for item in after_colon.split(","):
                item = item.strip().lower().replace(" ", "_").replace("-", "_")
                if not item:
                    continue
                for cat in FAILURE_CATEGORIES:
                    if cat in item or item in cat:
                        found.add(cat)
            if found:
                return list(found)

    # Method 2: Parse YES/NO answers for each Q#: category pattern
    found = set()
    for cat in FAILURE_CATEGORIES:
        cat_pattern = cat.replace("_", "[_ ]")
        # Pattern: **Q#: category_name** ... Answer YES or NO:\nYES
        pattern = rf"\*\*Q\d+[:\s]+{cat_pattern}\*\*.*?Answer YES or NO[:\s]*\n\s*YES\b"
        if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
            found.add(cat)

    return list(found)


async def grade_with_hybrid_prompt(
    task_id: str,
    gold_patch: str,
    trajectory_text: str,
    model: str = "claude-sonnet-4-20250514",
) -> dict:
    client = anthropic.Anthropic()
    prompt = HYBRID_PROMPT.format(gold_patch=gold_patch)

    full_prompt = f"""{prompt}

## AGENT TRAJECTORY
{trajectory_text}

Now answer each question Q1-Q25 with YES or NO and brief explanation, then provide FAILURE MODES summary."""

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


# Test tasks - different from example tasks to avoid data leakage
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
    print("Loading SWE-bench data...")
    swebench_data = load_swebench_data()

    print("\nTesting HYBRID prompt (hierarchical + few-shot examples)...\n")

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

        result = await grade_with_hybrid_prompt(task_id, gold_patch, traj_text)
        pred = result.get("predicted", [])
        overlap = set(pred) & set(ground_truth)

        print(f"\nPredicted: {pred}")
        print(f"Overlap: {list(overlap) if overlap else 'NONE'}")

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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "hybrid_prompt_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR / 'hybrid_prompt_results.json'}")


if __name__ == "__main__":
    asyncio.run(main())

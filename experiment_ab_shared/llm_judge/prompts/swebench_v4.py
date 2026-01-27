"""SWE-bench V4 prompt configuration for LLM judge feature extraction.

This module provides NEW semantic features that are orthogonal to V3 and unified features.
These features focus on:
1. Framework internals depth (not captured by domain_knowledge)
2. Edge case vs main path (not captured by atypicality)
3. Solution discovery vs implementation (different from solution_hint)
4. Behavioral change type (not captured by existing features)

V4 features (6 new, designed to be orthogonal):
- requires_framework_internals: Depth into framework-specific internals
- is_edge_case: Edge case/corner case vs main path bug
- solution_discovery_needed: How much investigation needed to find fix
- involves_timing_or_ordering: Order-dependent or timing-sensitive issues
- fix_pattern_type: Type of fix (value change, add method, behavior change, etc.)
- test_coverage_gap: Was this a gap in existing test coverage?
"""

from typing import Any, Dict

from experiment_ab_shared.llm_judge.prompt_config import FeatureDefinition, PromptConfig

SWEBENCH_V4_FEATURES = [
    FeatureDefinition(
        name="requires_framework_internals",
        min_value=1,
        max_value=5,
        description="How deep into framework-specific internals does the fix require? (1=surface API, 5=deep internals)",
    ),
    FeatureDefinition(
        name="is_edge_case",
        min_value=0,
        max_value=1,
        description="Is this an edge case/corner case (1) or main path bug (0)?",
    ),
    FeatureDefinition(
        name="solution_discovery_needed",
        min_value=1,
        max_value=5,
        description="How much investigation is needed to find the fix location/approach? (1=obvious, 5=extensive search)",
    ),
    FeatureDefinition(
        name="involves_timing_or_ordering",
        min_value=0,
        max_value=1,
        description="Does the bug involve timing, ordering, or race conditions? (0=no, 1=yes)",
    ),
    FeatureDefinition(
        name="fix_pattern_type",
        min_value=0,
        max_value=4,
        description="Type of fix (0=value/string change, 1=add method/feature, 2=fix logic/condition, 3=refactor/restructure, 4=behavior change)",
    ),
    FeatureDefinition(
        name="test_coverage_gap",
        min_value=0,
        max_value=1,
        description="Was this bug due to missing test coverage for this scenario? (0=no, 1=yes)",
    ),
]

SWEBENCH_V4_PROMPT_TEMPLATE = """You are analyzing a SWE-bench coding task to predict its difficulty.
Analyze ONLY the static task information (no code execution).

## TASK INFORMATION

**Instance ID:** {instance_id}
**Repository:** {repo}
**Version:** {version}

**Problem Statement:**
{problem_statement}

**Gold Patch (correct solution):**
```diff
{patch}
```

**Tests that should pass after fix (FAIL_TO_PASS):**
{fail_to_pass}

{hints_section}

## FEATURES TO EVALUATE

Analyze the problem and patch to evaluate these 6 features.
Focus on aspects NOT captured by generic difficulty measures.

### 1. Requires Framework Internals (requires_framework_internals: 1-5)
How deep into framework-specific internals does understanding/fixing this require?
- 1: Surface-level API usage, standard Python
- 2: Common framework patterns (Django views, pytest fixtures)
- 3: Framework-specific features (ORM queries, middleware)
- 4: Framework internals (meta-classes, signal handling)
- 5: Deep internals (migration engine, async machinery, SQL compilation)

### 2. Is Edge Case (is_edge_case: 0/1)
Is this an edge case/corner case or a main path bug?
- 0: Main path - common usage pattern, well-tested scenario
- 1: Edge case - unusual inputs, rare configurations, boundary conditions

### 3. Solution Discovery Needed (solution_discovery_needed: 1-5)
How much investigation is needed to find where/how to fix?
- 1: Location and approach are obvious from problem statement
- 2: Location clear, approach needs some thought
- 3: Need to trace through code to find right location
- 4: Multiple possible locations, need investigation
- 5: Extensive debugging/tracing required to find root cause

### 4. Involves Timing or Ordering (involves_timing_or_ordering: 0/1)
Does the bug involve timing, ordering dependencies, or race conditions?
- 0: No timing/ordering issues
- 1: Yes - order of operations matters, race conditions, async timing, initialization order

### 5. Fix Pattern Type (fix_pattern_type: 0-4)
What type of fix is needed?
- 0: Simple value/string/constant change
- 1: Add missing method, feature, or handler
- 2: Fix logic, condition, or control flow
- 3: Refactor or restructure existing code
- 4: Change existing behavior (potentially breaking)

### 6. Test Coverage Gap (test_coverage_gap: 0/1)
Was this bug likely due to missing test coverage for this specific scenario?
- 0: No - tests existed but didn't catch this
- 1: Yes - this scenario wasn't tested before

## OUTPUT FORMAT

Respond with ONLY a JSON object. No markdown, no extra text.

{{
    "requires_framework_internals": <1-5>,
    "is_edge_case": <0 or 1>,
    "solution_discovery_needed": <1-5>,
    "involves_timing_or_ordering": <0 or 1>,
    "fix_pattern_type": <0-4>,
    "test_coverage_gap": <0 or 1>,
    "reasoning": "<2-3 sentences on what makes this task hard or easy>"
}}
"""


def format_swebench_v4_prompt(task: Dict[str, Any]) -> str:
    """Format the SWE-bench V4 prompt with task-specific information."""
    hints_text = task.get("hints_text", "")
    hints_section = f"**Hints:**\n{hints_text}" if hints_text and hints_text.strip() else ""

    return SWEBENCH_V4_PROMPT_TEMPLATE.format(
        instance_id=task.get("instance_id", ""),
        repo=task.get("repo", ""),
        version=task.get("version", "unknown"),
        problem_statement=task.get("problem_statement", ""),
        patch=task.get("patch", ""),
        fail_to_pass=task.get("FAIL_TO_PASS", "[]"),
        hints_section=hints_section,
    )


SWEBENCH_V4_CONFIG = PromptConfig(
    name="swebench_v4",
    features=SWEBENCH_V4_FEATURES,
    prompt_template=SWEBENCH_V4_PROMPT_TEMPLATE,
    task_id_field="instance_id",
    format_prompt_fn=format_swebench_v4_prompt,
)

"""SWE-bench Pro V5 prompt configuration - final feature set.

Based on V4 validation, this version uses only the 5 most predictive features:

LLM features (4):
1. fix_complexity (1-5) - r=0.284**
2. verification_difficulty (1-5) - r=0.328***
3. standard_pattern_available (0/1) - r=-0.274**
4. integration_complexity (1-5) - r=0.212* (NEW in V4)

Deterministic features (1):
- num_files_modified - r=0.307** (computed from patch, not LLM)

Dropped features (not significant in V3/V4):
- fix_in_description (p=0.72)
- requires_external_spec (p=0.18)
- involves_async_or_timing (p=0.95)
- is_migration_or_compat (p=0.48)
- is_edge_case_handling (p=0.30)
- creates_new_abstraction (p=0.63)
- requires_new_parsing_logic (p=0.11)

This is the streamlined prompt for production use.
"""

from typing import Any, Dict

from experiment_ab_shared.llm_judge.prompt_config import FeatureDefinition, PromptConfig

SWEBENCH_PRO_V5_FEATURES = [
    FeatureDefinition(
        name="fix_complexity",
        min_value=1,
        max_value=5,
        description="How complex is the code change logic? (1=trivial, 5=very complex)",
    ),
    FeatureDefinition(
        name="verification_difficulty",
        min_value=1,
        max_value=5,
        description="How hard to test/verify the fix? (1=trivial, 5=very hard)",
    ),
    FeatureDefinition(
        name="standard_pattern_available",
        min_value=0,
        max_value=1,
        description="Is this a well-documented pattern with existing examples? (0=no, 1=yes)",
    ),
    FeatureDefinition(
        name="integration_complexity",
        min_value=1,
        max_value=5,
        description="How tightly must changes integrate with existing code? (1=isolated/greenfield, 5=deeply integrated)",
    ),
]

SWEBENCH_PRO_V5_PROMPT_TEMPLATE = """You are analyzing a SWE-bench coding task to predict its difficulty.
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

**Regression tests (PASS_TO_PASS):**
{pass_to_pass}

{hints_section}

## FEATURES TO EVALUATE

Analyze the problem and patch to evaluate these 4 features.
Focus on what makes the SOLUTION hard, not what the PROBLEM looks like.

### 1. Fix Complexity (fix_complexity: 1-5)
How complex is the actual code change logic?
- 1: Trivial (add parameter, change value)
- 2: Simple (straightforward logic change)
- 3: Moderate (understand context, multiple changes)
- 4: Complex (algorithmic, interdependent fixes)
- 5: Very complex (architectural, subtle edge cases)

### 2. Verification Difficulty (verification_difficulty: 1-5)
How hard is it to test/verify the fix works?
- 1: Trivial (obvious pass/fail)
- 2: Easy (straightforward test cases)
- 3: Moderate (some edge cases)
- 4: Hard (subtle correctness, complex setup)
- 5: Very hard (rare edge cases, hard to reproduce)

### 3. Standard Pattern Available (standard_pattern_available: 0/1)
Is this a well-documented pattern with existing examples?
- 0: Novel solution needed, no clear pattern to follow
- 1: Well-documented pattern (e.g., "add YAML representer", "implement __eq__", "add middleware")

### 4. Integration Complexity (integration_complexity: 1-5)
How tightly must the changes integrate with existing code?
Consider: Is this greenfield code or does it require understanding multiple subsystems?

- 1: Self-contained/greenfield - new code with clear boundaries, minimal dependencies on existing code
- 2: Simple extension - adds to existing code with clear interface, one touchpoint
- 3: Moderate integration - changes interact with several existing components
- 4: Deep integration - requires understanding multiple subsystems, threading state across components
- 5: Pervasive integration - affects system-wide behavior, many touchpoints, understanding full architecture

Examples:
- Level 1: Creating a new import script, new standalone utility function
- Level 2: Adding a new API endpoint that follows existing patterns
- Level 3: Modifying a component that interacts with 2-3 others
- Level 4: Bug fix requiring understanding of caching + sessions + audit logging
- Level 5: Refactoring core abstractions used throughout the codebase

## OUTPUT FORMAT

Respond with ONLY a JSON object. No markdown, no extra text.

{{
    "fix_complexity": <1-5>,
    "verification_difficulty": <1-5>,
    "standard_pattern_available": <0 or 1>,
    "integration_complexity": <1-5>,
    "reasoning": "<2-3 sentences on what makes the SOLUTION hard or easy>"
}}
"""


def format_swebench_pro_v5_prompt(task: Dict[str, Any]) -> str:
    """Format the SWE-bench Pro V5 prompt."""
    hints_text = task.get("hints_text", "") or ""
    hints_section = f"**Hints:**\n{hints_text}" if hints_text.strip() else ""

    problem_statement = task.get("problem_statement", "") or ""
    if len(problem_statement) > 12000:
        problem_statement = problem_statement[:12000]

    patch = task.get("patch", "") or ""
    if len(patch) > 8000:
        patch = patch[:8000]

    fail_to_pass = task.get("fail_to_pass") or task.get("FAIL_TO_PASS") or "[]"
    pass_to_pass = task.get("pass_to_pass") or task.get("PASS_TO_PASS") or "[]"

    return SWEBENCH_PRO_V5_PROMPT_TEMPLATE.format(
        instance_id=task.get("instance_id", ""),
        repo=task.get("repo", ""),
        version=task.get("version", "unknown"),
        problem_statement=problem_statement,
        patch=patch,
        fail_to_pass=fail_to_pass,
        pass_to_pass=pass_to_pass,
        hints_section=hints_section,
    )


SWEBENCH_PRO_V5_CONFIG = PromptConfig(
    name="swebench_pro_v5",
    features=SWEBENCH_PRO_V5_FEATURES,
    prompt_template=SWEBENCH_PRO_V5_PROMPT_TEMPLATE,
    task_id_field="instance_id",
    truncation_limits={
        "problem_statement": 12000,
        "patch": 8000,
    },
    format_prompt_fn=format_swebench_pro_v5_prompt,
)

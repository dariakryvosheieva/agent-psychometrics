"""SWE-bench V2 prompt configuration for LLM judge feature extraction.

This module combines the strong existing SWE-bench features (r > 0.1 or p < 0.01)
with 3 additional features from swebench_pro_v5.py.

Strong existing features (6 - kept):
- fix_complexity (r=0.541***)
- logical_reasoning_required (r=0.526***)
- fix_locality (r=0.462***)
- atypicality (r=0.455***)
- domain_knowledge_required (r=0.423***)
- fix_in_description (r=-0.279***)

Dropped (weak or not significant):
- error_message_provided (r=-0.014, not significant)
- reproduction_steps (r=0.119**, weak)
- problem_clarity (r=-0.097*, weak)

New V5 features (3):
- verification_difficulty (r=0.328*** in V5)
- standard_pattern_available (r=-0.274** in V5)
- integration_complexity (r=0.212* in V5)

Total: 9 LLM features
"""

from typing import Any, Dict

from experiment_ab_shared.llm_judge.prompt_config import FeatureDefinition, PromptConfig

SWEBENCH_V2_FEATURES = [
    # === Strong existing features (6) ===
    FeatureDefinition(
        name="fix_in_description",
        min_value=0,
        max_value=3,
        description="Does the problem statement contain or hint at the solution? (0=none, 3=exact fix)",
    ),
    FeatureDefinition(
        name="fix_locality",
        min_value=1,
        max_value=3,
        description="How localized is the fix? (1=single location, 3=multiple files)",
    ),
    FeatureDefinition(
        name="domain_knowledge_required",
        min_value=1,
        max_value=5,
        description="How much specialized knowledge is needed? (1=basic Python, 5=obscure APIs)",
    ),
    FeatureDefinition(
        name="fix_complexity",
        min_value=1,
        max_value=5,
        description="How complex is the actual fix? (1=trivial, 5=very complex)",
    ),
    FeatureDefinition(
        name="logical_reasoning_required",
        min_value=1,
        max_value=5,
        description="How much logical reasoning is needed? (1=mechanical, 5=deep reasoning)",
    ),
    FeatureDefinition(
        name="atypicality",
        min_value=1,
        max_value=5,
        description="How unusual is this bug pattern? (1=very common, 5=rare/novel)",
    ),
    # === New V5 features (3) ===
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
        description="How tightly must changes integrate with existing code? (1=isolated, 5=deeply integrated)",
    ),
]

SWEBENCH_V2_PROMPT_TEMPLATE = """You are analyzing a SWE-bench coding task to predict its difficulty.
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

Analyze the problem and patch to evaluate these 9 features.
Focus on what makes the SOLUTION hard, not what the PROBLEM looks like.

### 1. Fix Information in Description (fix_in_description: 0-3)
Does the problem statement hint at the solution?
- 0: No hint at all
- 1: Vague hint or general direction
- 2: Clear description of what needs to change
- 3: Exact code fix provided

### 2. Fix Locality (fix_locality: 1-3)
How localized is the fix based on the patch?
- 1: Single location, few lines (1-5 lines)
- 2: Multiple locations in same file, or moderate (6-20 lines)
- 3: Multiple files or significant changes (>20 lines)

### 3. Domain Knowledge Required (domain_knowledge_required: 1-5)
How much specialized knowledge is needed?
- 1: Basic Python, obvious fix
- 2: Standard library knowledge
- 3: Framework-specific knowledge (Django, pytest, numpy)
- 4: Deep library internals
- 5: Obscure APIs or highly specialized domain

### 4. Fix Complexity (fix_complexity: 1-5)
How complex is the actual code change logic?
- 1: Trivial (add parameter, change value)
- 2: Simple (straightforward logic change)
- 3: Moderate (understand context, multiple changes)
- 4: Complex (algorithmic, interdependent fixes)
- 5: Very complex (architectural, subtle edge cases)

### 5. Logical Reasoning Required (logical_reasoning_required: 1-5)
How much logical reasoning is needed?
- 1: Mechanical fix, no reasoning
- 2: Simple cause-effect reasoning
- 3: Multi-step reasoning required
- 4: Complex reasoning with multiple factors
- 5: Deep reasoning about edge cases, invariants

### 6. Atypicality (atypicality: 1-5)
How unusual is this bug pattern?
- 1: Very common (typo, off-by-one, missing null check)
- 2: Common (incorrect condition, wrong default)
- 3: Moderately unusual
- 4: Unusual bug pattern
- 5: Rare or novel bug pattern

### 7. Verification Difficulty (verification_difficulty: 1-5)
How hard is it to test/verify the fix works?
- 1: Trivial (obvious pass/fail)
- 2: Easy (straightforward test cases)
- 3: Moderate (some edge cases)
- 4: Hard (subtle correctness, complex setup)
- 5: Very hard (rare edge cases, hard to reproduce)

### 8. Standard Pattern Available (standard_pattern_available: 0/1)
Is this a well-documented pattern with existing examples?
- 0: Novel solution needed, no clear pattern to follow
- 1: Well-documented pattern (e.g., "add YAML representer", "implement __eq__", "add middleware")

### 9. Integration Complexity (integration_complexity: 1-5)
How tightly must the changes integrate with existing code?
- 1: Self-contained/greenfield - new code with clear boundaries
- 2: Simple extension - adds to existing with clear interface
- 3: Moderate integration - interacts with several components
- 4: Deep integration - requires understanding multiple subsystems
- 5: Pervasive integration - affects system-wide behavior

## OUTPUT FORMAT

Respond with ONLY a JSON object. No markdown, no extra text.

{{
    "fix_in_description": <0-3>,
    "fix_locality": <1-3>,
    "domain_knowledge_required": <1-5>,
    "fix_complexity": <1-5>,
    "logical_reasoning_required": <1-5>,
    "atypicality": <1-5>,
    "verification_difficulty": <1-5>,
    "standard_pattern_available": <0 or 1>,
    "integration_complexity": <1-5>,
    "reasoning": "<2-3 sentences on what makes the SOLUTION hard or easy>"
}}
"""


def format_swebench_v2_prompt(task: Dict[str, Any]) -> str:
    """Format the SWE-bench V2 prompt with task-specific information."""
    hints_text = task.get("hints_text", "")
    hints_section = f"**Hints:**\n{hints_text}" if hints_text and hints_text.strip() else ""

    problem_statement = task.get("problem_statement", "")
    if len(problem_statement) > 12000:
        problem_statement = problem_statement[:12000]

    patch = task.get("patch", "")
    if len(patch) > 8000:
        patch = patch[:8000]

    return SWEBENCH_V2_PROMPT_TEMPLATE.format(
        instance_id=task.get("instance_id", ""),
        repo=task.get("repo", ""),
        version=task.get("version", "unknown"),
        problem_statement=problem_statement,
        patch=patch,
        fail_to_pass=task.get("FAIL_TO_PASS", "[]"),
        pass_to_pass=task.get("PASS_TO_PASS", "[]"),
        hints_section=hints_section,
    )


SWEBENCH_V2_CONFIG = PromptConfig(
    name="swebench_v2",
    features=SWEBENCH_V2_FEATURES,
    prompt_template=SWEBENCH_V2_PROMPT_TEMPLATE,
    task_id_field="instance_id",
    truncation_limits={
        "problem_statement": 12000,
        "patch": 8000,
    },
    format_prompt_fn=format_swebench_v2_prompt,
)

"""SWE-bench prompt configuration for LLM judge feature extraction.

This module defines the prompt template and feature definitions for extracting
semantic features from SWE-bench Verified tasks.
"""

from typing import Any, Dict

from experiment_ab_shared.llm_judge.prompt_config import FeatureDefinition, PromptConfig

# Feature definitions for SWE-bench
SWEBENCH_FEATURES = [
    FeatureDefinition(
        name="fix_in_description",
        min_value=0,
        max_value=3,
        description="Does the problem statement contain or hint at the solution? (0=none, 3=exact fix)",
    ),
    FeatureDefinition(
        name="problem_clarity",
        min_value=1,
        max_value=5,
        description="How clear and well-specified is the problem? (1=vague, 5=crystal clear)",
    ),
    FeatureDefinition(
        name="error_message_provided",
        min_value=0,
        max_value=1,
        description="Does the problem include an error message or traceback? (0=no, 1=yes)",
    ),
    FeatureDefinition(
        name="reproduction_steps",
        min_value=0,
        max_value=1,
        description="Are concrete reproduction steps provided? (0=no, 1=yes)",
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
]

# The prompt template for SWE-bench tasks
SWEBENCH_PROMPT_TEMPLATE = """You are analyzing a SWE-bench coding task to predict its difficulty.
You will analyze ONLY the static task information (no code execution or environment access).

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

Analyze the problem statement and gold patch to evaluate these 9 semantic features.
Be precise and consistent with your ratings.

### 1. Fix Information in Description (fix_in_description: 0-3)
Does the problem statement contain or hint at the solution?
- 0: No hint at the solution at all
- 1: Vague hint or general direction
- 2: Clear description of what needs to change
- 3: Exact code fix or detailed solution provided

### 2. Problem Clarity (problem_clarity: 1-5)
How clear and well-specified is the problem?
- 1: Very vague, unclear what's actually wrong
- 2: Somewhat clear but missing key details
- 3: Reasonably clear, some ambiguity
- 4: Clear with good context
- 5: Crystal clear with reproduction steps and expected behavior

### 3. Error Message/Traceback (error_message_provided: 0/1)
Does the problem include an error message or traceback?
- 0: No error message provided
- 1: Error message, traceback, or exception shown

### 4. Reproduction Steps (reproduction_steps: 0/1)
Are concrete reproduction steps provided?
- 0: No concrete reproduction steps
- 1: Has reproduction steps (code snippet, test case, or commands)

### 5. Fix Locality (fix_locality: 1-3)
How localized is the fix based on the patch?
- 1: Single location, few lines changed (1-5 lines)
- 2: Multiple locations in same file, or moderate changes (6-20 lines)
- 3: Multiple files or significant changes (>20 lines)

### 6. Domain Knowledge Required (domain_knowledge_required: 1-5)
How much specialized knowledge is needed to understand and fix this?
- 1: Basic Python, obvious fix anyone could make
- 2: Standard library knowledge needed
- 3: Framework-specific knowledge (Django, pytest, numpy, etc.)
- 4: Deep understanding of the library's internals
- 5: Obscure APIs, protocols, or highly specialized domain knowledge

### 7. Fix Complexity (fix_complexity: 1-5)
How complex is the actual fix?
- 1: Trivial (add parameter, change value, simple one-liner)
- 2: Simple (straightforward logic change)
- 3: Moderate (requires understanding context, multiple changes)
- 4: Complex (algorithmic changes, multiple interdependent fixes)
- 5: Very complex (architectural changes, subtle edge cases, tricky bugs)

### 8. Logical Reasoning Required (logical_reasoning_required: 1-5)
How much logical reasoning is needed to arrive at the fix?
- 1: Mechanical fix, no reasoning needed
- 2: Simple cause-effect reasoning
- 3: Multi-step reasoning required
- 4: Complex reasoning with multiple factors
- 5: Deep reasoning about edge cases, invariants, or system behavior

### 9. Atypicality (atypicality: 1-5)
How unusual is this bug pattern?
- 1: Very common bug pattern (typo, off-by-one, missing null check)
- 2: Common pattern (incorrect condition, wrong default)
- 3: Moderately unusual
- 4: Unusual bug pattern
- 5: Rare or novel bug pattern

## OUTPUT FORMAT

Respond with ONLY a JSON object containing all features. No markdown, no extra text.

{{
    "fix_in_description": <0-3>,
    "problem_clarity": <1-5>,
    "error_message_provided": <0 or 1>,
    "reproduction_steps": <0 or 1>,
    "fix_locality": <1-3>,
    "domain_knowledge_required": <1-5>,
    "fix_complexity": <1-5>,
    "logical_reasoning_required": <1-5>,
    "atypicality": <1-5>,
    "reasoning": "<2-3 sentence summary of the key difficulty factors>"
}}
"""


def format_swebench_prompt(task: Dict[str, Any]) -> str:
    """Format the SWE-bench prompt with task-specific information.

    Args:
        task: SWE-bench task dict with keys:
            - instance_id: SWE-bench instance ID
            - repo: Repository name (e.g., "django/django")
            - version: Version string
            - problem_statement: The issue description
            - patch: The gold solution patch
            - FAIL_TO_PASS: Tests that should pass after fix
            - PASS_TO_PASS: Regression tests
            - hints_text: Optional hints (may be empty)

    Returns:
        Formatted prompt string
    """
    hints_text = task.get("hints_text", "")
    hints_section = ""
    if hints_text and hints_text.strip():
        hints_section = f"**Hints:**\n{hints_text}"

    # Truncate very long fields to avoid context overflow
    problem_statement = task.get("problem_statement", "")
    if len(problem_statement) > 12000:
        problem_statement = problem_statement[:12000]

    patch = task.get("patch", "")
    if len(patch) > 8000:
        patch = patch[:8000]

    return SWEBENCH_PROMPT_TEMPLATE.format(
        instance_id=task.get("instance_id", ""),
        repo=task.get("repo", ""),
        version=task.get("version", "unknown"),
        problem_statement=problem_statement,
        patch=patch,
        fail_to_pass=task.get("FAIL_TO_PASS", "[]"),
        pass_to_pass=task.get("PASS_TO_PASS", "[]"),
        hints_section=hints_section,
    )


# The main configuration object
SWEBENCH_CONFIG = PromptConfig(
    name="swebench",
    features=SWEBENCH_FEATURES,
    prompt_template=SWEBENCH_PROMPT_TEMPLATE,
    task_id_field="instance_id",
    truncation_limits={
        "problem_statement": 12000,
        "patch": 8000,
    },
    format_prompt_fn=format_swebench_prompt,
)

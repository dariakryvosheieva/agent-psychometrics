"""SWE-bench Pro unified prompt configuration.

This module defines the unified prompt for SWE-bench Pro tasks using
standardized features that enable fair comparison across datasets.

Features (9 total):
- 8 core features (shared with all datasets)
- 1 dataset-specific: integration_complexity (code changes integrate with codebase)

Task type: Bug fixes in Python repositories (more challenging than SWE-bench Verified)
"""

from typing import Any, Dict

from experiment_ab_shared.llm_judge.prompt_config import PromptConfig
from experiment_ab_shared.llm_judge.prompts.unified_features import (
    CODE_DATASET_FEATURES,
    NO_SOLUTION_FEATURES,
    COMPLETENESS_INSTRUCTION,
    OUTPUT_FORMAT_9_FEATURES_CODE,
    OUTPUT_FORMAT_7_FEATURES,
    SOLUTION_HINT_SCALE,
    PROBLEM_CLARITY_SCALE,
    SOLUTION_COMPLEXITY_SCALE_CODE,
    DOMAIN_KNOWLEDGE_SCALE_CODE,
    LOGICAL_REASONING_SCALE,
    ATYPICALITY_SCALE,
    VERIFICATION_DIFFICULTY_SCALE,
    STANDARD_PATTERN_SCALE,
    INTEGRATION_COMPLEXITY_SCALE,
)


SWEBENCH_PRO_UNIFIED_PROMPT_TEMPLATE = """You are analyzing a SWE-bench Pro coding task to predict its difficulty.
This is a BUG FIX task in a Python repository. SWE-bench Pro contains more challenging tasks
than the standard SWE-bench Verified dataset. You will analyze ONLY the static task information.

{completeness_instruction}

## TASK INFORMATION

**Instance ID:** {{instance_id}}
**Repository:** {{repo}}
**Version:** {{version}}

**Problem Statement:**
{{problem_statement}}

**Gold Patch (correct solution):**
```diff
{{patch}}
```

**Tests that should pass after fix (FAIL_TO_PASS):**
{{fail_to_pass}}

**Regression tests (PASS_TO_PASS):**
{{pass_to_pass}}

{{hints_section}}

## FEATURES TO EVALUATE

Analyze the problem statement and gold patch to evaluate these 9 features.
Focus on what makes the SOLUTION hard, not just what the PROBLEM looks like.
Be precise and consistent with your ratings.

{solution_hint_scale}

{problem_clarity_scale}

{solution_complexity_scale}

{domain_knowledge_scale}

{logical_reasoning_scale}

{atypicality_scale}

{verification_difficulty_scale}

{standard_pattern_scale}

{integration_complexity_scale}

{output_format}
""".format(
    completeness_instruction=COMPLETENESS_INSTRUCTION,
    solution_hint_scale=SOLUTION_HINT_SCALE,
    problem_clarity_scale=PROBLEM_CLARITY_SCALE,
    solution_complexity_scale=SOLUTION_COMPLEXITY_SCALE_CODE,
    domain_knowledge_scale=DOMAIN_KNOWLEDGE_SCALE_CODE,
    logical_reasoning_scale=LOGICAL_REASONING_SCALE,
    atypicality_scale=ATYPICALITY_SCALE,
    verification_difficulty_scale=VERIFICATION_DIFFICULTY_SCALE,
    standard_pattern_scale=STANDARD_PATTERN_SCALE,
    integration_complexity_scale=INTEGRATION_COMPLEXITY_SCALE,
    output_format=OUTPUT_FORMAT_9_FEATURES_CODE,
)


def format_swebench_pro_unified_prompt(task: Dict[str, Any]) -> str:
    """Format the SWE-bench Pro unified prompt with task-specific information.

    Args:
        task: SWE-bench Pro task dict with keys:
            - instance_id: SWE-bench instance ID
            - repo: Repository name (e.g., "django/django")
            - version: Version string
            - problem_statement: The issue description
            - patch: The gold solution patch
            - fail_to_pass or FAIL_TO_PASS: Tests that should pass after fix
            - pass_to_pass or PASS_TO_PASS: Regression tests
            - hints_text: Optional hints (may be empty)

    Returns:
        Formatted prompt string
    """
    hints_text = task.get("hints_text", "") or ""
    hints_section = ""
    if hints_text and hints_text.strip():
        hints_section = f"**Hints:**\n{hints_text}"

    # No truncation needed - Claude Opus 4.5 has 200K token context
    problem_statement = task.get("problem_statement", "") or ""
    patch = task.get("patch", "") or ""

    # Handle both lowercase and uppercase field names
    fail_to_pass = task.get("fail_to_pass") or task.get("FAIL_TO_PASS") or "[]"
    pass_to_pass = task.get("pass_to_pass") or task.get("PASS_TO_PASS") or "[]"

    return SWEBENCH_PRO_UNIFIED_PROMPT_TEMPLATE.format(
        instance_id=task.get("instance_id", ""),
        repo=task.get("repo", ""),
        version=task.get("version", "unknown"),
        problem_statement=problem_statement,
        patch=patch,
        fail_to_pass=fail_to_pass,
        pass_to_pass=pass_to_pass,
        hints_section=hints_section,
    )


# The main configuration object
SWEBENCH_PRO_UNIFIED_CONFIG = PromptConfig(
    name="swebench_pro_unified",
    features=CODE_DATASET_FEATURES,
    prompt_template=SWEBENCH_PRO_UNIFIED_PROMPT_TEMPLATE,
    task_id_field="instance_id",
    truncation_limits={},  # No truncation needed with Claude Opus 4.5's 200K context
    format_prompt_fn=format_swebench_pro_unified_prompt,
)


# =============================================================================
# No-Solution Variant (for ablation study)
# =============================================================================

SWEBENCH_PRO_UNIFIED_NO_SOLUTION_PROMPT_TEMPLATE = """You are analyzing a SWE-bench Pro coding task to predict its difficulty.
This is a BUG FIX task in a Python repository. You do not have access to the solution patch.

{completeness_instruction}

## TASK INFORMATION

**Instance ID:** {{instance_id}}
**Repository:** {{repo}}
**Version:** {{version}}

**Problem Statement:**
{{problem_statement}}

**Tests that should pass after fix (FAIL_TO_PASS):**
{{fail_to_pass}}

**Regression tests (PASS_TO_PASS):**
{{pass_to_pass}}

{{hints_section}}

## FEATURES TO EVALUATE

{solution_hint_scale}

{problem_clarity_scale}

{domain_knowledge_scale}

{logical_reasoning_scale}

{atypicality_scale}

{verification_difficulty_scale}

{standard_pattern_scale}

{output_format}
""".format(
    completeness_instruction=COMPLETENESS_INSTRUCTION,
    solution_hint_scale=SOLUTION_HINT_SCALE,
    problem_clarity_scale=PROBLEM_CLARITY_SCALE,
    domain_knowledge_scale=DOMAIN_KNOWLEDGE_SCALE_CODE,
    logical_reasoning_scale=LOGICAL_REASONING_SCALE,
    atypicality_scale=ATYPICALITY_SCALE,
    verification_difficulty_scale=VERIFICATION_DIFFICULTY_SCALE,
    standard_pattern_scale=STANDARD_PATTERN_SCALE,
    output_format=OUTPUT_FORMAT_7_FEATURES,
)


def format_swebench_pro_unified_no_solution_prompt(task: Dict[str, Any]) -> str:
    """Format the prompt without the gold patch."""
    hints_text = task.get("hints_text", "") or ""
    hints_section = f"**Hints:**\n{hints_text}" if hints_text and hints_text.strip() else ""

    fail_to_pass = task.get("fail_to_pass") or task.get("FAIL_TO_PASS") or "[]"
    pass_to_pass = task.get("pass_to_pass") or task.get("PASS_TO_PASS") or "[]"

    return SWEBENCH_PRO_UNIFIED_NO_SOLUTION_PROMPT_TEMPLATE.format(
        instance_id=task.get("instance_id", ""),
        repo=task.get("repo", ""),
        version=task.get("version", "unknown"),
        problem_statement=task.get("problem_statement", "") or "",
        fail_to_pass=fail_to_pass,
        pass_to_pass=pass_to_pass,
        hints_section=hints_section,
    )


SWEBENCH_PRO_UNIFIED_NO_SOLUTION_CONFIG = PromptConfig(
    name="swebench_pro_unified_no_solution",
    features=NO_SOLUTION_FEATURES,
    prompt_template=SWEBENCH_PRO_UNIFIED_NO_SOLUTION_PROMPT_TEMPLATE,
    task_id_field="instance_id",
    truncation_limits={},
    format_prompt_fn=format_swebench_pro_unified_no_solution_prompt,
)


# =============================================================================
# Problem-Only Variant (for ablation study)
# =============================================================================

SWEBENCH_PRO_UNIFIED_PROBLEM_ONLY_PROMPT_TEMPLATE = """You are analyzing a SWE-bench Pro coding task to predict its difficulty.
This is a BUG FIX task in a Python repository. You only have access to the problem statement.

{completeness_instruction}

## TASK INFORMATION

**Instance ID:** {{instance_id}}

**Problem Statement:**
{{problem_statement}}

## FEATURES TO EVALUATE

{solution_hint_scale}

{problem_clarity_scale}

{domain_knowledge_scale}

{logical_reasoning_scale}

{atypicality_scale}

{verification_difficulty_scale}

{standard_pattern_scale}

{output_format}
""".format(
    completeness_instruction=COMPLETENESS_INSTRUCTION,
    solution_hint_scale=SOLUTION_HINT_SCALE,
    problem_clarity_scale=PROBLEM_CLARITY_SCALE,
    domain_knowledge_scale=DOMAIN_KNOWLEDGE_SCALE_CODE,
    logical_reasoning_scale=LOGICAL_REASONING_SCALE,
    atypicality_scale=ATYPICALITY_SCALE,
    verification_difficulty_scale=VERIFICATION_DIFFICULTY_SCALE,
    standard_pattern_scale=STANDARD_PATTERN_SCALE,
    output_format=OUTPUT_FORMAT_7_FEATURES,
)


def format_swebench_pro_unified_problem_only_prompt(task: Dict[str, Any]) -> str:
    """Format the prompt with only problem statement."""
    return SWEBENCH_PRO_UNIFIED_PROBLEM_ONLY_PROMPT_TEMPLATE.format(
        instance_id=task.get("instance_id", ""),
        problem_statement=task.get("problem_statement", "") or "",
    )


SWEBENCH_PRO_UNIFIED_PROBLEM_ONLY_CONFIG = PromptConfig(
    name="swebench_pro_unified_problem_only",
    features=NO_SOLUTION_FEATURES,
    prompt_template=SWEBENCH_PRO_UNIFIED_PROBLEM_ONLY_PROMPT_TEMPLATE,
    task_id_field="instance_id",
    truncation_limits={},
    format_prompt_fn=format_swebench_pro_unified_problem_only_prompt,
)

"""GSO (Software Optimization Benchmark) unified prompt configuration.

This module defines the unified prompt for GSO tasks using
standardized features that enable fair comparison across datasets.

Features (9 total):
- 8 core features (shared with all datasets)
- 1 dataset-specific: integration_complexity (code changes integrate with codebase)

Task type: Performance optimization tasks (NOT bug fixes)
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
    SOLUTION_COMPLEXITY_SCALE_OPTIMIZATION,
    DOMAIN_KNOWLEDGE_SCALE_OPTIMIZATION,
    LOGICAL_REASONING_SCALE,
    ATYPICALITY_SCALE,
    VERIFICATION_DIFFICULTY_SCALE,
    STANDARD_PATTERN_SCALE,
    INTEGRATION_COMPLEXITY_SCALE,
)


GSO_UNIFIED_PROMPT_TEMPLATE = """You are analyzing a GSO (Software Optimization Benchmark) task to predict its difficulty.
This is a PERFORMANCE OPTIMIZATION task, NOT a bug fix. The goal is to make code run faster
while maintaining correctness. You will analyze ONLY the static task information.

{completeness_instruction}

## TASK INFORMATION

**Instance ID:** {{instance_id}}
**Repository:** {{repo}}
**API/Function being optimized:** {{api}}

**Test Script (performance scenario to optimize):**
```python
{{prob_script}}
```

**Gold Patch (optimization solution):**
```diff
{{gt_diff}}
```

{{hints_section}}

## FEATURES TO EVALUATE

Analyze the test script and optimization patch to evaluate these 9 features.
Focus on what makes the OPTIMIZATION hard, not just what the code looks like.
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
    solution_complexity_scale=SOLUTION_COMPLEXITY_SCALE_OPTIMIZATION,
    domain_knowledge_scale=DOMAIN_KNOWLEDGE_SCALE_OPTIMIZATION,
    logical_reasoning_scale=LOGICAL_REASONING_SCALE,
    atypicality_scale=ATYPICALITY_SCALE,
    verification_difficulty_scale=VERIFICATION_DIFFICULTY_SCALE,
    standard_pattern_scale=STANDARD_PATTERN_SCALE,
    integration_complexity_scale=INTEGRATION_COMPLEXITY_SCALE,
    output_format=OUTPUT_FORMAT_9_FEATURES_CODE,
)


def format_gso_unified_prompt(task: Dict[str, Any]) -> str:
    """Format the GSO unified prompt with task-specific information.

    Args:
        task: GSO task dict with keys:
            - instance_id: GSO task ID
            - repo: Repository name (e.g., "numpy/numpy")
            - api: API/function being optimized
            - prob_script: Test script showing performance scenario
            - gt_diff: Gold optimization patch
            - hints_text: Optional hints (may be empty)

    Returns:
        Formatted prompt string
    """
    hints_text = task.get("hints_text", "") or ""
    hints_section = ""
    if hints_text and hints_text.strip():
        hints_section = f"**Hints:**\n{hints_text}"

    # No truncation needed - Claude Opus 4.5 has 200K token context
    prob_script = task.get("prob_script", "") or ""
    gt_diff = task.get("gt_diff", "") or ""

    return GSO_UNIFIED_PROMPT_TEMPLATE.format(
        instance_id=task.get("instance_id", ""),
        repo=task.get("repo", ""),
        api=task.get("api", "unknown"),
        prob_script=prob_script,
        gt_diff=gt_diff,
        hints_section=hints_section,
    )


# The main configuration object
GSO_UNIFIED_CONFIG = PromptConfig(
    name="gso_unified",
    features=CODE_DATASET_FEATURES,
    prompt_template=GSO_UNIFIED_PROMPT_TEMPLATE,
    task_id_field="instance_id",
    truncation_limits={},  # No truncation needed with Claude Opus 4.5's 200K context
    format_prompt_fn=format_gso_unified_prompt,
)


# =============================================================================
# No-Solution Variant (for ablation study)
# =============================================================================

GSO_UNIFIED_NO_SOLUTION_PROMPT_TEMPLATE = """You are analyzing a GSO (Software Optimization Benchmark) task to predict its difficulty.
This is a PERFORMANCE OPTIMIZATION task, NOT a bug fix. You do not have access to the optimization solution.

{completeness_instruction}

## TASK INFORMATION

**Instance ID:** {{instance_id}}
**Repository:** {{repo}}
**API/Function being optimized:** {{api}}

**Test Script (performance scenario to optimize):**
```python
{{prob_script}}
```

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
    domain_knowledge_scale=DOMAIN_KNOWLEDGE_SCALE_OPTIMIZATION,
    logical_reasoning_scale=LOGICAL_REASONING_SCALE,
    atypicality_scale=ATYPICALITY_SCALE,
    verification_difficulty_scale=VERIFICATION_DIFFICULTY_SCALE,
    standard_pattern_scale=STANDARD_PATTERN_SCALE,
    output_format=OUTPUT_FORMAT_7_FEATURES,
)


def format_gso_unified_no_solution_prompt(task: Dict[str, Any]) -> str:
    """Format the prompt without the optimization patch."""
    hints_text = task.get("hints_text", "") or ""
    hints_section = f"**Hints:**\n{hints_text}" if hints_text and hints_text.strip() else ""

    return GSO_UNIFIED_NO_SOLUTION_PROMPT_TEMPLATE.format(
        instance_id=task.get("instance_id", ""),
        repo=task.get("repo", ""),
        api=task.get("api", "unknown"),
        prob_script=task.get("prob_script", "") or "",
        hints_section=hints_section,
    )


GSO_UNIFIED_NO_SOLUTION_CONFIG = PromptConfig(
    name="gso_unified_no_solution",
    features=NO_SOLUTION_FEATURES,
    prompt_template=GSO_UNIFIED_NO_SOLUTION_PROMPT_TEMPLATE,
    task_id_field="instance_id",
    truncation_limits={},
    format_prompt_fn=format_gso_unified_no_solution_prompt,
)


# =============================================================================
# Problem-Only Variant (for ablation study)
# =============================================================================

GSO_UNIFIED_PROBLEM_ONLY_PROMPT_TEMPLATE = """You are analyzing a GSO (Software Optimization Benchmark) task to predict its difficulty.
This is a PERFORMANCE OPTIMIZATION task. You only have access to the test script.

{completeness_instruction}

## TASK INFORMATION

**Instance ID:** {{instance_id}}

**Test Script (performance scenario to optimize):**
```python
{{prob_script}}
```

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
    domain_knowledge_scale=DOMAIN_KNOWLEDGE_SCALE_OPTIMIZATION,
    logical_reasoning_scale=LOGICAL_REASONING_SCALE,
    atypicality_scale=ATYPICALITY_SCALE,
    verification_difficulty_scale=VERIFICATION_DIFFICULTY_SCALE,
    standard_pattern_scale=STANDARD_PATTERN_SCALE,
    output_format=OUTPUT_FORMAT_7_FEATURES,
)


def format_gso_unified_problem_only_prompt(task: Dict[str, Any]) -> str:
    """Format the prompt with only test script."""
    return GSO_UNIFIED_PROBLEM_ONLY_PROMPT_TEMPLATE.format(
        instance_id=task.get("instance_id", ""),
        prob_script=task.get("prob_script", "") or "",
    )


GSO_UNIFIED_PROBLEM_ONLY_CONFIG = PromptConfig(
    name="gso_unified_problem_only",
    features=NO_SOLUTION_FEATURES,
    prompt_template=GSO_UNIFIED_PROBLEM_ONLY_PROMPT_TEMPLATE,
    task_id_field="instance_id",
    truncation_limits={},
    format_prompt_fn=format_gso_unified_problem_only_prompt,
)

"""TerminalBench unified prompt configuration.

This module defines the unified prompt for TerminalBench tasks using
standardized features that enable fair comparison across datasets.

Features (9 total):
- 8 core features (shared with all datasets)
- 1 dataset-specific: tooling_complexity (terminal environment complexity)

Task type: Terminal/shell tasks requiring command-line solutions
"""

from typing import Any, Dict, List

from experiment_ab_shared.llm_judge.prompt_config import PromptConfig
from experiment_ab_shared.llm_judge.prompts.unified_features import (
    TERMINAL_DATASET_FEATURES,
    NO_SOLUTION_FEATURES,
    COMPLETENESS_INSTRUCTION,
    OUTPUT_FORMAT_9_FEATURES_TERMINAL,
    OUTPUT_FORMAT_7_FEATURES,
    SOLUTION_HINT_SCALE,
    PROBLEM_CLARITY_SCALE,
    SOLUTION_COMPLEXITY_SCALE_TERMINAL,
    DOMAIN_KNOWLEDGE_SCALE_TERMINAL,
    LOGICAL_REASONING_SCALE,
    ATYPICALITY_SCALE,
    VERIFICATION_DIFFICULTY_SCALE,
    STANDARD_PATTERN_SCALE,
    TOOLING_COMPLEXITY_SCALE,
)


TERMINALBENCH_UNIFIED_PROMPT_TEMPLATE = """You are analyzing a TerminalBench terminal/shell task to predict its difficulty.
This task requires writing shell commands or scripts to accomplish a goal. You will analyze
the task instruction and reference solution to evaluate semantic features.

{completeness_instruction}

## TASK INFORMATION

**Task ID:** {{task_id}}
**Category:** {{category}}
**Tags:** {{tags}}
**Claimed Difficulty:** {{claimed_difficulty}}

**Task Instruction:**
{{instruction}}

**Reference Solution (solution.sh):**
```bash
{{solution}}
```

## FEATURES TO EVALUATE

Analyze the instruction and solution to evaluate these 9 features.
Focus on what makes the SOLUTION hard, not just what the TASK looks like.
Be precise and consistent with your ratings.

{solution_hint_scale}

{problem_clarity_scale}

{solution_complexity_scale}

{domain_knowledge_scale}

{logical_reasoning_scale}

{atypicality_scale}

{verification_difficulty_scale}

{standard_pattern_scale}

{tooling_complexity_scale}

{output_format}
""".format(
    completeness_instruction=COMPLETENESS_INSTRUCTION,
    solution_hint_scale=SOLUTION_HINT_SCALE,
    problem_clarity_scale=PROBLEM_CLARITY_SCALE,
    solution_complexity_scale=SOLUTION_COMPLEXITY_SCALE_TERMINAL,
    domain_knowledge_scale=DOMAIN_KNOWLEDGE_SCALE_TERMINAL,
    logical_reasoning_scale=LOGICAL_REASONING_SCALE,
    atypicality_scale=ATYPICALITY_SCALE,
    verification_difficulty_scale=VERIFICATION_DIFFICULTY_SCALE,
    standard_pattern_scale=STANDARD_PATTERN_SCALE,
    tooling_complexity_scale=TOOLING_COMPLEXITY_SCALE,
    output_format=OUTPUT_FORMAT_9_FEATURES_TERMINAL,
)


def format_terminalbench_unified_prompt(task: Dict[str, Any]) -> str:
    """Format the TerminalBench unified prompt with task-specific information.

    Args:
        task: TerminalBench task dict with keys:
            - task_id: TerminalBench task ID (e.g., "3d-model-format-legacy")
            - instruction: The task instruction from task.yaml
            - solution: The reference solution from solution.sh
            - category: Task category (e.g., "software-engineering")
            - tags: List of tags (e.g., ["coding", "file-operations"])
            - claimed_difficulty: Self-reported difficulty (e.g., "hard")

    Returns:
        Formatted prompt string
    """
    tags: List[str] = task.get("tags") or []

    # No truncation needed - Claude Opus 4.5 has 200K token context
    instruction = task.get("instruction", "")
    solution = task.get("solution", "")

    return TERMINALBENCH_UNIFIED_PROMPT_TEMPLATE.format(
        task_id=task.get("task_id", ""),
        category=task.get("category") or "N/A",
        tags=", ".join(tags) if tags else "N/A",
        claimed_difficulty=task.get("claimed_difficulty") or "N/A",
        instruction=instruction,
        solution=solution,
    )


# The main configuration object
TERMINALBENCH_UNIFIED_CONFIG = PromptConfig(
    name="terminalbench_unified",
    features=TERMINAL_DATASET_FEATURES,
    prompt_template=TERMINALBENCH_UNIFIED_PROMPT_TEMPLATE,
    task_id_field="task_id",
    truncation_limits={},  # No truncation needed with Claude Opus 4.5's 200K context
    format_prompt_fn=format_terminalbench_unified_prompt,
)


# =============================================================================
# No-Solution Variant (for ablation study)
# =============================================================================

TERMINALBENCH_UNIFIED_NO_SOLUTION_PROMPT_TEMPLATE = """You are analyzing a TerminalBench terminal/shell task to predict its difficulty.
This task requires writing shell commands or scripts to accomplish a goal. You do not have access to the reference solution.

{completeness_instruction}

## TASK INFORMATION

**Task ID:** {{task_id}}
**Category:** {{category}}
**Tags:** {{tags}}
**Claimed Difficulty:** {{claimed_difficulty}}

**Task Instruction:**
{{instruction}}

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
    domain_knowledge_scale=DOMAIN_KNOWLEDGE_SCALE_TERMINAL,
    logical_reasoning_scale=LOGICAL_REASONING_SCALE,
    atypicality_scale=ATYPICALITY_SCALE,
    verification_difficulty_scale=VERIFICATION_DIFFICULTY_SCALE,
    standard_pattern_scale=STANDARD_PATTERN_SCALE,
    output_format=OUTPUT_FORMAT_7_FEATURES,
)


def format_terminalbench_unified_no_solution_prompt(task: Dict[str, Any]) -> str:
    """Format the prompt without the reference solution."""
    tags: List[str] = task.get("tags") or []

    return TERMINALBENCH_UNIFIED_NO_SOLUTION_PROMPT_TEMPLATE.format(
        task_id=task.get("task_id", ""),
        category=task.get("category") or "N/A",
        tags=", ".join(tags) if tags else "N/A",
        claimed_difficulty=task.get("claimed_difficulty") or "N/A",
        instruction=task.get("instruction", ""),
    )


TERMINALBENCH_UNIFIED_NO_SOLUTION_CONFIG = PromptConfig(
    name="terminalbench_unified_no_solution",
    features=NO_SOLUTION_FEATURES,
    prompt_template=TERMINALBENCH_UNIFIED_NO_SOLUTION_PROMPT_TEMPLATE,
    task_id_field="task_id",
    truncation_limits={},
    format_prompt_fn=format_terminalbench_unified_no_solution_prompt,
)


# =============================================================================
# Problem-Only Variant (for ablation study)
# =============================================================================

TERMINALBENCH_UNIFIED_PROBLEM_ONLY_PROMPT_TEMPLATE = """You are analyzing a TerminalBench terminal/shell task to predict its difficulty.
This task requires writing shell commands or scripts. You only have access to the task instruction.

{completeness_instruction}

## TASK INFORMATION

**Task ID:** {{task_id}}

**Task Instruction:**
{{instruction}}

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
    domain_knowledge_scale=DOMAIN_KNOWLEDGE_SCALE_TERMINAL,
    logical_reasoning_scale=LOGICAL_REASONING_SCALE,
    atypicality_scale=ATYPICALITY_SCALE,
    verification_difficulty_scale=VERIFICATION_DIFFICULTY_SCALE,
    standard_pattern_scale=STANDARD_PATTERN_SCALE,
    output_format=OUTPUT_FORMAT_7_FEATURES,
)


def format_terminalbench_unified_problem_only_prompt(task: Dict[str, Any]) -> str:
    """Format the prompt with only task instruction."""
    return TERMINALBENCH_UNIFIED_PROBLEM_ONLY_PROMPT_TEMPLATE.format(
        task_id=task.get("task_id", ""),
        instruction=task.get("instruction", ""),
    )


TERMINALBENCH_UNIFIED_PROBLEM_ONLY_CONFIG = PromptConfig(
    name="terminalbench_unified_problem_only",
    features=NO_SOLUTION_FEATURES,
    prompt_template=TERMINALBENCH_UNIFIED_PROBLEM_ONLY_PROMPT_TEMPLATE,
    task_id_field="task_id",
    truncation_limits={},
    format_prompt_fn=format_terminalbench_unified_problem_only_prompt,
)

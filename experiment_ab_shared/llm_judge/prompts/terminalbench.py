"""TerminalBench prompt configuration for LLM judge feature extraction.

This module defines the prompt template and feature definitions for extracting
semantic features from TerminalBench terminal/shell tasks.
"""

from typing import Any, Dict, List

from experiment_ab_shared.llm_judge.prompt_config import FeatureDefinition, PromptConfig

# Feature definitions for TerminalBench
TERMINALBENCH_FEATURES = [
    FeatureDefinition(
        name="solution_in_instruction",
        min_value=0,
        max_value=3,
        description="Does the instruction contain or hint at the solution? (0=none, 3=exact commands)",
    ),
    FeatureDefinition(
        name="task_clarity",
        min_value=1,
        max_value=5,
        description="How clear and well-specified is the task? (1=vague, 5=crystal clear)",
    ),
    FeatureDefinition(
        name="solution_size",
        min_value=1,
        max_value=3,
        description="How large is the reference solution? (1=few commands, 3=complex script)",
    ),
    FeatureDefinition(
        name="domain_knowledge_required",
        min_value=1,
        max_value=5,
        description="How much specialized knowledge is needed? (1=basic shell, 5=obscure tools)",
    ),
    FeatureDefinition(
        name="task_complexity",
        min_value=1,
        max_value=5,
        description="How complex is the task? (1=trivial, 5=very complex)",
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
        description="How unusual is this task type? (1=very common, 5=rare/novel)",
    ),
    FeatureDefinition(
        name="tooling_complexity",
        min_value=1,
        max_value=5,
        description="How complex is the tooling/environment? (1=basic shell, 5=exotic toolchain)",
    ),
]

# The prompt template for TerminalBench tasks
TERMINALBENCH_PROMPT_TEMPLATE = """You are analyzing a TerminalBench terminal/shell task to predict its difficulty.
You will analyze the task instruction and reference solution to evaluate semantic features.

## TASK INFORMATION

**Task ID:** {task_id}
**Category:** {category}
**Tags:** {tags}
**Claimed Difficulty:** {claimed_difficulty}

**Task Instruction:**
{instruction}

**Reference Solution (solution.sh):**
```bash
{solution}
```

## FEATURES TO EVALUATE

Analyze the instruction and solution to evaluate these 8 semantic features.
Be precise and consistent with your ratings.

### 1. Solution Hints in Instruction (solution_in_instruction: 0-3)
Does the instruction contain or hint at the solution approach?
- 0: No hint at the solution at all
- 1: Vague hint or general direction
- 2: Clear description of approach needed
- 3: Exact commands or detailed solution steps provided

### 2. Task Clarity (task_clarity: 1-5)
How clear and well-specified is the task?
- 1: Very vague, unclear what's actually required
- 2: Somewhat clear but missing key details
- 3: Reasonably clear, some ambiguity about requirements
- 4: Clear with good context and success criteria
- 5: Crystal clear with explicit steps and expected outputs

### 3. Solution Size (solution_size: 1-3)
How large/complex is the reference solution script?
- 1: Simple, few commands (1-10 lines)
- 2: Moderate complexity (11-50 lines)
- 3: Large/complex script (>50 lines or multiple components)

### 4. Domain Knowledge Required (domain_knowledge_required: 1-5)
How much specialized knowledge is needed?
- 1: Basic shell commands anyone could use (ls, cd, cat, echo)
- 2: Standard Unix tools (grep, sed, awk, find)
- 3: Specialized tools or configurations (cmake, git internals, network tools)
- 4: Deep understanding of systems (kernel, filesystems, protocols)
- 5: Obscure tools, APIs, or highly specialized domain knowledge

### 5. Task Complexity (task_complexity: 1-5)
How complex is the actual task to complete?
- 1: Trivial (single command, simple file operation)
- 2: Simple (straightforward multi-step task)
- 3: Moderate (requires understanding context, multiple tools)
- 4: Complex (multiple interdependent steps, debugging needed)
- 5: Very complex (architectural changes, cross-system integration)

### 6. Logical Reasoning Required (logical_reasoning_required: 1-5)
How much logical reasoning is needed?
- 1: Mechanical execution, no reasoning needed
- 2: Simple cause-effect reasoning
- 3: Multi-step reasoning required
- 4: Complex reasoning with multiple factors
- 5: Deep reasoning about system behavior, edge cases

### 7. Atypicality (atypicality: 1-5)
How unusual is this type of terminal task?
- 1: Very common task (file manipulation, basic scripting)
- 2: Common task (process management, system configuration)
- 3: Moderately unusual
- 4: Unusual task pattern
- 5: Rare or novel task

### 8. Tooling Complexity (tooling_complexity: 1-5)
How complex is the tooling/environment setup?
- 1: No special tools needed (basic shell)
- 2: Standard development tools (git, make, pip)
- 3: Multiple specialized tools or complex configuration
- 4: Uncommon tools or complex build systems
- 5: Exotic toolchain, legacy systems, or cross-compilation

## OUTPUT FORMAT

Respond with ONLY a JSON object containing all features. No markdown, no extra text.

{{
    "solution_in_instruction": <0-3>,
    "task_clarity": <1-5>,
    "solution_size": <1-3>,
    "domain_knowledge_required": <1-5>,
    "task_complexity": <1-5>,
    "logical_reasoning_required": <1-5>,
    "atypicality": <1-5>,
    "tooling_complexity": <1-5>,
    "reasoning": "<2-3 sentence summary of the key difficulty factors>"
}}
"""


def format_terminalbench_prompt(task: Dict[str, Any]) -> str:
    """Format the TerminalBench prompt with task-specific information.

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

    # Truncate very long fields to avoid context overflow
    instruction = task.get("instruction", "")
    if len(instruction) > 12000:
        instruction = instruction[:12000]

    solution = task.get("solution", "")
    if len(solution) > 12000:
        solution = solution[:12000]

    return TERMINALBENCH_PROMPT_TEMPLATE.format(
        task_id=task.get("task_id", ""),
        category=task.get("category") or "N/A",
        tags=", ".join(tags) if tags else "N/A",
        claimed_difficulty=task.get("claimed_difficulty") or "N/A",
        instruction=instruction,
        solution=solution,
    )


# The main configuration object
TERMINALBENCH_CONFIG = PromptConfig(
    name="terminalbench",
    features=TERMINALBENCH_FEATURES,
    prompt_template=TERMINALBENCH_PROMPT_TEMPLATE,
    task_id_field="task_id",
    truncation_limits={
        "instruction": 12000,
        "solution": 12000,
    },
    format_prompt_fn=format_terminalbench_prompt,
)

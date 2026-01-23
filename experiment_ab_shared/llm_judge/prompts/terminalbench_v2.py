"""TerminalBench V2 prompt configuration for LLM judge feature extraction.

This module contains only statistically significant features (p<0.05) based on
pilot analysis correlating with oracle IRT task difficulties.

Significant LLM features (7):
- atypicality (r=0.423***)
- domain_knowledge_required (r=0.403***)
- verification_difficulty (r=0.377***) - NEW from V5
- task_complexity (r=0.339**)
- logical_reasoning_required (r=0.333**)
- standard_pattern_available (r=-0.281**) - NEW from V5
- task_clarity (r=-0.246*)

Dropped (not significant in pilot):
- solution_in_instruction (r=-0.125, p=0.24)
- integration_complexity (r=-0.006, p=0.96) - doesn't apply to terminal tasks

Significant deterministic features (added via --add-deterministic):
- log_lines (r=0.313**)
- num_lines (r=0.296**)
- num_pipes (r=0.272*)

Total: 7 LLM features + 3 deterministic = 10 features
"""

from typing import Any, Dict, List

from experiment_ab_shared.llm_judge.prompt_config import FeatureDefinition, PromptConfig

TERMINALBENCH_V2_FEATURES = [
    # === Significant existing features ===
    FeatureDefinition(
        name="task_clarity",
        min_value=1,
        max_value=5,
        description="How clear and well-specified is the task? (1=vague, 5=crystal clear)",
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
    # === Significant V5-inspired features ===
    FeatureDefinition(
        name="verification_difficulty",
        min_value=1,
        max_value=5,
        description="How hard to verify the solution works? (1=trivial, 5=very hard)",
    ),
    FeatureDefinition(
        name="standard_pattern_available",
        min_value=0,
        max_value=1,
        description="Is this a well-documented shell pattern with examples? (0=no, 1=yes)",
    ),
]

TERMINALBENCH_V2_PROMPT_TEMPLATE = """You are analyzing a TerminalBench terminal/shell task to predict its difficulty.
Analyze the task instruction and reference solution to evaluate semantic features.

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

Analyze the instruction and solution to evaluate these 7 features.
Focus on what makes the SOLUTION hard, not what the TASK description looks like.

### 1. Task Clarity (task_clarity: 1-5)
How clear and well-specified is the task?
- 1: Very vague, unclear what's actually required
- 2: Somewhat clear but missing key details
- 3: Reasonably clear, some ambiguity about requirements
- 4: Clear with good context and success criteria
- 5: Crystal clear with explicit steps and expected outputs

### 2. Domain Knowledge Required (domain_knowledge_required: 1-5)
How much specialized knowledge is needed?
- 1: Basic shell commands anyone could use (ls, cd, cat, echo)
- 2: Standard Unix tools (grep, sed, awk, find)
- 3: Specialized tools or configurations (cmake, git internals, network tools)
- 4: Deep understanding of systems (kernel, filesystems, protocols)
- 5: Obscure tools, APIs, or highly specialized domain knowledge

### 3. Task Complexity (task_complexity: 1-5)
How complex is the actual task to complete?
- 1: Trivial (single command, simple file operation)
- 2: Simple (straightforward multi-step task)
- 3: Moderate (requires understanding context, multiple tools)
- 4: Complex (multiple interdependent steps, debugging needed)
- 5: Very complex (architectural changes, cross-system integration)

### 4. Logical Reasoning Required (logical_reasoning_required: 1-5)
How much logical reasoning is needed?
- 1: Mechanical execution, no reasoning needed
- 2: Simple cause-effect reasoning
- 3: Multi-step reasoning required
- 4: Complex reasoning with multiple factors
- 5: Deep reasoning about system behavior, edge cases

### 5. Atypicality (atypicality: 1-5)
How unusual is this type of terminal task?
- 1: Very common task (file manipulation, basic scripting)
- 2: Common task (process management, system configuration)
- 3: Moderately unusual
- 4: Unusual task pattern
- 5: Rare or novel task

### 6. Verification Difficulty (verification_difficulty: 1-5)
How hard is it to verify/test the solution works correctly?
- 1: Trivial (obvious pass/fail, single output to check)
- 2: Easy (straightforward test cases, predictable output)
- 3: Moderate (some edge cases, state changes to verify)
- 4: Hard (subtle correctness, complex state, timing issues)
- 5: Very hard (rare conditions, hard to reproduce, side effects)

### 7. Standard Pattern Available (standard_pattern_available: 0/1)
Is this a well-documented shell pattern with existing examples?
- 0: Novel solution needed, no clear pattern or StackOverflow answer
- 1: Well-documented pattern (e.g., "find + xargs", "grep + awk pipeline", "process substitution")

## OUTPUT FORMAT

Respond with ONLY a JSON object. No markdown, no extra text.

{{
    "task_clarity": <1-5>,
    "domain_knowledge_required": <1-5>,
    "task_complexity": <1-5>,
    "logical_reasoning_required": <1-5>,
    "atypicality": <1-5>,
    "verification_difficulty": <1-5>,
    "standard_pattern_available": <0 or 1>,
    "reasoning": "<2-3 sentences on what makes the SOLUTION hard or easy>"
}}
"""


def format_terminalbench_v2_prompt(task: Dict[str, Any]) -> str:
    """Format the TerminalBench V2 prompt with task-specific information."""
    tags: List[str] = task.get("tags") or []

    # Truncate very long fields to avoid context overflow
    instruction = task.get("instruction", "")
    if len(instruction) > 12000:
        instruction = instruction[:12000]

    solution = task.get("solution", "")
    if len(solution) > 12000:
        solution = solution[:12000]

    return TERMINALBENCH_V2_PROMPT_TEMPLATE.format(
        task_id=task.get("task_id", ""),
        category=task.get("category") or "N/A",
        tags=", ".join(tags) if tags else "N/A",
        claimed_difficulty=task.get("claimed_difficulty") or "N/A",
        instruction=instruction,
        solution=solution,
    )


TERMINALBENCH_V2_CONFIG = PromptConfig(
    name="terminalbench_v2",
    features=TERMINALBENCH_V2_FEATURES,
    prompt_template=TERMINALBENCH_V2_PROMPT_TEMPLATE,
    task_id_field="task_id",
    truncation_limits={
        "instruction": 12000,
        "solution": 12000,
    },
    format_prompt_fn=format_terminalbench_v2_prompt,
)

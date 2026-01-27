"""SWE-bench V5 prompt configuration for LLM judge feature extraction.

This module provides NEW semantic features focusing on:
1. Error signal quality (different from problem_clarity)
2. External specification requirements (protocols, standards, RFCs)
3. Task type classification (bug fix vs enhancement)
4. Reproduction complexity
5. Code locality hints (does problem statement indicate where to look?)
6. Backwards compatibility concerns

V5 features (6 new):
- error_signal_clarity: How clear are error messages/stack traces?
- requires_external_spec: Does fix require knowledge of external protocols/standards?
- task_type: Bug fix vs enhancement/feature request
- reproduction_complexity: How complex is reproducing the issue?
- location_hints_provided: Does problem statement indicate where to look?
- backwards_compat_concern: Are there backwards compatibility concerns?
"""

from typing import Any, Dict

from experiment_ab_shared.llm_judge.prompt_config import FeatureDefinition, PromptConfig

SWEBENCH_V5_FEATURES = [
    FeatureDefinition(
        name="error_signal_clarity",
        min_value=1,
        max_value=5,
        description="How clear are the error signals (messages, stack traces, failures)? (1=no errors shown, 5=exact error with line numbers)",
    ),
    FeatureDefinition(
        name="requires_external_spec",
        min_value=0,
        max_value=1,
        description="Does fixing this require knowledge of external specifications (RFC, protocols, file formats, standards)? (0=no, 1=yes)",
    ),
    FeatureDefinition(
        name="task_type",
        min_value=0,
        max_value=2,
        description="What type of task is this? (0=bug fix, 1=enhancement/feature, 2=refactoring/cleanup)",
    ),
    FeatureDefinition(
        name="reproduction_complexity",
        min_value=1,
        max_value=5,
        description="How complex is reproducing this issue? (1=trivial/one-liner, 5=complex setup needed)",
    ),
    FeatureDefinition(
        name="location_hints_provided",
        min_value=0,
        max_value=1,
        description="Does the problem statement indicate WHERE to look in the code? (0=no hints, 1=file/function/line mentioned)",
    ),
    FeatureDefinition(
        name="backwards_compat_concern",
        min_value=0,
        max_value=1,
        description="Does the fix need to maintain backwards compatibility? (0=no concern, 1=must maintain compat)",
    ),
]

SWEBENCH_V5_PROMPT_TEMPLATE = """You are analyzing a SWE-bench coding task to predict its difficulty.
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
Focus on aspects that help predict task difficulty.

### 1. Error Signal Clarity (error_signal_clarity: 1-5)
How clear are the error signals (error messages, stack traces, test failures)?
- 1: No error shown, vague symptom description
- 2: General error message without details
- 3: Error message with some context
- 4: Clear error with relevant stack trace
- 5: Exact error with specific line numbers/locations

### 2. Requires External Specification (requires_external_spec: 0/1)
Does fixing this require knowledge of external specifications?
- 0: No external specs needed (pure Python, framework-internal)
- 1: Yes - requires RFC, protocol spec, file format spec, standard compliance, or external API docs

### 3. Task Type (task_type: 0-2)
What type of task is this based on the problem statement?
- 0: Bug fix - something is broken and needs to be fixed
- 1: Enhancement/feature - adding new capability or behavior
- 2: Refactoring/cleanup - improving code without changing behavior

### 4. Reproduction Complexity (reproduction_complexity: 1-5)
How complex is reproducing this issue based on the problem statement?
- 1: Trivial - one-liner or simple function call
- 2: Simple - few lines of setup code
- 3: Moderate - needs specific data or configuration
- 4: Complex - multiple components, specific state needed
- 5: Very complex - requires special environment, timing, or rare conditions

### 5. Location Hints Provided (location_hints_provided: 0/1)
Does the problem statement indicate WHERE to look in the code?
- 0: No location hints - must search for relevant code
- 1: Location hints provided - mentions file, function, class, or line numbers

### 6. Backwards Compatibility Concern (backwards_compat_concern: 0/1)
Does the fix need to maintain backwards compatibility?
- 0: No concern - new code, internal change, or allowed to break compat
- 1: Must maintain compat - public API, existing behavior depended on

## OUTPUT FORMAT

Respond with ONLY a JSON object. No markdown, no extra text.

{{
    "error_signal_clarity": <1-5>,
    "requires_external_spec": <0 or 1>,
    "task_type": <0-2>,
    "reproduction_complexity": <1-5>,
    "location_hints_provided": <0 or 1>,
    "backwards_compat_concern": <0 or 1>,
    "reasoning": "<2-3 sentences on what makes this task hard or easy>"
}}
"""


def format_swebench_v5_prompt(task: Dict[str, Any]) -> str:
    """Format the SWE-bench V5 prompt with task-specific information."""
    hints_text = task.get("hints_text", "")
    hints_section = f"**Hints:**\n{hints_text}" if hints_text and hints_text.strip() else ""

    return SWEBENCH_V5_PROMPT_TEMPLATE.format(
        instance_id=task.get("instance_id", ""),
        repo=task.get("repo", ""),
        version=task.get("version", "unknown"),
        problem_statement=task.get("problem_statement", ""),
        patch=task.get("patch", ""),
        fail_to_pass=task.get("FAIL_TO_PASS", "[]"),
        hints_section=hints_section,
    )


SWEBENCH_V5_CONFIG = PromptConfig(
    name="swebench_v5",
    features=SWEBENCH_V5_FEATURES,
    prompt_template=SWEBENCH_V5_PROMPT_TEMPLATE,
    task_id_field="instance_id",
    format_prompt_fn=format_swebench_v5_prompt,
)

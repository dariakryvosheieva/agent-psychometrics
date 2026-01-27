"""SWE-bench V6 prompt configuration for LLM judge feature extraction.

This module provides NEW semantic features focusing on:
1. Whether the solution is explicitly prescribed in the problem
2. Number of components that must interact for the fix
3. Scope of codebase understanding needed for debugging
4. Whether this is a semantic issue vs a crash/exception
5. Whether the fix changes public API surface
6. Whether the problem provides a minimal reproduction

V6 features (6 new):
- solution_prescribed: Does problem explicitly prescribe the solution?
- component_interaction: How many components need to interact?
- debugging_scope: How much of codebase must be understood?
- semantic_vs_crash: Is this semantic (0) or crash/exception (1)?
- api_surface_change: Does fix change public API?
- minimal_repro_provided: Is there a minimal reproduction case?
"""

from typing import Any, Dict

from experiment_ab_shared.llm_judge.prompt_config import FeatureDefinition, PromptConfig

SWEBENCH_V6_FEATURES = [
    FeatureDefinition(
        name="solution_prescribed",
        min_value=0,
        max_value=1,
        description="Does the problem statement explicitly prescribe the solution? (0=problem only, 1=solution proposed)",
    ),
    FeatureDefinition(
        name="component_interaction",
        min_value=1,
        max_value=3,
        description="How many components/subsystems must interact for the fix? (1=single, 2=two, 3=three+)",
    ),
    FeatureDefinition(
        name="debugging_scope",
        min_value=1,
        max_value=5,
        description="How much of the codebase must be understood to debug? (1=single function, 5=entire subsystem)",
    ),
    FeatureDefinition(
        name="semantic_vs_crash",
        min_value=0,
        max_value=1,
        description="Is this a semantic issue or a crash/exception? (0=semantic/behavioral, 1=crash/exception)",
    ),
    FeatureDefinition(
        name="api_surface_change",
        min_value=0,
        max_value=1,
        description="Does the fix change the public API surface? (0=internal only, 1=public API change)",
    ),
    FeatureDefinition(
        name="minimal_repro_provided",
        min_value=0,
        max_value=1,
        description="Is a minimal reproduction case provided in the problem? (0=no repro, 1=repro code provided)",
    ),
]

SWEBENCH_V6_PROMPT_TEMPLATE = """You are analyzing a SWE-bench coding task to predict its difficulty.
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

### 1. Solution Prescribed (solution_prescribed: 0/1)
Does the problem statement explicitly prescribe the solution?
- 0: Problem only - describes what's wrong but not how to fix it
- 1: Solution proposed - suggests specific code changes, API additions, or fixes

Examples of 0: "X doesn't work", "Bug in Y causes Z"
Examples of 1: "Add method X to class Y", "Change line Z from A to B"

### 2. Component Interaction (component_interaction: 1-3)
How many components/subsystems must interact for the fix?
- 1: Single component - fix is self-contained in one class/module
- 2: Two components - fix requires understanding two interacting parts
- 3: Three or more - fix spans multiple subsystems

### 3. Debugging Scope (debugging_scope: 1-5)
How much of the codebase must be understood to debug the root cause?
- 1: Single function - issue is obvious from the function alone
- 2: Single class - must understand class internals
- 3: Single module - must understand module structure
- 4: Multiple modules - must trace across module boundaries
- 5: Entire subsystem - requires deep understanding of architecture

### 4. Semantic vs Crash (semantic_vs_crash: 0/1)
Is this a semantic issue or a crash/exception?
- 0: Semantic/behavioral - wrong output, incorrect behavior, logic bug
- 1: Crash/exception - TypeError, ValueError, AttributeError, etc.

### 5. API Surface Change (api_surface_change: 0/1)
Does the fix change the public API surface?
- 0: Internal only - changes implementation without affecting public interface
- 1: Public API change - adds/modifies/removes public methods, parameters, or behavior

### 6. Minimal Repro Provided (minimal_repro_provided: 0/1)
Is a minimal reproduction case provided in the problem statement?
- 0: No repro - just description of the issue
- 1: Repro provided - code example that demonstrates the bug

## OUTPUT FORMAT

Respond with ONLY a JSON object. No markdown, no extra text.

{{
    "solution_prescribed": <0 or 1>,
    "component_interaction": <1-3>,
    "debugging_scope": <1-5>,
    "semantic_vs_crash": <0 or 1>,
    "api_surface_change": <0 or 1>,
    "minimal_repro_provided": <0 or 1>,
    "reasoning": "<2-3 sentences on what makes this task hard or easy>"
}}
"""


def format_swebench_v6_prompt(task: Dict[str, Any]) -> str:
    """Format the SWE-bench V6 prompt with task-specific information."""
    hints_text = task.get("hints_text", "")
    hints_section = f"**Hints:**\n{hints_text}" if hints_text and hints_text.strip() else ""

    return SWEBENCH_V6_PROMPT_TEMPLATE.format(
        instance_id=task.get("instance_id", ""),
        repo=task.get("repo", ""),
        version=task.get("version", "unknown"),
        problem_statement=task.get("problem_statement", ""),
        patch=task.get("patch", ""),
        fail_to_pass=task.get("FAIL_TO_PASS", "[]"),
        hints_section=hints_section,
    )


SWEBENCH_V6_CONFIG = PromptConfig(
    name="swebench_v6",
    features=SWEBENCH_V6_FEATURES,
    prompt_template=SWEBENCH_V6_PROMPT_TEMPLATE,
    task_id_field="instance_id",
    format_prompt_fn=format_swebench_v6_prompt,
)

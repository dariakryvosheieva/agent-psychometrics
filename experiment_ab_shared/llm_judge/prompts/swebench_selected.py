"""SWE-bench Selected Features prompt configuration for LLM judge feature extraction.

This module provides the 6 best-performing NEW features discovered through
iterative feature development (V3-V6). These features complement the existing
unified features with ~+0.05 R^2 incremental improvement.

Selected features (6 total):
From V3:
- domain_category (0-9): Primary domain classification

From V4:
- solution_discovery_needed (1-5): Does solving require exploration/discovery?
- fix_pattern_type (0-4): What type of fix pattern is needed?

From V5:
- location_hints_provided (0/1): Does problem statement hint at code location?

From V6:
- solution_prescribed (0/1): Is the solution explicitly described in problem?
- minimal_repro_provided (0/1): Is a reproduction case provided?
"""

from typing import Any, Dict

from experiment_ab_shared.llm_judge.prompt_config import FeatureDefinition, PromptConfig

SWEBENCH_SELECTED_FEATURES = [
    # V3: Domain category
    FeatureDefinition(
        name="domain_category",
        min_value=0,
        max_value=9,
        description="Primary domain/subsystem (0=infra/config, 1=auth, 2=database, 3=api, 4=ui, 5=data processing, 6=file/io, 7=networking, 8=testing, 9=other)",
    ),
    # V4: Solution discovery and fix pattern
    FeatureDefinition(
        name="solution_discovery_needed",
        min_value=1,
        max_value=5,
        description="Does solving require exploration/discovery? (1=solution obvious, 5=requires extensive exploration)",
    ),
    FeatureDefinition(
        name="fix_pattern_type",
        min_value=0,
        max_value=4,
        description="Fix pattern type (0=add missing check, 1=fix logic error, 2=add new method/function, 3=refactor existing, 4=architectural change)",
    ),
    # V5: Location hints
    FeatureDefinition(
        name="location_hints_provided",
        min_value=0,
        max_value=1,
        description="Does problem statement hint at code location? (0=no hints, 1=file/function/line mentioned)",
    ),
    # V6: Solution and repro
    FeatureDefinition(
        name="solution_prescribed",
        min_value=0,
        max_value=1,
        description="Is solution explicitly described in problem? (0=problem only, 1=solution proposed)",
    ),
    FeatureDefinition(
        name="minimal_repro_provided",
        min_value=0,
        max_value=1,
        description="Is a reproduction case provided? (0=no repro, 1=repro code provided)",
    ),
]

SWEBENCH_SELECTED_PROMPT_TEMPLATE = """You are analyzing a SWE-bench coding task to predict its difficulty.
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

Analyze the problem and patch to evaluate these 6 features that predict task difficulty.

### 1. Domain Category (domain_category: 0-9)
What is the primary domain/subsystem of this task?
- 0: Infrastructure/configuration (build, setup, deployment)
- 1: Authentication/authorization
- 2: Database/ORM
- 3: API/serialization
- 4: UI/templates/rendering
- 5: Data processing/transformation
- 6: File/IO operations
- 7: Networking/HTTP
- 8: Testing infrastructure
- 9: Other/mixed

### 2. Solution Discovery Needed (solution_discovery_needed: 1-5)
Does solving this require exploration and discovery, or is the fix obvious?
- 1: Solution is immediately obvious from problem description
- 2: Solution is fairly clear, minimal exploration needed
- 3: Some investigation required to find the right approach
- 4: Significant exploration needed to understand the issue
- 5: Requires extensive investigation and discovery

### 3. Fix Pattern Type (fix_pattern_type: 0-4)
What type of fix pattern does the gold patch represent?
- 0: Add missing check/validation (guard clause, null check, boundary check)
- 1: Fix logic error (incorrect condition, wrong operator, off-by-one)
- 2: Add new method/function (new capability, hook, API endpoint)
- 3: Refactor existing code (restructure, improve without behavior change)
- 4: Architectural change (cross-cutting, design-level modification)

### 4. Location Hints Provided (location_hints_provided: 0/1)
Does the problem statement indicate WHERE to look in the code?
- 0: No location hints - must search for relevant code
- 1: Location hints provided - mentions file, function, class, or line numbers

### 5. Solution Prescribed (solution_prescribed: 0/1)
Does the problem statement explicitly prescribe the solution?
- 0: Problem only - describes what's wrong but not how to fix
- 1: Solution proposed - suggests specific code changes or approach

### 6. Minimal Repro Provided (minimal_repro_provided: 0/1)
Is a minimal reproduction case provided in the problem statement?
- 0: No repro - just description of the issue
- 1: Repro provided - code example that demonstrates the bug

## OUTPUT FORMAT

Respond with ONLY a JSON object. No markdown, no extra text.

{{
    "domain_category": <0-9>,
    "solution_discovery_needed": <1-5>,
    "fix_pattern_type": <0-4>,
    "location_hints_provided": <0 or 1>,
    "solution_prescribed": <0 or 1>,
    "minimal_repro_provided": <0 or 1>,
    "reasoning": "<2-3 sentences on what makes this task hard or easy>"
}}
"""


def format_swebench_selected_prompt(task: Dict[str, Any]) -> str:
    """Format the SWE-bench selected features prompt with task-specific information."""
    hints_text = task.get("hints_text", "")
    hints_section = f"**Hints:**\n{hints_text}" if hints_text and hints_text.strip() else ""

    return SWEBENCH_SELECTED_PROMPT_TEMPLATE.format(
        instance_id=task.get("instance_id", ""),
        repo=task.get("repo", ""),
        version=task.get("version", "unknown"),
        problem_statement=task.get("problem_statement", ""),
        patch=task.get("patch", ""),
        fail_to_pass=task.get("FAIL_TO_PASS", "[]"),
        hints_section=hints_section,
    )


SWEBENCH_SELECTED_CONFIG = PromptConfig(
    name="swebench_selected",
    features=SWEBENCH_SELECTED_FEATURES,
    prompt_template=SWEBENCH_SELECTED_PROMPT_TEMPLATE,
    task_id_field="instance_id",
    format_prompt_fn=format_swebench_selected_prompt,
)

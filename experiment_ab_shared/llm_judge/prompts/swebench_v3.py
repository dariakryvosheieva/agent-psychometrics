"""SWE-bench V3 prompt configuration for LLM judge feature extraction.

This module provides NEW semantic features that complement the V2 features.
These features are designed to be extracted separately and concatenated with V2.

These features focus on cross-cutting concerns and domain categories that
showed strong signals in hard task analysis (from screenshots and JSON labels).

V3 NEW features (6):
- cross_cutting_fix: Requires coordinated edits across multiple modules
  (strong signal: hard tasks often need schema/backends coordination in Django)
- domain_category: Specific domain category (infrastructure, auth, database, etc.)
  (from JSON labels - enriched in zero-success tasks)
- change_scope: Whether changes are local, module, or system-wide
- api_boundary_crossing: Whether fix crosses public API boundaries
- implicit_requirements: Whether there are implicit/undocumented requirements
- coordination_complexity: How much inter-module coordination is needed

Total: 6 LLM features (designed to complement V2's 9 features)
"""

from typing import Any, Dict

from experiment_ab_shared.llm_judge.prompt_config import FeatureDefinition, PromptConfig

SWEBENCH_V3_FEATURES = [
    FeatureDefinition(
        name="cross_cutting_fix",
        min_value=1,
        max_value=5,
        description="Requires coordinated edits across multiple modules? (1=single module, 5=system-wide coordination)",
    ),
    FeatureDefinition(
        name="domain_category",
        min_value=0,
        max_value=9,
        description="Primary domain category (0=general, 1=infrastructure, 2=auth, 3=database, 4=devops, 5=web, 6=testing, 7=serialization, 8=networking, 9=other)",
    ),
    FeatureDefinition(
        name="change_scope",
        min_value=1,
        max_value=3,
        description="Scope of changes (1=local/function, 2=module/class, 3=system-wide)",
    ),
    FeatureDefinition(
        name="api_boundary_crossing",
        min_value=0,
        max_value=1,
        description="Does the fix cross public API boundaries? (0=no, 1=yes)",
    ),
    FeatureDefinition(
        name="implicit_requirements",
        min_value=1,
        max_value=5,
        description="How many implicit/undocumented requirements? (1=none, 5=many hidden assumptions)",
    ),
    FeatureDefinition(
        name="coordination_complexity",
        min_value=1,
        max_value=5,
        description="How much inter-component coordination needed? (1=none, 5=complex multi-system coordination)",
    ),
]

SWEBENCH_V3_PROMPT_TEMPLATE = """You are analyzing a SWE-bench coding task to predict its difficulty.
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
Focus on cross-cutting concerns and module coordination.

### 1. Cross-Cutting Fix (cross_cutting_fix: 1-5)
Does the fix require coordinated changes across multiple modules/subsystems?
(E.g., Django tasks often need coordinated schema + backends + forms changes)
- 1: Single module, self-contained fix
- 2: Two related modules
- 3: Multiple modules with clear interfaces
- 4: Several interconnected modules
- 5: System-wide coordination required

### 2. Domain Category (domain_category: 0-9)
Primary domain category of the fix:
- 0: General purpose / utilities
- 1: Infrastructure (config, deployment, logging, caching)
- 2: Authentication / Authorization / Security
- 3: Database / ORM / Data modeling
- 4: DevOps / CI/CD / Build systems
- 5: Web / HTTP / REST API
- 6: Testing / Mocking / Fixtures
- 7: Serialization / Parsing / Encoding
- 8: Networking / Protocols / I/O
- 9: Other specialized domain

### 3. Change Scope (change_scope: 1-3)
What is the scope of the changes?
- 1: Local/function level - single function or method
- 2: Module/class level - affects a whole class or module
- 3: System-wide - affects multiple modules or architectural patterns

### 4. API Boundary Crossing (api_boundary_crossing: 0/1)
Does the fix cross public API boundaries?
- 0: Internal/private changes only
- 1: Changes to public APIs, documented interfaces, or exported functions

### 5. Implicit Requirements (implicit_requirements: 1-5)
How many implicit/undocumented requirements exist?
- 1: All requirements are explicit in the problem statement
- 2: Minor implicit assumptions
- 3: Some undocumented requirements discovered via patch
- 4: Significant hidden requirements
- 5: Many implicit assumptions, undocumented behaviors

### 6. Coordination Complexity (coordination_complexity: 1-5)
How much inter-component coordination is required?
- 1: No coordination - changes are independent
- 2: Minimal coordination - simple dependencies
- 3: Moderate coordination - multiple components must agree
- 4: High coordination - complex interdependencies
- 5: Very high - requires understanding entire subsystem interactions

## OUTPUT FORMAT

Respond with ONLY a JSON object. No markdown, no extra text.

{{
    "cross_cutting_fix": <1-5>,
    "domain_category": <0-9>,
    "change_scope": <1-3>,
    "api_boundary_crossing": <0 or 1>,
    "implicit_requirements": <1-5>,
    "coordination_complexity": <1-5>,
    "reasoning": "<2-3 sentences on cross-cutting complexity and coordination>"
}}
"""


def format_swebench_v3_prompt(task: Dict[str, Any]) -> str:
    """Format the SWE-bench V3 prompt with task-specific information."""
    hints_text = task.get("hints_text", "")
    hints_section = f"**Hints:**\n{hints_text}" if hints_text and hints_text.strip() else ""

    return SWEBENCH_V3_PROMPT_TEMPLATE.format(
        instance_id=task.get("instance_id", ""),
        repo=task.get("repo", ""),
        version=task.get("version", "unknown"),
        problem_statement=task.get("problem_statement", ""),
        patch=task.get("patch", ""),
        fail_to_pass=task.get("FAIL_TO_PASS", "[]"),
        hints_section=hints_section,
    )


SWEBENCH_V3_CONFIG = PromptConfig(
    name="swebench_v3",
    features=SWEBENCH_V3_FEATURES,
    prompt_template=SWEBENCH_V3_PROMPT_TEMPLATE,
    task_id_field="instance_id",
    format_prompt_fn=format_swebench_v3_prompt,
)

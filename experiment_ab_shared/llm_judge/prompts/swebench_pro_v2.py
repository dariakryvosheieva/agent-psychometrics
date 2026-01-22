"""SWE-bench Pro V2 prompt configuration with improved features.

Based on residual analysis of V1 features, this version adds features that better
capture what makes tasks hard in practice:

V1 features that worked well:
- fix_complexity: Correlated with difficulty (r=0.315, p=0.001)

V1 features that didn't help much:
- problem_clarity, error_message_provided, reproduction_steps: Wrong sign or weak correlation
- domain_knowledge_required, atypicality: Overestimate difficulty for well-patterned solutions

New LLM features in V2:
- task_type: Bug fix (1) / New feature (2) / Optimization (3) - different difficulty profiles
- cross_module_coordination: Does fix require changes across module boundaries?
- verification_difficulty: How hard is it to test/verify the fix?
- standard_pattern_available: Is this a well-known pattern with documentation?

Deterministic features (computed from patch, not LLM):
- num_files_modified: Count distinct files in patch
- num_lines_changed: Count +/- lines in patch
These are added during regression, not during LLM extraction.

Key insight: V1 captured what the *problem* looks like, but not what the *solution* requires.
"""

from typing import Any, Dict

from experiment_ab_shared.llm_judge.prompt_config import FeatureDefinition, PromptConfig

# Feature definitions for SWE-bench Pro V2
#
# Note: Some features are computed deterministically from the patch (no LLM needed):
# - num_files_modified: count distinct files in patch
# - num_lines_changed: count +/- lines in patch
# These are added during regression, not during LLM extraction.
#
SWEBENCH_PRO_V2_FEATURES = [
    # === RETAINED FROM V1 (worked well) ===
    FeatureDefinition(
        name="fix_in_description",
        min_value=0,
        max_value=3,
        description="Does the problem statement contain or hint at the solution? (0=none, 3=exact fix)",
    ),
    FeatureDefinition(
        name="fix_complexity",
        min_value=1,
        max_value=5,
        description="How complex is the actual fix? (1=trivial, 5=very complex)",
    ),
    FeatureDefinition(
        name="domain_knowledge_required",
        min_value=1,
        max_value=5,
        description="How much specialized knowledge is needed? (1=basic Python, 5=obscure APIs)",
    ),
    # === NEW IN V2 ===
    FeatureDefinition(
        name="task_type",
        min_value=1,
        max_value=3,
        description="What type of change is this? (1=bug fix, 2=new feature, 3=optimization/refactoring)",
    ),
    FeatureDefinition(
        name="cross_module_coordination",
        min_value=1,
        max_value=5,
        description="Does fix require coordinating changes across module/package boundaries? (1=isolated, 5=extensive coordination)",
    ),
    FeatureDefinition(
        name="verification_difficulty",
        min_value=1,
        max_value=5,
        description="How hard is it to test/verify the fix works correctly? (1=trivial to verify, 5=very hard to verify)",
    ),
    FeatureDefinition(
        name="standard_pattern_available",
        min_value=0,
        max_value=1,
        description="Is this a well-known pattern with existing documentation/examples? (0=novel solution needed, 1=standard pattern exists)",
    ),
]

# The prompt template for SWE-bench Pro V2
SWEBENCH_PRO_V2_PROMPT_TEMPLATE = """You are analyzing a SWE-bench coding task to predict its difficulty.
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

Analyze the problem statement and gold patch to evaluate these 7 semantic features.
Be precise and consistent with your ratings.

IMPORTANT: Focus on what makes the SOLUTION hard to implement, not just what the PROBLEM looks like.
A conceptually simple change (like renaming symbols) can be very hard if it requires many coordinated edits.

### 1. Fix Information in Description (fix_in_description: 0-3)
Does the problem statement contain or hint at the solution?
- 0: No hint at the solution at all
- 1: Vague hint or general direction
- 2: Clear description of what needs to change
- 3: Exact code fix or detailed solution provided

### 2. Fix Complexity (fix_complexity: 1-5)
How complex is the actual code change logic?
- 1: Trivial (add parameter, change value, simple one-liner)
- 2: Simple (straightforward logic change)
- 3: Moderate (requires understanding context, multiple changes)
- 4: Complex (algorithmic changes, multiple interdependent fixes)
- 5: Very complex (architectural changes, subtle edge cases, tricky bugs)

### 3. Domain Knowledge Required (domain_knowledge_required: 1-5)
How much specialized knowledge is needed to understand and fix this?
- 1: Basic Python/language basics, obvious fix anyone could make
- 2: Standard library knowledge needed
- 3: Framework-specific knowledge (Django, pytest, numpy, etc.)
- 4: Deep understanding of the library's internals
- 5: Obscure APIs, protocols, or highly specialized domain knowledge

### 4. Task Type (task_type: 1-3)
What type of change is this? Different task types have different difficulty profiles.
- 1: Bug fix - fixing incorrect behavior, error handling, edge cases
- 2: New feature - adding new functionality, new API, new capability
- 3: Optimization/Refactoring - improving performance, restructuring code, encapsulation changes

### 5. Cross-Module Coordination (cross_module_coordination: 1-5)
Does the fix require understanding and coordinating changes across different modules/packages?
Consider: Are changes isolated, or do they need to be consistent across multiple subsystems?
- 1: Completely isolated change, no coordination needed
- 2: Minor coordination within same module
- 3: Changes touch multiple related modules
- 4: Significant coordination across different subsystems
- 5: Extensive coordination across many modules/packages (e.g., API changes affecting many callers)

### 6. Verification Difficulty (verification_difficulty: 1-5)
How hard is it to test and verify that the fix works correctly?
Consider: Are there edge cases? Is the behavior easy to observe? Does testing require complex setup?
- 1: Trivial to verify (obvious pass/fail, simple test)
- 2: Easy to verify (straightforward test cases)
- 3: Moderate verification (some edge cases to consider)
- 4: Hard to verify (subtle correctness, complex setup, timing-dependent)
- 5: Very hard to verify (rare edge cases, hard to reproduce, correctness is subtle)

### 7. Standard Pattern Available (standard_pattern_available: 0/1)
Is this a well-known solution pattern with existing documentation or examples?
Even "complex-looking" fixes can be easy if they follow a documented cookbook pattern.
- 0: No standard pattern exists; novel solution needed
- 1: Well-documented pattern (e.g., "add a YAML representer", "implement __eq__", "add middleware")

## OUTPUT FORMAT

Respond with ONLY a JSON object containing all features. No markdown, no extra text.

{{
    "fix_in_description": <0-3>,
    "fix_complexity": <1-5>,
    "domain_knowledge_required": <1-5>,
    "task_type": <1-3>,
    "cross_module_coordination": <1-5>,
    "verification_difficulty": <1-5>,
    "standard_pattern_available": <0 or 1>,
    "reasoning": "<2-3 sentence summary focusing on what makes the SOLUTION hard or easy>"
}}
"""


def format_swebench_pro_v2_prompt(task: Dict[str, Any]) -> str:
    """Format the SWE-bench Pro V2 prompt with task-specific information."""
    hints_text = task.get("hints_text", "") or ""
    hints_section = ""
    if hints_text.strip():
        hints_section = f"**Hints:**\n{hints_text}"

    # Truncate very long fields to avoid context overflow
    problem_statement = task.get("problem_statement", "") or ""
    if len(problem_statement) > 12000:
        problem_statement = problem_statement[:12000]

    patch = task.get("patch", "") or ""
    if len(patch) > 8000:
        patch = patch[:8000]

    # SWE-bench Pro may use lowercase field names
    fail_to_pass = task.get("fail_to_pass") or task.get("FAIL_TO_PASS") or "[]"
    pass_to_pass = task.get("pass_to_pass") or task.get("PASS_TO_PASS") or "[]"

    return SWEBENCH_PRO_V2_PROMPT_TEMPLATE.format(
        instance_id=task.get("instance_id", ""),
        repo=task.get("repo", ""),
        version=task.get("version", "unknown"),
        problem_statement=problem_statement,
        patch=patch,
        fail_to_pass=fail_to_pass,
        pass_to_pass=pass_to_pass,
        hints_section=hints_section,
    )


# The main configuration object for SWE-bench Pro V2
SWEBENCH_PRO_V2_CONFIG = PromptConfig(
    name="swebench_pro_v2",
    features=SWEBENCH_PRO_V2_FEATURES,
    prompt_template=SWEBENCH_PRO_V2_PROMPT_TEMPLATE,
    task_id_field="instance_id",
    truncation_limits={
        "problem_statement": 12000,
        "patch": 8000,
    },
    format_prompt_fn=format_swebench_pro_v2_prompt,
)

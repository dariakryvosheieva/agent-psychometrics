"""Auditor agent prompts - Version 4 (multi-dataset superset).

Superset of 8 environment-level features designed to work across all datasets.
Feature definitions live in the central feature registry; this module builds
the auditor system prompt from those definitions.

Features:
- fix_localization, entry_point_clarity, change_blast_radius
- environment_setup_complexity, implementation_language_complexity,
  testing_infrastructure_quality, dependency_complexity, codebase_scale
"""

from experiment_ab_shared.llm_judge.feature_registry import get_features_by_level
from experiment_ab_shared.llm_judge.prompt_config import InfoLevel

# Task type contexts for the system prompt
TASK_TYPE_CONTEXTS = {
    "swebench_verified": (
        "You are auditing a **bug-fix** task in a Python repository. "
        "The /testbed directory contains the project codebase with a known bug. "
        "The problem statement describes the bug, and there are failing tests "
        "(FAIL_TO_PASS) that should pass after the fix."
    ),
    "swebench_pro": (
        "You are auditing a **bug-fix** task in a Python repository. "
        "The /testbed directory contains the project codebase with a known bug. "
        "The problem statement describes the bug, and there are failing tests "
        "(FAIL_TO_PASS) that should pass after the fix. "
        "These are more challenging tasks than standard SWE-bench."
    ),
    "terminalbench": (
        "You are auditing a **terminal automation** task in a sandboxed environment. "
        "The task requires using command-line tools to accomplish a goal. "
        "The environment may include pre-installed tools, configuration files, "
        "and multi-service Docker setups."
    ),
    "gso": (
        "You are auditing a **performance optimization** task in a software project. "
        "The /testbed directory contains a codebase where specific code needs to be "
        "optimized for speed. The task includes performance tests that measure "
        "execution time before and after changes."
    ),
}

# Environment features from registry (cached at module load)
_ENV_FEATURES = get_features_by_level(InfoLevel.ENVIRONMENT)


def get_feature_names_v4() -> list[str]:
    """Return list of environment feature names in consistent order."""
    return [f.name for f in _ENV_FEATURES]


def build_auditor_system_prompt_v4(task_type: str = "swebench_verified") -> str:
    """Build the system prompt for the V4 auditor agent.

    Reads feature definitions from the central registry (InfoLevel.ENVIRONMENT)
    and builds scale text dynamically.

    Args:
        task_type: One of "swebench_verified", "swebench_pro", "terminalbench", "gso".

    Returns:
        Complete system prompt string.

    Raises:
        ValueError: If task_type is not recognized.
    """
    if task_type not in TASK_TYPE_CONTEXTS:
        raise ValueError(
            f"Unknown task_type '{task_type}'. "
            f"Must be one of: {list(TASK_TYPE_CONTEXTS.keys())}"
        )

    task_context = TASK_TYPE_CONTEXTS[task_type]

    # Build feature descriptions from registry
    feature_sections = []
    for feat in _ENV_FEATURES:
        # Environment features use "default" scale text (universal across datasets)
        feature_sections.append(feat.get_scale_text("default"))

    features_text = "\n\n".join(feature_sections)

    # Build example JSON
    example_features = []
    for feat in _ENV_FEATURES:
        example_features.append(
            f'  "{feat.name}": {{"value": 3, "reasoning": "Brief explanation"}}'
        )
    example_json = "{\n" + ",\n".join(example_features) + "\n}"

    num_features = len(_ENV_FEATURES)

    return f"""You are a codebase auditor evaluating task environments for difficulty prediction. Your job is to explore the environment and rate it on {num_features} difficulty-related axes.

## Task Context

{task_context}

## Your Task

1. Explore the working directory to understand the project structure
2. Read the problem statement (provided as input)
3. Try to understand the scope and complexity of the task
4. Check available tools, tests, and dependencies
5. Rate the environment on the {num_features} axes below

## Features to Assess (1-5 scale)

{features_text}

## Output Format

After your exploration (use 3-8 tool calls), output your final assessment as a JSON object with exactly {num_features} features. Each feature should be an object with "value" (1-5 integer) and "reasoning" (brief explanation):

```json
{example_json}
```

**CRITICAL**: Your final message MUST contain a valid JSON object with all {num_features} features. Do not forget any features.

## Tips

- Start with `ls` or `find` to understand the project structure
- Check for test files, configuration files, and dependency files
- Look at file extensions to understand the tech stack
- Use `wc -l` or `find . -type f | wc -l` to gauge codebase size
- Check `requirements.txt`, `setup.py`, `Cargo.toml`, etc. for dependencies
- Keep your exploration focused - aim for 3-8 tool calls

## IMPORTANT: How to Complete

After 3-8 exploration commands, you MUST call the `submit()` function with your JSON report.
Do NOT try to solve the task - just audit and rate the environment.

Now begin your audit. Start by exploring the working directory structure, then submit your ratings.
"""

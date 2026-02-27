"""Auditor agent prompts - Version 4 (multi-dataset superset).

Superset of 8 features designed to work across all datasets:
- SWE-bench Verified & Pro (bug fixes)
- Terminal Bench (terminal automation)
- GSO (performance optimization)

Includes 3 proven V3 features + 5 new environment-level features.
Ridge regression selects the relevant subset per dataset.

V3 features retained (strong correlation with IRT difficulty on SWE-bench):
- fix_localization (-0.587) - strongest predictor
- entry_point_clarity (-0.502)
- change_blast_radius (+0.502)

New features for broader coverage:
- environment_setup_complexity
- implementation_language_complexity
- testing_infrastructure_quality
- dependency_complexity
- codebase_scale
"""

# Task type contexts for the system prompt
TASK_TYPE_CONTEXTS = {
    "swebench": (
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

# The 8 features for V4 auditor
AUDITOR_FEATURES_V4 = {
    # === Retained from V3 (proven on SWE-bench) ===
    "fix_localization": {
        "description": "How spread out is the likely solution?",
        "scale": {
            1: "Solution requires changes across many modules/packages",
            2: "Solution spans multiple files across different directories",
            3: "Solution spans 2-3 files in the same module",
            4: "Solution is in 1-2 closely related files",
            5: "Solution is contained to a single function/method",
        },
    },
    "entry_point_clarity": {
        "description": "How easy is it to find where the problem manifests?",
        "scale": {
            1: "No clear entry point, requires deep architecture knowledge",
            2: "Entry point exists but buried in abstraction layers",
            3: "Entry point findable with moderate searching",
            4: "Problem statement or tests hint at the location",
            5: "Clear from problem statement exactly which file/function",
        },
    },
    "change_blast_radius": {
        "description": "How many components would be affected by changes? (Higher = harder)",
        "scale": {
            1: "Isolated change, no downstream effects",
            2: "Minor coupling, 1-2 related files to consider",
            3: "Moderate coupling, changes affect a subsystem",
            4: "High coupling, changes ripple across modules",
            5: "Core/shared code, changes affect entire codebase",
        },
    },
    # === New features for multi-dataset coverage ===
    "environment_setup_complexity": {
        "description": "How complex is the runtime/tooling environment?",
        "scale": {
            1: "Standard single-directory project, ready to run out of the box",
            2: "Minor configuration needed, clear project structure",
            3: "Multiple services or components, custom configurations",
            4: "Complex orchestration, specialized dependencies, non-trivial build steps",
            5: "Exotic environment, multi-container setup, hardware-specific requirements",
        },
    },
    "implementation_language_complexity": {
        "description": "How complex is the primary language/tech stack for the solution?",
        "scale": {
            1: "Pure Python or simple shell commands",
            2: "Python with standard libraries, basic scripting",
            3: "Mixed languages (Python + build tools), moderately complex shell",
            4: "Compiled languages (C/C++), complex build systems, framework-specific patterns",
            5: "Multi-language (C/Rust + Python bindings), SIMD/assembly, exotic toolchains",
        },
    },
    "testing_infrastructure_quality": {
        "description": "How good is the testing/validation setup for verifying a solution?",
        "scale": {
            1: "No test framework, no way to validate changes",
            2: "Basic tests exist but hard to run or incomplete",
            3: "Standard test framework, moderate coverage",
            4: "Good test coverage, easy to run tests, clear pass/fail signals",
            5: "Comprehensive test suite, fast feedback loops, detailed error messages",
        },
    },
    "dependency_complexity": {
        "description": "How complex are the project dependencies?",
        "scale": {
            1: "No external dependencies, standard library only",
            2: "Few well-known dependencies (e.g., requests, numpy)",
            3: "Moderate number of standard packages",
            4: "Many dependencies, some specialized or version-sensitive",
            5: "Complex dependency tree, C extensions, system-level deps, version conflicts",
        },
    },
    "codebase_scale": {
        "description": "How large/complex is the codebase the agent needs to work with?",
        "scale": {
            1: "Tiny project (<100 files, <5K lines)",
            2: "Small project (100-500 files)",
            3: "Medium project (500-2000 files)",
            4: "Large project (2000-10000 files)",
            5: "Massive project (10000+ files, complex module structure)",
        },
    },
}


def get_feature_names_v4() -> list[str]:
    """Return list of feature names in consistent order."""
    return list(AUDITOR_FEATURES_V4.keys())


def build_auditor_system_prompt_v4(task_type: str = "swebench") -> str:
    """Build the system prompt for the V4 auditor agent.

    Args:
        task_type: One of "swebench", "swebench_pro", "terminalbench", "gso".
            Controls the task-type-specific context in the prompt.

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

    # Build feature descriptions
    feature_sections = []
    for name, info in AUDITOR_FEATURES_V4.items():
        lines = [f"### {name}"]
        lines.append(f"**{info['description']}**")
        lines.append("")
        for score, desc in info["scale"].items():
            lines.append(f"- {score}: {desc}")
        feature_sections.append("\n".join(lines))

    features_text = "\n\n".join(feature_sections)

    # Build example JSON
    example_features = []
    for name in AUDITOR_FEATURES_V4:
        example_features.append(
            f'  "{name}": {{"value": 3, "reasoning": "Brief explanation"}}'
        )
    example_json = "{\n" + ",\n".join(example_features) + "\n}"

    num_features = len(AUDITOR_FEATURES_V4)

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

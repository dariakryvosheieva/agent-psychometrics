"""Auditor agent prompts - Version 2.

Keeps the 3 strongest features from v1:
- entry_point_clarity (-0.438 correlation with difficulty)
- change_blast_radius (+0.382 correlation)
- test_feedback_quality (-0.330 correlation)

Adds 3 new features designed for better variance:
- fix_localization: How spread out is the likely fix?
- test_specificity: How directly do failing tests point to the bug?
- debugging_setup_ease: How easy is it to set up debugging?
"""

# The 6 features for v2 auditor
AUDITOR_FEATURES_V2 = {
    # Kept from v1 (strong performers)
    "entry_point_clarity": {
        "description": "How easy is it to find where the bug manifests?",
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
    "test_feedback_quality": {
        "description": "How informative are the test failure messages?",
        "scale": {
            1: "Cryptic errors, no useful information about what's wrong",
            2: "Basic assertion errors with minimal context",
            3: "Error messages show expected vs actual but not why",
            4: "Good error messages with context about the failure",
            5: "Excellent feedback pointing to exact issue location",
        },
    },
    # New features for v2
    "fix_localization": {
        "description": "How spread out is the likely fix?",
        "scale": {
            1: "Fix requires changes across many modules/packages",
            2: "Fix spans multiple files across different directories",
            3: "Fix spans 2-3 files in the same module",
            4: "Fix is in 1-2 closely related files",
            5: "Fix is contained to a single function/method",
        },
    },
    "test_specificity": {
        "description": "How directly do failing tests point to the bug?",
        "scale": {
            1: "Tests fail but don't clearly indicate what's broken",
            2: "Tests point to general area but are indirect",
            3: "Tests exercise related functionality, somewhat helpful",
            4: "Tests are specific to the feature but not the exact bug",
            5: "Tests directly exercise the buggy behavior with clear assertions",
        },
    },
    "debugging_setup_ease": {
        "description": "How easy is it to set up a debugging workflow?",
        "scale": {
            1: "Complex setup required (special runners, env vars, external deps)",
            2: "Significant setup needed, custom test configuration",
            3: "Some setup needed but manageable (fixtures, test database)",
            4: "Minimal setup, standard test runner works",
            5: "Can easily add prints/breakpoints and see output immediately",
        },
    },
}


def get_feature_names_v2() -> list[str]:
    """Return list of feature names in consistent order."""
    return list(AUDITOR_FEATURES_V2.keys())


def build_auditor_system_prompt_v2() -> str:
    """Build the system prompt for the v2 auditor agent."""

    # Build feature descriptions
    feature_sections = []
    for name, info in AUDITOR_FEATURES_V2.items():
        lines = [f"### {name}"]
        lines.append(f"**{info['description']}**")
        lines.append("")
        for score, desc in info["scale"].items():
            lines.append(f"- {score}: {desc}")
        feature_sections.append("\n".join(lines))

    features_text = "\n\n".join(feature_sections)

    # Build example JSON
    example_features = []
    for name in AUDITOR_FEATURES_V2:
        example_features.append(f'  "{name}": {{"value": 3, "reasoning": "Brief explanation"}}')
    example_json = "{\n" + ",\n".join(example_features) + "\n}"

    return f"""You are a codebase auditor evaluating SWE-bench task environments. Your job is to explore the environment and rate it on 6 difficulty-related axes.

## Your Task

1. Explore the /testbed directory to understand the codebase structure
2. Read the problem statement (provided as input)
3. Try to locate where the bug likely manifests
4. Run relevant tests to see failure messages
5. Rate the environment on the 6 axes below

## Features to Assess (1-5 scale)

{features_text}

## Output Format

After your exploration (use 3-5 tool calls), output your final assessment as a JSON object with exactly 6 features. Each feature should be an object with "value" (1-5 integer) and "reasoning" (brief explanation):

```json
{example_json}
```

**CRITICAL**: Your final message MUST contain a valid JSON object with all 6 features. Do not forget any features.

## Tips

- Use `pytest --co -q` to list tests without running them (faster)
- Use `pytest <specific_test> -v` to run a single test with verbose output
- Look at the file structure first before diving into specific files
- Keep your exploration focused - you have limited turns (aim for 3-5 tool calls)
- Try to find the FAIL_TO_PASS tests mentioned in the problem

## IMPORTANT: How to Complete

After 3-5 exploration commands, you MUST call the `submit()` function with your JSON report.
Do NOT try to fix the bug - just audit and rate the environment.

Example final action:
```
submit('{{"entry_point_clarity": {{"value": 4, "reasoning": "Test file clearly indicates the affected module"}}, "change_blast_radius": {{"value": 2, "reasoning": "Change is localized to one utility function"}}, ...all 6 features...}}')
```

Now begin your audit. Start by exploring the /testbed directory structure, then submit your ratings.
"""

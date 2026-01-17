"""Cacheable prompt format for iterative refinement.

The key insight is to structure prompts with task content FIRST (expensive, static)
and feature instructions SECOND (cheap, varies per iteration). This enables
automatic prefix caching in OpenAI's GPT-5.2 API.

For 30 tasks × 5 iterations:
- Without caching: 30 × 5 × 3000 = 450K input tokens
- With caching: 30 × 3000 (first) + 30 × 4 × 800 (rest) = 186K tokens
- Savings: ~60% on input tokens
"""

from typing import List, Optional


# Default feature instructions (from experiment_a/llm_judge_prompt.py)
DEFAULT_FEATURE_INSTRUCTIONS = """Analyze the problem statement and gold patch to evaluate these semantic features.
Be precise and consistent with your ratings.

### 1. Fix Information in Description (fix_in_description: 0-3)
Does the problem statement contain or hint at the solution?
- 0: No hint at the solution at all
- 1: Vague hint or general direction
- 2: Clear description of what needs to change
- 3: Exact code fix or detailed solution provided

### 2. Problem Clarity (problem_clarity: 1-5)
How clear and well-specified is the problem?
- 1: Very vague, unclear what's actually wrong
- 2: Somewhat clear but missing key details
- 3: Reasonably clear, some ambiguity
- 4: Clear with good context
- 5: Crystal clear with reproduction steps and expected behavior

### 3. Error Message/Traceback (error_message_provided: 0/1)
Does the problem include an error message or traceback?
- 0: No error message provided
- 1: Error message, traceback, or exception shown

### 4. Reproduction Steps (reproduction_steps: 0/1)
Are concrete reproduction steps provided?
- 0: No concrete reproduction steps
- 1: Has reproduction steps (code snippet, test case, or commands)

### 5. Fix Locality (fix_locality: 1-3)
How localized is the fix based on the patch?
- 1: Single location, few lines changed (1-5 lines)
- 2: Multiple locations in same file, or moderate changes (6-20 lines)
- 3: Multiple files or significant changes (>20 lines)

### 6. Domain Knowledge Required (domain_knowledge_required: 1-5)
How much specialized knowledge is needed to understand and fix this?
- 1: Basic Python, obvious fix anyone could make
- 2: Standard library knowledge needed
- 3: Framework-specific knowledge (Django, pytest, numpy, etc.)
- 4: Deep understanding of the library's internals
- 5: Obscure APIs, protocols, or highly specialized domain knowledge

### 7. Fix Complexity (fix_complexity: 1-5)
How complex is the actual fix?
- 1: Trivial (add parameter, change value, simple one-liner)
- 2: Simple (straightforward logic change)
- 3: Moderate (requires understanding context, multiple changes)
- 4: Complex (algorithmic changes, multiple interdependent fixes)
- 5: Very complex (architectural changes, subtle edge cases, tricky bugs)

### 8. Logical Reasoning Required (logical_reasoning_required: 1-5)
How much logical reasoning is needed to arrive at the fix?
- 1: Mechanical fix, no reasoning needed
- 2: Simple cause-effect reasoning
- 3: Multi-step reasoning required
- 4: Complex reasoning with multiple factors
- 5: Deep reasoning about edge cases, invariants, or system behavior

### 9. Atypicality (atypicality: 1-5)
How unusual is this bug pattern?
- 1: Very common bug pattern (typo, off-by-one, missing null check)
- 2: Common pattern (incorrect condition, wrong default)
- 3: Moderately unusual
- 4: Unusual bug pattern
- 5: Rare or novel bug pattern

## OUTPUT FORMAT

Respond with ONLY a JSON object containing all features. No markdown, no extra text.

{
    "fix_in_description": <0-3>,
    "problem_clarity": <1-5>,
    "error_message_provided": <0 or 1>,
    "reproduction_steps": <0 or 1>,
    "fix_locality": <1-3>,
    "domain_knowledge_required": <1-5>,
    "fix_complexity": <1-5>,
    "logical_reasoning_required": <1-5>,
    "atypicality": <1-5>,
    "reasoning": "<2-3 sentence summary of the key difficulty factors>"
}
"""


def format_task_content(
    instance_id: str,
    repo: str,
    version: str,
    problem_statement: str,
    patch: str,
    fail_to_pass: str,
    pass_to_pass: str,
    hints_text: str = "",
    max_problem_len: int = 12000,
    max_patch_len: int = 8000,
) -> str:
    """Format the task content (cached prefix).

    This is the expensive part (~2000-8000 tokens) that stays constant
    across different feature instruction variants.

    Args:
        instance_id: SWE-bench instance ID
        repo: Repository name (e.g., "django/django")
        version: Version string
        problem_statement: The issue description
        patch: The gold solution patch
        fail_to_pass: Tests that should pass after fix
        pass_to_pass: Regression tests
        hints_text: Optional hints
        max_problem_len: Truncation limit for problem statement
        max_patch_len: Truncation limit for patch

    Returns:
        Formatted task content string
    """
    # Truncate long fields
    if len(problem_statement) > max_problem_len:
        problem_statement = problem_statement[:max_problem_len] + "\n... [truncated]"
    if len(patch) > max_patch_len:
        patch = patch[:max_patch_len] + "\n... [truncated]"

    hints_section = ""
    if hints_text and hints_text.strip():
        hints_section = f"\n**Hints:**\n{hints_text}\n"

    return f"""## TASK INFORMATION

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
---
"""


def format_cacheable_prompt(
    task: dict,
    feature_instructions: str = DEFAULT_FEATURE_INSTRUCTIONS,
) -> str:
    """Format prompt with task content first (cached) and instructions second (varies).

    This structure enables automatic prefix caching in OpenAI's GPT-5.2 API:
    1. Task content comes first (~2000-8000 tokens, static per task)
    2. Feature instructions come second (~800 tokens, varies per iteration)

    On subsequent calls with the same task but different instructions,
    the task content prefix is cached and only the instructions are charged.

    Args:
        task: Dict with instance_id, repo, version, problem_statement, patch, etc.
        feature_instructions: The feature extraction instructions (varies per iteration)

    Returns:
        Formatted prompt string
    """
    task_content = format_task_content(
        instance_id=task["instance_id"],
        repo=task["repo"],
        version=task.get("version", "unknown"),
        problem_statement=task["problem_statement"],
        patch=task["patch"],
        fail_to_pass=task.get("FAIL_TO_PASS", "[]"),
        pass_to_pass=task.get("PASS_TO_PASS", "[]"),
        hints_text=task.get("hints_text", ""),
    )

    return f"""{task_content}

## FEATURE EXTRACTION INSTRUCTIONS

You are analyzing a SWE-bench coding task to predict its difficulty.
Analyze ONLY the static task information above (no code execution).

{feature_instructions}
"""


def extract_feature_names_from_instructions(instructions: str) -> List[str]:
    """Extract feature names from instruction text.

    Looks for patterns like "feature_name:" or "(feature_name:" in the instructions.

    Args:
        instructions: The feature extraction instructions text

    Returns:
        List of feature names found
    """
    import re

    # Match patterns like "fix_in_description:" or "(fix_in_description:"
    pattern = r'\(?\b([a-z_]+):\s*(?:\d|<)'
    matches = re.findall(pattern, instructions)

    # Deduplicate while preserving order
    seen = set()
    features = []
    for match in matches:
        if match not in seen:
            seen.add(match)
            features.append(match)

    return features

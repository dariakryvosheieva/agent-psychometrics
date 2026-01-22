"""SWE-bench Pro V3 prompt configuration with additional discriminative features.

Based on residual analysis of V2, this version adds features that capture:
- External specification/API requirements (harder than expected)
- Async/timing complexity (harder than expected)
- Migration/compatibility work (harder than expected)
- Edge case handling (harder - subtle parsing, empty fields, boundary conditions)
- Creating new abstractions (easier - building new vs modifying existing)

V3 features (12 total = 9 LLM + 3 deterministic):

LLM features:
1. fix_in_description (0-3) - kept from V1
2. fix_complexity (1-5) - kept from V1
3. verification_difficulty (1-5) - NEW in V2, significant
4. requires_external_spec (0/1) - NEW in V3
5. involves_async_or_timing (0/1) - NEW in V3
6. is_migration_or_compat (0/1) - NEW in V3
7. standard_pattern_available (0/1) - kept from V2
8. is_edge_case_handling (0/1) - NEW in V3 (from further residual analysis)
9. creates_new_abstraction (0/1) - NEW in V3 (from further residual analysis)

Deterministic features (computed from patch, not LLM):
- num_files_modified: Count distinct files in patch
- num_lines_changed: Count +/- lines in patch
- patch_adds_new_file: Whether patch creates new files (0/1)

Key insight from V2 residuals:
- External specs make tasks HARDER (even if isolated)
- Async/timing issues make tasks HARDER (even if "standard pattern")
- Edge case handling (empty fields, boundary conditions) makes tasks HARDER
- Creating new abstractions (new module/class) is often EASIER
- Pure code movement is EASIER (even if many files)
"""

from typing import Any, Dict

from experiment_ab_shared.llm_judge.prompt_config import FeatureDefinition, PromptConfig

SWEBENCH_PRO_V3_FEATURES = [
    # === CORE FEATURES (proven useful) ===
    FeatureDefinition(
        name="fix_in_description",
        min_value=0,
        max_value=3,
        description="Does the problem statement hint at the solution? (0=none, 3=exact fix)",
    ),
    FeatureDefinition(
        name="fix_complexity",
        min_value=1,
        max_value=5,
        description="How complex is the code change logic? (1=trivial, 5=very complex)",
    ),
    FeatureDefinition(
        name="verification_difficulty",
        min_value=1,
        max_value=5,
        description="How hard to test/verify the fix? (1=trivial, 5=very hard)",
    ),
    # === NEW IN V3 ===
    FeatureDefinition(
        name="requires_external_spec",
        min_value=0,
        max_value=1,
        description="Does fix need external spec/protocol/API knowledge not in codebase? (0=no, 1=yes)",
    ),
    FeatureDefinition(
        name="involves_async_or_timing",
        min_value=0,
        max_value=1,
        description="Does fix involve race conditions, async handling, or timing issues? (0=no, 1=yes)",
    ),
    FeatureDefinition(
        name="is_migration_or_compat",
        min_value=0,
        max_value=1,
        description="Is this about backwards compatibility or version migration? (0=no, 1=yes)",
    ),
    FeatureDefinition(
        name="standard_pattern_available",
        min_value=0,
        max_value=1,
        description="Is this a well-documented pattern with existing examples? (0=no, 1=yes)",
    ),
    FeatureDefinition(
        name="is_edge_case_handling",
        min_value=0,
        max_value=1,
        description="Is this fixing an edge case, corner case, or boundary condition? (0=no, 1=yes)",
    ),
    FeatureDefinition(
        name="creates_new_abstraction",
        min_value=0,
        max_value=1,
        description="Does the fix create a new module, class, or interface (vs modifying existing)? (0=no, 1=yes)",
    ),
]

SWEBENCH_PRO_V3_PROMPT_TEMPLATE = """You are analyzing a SWE-bench coding task to predict its difficulty.
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

**Regression tests (PASS_TO_PASS):**
{pass_to_pass}

{hints_section}

## FEATURES TO EVALUATE

Analyze the problem and patch to evaluate these 9 features.
Focus on what makes the SOLUTION hard, not what the PROBLEM looks like.

### 1. Fix Information in Description (fix_in_description: 0-3)
Does the problem statement hint at the solution?
- 0: No hint at all
- 1: Vague hint or general direction
- 2: Clear description of what needs to change
- 3: Exact code fix provided

### 2. Fix Complexity (fix_complexity: 1-5)
How complex is the actual code change logic?
- 1: Trivial (add parameter, change value)
- 2: Simple (straightforward logic change)
- 3: Moderate (understand context, multiple changes)
- 4: Complex (algorithmic, interdependent fixes)
- 5: Very complex (architectural, subtle edge cases)

### 3. Verification Difficulty (verification_difficulty: 1-5)
How hard is it to test/verify the fix works?
- 1: Trivial (obvious pass/fail)
- 2: Easy (straightforward test cases)
- 3: Moderate (some edge cases)
- 4: Hard (subtle correctness, complex setup)
- 5: Very hard (rare edge cases, hard to reproduce)

### 4. Requires External Specification (requires_external_spec: 0/1)
Does the fix require knowledge of external specs, protocols, file formats, or APIs that are NOT documented in the codebase itself?
- 0: Fix can be understood entirely from the codebase
- 1: Fix requires external research (RFCs, file format specs, third-party API docs, protocol standards)

Examples of external specs: ZIP file format, OAuth protocol, JSON Schema, HTTP headers, database wire protocol, encoding standards.

### 5. Involves Async or Timing (involves_async_or_timing: 0/1)
Does the fix involve race conditions, async/await patterns, or timing-sensitive code?
- 0: No timing or concurrency concerns
- 1: Fix involves async handling, race conditions, debouncing, or other timing-sensitive code

Examples: double-click prevention, debouncing, request deduplication, async state management, concurrent updates.

### 6. Is Migration or Compatibility (is_migration_or_compat: 0/1)
Is this about backwards compatibility, version migration, or preserving behavior across upgrades?
- 0: Not a migration/compatibility issue
- 1: Fix involves backwards compatibility, migration paths, or cross-version behavior

Examples: database schema migration, API versioning, deprecation handling, upgrade paths.

### 7. Standard Pattern Available (standard_pattern_available: 0/1)
Is this a well-documented pattern with existing examples?
Note: Even if async/timing is involved, the PATTERN might still be standard.
- 0: Novel solution needed
- 1: Well-documented pattern (e.g., "add middleware", "implement __eq__", "isLoading state")

### 8. Is Edge Case Handling (is_edge_case_handling: 0/1)
Is this about handling a specific edge case, corner case, or boundary condition?
Edge cases are often subtle and require deep understanding of the system.
- 0: Not an edge case fix (general feature, refactoring, broad change)
- 1: Edge case fix (empty strings, null values, boundary conditions, rare input patterns)

Examples: handling empty fields in parsing, null pointer edge cases, off-by-one errors, unicode edge cases.

### 9. Creates New Abstraction (creates_new_abstraction: 0/1)
Does the patch primarily CREATE new code (new file, class, module) rather than MODIFY existing code?
Creating new abstractions from scratch is often easier than modifying complex existing code.
- 0: Primarily modifies existing code
- 1: Primarily creates new abstraction (new file, new class, new module, new interface)

## OUTPUT FORMAT

Respond with ONLY a JSON object. No markdown, no extra text.

{{
    "fix_in_description": <0-3>,
    "fix_complexity": <1-5>,
    "verification_difficulty": <1-5>,
    "requires_external_spec": <0 or 1>,
    "involves_async_or_timing": <0 or 1>,
    "is_migration_or_compat": <0 or 1>,
    "standard_pattern_available": <0 or 1>,
    "is_edge_case_handling": <0 or 1>,
    "creates_new_abstraction": <0 or 1>,
    "reasoning": "<2-3 sentences on what makes the SOLUTION hard or easy>"
}}
"""


def format_swebench_pro_v3_prompt(task: Dict[str, Any]) -> str:
    """Format the SWE-bench Pro V3 prompt."""
    hints_text = task.get("hints_text", "") or ""
    hints_section = f"**Hints:**\n{hints_text}" if hints_text.strip() else ""

    problem_statement = task.get("problem_statement", "") or ""
    if len(problem_statement) > 12000:
        problem_statement = problem_statement[:12000]

    patch = task.get("patch", "") or ""
    if len(patch) > 8000:
        patch = patch[:8000]

    fail_to_pass = task.get("fail_to_pass") or task.get("FAIL_TO_PASS") or "[]"
    pass_to_pass = task.get("pass_to_pass") or task.get("PASS_TO_PASS") or "[]"

    return SWEBENCH_PRO_V3_PROMPT_TEMPLATE.format(
        instance_id=task.get("instance_id", ""),
        repo=task.get("repo", ""),
        version=task.get("version", "unknown"),
        problem_statement=problem_statement,
        patch=patch,
        fail_to_pass=fail_to_pass,
        pass_to_pass=pass_to_pass,
        hints_section=hints_section,
    )


SWEBENCH_PRO_V3_CONFIG = PromptConfig(
    name="swebench_pro_v3",
    features=SWEBENCH_PRO_V3_FEATURES,
    prompt_template=SWEBENCH_PRO_V3_PROMPT_TEMPLATE,
    task_id_field="instance_id",
    truncation_limits={
        "problem_statement": 12000,
        "patch": 8000,
    },
    format_prompt_fn=format_swebench_pro_v3_prompt,
)

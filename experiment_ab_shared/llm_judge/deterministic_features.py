"""Compute deterministic features from patches, test patches, and problem statements.

This module provides functions to compute task features that can be derived
deterministically from the task data without LLM calls.

For SWE-bench (from patches):
- num_files_modified: Count distinct files in patch
- num_hunks: Count diff hunks (@@ headers)
- num_lines_changed: Sum of added + deleted lines
- log_lines_changed: log10(num_lines_changed + 1)
- patch_adds: Number of lines added
- patch_deletes: Number of lines deleted
- patch_chars: Total characters in patch
- patch_files_gt2: Boolean - more than 2 files modified (52.9% vs 14.2% for hard tasks)
- patch_files_gt3: Boolean - more than 3 files modified (23.5% vs 4.4%)
- patch_lines_gt20: Boolean - more than 20 lines changed (67.6% vs 19.0%)
- patch_lines_gt50: Boolean - more than 50 lines changed (23.5% vs 5.6%)

For SWE-bench (from test patches):
- test_patch_chars: Total characters in test patch
- test_patch_hunks: Number of hunks in test patch
- test_patch_files: Number of files in test patch
- test_patch_lines: Total lines changed in test patch

For SWE-bench (from problem statement):
- stmt_words: Word count
- stmt_chars: Character count
- stmt_lines: Line count
- stmt_lines_gt80: Boolean - more than 80 lines (26.5% vs 12.2% for hard tasks)
- has_http_link: Boolean - contains http:// or https:// (58.8% vs 44.6%)
- has_code_block: Boolean - contains ``` or indented code
- has_stack_trace: Boolean - contains "Traceback" or common error patterns
- feature_request_phrasing: Boolean - contains feature request language

For SWE-bench (from metadata):
- repo_name: Repository name (e.g., "django/django")
- is_django: Boolean - task is from Django repo
- is_sympy: Boolean - task is from Sympy repo

For TerminalBench (from solution.sh):
- num_lines: Total lines in solution script
- num_pipes: Count of pipes (|)
- log_lines: log10(num_lines + 1)
"""

import math
import re
from typing import Any, Dict, List, Optional

# Try to import unidiff, fall back to regex parsing if not available
try:
    from unidiff import PatchSet

    HAS_UNIDIFF = True
except ImportError:
    HAS_UNIDIFF = False


def compute_patch_features(patch: str, extended: bool = True) -> Dict[str, Any]:
    """Compute deterministic features from a unified diff patch.

    Args:
        patch: Unified diff string
        extended: If True, include all extended features. If False, only original 4.

    Returns:
        Dict with keys including:
        - Original: num_files_modified, num_hunks, num_lines_changed, log_lines_changed
        - Extended: patch_adds, patch_deletes, patch_chars, patch_files_gt2,
                   patch_files_gt3, patch_lines_gt20, patch_lines_gt50

    Raises:
        ValueError: If patch is empty or None
    """
    if not patch or not patch.strip():
        raise ValueError("Patch is empty or None - cannot compute deterministic features")

    if HAS_UNIDIFF:
        return _compute_patch_features_unidiff(patch, extended=extended)
    else:
        return _compute_patch_features_regex(patch, extended=extended)


def _compute_patch_features_unidiff(patch: str, extended: bool = True) -> Dict[str, Any]:
    """Compute patch features using unidiff library."""
    try:
        patchset = PatchSet(patch)

        num_files = len(patchset)
        num_hunks = sum(len(f) for f in patchset)
        lines_added = sum(f.added for f in patchset)
        lines_removed = sum(f.removed for f in patchset)
        num_lines_changed = lines_added + lines_removed
        log_lines_changed = math.log10(num_lines_changed + 1) if num_lines_changed > 0 else 0.0

        features = {
            "num_files_modified": num_files,
            "num_hunks": num_hunks,
            "num_lines_changed": num_lines_changed,
            "log_lines_changed": log_lines_changed,
        }

        if extended:
            # Extended features from screenshots analysis
            features.update({
                "patch_adds": lines_added,
                "patch_deletes": lines_removed,
                "patch_chars": len(patch),
                # Threshold features (strong signals from hard task analysis)
                "patch_files_gt2": int(num_files > 2),  # 52.9% vs 14.2% for hard tasks
                "patch_files_gt3": int(num_files > 3),  # 23.5% vs 4.4%
                "patch_lines_gt20": int(num_lines_changed > 20),  # 67.6% vs 19.0%
                "patch_lines_gt50": int(num_lines_changed > 50),  # 23.5% vs 5.6%
            })

        return features
    except Exception as e:
        # Fall back to regex if unidiff parsing fails
        return _compute_patch_features_regex(patch, extended=extended)


def _compute_patch_features_regex(patch: str, extended: bool = True) -> Dict[str, Any]:
    """Compute patch features using regex (fallback when unidiff unavailable)."""
    # Count files (--- a/... or +++ b/... lines)
    file_pattern = re.compile(r"^(?:---|\+\+\+) [ab]/(.+)$", re.MULTILINE)
    files = set(file_pattern.findall(patch))
    num_files = len(files)

    # Count hunks (@@ ... @@ lines)
    hunk_pattern = re.compile(r"^@@ .+ @@", re.MULTILINE)
    num_hunks = len(hunk_pattern.findall(patch))

    # Count added/removed lines
    lines_added = len(re.findall(r"^\+[^+]", patch, re.MULTILINE))
    lines_removed = len(re.findall(r"^-[^-]", patch, re.MULTILINE))
    num_lines_changed = lines_added + lines_removed

    log_lines_changed = math.log10(num_lines_changed + 1) if num_lines_changed > 0 else 0.0

    features = {
        "num_files_modified": num_files,
        "num_hunks": num_hunks,
        "num_lines_changed": num_lines_changed,
        "log_lines_changed": log_lines_changed,
    }

    if extended:
        features.update({
            "patch_adds": lines_added,
            "patch_deletes": lines_removed,
            "patch_chars": len(patch),
            "patch_files_gt2": int(num_files > 2),
            "patch_files_gt3": int(num_files > 3),
            "patch_lines_gt20": int(num_lines_changed > 20),
            "patch_lines_gt50": int(num_lines_changed > 50),
        })

    return features


def compute_solution_features(solution: str) -> Dict[str, Any]:
    """Compute deterministic features from a shell script solution.

    Only includes features that are statistically significant (p<0.05) based on
    pilot analysis correlating with oracle IRT task difficulties:
    - log_lines (r=0.313**)
    - num_lines (r=0.296**)
    - num_pipes (r=0.272*)

    Dropped (not significant):
    - num_conditionals (r=0.197, p=0.066)
    - uses_awk (r=-0.201, p=0.060)
    - uses_sed (r=-0.130, p=0.227)
    - uses_grep (r=-0.092, p=0.394)

    Args:
        solution: Shell script content (e.g., solution.sh for TerminalBench)

    Returns:
        Dict with keys: num_lines, num_pipes, log_lines

    Raises:
        ValueError: If solution is empty or None
    """
    if not solution or not solution.strip():
        raise ValueError("Solution is empty or None - cannot compute deterministic features")

    lines = solution.strip().split("\n")
    num_lines = len(lines)

    # Count pipes (basic heuristic - counts | characters not in strings)
    num_pipes = solution.count("|")

    log_lines = math.log10(num_lines + 1) if num_lines > 0 else 0.0

    return {
        "num_lines": num_lines,
        "num_pipes": num_pipes,
        "log_lines": log_lines,
    }


def compute_test_patch_features(test_patch: str) -> Dict[str, Any]:
    """Compute deterministic features from a test patch.

    Based on analysis showing test patches for hard tasks are larger:
    - test_patch_chars: Median 5,852 vs 2,698 for hard tasks
    - test_patch_hunks: Mean 6.57 vs 4.01
    - test_patch_files: Mean 2.82 vs 1.73

    Args:
        test_patch: Unified diff string for the test patch

    Returns:
        Dict with keys: test_patch_chars, test_patch_hunks, test_patch_files, test_patch_lines

    Raises:
        ValueError: If test_patch is empty or None
    """
    if not test_patch or not test_patch.strip():
        raise ValueError("Test patch is empty or None - cannot compute deterministic features")

    # Reuse patch parsing logic
    if HAS_UNIDIFF:
        try:
            patchset = PatchSet(test_patch)
            num_files = len(patchset)
            num_hunks = sum(len(f) for f in patchset)
            lines_added = sum(f.added for f in patchset)
            lines_removed = sum(f.removed for f in patchset)
            num_lines_changed = lines_added + lines_removed
        except Exception:
            # Fall back to regex
            file_pattern = re.compile(r"^(?:---|\+\+\+) [ab]/(.+)$", re.MULTILINE)
            files = set(file_pattern.findall(test_patch))
            num_files = len(files)
            hunk_pattern = re.compile(r"^@@ .+ @@", re.MULTILINE)
            num_hunks = len(hunk_pattern.findall(test_patch))
            lines_added = len(re.findall(r"^\+[^+]", test_patch, re.MULTILINE))
            lines_removed = len(re.findall(r"^-[^-]", test_patch, re.MULTILINE))
            num_lines_changed = lines_added + lines_removed
    else:
        file_pattern = re.compile(r"^(?:---|\+\+\+) [ab]/(.+)$", re.MULTILINE)
        files = set(file_pattern.findall(test_patch))
        num_files = len(files)
        hunk_pattern = re.compile(r"^@@ .+ @@", re.MULTILINE)
        num_hunks = len(hunk_pattern.findall(test_patch))
        lines_added = len(re.findall(r"^\+[^+]", test_patch, re.MULTILINE))
        lines_removed = len(re.findall(r"^-[^-]", test_patch, re.MULTILINE))
        num_lines_changed = lines_added + lines_removed

    return {
        "test_patch_chars": len(test_patch),
        "test_patch_hunks": num_hunks,
        "test_patch_files": num_files,
        "test_patch_lines": num_lines_changed,
    }


def compute_problem_statement_features(problem_statement: str) -> Dict[str, Any]:
    """Compute deterministic features from a problem statement.

    Based on analysis showing hard tasks have different statement characteristics:
    - stmt_words: Median 151.5 vs 139 for hard tasks
    - stmt_lines: Mean 48.1 vs 40.5 for hard tasks
    - has_http_link: 58.8% vs 44.6% for hard tasks
    - feature_request_phrasing: 8.8% vs 4.6% for hard tasks

    Args:
        problem_statement: The problem statement text

    Returns:
        Dict with keys: stmt_words, stmt_chars, stmt_lines, stmt_lines_gt80,
                       has_http_link, has_code_block, has_stack_trace, feature_request_phrasing

    Raises:
        ValueError: If problem_statement is empty or None
    """
    if not problem_statement or not problem_statement.strip():
        raise ValueError("Problem statement is empty or None - cannot compute deterministic features")

    lines = problem_statement.split("\n")
    words = problem_statement.split()

    # Basic counts
    stmt_lines = len(lines)
    stmt_words = len(words)
    stmt_chars = len(problem_statement)

    # Threshold feature
    stmt_lines_gt80 = int(stmt_lines > 80)  # 26.5% vs 12.2% for hard tasks

    # Content features
    has_http_link = int(bool(re.search(r"https?://", problem_statement)))  # 58.8% vs 44.6%

    # Code block detection (``` or 4-space indented code)
    has_code_block = int(
        "```" in problem_statement
        or bool(re.search(r"^\s{4,}\S", problem_statement, re.MULTILINE))
    )

    # Stack trace detection
    has_stack_trace = int(
        "Traceback" in problem_statement
        or "Error:" in problem_statement
        or "Exception:" in problem_statement
        or bool(re.search(r"File \".*\", line \d+", problem_statement))
    )

    # Feature request phrasing (8.8% vs 4.6% for hard tasks)
    feature_request_patterns = [
        r"it would be nice",
        r"i would like",
        r"would be great",
        r"support for",
        r"feature request",
        r"enhancement",
        r"add support",
        r"should support",
    ]
    feature_request_phrasing = int(
        any(re.search(p, problem_statement, re.IGNORECASE) for p in feature_request_patterns)
    )

    return {
        "stmt_words": stmt_words,
        "stmt_chars": stmt_chars,
        "stmt_lines": stmt_lines,
        "stmt_lines_gt80": stmt_lines_gt80,
        "has_http_link": has_http_link,
        "has_code_block": has_code_block,
        "has_stack_trace": has_stack_trace,
        "feature_request_phrasing": feature_request_phrasing,
    }


def compute_task_metadata_features(task: Dict[str, Any]) -> Dict[str, Any]:
    """Compute deterministic features from task metadata.

    Based on analysis showing repo distribution differs for hard tasks:
    - Django: 44.1% of hard tasks vs 46.2% overall
    - Sympy: 14.7% of hard tasks vs 15.0% overall

    Args:
        task: Task dictionary with instance_id or repo field

    Returns:
        Dict with keys: is_django, is_sympy, repo_name

    Raises:
        ValueError: If task is missing required fields
    """
    # Try to extract repo from instance_id (format: "repo__issue-number")
    instance_id = task.get("instance_id", "")
    repo = task.get("repo", "")

    if not instance_id and not repo:
        raise ValueError("Task missing both instance_id and repo fields")

    # Extract repo from instance_id if not directly available
    if not repo and instance_id:
        # Format: "django__django-12345" -> "django/django"
        parts = instance_id.split("__")
        if len(parts) >= 2:
            # Handle cases like "django__django-12345"
            repo_parts = parts[0].split("__")
            if len(repo_parts) == 1:
                # Try to infer from common patterns
                repo = parts[0].replace("__", "/")
            else:
                repo = "/".join(repo_parts)

    # Normalize repo name
    repo_lower = repo.lower() if repo else instance_id.lower()

    return {
        "is_django": int("django" in repo_lower),
        "is_sympy": int("sympy" in repo_lower),
        "repo_name": repo,  # Keep as string for potential categorical encoding
    }


def compute_all_swebench_deterministic_features(
    task: Dict[str, Any],
    patch_field: str = "patch",
    test_patch_field: str = "test_patch",
    problem_statement_field: str = "problem_statement",
) -> Dict[str, Any]:
    """Compute ALL deterministic features for a SWE-bench task.

    This combines:
    - Extended patch features (11 features)
    - Test patch features (4 features)
    - Problem statement features (8 features)
    - Task metadata features (3 features)

    Total: ~26 deterministic features

    Args:
        task: Task dictionary with all required fields
        patch_field: Field name for the gold patch
        test_patch_field: Field name for the test patch
        problem_statement_field: Field name for the problem statement

    Returns:
        Dict with all deterministic features

    Raises:
        ValueError: If required fields are missing or empty
    """
    features = {}

    # Patch features
    patch = task.get(patch_field, "")
    if patch:
        features.update(compute_patch_features(patch, extended=True))
    else:
        raise ValueError(f"Task missing required field '{patch_field}'")

    # Test patch features (optional - some tasks may not have test patches)
    test_patch = task.get(test_patch_field, "")
    if test_patch and test_patch.strip():
        features.update(compute_test_patch_features(test_patch))
    else:
        # Fill with zeros if no test patch
        features.update({
            "test_patch_chars": 0,
            "test_patch_hunks": 0,
            "test_patch_files": 0,
            "test_patch_lines": 0,
        })

    # Problem statement features
    problem_statement = task.get(problem_statement_field, "")
    if problem_statement:
        features.update(compute_problem_statement_features(problem_statement))
    else:
        raise ValueError(f"Task missing required field '{problem_statement_field}'")

    # Task metadata features
    features.update(compute_task_metadata_features(task))

    return features


def add_deterministic_features_to_df(
    df,  # pandas DataFrame
    task_data: Dict[str, Dict[str, Any]],
    dataset_type: str = "swebench",
    patch_field: str = "patch",
    solution_field: str = "solution",
) -> None:
    """Add deterministic features to a DataFrame in-place.

    Args:
        df: pandas DataFrame with task features (modified in-place)
        task_data: Dict mapping task_id to task dict with patch/solution
        dataset_type: "swebench" or "terminalbench"
        patch_field: Field name for patch in task_data
        solution_field: Field name for solution in task_data

    Raises:
        ValueError: If any task is missing from task_data or has empty patch/solution
    """
    import pandas as pd

    # Get task ID column
    task_id_col = None
    for col in ["_instance_id", "instance_id", "_task_id", "task_id"]:
        if col in df.columns:
            task_id_col = col
            break

    if task_id_col is None:
        raise ValueError("Could not find task ID column in DataFrame")

    # Compute features for each task
    feature_rows = []
    missing_tasks = []
    for idx, row in df.iterrows():
        task_id = row[task_id_col]

        # Clean task ID
        clean_id = task_id.replace("instance_", "") if isinstance(task_id, str) else task_id

        if clean_id not in task_data:
            # Try with instance_ prefix
            if f"instance_{clean_id}" in task_data:
                clean_id = f"instance_{clean_id}"
            else:
                missing_tasks.append(task_id)
                continue

        task = task_data[clean_id]

        if dataset_type == "swebench":
            patch = task.get(patch_field, "")
            features = compute_patch_features(patch)  # Will raise if empty
        else:  # terminalbench
            solution = task.get(solution_field, "")
            features = compute_solution_features(solution)  # Will raise if empty

        feature_rows.append(features)

    if missing_tasks:
        raise ValueError(
            f"{len(missing_tasks)} tasks missing from task_data: {missing_tasks[:5]}..."
        )

    # Add features to DataFrame
    features_df = pd.DataFrame(feature_rows, index=df.index)
    for col in features_df.columns:
        df[col] = features_df[col]


# Feature names for each dataset type

# Original patch features (backward compatible)
SWEBENCH_DETERMINISTIC_FEATURES = [
    "num_files_modified",
    "num_hunks",
    "num_lines_changed",
    "log_lines_changed",
]

# Extended patch features
SWEBENCH_EXTENDED_PATCH_FEATURES = [
    "num_files_modified",
    "num_hunks",
    "num_lines_changed",
    "log_lines_changed",
    "patch_adds",
    "patch_deletes",
    "patch_chars",
    "patch_files_gt2",
    "patch_files_gt3",
    "patch_lines_gt20",
    "patch_lines_gt50",
]

# Test patch features
SWEBENCH_TEST_PATCH_FEATURES = [
    "test_patch_chars",
    "test_patch_hunks",
    "test_patch_files",
    "test_patch_lines",
]

# Problem statement features
SWEBENCH_STATEMENT_FEATURES = [
    "stmt_words",
    "stmt_chars",
    "stmt_lines",
    "stmt_lines_gt80",
    "has_http_link",
    "has_code_block",
    "has_stack_trace",
    "feature_request_phrasing",
]

# Task metadata features
SWEBENCH_METADATA_FEATURES = [
    "is_django",
    "is_sympy",
    # "repo_name",  # String - excluded from numeric features
]

# All deterministic features for SWE-bench (numeric only, for ML)
SWEBENCH_ALL_DETERMINISTIC_FEATURES = (
    SWEBENCH_EXTENDED_PATCH_FEATURES
    + SWEBENCH_TEST_PATCH_FEATURES
    + SWEBENCH_STATEMENT_FEATURES
    + SWEBENCH_METADATA_FEATURES
)

TERMINALBENCH_DETERMINISTIC_FEATURES = [
    "num_lines",
    "num_pipes",
    "log_lines",
]

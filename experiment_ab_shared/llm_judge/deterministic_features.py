"""Compute deterministic features from patches and solution scripts.

This module provides functions to compute task features that can be derived
deterministically from the task data without LLM calls:

For SWE-bench (from patches):
- num_files_modified: Count distinct files in patch
- num_hunks: Count diff hunks (@@ headers)
- num_lines_changed: Sum of added + deleted lines
- log_lines_changed: log10(num_lines_changed + 1)

For TerminalBench (from solution.sh):
- num_lines: Total lines in solution script
- num_pipes: Count of pipes (|)
- num_conditionals: Count of if/elif/case statements
- log_lines: log10(num_lines + 1)
"""

import math
import re
from typing import Any, Dict, List

# Try to import unidiff, fall back to regex parsing if not available
try:
    from unidiff import PatchSet

    HAS_UNIDIFF = True
except ImportError:
    HAS_UNIDIFF = False


def compute_patch_features(patch: str) -> Dict[str, Any]:
    """Compute deterministic features from a unified diff patch.

    Args:
        patch: Unified diff string

    Returns:
        Dict with keys: num_files_modified, num_hunks, num_lines_changed, log_lines_changed

    Raises:
        ValueError: If patch is empty or None
    """
    if not patch or not patch.strip():
        raise ValueError("Patch is empty or None - cannot compute deterministic features")

    if HAS_UNIDIFF:
        return _compute_patch_features_unidiff(patch)
    else:
        return _compute_patch_features_regex(patch)


def _compute_patch_features_unidiff(patch: str) -> Dict[str, Any]:
    """Compute patch features using unidiff library."""
    try:
        patchset = PatchSet(patch)

        num_files = len(patchset)
        num_hunks = sum(len(f) for f in patchset)
        lines_added = sum(f.added for f in patchset)
        lines_removed = sum(f.removed for f in patchset)
        num_lines_changed = lines_added + lines_removed
        log_lines_changed = math.log10(num_lines_changed + 1) if num_lines_changed > 0 else 0.0

        return {
            "num_files_modified": num_files,
            "num_hunks": num_hunks,
            "num_lines_changed": num_lines_changed,
            "log_lines_changed": log_lines_changed,
        }
    except Exception as e:
        # Fall back to regex if unidiff parsing fails
        return _compute_patch_features_regex(patch)


def _compute_patch_features_regex(patch: str) -> Dict[str, Any]:
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

    return {
        "num_files_modified": num_files,
        "num_hunks": num_hunks,
        "num_lines_changed": num_lines_changed,
        "log_lines_changed": log_lines_changed,
    }


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
SWEBENCH_DETERMINISTIC_FEATURES = [
    "num_files_modified",
    "num_hunks",
    "num_lines_changed",
    "log_lines_changed",
]

TERMINALBENCH_DETERMINISTIC_FEATURES = [
    "num_lines",
    "num_pipes",
    "log_lines",
]

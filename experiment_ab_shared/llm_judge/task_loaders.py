"""Task loading functions for all supported datasets.

Each loader returns a list of task dicts with the fields expected by
TaskContext formatters (see task_context.py). Loaders validate that
required fields are non-empty and raise on problems.

Usage:
    from experiment_ab_shared.llm_judge.task_loaders import load_tasks

    tasks = load_tasks("swebench_verified")
    tasks = load_tasks("terminalbench")
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_swebench_tasks() -> List[Dict[str, Any]]:
    """Load SWE-bench Verified dataset from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets package required for SWE-bench. Install with: pip install datasets"
        )

    print("Loading SWE-bench Verified dataset from HuggingFace...")
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

    tasks = []
    for item in ds:
        tasks.append({
            "instance_id": item["instance_id"],
            "repo": item["repo"],
            "problem_statement": item["problem_statement"],
            "patch": item["patch"],
            "test_patch": item["test_patch"],
            "version": item["version"],
            "hints_text": item["hints_text"],
            "FAIL_TO_PASS": item["FAIL_TO_PASS"],
            "PASS_TO_PASS": item["PASS_TO_PASS"],
        })

    print(f"Loaded {len(tasks)} tasks")
    return tasks


def _normalize_swebench_pro_task_id(task_id: str) -> str:
    """Normalize SWE-bench Pro task IDs to match IRT format.

    Removes 'instance_' prefix and version suffix (-v<hex> or -vnan).
    """
    import re

    if task_id.startswith('instance_'):
        task_id = task_id[9:]

    task_id = re.sub(r'-v(nan|[a-f0-9]+)$', '', task_id)

    return task_id


def load_swebench_pro_tasks() -> List[Dict[str, Any]]:
    """Load SWE-bench Pro dataset from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets package required for SWE-bench Pro. Install with: pip install datasets"
        )

    print("Loading SWE-bench Pro dataset from HuggingFace (ScaleAI/SWE-bench_Pro)...")
    ds = load_dataset("ScaleAI/SWE-bench_Pro", split="test")

    tasks = []
    for item in ds:
        normalized_id = _normalize_swebench_pro_task_id(item["instance_id"])
        tasks.append({
            "instance_id": normalized_id,
            "repo": item["repo"],
            "problem_statement": item["problem_statement"],
            "patch": item["patch"],
            "test_patch": item.get("test_patch", ""),
            "version": item.get("version", "unknown"),
            "hints_text": item.get("hints_text", ""),
            "fail_to_pass": item.get("fail_to_pass", "[]"),
            "pass_to_pass": item.get("pass_to_pass", "[]"),
        })

    print(f"Loaded {len(tasks)} tasks")
    return tasks


def load_gso_tasks() -> List[Dict[str, Any]]:
    """Load GSO (Software Optimization Benchmark) dataset from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets package required for GSO. Install with: pip install datasets"
        )

    print("Loading GSO dataset from HuggingFace (gso-bench/gso)...")
    ds = load_dataset("gso-bench/gso", split="test")

    tasks = []
    for item in ds:
        tasks.append({
            "instance_id": item["instance_id"],
            "repo": item.get("repo", ""),
            "api": item.get("api", ""),
            "prob_script": item.get("prob_script", ""),
            "gt_diff": item.get("gt_diff", ""),
            "hints_text": item.get("hints_text", ""),
        })

    print(f"Loaded {len(tasks)} tasks")
    return tasks


def load_terminalbench_tasks() -> List[Dict[str, Any]]:
    """Load TerminalBench tasks from the pre-extracted tasks JSONL.

    Reads from data/terminalbench/tasks.jsonl and cross-checks against
    the IRT items file to ensure they cover the same task set.

    Raises:
        FileNotFoundError: If the tasks JSONL is missing.
        ValueError: If tasks are missing required fields or the JSONL
            task set doesn't match the IRT items.
    """
    tasks_jsonl = Path("data/terminalbench/tasks.jsonl")
    if not tasks_jsonl.exists():
        raise FileNotFoundError(
            f"Tasks JSONL not found: {tasks_jsonl}. "
            f"Generate it with: python swebench_irt/scrape_terminal_bench_statements.py"
        )

    # Load all tasks from JSONL
    all_tasks = {}
    with open(tasks_jsonl, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                all_tasks[record["task_id"]] = record

    # Cross-check against IRT items
    irt_items_path = Path("data/terminalbench/irt/1d_1pl/items.csv")
    if irt_items_path.exists():
        import pandas as pd
        items_df = pd.read_csv(irt_items_path, index_col=0)
        irt_task_ids = set(items_df.index.tolist())
        jsonl_task_ids = set(all_tasks.keys())

        missing_from_jsonl = irt_task_ids - jsonl_task_ids
        missing_from_irt = jsonl_task_ids - irt_task_ids

        if missing_from_jsonl:
            raise ValueError(
                f"IRT items file has {len(missing_from_jsonl)} task(s) not in "
                f"tasks JSONL: {sorted(missing_from_jsonl)[:10]}. "
                f"Regenerate tasks JSONL or update IRT items."
            )
        if missing_from_irt:
            raise ValueError(
                f"Tasks JSONL has {len(missing_from_irt)} task(s) not in "
                f"IRT items: {sorted(missing_from_irt)[:10]}. "
                f"These task sets should match."
            )

        task_ids = list(irt_task_ids)
    else:
        task_ids = list(all_tasks.keys())

    # Validate and collect tasks
    tasks = []
    errors = []
    for task_id in sorted(task_ids):
        record = all_tasks[task_id]
        problem_statement = record.get("problem_statement", "").strip()
        patch = record.get("patch", "").strip()
        if not problem_statement:
            errors.append(f"Task {task_id}: empty problem_statement")
            continue
        if not patch:
            errors.append(f"Task {task_id}: empty patch")
            continue
        tasks.append({
            "task_id": task_id,
            "problem_statement": problem_statement,
            "patch": patch,
            "tests": record.get("tests", ""),
            "tags": record.get("tags", []),
            "category": record.get("category", ""),
            "difficulty": record.get("difficulty", ""),
        })

    if errors:
        error_summary = "\n".join(errors[:10])
        if len(errors) > 10:
            error_summary += f"\n... and {len(errors) - 10} more errors"
        raise ValueError(
            f"Found {len(errors)} tasks with missing data:\n{error_summary}"
        )

    print(f"Loaded {len(tasks)} TerminalBench tasks from {tasks_jsonl}")
    return tasks


def load_tasks_from_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load tasks from a JSONL file."""
    tasks = []
    with open(path) as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    print(f"Loaded {len(tasks)} tasks from {path}")
    return tasks


# Dataset name -> loader function
_LOADERS = {
    "swebench_verified": load_swebench_tasks,
    "swebench_pro": load_swebench_pro_tasks,
    "terminalbench": load_terminalbench_tasks,
    "gso": load_gso_tasks,
}

SUPPORTED_DATASETS = sorted(_LOADERS.keys())


def load_tasks(dataset: str) -> List[Dict[str, Any]]:
    """Load tasks for a named dataset.

    Args:
        dataset: Dataset name (swebench, swebench_pro, terminalbench, gso)

    Raises:
        ValueError: If dataset is unknown
    """
    if dataset not in _LOADERS:
        raise ValueError(
            f"Unknown dataset: '{dataset}'. "
            f"Available: {sorted(_LOADERS.keys())}"
        )
    return _LOADERS[dataset]()

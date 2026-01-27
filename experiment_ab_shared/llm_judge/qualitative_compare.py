"""Qualitative comparison of hard vs easy tasks for manual feature validation.

This module samples and displays hard vs easy tasks side-by-side to verify
that features behave as expected before running quantitative analysis.

Usage:
    python -m experiment_ab_shared.llm_judge.qualitative_compare \
        --dataset swebench --n-pairs 3

    python -m experiment_ab_shared.llm_judge.qualitative_compare \
        --dataset swebench --feature patch_files_gt2
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def load_tasks_and_difficulties(
    dataset: str,
    irt_items_path: Optional[Path] = None,
) -> tuple:
    """Load tasks and their IRT difficulties.

    Returns:
        Tuple of (tasks_dict, difficulties_series)
    """
    from experiment_ab_shared.llm_judge.extract_pipeline import (
        _load_tasks_for_dataset,
        _get_task_id,
    )

    # Load tasks
    tasks = _load_tasks_for_dataset(dataset)
    task_dict = {_get_task_id(t, dataset): t for t in tasks}

    # Load IRT difficulties
    if irt_items_path is None:
        if dataset == "swebench":
            irt_items_path = Path("clean_data/swebench_verified_20251120_full/1d/items.csv")
        elif dataset == "swebench_pro":
            irt_items_path = Path("clean_data/swebench_pro/1d/items.csv")
        elif dataset == "gso":
            irt_items_path = Path("clean_data/gso/1d/items.csv")
        elif dataset == "terminalbench":
            irt_items_path = Path("clean_data/terminalbench/1d/items.csv")
        else:
            raise ValueError(f"No default IRT items path for dataset: {dataset}")

    if not irt_items_path.exists():
        raise FileNotFoundError(f"IRT items not found at: {irt_items_path}")

    irt_df = pd.read_csv(irt_items_path, index_col=0)
    irt_df.index = irt_df.index.str.replace(r"^instance_", "", regex=True)
    difficulties = irt_df["b"]

    return task_dict, difficulties


def sample_hard_easy_pair(
    task_dict: Dict[str, Dict],
    difficulties: pd.Series,
    hard_percentile: float = 0.8,
    easy_percentile: float = 0.2,
    seed: Optional[int] = None,
) -> tuple:
    """Sample one hard and one easy task.

    Args:
        task_dict: Dict mapping task_id -> task
        difficulties: Series mapping task_id -> difficulty
        hard_percentile: Percentile threshold for hard tasks (default: top 20%)
        easy_percentile: Percentile threshold for easy tasks (default: bottom 20%)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (hard_task_id, easy_task_id)
    """
    rng = np.random.default_rng(seed)

    # Get common task IDs
    common_ids = [t for t in task_dict.keys() if t in difficulties.index]

    if not common_ids:
        raise ValueError("No common task IDs between tasks and difficulties")

    # Split by difficulty percentiles
    hard_threshold = difficulties.loc[common_ids].quantile(hard_percentile)
    easy_threshold = difficulties.loc[common_ids].quantile(easy_percentile)

    hard_ids = [t for t in common_ids if difficulties.loc[t] >= hard_threshold]
    easy_ids = [t for t in common_ids if difficulties.loc[t] <= easy_threshold]

    if not hard_ids or not easy_ids:
        raise ValueError("Not enough tasks in hard or easy category")

    hard_id = rng.choice(hard_ids)
    easy_id = rng.choice(easy_ids)

    return hard_id, easy_id


def format_task_display(
    task: Dict[str, Any],
    task_id: str,
    difficulty: float,
    difficulties: pd.Series,
    features_df: Optional[pd.DataFrame] = None,
    feature_to_highlight: Optional[str] = None,
    max_statement_chars: int = 800,
) -> str:
    """Format a task for display.

    Args:
        task: Task dictionary
        task_id: Task ID
        difficulty: Difficulty value
        difficulties: All difficulties (for percentile calculation)
        features_df: Optional DataFrame with computed features
        feature_to_highlight: Optional feature to highlight
        max_statement_chars: Max characters to show from problem statement

    Returns:
        Formatted string for display
    """
    from experiment_ab_shared.llm_judge.deterministic_features import (
        compute_patch_features,
        compute_test_patch_features,
        compute_problem_statement_features,
    )

    # Calculate percentile
    percentile = (difficulties < difficulty).mean() * 100

    lines = []
    lines.append(f"Task ID: {task_id}")
    lines.append(f"Difficulty: {difficulty:.2f} (percentile: {percentile:.1f}%)")

    # Repository
    repo = task.get("repo", "")
    if not repo:
        # Extract from instance_id
        parts = task_id.split("__")
        if len(parts) >= 1:
            repo = parts[0].replace("_", "/")
    lines.append(f"Repository: {repo}")

    # Problem statement
    stmt = task.get("problem_statement", "")
    if stmt:
        stmt_truncated = stmt[:max_statement_chars]
        if len(stmt) > max_statement_chars:
            stmt_truncated += "..."
        lines.append(f"\nProblem Statement ({len(stmt)} chars):")
        # Indent each line
        for line in stmt_truncated.split("\n")[:15]:
            lines.append(f"  {line}")
        if stmt.count("\n") > 15:
            lines.append(f"  ... ({stmt.count(chr(10)) - 15} more lines)")

    # Patch stats
    patch = task.get("patch", "")
    if patch:
        try:
            patch_features = compute_patch_features(patch, extended=True)
            lines.append(f"\nGold Patch:")
            lines.append(f"  Files: {patch_features['num_files_modified']}")
            lines.append(f"  Lines changed: {patch_features['num_lines_changed']} (+{patch_features['patch_adds']}/-{patch_features['patch_deletes']})")
            lines.append(f"  Hunks: {patch_features['num_hunks']}")
        except Exception:
            lines.append(f"\nGold Patch: {len(patch)} chars")

    # Test patch stats
    test_patch = task.get("test_patch", "")
    if test_patch:
        try:
            tp_features = compute_test_patch_features(test_patch)
            lines.append(f"\nTest Patch:")
            lines.append(f"  Files: {tp_features['test_patch_files']}")
            lines.append(f"  Lines: {tp_features['test_patch_lines']}")
            lines.append(f"  Chars: {tp_features['test_patch_chars']}")
        except Exception:
            lines.append(f"\nTest Patch: {len(test_patch)} chars")

    # Problem statement features
    stmt = task.get("problem_statement", "")
    if stmt:
        try:
            stmt_features = compute_problem_statement_features(stmt)
            lines.append(f"\nStatement Features:")
            lines.append(f"  Words: {stmt_features['stmt_words']}, Lines: {stmt_features['stmt_lines']}")
            lines.append(f"  Has HTTP link: {bool(stmt_features['has_http_link'])}")
            lines.append(f"  Has code block: {bool(stmt_features['has_code_block'])}")
            lines.append(f"  Has stack trace: {bool(stmt_features['has_stack_trace'])}")
            lines.append(f"  Feature request phrasing: {bool(stmt_features['feature_request_phrasing'])}")
        except Exception:
            pass

    # Highlighted feature
    if feature_to_highlight and features_df is not None:
        clean_id = task_id.replace("instance_", "")
        if clean_id in features_df.index and feature_to_highlight in features_df.columns:
            value = features_df.loc[clean_id, feature_to_highlight]
            lines.append(f"\n>>> Feature [{feature_to_highlight}]: {value}")

    return "\n".join(lines)


def compare_pair(
    task_dict: Dict[str, Dict],
    difficulties: pd.Series,
    hard_id: str,
    easy_id: str,
    features_df: Optional[pd.DataFrame] = None,
    feature_to_highlight: Optional[str] = None,
) -> None:
    """Display comparison of a hard vs easy task pair."""
    hard_task = task_dict[hard_id]
    easy_task = task_dict[easy_id]

    hard_diff = difficulties.loc[hard_id]
    easy_diff = difficulties.loc[easy_id]

    print("\n" + "=" * 80)
    print("HARD TASK")
    print("=" * 80)
    print(format_task_display(
        hard_task, hard_id, hard_diff, difficulties,
        features_df, feature_to_highlight
    ))

    print("\n" + "=" * 80)
    print("EASY TASK")
    print("=" * 80)
    print(format_task_display(
        easy_task, easy_id, easy_diff, difficulties,
        features_df, feature_to_highlight
    ))

    print("\n" + "=" * 80)


def run_comparison(
    dataset: str,
    n_pairs: int = 1,
    feature_to_highlight: Optional[str] = None,
    features_csv: Optional[Path] = None,
    irt_items_path: Optional[Path] = None,
    seed: Optional[int] = None,
    interactive: bool = False,
) -> None:
    """Run qualitative comparison.

    Args:
        dataset: Dataset name
        n_pairs: Number of pairs to show
        feature_to_highlight: Optional feature to highlight
        features_csv: Optional path to features CSV
        irt_items_path: Optional path to IRT items
        seed: Random seed
        interactive: If True, prompt for more pairs
    """
    # Load data
    task_dict, difficulties = load_tasks_and_difficulties(dataset, irt_items_path)

    # Load features if provided
    features_df = None
    if features_csv:
        features_df = pd.read_csv(features_csv)
        # Find task ID column and set as index
        for col in ["_instance_id", "instance_id", "_task_id", "task_id"]:
            if col in features_df.columns:
                features_df = features_df.set_index(col)
                break
        features_df.index = features_df.index.str.replace(r"^instance_", "", regex=True)

    print(f"\nDataset: {dataset}")
    print(f"Tasks: {len(task_dict)}")
    print(f"With difficulties: {len(difficulties)}")

    if feature_to_highlight:
        print(f"Highlighting feature: {feature_to_highlight}")

    rng = np.random.default_rng(seed)

    for i in range(n_pairs):
        pair_seed = rng.integers(0, 10000)
        hard_id, easy_id = sample_hard_easy_pair(task_dict, difficulties, seed=pair_seed)
        compare_pair(task_dict, difficulties, hard_id, easy_id, features_df, feature_to_highlight)

        if interactive and i < n_pairs - 1:
            response = input("\nPress Enter for next pair, 'q' to quit: ")
            if response.lower() == 'q':
                break

    if interactive:
        while True:
            response = input("\nPress Enter for another pair, 'q' to quit: ")
            if response.lower() == 'q':
                break
            pair_seed = rng.integers(0, 10000)
            hard_id, easy_id = sample_hard_easy_pair(task_dict, difficulties, seed=pair_seed)
            compare_pair(task_dict, difficulties, hard_id, easy_id, features_df, feature_to_highlight)


def main():
    parser = argparse.ArgumentParser(
        description="Compare hard vs easy tasks for qualitative validation"
    )
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--n-pairs", type=int, default=1, help="Number of pairs to show")
    parser.add_argument("--feature", help="Feature to highlight")
    parser.add_argument("--features-csv", type=Path, help="Path to features CSV")
    parser.add_argument("--irt-items", type=Path, help="Path to IRT items.csv")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")

    args = parser.parse_args()

    run_comparison(
        dataset=args.dataset,
        n_pairs=args.n_pairs,
        feature_to_highlight=args.feature,
        features_csv=args.features_csv,
        irt_items_path=args.irt_items,
        seed=args.seed,
        interactive=args.interactive,
    )


if __name__ == "__main__":
    main()

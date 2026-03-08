"""Extract features on a sample of tasks and check correlation with difficulty.

This script is for validating feature design before running on all 34 frontier tasks.
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from experiment_appendix_h_hard_tasks.trajectory_features.simple_extractor import SimpleFeatureExtractor
from experiment_appendix_h_hard_tasks.swebench.config import SWEBenchConfig
from experiment_appendix_h_hard_tasks.trajectory_features.prompts_frontier_v1 import get_frontier_v1_config
from experiment_appendix_h_hard_tasks.trajectory_features.utils import (
    build_task_dicts,
    load_frontier_tasks_with_difficulties,
)


def select_sample_tasks(
    frontier_tasks: List[str],
    oracle_items: pd.DataFrame,
    n_per_group: int = 2,
) -> List[Tuple[str, float]]:
    """Select sample tasks across the difficulty spectrum.

    Returns list of (task_id, difficulty) tuples.
    """
    # Get difficulties and sort
    task_diffs = []
    for task_id in frontier_tasks:
        if task_id in oracle_items.index:
            task_diffs.append((task_id, oracle_items.loc[task_id, "b"]))

    task_diffs.sort(key=lambda x: x[1])
    n = len(task_diffs)

    # Select from easy, medium, hard
    easy = task_diffs[:n_per_group]
    medium = task_diffs[n//2 - n_per_group//2 : n//2 + n_per_group//2 + n_per_group % 2]
    hard = task_diffs[-n_per_group:]

    selected = easy + medium + hard

    print(f"Selected {len(selected)} sample tasks:")
    for task_id, diff in selected:
        label = "EASY" if diff < 2.5 else "HARD" if diff > 3.5 else "MEDIUM"
        print(f"  {label}: {task_id} (β={diff:.3f})")

    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Extract features on sample tasks and check correlation"
    )
    parser.add_argument(
        "--n-per-group",
        type=int,
        default=2,
        help="Number of tasks per difficulty group (default: 2)",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="20250415_openhands",
        help="Agent to use for trajectories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("chris_output/trajectory_features/sample_test"),
        help="Output directory for sample extraction",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Just show what would be extracted",
    )

    args = parser.parse_args()

    # Load config
    config = SWEBenchConfig()
    prompt_config = get_frontier_v1_config()

    # Load frontier tasks
    print("Loading frontier tasks...")
    frontier_tasks, oracle_items, _, _ = load_frontier_tasks_with_difficulties(config)
    print(f"  Total frontier tasks: {len(frontier_tasks)}")

    # Select sample
    print("\nSelecting sample tasks...")
    sample_tasks = select_sample_tasks(frontier_tasks, oracle_items, args.n_per_group)
    sample_task_ids = [t[0] for t in sample_tasks]

    if args.dry_run:
        print("\n[DRY RUN] Would extract features for the above tasks")
        return

    # Build task dicts for sample only
    print(f"\nLoading trajectories for {len(sample_task_ids)} sample tasks...")
    trajs_dir = Path("experiment_appendix_h_hard_tasks/trajectory_data/unified_trajs")
    task_dicts, missing = build_task_dicts(
        sample_task_ids, args.agent, trajs_dir,
        max_messages=200,  # More messages for fuller trajectory
        max_chars_per_message=3000,
    )

    if missing:
        print(f"  Warning: {len(missing)} missing trajectories")

    # Extract features
    print(f"\nExtracting features...")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    extractor = SimpleFeatureExtractor(
        prompt_config=prompt_config,
        output_dir=args.output_dir,
        provider="anthropic",
    )

    csv_path = extractor.run(task_dicts, skip_existing=False)

    if not csv_path or not csv_path.exists():
        print("Feature extraction failed")
        return

    # Load features and compute correlations
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)

    features_df = pd.read_csv(csv_path)

    # Merge with difficulties
    task_id_col = "_task_id" if "_task_id" in features_df.columns else "task_id"
    features_df = features_df.set_index(task_id_col)
    merged = features_df.join(oracle_items[["b"]], how="inner")

    # Identify feature columns
    feature_cols = [c for c in features_df.columns
                    if not c.startswith("_") and c != "reasoning"]

    print(f"\nSample size: n={len(merged)}")
    print(f"Difficulty range: {merged['b'].min():.3f} to {merged['b'].max():.3f}")

    print(f"\n{'Feature':<30} {'Pearson r':>12} {'p-value':>10} {'Direction':>12}")
    print("-" * 70)

    for feature in feature_cols:
        if feature not in merged.columns:
            continue
        r, p = stats.pearsonr(merged["b"], merged[feature])
        sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
        direction = "↑ harder" if r > 0 else "↓ easier"
        print(f"{feature:<30} {r:>+10.3f}   {p:>8.3f}{sig:<2} {direction:>12}")

    # Show raw data
    print("\n" + "=" * 70)
    print("RAW FEATURE VALUES")
    print("=" * 70)

    cols_to_show = ["b"] + [c for c in feature_cols if c in merged.columns]
    print(merged[cols_to_show].to_string())


if __name__ == "__main__":
    main()

"""Compute execution features for trajectories (deterministic, no API calls).

This script extracts execution-time features from agent trajectories for
Experiment B. Features are computed deterministically from trajectory JSON
without any LLM API calls.

Usage:
    # Dry run to see execution plan
    python -m experiment_b.compute_execution_features --dry_run

    # Run on small subset for validation
    python -m experiment_b.compute_execution_features --limit 50

    # Full run (all M1×D_train + M2×D_valid)
    python -m experiment_b.compute_execution_features

    # Analyze feature distributions
    python -m experiment_b.compute_execution_features --analyze
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

# Add parent to path for imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_b.config import ExperimentConfig
from experiment_b.data_splits import create_experiment_split
from experiment_b.trajectory_features_v2 import (
    MECHANICAL_FEATURE_NAMES,
    EXECUTION_FEATURE_NAMES,  # backwards compat alias
    extract_mechanical_features,
    aggregate_mechanical_features,
    parse_gold_patch_files,
    MechanicalFeatures,
)


# Directories
UNIFIED_TRAJS_DIR = ROOT / "trajectory_data" / "unified_trajs"
OUTPUT_DIR = ROOT / "chris_output" / "experiment_b" / "execution_features"


def load_swebench_patches() -> Dict[str, str]:
    """Load gold patches from SWE-bench Verified dataset.

    Returns:
        Dict mapping task_id -> patch string
    """
    try:
        from datasets import load_dataset
        print("Loading SWE-bench Verified patches...")
        ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
        patches = {item["instance_id"]: item["patch"] for item in ds}
        print(f"Loaded patches for {len(patches)} tasks")
        return patches
    except ImportError:
        print("Warning: datasets library not available, patches won't be loaded")
        return {}


def load_trajectory(agent: str, task_id: str) -> Optional[dict]:
    """Load a unified trajectory JSON file."""
    traj_path = UNIFIED_TRAJS_DIR / agent / f"{task_id}.json"
    if not traj_path.exists():
        return None

    try:
        with open(traj_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def compute_features_for_task(
    task_id: str,
    agents: List[str],
    gold_patch: Optional[str] = None,
) -> Dict[str, dict]:
    """Compute mechanical features for all agents on a task.

    Args:
        task_id: Task instance ID
        agents: List of agent names
        gold_patch: Optional gold patch string (unused for mechanical features)

    Returns:
        Dict mapping agent -> feature dict
    """
    results = {}
    for agent in agents:
        trajectory = load_trajectory(agent, task_id)
        if trajectory is None:
            continue

        features = extract_mechanical_features(trajectory)

        # Convert to serializable dict
        results[agent] = {
            "syntax_error_count": features.syntax_error_count,
            "test_run_count": features.test_run_count,
            "traceback_count": features.traceback_count,
            "unique_files_edited": list(features.unique_files_edited),
            "total_commands": features.total_commands,
            "edit_attempts": features.edit_attempts,
        }

    return results


def analyze_features(output_dir: Path):
    """Analyze computed feature distributions.

    Args:
        output_dir: Directory containing computed features
    """
    print("\n=== Execution Feature Analysis ===\n")

    all_features = {name: [] for name in EXECUTION_FEATURE_NAMES}
    task_count = 0

    # Load all feature files
    for task_file in output_dir.glob("*.json"):
        if task_file.name.startswith("_"):
            continue

        try:
            with open(task_file) as f:
                data = json.load(f)

            if "aggregated" in data:
                task_count += 1
                agg = data["aggregated"]
                for i, name in enumerate(EXECUTION_FEATURE_NAMES):
                    if i < len(agg):
                        all_features[name].append(agg[i])
        except (json.JSONDecodeError, IOError):
            continue

    print(f"Analyzed {task_count} tasks\n")

    print("Feature distributions:")
    print("-" * 60)
    for name in EXECUTION_FEATURE_NAMES:
        values = all_features[name]
        if values:
            arr = np.array(values)
            print(f"{name}:")
            print(f"  mean={arr.mean():.3f}, std={arr.std():.3f}")
            print(f"  min={arr.min():.3f}, max={arr.max():.3f}")
            print(f"  25%={np.percentile(arr, 25):.3f}, "
                  f"50%={np.percentile(arr, 50):.3f}, "
                  f"75%={np.percentile(arr, 75):.3f}")
            print()


def main():
    parser = argparse.ArgumentParser(description="Compute execution features")
    parser.add_argument("--dry_run", action="store_true",
                        help="Show plan without computing")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of tasks to process")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze existing feature distributions")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip tasks that already have features")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.analyze:
        analyze_features(OUTPUT_DIR)
        return

    # Load config and create splits
    config = ExperimentConfig()
    print("Creating experiment splits...")
    split = create_experiment_split(
        responses_path=ROOT / config.responses_path,
        trajectories_dir=ROOT / config.trajectories_dir,
        weak_threshold=config.weak_threshold,
        strong_min_improvement=config.strong_min_improvement,
        m1_fraction=config.m1_fraction,
        m2_fraction=config.m2_fraction,
    )

    # Collect agent-task pairs to process
    pairs_to_compute = []

    # M1 agents × D_train tasks (for posterior training)
    for task_id in split.d_train_tasks:
        for agent in split.m1_agents:
            pairs_to_compute.append((task_id, agent, "train"))

    # M2 agents × D_valid tasks (for posterior evaluation)
    for task_id in split.d_valid_tasks:
        for agent in split.m2_agents:
            pairs_to_compute.append((task_id, agent, "valid"))

    print(f"\nTotal agent-task pairs: {len(pairs_to_compute)}")
    print(f"  M1 agents: {len(split.m1_agents)}")
    print(f"  M2 agents: {len(split.m2_agents)}")
    print(f"  D_train tasks: {len(split.d_train_tasks)}")
    print(f"  D_valid tasks: {len(split.d_valid_tasks)}")

    # Group by task for batch processing
    task_agents = {}
    for task_id, agent, split_type in pairs_to_compute:
        if task_id not in task_agents:
            task_agents[task_id] = {"agents": [], "split": split_type}
        task_agents[task_id]["agents"].append(agent)

    print(f"\nUnique tasks to process: {len(task_agents)}")

    if args.limit > 0:
        task_ids = list(task_agents.keys())[:args.limit]
        task_agents = {tid: task_agents[tid] for tid in task_ids}
        print(f"Limited to {len(task_agents)} tasks")

    if args.dry_run:
        print("\n=== DRY RUN - Would process: ===")
        for i, (task_id, info) in enumerate(list(task_agents.items())[:10]):
            print(f"  {task_id}: {len(info['agents'])} agents ({info['split']})")
        if len(task_agents) > 10:
            print(f"  ... and {len(task_agents) - 10} more tasks")
        return

    # Load gold patches
    patches = load_swebench_patches()

    # Process tasks
    print(f"\nComputing execution features...")
    start_time = datetime.now()

    processed = 0
    skipped = 0

    for task_id, info in task_agents.items():
        output_file = OUTPUT_DIR / f"{task_id}.json"

        if args.skip_existing and output_file.exists():
            skipped += 1
            continue

        # Compute features
        gold_patch = patches.get(task_id)
        agent_features = compute_features_for_task(
            task_id, info["agents"], gold_patch
        )

        if not agent_features:
            continue

        # Aggregate features
        features_dict = {}
        for agent, feat_dict in agent_features.items():
            features_dict[agent] = MechanicalFeatures(
                syntax_error_count=feat_dict["syntax_error_count"],
                test_run_count=feat_dict["test_run_count"],
                traceback_count=feat_dict["traceback_count"],
                unique_files_edited=set(feat_dict["unique_files_edited"]),
                total_commands=feat_dict["total_commands"],
                edit_attempts=feat_dict["edit_attempts"],
            )

        aggregated = aggregate_mechanical_features(features_dict)

        # Save results
        result = {
            "task_id": task_id,
            "split": info["split"],
            "n_agents": len(agent_features),
            "per_agent": agent_features,
            "aggregated": aggregated.tolist(),
            "feature_names": MECHANICAL_FEATURE_NAMES,
        }

        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)

        processed += 1
        if processed % 20 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"  Processed {processed}/{len(task_agents)} tasks "
                  f"({elapsed:.1f}s elapsed)")

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\nDone! Processed {processed} tasks, skipped {skipped}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

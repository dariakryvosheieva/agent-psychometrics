"""Compute test progression features for all trajectories.

This script extracts test progression features from agent trajectories.
No LLM calls needed - purely regex-based parsing.

Usage:
    # Dry run to see execution plan
    python -m experiment_b.test_progression.compute_features --dry_run

    # Run on small subset for validation
    python -m experiment_b.test_progression.compute_features --limit 100

    # Full run
    python -m experiment_b.test_progression.compute_features

    # Analyze feature distributions
    python -m experiment_b.test_progression.compute_features --analyze
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Add parent to path for imports
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_b.test_progression.features import (
    TEST_PROGRESSION_FEATURE_NAMES,
    compute_test_progression_features,
    extract_test_progression,
    features_to_dict,
)
from experiment_b.test_progression.aggregator import compute_coverage_stats


# Directories
UNIFIED_TRAJS_DIR = ROOT / "trajectory_data" / "unified_trajs"
OUTPUT_DIR = ROOT / "chris_output" / "experiment_b" / "test_progression_features"


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


def get_all_agent_task_pairs() -> List[tuple]:
    """Get all available agent-task pairs from trajectory directory.

    Returns:
        List of (agent, task_id) tuples
    """
    pairs = []

    if not UNIFIED_TRAJS_DIR.exists():
        print(f"Warning: Trajectories directory not found: {UNIFIED_TRAJS_DIR}")
        return pairs

    for agent_dir in UNIFIED_TRAJS_DIR.iterdir():
        if not agent_dir.is_dir():
            continue

        agent = agent_dir.name
        for task_file in agent_dir.glob("*.json"):
            task_id = task_file.stem
            pairs.append((agent, task_id))

    return pairs


def compute_features_for_trajectory(
    agent: str,
    task_id: str,
) -> Optional[Dict]:
    """Compute test progression features for a single trajectory.

    Args:
        agent: Agent name
        task_id: Task instance ID

    Returns:
        Feature dict or None if trajectory not found
    """
    trajectory = load_trajectory(agent, task_id)
    if trajectory is None:
        return None

    progression = extract_test_progression(trajectory)
    features = compute_test_progression_features(progression)

    # Convert to dict for JSON
    result = features_to_dict(features)
    result["task_id"] = task_id
    result["agent"] = agent
    result["resolved"] = progression.resolved

    return result


def analyze_features(output_dir: Path):
    """Analyze computed feature distributions.

    Args:
        output_dir: Directory containing computed features
    """
    print("\n=== Test Progression Feature Analysis ===\n")

    all_features: Dict[str, List] = {name: [] for name in TEST_PROGRESSION_FEATURE_NAMES}
    total_count = 0
    has_test_output_count = 0
    has_granular_count = 0
    multi_run_count = 0
    agent_counts: Dict[str, int] = {}
    framework_counts: Dict[str, int] = {}

    # Load all feature files
    for agent_dir in output_dir.iterdir():
        if not agent_dir.is_dir():
            continue

        agent = agent_dir.name
        agent_counts[agent] = 0

        for task_file in agent_dir.glob("*.json"):
            try:
                with open(task_file) as f:
                    data = json.load(f)

                total_count += 1
                agent_counts[agent] += 1

                if data.get("has_test_output", False):
                    has_test_output_count += 1
                if data.get("has_granular_data", False):
                    has_granular_count += 1
                if data.get("num_test_runs", 0) >= 2:
                    multi_run_count += 1

                framework = data.get("framework", "unknown")
                framework_counts[framework] = framework_counts.get(framework, 0) + 1

                # Collect numeric features
                for name in TEST_PROGRESSION_FEATURE_NAMES:
                    if name in data:
                        val = data[name]
                        if isinstance(val, (int, float)):
                            all_features[name].append(val)
                        elif isinstance(val, bool):
                            all_features[name].append(1.0 if val else 0.0)

            except (json.JSONDecodeError, IOError):
                continue

    print(f"Analyzed {total_count} agent-task pairs")
    print(f"Agents: {len(agent_counts)}")
    print()

    print("Coverage statistics:")
    print("-" * 60)
    print(f"  Has test output: {has_test_output_count} ({100*has_test_output_count/total_count:.1f}%)")
    print(f"  Has granular data: {has_granular_count} ({100*has_granular_count/total_count:.1f}%)")
    print(f"  Multi-run (>=2): {multi_run_count} ({100*multi_run_count/total_count:.1f}%)")
    print()

    print("Framework distribution:")
    print("-" * 60)
    for fw, count in sorted(framework_counts.items(), key=lambda x: -x[1]):
        print(f"  {fw}: {count} ({100*count/total_count:.1f}%)")
    print()

    print("Feature distributions:")
    print("-" * 60)
    for name in TEST_PROGRESSION_FEATURE_NAMES:
        values = all_features[name]
        if values:
            arr = np.array(values)
            print(f"{name}:")
            print(f"  n={len(values)}, mean={arr.mean():.3f}, std={arr.std():.3f}")
            print(f"  min={arr.min():.3f}, median={np.median(arr):.3f}, max={arr.max():.3f}")
            print()


def main():
    parser = argparse.ArgumentParser(description="Compute test progression features")
    parser.add_argument("--dry_run", action="store_true",
                        help="Show plan without computing")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of trajectories to process")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze existing feature distributions")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                        help="Skip pairs that already have features (default: True)")
    parser.add_argument("--force", action="store_true",
                        help="Force recompute all features")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.analyze:
        analyze_features(OUTPUT_DIR)
        return

    # Get all available trajectories
    print("Scanning trajectory directory...")
    pairs = get_all_agent_task_pairs()
    print(f"Found {len(pairs)} agent-task pairs")

    # Filter out existing if requested
    if args.skip_existing and not args.force:
        filtered_pairs = []
        for agent, task_id in pairs:
            output_file = OUTPUT_DIR / agent / f"{task_id}.json"
            if not output_file.exists():
                filtered_pairs.append((agent, task_id))
        print(f"After skipping existing: {len(filtered_pairs)} pairs to process")
        pairs = filtered_pairs

    if args.limit > 0:
        pairs = pairs[:args.limit]
        print(f"Limited to {len(pairs)} pairs")

    if args.dry_run:
        print(f"\n=== DRY RUN - Would process {len(pairs)} pairs ===")

        # Show sample
        for i, (agent, task_id) in enumerate(pairs[:10]):
            print(f"  {agent}/{task_id}")
        if len(pairs) > 10:
            print(f"  ... and {len(pairs) - 10} more pairs")

        # Show unique agents
        agents = set(agent for agent, _ in pairs)
        print(f"\nUnique agents: {len(agents)}")

        return

    # Process pairs
    print(f"\nComputing test progression features...")
    start_time = datetime.now()

    processed = 0
    skipped = 0
    has_test_output = 0
    has_granular = 0

    for agent, task_id in pairs:
        # Create agent directory
        agent_dir = OUTPUT_DIR / agent
        agent_dir.mkdir(parents=True, exist_ok=True)
        output_file = agent_dir / f"{task_id}.json"

        if args.skip_existing and not args.force and output_file.exists():
            skipped += 1
            continue

        # Compute features
        result = compute_features_for_trajectory(agent, task_id)

        if result is None:
            skipped += 1
            continue

        # Track coverage
        if result.get("has_test_output", False):
            has_test_output += 1
        if result.get("has_granular_data", False):
            has_granular += 1

        # Save
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)

        processed += 1

        if processed % 1000 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = processed / elapsed if elapsed > 0 else 0
            print(f"  Processed {processed}/{len(pairs)} ({rate:.0f}/s)")

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\nDone! Processed {processed}, skipped {skipped}")
    print(f"  Has test output: {has_test_output} ({100*has_test_output/max(processed,1):.1f}%)")
    print(f"  Has granular data: {has_granular} ({100*has_granular/max(processed,1):.1f}%)")
    print(f"Total time: {elapsed:.1f}s ({processed/elapsed:.0f} trajectories/s)")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

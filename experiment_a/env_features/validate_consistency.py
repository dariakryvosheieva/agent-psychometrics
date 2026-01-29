"""Validate consistency of environment feature extraction across runs.

Compares the metadata from two evaluation runs and reports any differences.
All features should be identical across runs since they are deterministic.

Usage:
    python -m experiment_a.env_features.validate_consistency \\
        chris_output/env_features/validation_run1/ \\
        chris_output/env_features/validation_run2/
"""

import argparse
import sys
from pathlib import Path
from typing import Any

from inspect_ai.log import read_eval_log

from experiment_a.env_features.feature_definitions import get_feature_names


def find_log_files(log_dir: Path) -> list[Path]:
    """Find all evaluation log files in a directory."""
    # Inspect uses .eval files (binary msgpack format)
    log_files = list(log_dir.glob("*.eval")) + list(log_dir.glob("logs/*.eval"))
    return sorted(log_files)


def load_sample_metadata(log_path: Path) -> dict[str, dict[str, Any]]:
    """Load sample metadata from an Inspect log file.

    Returns:
        Dict mapping sample_id -> metadata dict
    """
    # Use Inspect's API to read the binary log format
    log = read_eval_log(str(log_path))

    result = {}
    for sample in log.samples or []:
        sample_id = sample.id
        if sample_id:
            # Get metadata from the sample's metadata dict
            metadata = dict(sample.metadata) if sample.metadata else {}
            result[str(sample_id)] = metadata

    return result


def compare_metadata(
    run1_metadata: dict[str, dict[str, Any]],
    run2_metadata: dict[str, dict[str, Any]],
    feature_names: list[str],
) -> tuple[bool, list[str]]:
    """Compare metadata from two runs.

    Args:
        run1_metadata: Sample metadata from run 1
        run2_metadata: Sample metadata from run 2
        feature_names: List of feature names to compare

    Returns:
        (all_match, list of difference messages)
    """
    differences: list[str] = []

    # Check for sample ID mismatches
    run1_ids = set(run1_metadata.keys())
    run2_ids = set(run2_metadata.keys())

    if run1_ids != run2_ids:
        only_in_1 = run1_ids - run2_ids
        only_in_2 = run2_ids - run1_ids
        if only_in_1:
            differences.append(f"Samples only in run1: {only_in_1}")
        if only_in_2:
            differences.append(f"Samples only in run2: {only_in_2}")

    # Compare features for common samples
    common_ids = run1_ids & run2_ids
    for sample_id in sorted(common_ids):
        meta1 = run1_metadata[sample_id]
        meta2 = run2_metadata[sample_id]

        for feature_name in feature_names:
            val1 = meta1.get(feature_name)
            val2 = meta2.get(feature_name)

            if val1 != val2:
                differences.append(
                    f"{sample_id}/{feature_name}: {val1} != {val2}"
                )

    all_match = len(differences) == 0
    return all_match, differences


def main():
    parser = argparse.ArgumentParser(
        description="Validate consistency of environment features across two runs"
    )
    parser.add_argument(
        "run1_dir",
        type=Path,
        help="Directory containing logs from first run",
    )
    parser.add_argument(
        "run2_dir",
        type=Path,
        help="Directory containing logs from second run",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show all compared values",
    )
    args = parser.parse_args()

    # Find log files
    run1_logs = find_log_files(args.run1_dir)
    run2_logs = find_log_files(args.run2_dir)

    if not run1_logs:
        print(f"ERROR: No log files found in {args.run1_dir}")
        sys.exit(1)
    if not run2_logs:
        print(f"ERROR: No log files found in {args.run2_dir}")
        sys.exit(1)

    print(f"Run 1: Found {len(run1_logs)} log file(s) in {args.run1_dir}")
    print(f"Run 2: Found {len(run2_logs)} log file(s) in {args.run2_dir}")

    # Load metadata from all logs
    run1_metadata: dict[str, dict[str, Any]] = {}
    run2_metadata: dict[str, dict[str, Any]] = {}

    for log_path in run1_logs:
        run1_metadata.update(load_sample_metadata(log_path))

    for log_path in run2_logs:
        run2_metadata.update(load_sample_metadata(log_path))

    print(f"\nRun 1: {len(run1_metadata)} samples")
    print(f"Run 2: {len(run2_metadata)} samples")

    # Get feature names to compare
    feature_names = get_feature_names()
    print(f"\nComparing {len(feature_names)} features...")

    # Compare
    all_match, differences = compare_metadata(
        run1_metadata, run2_metadata, feature_names
    )

    if args.verbose and all_match:
        print("\nAll values match:")
        for sample_id in sorted(run1_metadata.keys()):
            print(f"\n  {sample_id}:")
            for feature_name in feature_names:
                val = run1_metadata[sample_id].get(feature_name)
                print(f"    {feature_name}: {val}")

    # Report results
    if all_match:
        print("\n" + "=" * 60)
        print("SUCCESS: All features match between runs!")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print(f"FAILURE: {len(differences)} difference(s) found!")
        print("=" * 60)
        for diff in differences[:50]:  # Limit output
            print(f"  - {diff}")
        if len(differences) > 50:
            print(f"  ... and {len(differences) - 50} more")
        sys.exit(1)


if __name__ == "__main__":
    main()

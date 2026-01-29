"""Run environment feature extraction with batching and Docker cleanup.

Processes SWE-bench tasks in batches to prevent memory overflow, cleaning
Docker state between batches.

Usage:
    # Run with default settings (batch_size=10, max_connections=10)
    python -m experiment_a.env_features.run_extraction

    # Custom batch size and parallelism
    python -m experiment_a.env_features.run_extraction --batch_size 5 --max_connections 5

    # Just aggregate existing logs to CSV (skip extraction)
    python -m experiment_a.env_features.run_extraction --aggregate_only
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset
from inspect_ai.log import read_eval_log

from experiment_a.env_features.feature_definitions import get_feature_names


def get_swebench_images() -> list[str]:
    """Get list of currently loaded SWE-bench eval images."""
    result = subprocess.run(
        ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []

    images = []
    for line in result.stdout.strip().split("\n"):
        # Match any swebench image (eval or env)
        if line and line.startswith("swebench/"):
            images.append(line)
    return images


def run_docker_cleanup(remove_images: bool = True):
    """Clean up Docker state to free memory.

    Args:
        remove_images: If True, remove SWE-bench eval images entirely (frees ~500MB per image)
    """
    print("\n--- Cleaning Docker state ---")

    # First, stop any running containers using swebench images
    result = subprocess.run(
        ["docker", "ps", "-q", "--filter", "ancestor=swebench"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        container_ids = result.stdout.strip().split("\n")
        print(f"  Stopping {len(container_ids)} running containers...")
        subprocess.run(["docker", "stop"] + container_ids, capture_output=True)

    # Remove stopped containers
    result = subprocess.run(
        ["docker", "container", "prune", "-f"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        print(f"  Containers pruned")

    # Remove unused volumes
    result = subprocess.run(
        ["docker", "volume", "prune", "-f"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        print(f"  Volumes pruned")

    # Remove SWE-bench images entirely
    if remove_images:
        swebench_images = get_swebench_images()
        if swebench_images:
            print(f"  Removing {len(swebench_images)} SWE-bench images...")
            for image in swebench_images:
                result = subprocess.run(
                    ["docker", "rmi", "-f", image],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    print(f"    Warning: Could not remove {image}: {result.stderr.strip()}")

            # Verify removal
            remaining = get_swebench_images()
            if remaining:
                print(f"  WARNING: {len(remaining)} images still remain!")
                for img in remaining:
                    print(f"    - {img}")
            else:
                print(f"  Successfully removed all {len(swebench_images)} images")
        else:
            print("  No SWE-bench images to remove")

    # Final prune to clean up any dangling layers
    subprocess.run(
        ["docker", "system", "prune", "-f"],
        capture_output=True,
        text=True,
    )

    print("--- Docker cleanup complete ---\n")


def run_inspect_batch(
    batch_num: int,
    sample_ids: list[str],
    max_connections: int,
    log_dir: Path,
) -> bool:
    """Run Inspect evaluation for a batch of tasks.

    Args:
        batch_num: Batch number for logging
        sample_ids: List of instance IDs to process in this batch
        max_connections: Max parallel containers
        log_dir: Directory to store logs

    Returns:
        True if successful, False otherwise
    """
    batch_log_dir = log_dir / f"batch_{batch_num:03d}"
    batch_log_dir.mkdir(parents=True, exist_ok=True)

    # Use --sample-id to specify exactly which samples to run
    # This ensures we only download/run the images for this batch
    # Use @env_feature_extraction to explicitly select the full dataset task
    sample_ids_str = ",".join(sample_ids)
    cmd = [
        "inspect",
        "eval",
        "experiment_a/env_features/inspect_task.py@env_feature_extraction",
        f"--max-connections={max_connections}",
        f"--log-dir={batch_log_dir}",
        f"--sample-id={sample_ids_str}",
    ]

    print(f"\n=== Batch {batch_num}: Running {len(sample_ids)} tasks ===")
    print(f"Sample IDs: {sample_ids[0]}...{sample_ids[-1]}")

    # Run from project root (3 levels up from this file)
    project_root = Path(__file__).parent.parent.parent
    result = subprocess.run(cmd, cwd=project_root)

    if result.returncode != 0:
        print(f"WARNING: Batch {batch_num} returned non-zero exit code: {result.returncode}")
        return False

    return True


def load_all_metadata(log_dir: Path) -> dict[str, dict[str, Any]]:
    """Load metadata from all log files in a directory."""
    all_metadata: dict[str, dict[str, Any]] = {}

    # Find all .eval log files (binary msgpack format)
    log_files = list(log_dir.rglob("*.eval"))
    print(f"Found {len(log_files)} log files")

    for log_path in log_files:
        try:
            # Use Inspect's API to read the binary log format
            log = read_eval_log(str(log_path))

            for sample in log.samples or []:
                sample_id = sample.id
                if sample_id:
                    # Get metadata from the sample's metadata dict
                    metadata = dict(sample.metadata) if sample.metadata else {}
                    all_metadata[str(sample_id)] = metadata

        except Exception as e:
            print(f"  Warning: Could not load {log_path}: {e}")

    return all_metadata


def aggregate_to_csv(log_dir: Path, output_path: Path) -> pd.DataFrame:
    """Aggregate all log metadata into a CSV file.

    Args:
        log_dir: Directory containing Inspect log files
        output_path: Path to write CSV output

    Returns:
        DataFrame with aggregated features
    """
    print(f"\n=== Aggregating logs from {log_dir} ===")

    # Load all metadata
    all_metadata = load_all_metadata(log_dir)
    print(f"Loaded metadata for {len(all_metadata)} samples")

    if not all_metadata:
        raise ValueError(f"No samples found in {log_dir}")

    # Get feature names
    feature_names = get_feature_names()

    # Build DataFrame
    rows = []
    for sample_id, metadata in sorted(all_metadata.items()):
        row = {"instance_id": sample_id}
        for feature_name in feature_names:
            row[feature_name] = metadata.get(feature_name, -1)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Set instance_id as index
    df = df.set_index("instance_id")

    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    print(f"Saved {len(df)} rows to {output_path}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Run environment feature extraction with batching"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Number of tasks per batch (default: 10)",
    )
    parser.add_argument(
        "--max_connections",
        type=int,
        default=10,
        help="Max parallel containers per batch (default: 10)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("chris_output/env_features/swebench_verified"),
        help="Output directory for logs and CSV",
    )
    parser.add_argument(
        "--aggregate_only",
        action="store_true",
        help="Skip extraction, just aggregate existing logs to CSV",
    )
    parser.add_argument(
        "--no_cleanup",
        action="store_true",
        help="Skip Docker cleanup between batches",
    )
    parser.add_argument(
        "--keep_images",
        action="store_true",
        help="Don't remove Docker images during cleanup (only prune containers/volumes)",
    )
    args = parser.parse_args()

    log_dir = args.output_dir / "logs"
    csv_path = args.output_dir / "env_features.csv"

    if args.aggregate_only:
        # Just aggregate existing logs
        df = aggregate_to_csv(log_dir, csv_path)
        print(f"\nDone! {len(df)} samples aggregated to {csv_path}")
        return

    # Load dataset to get all instance IDs
    print("Loading dataset to get instance IDs...")
    dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    all_instance_ids = [row["instance_id"] for row in dataset]
    total_tasks = len(all_instance_ids)

    # Check for already-completed samples to support resuming
    existing_metadata = load_all_metadata(log_dir)
    completed_ids = set(existing_metadata.keys())
    remaining_ids = [iid for iid in all_instance_ids if iid not in completed_ids]

    if completed_ids:
        print(f"Found {len(completed_ids)} already completed, {len(remaining_ids)} remaining")

    # Calculate number of batches for remaining tasks
    num_batches = (len(remaining_ids) + args.batch_size - 1) // args.batch_size
    print(f"Processing {len(remaining_ids)} tasks in {num_batches} batches of {args.batch_size}")
    print(f"Parallelism: {args.max_connections} containers per batch")
    print(f"Output directory: {args.output_dir}")
    if not args.keep_images:
        print("Docker images will be removed after each batch to save disk space")

    # Run batches with specific sample IDs
    try:
        for batch_num in range(num_batches):
            start_idx = batch_num * args.batch_size
            end_idx = min(start_idx + args.batch_size, len(remaining_ids))
            batch_ids = remaining_ids[start_idx:end_idx]

            success = run_inspect_batch(
                batch_num=batch_num,
                sample_ids=batch_ids,
                max_connections=args.max_connections,
                log_dir=log_dir,
            )

            # ALWAYS clean up Docker after each batch, even if it failed
            if not args.no_cleanup:
                run_docker_cleanup(remove_images=not args.keep_images)

            if not success:
                print(f"Batch {batch_num} failed, stopping")
                break

            # Check progress
            metadata = load_all_metadata(log_dir)
            print(f"Progress: {len(metadata)}/{total_tasks} samples completed")

            if len(metadata) >= total_tasks:
                print("All samples completed!")
                break
    finally:
        # Final cleanup to ensure no images are left behind
        if not args.no_cleanup:
            print("\n=== Final Docker cleanup ===")
            run_docker_cleanup(remove_images=not args.keep_images)

    # Aggregate results
    df = aggregate_to_csv(log_dir, csv_path)
    print(f"\nDone! {len(df)} samples extracted to {csv_path}")


if __name__ == "__main__":
    main()

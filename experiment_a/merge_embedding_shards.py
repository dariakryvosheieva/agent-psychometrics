#!/usr/bin/env python3
"""Merge sharded embedding files from parallel SLURM jobs.

When embeddings are generated in parallel across multiple GPUs using
task sharding (--start_idx, --n_inputs), this script merges the
resulting .npz files into a single file.

Example usage:
    python -m experiment_a.merge_embedding_shards \
        --shard_pattern "chris_output/experiment_a/embeddings/embeddings__*__shard*.npz" \
        --out_path chris_output/experiment_a/embeddings/embeddings__merged.npz

Or with explicit shard files:
    python -m experiment_a.merge_embedding_shards \
        --shard_files shard0.npz shard1.npz \
        --out_path merged.npz
"""

import argparse
import sys
from glob import glob
from pathlib import Path
from typing import List, Optional

import numpy as np


def load_and_validate_shards(shard_paths: List[Path]) -> List[dict]:
    """Load shards and validate they have compatible metadata.

    Args:
        shard_paths: List of paths to shard .npz files

    Returns:
        List of loaded shard data dicts

    Raises:
        ValueError: If shards have incompatible metadata
    """
    shards = []
    reference_metadata = None

    for path in shard_paths:
        data = dict(np.load(path, allow_pickle=True))

        # Extract metadata for validation
        metadata = {
            "backbone": str(data["backbone"][0]) if "backbone" in data else None,
            "max_length": int(data["max_length"][0]) if "max_length" in data else None,
            "embedding_dim": int(data["embedding_dim"][0]) if "embedding_dim" in data else None,
            "embedding_layer": int(data["embedding_layer"][0]) if "embedding_layer" in data else None,
            "dataset_name": str(data["dataset_name"][0]) if "dataset_name" in data else None,
            "split": str(data["split"][0]) if "split" in data else None,
        }

        if reference_metadata is None:
            reference_metadata = metadata
        else:
            # Validate compatibility
            for key in ["backbone", "max_length", "embedding_dim", "embedding_layer", "dataset_name", "split"]:
                if metadata[key] != reference_metadata[key]:
                    raise ValueError(
                        f"Incompatible shards: {key} mismatch. "
                        f"Expected {reference_metadata[key]}, got {metadata[key]} in {path}"
                    )

        shards.append(data)
        print(f"Loaded shard: {path} ({len(data['task_ids'])} tasks)")

    return shards


def merge_shards(shards: List[dict]) -> dict:
    """Merge multiple shard dicts into a single embedding dict.

    Args:
        shards: List of shard data dicts from load_and_validate_shards

    Returns:
        Merged data dict ready for np.savez_compressed
    """
    # Concatenate task_ids and embeddings
    all_task_ids = []
    all_embeddings = []

    for shard in shards:
        all_task_ids.extend(shard["task_ids"])
        all_embeddings.append(shard["X"])

    merged_task_ids = np.array(all_task_ids, dtype=object)
    merged_X = np.vstack(all_embeddings).astype(np.float32)

    # Check for duplicates
    unique_ids = set(merged_task_ids)
    if len(unique_ids) != len(merged_task_ids):
        duplicates = [tid for tid in merged_task_ids if list(merged_task_ids).count(tid) > 1]
        raise ValueError(f"Duplicate task IDs found across shards: {set(duplicates)}")

    # Use metadata from first shard
    first_shard = shards[0]

    merged = {
        "task_ids": merged_task_ids,
        "X": merged_X,
        "backbone": first_shard.get("backbone", np.array(["unknown"], dtype=object)),
        "max_length": first_shard.get("max_length", np.array([0], dtype=np.int64)),
        "embedding_dim": np.array([merged_X.shape[1]], dtype=np.int64),
        "embedding_layer": first_shard.get("embedding_layer", np.array([-1], dtype=np.int64)),
        "instruction": first_shard.get("instruction", np.array([""], dtype=object)),
        "dataset_name": first_shard.get("dataset_name", np.array([""], dtype=object)),
        "split": first_shard.get("split", np.array([""], dtype=object)),
        # Update counts to reflect merged data
        "start_idx": np.array([0], dtype=np.int64),
        "n_inputs": np.array([len(merged_task_ids)], dtype=np.int64),
    }

    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Merge sharded embedding files from parallel SLURM jobs"
    )
    parser.add_argument(
        "--shard_pattern",
        type=str,
        help="Glob pattern to match shard files (e.g., 'embeddings__*__shard*.npz')",
    )
    parser.add_argument(
        "--shard_files",
        type=str,
        nargs="+",
        help="Explicit list of shard files to merge",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Output path for merged embeddings",
    )
    args = parser.parse_args()

    # Get shard paths
    if args.shard_pattern:
        shard_paths = sorted(glob(args.shard_pattern))
        if not shard_paths:
            print(f"No files matched pattern: {args.shard_pattern}")
            sys.exit(1)
    elif args.shard_files:
        shard_paths = args.shard_files
    else:
        print("Must specify either --shard_pattern or --shard_files")
        sys.exit(1)

    shard_paths = [Path(p) for p in shard_paths]
    print(f"Found {len(shard_paths)} shard files to merge")

    # Load and validate
    shards = load_and_validate_shards(shard_paths)

    # Merge
    merged = merge_shards(shards)
    print(f"Merged: {len(merged['task_ids'])} total tasks, dim={merged['X'].shape[1]}")

    # Save
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **merged)
    print(f"Saved merged embeddings to: {out_path}")


if __name__ == "__main__":
    main()

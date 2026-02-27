"""Run V4 auditor agent with batching, Docker cleanup, and incremental CSV resumability.

Supports multiple datasets: SWE-bench Verified, SWE-bench Pro, Terminal Bench, GSO.
After each batch, results are immediately appended to an incremental CSV so that
the process can resume from where it left off if interrupted (e.g., spot instance
termination).

Usage:
    # Run on SWE-bench Verified (default)
    python -m experiment_a.auditor_agent.run_auditor --dataset swebench

    # Run on Terminal Bench
    python -m experiment_a.auditor_agent.run_auditor --dataset terminalbench

    # Run on GSO with small batch for testing
    python -m experiment_a.auditor_agent.run_auditor --dataset gso --limit 2

    # Custom batch size and model
    python -m experiment_a.auditor_agent.run_auditor --dataset swebench --batch_size 5

    # Just aggregate existing logs to CSV (skip running)
    python -m experiment_a.auditor_agent.run_auditor --dataset swebench --aggregate_only
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset

# Add project root to path
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Reuse Docker utilities from env_features
from experiment_a.env_features.run_extraction import run_docker_cleanup
from experiment_a.auditor_agent.parse_outputs import (
    parse_all_logs,
    validate_results,
    EXPECTED_FEATURES_V4,
)

# Dataset configurations: maps dataset name to inspect task path, ID source, etc.
DATASET_CONFIGS = {
    "swebench": {
        "inspect_task": "experiment_a/auditor_agent/inspect_task.py@auditor_task_v4",
        "hf_dataset": "princeton-nlp/SWE-bench_Verified",
        "id_field": "instance_id",
        "log_dir_name": "swebench_verified_v4",
    },
    "swebench_pro": {
        "inspect_task": "experiment_a/swebench_pro/inspect_task.py@auditor_task_v4_swebench_pro",
        "hf_dataset": "ScaleAI/SWE-bench_Pro",
        "id_field": "instance_id",
        "log_dir_name": "swebench_pro_v4",
    },
    "terminalbench": {
        "inspect_task": "experiment_a/terminalbench/inspect_task.py@auditor_task_v4_terminalbench",
        "items_csv": "chris_output/terminal_bench_2.0_binomial_1pl/1d/items.csv",
        "log_dir_name": "terminalbench_v4",
    },
    "gso": {
        "inspect_task": "experiment_a/gso/inspect_task.py@auditor_task_v4_gso",
        "hf_dataset": "gso-bench/gso",
        "id_field": "instance_id",
        "log_dir_name": "gso_v4",
    },
}

INCREMENTAL_CSV_NAME = "auditor_features_incremental.csv"


def get_all_instance_ids(dataset: str) -> list[str]:
    """Get all instance IDs for a dataset.

    Args:
        dataset: Dataset name (one of DATASET_CONFIGS keys).

    Returns:
        List of instance ID strings.

    Raises:
        FileNotFoundError: If a local items CSV doesn't exist.
        ValueError: If no ID source is configured for the dataset.
    """
    config = DATASET_CONFIGS[dataset]

    if "hf_dataset" in config:
        print(f"Loading {config['hf_dataset']} dataset...")
        ds = load_dataset(config["hf_dataset"], split="test")
        return [sample[config["id_field"]] for sample in ds]
    elif "items_csv" in config:
        items_path = Path(config["items_csv"])
        if not items_path.exists():
            raise FileNotFoundError(f"Items file not found: {items_path}")
        items_df = pd.read_csv(items_path, index_col=0)
        return list(items_df.index.astype(str))
    else:
        raise ValueError(f"No ID source configured for dataset: {dataset}")


def get_completed_ids_from_csv(log_dir: Path) -> set[str]:
    """Get completed instance IDs from the incremental CSV.

    Only counts an instance as completed if ALL 8 V4 features have valid values.
    If any feature is missing, the instance will be re-extracted.

    Args:
        log_dir: Directory containing the incremental CSV.

    Returns:
        Set of completed instance ID strings.
    """
    incremental_csv = log_dir / INCREMENTAL_CSV_NAME
    if not incremental_csv.exists():
        return set()

    df = pd.read_csv(incremental_csv)
    if "instance_id" not in df.columns:
        return set()

    # Only count rows where ALL features have valid values
    feature_cols = [c for c in EXPECTED_FEATURES_V4 if c in df.columns]
    if len(feature_cols) < len(EXPECTED_FEATURES_V4):
        # Some feature columns are missing entirely — nothing is complete
        return set()

    valid = df.dropna(subset=feature_cols, how="any")
    return set(valid["instance_id"].astype(str))


def append_batch_to_incremental_csv(
    batch_log_dir: Path,
    log_dir: Path,
    expected_features: list[str],
) -> None:
    """Parse a batch's logs and append results to the incremental CSV.

    Args:
        batch_log_dir: Directory containing .eval files for one batch.
        log_dir: Parent log directory (where incremental CSV lives).
        expected_features: Feature names to extract.
    """
    batch_df = parse_all_logs(batch_log_dir, expected_features)
    if batch_df.empty:
        return

    incremental_csv = log_dir / INCREMENTAL_CSV_NAME
    if incremental_csv.exists():
        existing = pd.read_csv(incremental_csv)
        combined = pd.concat([existing, batch_df]).drop_duplicates(
            subset=["instance_id"], keep="last"
        )
        combined.to_csv(incremental_csv, index=False)
        print(f"Incremental CSV updated: {len(batch_df)} new, {len(combined)} total")
    else:
        batch_df.to_csv(incremental_csv, index=False)
        print(f"Incremental CSV created: {len(batch_df)} samples")


def run_inspect_batch(
    batch_num: int,
    sample_ids: list[str],
    model: str,
    max_connections: int,
    log_dir: Path,
    inspect_task: str,
) -> Path:
    """Run Inspect evaluation for a batch of tasks.

    Args:
        batch_num: Batch number for logging.
        sample_ids: List of instance IDs to process in this batch.
        model: Model to use for the auditor agent.
        max_connections: Max parallel containers.
        log_dir: Directory to store logs.
        inspect_task: Full inspect task path (e.g., "module.py@task_fn").

    Returns:
        Path to the batch log directory.
    """
    batch_log_dir = log_dir / f"batch_{batch_num:03d}"
    batch_log_dir.mkdir(parents=True, exist_ok=True)

    sample_ids_str = ",".join(sample_ids)
    cmd = [
        "inspect", "eval",
        inspect_task,
        f"--model={model}",
        f"--max-connections={max_connections}",
        f"--log-dir={batch_log_dir}",
        f"--sample-id={sample_ids_str}",
    ]

    print(f"\n=== Batch {batch_num}: Running {len(sample_ids)} tasks ===")
    print(f"Sample IDs: {sample_ids[0]}...{sample_ids[-1]}")
    print(f"Model: {model}, Max connections: {max_connections}")

    result = subprocess.run(cmd, cwd=_project_root)

    if result.returncode != 0:
        print(f"WARNING: Batch {batch_num} returned non-zero exit code: {result.returncode}")

    return batch_log_dir


def main():
    parser = argparse.ArgumentParser(
        description="Run V4 auditor agent with batching and Docker cleanup"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="swebench",
        choices=list(DATASET_CONFIGS.keys()),
        help="Dataset to run on (default: swebench)",
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
        help="Max parallel Docker containers (default: 10)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="anthropic/claude-opus-4-5-20251101",
        help="Model to use (default: anthropic/claude-opus-4-5-20251101)",
    )
    parser.add_argument(
        "--log_dir",
        type=Path,
        default=None,
        help="Directory to store logs (default: chris_output/auditor_features/{dataset})",
    )
    parser.add_argument(
        "--aggregate_only",
        action="store_true",
        help="Only aggregate existing logs to CSV (skip running)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N tasks (for testing)",
    )
    parser.add_argument(
        "--sample_ids",
        type=str,
        default=None,
        help="Comma-separated list of specific instance IDs to process",
    )
    parser.add_argument(
        "--sample_ids_file",
        type=Path,
        default=None,
        help="File containing instance IDs (one per line)",
    )
    parser.add_argument(
        "--skip_cleanup",
        action="store_true",
        help="Skip Docker cleanup between batches",
    )

    args = parser.parse_args()

    # Get dataset config
    config = DATASET_CONFIGS[args.dataset]
    expected_features = EXPECTED_FEATURES_V4

    # Set default log_dir based on dataset
    if args.log_dir is None:
        args.log_dir = Path("chris_output/auditor_features") / config["log_dir_name"]

    # Create log directory
    args.log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {args.dataset}")
    print(f"Inspect task: {config['inspect_task']}")
    print(f"Log directory: {args.log_dir}")

    if args.aggregate_only:
        print("=== Aggregating existing logs ===")
        df = parse_all_logs(args.log_dir, expected_features)
        if not df.empty:
            validate_results(df, expected_features)
            output_path = args.log_dir / "auditor_features.csv"
            df.to_csv(output_path, index=False)
            print(f"\nSaved to {output_path}")
        return

    # Determine which instance IDs to process
    if args.sample_ids:
        all_instance_ids = [s.strip() for s in args.sample_ids.split(",")]
        print(f"Using {len(all_instance_ids)} specified sample IDs")
    elif args.sample_ids_file:
        with open(args.sample_ids_file) as f:
            all_instance_ids = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(all_instance_ids)} sample IDs from {args.sample_ids_file}")
    else:
        all_instance_ids = get_all_instance_ids(args.dataset)

    if args.limit:
        all_instance_ids = all_instance_ids[:args.limit]

    print(f"Total tasks: {len(all_instance_ids)}")

    # Get already completed instance IDs from incremental CSV
    completed_ids = get_completed_ids_from_csv(args.log_dir)
    print(f"Already completed (from incremental CSV): {len(completed_ids)}")

    # Filter to remaining tasks
    remaining_ids = [iid for iid in all_instance_ids if iid not in completed_ids]
    print(f"Remaining tasks: {len(remaining_ids)}")

    if not remaining_ids:
        print("All tasks already completed!")
        # Final aggregation from all logs
        df = parse_all_logs(args.log_dir, expected_features)
        if not df.empty:
            validate_results(df, expected_features)
            output_path = args.log_dir / "auditor_features.csv"
            df.to_csv(output_path, index=False)
            print(f"\nSaved to {output_path}")
        return

    # Split into batches
    batches = [
        remaining_ids[i:i + args.batch_size]
        for i in range(0, len(remaining_ids), args.batch_size)
    ]
    print(f"Number of batches: {len(batches)}")

    # Run batches
    for batch_num, batch_ids in enumerate(batches):
        batch_log_dir = run_inspect_batch(
            batch_num,
            batch_ids,
            args.model,
            args.max_connections,
            args.log_dir,
            config["inspect_task"],
        )

        # Immediately append batch results to incremental CSV
        append_batch_to_incremental_csv(
            batch_log_dir, args.log_dir, expected_features
        )

        # Cleanup between batches
        if not args.skip_cleanup:
            run_docker_cleanup()

    # Final aggregation
    print("\n=== Final aggregation ===")
    df = parse_all_logs(args.log_dir, expected_features)
    if not df.empty:
        validate_results(df, expected_features)
        output_path = args.log_dir / "auditor_features.csv"
        df.to_csv(output_path, index=False)
        print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()

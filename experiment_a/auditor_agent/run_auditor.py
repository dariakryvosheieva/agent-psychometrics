"""Run auditor agent with batching and Docker cleanup.

Processes SWE-bench tasks in batches to prevent memory overflow, cleaning
Docker state between batches.

Usage:
    # Run on all tasks with default settings (batch_size=10, max_connections=10)
    python -m experiment_a.auditor_agent.run_auditor

    # Custom batch size and model
    python -m experiment_a.auditor_agent.run_auditor --batch_size 5 --max_connections 5

    # Just aggregate existing logs to CSV (skip running)
    python -m experiment_a.auditor_agent.run_auditor --aggregate_only

    # Resume from a specific batch
    python -m experiment_a.auditor_agent.run_auditor --resume_from_batch 10
"""

import argparse
import subprocess
import sys
from pathlib import Path

from datasets import load_dataset

# Add project root to path
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Reuse Docker utilities from env_features
from experiment_a.env_features.run_extraction import run_docker_cleanup
from experiment_a.auditor_agent.parse_outputs import parse_all_logs, validate_results


def run_inspect_batch(
    batch_num: int,
    sample_ids: list[str],
    model: str,
    max_connections: int,
    log_dir: Path,
    task_name: str = "auditor_task",
) -> bool:
    """Run Inspect evaluation for a batch of tasks.

    Args:
        batch_num: Batch number for logging
        sample_ids: List of instance IDs to process in this batch
        model: Model to use for the auditor agent
        max_connections: Max parallel containers
        log_dir: Directory to store logs
        task_name: Task function name (auditor_task or auditor_task_v2)

    Returns:
        True if successful, False otherwise
    """
    batch_log_dir = log_dir / f"batch_{batch_num:03d}"
    batch_log_dir.mkdir(parents=True, exist_ok=True)

    sample_ids_str = ",".join(sample_ids)
    cmd = [
        "inspect", "eval",
        f"experiment_a/auditor_agent/inspect_task.py@{task_name}",
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
        return False

    return True


def get_completed_instance_ids(log_dir: Path) -> set[str]:
    """Get set of instance IDs that have already been processed successfully."""
    completed = set()

    for log_path in log_dir.rglob("*.eval"):
        try:
            from inspect_ai.log import read_eval_log
            log = read_eval_log(str(log_path))
            for sample in log.samples or []:
                if sample.id:
                    # Check if sample has valid completion with all features
                    completion = sample.output.completion if sample.output else None
                    if completion and "test_runability" in completion and "test_feedback_quality" in completion:
                        completed.add(str(sample.id))
        except Exception:
            pass

    return completed


def main():
    parser = argparse.ArgumentParser(
        description="Run auditor agent with batching and Docker cleanup"
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
        default=Path("chris_output/auditor_features/swebench_verified"),
        help="Directory to store logs",
    )
    parser.add_argument(
        "--aggregate_only",
        action="store_true",
        help="Only aggregate existing logs to CSV (skip running)",
    )
    parser.add_argument(
        "--resume_from_batch",
        type=int,
        default=0,
        help="Resume from a specific batch number",
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
    parser.add_argument(
        "--task",
        type=str,
        default="auditor_task",
        choices=["auditor_task", "auditor_task_v2"],
        help="Task to run (default: auditor_task)",
    )

    args = parser.parse_args()

    # Create log directory
    args.log_dir.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        print("=== Aggregating existing logs ===")
        df = parse_all_logs(args.log_dir)
        if not df.empty:
            validate_results(df)
            output_path = args.log_dir / "auditor_features.csv"
            df.to_csv(output_path, index=False)
            print(f"\nSaved to {output_path}")
        return

    # Determine which instance IDs to process
    if args.sample_ids:
        # Use explicit comma-separated list
        all_instance_ids = [s.strip() for s in args.sample_ids.split(",")]
        print(f"Using {len(all_instance_ids)} specified sample IDs")
    elif args.sample_ids_file:
        # Read from file (one ID per line)
        with open(args.sample_ids_file) as f:
            all_instance_ids = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(all_instance_ids)} sample IDs from {args.sample_ids_file}")
    else:
        # Load dataset to get all instance IDs
        print("Loading SWE-bench Verified dataset...")
        ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
        all_instance_ids = [sample["instance_id"] for sample in ds]

    if args.limit:
        all_instance_ids = all_instance_ids[:args.limit]

    print(f"Total tasks: {len(all_instance_ids)}")

    # Get already completed instance IDs
    completed_ids = get_completed_instance_ids(args.log_dir)
    print(f"Already completed: {len(completed_ids)}")

    # Filter to remaining tasks
    remaining_ids = [iid for iid in all_instance_ids if iid not in completed_ids]
    print(f"Remaining tasks: {len(remaining_ids)}")

    if not remaining_ids:
        print("All tasks already completed!")
        # Aggregate results
        df = parse_all_logs(args.log_dir)
        if not df.empty:
            validate_results(df)
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
        if batch_num < args.resume_from_batch:
            print(f"Skipping batch {batch_num} (resuming from {args.resume_from_batch})")
            continue

        success = run_inspect_batch(
            batch_num,
            batch_ids,
            args.model,
            args.max_connections,
            args.log_dir,
            args.task,
        )

        if not success:
            print(f"Batch {batch_num} failed, continuing with next batch...")

        # Cleanup between batches
        if not args.skip_cleanup:
            run_docker_cleanup()

    # Final aggregation
    print("\n=== Final aggregation ===")
    df = parse_all_logs(args.log_dir)
    if not df.empty:
        validate_results(df)
        output_path = args.log_dir / "auditor_features.csv"
        df.to_csv(output_path, index=False)
        print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()

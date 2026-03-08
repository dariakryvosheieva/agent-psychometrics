#!/usr/bin/env python3
"""Main entry point for API-based trajectory summarization.

Usage:
    # Dry run to see what would be processed
    python -m trajectory_summarization_api.run_summarization --dry_run --limit 10

    # Run on subset for testing
    python -m trajectory_summarization_api.run_summarization --limit 5

    # Full run
    python -m trajectory_summarization_api.run_summarization
"""

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

from tqdm import tqdm

from .config import SummarizationConfig
from .data_loader import discover_trajectories
from .summarizer import TrajectorySummarizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Summarize SWE-bench agent trajectories using OpenAI API"
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",
        help="OpenAI model to use",
    )
    # Note: temperature is not supported for gpt-5-mini
    parser.add_argument(
        "--max_output_tokens",
        type=int,
        default=2000,
        help="Maximum output tokens (buffer to avoid incomplete responses)",
    )

    # Data arguments
    parser.add_argument(
        "--trajectory_dir",
        type=Path,
        default=Path("experiment_appendix_h_hard_tasks/trajectory_data/unified_trajs"),
        help="Directory containing trajectory JSON files",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("chris_output/trajectory_summaries_api"),
        help="Directory to save summaries",
    )

    # Parallelization arguments
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=200,
        help="Maximum concurrent API requests",
    )
    parser.add_argument(
        "--rpm",
        type=int,
        default=5000,
        help="Requests per minute limit",
    )

    # Processing arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size for checkpoint saves",
    )
    parser.add_argument(
        "--no_skip_existing",
        action="store_true",
        help="Reprocess trajectories even if output exists",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of trajectories to process",
    )

    # Filtering arguments
    parser.add_argument(
        "--agents",
        nargs="+",
        help="Specific agents to process",
    )
    parser.add_argument(
        "--task_ids",
        nargs="+",
        help="Specific task IDs to process",
    )

    # Optimization arguments
    parser.add_argument(
        "--sort_by_task",
        action="store_true",
        help="Sort by task_id first (instead of agent) to maximize prompt caching",
    )
    parser.add_argument(
        "--max_trajectory_chars",
        type=int,
        default=800_000,
        help="Max characters for trajectory text (default: 800K = ~200K tokens)",
    )

    # Retry arguments
    parser.add_argument(
        "--retry_failures",
        action="store_true",
        help="Only retry trajectories that failed in previous run (from checkpoint)",
    )

    # Debug arguments
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be processed without running",
    )

    return parser.parse_args()


def is_valid_summary(output_file: Path) -> bool:
    """Check if an existing summary file contains a valid summary.

    Args:
        output_file: Path to the summary JSON file

    Returns:
        True if the file contains a non-empty summary, False otherwise
    """
    try:
        with open(output_file) as f:
            data = json.load(f)
        summary = data.get("summary", "")
        # Check that summary exists and has meaningful content (at least 50 chars)
        return isinstance(summary, str) and len(summary.strip()) >= 50
    except (json.JSONDecodeError, IOError, KeyError):
        return False


def filter_existing_summaries(
    all_trajectories: list,
    output_dir: Path,
) -> tuple:
    """Filter out trajectories that already have valid summaries.

    Args:
        all_trajectories: List of (agent_id, task_id, filepath) tuples
        output_dir: Directory where summaries are saved

    Returns:
        Tuple of (to_process, skipped, invalid_removed)
    """
    to_process = []
    skipped = 0
    invalid_removed = 0

    for agent, task_id, filepath in all_trajectories:
        output_file = output_dir / agent / f"{task_id}.json"

        if not output_file.exists():
            to_process.append((agent, task_id, filepath))
        elif is_valid_summary(output_file):
            skipped += 1
        else:
            # Invalid/empty summary - remove and reprocess
            logger.warning(f"Removing invalid summary: {output_file}")
            output_file.unlink()
            to_process.append((agent, task_id, filepath))
            invalid_removed += 1

    return to_process, skipped, invalid_removed


async def run_async(config: SummarizationConfig, all_trajectories: list):
    """Run the async summarization pipeline."""
    summarizer = TrajectorySummarizer(config)

    # Process in batches
    total_processed = 0
    total_successes = 0
    total_failures = 0
    total_api_failures = 0
    start_time = time.time()

    # Create batches
    batches = []
    current_batch = []
    for item in all_trajectories:
        current_batch.append(item)
        if len(current_batch) >= config.batch_size:
            batches.append(current_batch)
            current_batch = []
    if current_batch:
        batches.append(current_batch)

    logger.info(f"Processing {len(batches)} batches of up to {config.batch_size} trajectories...")

    for batch_idx, batch_items in enumerate(tqdm(batches, desc="Processing batches")):
        successes, failures, api_failures = await summarizer.process_batch(batch_items)

        total_successes += successes
        total_failures += failures
        total_api_failures += api_failures
        total_processed += len(batch_items)

        # Update checkpoint with usage stats from client
        summarizer.checkpoint.total_input_tokens = summarizer.client.usage.total_input_tokens
        summarizer.checkpoint.total_output_tokens = summarizer.client.usage.total_output_tokens
        summarizer.checkpoint.save(config.checkpoint_file)

        # Log progress periodically
        if (batch_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = total_processed / elapsed if elapsed > 0 else 0
            cost = summarizer.client.usage.estimated_cost
            logger.info(
                f"Progress: {total_processed}/{len(all_trajectories)} "
                f"({rate:.2f} traj/s, ${cost:.4f} cost, "
                f"{total_api_failures} API failures)"
            )

    return summarizer, total_successes, total_failures, total_api_failures


def main():
    """Main entry point."""
    args = parse_args()

    # Build config
    config = SummarizationConfig(
        model=args.model,
        max_output_tokens=args.max_output_tokens,
        max_trajectory_chars=args.max_trajectory_chars,
        trajectory_dir=args.trajectory_dir,
        output_dir=args.output_dir,
        checkpoint_file=args.output_dir / ".checkpoint.json",
        max_concurrent_requests=args.max_concurrent,
        requests_per_minute=args.rpm,
        batch_size=args.batch_size,
        skip_existing=not args.no_skip_existing,
        agents=args.agents,
        task_ids=args.task_ids,
        dry_run=args.dry_run,
        limit=args.limit,
    )

    # Log configuration
    logger.info("=" * 60)
    logger.info("Trajectory Summarization Pipeline (OpenAI API)")
    logger.info("=" * 60)
    logger.info(f"Model: {config.model}")
    logger.info(f"Max output tokens: {config.max_output_tokens}")
    logger.info(f"Max trajectory chars: {config.max_trajectory_chars:,}")
    logger.info(f"Trajectory dir: {config.trajectory_dir}")
    logger.info(f"Output dir: {config.output_dir}")
    logger.info(f"Max concurrent: {config.max_concurrent_requests}")
    logger.info(f"RPM limit: {config.requests_per_minute}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Skip existing: {config.skip_existing}")
    if config.agents:
        logger.info(f"Agents filter: {config.agents}")
    if config.task_ids:
        logger.info(f"Task IDs filter: {config.task_ids}")
    if config.limit:
        logger.info(f"Limit: {config.limit}")
    if config.dry_run:
        logger.info("DRY RUN MODE - no API calls will be made")
    logger.info("=" * 60)

    # Discover trajectories
    logger.info(f"Discovering trajectories from {config.trajectory_dir}...")
    all_trajectories = discover_trajectories(
        config.trajectory_dir,
        agents=config.agents,
        task_ids=config.task_ids,
    )
    logger.info(f"Found {len(all_trajectories)} trajectories")

    # Filter to only retry failures if requested
    if args.retry_failures:
        # Load checkpoint and get api_failures
        checkpoint_file = config.output_dir / ".checkpoint.json"
        if checkpoint_file.exists():
            with open(checkpoint_file) as f:
                checkpoint_data = json.load(f)
            api_failures = set(checkpoint_data.get("api_failures", []))
            logger.info(f"Found {len(api_failures)} API failures in checkpoint")

            # Filter trajectories to only those that failed
            all_trajectories = [
                (agent, task_id, filepath)
                for agent, task_id, filepath in all_trajectories
                if f"{agent}/{task_id}" in api_failures
            ]
            logger.info(f"Retrying {len(all_trajectories)} failed trajectories")

            # Clear api_failures from checkpoint so we can track new ones
            # (only if not dry_run - don't modify checkpoint in dry_run mode)
            if not config.dry_run:
                checkpoint_data["api_failures"] = []
                with open(checkpoint_file, "w") as f:
                    json.dump(checkpoint_data, f, indent=2)
        else:
            logger.warning("No checkpoint file found - nothing to retry")
            return

    # Sort by task_id for prompt caching if requested
    if args.sort_by_task:
        all_trajectories = sorted(all_trajectories, key=lambda x: (x[1], x[0]))  # (task_id, agent_id)
        logger.info("Sorted by task_id for prompt caching optimization")

    if config.limit:
        all_trajectories = all_trajectories[: config.limit]
        logger.info(f"Limited to {config.limit} trajectories")

    # Filter existing if skip_existing
    if config.skip_existing:
        all_trajectories, skipped, invalid_removed = filter_existing_summaries(
            all_trajectories,
            config.output_dir,
        )
        if skipped > 0:
            logger.info(f"Skipping {skipped} existing valid summaries")
        if invalid_removed > 0:
            logger.warning(f"Removed {invalid_removed} invalid/empty summaries for reprocessing")

    if not all_trajectories:
        logger.info("No trajectories to process")
        return

    logger.info(f"Trajectories to process: {len(all_trajectories)}")

    if config.dry_run:
        logger.info("\nDRY RUN - would process:")
        for agent, task_id, filepath in all_trajectories[:20]:
            logger.info(f"  {agent}/{task_id}")
        if len(all_trajectories) > 20:
            logger.info(f"  ... and {len(all_trajectories) - 20} more")
        return

    # Run async pipeline
    start_time = time.time()
    summarizer, successes, failures, api_failures = asyncio.run(
        run_async(config, all_trajectories)
    )
    elapsed = time.time() - start_time

    # Final summary
    rate = successes / elapsed if elapsed > 0 else 0
    cost = summarizer.client.usage.estimated_cost

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Successes: {successes}")
    logger.info(f"Failures (load errors): {failures}")
    logger.info(f"API failures (returned None): {api_failures}")
    logger.info(f"Total time: {elapsed:.1f}s")
    logger.info(f"Rate: {rate:.2f} trajectories/second")
    logger.info(f"Total input tokens: {summarizer.client.usage.total_input_tokens:,}")
    logger.info(f"Total output tokens: {summarizer.client.usage.total_output_tokens:,}")
    logger.info(f"Total cost: ${cost:.4f}")
    logger.info(f"Output directory: {config.output_dir}")

    # Log checkpoint summary
    summarizer.log_checkpoint_summary()

    # Count output files
    output_count = sum(1 for _ in config.output_dir.rglob("*.json") if not _.name.startswith("."))
    logger.info(f"Total summaries on disk: {output_count}")

    # Warn about API failures
    if api_failures > 0:
        logger.warning(
            f"\n⚠️  {api_failures} API failures occurred. "
            f"Check checkpoint file for details: {config.checkpoint_file}"
        )


if __name__ == "__main__":
    main()

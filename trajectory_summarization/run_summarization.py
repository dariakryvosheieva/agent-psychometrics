#!/usr/bin/env python3
"""Main entry point for trajectory summarization.

Usage:
    # Dry run to see what would be processed
    python -m trajectory_summarization.run_summarization --dry_run --limit 10

    # Run on subset for testing
    python -m trajectory_summarization.run_summarization --limit 100

    # Full run with sharding (for 2 GPU data parallelism)
    python -m trajectory_summarization.run_summarization --shard_id 0 --num_shards 2
"""

import argparse
import logging
import os
import time
from pathlib import Path

from tqdm import tqdm

from .config import SummarizationConfig
from .data_loader import discover_trajectories, load_trajectory
from .summarizer import TrajectorySummarizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Summarize SWE-bench agent trajectories using vLLM"
    )

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-8B-Instruct",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="fp8",
        choices=["fp8", "awq", "gptq", "none"],
        help="Quantization method",
    )

    # Data arguments
    parser.add_argument(
        "--trajectory_dir",
        type=Path,
        default=Path("trajectory_data/unified_trajs"),
        help="Directory containing trajectory JSON files",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("chris_output/trajectory_summaries"),
        help="Directory to save summaries",
    )

    # Sharding arguments
    parser.add_argument(
        "--shard_id",
        type=int,
        default=0,
        help="Which shard to process (0-indexed)",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Total number of shards",
    )

    # Processing arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for vLLM inference",
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

    # vLLM arguments
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.90,
        help="GPU memory utilization for vLLM",
    )
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=16,
        help="Maximum number of sequences for continuous batching",
    )

    # Debug arguments
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be processed without running inference",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Build config
    config = SummarizationConfig(
        model_name=args.model_name,
        quantization=args.quantization,
        trajectory_dir=args.trajectory_dir,
        output_dir=args.output_dir,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        batch_size=args.batch_size,
        skip_existing=not args.no_skip_existing,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        dry_run=args.dry_run,
        limit=args.limit,
    )

    # Log configuration
    logger.info("=" * 60)
    logger.info("Trajectory Summarization Pipeline")
    logger.info("=" * 60)
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Quantization: {config.quantization}")
    logger.info(f"Trajectory dir: {config.trajectory_dir}")
    logger.info(f"Output dir: {config.output_dir}")
    logger.info(f"Shard: {config.shard_id}/{config.num_shards}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Skip existing: {config.skip_existing}")
    if config.limit:
        logger.info(f"Limit: {config.limit}")
    if config.dry_run:
        logger.info("DRY RUN MODE - no inference will be performed")
    logger.info("=" * 60)

    # Discover trajectories
    logger.info(f"Discovering trajectories from {config.trajectory_dir}...")
    all_trajectories = discover_trajectories(
        config.trajectory_dir,
        shard_id=config.shard_id,
        num_shards=config.num_shards,
    )
    logger.info(
        f"Found {len(all_trajectories)} trajectories for shard {config.shard_id}/{config.num_shards}"
    )

    if config.limit:
        all_trajectories = all_trajectories[: config.limit]
        logger.info(f"Limited to {config.limit} trajectories")

    # Filter existing if skip_existing
    output_dir = Path(config.output_dir)
    if config.skip_existing:
        to_process = []
        for agent, task_id, filepath in all_trajectories:
            output_file = output_dir / agent / f"{task_id}.json"
            if not output_file.exists():
                to_process.append((agent, task_id, filepath))
        skipped = len(all_trajectories) - len(to_process)
        if skipped > 0:
            logger.info(f"Skipping {skipped} existing summaries")
        all_trajectories = to_process

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

    # Initialize summarizer
    summarizer = TrajectorySummarizer(config)

    # Process in batches
    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    errors = 0
    total_input_tokens = 0
    total_output_tokens = 0
    start_time = time.time()

    # Create batches
    batches = []
    current_batch = []
    for agent, task_id, filepath in all_trajectories:
        current_batch.append((agent, task_id, filepath))
        if len(current_batch) >= config.batch_size:
            batches.append(current_batch)
            current_batch = []
    if current_batch:
        batches.append(current_batch)

    logger.info(f"Processing {len(batches)} batches...")

    for batch_idx, batch_items in enumerate(tqdm(batches, desc="Processing batches")):
        # Load trajectories
        trajectories = []
        for agent, task_id, filepath in batch_items:
            traj = load_trajectory(filepath)
            if traj:
                trajectories.append(traj)
            else:
                errors += 1

        if not trajectories:
            continue

        # Summarize
        try:
            summaries = summarizer.summarize_batch(trajectories)
            for summary in summaries:
                summarizer.save_summary(summary, output_dir)
                processed += 1
                total_input_tokens += summary.metadata.get("input_tokens", 0)
                total_output_tokens += summary.metadata.get("output_tokens", 0)
        except Exception as e:
            logger.error(f"Error processing batch {batch_idx}: {e}")
            errors += len(trajectories)

        # Periodic logging
        if (batch_idx + 1) % config.log_every == 0:
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            logger.info(
                f"Progress: {processed}/{len(all_trajectories)} "
                f"({rate:.2f} traj/s, {errors} errors)"
            )

    # Final summary
    elapsed = time.time() - start_time
    rate = processed / elapsed if elapsed > 0 else 0

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Processed: {processed}")
    logger.info(f"Errors: {errors}")
    logger.info(f"Total time: {elapsed:.1f}s")
    logger.info(f"Rate: {rate:.2f} trajectories/second")
    logger.info(f"Total input tokens: {total_input_tokens:,}")
    logger.info(f"Total output tokens: {total_output_tokens:,}")
    logger.info(f"Output directory: {output_dir}")

    # Count output files
    output_count = sum(1 for _ in output_dir.rglob("*.json"))
    logger.info(f"Total summaries on disk: {output_count}")


if __name__ == "__main__":
    main()

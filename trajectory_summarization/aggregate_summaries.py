#!/usr/bin/env python3
"""Aggregate individual summary JSON files into a single JSONL file.

Usage:
    python -m trajectory_summarization.aggregate_summaries \
        --input_dir chris_output/trajectory_summaries \
        --output_file chris_output/trajectory_summaries/all_summaries.jsonl
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def aggregate_summaries(input_dir: Path, output_file: Path) -> int:
    """Aggregate all summary JSON files into a single JSONL file.

    Args:
        input_dir: Directory containing agent subdirectories with summary JSONs
        output_file: Output JSONL file path

    Returns:
        Number of summaries aggregated
    """
    summaries = []

    # Find all JSON files
    json_files = list(input_dir.rglob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files")

    for json_file in json_files:
        # Skip non-summary files
        if json_file.name.startswith("_") or json_file.name == "all_summaries.jsonl":
            continue

        try:
            with open(json_file) as f:
                data = json.load(f)

            # Validate it's a summary
            if "task_id" in data and "summary" in data:
                summaries.append(data)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load {json_file}: {e}")

    # Sort by agent and task_id for consistency
    summaries.sort(key=lambda x: (x.get("agent", ""), x.get("task_id", "")))

    # Write JSONL
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for summary in summaries:
            f.write(json.dumps(summary) + "\n")

    logger.info(f"Aggregated {len(summaries)} summaries to {output_file}")
    return len(summaries)


def compute_stats(input_dir: Path) -> dict:
    """Compute statistics about the summaries.

    Args:
        input_dir: Directory containing agent subdirectories with summary JSONs

    Returns:
        Dictionary of statistics
    """
    stats = {
        "total_summaries": 0,
        "total_agents": 0,
        "total_tasks": set(),
        "resolved_count": 0,
        "failed_count": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "summary_lengths": [],
    }

    agents = set()

    for json_file in input_dir.rglob("*.json"):
        if json_file.name.startswith("_"):
            continue

        try:
            with open(json_file) as f:
                data = json.load(f)

            if "task_id" not in data or "summary" not in data:
                continue

            stats["total_summaries"] += 1
            agents.add(data.get("agent", ""))
            stats["total_tasks"].add(data.get("task_id", ""))

            if data.get("resolved"):
                stats["resolved_count"] += 1
            else:
                stats["failed_count"] += 1

            metadata = data.get("metadata", {})
            stats["total_input_tokens"] += metadata.get("input_tokens", 0)
            stats["total_output_tokens"] += metadata.get("output_tokens", 0)
            stats["summary_lengths"].append(len(data.get("summary", "")))

        except (json.JSONDecodeError, IOError):
            continue

    stats["total_agents"] = len(agents)
    stats["total_tasks"] = len(stats["total_tasks"])

    if stats["summary_lengths"]:
        lengths = stats["summary_lengths"]
        stats["avg_summary_length"] = sum(lengths) / len(lengths)
        stats["min_summary_length"] = min(lengths)
        stats["max_summary_length"] = max(lengths)
    del stats["summary_lengths"]

    return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Aggregate trajectory summaries into a single JSONL file"
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("chris_output/trajectory_summaries"),
        help="Directory containing summary JSON files",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=None,
        help="Output JSONL file (default: {input_dir}/all_summaries.jsonl)",
    )
    parser.add_argument(
        "--stats_only",
        action="store_true",
        help="Only compute and print statistics",
    )

    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = args.input_dir / "all_summaries.jsonl"

    if args.stats_only:
        stats = compute_stats(args.input_dir)
        logger.info("\n=== Summary Statistics ===")
        for key, value in stats.items():
            if isinstance(value, float):
                logger.info(f"{key}: {value:.1f}")
            elif isinstance(value, int) and value > 1000:
                logger.info(f"{key}: {value:,}")
            else:
                logger.info(f"{key}: {value}")
        return

    count = aggregate_summaries(args.input_dir, args.output_file)

    # Also print stats
    stats = compute_stats(args.input_dir)
    logger.info("\n=== Summary Statistics ===")
    for key, value in stats.items():
        if isinstance(value, float):
            logger.info(f"{key}: {value:.1f}")
        elif isinstance(value, int) and value > 1000:
            logger.info(f"{key}: {value:,}")
        else:
            logger.info(f"{key}: {value}")


if __name__ == "__main__":
    main()

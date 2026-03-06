"""CLI entry point for LLM Judge feature extraction.

Usage:
    # Extract specific features for a dataset
    python -m experiment_ab_shared.llm_judge extract \
        --features solution_hint,problem_clarity,test_comprehensiveness \
        --dataset swebench_verified

    # Extract all non-environment features
    python -m experiment_ab_shared.llm_judge extract --all --dataset swebench_verified

    # Dry run to see batch plan and cost estimate
    python -m experiment_ab_shared.llm_judge extract --all --dataset swebench_verified --dry-run

    # Custom provider/model
    python -m experiment_ab_shared.llm_judge extract --all --dataset gso \
        --provider anthropic --model claude-sonnet-4-6

    # Parallel extraction
    python -m experiment_ab_shared.llm_judge extract --all --dataset swebench_verified \
        --parallel --concurrency 10

    # Aggregate existing JSONs to CSV
    python -m experiment_ab_shared.llm_judge aggregate \
        --dataset swebench_verified \
        --output-dir chris_output/experiment_a/llm_judge_features

    # Analyze feature correlations with IRT difficulty
    python -m experiment_ab_shared.llm_judge correlations \
        --features-csv path/to/features.csv --irt-items path/to/items.csv
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

from experiment_ab_shared.llm_judge.batched_extractor import BatchedFeatureExtractor
from experiment_ab_shared.llm_judge.feature_registry import get_features_by_level
from experiment_ab_shared.llm_judge.prompt_config import InfoLevel
from experiment_ab_shared.llm_judge.task_context import get_task_context
from experiment_ab_shared.llm_judge.task_loaders import (
    load_tasks,
    load_tasks_from_jsonl,
    SUPPORTED_DATASETS,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIRS = {
    "swebench_verified": Path("chris_output/experiment_a/llm_judge_features"),
    "swebench_pro": Path("chris_output/experiment_a_swebench_pro/llm_judge_features"),
    "terminalbench": Path("chris_output/experiment_a_terminalbench/llm_judge_features"),
    "gso": Path("chris_output/gso_llm_judge_features"),
}


def _get_non_environment_feature_names() -> List[str]:
    """All feature names except environment-level (handled by auditor agent)."""
    names = []
    for level in (InfoLevel.PROBLEM, InfoLevel.TEST, InfoLevel.SOLUTION):
        names.extend(f.name for f in get_features_by_level(level))
    return names


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    return DEFAULT_OUTPUT_DIRS.get(
        args.dataset,
        Path(f"chris_output/llm_judge_features/{args.dataset}"),
    )


def cmd_extract(args: argparse.Namespace) -> None:
    """Handle the 'extract' subcommand."""
    if not args.dataset:
        print("Error: --dataset is required")
        sys.exit(1)

    if args.all:
        feature_names = _get_non_environment_feature_names()
    elif args.features:
        feature_names = [f.strip() for f in args.features.split(",")]
    else:
        print("Error: specify --features or --all")
        sys.exit(1)

    output_dir = _resolve_output_dir(args)
    task_context = get_task_context(args.dataset)

    if args.tasks:
        tasks = load_tasks_from_jsonl(Path(args.tasks))
    else:
        tasks = load_tasks(args.dataset)

    extractor = BatchedFeatureExtractor(
        feature_names=feature_names,
        task_context=task_context,
        provider=args.provider,
        model=args.model,
        batch_size=args.batch_size,
    )

    task_ids = args.task_ids.split(",") if args.task_ids else None

    if args.dry_run:
        extractor.dry_run(
            tasks=tasks,
            output_dir=output_dir,
            skip_existing=not args.no_skip_existing,
            limit=args.limit,
            task_ids=task_ids,
        )
    elif args.parallel:
        extractor.run_parallel(
            tasks=tasks,
            output_dir=output_dir,
            skip_existing=not args.no_skip_existing,
            limit=args.limit,
            task_ids=task_ids,
            concurrency=args.concurrency,
        )
    else:
        extractor.run(
            tasks=tasks,
            output_dir=output_dir,
            skip_existing=not args.no_skip_existing,
            delay=args.delay,
            limit=args.limit,
            task_ids=task_ids,
        )


def cmd_aggregate(args: argparse.Namespace) -> None:
    """Handle the 'aggregate' subcommand."""
    if not args.dataset:
        print("Error: --dataset is required")
        sys.exit(1)

    output_dir = _resolve_output_dir(args)
    if not output_dir.exists():
        print(f"Error: Output directory does not exist: {output_dir}")
        sys.exit(1)

    if args.features:
        feature_names = [f.strip() for f in args.features.split(",")]
    else:
        feature_names = _get_non_environment_feature_names()

    task_context = get_task_context(args.dataset)

    extractor = BatchedFeatureExtractor(
        feature_names=feature_names,
        task_context=task_context,
        provider=args.provider,
    )

    csv_path = extractor._aggregate_to_csv(output_dir)
    if csv_path:
        print(f"Aggregated CSV: {csv_path}")
    else:
        print("No JSON files found to aggregate")


def cmd_correlations(args: argparse.Namespace) -> None:
    """Handle the 'correlations' subcommand."""
    from experiment_ab_shared.llm_judge.analyze_feature_correlations import (
        analyze_features,
    )

    analyze_features(
        features_path=Path(args.features_csv),
        irt_items_path=Path(args.irt_items),
        dataset_name=args.dataset,
        output_path=Path(args.output) if args.output else None,
    )


# ============================================================================
# Argument Parser Setup
# ============================================================================

def _setup_extract_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "extract", help="Extract LLM judge features from tasks",
    )
    parser.add_argument(
        "--dataset", type=str, choices=SUPPORTED_DATASETS,
        help="Dataset to extract features for",
    )
    parser.add_argument(
        "--features", type=str,
        help="Comma-separated list of feature names to extract",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Extract all non-environment features",
    )
    parser.add_argument(
        "--tasks", type=str,
        help="Path to tasks JSONL file (overrides --dataset loader)",
    )
    parser.add_argument(
        "--output-dir", type=str, dest="output_dir",
        help="Output directory for JSON files and CSV",
    )
    parser.add_argument(
        "--provider", type=str, default="openai",
        choices=["anthropic", "openai"],
        help="LLM provider (default: openai)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Specific model to use (default: provider's default)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=7, dest="batch_size",
        help="Max features per API call (default: 7)",
    )
    parser.add_argument(
        "--limit", type=int,
        help="Maximum number of tasks to process",
    )
    parser.add_argument(
        "--task-ids", type=str, dest="task_ids",
        help="Comma-separated list of specific task IDs to process",
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Delay between tasks in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--no-skip-existing", action="store_true", dest="no_skip_existing",
        help="Re-process tasks with existing feature files",
    )
    parser.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="Show execution plan and cost estimate without running",
    )
    parser.add_argument(
        "--parallel", action="store_true",
        help="Run extraction in parallel (async API calls)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=10,
        help="Max concurrent tasks when --parallel is used (default: 10)",
    )


def _setup_aggregate_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "aggregate", help="Aggregate existing JSON files into CSV",
    )
    parser.add_argument(
        "--dataset", type=str, choices=SUPPORTED_DATASETS,
        help="Dataset (for task context and column ordering)",
    )
    parser.add_argument(
        "--features", type=str,
        help="Comma-separated feature names for column ordering",
    )
    parser.add_argument(
        "--output-dir", type=str, dest="output_dir",
        help="Directory containing JSON files",
    )
    parser.add_argument(
        "--provider", type=str, default="openai",
        help="Provider (only used for BatchedFeatureExtractor init)",
    )


def _setup_correlations_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "correlations", help="Analyze feature correlations with IRT difficulty",
    )
    parser.add_argument(
        "--features-csv", type=str, required=True, dest="features_csv",
        help="Path to features CSV file",
    )
    parser.add_argument(
        "--irt-items", type=str, required=True, dest="irt_items",
        help="Path to IRT items.csv with difficulty column 'b'",
    )
    parser.add_argument(
        "--dataset", type=str, default="unknown",
        help="Dataset name for display",
    )
    parser.add_argument(
        "--output", type=str,
        help="Path to save results JSON",
    )


def main():
    parser = argparse.ArgumentParser(
        description="LLM Judge feature extraction for task difficulty prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    _setup_extract_parser(subparsers)
    _setup_aggregate_parser(subparsers)
    _setup_correlations_parser(subparsers)

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    command_handlers = {
        "extract": cmd_extract,
        "aggregate": cmd_aggregate,
        "correlations": cmd_correlations,
    }

    if args.command in command_handlers:
        command_handlers[args.command](args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

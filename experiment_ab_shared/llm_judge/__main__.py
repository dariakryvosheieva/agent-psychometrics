"""CLI entry point for LLM Judge feature extraction.

Usage:
    # Extract features for built-in datasets
    python -m experiment_ab_shared.llm_judge extract --dataset swebench
    python -m experiment_ab_shared.llm_judge extract --dataset terminalbench

    # Dry run to see cost estimate
    python -m experiment_ab_shared.llm_judge extract --dataset swebench --dry-run

    # Custom options
    python -m experiment_ab_shared.llm_judge extract \\
        --dataset swebench \\
        --provider anthropic \\
        --model claude-sonnet-4-20250514 \\
        --limit 50 \\
        --output-dir chris_output/experiment_a/llm_judge_features

    # Aggregate only (combine existing JSONs to CSV)
    python -m experiment_ab_shared.llm_judge aggregate \\
        --dataset swebench \\
        --output-dir chris_output/experiment_a/llm_judge_features

    # Use custom prompt config file
    python -m experiment_ab_shared.llm_judge extract \\
        --config path/to/custom_config.py \\
        --tasks path/to/tasks.jsonl
"""

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from experiment_ab_shared.llm_judge.extractor import LLMFeatureExtractor
from experiment_ab_shared.llm_judge.prompt_config import PromptConfig
from experiment_ab_shared.llm_judge.prompts import get_prompt_config, list_datasets

logger = logging.getLogger(__name__)

# Default output directories per dataset
DEFAULT_OUTPUT_DIRS = {
    "swebench": Path("chris_output/experiment_a/llm_judge_features"),
    "swebench_pro": Path("chris_output/experiment_a_swebench_pro/llm_judge_features"),
    "terminalbench": Path("chris_output/experiment_a_terminalbench/llm_judge_features"),
    "gso": Path("chris_output/gso_llm_judge_features"),
}


def load_swebench_tasks() -> List[Dict[str, Any]]:
    """Load SWE-bench Verified dataset from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets package required for SWE-bench. Install with: pip install datasets"
        )

    print("Loading SWE-bench Verified dataset from HuggingFace...")
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

    tasks = []
    for item in ds:
        tasks.append({
            "instance_id": item["instance_id"],
            "repo": item["repo"],
            "problem_statement": item["problem_statement"],
            "patch": item["patch"],
            "test_patch": item["test_patch"],
            "version": item["version"],
            "hints_text": item["hints_text"],
            "FAIL_TO_PASS": item["FAIL_TO_PASS"],
            "PASS_TO_PASS": item["PASS_TO_PASS"],
        })

    print(f"Loaded {len(tasks)} tasks")
    return tasks


def _normalize_swebench_pro_task_id(task_id: str) -> str:
    """Normalize SWE-bench Pro task IDs to match IRT format.

    SWE-bench Pro task IDs from HuggingFace have the format:
        instance_<repo>__<repo>-<commit>-v<version>

    But the IRT data uses the format:
        <repo>__<repo>-<commit>

    This function normalizes to the IRT format by:
    1. Removing 'instance_' prefix
    2. Removing version suffix (-v<hex> or -vnan)
    """
    import re

    # Remove instance_ prefix
    if task_id.startswith('instance_'):
        task_id = task_id[9:]

    # Remove version suffix (format: -v followed by 'nan' or hex chars at end)
    task_id = re.sub(r'-v(nan|[a-f0-9]+)$', '', task_id)

    return task_id


def load_swebench_pro_tasks() -> List[Dict[str, Any]]:
    """Load SWE-bench Pro dataset from HuggingFace.

    Data source: ScaleAI/SWE-bench_Pro

    Note: Task IDs are normalized to match the IRT data format by removing
    the 'instance_' prefix and version suffix.

    Returns:
        List of task dictionaries with fields:
        - instance_id: Task identifier (normalized)
        - repo: Repository name
        - problem_statement: Issue description
        - patch: Gold solution patch
        - fail_to_pass: Tests that should pass after fix
        - pass_to_pass: Regression tests
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets package required for SWE-bench Pro. Install with: pip install datasets"
        )

    print("Loading SWE-bench Pro dataset from HuggingFace (ScaleAI/SWE-bench_Pro)...")
    ds = load_dataset("ScaleAI/SWE-bench_Pro", split="test")

    tasks = []
    for item in ds:
        # Normalize task ID to match IRT format
        normalized_id = _normalize_swebench_pro_task_id(item["instance_id"])
        tasks.append({
            "instance_id": normalized_id,
            "repo": item["repo"],
            "problem_statement": item["problem_statement"],
            "patch": item["patch"],
            "test_patch": item.get("test_patch", ""),
            "version": item.get("version", "unknown"),
            "hints_text": item.get("hints_text", ""),
            # SWE-bench Pro uses lowercase field names
            "fail_to_pass": item.get("fail_to_pass", "[]"),
            "pass_to_pass": item.get("pass_to_pass", "[]"),
        })

    print(f"Loaded {len(tasks)} tasks")
    return tasks


def load_gso_tasks() -> List[Dict[str, Any]]:
    """Load GSO (Software Optimization Benchmark) dataset from HuggingFace.

    Data source: gso-bench/gso

    Returns:
        List of task dictionaries with fields:
        - instance_id: Task identifier (owner__repo-commit format)
        - repo: Repository name
        - api: API/function being optimized
        - prob_script: Test script showing performance scenario
        - gt_diff: Gold optimization patch
        - hints_text: Optional hints
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets package required for GSO. Install with: pip install datasets"
        )

    print("Loading GSO dataset from HuggingFace (gso-bench/gso)...")
    ds = load_dataset("gso-bench/gso", split="test")

    tasks = []
    for item in ds:
        tasks.append({
            "instance_id": item["instance_id"],
            "repo": item.get("repo", ""),
            "api": item.get("api", ""),
            "prob_script": item.get("prob_script", ""),
            "gt_diff": item.get("gt_diff", ""),
            "hints_text": item.get("hints_text", ""),
        })

    print(f"Loaded {len(tasks)} tasks")
    return tasks


def load_terminalbench_tasks(
    items_path: Optional[Path] = None,
    repo_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Load TerminalBench tasks from the pre-extracted tasks JSONL.

    Reads from data/terminalbench/tasks.jsonl (generated by
    swebench_irt/scrape_terminal_bench_statements.py from terminal-bench-2).
    Maps JSONL field names to the names expected by prompt formatters.

    Args:
        items_path: Path to items CSV (filters to IRT task set). If None,
            uses default IRT items path.
        repo_path: Unused (kept for CLI compatibility).

    Returns:
        List of task dicts with task_id, instruction, solution, tags,
        category, claimed_difficulty.

    Raises:
        FileNotFoundError: If the tasks JSONL or items CSV is missing.
        ValueError: If tasks are missing required fields.
    """
    tasks_jsonl = Path("data/terminalbench/tasks.jsonl")
    if not tasks_jsonl.exists():
        raise FileNotFoundError(
            f"Tasks JSONL not found: {tasks_jsonl}. "
            f"Generate it with: python swebench_irt/scrape_terminal_bench_statements.py"
        )

    # Load all tasks from JSONL
    all_tasks = {}
    with open(tasks_jsonl, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                all_tasks[record["task_id"]] = record

    # Filter to IRT task set
    if items_path is None:
        items_path = Path("data/terminalbench/irt/1d_1pl/items.csv")
    if items_path.exists():
        import pandas as pd
        items_df = pd.read_csv(items_path, index_col=0)
        task_ids = items_df.index.tolist()
    else:
        task_ids = list(all_tasks.keys())

    # Map JSONL fields to prompt formatter field names
    tasks = []
    errors = []
    for task_id in task_ids:
        if task_id not in all_tasks:
            errors.append(f"Task {task_id}: not found in {tasks_jsonl}")
            continue
        record = all_tasks[task_id]
        instruction = record.get("problem_statement", "").strip()
        solution = record.get("patch", "").strip()
        if not instruction:
            errors.append(f"Task {task_id}: empty problem_statement")
            continue
        if not solution:
            errors.append(f"Task {task_id}: empty patch")
            continue
        tasks.append({
            "task_id": task_id,
            "instruction": instruction,
            "solution": solution,
            "tags": record.get("tags", []),
            "category": record.get("category", ""),
            "claimed_difficulty": record.get("difficulty", ""),
        })

    if errors:
        error_summary = "\n".join(errors[:10])
        if len(errors) > 10:
            error_summary += f"\n... and {len(errors) - 10} more errors"
        raise ValueError(
            f"Found {len(errors)} tasks with missing data:\n{error_summary}"
        )

    print(f"Loaded {len(tasks)} TerminalBench tasks from {tasks_jsonl}")
    return tasks


def load_tasks_from_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load tasks from a JSONL file."""
    tasks = []
    with open(path) as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    print(f"Loaded {len(tasks)} tasks from {path}")
    return tasks


def load_custom_config(config_path: Path) -> PromptConfig:
    """Load a custom PromptConfig from a Python file.

    The Python file should define a variable named CONFIG that is a PromptConfig instance.

    Args:
        config_path: Path to Python file containing CONFIG

    Returns:
        PromptConfig instance
    """
    spec = importlib.util.spec_from_file_location("custom_config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load config from {config_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "CONFIG"):
        raise ValueError(f"Config file must define a CONFIG variable: {config_path}")

    config = module.CONFIG
    if not isinstance(config, PromptConfig):
        raise TypeError(f"CONFIG must be a PromptConfig instance, got {type(config)}")

    return config


def load_tasks_for_dataset(
    dataset: str,
    items_path: Optional[Path] = None,
    repo_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Load tasks for a built-in dataset."""
    if dataset in ("swebench", "swebench_with_test", "swebench_test_quality_with_solution", "swebench_test_quality_no_solution"):
        return load_swebench_tasks()
    elif dataset == "swebench_pro":
        return load_swebench_pro_tasks()
    elif dataset == "terminalbench":
        return load_terminalbench_tasks(items_path, repo_path)
    elif dataset == "gso":
        return load_gso_tasks()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def cmd_extract(args: argparse.Namespace) -> None:
    """Handle the 'extract' subcommand."""
    # Load prompt config
    if args.config:
        config = load_custom_config(Path(args.config))
        dataset_name = config.name
    else:
        if not args.dataset:
            print("Error: Must specify --dataset or --config")
            sys.exit(1)
        config = get_prompt_config(args.dataset)
        dataset_name = args.dataset

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = DEFAULT_OUTPUT_DIRS.get(
            dataset_name,
            Path(f"chris_output/llm_judge_features/{dataset_name}"),
        )

    # Load tasks
    if args.tasks:
        tasks = load_tasks_from_jsonl(Path(args.tasks))
    else:
        if not args.dataset:
            print("Error: Must specify --dataset or --tasks")
            sys.exit(1)
        tasks = load_tasks_for_dataset(
            args.dataset,
            items_path=Path(args.items_path) if args.items_path else None,
            repo_path=Path(args.repo_path) if args.repo_path else None,
        )

    # Create extractor
    extractor = LLMFeatureExtractor(
        prompt_config=config,
        output_dir=output_dir,
        provider=args.provider,
        model=args.model,
    )

    # Parse task_ids if provided
    task_ids = args.task_ids.split(",") if args.task_ids else None

    # Run extraction or dry run
    csv_path = None
    if args.dry_run:
        extractor.dry_run(
            tasks=tasks,
            limit=args.limit,
            task_ids=task_ids,
            skip_existing=not args.no_skip_existing,
        )
    elif args.parallel:
        csv_path = extractor.run_parallel(
            tasks=tasks,
            skip_existing=not args.no_skip_existing,
            limit=args.limit,
            task_ids=task_ids,
            concurrency=args.concurrency,
        )
    else:
        csv_path = extractor.run(
            tasks=tasks,
            skip_existing=not args.no_skip_existing,
            delay=args.delay,
            limit=args.limit,
            task_ids=task_ids,
        )



def cmd_aggregate(args: argparse.Namespace) -> None:
    """Handle the 'aggregate' subcommand."""
    # Load prompt config for column ordering
    if args.config:
        config = load_custom_config(Path(args.config))
        dataset_name = config.name
    elif args.dataset:
        config = get_prompt_config(args.dataset)
        dataset_name = args.dataset
    else:
        print("Error: Must specify --dataset or --config for column ordering")
        sys.exit(1)

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = DEFAULT_OUTPUT_DIRS.get(
            dataset_name,
            Path(f"chris_output/llm_judge_features/{dataset_name}"),
        )

    if not output_dir.exists():
        print(f"Error: Output directory does not exist: {output_dir}")
        sys.exit(1)

    # Create extractor just for aggregation
    extractor = LLMFeatureExtractor(
        prompt_config=config,
        output_dir=output_dir,
    )

    csv_path = extractor.aggregate_to_csv()
    if csv_path:
        print(f"Aggregated CSV: {csv_path}")
    else:
        print("No JSON files found to aggregate")


def cmd_quick_eval(args: argparse.Namespace) -> None:
    """Handle the 'quick_eval' subcommand."""
    import pandas as pd
    from experiment_ab_shared.llm_judge.quick_eval import evaluate_features

    # Load features CSV
    features_df = pd.read_csv(args.features_csv)

    # Find task ID column and set as index
    task_id_col = None
    for col in ["_instance_id", "instance_id", "_task_id", "task_id"]:
        if col in features_df.columns:
            task_id_col = col
            break
    if task_id_col is None:
        task_id_col = features_df.columns[0]
    features_df = features_df.set_index(task_id_col)

    # Run evaluation
    result = evaluate_features(
        features_df,
        Path(args.irt_items),
        correlation_threshold=args.correlation_threshold,
        redundancy_threshold=args.redundancy_threshold,
    )

    # Print report
    result.print_report()

    # Save if output path provided
    if args.output:
        result.to_json(Path(args.output))


def cmd_compare(args: argparse.Namespace) -> None:
    """Handle the 'compare' subcommand."""
    from experiment_ab_shared.llm_judge.qualitative_compare import run_comparison

    run_comparison(
        dataset=args.dataset,
        n_pairs=args.n_pairs,
        feature_to_highlight=args.feature,
        features_csv=Path(args.features_csv) if args.features_csv else None,
        irt_items_path=Path(args.irt_items) if args.irt_items else None,
        seed=args.seed,
        interactive=args.interactive,
    )


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
# Argument Parser Setup Functions
# ============================================================================


def _setup_extract_parser(subparsers) -> None:
    """Setup the 'extract' subcommand parser."""
    parser = subparsers.add_parser(
        "extract",
        help="Extract LLM judge features from tasks",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list_datasets(),
        help="Built-in dataset to use",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom PromptConfig Python file",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        help="Path to tasks JSONL file (for custom configs)",
    )
    parser.add_argument(
        "--items-path",
        type=str,
        dest="items_path",
        help="Path to items CSV (TerminalBench only)",
    )
    parser.add_argument(
        "--repo-path",
        type=str,
        dest="repo_path",
        help="Path to terminal-bench repo (TerminalBench only)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        dest="output_dir",
        help="Output directory for JSON files and CSV",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="anthropic",
        choices=["anthropic", "openai"],
        help="LLM provider (default: anthropic)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Specific model to use (default: provider's default)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of tasks to process",
    )
    parser.add_argument(
        "--task-ids",
        type=str,
        dest="task_ids",
        help="Comma-separated list of specific task IDs to process",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between API calls in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        dest="no_skip_existing",
        help="Re-process tasks with existing feature files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Show execution plan and cost estimate without running",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run extraction in parallel (faster, uses async API calls)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Max concurrent API calls when --parallel is used (default: 10)",
    )


def _setup_aggregate_parser(subparsers) -> None:
    """Setup the 'aggregate' subcommand parser."""
    parser = subparsers.add_parser(
        "aggregate",
        help="Aggregate existing JSON files into CSV",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list_datasets(),
        help="Dataset for column ordering",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom PromptConfig Python file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        dest="output_dir",
        help="Directory containing JSON files",
    )


def _setup_quick_eval_parser(subparsers) -> None:
    """Setup the 'quick_eval' subcommand parser."""
    parser = subparsers.add_parser(
        "quick_eval",
        help="Quick evaluation of feature correlations with difficulty",
    )
    parser.add_argument(
        "--features-csv",
        type=str,
        required=True,
        dest="features_csv",
        help="Path to features CSV file",
    )
    parser.add_argument(
        "--irt-items",
        type=str,
        required=True,
        dest="irt_items",
        help="Path to IRT items.csv with difficulty column 'b'",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save results JSON",
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.05,
        dest="correlation_threshold",
        help="p-value threshold for significance (default: 0.05)",
    )
    parser.add_argument(
        "--redundancy-threshold",
        type=float,
        default=0.9,
        dest="redundancy_threshold",
        help="Correlation threshold for redundant pairs (default: 0.9)",
    )


def _setup_compare_parser(subparsers) -> None:
    """Setup the 'compare' subcommand parser."""
    parser = subparsers.add_parser(
        "compare",
        help="Compare hard vs easy tasks for qualitative validation",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name",
    )
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=1,
        dest="n_pairs",
        help="Number of pairs to show (default: 1)",
    )
    parser.add_argument(
        "--feature",
        type=str,
        help="Feature to highlight",
    )
    parser.add_argument(
        "--features-csv",
        type=str,
        dest="features_csv",
        help="Path to features CSV",
    )
    parser.add_argument(
        "--irt-items",
        type=str,
        dest="irt_items",
        help="Path to IRT items.csv",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed",
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Interactive mode (prompt for more pairs)",
    )


def _setup_correlations_parser(subparsers) -> None:
    """Setup the 'correlations' subcommand parser."""
    parser = subparsers.add_parser(
        "correlations",
        help="Analyze feature correlations with IRT difficulty",
    )
    parser.add_argument(
        "--features-csv",
        type=str,
        required=True,
        dest="features_csv",
        help="Path to features CSV file",
    )
    parser.add_argument(
        "--irt-items",
        type=str,
        required=True,
        dest="irt_items",
        help="Path to IRT items.csv",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="unknown",
        help="Dataset name for display",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save results JSON",
    )


def main():
    parser = argparse.ArgumentParser(
        description="LLM Judge feature extraction for task difficulty prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Setup all subcommand parsers
    _setup_extract_parser(subparsers)
    _setup_aggregate_parser(subparsers)
    _setup_quick_eval_parser(subparsers)
    _setup_compare_parser(subparsers)
    _setup_correlations_parser(subparsers)

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Dispatch to subcommand
    command_handlers = {
        "extract": cmd_extract,
        "aggregate": cmd_aggregate,
        "quick_eval": cmd_quick_eval,
        "compare": cmd_compare,
        "correlations": cmd_correlations,
    }

    if args.command in command_handlers:
        command_handlers[args.command](args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

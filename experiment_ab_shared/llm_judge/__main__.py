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


def load_swebench_pro_tasks() -> List[Dict[str, Any]]:
    """Load SWE-bench Pro dataset from HuggingFace.

    Data source: ScaleAI/SWE-bench_Pro

    Returns:
        List of task dictionaries with fields:
        - instance_id: Task identifier
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
        tasks.append({
            "instance_id": item["instance_id"],
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


def load_terminalbench_tasks(
    items_path: Optional[Path] = None,
    repo_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Load TerminalBench tasks from local files.

    TerminalBench tasks are stored in individual directories with:
    - task.yaml: Contains instruction, tags, and metadata
    - solution.sh: Reference solution script

    Args:
        items_path: Path to items CSV (oracle IRT items with task_id as index)
        repo_path: Path to terminal-bench repository root

    Returns:
        List of task dictionaries with task_id, instruction, solution, tags

    Raises:
        FileNotFoundError: If required files are missing
        ValueError: If any task is missing required fields (instruction, solution, tags)
    """
    import yaml

    # Default paths
    if items_path is None:
        items_path = Path("chris_output/terminal_bench_2.0/1d_1pl/items.csv")
    if repo_path is None:
        repo_path = Path("terminal-bench")

    if not items_path.exists():
        raise FileNotFoundError(f"Items file not found: {items_path}")
    if not repo_path.exists():
        raise FileNotFoundError(f"TerminalBench repo not found: {repo_path}")

    import pandas as pd

    # Load task IDs from items CSV (task_id is the index column)
    items_df = pd.read_csv(items_path, index_col=0)
    task_ids = items_df.index.tolist()

    print(f"Loading {len(task_ids)} TerminalBench tasks from {repo_path}...")

    tasks = []
    errors = []

    for task_id in task_ids:
        task_dir = repo_path / "tasks" / task_id

        # Load task.yaml (required - contains instruction and tags)
        task_yaml_path = task_dir / "task.yaml"
        if not task_yaml_path.exists():
            errors.append(f"Task {task_id}: missing task.yaml")
            continue

        with open(task_yaml_path) as f:
            task_yaml = yaml.safe_load(f)

        # Extract instruction (required)
        instruction = task_yaml.get("instruction", "").strip()
        if not instruction:
            errors.append(f"Task {task_id}: task.yaml missing instruction field")
            continue

        # Extract tags (optional - can be empty list, still useful context if present)
        tags = task_yaml.get("tags", [])

        # Load solution.sh (required)
        solution_path = task_dir / "solution.sh"
        if not solution_path.exists():
            errors.append(f"Task {task_id}: missing solution.sh")
            continue

        with open(solution_path) as f:
            solution = f.read().strip()

        if not solution:
            errors.append(f"Task {task_id}: solution.sh is empty")
            continue

        tasks.append({
            "task_id": task_id,
            "instruction": instruction,
            "solution": solution,
            "tags": tags,
        })

    # Fail loudly if any tasks are missing required data
    if errors:
        error_summary = "\n".join(errors[:10])
        if len(errors) > 10:
            error_summary += f"\n... and {len(errors) - 10} more errors"
        raise ValueError(
            f"Found {len(errors)} tasks with missing or empty required fields:\n{error_summary}"
        )

    print(f"Loaded {len(tasks)} tasks (all with instruction and solution)")
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
    if dataset == "swebench":
        return load_swebench_tasks()
    elif dataset in ("swebench_pro", "swebench_pro_v2", "swebench_pro_v3", "swebench_pro_v4", "swebench_pro_v5"):
        # V2/V3/V4/V5 use same data as swebench_pro, just different prompts
        return load_swebench_pro_tasks()
    elif dataset == "terminalbench":
        return load_terminalbench_tasks(items_path, repo_path)
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
    if args.dry_run:
        extractor.dry_run(
            tasks=tasks,
            limit=args.limit,
            task_ids=task_ids,
            skip_existing=not args.no_skip_existing,
        )
    elif args.parallel:
        extractor.run_parallel(
            tasks=tasks,
            skip_existing=not args.no_skip_existing,
            limit=args.limit,
            task_ids=task_ids,
            concurrency=args.concurrency,
        )
    else:
        extractor.run(
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

    # Extract subcommand
    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract LLM judge features from tasks",
    )
    extract_parser.add_argument(
        "--dataset",
        type=str,
        choices=list_datasets(),
        help="Built-in dataset to use",
    )
    extract_parser.add_argument(
        "--config",
        type=str,
        help="Path to custom PromptConfig Python file",
    )
    extract_parser.add_argument(
        "--tasks",
        type=str,
        help="Path to tasks JSONL file (for custom configs)",
    )
    extract_parser.add_argument(
        "--items-path",
        type=str,
        dest="items_path",
        help="Path to items CSV (TerminalBench only)",
    )
    extract_parser.add_argument(
        "--repo-path",
        type=str,
        dest="repo_path",
        help="Path to terminal-bench repo (TerminalBench only)",
    )
    extract_parser.add_argument(
        "--output-dir",
        type=str,
        dest="output_dir",
        help="Output directory for JSON files and CSV",
    )
    extract_parser.add_argument(
        "--provider",
        type=str,
        default="anthropic",
        choices=["anthropic", "openai"],
        help="LLM provider (default: anthropic)",
    )
    extract_parser.add_argument(
        "--model",
        type=str,
        help="Specific model to use (default: provider's default)",
    )
    extract_parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of tasks to process",
    )
    extract_parser.add_argument(
        "--task-ids",
        type=str,
        dest="task_ids",
        help="Comma-separated list of specific task IDs to process",
    )
    extract_parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between API calls in seconds (default: 0.5)",
    )
    extract_parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        dest="no_skip_existing",
        help="Re-process tasks with existing feature files",
    )
    extract_parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Show execution plan and cost estimate without running",
    )
    extract_parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run extraction in parallel (faster, uses async API calls)",
    )
    extract_parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Max concurrent API calls when --parallel is used (default: 10)",
    )

    # Aggregate subcommand
    aggregate_parser = subparsers.add_parser(
        "aggregate",
        help="Aggregate existing JSON files into CSV",
    )
    aggregate_parser.add_argument(
        "--dataset",
        type=str,
        choices=list_datasets(),
        help="Dataset for column ordering",
    )
    aggregate_parser.add_argument(
        "--config",
        type=str,
        help="Path to custom PromptConfig Python file",
    )
    aggregate_parser.add_argument(
        "--output-dir",
        type=str,
        dest="output_dir",
        help="Directory containing JSON files",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Dispatch to subcommand
    if args.command == "extract":
        cmd_extract(args)
    elif args.command == "aggregate":
        cmd_aggregate(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Remove headless-terminal task from all TerminalBench data files and retrain IRT.

This script:
1. Removes headless-terminal entries from JSONL response data files (both binary and raw)
2. Removes headless-terminal from llm_judge_features.csv
3. Deletes the individual headless-terminal.json feature file
4. Deletes ALL IRT directories for terminal bench (need complete retraining)
5. Removes the task directory from terminal-bench/tasks/
6. Retrains the primary binary IRT model (terminal_bench_2.0)

Usage:
    python scripts/remove_headless_terminal.py --dry-run  # Preview changes
    python scripts/remove_headless_terminal.py            # Apply changes
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASK_TO_REMOVE = "headless-terminal"


def remove_task_from_responses(file_path: Path, dry_run: bool) -> int:
    """Remove the task from each agent's responses in a JSONL file.

    The JSONL format has: {"subject_id": "agent", "responses": {"task1": ..., "task2": ...}}
    We need to remove TASK_TO_REMOVE from each responses dict.

    Returns number of entries modified.
    """
    if not file_path.exists():
        return 0

    lines = file_path.read_text().strip().split("\n")
    modified_count = 0
    new_lines = []

    for line in lines:
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            if "responses" in data and TASK_TO_REMOVE in data["responses"]:
                del data["responses"][TASK_TO_REMOVE]
                modified_count += 1
            new_lines.append(json.dumps(data))
        except json.JSONDecodeError:
            new_lines.append(line)

    if modified_count > 0:
        print(f"  {file_path}: removed task from {modified_count} agents")
        if not dry_run:
            file_path.write_text("\n".join(new_lines) + "\n" if new_lines else "")

    return modified_count


def remove_from_csv(file_path: Path, dry_run: bool) -> int:
    """Remove rows containing the task from a CSV file.

    Returns number of rows removed.
    """
    if not file_path.exists():
        return 0

    lines = file_path.read_text().strip().split("\n")
    original_count = len(lines)

    # Keep header and lines that don't contain the task
    filtered_lines = [lines[0]] if lines else []  # Keep header
    for line in lines[1:]:
        if TASK_TO_REMOVE not in line:
            filtered_lines.append(line)

    removed = original_count - len(filtered_lines)

    if removed > 0:
        print(f"  {file_path}: removing {removed} rows")
        if not dry_run:
            file_path.write_text("\n".join(filtered_lines) + "\n" if filtered_lines else "")

    return removed


def remove_from_meta_json(file_path: Path, dry_run: bool) -> bool:
    """Remove the task from meta.json agent results.

    Returns True if task was found and would be removed.
    """
    if not file_path.exists():
        return False

    data = json.loads(file_path.read_text())
    found = False

    # The meta.json has agent_results with task ratings
    for agent_name, agent_data in data.get("agent_results", {}).items():
        if TASK_TO_REMOVE in agent_data:
            found = True
            if not dry_run:
                del agent_data[TASK_TO_REMOVE]

    if found:
        print(f"  {file_path}: removing {TASK_TO_REMOVE} from agent results")
        if not dry_run:
            file_path.write_text(json.dumps(data, indent=2))

    return found


def delete_file(file_path: Path, dry_run: bool) -> bool:
    """Delete a file if it exists."""
    if file_path.exists():
        print(f"  Deleting: {file_path}")
        if not dry_run:
            file_path.unlink()
        return True
    return False


def delete_directory(dir_path: Path, dry_run: bool) -> bool:
    """Delete a directory and its contents if it exists."""
    if dir_path.exists():
        print(f"  Deleting directory: {dir_path}")
        if not dry_run:
            shutil.rmtree(dir_path)
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Remove headless-terminal from TerminalBench data")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying")
    parser.add_argument("--skip-retrain", action="store_true", help="Skip IRT retraining step")
    args = parser.parse_args()

    if args.dry_run:
        print("=== DRY RUN MODE - No changes will be made ===\n")

    # 1. Primary response data files
    print("1. Removing task from response data files...")
    data_dir = ROOT / "data" / "terminal_bench"

    # Both binary and raw (binomial) versions need to be updated
    remove_task_from_responses(data_dir / "terminal_bench_2.0.jsonl", args.dry_run)
    remove_task_from_responses(data_dir / "terminal_bench_2.0_raw.jsonl", args.dry_run)
    remove_from_meta_json(data_dir / "terminal_bench_2.0.meta.json", args.dry_run)

    # 2. Delete ALL IRT model directories (they need complete retraining)
    print("\n2. Deleting IRT model directories (need complete retraining)...")

    # Primary terminal bench IRT outputs - delete all of them
    delete_directory(ROOT / "chris_output" / "terminal_bench_2.0", args.dry_run)
    delete_directory(ROOT / "chris_output" / "terminal_bench_2.0_binomial", args.dry_run)
    delete_directory(ROOT / "chris_output" / "terminal_bench_2.0_binomial_1pl", args.dry_run)

    # Experiment A terminalbench IRT splits
    delete_directory(ROOT / "chris_output" / "experiment_a_terminalbench" / "irt_splits", args.dry_run)

    # Experiment B terminalbench baseline IRT cache
    delete_directory(ROOT / "chris_output" / "experiment_b" / "terminalbench" / "baseline_irt", args.dry_run)

    # Binomial vs binary experiment
    delete_directory(ROOT / "chris_output" / "experiment_a_terminalbench" / "binomial_vs_binary", args.dry_run)

    # 3. LLM judge feature files
    print("\n3. Removing from LLM judge features...")
    llm_judge_dir = ROOT / "chris_output" / "experiment_a_terminalbench" / "llm_judge_features"

    delete_file(llm_judge_dir / "headless-terminal.json", args.dry_run)
    remove_from_csv(llm_judge_dir / "llm_judge_features.csv", args.dry_run)

    # 4. Output directory files (various experiment outputs)
    print("\n4. Removing from output directory files...")
    out_dirs = [
        ROOT / "out" / "multi_benchmark_evaluate_in_distribution",
        ROOT / "out" / "multi_benchmark" / "irt_model_scaffold_1pl",
        ROOT / "out" / "terminal_bench",
        ROOT / "out" / "chris_irt",
    ]

    for out_dir in out_dirs:
        if out_dir.exists():
            for csv_file in out_dir.glob("*.csv"):
                remove_from_csv(csv_file, args.dry_run)
            for jsonl_file in out_dir.glob("*.jsonl"):
                remove_task_from_responses(jsonl_file, args.dry_run)

    # 5. Delete the task directory from terminal-bench repo
    print("\n5. Deleting task directory from terminal-bench repo...")
    task_dir = ROOT / "terminal-bench" / "tasks" / TASK_TO_REMOVE
    delete_directory(task_dir, args.dry_run)

    # 6. Retrain primary IRT model (binary version - the default)
    print("\n6. Retraining primary IRT model (terminal_bench_2.0 - binary)...")
    if args.dry_run:
        print("  [DRY RUN] Would run: python swebench_irt/train.py --dims 1 --model 1pl \\")
        print("      --data_path data/terminal_bench/terminal_bench_2.0.jsonl \\")
        print("      --output_dir chris_output/terminal_bench_2.0")
    elif not args.skip_retrain:
        cmd = [
            sys.executable, "-m", "swebench_irt.train",
            "--dims", "1",
            "--model", "1pl",
            "--data_path", str(data_dir / "terminal_bench_2.0.jsonl"),
            "--output_dir", str(ROOT / "chris_output" / "terminal_bench_2.0"),
        ]
        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            print("  ERROR: IRT training failed!")
            sys.exit(1)
    else:
        print("  [SKIPPED] --skip-retrain flag was set")

    print("\n" + "=" * 60)
    if args.dry_run:
        print("DRY RUN COMPLETE - Run without --dry-run to apply changes")
    else:
        print("CLEANUP COMPLETE")
        print("\nPrimary binary IRT model has been retrained.")
        print("Cached splits for experiments will be regenerated automatically when needed.")
        print("\nNote: Binomial IRT (terminal_bench_2.0_binomial_1pl) not retrained.")
        print("Re-run if you need binomial mode for experiments.")


if __name__ == "__main__":
    main()

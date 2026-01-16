"""
Run pass@k experiments: k attempts per (model, task) pair.

This script runs agents multiple times on each task to measure pass@k rates,
then correlates with IRT difficulty scores.

Runs are parallelized using concurrent.futures for efficiency.

Usage:
    # Run M1 model (o1) on selected tasks with k=10 attempts
    python -m experiment_pass_at_k.run_pass_k --model openai/o1-2024-12-05

    # Run M2 model (o3) on selected tasks
    python -m experiment_pass_at_k.run_pass_k --model openai/o3-2025-04-16

    # Custom k value and parallelism
    python -m experiment_pass_at_k.run_pass_k --model openai/o1-2024-12-05 --k 20 --parallel 30

    # Resume from previous run
    python -m experiment_pass_at_k.run_pass_k --model openai/o1-2024-12-05 --resume

    # Run specific tasks only
    python -m experiment_pass_at_k.run_pass_k --model openai/o1-2024-12-05 --task_ids "django__django-15315,sympy__sympy-12481"
"""

import argparse
import json
import subprocess
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

from experiment_pass_at_k.config import ExperimentPassKConfig

# Pricing per 1M tokens (as of Jan 2026)
# Sources: https://openai.com/api/pricing/, https://platform.openai.com/docs/pricing
PRICING = {
    "openai/o1-2024-12-17": {"input": 15.0, "output": 60.0},  # o1 pricing
    "openai/o3-2025-04-16": {"input": 2.0, "output": 8.0},    # o3 pricing (post June 2025 80% discount)
    "openai/o4-mini": {"input": 1.10, "output": 4.40},        # o4-mini pricing
    "openai/gpt-5-mini": {"input": 0.25, "output": 2.0},      # gpt-5-mini pricing
    "openai/o1-preview-2024-09-12": {"input": 15.0, "output": 60.0},
    "openai/gpt-4o-2024-08-06": {"input": 2.5, "output": 10.0},
    "openai/gpt-4o": {"input": 2.5, "output": 10.0},
}

# Lock for thread-safe printing
print_lock = Lock()


def safe_print(*args, **kwargs):
    """Thread-safe print."""
    with print_lock:
        print(*args, **kwargs)


def get_docker_image_name(task_id: str) -> str:
    """Get the Docker image name for a SWE-bench task.

    Format: swebench/sweb.eval.x86_64.<repo>_1776_<repo>-<issue>:latest
    Example: django__django-16569 -> swebench/sweb.eval.x86_64.django_1776_django-16569:latest
    """
    repo = task_id.split("__")[0]
    return f"swebench/sweb.eval.x86_64.{repo}_1776_{task_id}:latest"


def delete_docker_image(image_name: str) -> bool:
    """Delete a Docker image. Returns True if successful."""
    try:
        result = subprocess.run(
            ["docker", "rmi", image_name],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            safe_print(f"  [Docker] Deleted image: {image_name}")
            return True
        return False
    except Exception as e:
        safe_print(f"  [Docker] Error deleting {image_name}: {e}")
        return False


def get_repo_from_task_id(task_id: str) -> str:
    """Extract repo name from task_id (e.g., 'django__django-16569' -> 'django')."""
    return task_id.split("__")[0]


def sort_tasks_by_repo(tasks: List[Dict]) -> List[Dict]:
    """Sort tasks by repo to group tasks using similar Docker images."""
    return sorted(tasks, key=lambda t: get_repo_from_task_id(t["task_id"]))


def extract_result_from_eval_log(log_path: Path, model: str) -> Dict:
    """Extract success/failure and stats from an .eval log file.

    Inspect .eval files are ZIP archives containing:
    - header.json: stats, model_usage
    - samples.json: individual sample results with scores

    Returns:
        Dict with success, tokens, cost, etc.
    """
    result = {}

    try:
        with zipfile.ZipFile(log_path, 'r') as zf:
            # Extract header for stats
            with zf.open('header.json') as f:
                header = json.load(f)

            # Extract model usage
            if "stats" in header and "model_usage" in header["stats"]:
                model_usage = header["stats"]["model_usage"]
                if model in model_usage:
                    usage = model_usage[model]
                    result["input_tokens"] = usage.get("input_tokens", 0)
                    result["output_tokens"] = usage.get("output_tokens", 0)
                    result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
                    result["input_tokens_cache_read"] = usage.get("input_tokens_cache_read", 0)

                    # Calculate cost
                    if model in PRICING:
                        pricing = PRICING[model]
                        input_cost = (result["input_tokens"] / 1_000_000) * pricing["input"]
                        output_cost = (result["output_tokens"] / 1_000_000) * pricing["output"]
                        result["cost_usd"] = input_cost + output_cost

            # Extract success from samples
            # Inspect uses either samples.json OR samples/<task_id>_epoch_<n>.json
            sample = None
            if 'samples.json' in zf.namelist():
                with zf.open('samples.json') as f:
                    samples = json.load(f)
                    if samples and len(samples) > 0:
                        sample = samples[0]
            else:
                # Look for samples/<task_id>_epoch_1.json pattern
                sample_files = [n for n in zf.namelist() if n.startswith('samples/') and n.endswith('.json')]
                if sample_files:
                    with zf.open(sample_files[0]) as f:
                        sample = json.load(f)

            if sample and "scores" in sample:
                # SWE-bench uses a custom scorer
                # Success is typically indicated by a score > 0
                scores = sample["scores"]
                if scores:
                    # Get the first score value
                    for scorer_name, score_data in scores.items():
                        if isinstance(score_data, dict) and "value" in score_data:
                            result["success"] = score_data["value"] == 1 or score_data["value"] == 1.0 or score_data["value"] == "C"
                            result["score"] = score_data["value"]
                            break
                        elif isinstance(score_data, (int, float)):
                            result["success"] = score_data == 1 or score_data == 1.0
                            result["score"] = score_data
                            break

    except Exception as e:
        result["extraction_error"] = str(e)

    return result


def run_single_attempt(
    model: str,
    task_id: str,
    attempt_num: int,
    message_limit: int = 50,
    sandbox: str = "docker",
    log_dir: Path = None,
) -> Dict:
    """Run one attempt of an agent on a task.

    Returns:
        Dict with attempt details: success, timing, tokens, cost, etc.
    """
    result = {
        "attempt": attempt_num,
        "task_id": task_id,
        "timestamp": datetime.now().isoformat(),
    }

    start_time = time.time()

    # Build inspect eval command
    # Use instance_ids to limit which Docker images are built (only the one we need)
    # This prevents inspect from trying to pull/build all 500 SWE-bench images
    cmd = [
        "inspect",
        "eval",
        "inspect_evals/swe_bench",
        "-T",
        "dataset=princeton-nlp/SWE-bench_Verified",
        "-T",
        f"instance_ids=['{task_id}']",
        "--model",
        model,
        "--sandbox",
        sandbox,
        "--message-limit",
        str(message_limit),
    ]

    # Add log directory if specified
    if log_dir:
        cmd.extend(["--log-dir", str(log_dir)])

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        result["elapsed_seconds"] = time.time() - start_time
        result["return_code"] = proc.returncode

        if proc.returncode != 0:
            result["error"] = (proc.stderr or proc.stdout)[:500]
            result["success"] = False
            safe_print(f"  [{task_id}] Attempt {attempt_num}: FAILED ({result['elapsed_seconds']:.1f}s)")
        else:
            # Extract results from the most recent .eval log in our log dir
            search_dir = log_dir if log_dir else Path("logs")
            if search_dir.exists():
                log_files = sorted(search_dir.glob("*.eval"), key=lambda p: p.stat().st_mtime, reverse=True)
                if log_files:
                    eval_result = extract_result_from_eval_log(log_files[0], model)
                    result.update(eval_result)
                    # Store log file path for reference
                    result["log_file"] = str(log_files[0])

            success = result.get("success", False)
            tokens = result.get("total_tokens", 0)
            cost = result.get("cost_usd", 0)

            status = "SUCCESS" if success else "FAIL"
            safe_print(f"  [{task_id}] Attempt {attempt_num}: {status} ({result['elapsed_seconds']:.1f}s, {tokens:,} tokens, ${cost:.4f})")

    except Exception as e:
        result["elapsed_seconds"] = time.time() - start_time
        result["error"] = str(e)
        result["success"] = False
        safe_print(f"  [{task_id}] Attempt {attempt_num}: ERROR - {e}")

    return result


def run_single_job(job: Tuple) -> Dict:
    """Run a single (task, attempt) job. Used for parallel execution."""
    model, task_id, attempt_num, message_limit, sandbox, log_dir = job
    return run_single_attempt(
        model=model,
        task_id=task_id,
        attempt_num=attempt_num,
        message_limit=message_limit,
        sandbox=sandbox,
        log_dir=log_dir,
    )


def load_existing_results(output_dir: Path, model: str, task_id: str) -> Optional[Dict]:
    """Load existing results for a task if available."""
    model_short = model.split("/")[-1]
    task_file = output_dir / f"results_{model_short}" / f"{task_id}.json"

    if task_file.exists():
        with open(task_file) as f:
            return json.load(f)
    return None


def save_task_result(
    output_dir: Path,
    model: str,
    task_id: str,
    irt_difficulty: float,
    k: int,
    attempts: List[Dict],
):
    """Save results for a single task."""
    # Compute summary
    successes = sum(1 for a in attempts if a.get("success", False))
    pass_rate = successes / len(attempts) if attempts else 0

    # Find first success
    first_success = None
    for a in sorted(attempts, key=lambda x: x["attempt"]):
        if a.get("success", False):
            first_success = a["attempt"]
            break

    summary = {
        "total_attempts": len(attempts),
        "successful_attempts": successes,
        "pass_rate": pass_rate,
        "first_success_at": first_success,
        "total_cost_usd": sum(a.get("cost_usd", 0) for a in attempts),
        "total_time_seconds": sum(a.get("elapsed_seconds", 0) for a in attempts),
        "total_tokens": sum(a.get("total_tokens", 0) for a in attempts),
    }

    result = {
        "task_id": task_id,
        "model": model,
        "irt_difficulty": irt_difficulty,
        "k": k,
        "attempts": sorted(attempts, key=lambda x: x["attempt"]),
        "summary": summary,
    }

    # Save to task-specific file
    model_short = model.split("/")[-1]
    task_file = output_dir / f"results_{model_short}" / f"{task_id}.json"
    task_file.parent.mkdir(parents=True, exist_ok=True)

    with open(task_file, "w") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="Run pass@k experiments")
    parser.add_argument(
        "--model",
        type=str,
        help="Model to use. If not specified, uses config default.",
    )
    parser.add_argument(
        "--tasks_file",
        type=str,
        default="chris_output/experiment_pass_at_k/selected_tasks.json",
        help="JSON file with selected tasks",
    )
    parser.add_argument(
        "--k",
        type=int,
        help="Number of attempts per task (default: from config)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous run, continuing incomplete tasks",
    )
    parser.add_argument(
        "--task_ids",
        type=str,
        help="Comma-separated list of specific task IDs to run",
    )
    parser.add_argument(
        "--message_limit",
        type=int,
        help="Message limit for agent (default: from config)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=10,
        help="Number of parallel attempts per task (default: 10)",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        default=True,
        help="Delete Docker image after completing all attempts for each task (default: True)",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_false",
        dest="cleanup",
        help="Keep Docker images after task completion",
    )
    parser.add_argument(
        "--parallel-tasks",
        type=int,
        default=1,
        help="Number of tasks to run in parallel (default: 1 = sequential). Higher values use more disk for Docker images.",
    )
    args = parser.parse_args()

    # Load config
    config = ExperimentPassKConfig()

    # Override config with CLI args
    model = args.model or config.m1_model
    k = args.k or config.k_attempts
    message_limit = args.message_limit or config.message_limit
    max_parallel = args.parallel
    cleanup_images = args.cleanup
    parallel_tasks = args.parallel_tasks

    # Load tasks
    if args.task_ids:
        task_ids = [t.strip() for t in args.task_ids.split(",")]
        # Load IRT difficulties for these tasks
        import pandas as pd
        items_df = pd.read_csv(config.items_path, index_col=0)
        tasks = [
            {"task_id": tid, "difficulty": float(items_df.loc[tid, "b"])}
            for tid in task_ids
            if tid in items_df.index
        ]
    else:
        with open(args.tasks_file) as f:
            tasks_data = json.load(f)
        tasks = tasks_data["tasks"]

    # Sort tasks by repo to maximize Docker layer cache hits between tasks
    tasks = sort_tasks_by_repo(tasks)

    print("=" * 60)
    print("PASS@K EXPERIMENT")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"K (attempts per task): {k}")
    print(f"Tasks: {len(tasks)}")
    print(f"Total runs planned: {len(tasks) * k}")
    print(f"Parallel attempts per task: {max_parallel}")
    print(f"Docker cleanup after each task: {cleanup_images}")
    print(f"Output dir: {config.output_dir}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)

    # Create log directory
    base_log_dir = config.output_dir / "logs"
    base_log_dir.mkdir(parents=True, exist_ok=True)

    # Build list of tasks to run with their needed attempts
    tasks_to_run = []
    for task in tasks:
        task_id = task["task_id"]
        difficulty = task["difficulty"]

        # Check for existing results
        existing_attempts = []
        if args.resume:
            existing = load_existing_results(config.output_dir, model, task_id)
            if existing:
                existing_attempts = existing.get("attempts", [])
                if len(existing_attempts) >= k:
                    print(f"  Skipping {task_id} (already have {len(existing_attempts)} attempts)")
                    continue

        # Determine which attempts are needed
        existing_nums = {a["attempt"] for a in existing_attempts}
        needed_attempts = [i for i in range(1, k + 1) if i not in existing_nums]

        if not needed_attempts:
            continue

        tasks_to_run.append({
            "task_id": task_id,
            "difficulty": difficulty,
            "existing": existing_attempts,
            "needed": needed_attempts,
        })

    total_jobs = sum(len(t["needed"]) for t in tasks_to_run)
    print(f"\nTasks to run: {len(tasks_to_run)}")
    print(f"Total attempts needed: {total_jobs}")

    if not tasks_to_run:
        print("No work to do!")
        return

    # Process tasks in batches based on parallel_tasks setting
    print(f"\nProcessing tasks (parallel_tasks={parallel_tasks}, parallel_attempts={max_parallel}, cleanup={cleanup_images})...")
    print("-" * 60)

    all_results = []
    completed_jobs = 0
    start_time = time.time()

    # Process tasks in batches
    for batch_start in range(0, len(tasks_to_run), parallel_tasks):
        batch = tasks_to_run[batch_start:batch_start + parallel_tasks]
        batch_task_ids = [t["task_id"] for t in batch]

        print(f"\n=== Batch {batch_start // parallel_tasks + 1}: {len(batch)} tasks ===")
        for t in batch:
            print(f"  - {t['task_id']} ({len(t['needed'])} attempts needed)")

        # Build ALL jobs for ALL tasks in this batch
        all_jobs = []
        job_to_task = {}  # Map job to task info for result aggregation
        for task_info in batch:
            task_id = task_info["task_id"]
            for attempt_num in task_info["needed"]:
                job_log_dir = base_log_dir / f"{task_id}_attempt{attempt_num}"
                job_log_dir.mkdir(parents=True, exist_ok=True)
                job = (
                    model,
                    task_id,
                    attempt_num,
                    message_limit,
                    config.sandbox,
                    job_log_dir,
                )
                all_jobs.append(job)
                job_to_task[job] = task_info

        # Initialize results storage for each task in batch
        batch_results = {t["task_id"]: t["existing"].copy() for t in batch}

        # Run ALL jobs for ALL tasks in batch in parallel
        total_workers = min(max_parallel * parallel_tasks, len(all_jobs))
        print(f"\nRunning {len(all_jobs)} jobs with {total_workers} workers...")

        with ThreadPoolExecutor(max_workers=total_workers) as executor:
            futures = {executor.submit(run_single_job, job): job for job in all_jobs}
            for future in as_completed(futures):
                job = futures[future]
                task_id = job[1]  # task_id is second element of job tuple
                try:
                    result = future.result()
                    batch_results[task_id].append(result)
                    completed_jobs += 1

                    # Progress update
                    elapsed = time.time() - start_time
                    rate = completed_jobs / elapsed if elapsed > 0 else 0
                    eta = (total_jobs - completed_jobs) / rate if rate > 0 else 0
                    safe_print(f"  [{completed_jobs}/{total_jobs}] {task_id} attempt {result.get('attempt', '?')}: {'SUCCESS' if result.get('success') else 'FAIL'} - ETA: {eta/60:.1f} min")
                except Exception as e:
                    safe_print(f"  ERROR on {task_id}: {e}")
                    completed_jobs += 1

        # Save results for each task in batch
        for task_info in batch:
            task_id = task_info["task_id"]
            task_results = batch_results[task_id]
            task_result = save_task_result(
                output_dir=config.output_dir,
                model=model,
                task_id=task_id,
                irt_difficulty=task_info["difficulty"],
                k=k,
                attempts=task_results,
            )
            all_results.append(task_result)
            successes = sum(1 for a in task_results if a.get("success", False))
            print(f"  {task_id}: {successes}/{len(task_results)} passed")

        # Delete Docker images for tasks in this batch
        if cleanup_images:
            print(f"\nCleaning up Docker images for batch...")
            for task_id in batch_task_ids:
                image_name = get_docker_image_name(task_id)
                delete_docker_image(image_name)

    # Also include tasks that were already complete (skipped above)
    for task in tasks:
        task_id = task["task_id"]
        already_in_results = any(r["task_id"] == task_id for r in all_results)
        if not already_in_results:
            existing = load_existing_results(config.output_dir, model, task_id)
            if existing:
                all_results.append(existing)

    # Save summary
    model_short = model.split("/")[-1]
    summary_file = config.output_dir / f"results_{model_short}" / "summary.json"

    total_elapsed = time.time() - start_time

    summary = {
        "model": model,
        "k": k,
        "n_tasks": len(all_results),
        "parallel_workers": max_parallel,
        "wall_clock_seconds": total_elapsed,
        "timestamp": datetime.now().isoformat(),
        "tasks": [
            {
                "task_id": r["task_id"],
                "irt_difficulty": r["irt_difficulty"],
                "pass_rate": r["summary"]["pass_rate"],
                "first_success_at": r["summary"]["first_success_at"],
                "total_cost_usd": r["summary"]["total_cost_usd"],
            }
            for r in all_results
        ],
        "totals": {
            "total_attempts": sum(r["summary"]["total_attempts"] for r in all_results),
            "total_successes": sum(r["summary"]["successful_attempts"] for r in all_results),
            "total_cost_usd": sum(r["summary"]["total_cost_usd"] for r in all_results),
            "total_time_seconds": sum(r["summary"]["total_time_seconds"] for r in all_results),
        },
    }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Print final summary
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Tasks: {summary['n_tasks']}")
    print(f"Total attempts: {summary['totals']['total_attempts']}")
    print(f"Total successes: {summary['totals']['total_successes']}")
    if summary['totals']['total_attempts'] > 0:
        print(f"Overall pass rate: {summary['totals']['total_successes'] / summary['totals']['total_attempts']:.1%}")
    print(f"Total cost: ${summary['totals']['total_cost_usd']:.2f}")
    print(f"Total compute time: {summary['totals']['total_time_seconds'] / 60:.1f} min")
    print(f"Wall clock time: {total_elapsed / 60:.1f} min")
    print(f"Speedup from parallelization: {summary['totals']['total_time_seconds'] / total_elapsed:.1f}x")
    print(f"\nResults saved to {summary_file}")


if __name__ == "__main__":
    main()

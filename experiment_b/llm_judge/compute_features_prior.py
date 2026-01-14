"""Compute LLM judge prior features (problem-only, no trajectory).

This script extracts the 9 problem-level features for tasks in D_train and D_valid.
These features establish the baseline for comparing against trajectory-augmented models.

Usage:
    # Dry run to see execution plan
    python -m experiment_b.llm_judge.compute_features_prior --dry_run

    # Run on small subset for validation
    python -m experiment_b.llm_judge.compute_features_prior --limit 5

    # Full run
    python -m experiment_b.llm_judge.compute_features_prior

    # Use specific model
    python -m experiment_b.llm_judge.compute_features_prior --model gpt-4o-mini
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add parent to path for imports
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_b.config import ExperimentConfig
from experiment_b.data_splits import create_experiment_split
from experiment_b.llm_judge.features_prior import (
    LLM_JUDGE_PRIOR_FEATURE_NAMES,
    LLMJudgePriorFeatures,
    format_prior_prompt,
)


# Output directory
OUTPUT_DIR = ROOT / "chris_output" / "experiment_b" / "llm_judge_prior_features"


def load_swebench_tasks() -> Dict[str, dict]:
    """Load SWE-bench Verified task metadata.

    Returns:
        Dict mapping task_id -> task dict with all metadata
    """
    try:
        from datasets import load_dataset
        print("Loading SWE-bench Verified tasks...")
        ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
        tasks = {
            item["instance_id"]: {
                "instance_id": item["instance_id"],
                "repo": item["repo"],
                "version": item["version"],
                "problem_statement": item["problem_statement"],
                "patch": item["patch"],
                "hints_text": item["hints_text"],
                "FAIL_TO_PASS": item["FAIL_TO_PASS"],
                "PASS_TO_PASS": item["PASS_TO_PASS"],
            }
            for item in ds
        }
        print(f"Loaded {len(tasks)} tasks")
        return tasks
    except ImportError:
        print("Error: datasets library not available")
        return {}


def call_llm(prompt: str, model: str = "gpt-4o-mini") -> Optional[dict]:
    """Call LLM API to get judge response.

    Args:
        prompt: The prompt to send
        model: Model to use

    Returns:
        Parsed JSON response or None on failure
    """
    try:
        if model.startswith("claude") or model.startswith("anthropic"):
            content = _call_anthropic(prompt, model)
        else:
            content = _call_openai(prompt, model)

        if not content:
            return None

        # Parse JSON from response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        return json.loads(content.strip())

    except Exception as e:
        print(f"    LLM call failed: {e}")
        return None


def _call_anthropic(prompt: str, model: str) -> Optional[str]:
    """Call Anthropic API."""
    import anthropic
    client = anthropic.Anthropic()

    response = client.messages.create(
        model=model,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )

    if response.content and len(response.content) > 0:
        return response.content[0].text
    return None


def _call_openai(prompt: str, model: str) -> Optional[str]:
    """Call OpenAI API."""
    from openai import OpenAI
    client = OpenAI()

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=500,
    )

    if response.choices and response.choices[0].message.content:
        return response.choices[0].message.content
    return None


def compute_features_for_task(
    task_id: str,
    task_info: dict,
    model: str = "gpt-4o-mini",
) -> Optional[dict]:
    """Compute prior features for a single task.

    Args:
        task_id: Task instance ID
        task_info: Task metadata
        model: LLM model to use

    Returns:
        Feature dict or None on failure
    """
    prompt = format_prior_prompt(
        instance_id=task_id,
        repo=task_info["repo"],
        version=task_info.get("version", "unknown"),
        problem_statement=task_info["problem_statement"],
        patch=task_info["patch"],
        fail_to_pass=task_info.get("FAIL_TO_PASS", "[]"),
        pass_to_pass=task_info.get("PASS_TO_PASS", "[]"),
        hints_text=task_info.get("hints_text", ""),
    )

    response = call_llm(prompt, model=model)
    if response is None:
        return None

    # Validate response has required fields
    for field in LLM_JUDGE_PRIOR_FEATURE_NAMES:
        if field not in response:
            print(f"    Missing field: {field}")
            return None

    return response


def main():
    parser = argparse.ArgumentParser(description="Compute LLM judge prior features")
    parser.add_argument("--dry_run", action="store_true",
                        help="Show plan without computing")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of tasks to process")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                        help="Skip tasks that already have features (default: True)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="LLM model to use (default: gpt-4o-mini)")
    parser.add_argument("--rate_limit_delay", type=float, default=0.5,
                        help="Delay between API calls in seconds (default: 0.5)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check for API key based on model
    if args.model.startswith("claude") or args.model.startswith("anthropic"):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("Error: ANTHROPIC_API_KEY environment variable not set")
            return
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            print("Error: OPENAI_API_KEY environment variable not set")
            return

    # Load config and create splits
    config = ExperimentConfig()
    print("Creating experiment splits...")
    split = create_experiment_split(
        responses_path=ROOT / config.responses_path,
        trajectories_dir=ROOT / config.trajectories_dir,
        weak_threshold=config.weak_threshold,
        strong_min_improvement=config.strong_min_improvement,
        m1_fraction=config.m1_fraction,
        m2_fraction=config.m2_fraction,
    )

    # Collect tasks to process (D_train + D_valid)
    all_tasks = list(set(split.d_train_tasks) | set(split.d_valid_tasks))

    print(f"\nTotal tasks: {len(all_tasks)}")
    print(f"  D_train tasks: {len(split.d_train_tasks)}")
    print(f"  D_valid tasks: {len(split.d_valid_tasks)}")

    # Filter out existing
    if args.skip_existing:
        filtered_tasks = []
        for task_id in all_tasks:
            output_file = OUTPUT_DIR / f"{task_id}.json"
            if not output_file.exists():
                filtered_tasks.append(task_id)
        print(f"\nAfter skipping existing: {len(filtered_tasks)} tasks to process")
        all_tasks = filtered_tasks

    if args.limit > 0:
        all_tasks = all_tasks[:args.limit]
        print(f"Limited to {len(all_tasks)} tasks")

    if args.dry_run:
        print(f"\n=== DRY RUN - Would process {len(all_tasks)} tasks ===")
        print(f"Model: {args.model}")
        print(f"Rate limit delay: {args.rate_limit_delay}s")
        print(f"Output directory: {OUTPUT_DIR}")
        print(f"\nSample tasks (first 10):")
        for task_id in all_tasks[:10]:
            print(f"  {task_id}")
        if len(all_tasks) > 10:
            print(f"  ... and {len(all_tasks) - 10} more tasks")
        return

    # Load task metadata
    tasks = load_swebench_tasks()
    if not tasks:
        print("Failed to load task metadata")
        return

    # Process tasks
    print(f"\nComputing prior features using {args.model}...")
    start_time = datetime.now()

    processed = 0
    skipped = 0
    failed = 0

    for task_id in all_tasks:
        output_file = OUTPUT_DIR / f"{task_id}.json"

        if args.skip_existing and output_file.exists():
            skipped += 1
            continue

        if task_id not in tasks:
            print(f"  Skipping {task_id}: no task metadata")
            skipped += 1
            continue

        print(f"[{processed + failed + 1}/{len(all_tasks)}] {task_id}...")

        # Compute features
        result = compute_features_for_task(
            task_id, tasks[task_id], model=args.model
        )

        if result is None:
            failed += 1
            print(f"    FAILED")
            continue

        # Add metadata and save
        result["task_id"] = task_id
        result["model"] = args.model
        result["computed_at"] = datetime.now().isoformat()

        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)

        processed += 1
        print(f"    -> fix_complexity={result.get('fix_complexity')}, "
              f"domain_knowledge={result.get('domain_knowledge_required')}")

        # Rate limiting
        time.sleep(args.rate_limit_delay)

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\nDone! Processed {processed}, skipped {skipped}, failed {failed}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

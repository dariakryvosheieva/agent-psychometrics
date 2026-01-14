"""Compute LLM judge combined features (problem + trajectory).

This script extracts 13 features (9 problem + 4 trajectory) for tasks in D_train and D_valid.
It uses a single agent's trajectories, auto-selected for best coverage.

Usage:
    # Dry run to see execution plan and agent selection
    python -m experiment_b.llm_judge.compute_features_combined --dry_run

    # Run on small subset for validation
    python -m experiment_b.llm_judge.compute_features_combined --limit 5

    # Full run
    python -m experiment_b.llm_judge.compute_features_combined

    # Use specific agent instead of auto-select
    python -m experiment_b.llm_judge.compute_features_combined --agent 20240402_sweagent_gpt4
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent to path for imports
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_b.config import ExperimentConfig
from experiment_b.data_splits import create_experiment_split
from experiment_b.llm_judge.features_combined import (
    LLM_JUDGE_COMBINED_FEATURE_NAMES,
    LLMJudgeCombinedFeatures,
    format_combined_prompt,
)


# Directories
UNIFIED_TRAJS_DIR = ROOT / "trajectory_data" / "unified_trajs"
OUTPUT_DIR = ROOT / "chris_output" / "experiment_b" / "llm_judge_combined_features"


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


def load_trajectory(agent: str, task_id: str) -> Optional[dict]:
    """Load a unified trajectory JSON file."""
    traj_path = UNIFIED_TRAJS_DIR / agent / f"{task_id}.json"
    if not traj_path.exists():
        return None

    try:
        with open(traj_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def select_best_agent(
    m1_agents: List[str],
    task_ids: List[str],
) -> Tuple[str, int, int]:
    """Select the M1 agent with best coverage of tasks.

    Args:
        m1_agents: List of M1 (oldest) agent names
        task_ids: List of task IDs to check coverage for

    Returns:
        Tuple of (best_agent_name, coverage_count, failure_count)
    """
    print("\nAnalyzing agent trajectory coverage...")

    best_agent = None
    best_coverage = 0
    best_failures = 0

    for agent in m1_agents:
        agent_dir = UNIFIED_TRAJS_DIR / agent
        if not agent_dir.exists():
            continue

        coverage = 0
        failures = 0

        for task_id in task_ids:
            traj_path = agent_dir / f"{task_id}.json"
            if traj_path.exists():
                coverage += 1
                # Check if agent failed (more informative trajectories)
                try:
                    with open(traj_path) as f:
                        traj = json.load(f)
                    if not traj.get("resolved", False):
                        failures += 1
                except (json.JSONDecodeError, IOError):
                    pass

        if coverage > best_coverage:
            best_coverage = coverage
            best_failures = failures
            best_agent = agent

        if coverage > 0:
            print(f"  {agent}: {coverage}/{len(task_ids)} tasks ({failures} failures)")

    return best_agent, best_coverage, best_failures


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
        max_tokens=800,
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
        max_tokens=800,
    )

    if response.choices and response.choices[0].message.content:
        return response.choices[0].message.content
    return None


def compute_features_for_task(
    task_id: str,
    task_info: dict,
    trajectory: dict,
    model: str = "gpt-4o-mini",
) -> Optional[dict]:
    """Compute combined features for a single task.

    Args:
        task_id: Task instance ID
        task_info: Task metadata
        trajectory: Agent trajectory
        model: LLM model to use

    Returns:
        Feature dict or None on failure
    """
    prompt = format_combined_prompt(
        instance_id=task_id,
        repo=task_info["repo"],
        version=task_info.get("version", "unknown"),
        problem_statement=task_info["problem_statement"],
        patch=task_info["patch"],
        fail_to_pass=task_info.get("FAIL_TO_PASS", "[]"),
        hints_text=task_info.get("hints_text", ""),
        trajectory=trajectory,
    )

    response = call_llm(prompt, model=model)
    if response is None:
        return None

    # Validate response has required fields
    for field in LLM_JUDGE_COMBINED_FEATURE_NAMES:
        if field not in response:
            print(f"    Missing field: {field}")
            return None

    return response


def main():
    parser = argparse.ArgumentParser(description="Compute LLM judge combined features")
    parser.add_argument("--dry_run", action="store_true",
                        help="Show plan without computing")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of tasks to process")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                        help="Skip tasks that already have features (default: True)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="LLM model to use (default: gpt-4o-mini)")
    parser.add_argument("--agent", type=str, default=None,
                        help="Specific agent to use (default: auto-select best M1 agent)")
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

    # Select agent
    if args.agent:
        selected_agent = args.agent
        # Check coverage
        agent_dir = UNIFIED_TRAJS_DIR / selected_agent
        if not agent_dir.exists():
            print(f"Error: Agent directory not found: {agent_dir}")
            return
        coverage = sum(1 for t in all_tasks if (agent_dir / f"{t}.json").exists())
        print(f"\nUsing specified agent: {selected_agent} ({coverage}/{len(all_tasks)} coverage)")
    else:
        # Auto-select best M1 agent
        selected_agent, coverage, failures = select_best_agent(split.m1_agents, all_tasks)
        if selected_agent is None:
            print("Error: No agent found with trajectory coverage")
            return
        print(f"\nAuto-selected agent: {selected_agent}")
        print(f"  Coverage: {coverage}/{len(all_tasks)} tasks")
        print(f"  Failures: {failures} (more informative for difficulty prediction)")

    # Filter to tasks with trajectories
    tasks_with_trajs = []
    for task_id in all_tasks:
        traj_path = UNIFIED_TRAJS_DIR / selected_agent / f"{task_id}.json"
        if traj_path.exists():
            tasks_with_trajs.append(task_id)

    print(f"\nTasks with trajectories: {len(tasks_with_trajs)}")

    # Filter out existing
    if args.skip_existing:
        filtered_tasks = []
        for task_id in tasks_with_trajs:
            output_file = OUTPUT_DIR / f"{task_id}.json"
            if not output_file.exists():
                filtered_tasks.append(task_id)
        print(f"After skipping existing: {len(filtered_tasks)} tasks to process")
        tasks_with_trajs = filtered_tasks

    if args.limit > 0:
        tasks_with_trajs = tasks_with_trajs[:args.limit]
        print(f"Limited to {len(tasks_with_trajs)} tasks")

    if args.dry_run:
        print(f"\n=== DRY RUN - Would process {len(tasks_with_trajs)} tasks ===")
        print(f"Agent: {selected_agent}")
        print(f"Model: {args.model}")
        print(f"Rate limit delay: {args.rate_limit_delay}s")
        print(f"Output directory: {OUTPUT_DIR}")
        print(f"\nSample tasks (first 10):")
        for task_id in tasks_with_trajs[:10]:
            print(f"  {task_id}")
        if len(tasks_with_trajs) > 10:
            print(f"  ... and {len(tasks_with_trajs) - 10} more tasks")
        return

    # Load task metadata
    tasks = load_swebench_tasks()
    if not tasks:
        print("Failed to load task metadata")
        return

    # Process tasks
    print(f"\nComputing combined features using {args.model}...")
    start_time = datetime.now()

    processed = 0
    skipped = 0
    failed = 0

    for task_id in tasks_with_trajs:
        output_file = OUTPUT_DIR / f"{task_id}.json"

        if args.skip_existing and output_file.exists():
            skipped += 1
            continue

        if task_id not in tasks:
            print(f"  Skipping {task_id}: no task metadata")
            skipped += 1
            continue

        # Load trajectory
        trajectory = load_trajectory(selected_agent, task_id)
        if trajectory is None:
            print(f"  Skipping {task_id}: no trajectory")
            skipped += 1
            continue

        print(f"[{processed + failed + 1}/{len(tasks_with_trajs)}] {task_id}...")

        # Compute features
        result = compute_features_for_task(
            task_id, tasks[task_id], trajectory, model=args.model
        )

        if result is None:
            failed += 1
            print(f"    FAILED")
            continue

        # Add metadata and save
        result["task_id"] = task_id
        result["agent"] = selected_agent
        result["resolved"] = trajectory.get("resolved", False)
        result["model"] = args.model
        result["computed_at"] = datetime.now().isoformat()

        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)

        processed += 1
        print(f"    -> fix_complexity={result.get('fix_complexity')}, "
              f"location_alignment={result.get('location_vs_fix_alignment')}")

        # Rate limiting
        time.sleep(args.rate_limit_delay)

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\nDone! Processed {processed}, skipped {skipped}, failed {failed}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

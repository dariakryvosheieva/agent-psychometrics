"""Compute LLM judge v7 features for trajectories.

This script extracts semantic features from agent trajectories using an LLM
as judge. Features are designed to capture signal orthogonal to what the
embedding prior captures.

V7 features (all 1-5 scale, higher = harder):
1. error_misdirection - Did errors point to wrong location?
2. bug_reproduction_quality - How well did agent reproduce the bug?
3. location_vs_fix_gap - Found right location but couldn't fix?
4. solution_discoverability - Can solution be found from reading code?

Usage:
    # Dry run to see execution plan
    python -m experiment_b.compute_llm_judge_v7_features --dry_run

    # Run on small subset for validation
    python -m experiment_b.compute_llm_judge_v7_features --limit 10

    # Full run with specific model
    python -m experiment_b.compute_llm_judge_v7_features --model gpt-4o-mini

    # Analyze feature distributions
    python -m experiment_b.compute_llm_judge_v7_features --analyze
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Add parent to path for imports
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_b.config import ExperimentConfig
from experiment_b.data_splits import create_experiment_split
from experiment_b.trajectory_features_v2 import parse_gold_patch_files
from experiment_b.llm_judge.features_v7 import (
    LLM_JUDGE_V7_FEATURE_NAMES,
    LLMJudgeV7Features,
    build_v7_prompt,
)


# Directories
UNIFIED_TRAJS_DIR = ROOT / "trajectory_data" / "unified_trajs"
OUTPUT_DIR = ROOT / "chris_output" / "experiment_b" / "llm_judge_v7_features"


def load_swebench_tasks() -> Dict[str, dict]:
    """Load SWE-bench Verified task metadata.

    Returns:
        Dict mapping task_id -> task dict with problem_statement, patch, repo
    """
    try:
        from datasets import load_dataset
        print("Loading SWE-bench Verified tasks...")
        ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
        tasks = {
            item["instance_id"]: {
                "problem_statement": item["problem_statement"],
                "patch": item["patch"],
                "repo": item["repo"],
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


def call_llm(prompt: str, model: str = "claude-opus-4-5-20251101") -> Optional[dict]:
    """Call LLM API to get judge response.

    Args:
        prompt: The prompt to send
        model: Model to use. Supports:
            - Anthropic models: claude-opus-4-5-20251101, claude-sonnet-4-20250514, etc.
            - OpenAI models: gpt-4o, gpt-4o-mini, gpt-5.2, etc.

    Returns:
        Parsed JSON response or None on failure
    """
    try:
        # Determine which API to use based on model name
        if model.startswith("claude") or model.startswith("anthropic"):
            content = _call_anthropic(prompt, model)
        else:
            content = _call_openai(prompt, model)

        if not content:
            return None

        # Parse JSON from response
        # Handle markdown code blocks
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


def compute_features_for_agent_task(
    task_id: str,
    agent: str,
    task_info: dict,
    model: str = "gpt-4o-mini",
) -> Optional[dict]:
    """Compute v7 features for a single agent-task pair.

    Args:
        task_id: Task instance ID
        agent: Agent name
        task_info: Task metadata (problem_statement, patch, repo)
        model: LLM model to use

    Returns:
        Feature dict or None on failure
    """
    trajectory = load_trajectory(agent, task_id)
    if trajectory is None:
        return None

    gold_patch_files = list(parse_gold_patch_files(task_info["patch"]))

    prompt = build_v7_prompt(
        instance_id=task_id,
        repo=task_info["repo"],
        problem_statement=task_info["problem_statement"],
        patch=task_info["patch"],
        trajectory=trajectory,
        gold_patch_files=gold_patch_files,
    )

    response = call_llm(prompt, model=model)
    if response is None:
        return None

    # Validate response has required fields
    required_fields = ["error_misdirection", "bug_reproduction_quality",
                       "location_vs_fix_gap", "solution_discoverability"]
    for field in required_fields:
        if field not in response:
            print(f"    Missing field: {field}")
            return None
        # Ensure values are in valid range
        val = response[field]
        if not isinstance(val, int) or val < 1 or val > 5:
            print(f"    Invalid value for {field}: {val}")
            return None

    return response


def analyze_features(output_dir: Path):
    """Analyze computed feature distributions.

    Args:
        output_dir: Directory containing computed features
    """
    print("\n=== LLM Judge V7 Feature Analysis ===\n")

    all_features = {name: [] for name in LLM_JUDGE_V7_FEATURE_NAMES}
    total_count = 0
    agent_counts = {}

    # Load all feature files (organized by agent)
    for agent_dir in output_dir.iterdir():
        if not agent_dir.is_dir():
            continue

        agent = agent_dir.name
        agent_counts[agent] = 0

        for task_file in agent_dir.glob("*.json"):
            try:
                with open(task_file) as f:
                    data = json.load(f)

                total_count += 1
                agent_counts[agent] += 1

                for name in LLM_JUDGE_V7_FEATURE_NAMES:
                    if name in data:
                        all_features[name].append(data[name])

            except (json.JSONDecodeError, IOError):
                continue

    print(f"Analyzed {total_count} agent-task pairs")
    print(f"Agents: {len(agent_counts)}")
    print()

    print("Feature distributions (raw 1-5 scale):")
    print("-" * 60)
    for name in LLM_JUDGE_V7_FEATURE_NAMES:
        values = all_features[name]
        if values:
            arr = np.array(values)
            print(f"{name}:")
            print(f"  mean={arr.mean():.2f}, std={arr.std():.2f}")
            print(f"  min={arr.min()}, max={arr.max()}")
            print(f"  distribution: {dict(zip(*np.unique(arr, return_counts=True)))}")
            print()


def main():
    parser = argparse.ArgumentParser(description="Compute LLM judge v7 features")
    parser.add_argument("--dry_run", action="store_true",
                        help="Show plan without computing")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of agent-task pairs to process")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze existing feature distributions")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                        help="Skip pairs that already have features (default: True)")
    parser.add_argument("--model", type=str, default="claude-opus-4-5-20251101",
                        help="LLM model to use (default: claude-opus-4-5-20251101)")
    parser.add_argument("--rate_limit_delay", type=float, default=0.5,
                        help="Delay between API calls in seconds (default: 0.5)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.analyze:
        analyze_features(OUTPUT_DIR)
        return

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

    # Collect agent-task pairs to process
    pairs_to_compute = []

    # M1 agents x D_train tasks (for posterior training)
    for task_id in split.d_train_tasks:
        for agent in split.m1_agents:
            pairs_to_compute.append((task_id, agent, "train"))

    # M2 agents x D_valid tasks (for posterior evaluation)
    for task_id in split.d_valid_tasks:
        for agent in split.m2_agents:
            pairs_to_compute.append((task_id, agent, "valid"))

    print(f"\nTotal agent-task pairs: {len(pairs_to_compute)}")
    print(f"  M1 agents: {len(split.m1_agents)}")
    print(f"  M2 agents: {len(split.m2_agents)}")
    print(f"  D_train tasks: {len(split.d_train_tasks)}")
    print(f"  D_valid tasks: {len(split.d_valid_tasks)}")

    # Filter out existing
    if args.skip_existing:
        filtered_pairs = []
        for task_id, agent, split_type in pairs_to_compute:
            output_file = OUTPUT_DIR / agent / f"{task_id}.json"
            if not output_file.exists():
                filtered_pairs.append((task_id, agent, split_type))
        print(f"\nAfter skipping existing: {len(filtered_pairs)} pairs to process")
        pairs_to_compute = filtered_pairs

    if args.limit > 0:
        pairs_to_compute = pairs_to_compute[:args.limit]
        print(f"Limited to {len(pairs_to_compute)} pairs")

    if args.dry_run:
        print(f"\n=== DRY RUN - Would process {len(pairs_to_compute)} pairs ===")
        print(f"Model: {args.model}")
        print(f"Rate limit delay: {args.rate_limit_delay}s")
        for i, (task_id, agent, split_type) in enumerate(pairs_to_compute[:10]):
            print(f"  {agent}/{task_id} ({split_type})")
        if len(pairs_to_compute) > 10:
            print(f"  ... and {len(pairs_to_compute) - 10} more pairs")
        return

    # Load task metadata
    tasks = load_swebench_tasks()
    if not tasks:
        print("Failed to load task metadata")
        return

    # Process pairs
    print(f"\nComputing v7 features using {args.model}...")
    start_time = datetime.now()

    processed = 0
    skipped = 0
    failed = 0

    for task_id, agent, split_type in pairs_to_compute:
        # Create agent directory
        agent_dir = OUTPUT_DIR / agent
        agent_dir.mkdir(parents=True, exist_ok=True)
        output_file = agent_dir / f"{task_id}.json"

        if args.skip_existing and output_file.exists():
            skipped += 1
            continue

        if task_id not in tasks:
            print(f"  Skipping {task_id}: no task metadata")
            skipped += 1
            continue

        # Compute features
        result = compute_features_for_agent_task(
            task_id, agent, tasks[task_id], model=args.model
        )

        if result is None:
            failed += 1
            continue

        # Add metadata and save
        result["task_id"] = task_id
        result["agent"] = agent
        result["split"] = split_type
        result["model"] = args.model

        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)

        processed += 1

        if processed % 10 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = processed / elapsed if elapsed > 0 else 0
            print(f"  Processed {processed}/{len(pairs_to_compute)} "
                  f"({rate:.1f}/s, {failed} failed)")

        # Rate limiting
        time.sleep(args.rate_limit_delay)

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\nDone! Processed {processed}, skipped {skipped}, failed {failed}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

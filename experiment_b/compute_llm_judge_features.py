"""Compute LLM judge features for trajectories (direct API, no Lunette sandbox).

This script uses direct LLM API calls (Anthropic/OpenAI) to extract semantic
features from agent trajectories. It only computes features for the specific
agent-task combinations needed for Experiment B:
- M1 agents × D_train tasks (for posterior training)
- M2 agents × D_valid tasks (for posterior evaluation)

Usage:
    # Dry run to see execution plan
    python -m experiment_b.compute_llm_judge_features --dry_run

    # Run on small subset for validation
    python -m experiment_b.compute_llm_judge_features --limit 50

    # Run on specific agents
    python -m experiment_b.compute_llm_judge_features --agents 20240620_sweagent_claude3.5sonnet

    # Use OpenAI instead of Anthropic
    python -m experiment_b.compute_llm_judge_features --provider openai --model gpt-4o

    # Full run (all M1×D_train + M2×D_valid)
    python -m experiment_b.compute_llm_judge_features
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent to path for imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_b.config import ExperimentConfig
from experiment_b.data_splits import create_experiment_split
from experiment_b.llm_judge_features import TRAJECTORY_GRADING_PROMPT, LLM_JUDGE_FEATURE_NAMES


# Try to import API clients
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# Directories
UNIFIED_TRAJS_DIR = ROOT / "trajectory_data" / "unified_trajs"
OUTPUT_DIR = ROOT / "chris_output" / "experiment_b" / "llm_judge_features"


def load_swebench_metadata() -> Dict[str, dict]:
    """Load full SWE-bench Verified metadata from HuggingFace."""
    from datasets import load_dataset

    print("Loading SWE-bench Verified dataset...")
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

    metadata = {}
    for item in ds:
        instance_id = item["instance_id"]
        metadata[instance_id] = {
            "repo": item["repo"],
            "problem_statement": item["problem_statement"],
            "patch": item["patch"],
            "test_patch": item["test_patch"],
            "version": item["version"],
            "hints_text": item["hints_text"],
            "base_commit": item["base_commit"],
            "FAIL_TO_PASS": item["FAIL_TO_PASS"],
            "PASS_TO_PASS": item["PASS_TO_PASS"],
        }

    print(f"Loaded metadata for {len(metadata)} tasks")
    return metadata


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


def format_trajectory_for_prompt(traj: dict, max_chars: int = 50000) -> str:
    """Format trajectory messages into a readable string, truncating if needed."""
    messages = traj.get("messages", [])
    if not messages:
        return "(No messages in trajectory)"

    lines = []
    total_chars = 0

    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        # Truncate individual messages if very long
        if len(content) > 5000:
            content = content[:5000] + "\n... [truncated]"

        line = f"[{role.upper()}]\n{content}\n"

        # Check if we'd exceed max chars
        if total_chars + len(line) > max_chars:
            lines.append(f"\n... [trajectory truncated at message {i}/{len(messages)}]")
            break

        lines.append(line)
        total_chars += len(line)

    return "\n".join(lines)


def parse_llm_response(text: str) -> Optional[dict]:
    """Parse the LLM response to extract features."""
    if not text:
        return None

    # Try to extract JSON from response
    # First try: look for ```json block
    json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if json_match:
        text = json_match.group(1)
    else:
        # Second try: look for any JSON object
        json_match = re.search(r'\{[^{}]*"llm_judge_difficulty_score"[^{}]*\}', text, re.DOTALL)
        if json_match:
            text = json_match.group(0)

    try:
        data = json.loads(text)

        # Validate required fields
        if "llm_judge_difficulty_score" not in data:
            return None

        return data
    except json.JSONDecodeError:
        return None


def call_anthropic(prompt: str, model: str = "claude-opus-4-5-20251101") -> str:
    """Call Anthropic API and return response text."""
    if not HAS_ANTHROPIC:
        raise ImportError("anthropic package not installed")

    client = anthropic.Anthropic()

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text.strip()


def call_openai(prompt: str, model: str = "gpt-4o") -> str:
    """Call OpenAI API and return response text."""
    if not HAS_OPENAI:
        raise ImportError("openai package not installed")

    client = openai.OpenAI()

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
    )

    return response.choices[0].message.content.strip()


def grade_trajectory_with_llm(
    task_id: str,
    trajectory: dict,
    swebench_meta: dict,
    provider: str = "anthropic",
    model: Optional[str] = None,
) -> Optional[dict]:
    """Call LLM to extract features from trajectory + metadata.

    Args:
        task_id: Task instance ID
        trajectory: Loaded trajectory dict
        swebench_meta: SWE-bench metadata for the task
        provider: "anthropic" or "openai"
        model: Specific model to use (default: claude-sonnet-4-20250514 or gpt-4o)

    Returns:
        Parsed feature dict or None if failed
    """
    # Build prompt with full metadata
    hints_section = ""
    if swebench_meta.get("hints_text"):
        hints_section = f"**Hints:**\n{swebench_meta['hints_text']}"

    prompt = TRAJECTORY_GRADING_PROMPT.format(
        instance_id=task_id,
        repo=swebench_meta["repo"],
        version=swebench_meta.get("version", "unknown"),
        problem_statement=swebench_meta["problem_statement"][:8000],
        patch=swebench_meta["patch"][:6000],
        fail_to_pass=swebench_meta.get("FAIL_TO_PASS", "[]"),
        hints_section=hints_section,
        trajectory_text=format_trajectory_for_prompt(trajectory),
        resolved_status="RESOLVED (agent succeeded)" if trajectory.get("resolved") else "UNRESOLVED (agent failed)",
    )

    try:
        if provider == "anthropic":
            model = model or "claude-opus-4-5-20251101"
            response_text = call_anthropic(prompt, model)
        elif provider == "openai":
            model = model or "gpt-4o"
            response_text = call_openai(prompt, model)
        else:
            raise ValueError(f"Unknown provider: {provider}")

        features = parse_llm_response(response_text)
        if features:
            features["_model"] = model
            features["_provider"] = provider
        return features

    except Exception as e:
        print(f"    Error calling LLM: {e}")
        return None


def get_experiment_combinations(
    split,
    trajectories_dir: Path,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Get the agent-task combinations needed for Experiment B.

    Returns:
        (train_combinations, valid_combinations) where each is a list of (agent, task_id) tuples
    """
    train_combinations = []
    valid_combinations = []

    # M1 agents × D_train tasks (for training)
    for agent in split.m1_agents:
        agent_dir = trajectories_dir / agent
        if not agent_dir.exists():
            continue
        for task_id in split.d_train_tasks:
            if (agent_dir / f"{task_id}.json").exists():
                train_combinations.append((agent, task_id))

    # M2 agents × D_valid tasks (for validation)
    for agent in split.m2_agents:
        agent_dir = trajectories_dir / agent
        if not agent_dir.exists():
            continue
        for task_id in split.d_valid_tasks:
            if (agent_dir / f"{task_id}.json").exists():
                valid_combinations.append((agent, task_id))

    return train_combinations, valid_combinations


def main():
    parser = argparse.ArgumentParser(description="Compute LLM judge features")
    parser.add_argument("--dry_run", action="store_true", help="Show plan without running")
    parser.add_argument("--limit", type=int, help="Limit total number of API calls")
    parser.add_argument("--agents", nargs="+", help="Specific agents to process (overrides auto-selection)")
    parser.add_argument("--tasks", nargs="+", help="Specific tasks to process (overrides auto-selection)")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                       help="Skip tasks with existing features")
    parser.add_argument("--provider", type=str, default="anthropic",
                       choices=["anthropic", "openai"],
                       help="LLM provider to use")
    parser.add_argument("--model", type=str, default=None,
                       help="Specific model to use")
    parser.add_argument("--delay", type=float, default=0.5,
                       help="Delay between API calls (seconds)")
    parser.add_argument("--weak_threshold", type=float, default=0.2,
                       help="Pass rate threshold for weak agents")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load config and create splits
    config = ExperimentConfig(weak_threshold=args.weak_threshold)
    responses_path = ROOT / config.responses_path
    trajectories_dir = ROOT / config.trajectories_dir

    print("Creating experiment split...")
    split = create_experiment_split(
        responses_path=responses_path,
        trajectories_dir=trajectories_dir,
        weak_threshold=config.weak_threshold,
        strong_min_improvement=config.strong_min_improvement,
        m1_fraction=config.m1_fraction,
        m2_fraction=config.m2_fraction,
    )

    print(f"\nExperiment split:")
    print(f"  M1 agents: {len(split.m1_agents)}")
    print(f"  M2 agents: {len(split.m2_agents)}")
    print(f"  D_train tasks: {len(split.d_train_tasks)}")
    print(f"  D_valid tasks: {len(split.d_valid_tasks)}")

    # Get combinations to process
    if args.agents and args.tasks:
        # Manual override: specific agents × specific tasks
        combinations = [(a, t) for a in args.agents for t in args.tasks]
        print(f"\n  Manual selection: {len(combinations)} combinations")
    elif args.agents:
        # Specific agents × all relevant tasks
        combinations = []
        for agent in args.agents:
            agent_dir = UNIFIED_TRAJS_DIR / agent
            if not agent_dir.exists():
                print(f"  Warning: Agent {agent} not found in {UNIFIED_TRAJS_DIR}")
                continue
            # Use D_train if agent is in M1, D_valid if in M2
            if agent in split.m1_agents:
                tasks = split.d_train_tasks
            elif agent in split.m2_agents:
                tasks = split.d_valid_tasks
            else:
                # Default to both
                tasks = list(set(split.d_train_tasks + split.d_valid_tasks))
            for task_id in tasks:
                if (agent_dir / f"{task_id}.json").exists():
                    combinations.append((agent, task_id))
        print(f"\n  Selected agents: {len(combinations)} combinations")
    else:
        # Auto: M1×D_train + M2×D_valid
        train_combs, valid_combs = get_experiment_combinations(split, UNIFIED_TRAJS_DIR)
        combinations = train_combs + valid_combs
        print(f"\n  Auto-selected: {len(train_combs)} train + {len(valid_combs)} valid = {len(combinations)} combinations")

    # Apply limit
    if args.limit and len(combinations) > args.limit:
        combinations = combinations[:args.limit]
        print(f"  Limited to {args.limit} combinations")

    # Filter out existing
    if args.skip_existing:
        original_count = len(combinations)
        combinations = [
            (a, t) for a, t in combinations
            if not (OUTPUT_DIR / a / f"{t}.json").exists()
        ]
        skipped = original_count - len(combinations)
        if skipped > 0:
            print(f"  Skipping {skipped} existing, {len(combinations)} remaining")

    print(f"\n  Total API calls to make: {len(combinations)}")

    if args.dry_run:
        print("\n=== DRY RUN ===")
        print(f"\nProvider: {args.provider}")
        print(f"Model: {args.model or 'default'}")

        # Show sample of combinations
        print(f"\nSample combinations (first 10):")
        for agent, task_id in combinations[:10]:
            print(f"  {agent} × {task_id}")
        if len(combinations) > 10:
            print(f"  ... and {len(combinations) - 10} more")

        # Estimate cost
        calls = len(combinations)
        cost_per_call = 0.04  # Rough estimate for Claude Sonnet
        print(f"\nEstimated cost: ~${calls * cost_per_call:.2f} ({calls} calls × ${cost_per_call}/call)")
        return

    # Load SWE-bench metadata
    swebench_meta = load_swebench_metadata()

    # Process combinations
    stats = {
        "total": len(combinations),
        "success": 0,
        "failed": 0,
        "no_trajectory": 0,
        "no_metadata": 0,
    }

    for i, (agent, task_id) in enumerate(combinations):
        output_file = OUTPUT_DIR / agent / f"{task_id}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        print(f"[{i+1}/{len(combinations)}] {agent} × {task_id}...")

        # Load trajectory
        trajectory = load_trajectory(agent, task_id)
        if not trajectory:
            print(f"    No trajectory found")
            stats["no_trajectory"] += 1
            continue

        # Check metadata
        if task_id not in swebench_meta:
            print(f"    No SWE-bench metadata found")
            stats["no_metadata"] += 1
            continue

        # Grade with LLM
        features = grade_trajectory_with_llm(
            task_id=task_id,
            trajectory=trajectory,
            swebench_meta=swebench_meta[task_id],
            provider=args.provider,
            model=args.model,
        )

        if features:
            # Save features
            with open(output_file, "w") as f:
                json.dump(features, f, indent=2)
            stats["success"] += 1
            print(f"    -> difficulty_score: {features.get('llm_judge_difficulty_score', 'N/A')}")
        else:
            stats["failed"] += 1
            print(f"    Failed to extract features")

        # Rate limiting
        if args.delay and i < len(combinations) - 1:
            time.sleep(args.delay)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total processed: {stats['total']}")
    print(f"Success: {stats['success']}")
    print(f"Failed: {stats['failed']}")
    print(f"No trajectory: {stats['no_trajectory']}")
    print(f"No metadata: {stats['no_metadata']}")

    # Save stats
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stats_file = OUTPUT_DIR / f"compute_stats_{timestamp}.json"
    with open(stats_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "provider": args.provider,
            "model": args.model,
            "stats": stats,
            "combinations_processed": len(combinations),
        }, f, indent=2)
    print(f"\nStats saved to: {stats_file}")


if __name__ == "__main__":
    main()

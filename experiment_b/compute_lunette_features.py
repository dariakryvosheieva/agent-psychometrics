"""Compute Lunette features for trajectories.

This script uses Lunette's GradingPlan API to extract semantic features
from agent trajectories. Features are saved to disk for use by the
posterior model.

Usage:
    python -m experiment_b.compute_lunette_features --dry_run
    python -m experiment_b.compute_lunette_features --limit 10
    python -m experiment_b.compute_lunette_features --agents agent1 agent2
"""

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from lunette import LunetteClient
from lunette.analysis import GradingPlan

# Add parent to path for imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_b.config import ExperimentConfig
from experiment_b.data_splits import create_experiment_split
from experiment_b.lunette_features import TRAJECTORY_GRADING_PROMPT, LUNETTE_FEATURE_NAMES


# Directories
UNIFIED_TRAJS_DIR = ROOT / "trajectory_data" / "unified_trajs"
OUTPUT_DIR = ROOT / "chris_output" / "experiment_b" / "lunette_features"


def load_lunette_tracking(agent: str) -> Optional[dict]:
    """Load Lunette tracking file for an agent."""
    tracking_file = UNIFIED_TRAJS_DIR / agent / "_lunette_uploads.json"
    if not tracking_file.exists():
        return None
    with open(tracking_file) as f:
        return json.load(f)


def find_run_info_for_task(tracking: dict, task_id: str) -> Optional[dict]:
    """Find the Lunette run_id and trajectory_id for a specific task."""
    for traj in tracking.get("trajectories", []):
        if traj.get("task_id") == task_id:
            return {
                "run_id": traj.get("run_id"),
                "trajectory_id": traj.get("trajectory_id"),
            }
    return None


def parse_features_json(text: str) -> Optional[dict]:
    """Parse features JSON from LLM response."""
    if not text:
        return None

    # Try to find JSON in the response
    json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if not json_match:
        return None

    try:
        data = json.loads(json_match.group())
        # Validate required keys
        if all(k in data for k in LUNETTE_FEATURE_NAMES):
            return data
        # Partial match is still useful
        if any(k in data for k in LUNETTE_FEATURE_NAMES):
            return data
        return None
    except json.JSONDecodeError:
        return None


def parse_features_from_explanation(explanation: str) -> Optional[dict]:
    """Parse features from Lunette's explanation text.

    Lunette returns {name, score, explanation} format. This function
    extracts feature scores from the explanation text which contains
    patterns like "backtracking_exploration: 3/4" or "Backtracking (2/4)".
    """
    if not explanation:
        return None

    features = {}

    # Parse competency scores - various formats:
    # "backtracking_exploration: 3/4"
    # "Backtracking (2/4)"
    # "1. backtracking_exploration: 3/4"
    competency_patterns = [
        (r'backtracking[_\s]*(?:exploration)?[:\s(]+(\d)[/\s]*4', 'backtracking_exploration'),
        (r'task[_\s]*decomposition[:\s(]+(\d)[/\s]*4', 'task_decomposition'),
        (r'observation[_\s]*reading[:\s(]+(\d)[/\s]*4', 'observation_reading'),
        (r'self[_\s]*verification[:\s(]+(\d)[/\s]*4', 'self_verification'),
    ]

    for pattern, feature_name in competency_patterns:
        match = re.search(pattern, explanation, re.IGNORECASE)
        if match:
            features[feature_name] = int(match.group(1))

    # Parse failure mode indicators - look for positive mentions
    # Exclude negations like "no strategy_defect" or "all 0"
    failure_patterns = {
        'localization_failure': r'(?<!no\s)(?<!No\s)localization[_\s]*failure[s]?(?!\s*\(all\s*0)',
        'strategy_defect': r'(?:Failures\s*detected:.*|Failures:.*)?strategy[_\s]*defect(?:\s*\([^)]+\))?',
        'implementation_defect': r'(?:Failures\s*detected:.*|Failures:.*)?implementation[_\s]*defect(?:\s*\([^)]+\))?',
        'incomplete_repair': r'(?:Failures\s*detected:.*|Failures:.*)?incomplete[_\s]*repair(?:\s*\([^)]+\))?',
        'verification_failure': r'(?:Failures\s*detected:.*|Failures:.*)?verification[_\s]*failure(?:\s*\([^)]+\))?',
    }

    # Check for "all 0" or "no failures" patterns
    no_failures = bool(re.search(r'failure\s*modes?\s*\(all\s*0\)|no\s+(?:failures|defects)', explanation, re.IGNORECASE))

    for feature_name, pattern in failure_patterns.items():
        if no_failures:
            features[feature_name] = 0
        else:
            # Look for positive mentions in "Failures detected:" section
            failures_section = re.search(r'Failures\s*(?:detected)?:([^.]+)', explanation, re.IGNORECASE)
            if failures_section:
                section_text = failures_section.group(1)
                if feature_name.replace('_', '') in section_text.lower().replace('_', '').replace(' ', ''):
                    features[feature_name] = 1
                else:
                    features[feature_name] = 0
            else:
                # Fallback: check whole text but avoid negations
                if re.search(pattern, explanation, re.IGNORECASE):
                    features[feature_name] = 1
                else:
                    features[feature_name] = 0

    # Parse trajectory signals - similar logic
    no_signals = bool(re.search(r'trajectory\s*signals?\s*\(all\s*0\)|signals?\s*\(all\s*0\)', explanation, re.IGNORECASE))

    signal_patterns = {
        'agent_looping': r'agent[_\s]*looping|looping\s*\(',
        'agent_gave_up_early': r'gave[_\s]*up|stopped\s*early|abandoned',
        'agent_wrong_focus': r'wrong[_\s]*focus|fixated\s*on\s*irrelevant',
        'context_overflow': r'context[_\s]*overflow|lost\s*track|forgot',
    }

    for feature_name, pattern in signal_patterns.items():
        if no_signals:
            features[feature_name] = 0
        else:
            # Look for positive mentions in "Signals:" section
            signals_section = re.search(r'Signals?:([^.]+)', explanation, re.IGNORECASE)
            if signals_section:
                section_text = signals_section.group(1)
                if feature_name.replace('agent_', '').replace('_', '') in section_text.lower().replace('_', '').replace(' ', ''):
                    features[feature_name] = 1
                else:
                    features[feature_name] = 0
            else:
                if re.search(pattern, explanation, re.IGNORECASE):
                    features[feature_name] = 1
                else:
                    features[feature_name] = 0

    # Return if we found at least some features
    if len(features) >= 4:
        return features
    return None


async def grade_trajectory(
    client: LunetteClient,
    run_id: str,
    task_id: str,
) -> Optional[dict]:
    """Grade a single trajectory using Lunette.

    Args:
        client: LunetteClient instance
        run_id: Lunette run ID
        task_id: SWE-bench task ID

    Returns:
        Feature dict with lunette_difficulty_score and parsed features
    """
    try:
        plan = GradingPlan(name="difficulty-prediction", prompt=TRAJECTORY_GRADING_PROMPT)

        results = await client.investigate(
            run_id=run_id,
            plan=plan,
            limit=1,
        )

        if not results.results:
            print(f"    No results returned for {task_id}")
            return None

        result_data = results.results[0].data

        # Lunette returns {name, score, explanation}
        if isinstance(result_data, dict):
            # Get the difficulty score (0-1) from Lunette's score field
            lunette_score = result_data.get("score", 0.5)
            explanation = result_data.get("explanation", "")

            # Start with the main score
            features = {"lunette_difficulty_score": float(lunette_score)}

            # Parse additional features from explanation
            explanation_features = parse_features_from_explanation(explanation)
            if explanation_features:
                features.update(explanation_features)

            return features
        else:
            # Unexpected format
            return {"lunette_difficulty_score": 0.5, "_raw": str(result_data), "_unexpected_format": True}

    except Exception as e:
        print(f"    Error grading {task_id}: {e}")
        return None


def get_available_agents() -> List[str]:
    """Get list of agents with Lunette uploads."""
    agents = []
    for agent_dir in UNIFIED_TRAJS_DIR.iterdir():
        if agent_dir.is_dir():
            tracking_file = agent_dir / "_lunette_uploads.json"
            if tracking_file.exists():
                agents.append(agent_dir.name)
    return sorted(agents)


async def compute_features_for_agent(
    client: LunetteClient,
    agent: str,
    task_ids: List[str],
    output_dir: Path,
    skip_existing: bool = True,
) -> dict:
    """Compute features for all tasks for one agent.

    Returns:
        Stats dict with success/failure counts
    """
    tracking = load_lunette_tracking(agent)
    if not tracking:
        print(f"  No Lunette tracking for {agent}")
        return {"agent": agent, "skipped": True, "reason": "no_tracking"}

    agent_output_dir = output_dir / agent
    agent_output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "agent": agent,
        "total": len(task_ids),
        "success": 0,
        "failed": 0,
        "skipped": 0,
        "no_run_id": 0,
    }

    for i, task_id in enumerate(task_ids):
        output_file = agent_output_dir / f"{task_id}.json"

        # Skip if exists
        if skip_existing and output_file.exists():
            stats["skipped"] += 1
            continue

        # Find run info
        run_info = find_run_info_for_task(tracking, task_id)
        if not run_info or not run_info.get("run_id"):
            stats["no_run_id"] += 1
            continue

        print(f"  [{i+1}/{len(task_ids)}] {task_id}...")

        features = await grade_trajectory(client, run_info["run_id"], task_id)

        if features and not features.get("_parse_failed"):
            # Save features
            with open(output_file, "w") as f:
                json.dump(features, f, indent=2)
            stats["success"] += 1
        else:
            stats["failed"] += 1
            if features and features.get("_raw"):
                # Save raw response for debugging
                debug_file = agent_output_dir / f"{task_id}.debug.txt"
                with open(debug_file, "w") as f:
                    f.write(features["_raw"])

    return stats


async def main():
    parser = argparse.ArgumentParser(description="Compute Lunette features")
    parser.add_argument("--dry_run", action="store_true", help="Show plan without running")
    parser.add_argument("--limit", type=int, help="Limit number of tasks per agent")
    parser.add_argument("--agents", nargs="+", help="Specific agents to process")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                       help="Skip tasks with existing features")
    parser.add_argument("--weak_threshold", type=float, default=0.2,
                       help="Pass rate threshold for weak agents")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load config and create splits to get agent lists
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

    # Determine which agents to process
    if args.agents:
        agents = args.agents
    else:
        # Use M1 and M2 agents (the ones needed for training/validation)
        agents = split.m1_agents + split.m2_agents

    # Filter to agents with Lunette uploads
    available_agents = set(get_available_agents())
    agents = [a for a in agents if a in available_agents]

    # Get all task IDs (both train and valid)
    all_tasks = list(set(split.d_train_tasks + split.d_valid_tasks))
    if args.limit:
        all_tasks = all_tasks[:args.limit]

    print(f"\nConfiguration:")
    print(f"  Agents to process: {len(agents)}")
    print(f"  Tasks to process: {len(all_tasks)}")
    print(f"  Total combinations: {len(agents) * len(all_tasks)}")
    print(f"  Output directory: {OUTPUT_DIR}")

    if args.dry_run:
        print("\n=== DRY RUN ===")
        print(f"\nAgents ({len(agents)}):")
        for a in agents[:10]:
            print(f"  - {a}")
        if len(agents) > 10:
            print(f"  ... and {len(agents) - 10} more")
        print(f"\nTasks ({len(all_tasks)}):")
        for t in all_tasks[:5]:
            print(f"  - {t}")
        if len(all_tasks) > 5:
            print(f"  ... and {len(all_tasks) - 5} more")
        return

    # Process
    all_stats = []

    async with LunetteClient() as client:
        for agent in agents:
            print(f"\nProcessing {agent}...")
            stats = await compute_features_for_agent(
                client,
                agent,
                all_tasks,
                OUTPUT_DIR,
                skip_existing=args.skip_existing,
            )
            all_stats.append(stats)
            print(f"  Done: {stats}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_success = sum(s.get("success", 0) for s in all_stats)
    total_failed = sum(s.get("failed", 0) for s in all_stats)
    total_skipped = sum(s.get("skipped", 0) for s in all_stats)

    print(f"Total successful: {total_success}")
    print(f"Total failed: {total_failed}")
    print(f"Total skipped (existing): {total_skipped}")

    # Save stats
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stats_file = OUTPUT_DIR / f"compute_stats_{timestamp}.json"
    with open(stats_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "agents": agents,
            "tasks": all_tasks,
            "agent_stats": all_stats,
        }, f, indent=2)
    print(f"\nStats saved to: {stats_file}")


if __name__ == "__main__":
    asyncio.run(main())

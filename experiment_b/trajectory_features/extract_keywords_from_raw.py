"""Extract keyword features directly from raw trajectory text.

This script extracts keyword features by searching for patterns in the raw
trajectory JSON (agent messages) without requiring an LLM call. This is
useful for agents that only need keyword features (not judge features).

Usage:
    # Extract keywords for one agent from custom task list
    python -m experiment_b.trajectory_features.extract_keywords_from_raw \
        --agent amazon \
        --task-list chris_output/trajectory_features/all_fail_nonfrontier_tasks.json

    # Extract for all keyword-only agents
    python -m experiment_b.trajectory_features.extract_keywords_from_raw --all-keyword-agents
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd

from experiment_b.trajectory_features.build_frontier_model import FULL_COVERAGE_AGENTS
from experiment_b.trajectory_features.keyword_features import (
    KEYWORD_FEATURES,
    extract_keyword_features,
)

# Agents that need LLM judge features (solution_stability, progress_linearity, etc.)
AGENTS_NEEDING_JUDGE_FEATURES = {"masai", "autocoderover"}

# Agents that only need keyword features (can extract from raw trajectory)
KEYWORD_ONLY_AGENTS = {
    "amazon", "openhands_sonnet", "lingma72b", "navie", "honeycomb", "openhands"
}

# Agents with no significant features (can skip entirely)
AGENTS_TO_SKIP = {
    "sweagent_sonnet", "agentless", "agentless_sonnet", "epam_sonnet", "marscode"
}


def load_trajectory(agent_full_name: str, task_id: str, trajs_dir: Path) -> dict:
    """Load a single trajectory file."""
    traj_path = trajs_dir / agent_full_name / f"{task_id}.json"
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory not found: {traj_path}")
    with open(traj_path) as f:
        return json.load(f)


def extract_trajectory_text(trajectory: dict) -> str:
    """Extract all text content from a trajectory.

    Concatenates all assistant messages to get the full trajectory text.
    """
    messages = trajectory.get("messages", [])
    text_parts = []
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("content"):
            text_parts.append(msg["content"])
    return "\n\n".join(text_parts)


def extract_keywords_for_agent(
    agent_short_name: str,
    task_ids: List[str],
    trajs_dir: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """Extract keyword features for one agent from raw trajectories.

    Args:
        agent_short_name: Short agent name (e.g., "amazon")
        task_ids: List of task IDs to process
        trajs_dir: Directory containing trajectory files
        output_dir: Directory to save output CSV

    Returns:
        DataFrame with keyword features
    """
    agent_full_name = FULL_COVERAGE_AGENTS[agent_short_name]
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    missing = []

    for task_id in task_ids:
        try:
            trajectory = load_trajectory(agent_full_name, task_id, trajs_dir)
            trajectory_text = extract_trajectory_text(trajectory)

            # Extract keyword features from raw trajectory text
            features = extract_keyword_features(trajectory_text, agent_short_name)
            features["task_id"] = task_id
            features["_agent"] = agent_short_name
            results.append(features)

        except FileNotFoundError:
            missing.append(task_id)

    if missing:
        print(f"  Warning: {len(missing)} trajectories not found for {agent_short_name}")

    df = pd.DataFrame(results)

    # Save to CSV
    csv_path = output_dir / f"raw_keywords_{agent_short_name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved {len(df)} tasks to {csv_path}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Extract keyword features from raw trajectories"
    )
    parser.add_argument(
        "--agent",
        type=str,
        default=None,
        help="Agent short name to extract for (e.g., 'amazon')",
    )
    parser.add_argument(
        "--all-keyword-agents",
        action="store_true",
        help="Extract for all keyword-only agents",
    )
    parser.add_argument(
        "--task-list",
        type=Path,
        default=Path("chris_output/trajectory_features/all_fail_nonfrontier_tasks.json"),
        help="JSON file with task IDs to process",
    )
    parser.add_argument(
        "--trajs-dir",
        type=Path,
        default=Path("trajectory_data/unified_trajs"),
        help="Directory containing trajectory files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("chris_output/trajectory_features/raw_keywords"),
        help="Output directory for CSV files",
    )
    args = parser.parse_args()

    # Load task list
    print(f"Loading task list from {args.task_list}...")
    with open(args.task_list) as f:
        task_ids = json.load(f)
    print(f"  Loaded {len(task_ids)} tasks")

    # Determine which agents to process
    if args.all_keyword_agents:
        agents_to_process = list(KEYWORD_ONLY_AGENTS)
    elif args.agent:
        if args.agent not in FULL_COVERAGE_AGENTS:
            raise ValueError(f"Unknown agent: {args.agent}")
        agents_to_process = [args.agent]
    else:
        raise ValueError("Must specify --agent or --all-keyword-agents")

    print(f"\nExtracting keywords for {len(agents_to_process)} agents:")
    for agent in sorted(agents_to_process):
        print(f"  - {agent}")

    # Extract for each agent
    all_dfs = []
    for agent in sorted(agents_to_process):
        print(f"\nProcessing {agent}...")
        df = extract_keywords_for_agent(
            agent, task_ids, args.trajs_dir, args.output_dir
        )
        all_dfs.append(df)

    # Combine all results
    if len(all_dfs) > 1:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_path = args.output_dir / "raw_keywords_all.csv"
        combined.to_csv(combined_path, index=False)
        print(f"\nCombined results saved to {combined_path}")

    print("\nDone!")
    print(f"  Agents needing LLM judge features: {sorted(AGENTS_NEEDING_JUDGE_FEATURES)}")
    print(f"  Agents to skip (no significant features): {sorted(AGENTS_TO_SKIP)}")


if __name__ == "__main__":
    main()

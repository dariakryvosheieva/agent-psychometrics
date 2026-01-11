"""
Upload only the annotated trajectories from IssueSolvingEmpirical to Lunette.

This creates a new run with just the ~200 annotated trajectories for efficient grading.
"""

import asyncio
import json
import re
from pathlib import Path

from lunette import LunetteClient
from lunette.models.run import Run
from lunette.models.trajectory import Trajectory, ScalarScore
from lunette.models.messages import UserMessage, AssistantMessage

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ISSUE_SOLVING_DIR = PROJECT_ROOT / "IssueSolvingEmpirical" / "dataset"
UNIFIED_TRAJS_DIR = PROJECT_ROOT / "trajectory_data" / "unified_trajs"
OUTPUT_DIR = PROJECT_ROOT / "chris_output" / "experiment_b1"

# Agent mapping
AGENT_MAPPING = {
    "openhands": "20241029_OpenHands-CodeAct-2.1-sonnet-20241022",
    "agentless": "20241028_agentless-1.5_gpt4o",
}


def load_annotations(agent_name: str) -> dict[str, list[str]]:
    """Load annotations for an agent."""
    annotation_file = ISSUE_SOLVING_DIR / f"annotations_{agent_name}.json"

    try:
        with open(annotation_file) as f:
            raw = json.load(f)
    except json.JSONDecodeError:
        with open(annotation_file) as f:
            content = f.read()
        task_pattern = r'"([a-z_]+__[a-z_]+-\d+)":\s*\{([^}]+)\}'
        category_pattern = r'"category":\s*"([^"]+)"'
        raw = {}
        for match in re.finditer(task_pattern, content, re.DOTALL):
            task_id = match.group(1)
            task_content = match.group(2)
            categories = re.findall(category_pattern, task_content)
            if categories:
                raw[task_id] = {"extracted": [{"category": c} for c in categories]}

    result = {}
    for task_id, action_annotations in raw.items():
        categories = set()
        for action_data in action_annotations.values():
            if isinstance(action_data, list):
                for ann in action_data:
                    if "category" in ann:
                        categories.add(ann["category"])
        result[task_id] = list(categories)

    return result


def load_unified_trajectory(agent_dir: str, task_id: str) -> dict | None:
    """Load a unified trajectory JSON file."""
    traj_file = UNIFIED_TRAJS_DIR / agent_dir / f"{task_id}.json"
    if not traj_file.exists():
        return None
    with open(traj_file) as f:
        return json.load(f)


def convert_to_lunette_trajectory(task_id: str, unified_traj: dict) -> Trajectory:
    """Convert unified trajectory format to Lunette Trajectory."""
    messages = []

    # Convert messages from unified format
    for i, msg in enumerate(unified_traj.get("messages", [])):
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "user":
            messages.append(UserMessage(position=i, content=content))
        elif role == "assistant":
            messages.append(AssistantMessage(position=i, content=content))

    # Determine resolution status
    resolved = unified_traj.get("resolved", False)
    scores = {"resolved": ScalarScore(value=1.0 if resolved else 0.0)}

    return Trajectory(
        sample=task_id,
        messages=messages,
        scores=scores,
        metadata=unified_traj.get("metadata", {}),
    )


async def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all annotated task IDs
    print("Collecting annotated tasks...")
    tasks_to_upload = []

    for paper_agent, our_agent in AGENT_MAPPING.items():
        annotations = load_annotations(paper_agent)
        print(f"  {paper_agent}: {len(annotations)} annotated tasks")

        for task_id, ground_truth in annotations.items():
            unified_traj = load_unified_trajectory(our_agent, task_id)
            if unified_traj:
                tasks_to_upload.append({
                    "task_id": task_id,
                    "paper_agent": paper_agent,
                    "our_agent": our_agent,
                    "ground_truth": ground_truth,
                    "unified_traj": unified_traj,
                })
            else:
                print(f"    Warning: No unified trajectory for {task_id}")

    print(f"\nTotal trajectories to upload: {len(tasks_to_upload)}")

    # Convert to Lunette trajectories
    print("\nConverting to Lunette format...")
    lunette_trajectories = []
    for task in tasks_to_upload:
        traj = convert_to_lunette_trajectory(task["task_id"], task["unified_traj"])
        lunette_trajectories.append(traj)

    # Upload to Lunette - 1 trajectory per run to allow reliable grading
    # The Lunette investigate API doesn't support offset pagination, so we need
    # separate runs to grade each trajectory individually
    BATCH_SIZE = 1
    print(f"\nUploading {len(lunette_trajectories)} trajectories in batches of {BATCH_SIZE}...")

    run_ids = []
    trajectory_to_run = {}  # task_id -> run_id mapping

    async with LunetteClient() as client:
        for batch_idx in range(0, len(lunette_trajectories), BATCH_SIZE):
            batch = lunette_trajectories[batch_idx:batch_idx + BATCH_SIZE]
            batch_tasks = tasks_to_upload[batch_idx:batch_idx + BATCH_SIZE]

            run = Run(
                task="swebench-failure-analysis",
                model="annotated-trajectories",
                trajectories=batch,
            )

            run_meta = await client.save_run(run)
            run_id = run_meta["run_id"]
            run_ids.append(run_id)

            # Track which tasks are in this run
            for task in batch_tasks:
                trajectory_to_run[task["task_id"]] = run_id

            print(f"  Batch {batch_idx // BATCH_SIZE + 1}: {len(batch)} trajectories -> run {run_id[:16]}...")

    # Save tracking info
    tracking = {
        "run_ids": run_ids,
        "trajectory_count": len(lunette_trajectories),
        "task_to_run": trajectory_to_run,
        "tasks": [
            {
                "task_id": t["task_id"],
                "paper_agent": t["paper_agent"],
                "ground_truth": t["ground_truth"],
                "run_id": trajectory_to_run[t["task_id"]],
            }
            for t in tasks_to_upload
        ],
    }

    tracking_file = OUTPUT_DIR / "annotated_trajectories_upload.json"
    with open(tracking_file, "w") as f:
        json.dump(tracking, f, indent=2)

    print(f"\nUpload complete!")
    print(f"  Total runs: {len(run_ids)}")
    print(f"  Total trajectories: {len(lunette_trajectories)}")
    print(f"  Tracking file: {tracking_file}")


if __name__ == "__main__":
    asyncio.run(main())

"""Test batch grading with 2 trajectories in one run."""

import asyncio
import json
from pathlib import Path

from lunette import LunetteClient
from lunette.analysis import GradingPlan
from lunette.models.run import Run
from lunette.models.trajectory import Trajectory, ScalarScore
from lunette.models.messages import UserMessage, AssistantMessage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UNIFIED_TRAJS_DIR = PROJECT_ROOT / "trajectory_data" / "unified_trajs"

SIMPLE_PROMPT = """Analyze this FAILED SWE-bench trajectory. List the primary failure mode."""


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
    for i, msg in enumerate(unified_traj.get("messages", [])[:20]):  # Limit messages
        role = msg.get("role", "user")
        content = msg.get("content", "")[:2000]  # Truncate long content

        if role == "user":
            messages.append(UserMessage(position=i, content=content))
        elif role == "assistant":
            messages.append(AssistantMessage(position=i, content=content))

    resolved = unified_traj.get("resolved", False)
    scores = {"resolved": ScalarScore(value=1.0 if resolved else 0.0)}

    return Trajectory(
        sample=task_id,
        messages=messages,
        scores=scores,
        metadata={},
    )


async def main():
    agent_dir = "20241029_OpenHands-CodeAct-2.1-sonnet-20241022"

    # Pick 2 tasks
    task_ids = ["astropy__astropy-13236", "astropy__astropy-13398"]

    # Convert to trajectories
    trajectories = []
    for task_id in task_ids:
        traj = load_unified_trajectory(agent_dir, task_id)
        if traj:
            lunette_traj = convert_to_lunette_trajectory(task_id, traj)
            trajectories.append(lunette_traj)
            print(f"Loaded {task_id}: {len(lunette_traj.messages)} messages")

    if len(trajectories) < 2:
        print("Could not load both trajectories")
        return

    # Upload as a single run
    print(f"\nUploading {len(trajectories)} trajectories as 1 run...")
    async with LunetteClient() as client:
        run = Run(
            task="batch-grading-test",
            model="test",
            trajectories=trajectories,
        )
        run_meta = await client.save_run(run)
        run_id = run_meta["run_id"]
        print(f"Created run: {run_id}")

        # Grade without limit - should grade all trajectories
        print("\nRunning investigate() on the run...")
        results = await client.investigate(
            run_id=run_id,
            plan=GradingPlan(
                name="test-grading",
                prompt=SIMPLE_PROMPT,
            ),
        )

        print(f"\nGot {len(results.results)} results:")
        for r in results.results:
            print(f"  - original_trajectory_id: {r.original_trajectory_id}")
            print(f"    explanation: {r.data.get('explanation', 'N/A')[:200]}...")
            print()


if __name__ == "__main__":
    asyncio.run(main())

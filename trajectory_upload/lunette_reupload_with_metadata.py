"""
Delete incorrectly formatted Lunette uploads and re-upload with proper SWE-bench metadata.

This script:
1. Deletes all existing runs for specified agents
2. Re-uploads trajectories with proper SWE-bench metadata (repo, patch, test_patch, etc.)

Usage:
    # Test with one agent
    python trajectory_upload/lunette_reupload_with_metadata.py --agents 20240620_sweagent_claude3.5sonnet --limit 5

    # Dry run to see what would happen
    python trajectory_upload/lunette_reupload_with_metadata.py --agents 20240620_sweagent_claude3.5sonnet --dry_run

    # Process all agents
    python trajectory_upload/lunette_reupload_with_metadata.py

    # Skip deletion (if already deleted)
    python trajectory_upload/lunette_reupload_with_metadata.py --skip_delete
"""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from datasets import load_dataset

from lunette import LunetteClient
from lunette.models.run import Run
from lunette.models.trajectory import Trajectory, ScalarScore
from lunette.models.messages import SystemMessage, UserMessage, AssistantMessage


def load_lunette_api_key() -> str:
    """Load Lunette API key from config file."""
    config_path = Path.home() / ".lunette" / "config.json"
    with open(config_path) as f:
        return json.load(f)["api_key"]


def load_swebench_metadata() -> dict[str, dict]:
    """Load SWE-bench Verified dataset and return metadata dict by instance_id."""
    print("Loading SWE-bench Verified dataset...")
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

    metadata = {}
    for item in ds:
        instance_id = item["instance_id"]
        metadata[instance_id] = {
            "repo": item["repo"],
            "patch": item["patch"],
            "test_patch": item["test_patch"],
            "version": item["version"],
            "created_at": item["created_at"],
            "hints_text": item["hints_text"],
            "base_commit": item["base_commit"],
            "FAIL_TO_PASS": item["FAIL_TO_PASS"],
            "PASS_TO_PASS": item["PASS_TO_PASS"],
        }

    print(f"Loaded metadata for {len(metadata)} tasks")
    return metadata


def load_results_for_agent(agent_name: str) -> dict[str, bool]:
    """Load results.json to get resolved/unresolved status for each task."""
    experiments_dir = Path(__file__).resolve().parents[1] / "experiments"
    results_path = experiments_dir / "evaluation" / "verified" / agent_name / "results" / "results.json"

    if not results_path.exists():
        print(f"  Warning: results.json not found at {results_path}")
        return {}

    with open(results_path) as f:
        results = json.load(f)

    resolved_set = set(results.get("resolved", []))
    return {task: task in resolved_set for task in resolved_set}


def load_converted_trajectory(file_path: Path) -> dict:
    """Load a converted trajectory JSON file."""
    with open(file_path) as f:
        return json.load(f)


def convert_to_lunette_format_with_metadata(
    unified_traj: dict,
    resolved: bool,
    model_name: str,
    swebench_metadata: dict,
) -> Trajectory:
    """Convert unified trajectory format to Lunette Trajectory with proper SWE-bench metadata."""
    messages = []

    for i, msg in enumerate(unified_traj.get("messages", [])):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        position = msg.get("position", i)

        if role == "system":
            messages.append(SystemMessage(position=position, content=content))
        elif role == "assistant":
            messages.append(AssistantMessage(position=position, content=content))
        else:
            messages.append(UserMessage(position=position, content=content))

    task_id = unified_traj.get("task_id", "unknown")

    # Get solution from metadata if available
    info = unified_traj.get("metadata", {}).get("info", {})
    solution = info.get("submission", "")

    scores = {"resolved": ScalarScore(value=1.0 if resolved else 0.0)}

    # Build metadata with SWE-bench fields
    task_meta = swebench_metadata.get(task_id, {})
    metadata = {
        # SWE-bench task metadata
        "repo": task_meta.get("repo", ""),
        "patch": task_meta.get("patch", ""),
        "test_patch": task_meta.get("test_patch", ""),
        "version": task_meta.get("version", ""),
        "created_at": task_meta.get("created_at", ""),
        "hints_text": task_meta.get("hints_text", ""),
        "base_commit": task_meta.get("base_commit", ""),
        "FAIL_TO_PASS": task_meta.get("FAIL_TO_PASS", []),
        "PASS_TO_PASS": task_meta.get("PASS_TO_PASS", []),
    }

    return Trajectory(
        sample=task_id,
        messages=messages,
        scores=scores,
        solution=solution,
        metadata=metadata,
    )


def delete_runs(run_ids: list[str], api_key: str, dry_run: bool = False) -> dict:
    """Delete runs from Lunette."""
    results = {"deleted": 0, "failed": 0, "errors": []}

    with httpx.Client(base_url="https://lunette.dev/api", headers={"X-API-Key": api_key}, timeout=30.0) as client:
        for i, run_id in enumerate(run_ids):
            if dry_run:
                print(f"  [{i+1}/{len(run_ids)}] Would delete run: {run_id[:8]}...")
                results["deleted"] += 1
                continue

            try:
                r = client.delete(f"/runs/{run_id}")
                if r.status_code == 200:
                    resp = r.json()
                    traj_count = resp.get("deleted_trajectories", 0)
                    print(f"  [{i+1}/{len(run_ids)}] Deleted run {run_id[:8]}... ({traj_count} trajectories)")
                    results["deleted"] += 1
                else:
                    print(f"  [{i+1}/{len(run_ids)}] Failed to delete {run_id[:8]}: {r.status_code}")
                    results["failed"] += 1
                    results["errors"].append({"run_id": run_id, "status": r.status_code})
            except Exception as e:
                print(f"  [{i+1}/{len(run_ids)}] Error deleting {run_id[:8]}: {e}")
                results["failed"] += 1
                results["errors"].append({"run_id": run_id, "error": str(e)})

    return results


def load_existing_upload(agent_dir: Path) -> dict | None:
    """Load existing upload tracking file."""
    tracking_file = agent_dir / "_lunette_uploads.json"
    if tracking_file.exists():
        with open(tracking_file) as f:
            return json.load(f)
    return None


def save_upload_tracking(agent_dir: Path, upload_info: dict):
    """Save upload tracking info."""
    tracking_file = agent_dir / "_lunette_uploads.json"
    with open(tracking_file, "w") as f:
        json.dump(upload_info, f, indent=2)


async def upload_agent_with_metadata(
    client: LunetteClient,
    agent_dir: Path,
    agent_name: str,
    swebench_metadata: dict,
    dry_run: bool = False,
    batch_size: int = 100,
    limit: Optional[int] = None,
) -> dict:
    """Upload trajectories for an agent with proper SWE-bench metadata."""

    results = load_results_for_agent(agent_name)

    # Find all trajectory JSON files
    json_files = sorted(agent_dir.glob("*.json"))
    json_files = [f for f in json_files if not f.name.startswith("_")]

    if limit:
        json_files = json_files[:limit]

    if not json_files:
        print(f"  {agent_name}: No trajectory files found")
        return {"error": "No trajectories"}

    print(f"  {agent_name}: Converting {len(json_files)} trajectories with SWE-bench metadata...")

    # Convert all trajectories
    trajectories = []
    trajectory_info = []

    for file_path in json_files:
        task_id = file_path.stem
        resolved = results.get(task_id, False)

        try:
            unified = load_converted_trajectory(file_path)
            lunette_traj = convert_to_lunette_format_with_metadata(
                unified,
                resolved=resolved,
                model_name=agent_name,
                swebench_metadata=swebench_metadata,
            )
            trajectories.append(lunette_traj)
            trajectory_info.append({
                "task_id": task_id,
                "resolved": resolved,
                "message_count": len(unified.get("messages", [])),
            })
        except Exception as e:
            print(f"    Error converting {task_id}: {e}")

    if not trajectories:
        print(f"  {agent_name}: No valid trajectories after conversion")
        return {"error": "No valid trajectories"}

    if dry_run:
        num_batches = (len(trajectories) + batch_size - 1) // batch_size
        print(f"  {agent_name}: Would upload {len(trajectories)} trajectories in {num_batches} batch(es)")
        return {"dry_run": True, "trajectory_count": len(trajectories)}

    # Upload in batches with retry logic for 413 errors
    all_run_ids = []
    all_traj_ids = []
    batch_mapping = []
    failed_trajectories = []

    async def upload_batch(batch_trajs: list, label: str, current_batch_size: int = None) -> bool:
        """Upload a batch, returns True on success."""
        if current_batch_size is None:
            current_batch_size = len(batch_trajs)

        run = Run(
            task="swebench-verified",
            model=agent_name,
            trajectories=batch_trajs,
        )

        try:
            run_meta = await client.save_run(run)
            run_id = run_meta["run_id"]
            traj_ids = run_meta.get("trajectory_ids", [])

            all_run_ids.append(run_id)
            all_traj_ids.extend(traj_ids)

            for traj_id in traj_ids:
                batch_mapping.append({
                    "run_id": run_id,
                    "trajectory_id": traj_id,
                })

            print(f"    {label}: {len(traj_ids)} trajectories -> run:{run_id[:8]}...")
            return True

        except Exception as e:
            error_str = str(e)
            if "413" in error_str and len(batch_trajs) > 10:
                # Try smaller batches
                smaller_size = max(10, len(batch_trajs) // 2)
                print(f"    {label}: 413 error, retrying with batch size {smaller_size}...")
                sub_batches = [batch_trajs[i:i+smaller_size] for i in range(0, len(batch_trajs), smaller_size)]
                success = True
                for j, sub_batch in enumerate(sub_batches):
                    sub_label = f"{label}.{j+1}"
                    if not await upload_batch(sub_batch, sub_label, smaller_size):
                        success = False
                return success
            else:
                print(f"    {label}: FAILED - {e}")
                failed_trajectories.extend(batch_trajs)
                return False

    num_batches = (len(trajectories) + batch_size - 1) // batch_size
    print(f"  {agent_name}: Uploading {len(trajectories)} trajectories in {num_batches} batch(es)...")

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(trajectories))
        batch_trajs = trajectories[start:end]

        await upload_batch(batch_trajs, f"Batch {batch_idx + 1}/{num_batches}")

    if not all_traj_ids:
        return {"error": "All batches failed", "failed_count": len(failed_trajectories)}

    if failed_trajectories:
        print(f"  {agent_name}: PARTIAL - {len(all_traj_ids)} uploaded, {len(failed_trajectories)} failed")
    else:
        print(f"  {agent_name}: SUCCESS - {len(all_traj_ids)} trajectories in {len(all_run_ids)} run(s)")

    # Save tracking info
    upload_info = {
        "agent": agent_name,
        "uploaded_at": datetime.now().isoformat(),
        "run_id": all_run_ids[0] if len(all_run_ids) == 1 else None,
        "run_ids": all_run_ids,
        "num_batches": num_batches,
        "batch_size": batch_size,
        "trajectory_count": len(all_traj_ids),
        "trajectory_ids": all_traj_ids,
        "has_swebench_metadata": True,
        "trajectories": [
            {
                **trajectory_info[i],
                "trajectory_id": batch_mapping[i]["trajectory_id"] if i < len(batch_mapping) else None,
                "run_id": batch_mapping[i]["run_id"] if i < len(batch_mapping) else None,
            }
            for i in range(len(trajectory_info))
        ],
    }
    save_upload_tracking(agent_dir, upload_info)

    return {"success": True, "run_ids": all_run_ids, "trajectory_count": len(all_traj_ids)}


async def main():
    parser = argparse.ArgumentParser(description="Delete and re-upload trajectories with proper SWE-bench metadata")
    parser.add_argument("--agents", nargs="+", help="Specific agents to process (default: all)")
    parser.add_argument("--dry_run", action="store_true", help="Show what would happen without making changes")
    parser.add_argument("--skip_delete", action="store_true", help="Skip deletion step (if already deleted)")
    parser.add_argument("--input_dir", type=str, default="trajectory_data/unified_trajs",
                        help="Base directory containing agent folders")
    parser.add_argument("--batch_size", type=int, default=50, help="Max trajectories per batch (default: 50 for metadata-rich uploads)")
    parser.add_argument("--limit", type=int, help="Limit number of trajectories per agent (for testing)")

    args = parser.parse_args()

    input_base = Path(args.input_dir)
    if not input_base.exists():
        print(f"Error: Input directory not found: {input_base}")
        return

    # Find agent directories
    if args.agents:
        agent_dirs = [input_base / a for a in args.agents]
        agent_dirs = [d for d in agent_dirs if d.exists()]
    else:
        agent_dirs = sorted([
            d for d in input_base.iterdir()
            if d.is_dir() and not d.name.startswith("_")
        ])

    print(f"=== Lunette Re-upload with SWE-bench Metadata ===")
    print(f"Agents to process: {len(agent_dirs)}")
    if args.dry_run:
        print("DRY RUN - no changes will be made\n")
    if args.limit:
        print(f"LIMIT: Processing only {args.limit} trajectories per agent\n")

    # Load SWE-bench metadata once
    swebench_metadata = load_swebench_metadata()

    # Load API key for deletions
    api_key = load_lunette_api_key()

    summary = {
        "started": datetime.now().isoformat(),
        "agents_processed": 0,
        "runs_deleted": 0,
        "trajectories_uploaded": 0,
        "errors": [],
    }

    async with LunetteClient() as client:
        for i, agent_dir in enumerate(agent_dirs):
            agent_name = agent_dir.name
            print(f"\n[{i+1}/{len(agent_dirs)}] {agent_name}")

            # Step 1: Delete existing runs
            if not args.skip_delete:
                existing = load_existing_upload(agent_dir)
                if existing:
                    run_ids = existing.get("run_ids", [])
                    if existing.get("run_id") and not run_ids:
                        run_ids = [existing["run_id"]]

                    if run_ids:
                        print(f"  Deleting {len(run_ids)} existing runs...")
                        delete_result = delete_runs(run_ids, api_key, dry_run=args.dry_run)
                        summary["runs_deleted"] += delete_result["deleted"]

            # Step 2: Re-upload with metadata
            try:
                result = await upload_agent_with_metadata(
                    client=client,
                    agent_dir=agent_dir,
                    agent_name=agent_name,
                    swebench_metadata=swebench_metadata,
                    dry_run=args.dry_run,
                    batch_size=args.batch_size,
                    limit=args.limit,
                )

                if result.get("success") or result.get("dry_run"):
                    summary["agents_processed"] += 1
                    summary["trajectories_uploaded"] += result.get("trajectory_count", 0)
                else:
                    summary["errors"].append({"agent": agent_name, "error": result.get("error")})

            except Exception as e:
                print(f"  ERROR: {e}")
                summary["errors"].append({"agent": agent_name, "error": str(e)})

    summary["completed"] = datetime.now().isoformat()

    print(f"\n=== COMPLETE ===")
    print(f"Agents processed: {summary['agents_processed']}")
    print(f"Runs deleted: {summary['runs_deleted']}")
    print(f"Trajectories uploaded: {summary['trajectories_uploaded']}")
    if summary["errors"]:
        print(f"Errors: {len(summary['errors'])}")

    # Save summary
    summary_path = input_base / "_reupload_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())

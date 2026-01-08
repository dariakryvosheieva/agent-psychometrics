"""
Upload converted trajectories to Lunette.

Takes the unified JSON trajectories from trajectory_converter.py and uploads
them to Lunette for grading/analysis.

Usage:
    # Upload all converted trajectories for one agent
    python llm_judge/lunette_upload.py --agent 20240620_sweagent_claude3.5sonnet

    # Upload from a specific directory
    python llm_judge/lunette_upload.py --input_dir chris_output/unified_trajs/20240620_sweagent_claude3.5sonnet

    # Upload with a limit
    python llm_judge/lunette_upload.py --agent 20240620_sweagent_claude3.5sonnet --limit 50

    # Dry run (show what would be uploaded)
    python llm_judge/lunette_upload.py --agent 20240620_sweagent_claude3.5sonnet --dry_run
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from lunette import LunetteClient
from lunette.models.run import Run
from lunette.models.trajectory import Trajectory, ScalarScore
from lunette.models.messages import SystemMessage, UserMessage, AssistantMessage


def load_converted_trajectory(file_path: Path) -> dict:
    """Load a converted trajectory JSON file."""
    with open(file_path) as f:
        return json.load(f)


def load_results_for_agent(agent_name: str) -> dict[str, bool]:
    """Load results.json to get resolved/unresolved status for each task."""
    experiments_dir = Path(__file__).resolve().parents[1] / 'experiments'
    results_path = experiments_dir / 'evaluation' / 'verified' / agent_name / 'results' / 'results.json'

    if not results_path.exists():
        print(f"Warning: results.json not found at {results_path}")
        return {}

    with open(results_path) as f:
        results = json.load(f)

    resolved_set = set(results.get('resolved', []))
    # Return dict mapping task_id -> resolved status
    return {task: task in resolved_set for task in resolved_set}


def convert_to_lunette_format(
    unified_traj: dict,
    resolved: bool = False,
    model_name: str = "unknown",
) -> Trajectory:
    """Convert unified trajectory format to Lunette Trajectory object."""
    messages = []

    for i, msg in enumerate(unified_traj.get('messages', [])):
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        position = msg.get('position', i)  # Use index as position if not specified

        if role == 'system':
            messages.append(SystemMessage(position=position, content=content))
        elif role == 'assistant':
            messages.append(AssistantMessage(position=position, content=content))
        else:
            messages.append(UserMessage(position=position, content=content))

    task_id = unified_traj.get('task_id', 'unknown')

    # Get solution from metadata if available
    info = unified_traj.get('metadata', {}).get('info', {})
    solution = info.get('submission', '')

    scores = {'resolved': ScalarScore(value=1.0 if resolved else 0.0)}

    return Trajectory(
        sample=task_id,
        messages=messages,
        scores=scores,
        solution=solution,
        metadata={
            'environment': unified_traj.get('environment', ''),
            'original_format': unified_traj.get('metadata', {}).get('source_format', ''),
            'total_messages': len(messages),
        }
    )


def load_existing_uploads(input_dir: Path) -> dict[str, dict]:
    """Load existing upload tracking file to support resume."""
    tracking_file = input_dir / '_lunette_uploads.json'
    if tracking_file.exists():
        with open(tracking_file) as f:
            data = json.load(f)
        return {item['task_id']: item for item in data.get('uploads', [])}
    return {}


def save_upload_tracking(input_dir: Path, uploads: list[dict], agent_name: str):
    """Save upload tracking info for later access."""
    from datetime import datetime
    tracking_file = input_dir / '_lunette_uploads.json'

    data = {
        'agent': agent_name,
        'last_updated': datetime.now().isoformat(),
        'total_uploads': len(uploads),
        'uploads': uploads,
    }

    with open(tracking_file, 'w') as f:
        json.dump(data, f, indent=2)


async def upload_trajectories(
    client: LunetteClient,
    input_dir: Path,
    agent_name: str,
    results: dict[str, bool],
    limit: Optional[int] = None,
    dry_run: bool = False,
    skip_existing: bool = True,
) -> dict:
    """Upload trajectories from input_dir to Lunette."""
    summary = {
        'total': 0,
        'uploaded': 0,
        'skipped': 0,
        'errors': 0,
        'uploads': [],
    }

    # Load existing uploads for resume capability
    existing_uploads = load_existing_uploads(input_dir) if skip_existing else {}
    if existing_uploads:
        print(f"Found {len(existing_uploads)} existing uploads (will skip)")

    # Find all JSON files
    json_files = sorted(input_dir.glob('*.json'))
    json_files = [f for f in json_files if not f.name.startswith('_')]  # Skip summary files

    if limit:
        json_files = json_files[:limit]

    print(f"Found {len(json_files)} trajectory files to process")

    for i, file_path in enumerate(json_files):
        summary['total'] += 1
        task_id = file_path.stem

        # Skip if already uploaded
        if task_id in existing_uploads:
            summary['skipped'] += 1
            summary['uploads'].append(existing_uploads[task_id])
            continue

        # Get resolved status
        resolved = results.get(task_id, False)

        if dry_run:
            print(f"[{i+1}/{len(json_files)}] Would upload: {task_id} (resolved={resolved})")
            summary['uploaded'] += 1
            continue

        try:
            # Load and convert
            unified = load_converted_trajectory(file_path)
            lunette_traj = convert_to_lunette_format(
                unified,
                resolved=resolved,
                model_name=agent_name,
            )

            # Create run
            run = Run(
                task="swebench-verified",
                model=agent_name,
                trajectories=[lunette_traj],
            )

            # Upload
            print(f"[{i+1}/{len(json_files)}] Uploading {task_id}...", end=" ", flush=True)
            run_meta = await client.save_run(run)
            run_id = run_meta['run_id']
            traj_id = run_meta.get('trajectory_ids', [None])[0]
            print(f"-> run:{run_id[:8]}... traj:{traj_id[:8] if traj_id else 'N/A'}...")

            upload_record = {
                'task_id': task_id,
                'run_id': run_id,
                'trajectory_id': traj_id,
                'resolved': resolved,
                'message_count': len(unified.get('messages', [])),
            }
            summary['uploaded'] += 1
            summary['uploads'].append(upload_record)

            # Save progress periodically (every 10 uploads)
            if not dry_run and summary['uploaded'] % 10 == 0:
                save_upload_tracking(input_dir, summary['uploads'], agent_name)

        except Exception as e:
            print(f"[{i+1}/{len(json_files)}] Error uploading {task_id}: {e}")
            summary['errors'] += 1

    # Final save
    if not dry_run and summary['uploads']:
        save_upload_tracking(input_dir, summary['uploads'], agent_name)

    return summary


async def main():
    parser = argparse.ArgumentParser(
        description='Upload converted trajectories to Lunette',
    )

    parser.add_argument('--agent', type=str, help='Agent name (looks in chris_output/unified_trajs/<agent>/)')
    parser.add_argument('--input_dir', type=str, help='Directory containing converted trajectory JSONs')
    parser.add_argument('--limit', type=int, help='Max number of trajectories to upload')
    parser.add_argument('--dry_run', action='store_true', help='Show what would be uploaded without uploading')
    parser.add_argument('--output', type=str, help='Output path for upload summary')

    args = parser.parse_args()

    if not args.agent and not args.input_dir:
        parser.print_help()
        return

    # Determine input directory
    if args.input_dir:
        input_dir = Path(args.input_dir)
        agent_name = input_dir.name
    else:
        input_dir = Path(f'chris_output/unified_trajs/{args.agent}')
        agent_name = args.agent

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        print(f"\nHave you run the converter? Try:")
        print(f"  python llm_judge/trajectory_converter.py --agent {agent_name}")
        return

    # Load results
    results = load_results_for_agent(agent_name)
    print(f"Loaded results for {len(results)} resolved tasks")

    # Upload
    if args.dry_run:
        print("\n=== DRY RUN (no uploads will be made) ===\n")
        # Create a mock for dry run
        summary = {
            'total': 0,
            'uploaded': 0,
            'errors': 0,
            'run_ids': [],
        }
        json_files = sorted(input_dir.glob('*.json'))
        json_files = [f for f in json_files if not f.name.startswith('_')]
        if args.limit:
            json_files = json_files[:args.limit]

        for i, f in enumerate(json_files):
            task_id = f.stem
            resolved = results.get(task_id, False)
            print(f"[{i+1}/{len(json_files)}] Would upload: {task_id} (resolved={resolved})")
            summary['total'] += 1
            summary['uploaded'] += 1

        print(f"\nWould upload {summary['uploaded']} trajectories")
    else:
        async with LunetteClient() as client:
            summary = await upload_trajectories(
                client=client,
                input_dir=input_dir,
                agent_name=agent_name,
                results=results,
                limit=args.limit,
                dry_run=args.dry_run,
            )

        print(f"\n=== UPLOAD SUMMARY ===")
        print(f"Total processed: {summary['total']}")
        print(f"Uploaded: {summary['uploaded']}")
        print(f"Skipped (existing): {summary['skipped']}")
        print(f"Errors: {summary['errors']}")
        print(f"Tracking saved to: {input_dir / '_lunette_uploads.json'}")


if __name__ == '__main__':
    asyncio.run(main())

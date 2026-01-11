"""
DEPRECATED: This script uploads without SWE-bench metadata.
Use lunette_reupload_with_metadata.py instead.

Batch upload all converted trajectories to Lunette.

Uploads ALL trajectories for each agent in a SINGLE run (not one run per trajectory).

Usage:
    # RECOMMENDED: Use the metadata-aware uploader instead
    python trajectory_upload/lunette_reupload_with_metadata.py

    # Legacy (no metadata):
    python trajectory_upload/lunette_batch_upload.py --agents 20240620_sweagent_claude3.5sonnet
"""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from lunette import LunetteClient
from lunette.models.run import Run

from lunette_upload import (
    load_results_for_agent,
    load_converted_trajectory,
    convert_to_lunette_format,
)


def load_existing_upload(agent_dir: Path) -> dict | None:
    """Load existing upload tracking file."""
    tracking_file = agent_dir / '_lunette_uploads.json'
    if tracking_file.exists():
        with open(tracking_file) as f:
            return json.load(f)
    return None


def save_upload_tracking(agent_dir: Path, upload_info: dict):
    """Save upload tracking info."""
    tracking_file = agent_dir / '_lunette_uploads.json'
    with open(tracking_file, 'w') as f:
        json.dump(upload_info, f, indent=2)


async def upload_agent_batch(
    client: LunetteClient,
    agent_dir: Path,
    agent_name: str,
    dry_run: bool = False,
    batch_size: int = 100,
) -> dict:
    """Upload trajectories for an agent, batching if needed to avoid 413 errors."""

    # Check if already uploaded
    existing = load_existing_upload(agent_dir)
    if existing and (existing.get('run_id') or existing.get('run_ids')):
        run_info = existing.get('run_id', existing.get('run_ids', ['?'])[0])[:8] if existing.get('run_id') else f"{len(existing.get('run_ids', []))} runs"
        print(f"  {agent_name}: Already uploaded ({run_info})")
        return {'skipped': True, 'existing': existing}

    results = load_results_for_agent(agent_name)

    # Find all trajectory JSON files
    json_files = sorted(agent_dir.glob('*.json'))
    json_files = [f for f in json_files if not f.name.startswith('_')]

    if not json_files:
        print(f"  {agent_name}: No trajectory files found")
        return {'error': 'No trajectories'}

    print(f"  {agent_name}: Converting {len(json_files)} trajectories...")

    # Convert all trajectories
    trajectories = []
    trajectory_info = []

    for file_path in json_files:
        task_id = file_path.stem
        resolved = results.get(task_id, False)

        try:
            unified = load_converted_trajectory(file_path)
            lunette_traj = convert_to_lunette_format(
                unified,
                resolved=resolved,
                model_name=agent_name
            )
            trajectories.append(lunette_traj)
            trajectory_info.append({
                'task_id': task_id,
                'resolved': resolved,
                'message_count': len(unified.get('messages', [])),
            })
        except Exception as e:
            print(f"    Error converting {task_id}: {e}")

    if not trajectories:
        print(f"  {agent_name}: No valid trajectories after conversion")
        return {'error': 'No valid trajectories'}

    if dry_run:
        num_batches = (len(trajectories) + batch_size - 1) // batch_size
        print(f"  {agent_name}: Would upload {len(trajectories)} trajectories in {num_batches} batch(es)")
        return {'dry_run': True, 'trajectory_count': len(trajectories)}

    # Upload in batches to avoid 413 errors
    all_run_ids = []
    all_traj_ids = []
    batch_mapping = []  # Track which batch/run each trajectory belongs to

    num_batches = (len(trajectories) + batch_size - 1) // batch_size
    print(f"  {agent_name}: Uploading {len(trajectories)} trajectories in {num_batches} batch(es)...")

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(trajectories))
        batch_trajs = trajectories[start:end]

        run = Run(
            task="swebench-verified",
            model=agent_name,
            trajectories=batch_trajs,
        )

        try:
            run_meta = await client.save_run(run)
            run_id = run_meta['run_id']
            traj_ids = run_meta.get('trajectory_ids', [])

            all_run_ids.append(run_id)
            all_traj_ids.extend(traj_ids)

            # Track batch info for each trajectory
            for traj_id in traj_ids:
                batch_mapping.append({
                    'batch_index': batch_idx,
                    'batch_total': num_batches,
                    'run_id': run_id,
                    'trajectory_id': traj_id,
                })

            print(f"    Batch {batch_idx + 1}/{num_batches}: {len(traj_ids)} trajectories -> run:{run_id[:8]}...")

        except Exception as e:
            print(f"    Batch {batch_idx + 1}/{num_batches}: FAILED - {e}")
            # Continue with remaining batches

    if not all_traj_ids:
        return {'error': 'All batches failed'}

    print(f"  {agent_name}: SUCCESS - {len(all_traj_ids)} trajectories in {len(all_run_ids)} run(s)")

    # Save tracking info with batch details
    upload_info = {
        'agent': agent_name,
        'uploaded_at': datetime.now().isoformat(),
        'run_id': all_run_ids[0] if len(all_run_ids) == 1 else None,
        'run_ids': all_run_ids,
        'num_batches': num_batches,
        'batch_size': batch_size,
        'trajectory_count': len(all_traj_ids),
        'trajectory_ids': all_traj_ids,
        'trajectories': [
            {
                **trajectory_info[i],
                'trajectory_id': batch_mapping[i]['trajectory_id'] if i < len(batch_mapping) else None,
                'batch_index': batch_mapping[i]['batch_index'] if i < len(batch_mapping) else None,
                'run_id': batch_mapping[i]['run_id'] if i < len(batch_mapping) else None,
            }
            for i in range(len(trajectory_info))
        ],
    }
    save_upload_tracking(agent_dir, upload_info)

    return {'success': True, 'run_ids': all_run_ids, 'trajectory_count': len(all_traj_ids)}


async def main():
    parser = argparse.ArgumentParser(description='Batch upload trajectories to Lunette (one run per agent)')
    parser.add_argument('--agents', nargs='+', help='Specific agents to upload (default: all)')
    parser.add_argument('--dry_run', action='store_true', help='Show what would be uploaded')
    parser.add_argument('--input_dir', type=str, default='trajectory_data/unified_trajs',
                        help='Base directory containing agent folders')
    parser.add_argument('--output', type=str, help='Output path for batch summary')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Max trajectories per batch (default: 100, use smaller for large trajectories)')

    args = parser.parse_args()

    input_base = Path(args.input_dir)
    if not input_base.exists():
        print(f"Error: Input directory not found: {input_base}")
        return

    # Find all agent directories
    if args.agents:
        agent_dirs = [input_base / a for a in args.agents]
        agent_dirs = [d for d in agent_dirs if d.exists()]
    else:
        agent_dirs = sorted([
            d for d in input_base.iterdir()
            if d.is_dir() and not d.name.startswith('_')
        ])

    print(f"=== Batch Upload to Lunette (1 run per agent) ===")
    print(f"Found {len(agent_dirs)} agents to process")
    if args.dry_run:
        print("DRY RUN - no uploads will be made\n")

    batch_summary = {
        'started': datetime.now().isoformat(),
        'total_agents': len(agent_dirs),
        'successful': 0,
        'skipped': 0,
        'failed': 0,
        'agents': {},
    }

    async with LunetteClient() as client:
        for i, agent_dir in enumerate(agent_dirs):
            agent_name = agent_dir.name
            print(f"\n[{i+1}/{len(agent_dirs)}] {agent_name}")

            try:
                result = await upload_agent_batch(
                    client=client,
                    agent_dir=agent_dir,
                    agent_name=agent_name,
                    dry_run=args.dry_run,
                    batch_size=args.batch_size,
                )

                batch_summary['agents'][agent_name] = result

                if result.get('success'):
                    batch_summary['successful'] += 1
                elif result.get('skipped'):
                    batch_summary['skipped'] += 1
                elif result.get('dry_run'):
                    batch_summary['successful'] += 1
                else:
                    batch_summary['failed'] += 1

            except Exception as e:
                print(f"  {agent_name}: ERROR - {e}")
                batch_summary['agents'][agent_name] = {'error': str(e)}
                batch_summary['failed'] += 1

    batch_summary['completed'] = datetime.now().isoformat()

    print(f"\n=== BATCH UPLOAD COMPLETE ===")
    print(f"Successful: {batch_summary['successful']}")
    print(f"Skipped (already uploaded): {batch_summary['skipped']}")
    print(f"Failed: {batch_summary['failed']}")

    # Save batch summary
    output_path = Path(args.output) if args.output else input_base / '_batch_upload_summary.json'
    with open(output_path, 'w') as f:
        json.dump(batch_summary, f, indent=2)
    print(f"Summary saved to: {output_path}")


if __name__ == '__main__':
    asyncio.run(main())

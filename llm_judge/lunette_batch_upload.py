"""
Batch upload all converted trajectories to Lunette.

Usage:
    # Upload all agents
    python llm_judge/lunette_batch_upload.py

    # Upload specific agents
    python llm_judge/lunette_batch_upload.py --agents 20240620_sweagent_claude3.5sonnet 20240728_sweagent_gpt4o

    # Limit trajectories per agent (for testing)
    python llm_judge/lunette_batch_upload.py --limit_per_agent 50

    # Dry run
    python llm_judge/lunette_batch_upload.py --dry_run
"""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from lunette import LunetteClient

from lunette_upload import (
    load_results_for_agent,
    load_existing_uploads,
    save_upload_tracking,
    load_converted_trajectory,
    convert_to_lunette_format,
)
from lunette.models.run import Run


async def upload_agent(
    client: LunetteClient,
    agent_dir: Path,
    agent_name: str,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict:
    """Upload all trajectories for a single agent."""
    results = load_results_for_agent(agent_name)
    existing_uploads = load_existing_uploads(agent_dir)

    json_files = sorted(agent_dir.glob('*.json'))
    json_files = [f for f in json_files if not f.name.startswith('_')]

    if limit:
        json_files = json_files[:limit]

    summary = {
        'agent': agent_name,
        'total': len(json_files),
        'uploaded': 0,
        'skipped': len([f for f in json_files if f.stem in existing_uploads]),
        'errors': 0,
        'uploads': list(existing_uploads.values()),
    }

    to_upload = [f for f in json_files if f.stem not in existing_uploads]

    if not to_upload:
        print(f"  {agent_name}: All {len(json_files)} trajectories already uploaded")
        return summary

    print(f"  {agent_name}: Uploading {len(to_upload)}/{len(json_files)} trajectories...")

    for i, file_path in enumerate(to_upload):
        task_id = file_path.stem
        resolved = results.get(task_id, False)

        if dry_run:
            summary['uploaded'] += 1
            continue

        try:
            unified = load_converted_trajectory(file_path)
            lunette_traj = convert_to_lunette_format(unified, resolved=resolved, model_name=agent_name)
            run = Run(task="swebench-verified", model=agent_name, trajectories=[lunette_traj])

            run_meta = await client.save_run(run)
            run_id = run_meta['run_id']
            traj_id = run_meta.get('trajectory_ids', [None])[0]

            upload_record = {
                'task_id': task_id,
                'run_id': run_id,
                'trajectory_id': traj_id,
                'resolved': resolved,
                'message_count': len(unified.get('messages', [])),
            }
            summary['uploaded'] += 1
            summary['uploads'].append(upload_record)

            # Save progress every 50 uploads
            if summary['uploaded'] % 50 == 0:
                save_upload_tracking(agent_dir, summary['uploads'], agent_name)
                print(f"    Progress: {summary['uploaded']}/{len(to_upload)}")

        except Exception as e:
            print(f"    Error uploading {task_id}: {e}")
            summary['errors'] += 1

    # Final save
    if not dry_run and summary['uploads']:
        save_upload_tracking(agent_dir, summary['uploads'], agent_name)

    return summary


async def main():
    parser = argparse.ArgumentParser(description='Batch upload trajectories to Lunette')
    parser.add_argument('--agents', nargs='+', help='Specific agents to upload (default: all)')
    parser.add_argument('--limit_per_agent', type=int, help='Max trajectories per agent')
    parser.add_argument('--dry_run', action='store_true', help='Show what would be uploaded')
    parser.add_argument('--input_dir', type=str, default='trajectory_data/unified_trajs',
                        help='Base directory containing agent folders')
    parser.add_argument('--output', type=str, help='Output path for batch summary')

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

    print(f"=== Batch Upload to Lunette ===")
    print(f"Found {len(agent_dirs)} agents to process")
    if args.dry_run:
        print("DRY RUN - no uploads will be made\n")

    batch_summary = {
        'started': datetime.now().isoformat(),
        'total_agents': len(agent_dirs),
        'total_uploaded': 0,
        'total_skipped': 0,
        'total_errors': 0,
        'agents': {},
    }

    async with LunetteClient() as client:
        for i, agent_dir in enumerate(agent_dirs):
            agent_name = agent_dir.name
            print(f"\n[{i+1}/{len(agent_dirs)}] Processing {agent_name}...")

            try:
                summary = await upload_agent(
                    client=client,
                    agent_dir=agent_dir,
                    agent_name=agent_name,
                    limit=args.limit_per_agent,
                    dry_run=args.dry_run,
                )
                batch_summary['agents'][agent_name] = {
                    'uploaded': summary['uploaded'],
                    'skipped': summary['skipped'],
                    'errors': summary['errors'],
                    'total': summary['total'],
                }
                batch_summary['total_uploaded'] += summary['uploaded']
                batch_summary['total_skipped'] += summary['skipped']
                batch_summary['total_errors'] += summary['errors']

            except Exception as e:
                print(f"  Error processing {agent_name}: {e}")
                batch_summary['agents'][agent_name] = {'error': str(e)}

    batch_summary['completed'] = datetime.now().isoformat()

    print(f"\n=== BATCH UPLOAD COMPLETE ===")
    print(f"Agents processed: {len(agent_dirs)}")
    print(f"Total uploaded: {batch_summary['total_uploaded']}")
    print(f"Total skipped: {batch_summary['total_skipped']}")
    print(f"Total errors: {batch_summary['total_errors']}")

    # Save batch summary
    output_path = Path(args.output) if args.output else input_base / '_batch_upload_summary.json'
    with open(output_path, 'w') as f:
        json.dump(batch_summary, f, indent=2)
    print(f"Summary saved to: {output_path}")


if __name__ == '__main__':
    asyncio.run(main())

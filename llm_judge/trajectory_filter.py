"""
Filter SWE-bench trajectories to extract only editing behavior.

Removes chat messages, thoughts, and other potentially spurious signals,
keeping only the actions that modify code and their direct observations.

Usage:
    # Filter a single trajectory and print to stdout
    python llm_judge/trajectory_filter.py --traj path/to/task.traj

    # Filter a single trajectory and save to file
    python llm_judge/trajectory_filter.py --traj path/to/task.traj --output filtered.json

    # Filter all trajectories for an agent
    python llm_judge/trajectory_filter.py --agent 20240620_sweagent_claude3.5sonnet --output_dir filtered_trajs/

    # Filter with model name redaction disabled
    python llm_judge/trajectory_filter.py --traj path/to/task.traj --no_redact
"""

import argparse
import json
import re
from pathlib import Path
from typing import Optional


# Model/agent names to redact from observations
MODEL_PATTERNS = [
    r'claude', r'gpt-?4', r'gpt-?3\.?5', r'sonnet', r'opus', r'haiku',
    r'gemini', r'llama', r'mistral', r'qwen', r'deepseek',
    r'sweagent', r'swe-agent', r'autocoderover', r'openhands',
    r'anthropic', r'openai',
]


def redact_model_info(text: str, patterns: list[str] = MODEL_PATTERNS) -> str:
    """Redact potential model-identifying information from text."""
    result = text
    for pattern in patterns:
        result = re.sub(pattern, '[REDACTED]', result, flags=re.IGNORECASE)
    return result


def filter_trajectory(
    trajectory: dict,
    redact_models: bool = True,
    keep_thoughts: bool = False,
) -> dict:
    """
    Filter a trajectory to keep only editing-related actions.

    Args:
        trajectory: Raw SWE-bench trajectory dict
        redact_models: Whether to redact model names from observations
        keep_thoughts: Whether to keep 'thought' and 'response' fields

    Returns:
        Filtered trajectory dict with only edit-related steps
    """
    traj_list = trajectory.get('trajectory', [])
    filtered_steps = []

    for step in traj_list:
        action = step.get('action', '')
        observation = step.get('observation', '')
        action_lower = action.lower().strip()

        # Determine if this action is edit-related
        is_edit_related = any([
            # Direct file modifications
            action_lower.startswith('edit '),
            action_lower.startswith('create '),

            # File viewing (shows code context)
            action_lower.startswith('open '),
            action_lower.startswith('cat '),
            'File:' in observation,

            # Code execution (shows behavior)
            action_lower.startswith('python '),
            action_lower.startswith('pytest'),
            action_lower.startswith('make '),

            # Search that found results (shows what code was examined)
            action_lower.startswith('grep') and 'matches' in observation.lower(),
            action_lower.startswith('find_file') and 'Found' in observation,
            action_lower.startswith('search_file') and 'Found' in observation,
        ])

        if is_edit_related:
            # Process observation
            if redact_models:
                observation = redact_model_info(observation)

            filtered_step = {
                'action': action,
                'observation': observation,
            }

            # Optionally keep thought/response (usually omitted to avoid model signatures)
            if keep_thoughts:
                if 'thought' in step:
                    thought = step['thought']
                    if redact_models:
                        thought = redact_model_info(thought)
                    filtered_step['thought'] = thought
                if 'response' in step:
                    response = step['response']
                    if redact_models:
                        response = redact_model_info(response)
                    filtered_step['response'] = response

            filtered_steps.append(filtered_step)

    return {
        'trajectory': filtered_steps,
        'info': trajectory.get('info', {}),
        'environment': trajectory.get('environment', ''),
        '_filtered': True,
        '_original_steps': len(traj_list),
        '_filtered_steps': len(filtered_steps),
    }


def load_trajectory(path: Path) -> dict:
    """Load a trajectory from a .traj file."""
    with open(path) as f:
        return json.load(f)


def save_trajectory(trajectory: dict, path: Path) -> None:
    """Save a trajectory to a JSON file."""
    with open(path, 'w') as f:
        json.dump(trajectory, f, indent=2)


def filter_agent_trajectories(
    agent_dir: Path,
    output_dir: Path,
    redact_models: bool = True,
    keep_thoughts: bool = False,
) -> dict:
    """
    Filter all trajectories for an agent.

    Returns:
        Summary dict with counts
    """
    trajs_dir = agent_dir / 'trajs'
    if not trajs_dir.exists():
        raise FileNotFoundError(f"No trajectories found at {trajs_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        'total': 0,
        'filtered': 0,
        'original_steps': 0,
        'filtered_steps': 0,
    }

    for traj_path in sorted(trajs_dir.glob('*.traj')):
        traj = load_trajectory(traj_path)
        filtered = filter_trajectory(traj, redact_models=redact_models, keep_thoughts=keep_thoughts)

        output_path = output_dir / traj_path.name
        save_trajectory(filtered, output_path)

        summary['total'] += 1
        summary['filtered'] += 1
        summary['original_steps'] += filtered['_original_steps']
        summary['filtered_steps'] += filtered['_filtered_steps']

    return summary


def print_filtered_trajectory(trajectory: dict, max_obs_length: int = 500) -> None:
    """Pretty-print a filtered trajectory."""
    steps = trajectory.get('trajectory', [])
    original = trajectory.get('_original_steps', '?')
    filtered = trajectory.get('_filtered_steps', len(steps))

    print(f"Filtered trajectory: {filtered}/{original} steps kept")
    print("=" * 80)

    for i, step in enumerate(steps):
        action = step.get('action', '')
        observation = step.get('observation', '')

        # Truncate long observations
        if len(observation) > max_obs_length:
            observation = observation[:max_obs_length] + f"... [{len(observation) - max_obs_length} more chars]"

        print(f"\n[Step {i+1}]")
        print(f"ACTION: {action[:200]}{'...' if len(action) > 200 else ''}")
        print(f"OBSERVATION: {observation}")


def main():
    parser = argparse.ArgumentParser(
        description='Filter SWE-bench trajectories to editing behavior only',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View filtered trajectory
  python llm_judge/trajectory_filter.py --traj experiments/evaluation/verified/20240620_sweagent_claude3.5sonnet/trajs/django__django-10880.traj

  # Save filtered trajectory
  python llm_judge/trajectory_filter.py --traj path/to/task.traj --output filtered.json

  # Filter all trajectories for an agent
  python llm_judge/trajectory_filter.py --agent 20240620_sweagent_claude3.5sonnet --output_dir chris_output/filtered_trajs/
        """
    )

    parser.add_argument('--traj', type=str, help='Path to a single .traj file to filter')
    parser.add_argument('--agent', type=str, help='Agent name to filter all trajectories for')
    parser.add_argument('--output', type=str, help='Output path for single trajectory')
    parser.add_argument('--output_dir', type=str, help='Output directory for agent trajectories')
    parser.add_argument('--no_redact', action='store_true', help='Disable model name redaction')
    parser.add_argument('--keep_thoughts', action='store_true', help='Keep thought/response fields')
    parser.add_argument('--max_obs', type=int, default=500, help='Max observation length to print (default: 500)')

    args = parser.parse_args()

    if args.traj:
        # Filter single trajectory
        traj_path = Path(args.traj)
        if not traj_path.exists():
            print(f"Error: File not found: {traj_path}")
            return

        traj = load_trajectory(traj_path)
        filtered = filter_trajectory(
            traj,
            redact_models=not args.no_redact,
            keep_thoughts=args.keep_thoughts
        )

        if args.output:
            save_trajectory(filtered, Path(args.output))
            print(f"Saved filtered trajectory to {args.output}")
            print(f"  Original steps: {filtered['_original_steps']}")
            print(f"  Filtered steps: {filtered['_filtered_steps']}")
        else:
            print_filtered_trajectory(filtered, max_obs_length=args.max_obs)

    elif args.agent:
        # Filter all trajectories for agent
        experiments_dir = Path(__file__).resolve().parents[1] / 'experiments'
        agent_dir = experiments_dir / 'evaluation' / 'verified' / args.agent

        if not agent_dir.exists():
            print(f"Error: Agent directory not found: {agent_dir}")
            return

        output_dir = Path(args.output_dir) if args.output_dir else Path(f'chris_output/filtered_trajs/{args.agent}')

        print(f"Filtering trajectories for {args.agent}...")
        summary = filter_agent_trajectories(
            agent_dir,
            output_dir,
            redact_models=not args.no_redact,
            keep_thoughts=args.keep_thoughts
        )

        print(f"\nFiltered {summary['filtered']} trajectories")
        print(f"  Original steps: {summary['original_steps']}")
        print(f"  Filtered steps: {summary['filtered_steps']}")
        print(f"  Reduction: {100 * (1 - summary['filtered_steps'] / summary['original_steps']):.1f}%")
        print(f"  Output directory: {output_dir}")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()

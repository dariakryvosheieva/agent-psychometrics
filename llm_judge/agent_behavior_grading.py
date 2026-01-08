"""
Batch grading of agent behavioral signatures using Lunette.

This script:
1. Filters trajectories to editing behavior only (removes thoughts, chat, etc.)
2. Batch uploads filtered trajectories to Lunette
3. Grades them using a structured prompt for agent identification features

All features have discrete numerical outputs for downstream analysis.

Usage:
    # Upload and grade trajectories for one agent
    python llm_judge/agent_behavior_grading.py --agent 20240620_sweagent_claude3.5sonnet --num_tasks 50

    # Grade multiple agents for comparison
    python llm_judge/agent_behavior_grading.py --agents agent1,agent2,agent3 --num_tasks 30

    # Grade already-uploaded runs by ID
    python llm_judge/agent_behavior_grading.py --grade_runs run_id1,run_id2,run_id3
"""

import argparse
import asyncio
import json
import random
from pathlib import Path
from typing import Optional

import pandas as pd
import sys

# Add parent dir to path for local imports
sys.path.insert(0, str(Path(__file__).parent))
from trajectory_filter import filter_trajectory, load_trajectory

try:
    from lunette import LunetteClient
    from lunette.analysis import GradingPlan
    from lunette.models.run import Run
    from lunette.models.trajectory import Trajectory, ScalarScore
    from lunette.models.messages import UserMessage, AssistantMessage
    LUNETTE_AVAILABLE = True
except ImportError:
    LUNETTE_AVAILABLE = False
    print("Warning: lunette not installed. Run: pip install lunette-sdk")


# =============================================================================
# GRADING PROMPT: Agent Behavioral Signatures
# =============================================================================
# Each feature has a discrete numerical scale for quantitative analysis.
# Features are designed to distinguish problem-solving strategies, not artifacts.

AGENT_BEHAVIOR_GRADING_PROMPT = """You are analyzing a coding agent's trajectory to identify behavioral patterns.

IMPORTANT: You are seeing a FILTERED trajectory that contains only:
- File edits and creations
- Code execution results (python, pytest)
- File viewing operations
The agent's internal reasoning has been removed. Judge based on ACTIONS and RESULTS only.

## Behavioral Features to Grade

For each feature, provide a score on the specified scale.

### 1. LOCALIZATION STRATEGY (1-5)
How did the agent find the location of the bug/code to modify?
- 1: Appears random or lucky - jumped directly to a file without visible search
- 2: Simple keyword search - one grep/find that happened to work
- 3: Iterative search - multiple searches, narrowing down
- 4: Structural navigation - followed imports, class hierarchy, or call chains
- 5: Systematic exploration - methodically explored directory structure, read related files

### 2. HYPOTHESIS TESTING (0-3)
Did the agent test hypotheses before committing to a fix?
- 0: No evidence - went straight to editing without any verification
- 1: Minimal - ran the code once before/after
- 2: Moderate - created a reproduction script OR ran existing tests
- 3: Thorough - created reproduction AND ran tests, or multiple verification steps

### 3. INCREMENTAL VS BIG-BANG EDITING (1-5)
How did the agent apply changes?
- 1: Single large edit - one edit command with many lines changed
- 2: Few large edits - 2-3 substantial edits
- 3: Mixed approach - some large, some small edits
- 4: Mostly incremental - many small targeted edits
- 5: Highly incremental - very small, surgical edits with testing between each

### 4. ERROR RECOVERY (0-4)
How did the agent respond to errors or failed attempts?
- 0: No errors encountered OR gave up immediately after first error
- 1: Simple retry - repeated similar action hoping for different result
- 2: Minor adjustment - small tweak to the same approach
- 3: Approach change - tried a different strategy after failure
- 4: Systematic debugging - diagnosed the error, understood root cause, then fixed

### 5. VERIFICATION APPROACH (0-3)
How did the agent verify their fix worked?
- 0: No verification - edited and submitted without checking
- 1: Manual inspection - only looked at the code, no execution
- 2: Basic execution - ran the code or a simple test
- 3: Comprehensive - ran test suite OR created new tests OR multiple verification methods

### 6. EXPLORATION DEPTH (1-5)
How much of the codebase did the agent examine?
- 1: Minimal (1-2 files) - barely looked around
- 2: Shallow (3-5 files) - quick survey
- 3: Moderate (6-10 files) - reasonable exploration
- 4: Deep (11-20 files) - thorough investigation
- 5: Very deep (20+ files) - exhaustive exploration

### 7. EDIT PRECISION (1-5)
How focused and clean were the code changes?
- 1: Messy - unnecessary changes, formatting changes, unrelated modifications
- 2: Somewhat unfocused - some unnecessary changes mixed with the fix
- 3: Adequate - mostly relevant changes with minor extras
- 4: Clean - focused changes, minimal diff
- 5: Surgical - exactly the minimum changes needed, nothing extra

### 8. ITERATION COUNT (1-5)
How many edit-test cycles did the agent perform?
- 1: Single shot - one edit attempt (success or failure)
- 2: Two attempts - edited, tested, edited again
- 3: Few iterations (3-4 cycles)
- 4: Several iterations (5-7 cycles)
- 5: Many iterations (8+ cycles)

### 9. TEST CREATION (0-2)
Did the agent create new test code?
- 0: No - did not create any test files or test functions
- 1: Reproduction only - created a script to reproduce the issue but not formal tests
- 2: Yes - created actual test cases (pytest, unittest, etc.)

### 10. CONTEXT GATHERING (1-5)
How much context did the agent gather before editing?
- 1: None - started editing immediately
- 2: Minimal - glanced at one related file
- 3: Moderate - read the file being edited and maybe one other
- 4: Good - examined related files, imports, or dependencies
- 5: Thorough - studied multiple related files, understood the broader system

## Output Format

Respond with ONLY a valid JSON object (no markdown, no explanation outside JSON):
{
    "localization_strategy": <1-5>,
    "hypothesis_testing": <0-3>,
    "incremental_vs_big_bang": <1-5>,
    "error_recovery": <0-4>,
    "verification_approach": <0-3>,
    "exploration_depth": <1-5>,
    "edit_precision": <1-5>,
    "iteration_count": <1-5>,
    "test_creation": <0-2>,
    "context_gathering": <1-5>,
    "brief_summary": "<1-2 sentence description of the agent's approach>"
}
"""


def get_task_lists(submission_dir: Path) -> tuple[list[str], list[str]]:
    """Get sorted lists of resolved and unresolved task IDs."""
    trajs_dir = submission_dir / 'trajs'
    results_path = submission_dir / 'results' / 'results.json'

    with open(results_path) as f:
        results = json.load(f)

    resolved_set = set(results.get('resolved', []))
    all_tasks = sorted([f.stem for f in trajs_dir.glob('*.traj')])

    resolved = sorted([t for t in all_tasks if t in resolved_set])
    unresolved = sorted([t for t in all_tasks if t not in resolved_set])

    return resolved, unresolved


def convert_filtered_to_lunette(
    task_id: str,
    filtered_traj: dict,
    resolved: bool,
    agent_name: str,
) -> Trajectory:
    """Convert a filtered trajectory to Lunette format."""
    messages = []
    steps = filtered_traj.get('trajectory', [])

    for i, step in enumerate(steps):
        action = step.get('action', '')
        observation = step.get('observation', '')

        # Alternate user (action) and assistant (observation) messages
        messages.append(UserMessage(position=i*2, content=f"Action:\n{action}"))
        # Truncate very long observations
        obs_truncated = observation[:8000] if len(observation) > 8000 else observation
        messages.append(AssistantMessage(position=i*2+1, content=f"Result:\n{obs_truncated}"))

    info = filtered_traj.get('info', {})
    solution = info.get('submission')

    return Trajectory(
        sample=task_id,
        messages=messages,
        scores={'resolved': ScalarScore(value=1.0 if resolved else 0.0)},
        solution=solution,
        metadata={
            'agent': agent_name,
            'filtered': True,
            'original_steps': filtered_traj.get('_original_steps', 'unknown'),
            'filtered_steps': filtered_traj.get('_filtered_steps', len(steps)),
        }
    )


async def upload_filtered_trajectories(
    client,
    submission_dir: Path,
    agent_name: str,
    task_ids: list[str],
    resolved_set: set[str],
) -> dict[str, str]:
    """
    Batch upload filtered trajectories to Lunette.

    Returns:
        Dict mapping task_id -> run_id
    """
    trajs_dir = submission_dir / 'trajs'
    task_to_run = {}

    print(f"\nUploading {len(task_ids)} filtered trajectories...")

    for i, task_id in enumerate(task_ids):
        traj_path = trajs_dir / f"{task_id}.traj"
        if not traj_path.exists():
            print(f"  [{i+1}/{len(task_ids)}] {task_id}: SKIP (not found)")
            continue

        try:
            # Load and filter
            raw_traj = load_trajectory(traj_path)
            filtered_traj = filter_trajectory(raw_traj, redact_models=True, keep_thoughts=False)

            # Convert to Lunette format
            lunette_traj = convert_filtered_to_lunette(
                task_id=task_id,
                filtered_traj=filtered_traj,
                resolved=task_id in resolved_set,
                agent_name=agent_name,
            )

            # Upload
            run = Run(
                task="swebench-verified",
                model=agent_name,
                trajectories=[lunette_traj],
            )
            run_meta = await client.save_run(run)
            run_id = run_meta['run_id']
            task_to_run[task_id] = run_id

            orig = filtered_traj.get('_original_steps', '?')
            filt = filtered_traj.get('_filtered_steps', '?')
            print(f"  [{i+1}/{len(task_ids)}] {task_id}: {run_id} ({filt}/{orig} steps)")

        except Exception as e:
            print(f"  [{i+1}/{len(task_ids)}] {task_id}: ERROR - {e}")

    return task_to_run


async def grade_runs_batch(
    client,
    task_to_run: dict[str, str],
    resolved_set: set[str],
    agent_name: str,
    output_path: Path,
) -> pd.DataFrame:
    """
    Batch grade uploaded runs.

    Saves results incrementally to output_path.
    """
    all_results = []

    print(f"\nGrading {len(task_to_run)} runs...")

    for i, (task_id, run_id) in enumerate(task_to_run.items()):
        print(f"  [{i+1}/{len(task_to_run)}] {task_id}...", end=" ", flush=True)

        try:
            results = await client.investigate(
                run_id=run_id,
                plan=GradingPlan(
                    name="agent-behavior",
                    prompt=AGENT_BEHAVIOR_GRADING_PROMPT,
                ),
                limit=1,
            )

            if results.results:
                result_data = results.results[0].data
                result_data["task_id"] = task_id
                result_data["run_id"] = run_id
                result_data["agent"] = agent_name
                result_data["resolved"] = task_id in resolved_set
                all_results.append(result_data)

                # Print key metrics
                loc = result_data.get('localization_strategy', '?')
                prec = result_data.get('edit_precision', '?')
                print(f"loc={loc}, precision={prec}")
            else:
                print("NO RESULTS")
                all_results.append({
                    "task_id": task_id,
                    "run_id": run_id,
                    "agent": agent_name,
                    "error": "No grading results"
                })

            # Save incrementally
            df = pd.DataFrame(all_results)
            df.to_csv(output_path, index=False)

        except Exception as e:
            print(f"ERROR - {e}")
            all_results.append({
                "task_id": task_id,
                "run_id": run_id,
                "agent": agent_name,
                "error": str(e)
            })

    return pd.DataFrame(all_results)


async def process_agent(
    client,
    agent_name: str,
    submission_dir: Path,
    num_tasks: int,
    output_dir: Path,
    seed: int,
) -> pd.DataFrame:
    """Process a single agent: upload filtered trajectories and grade them."""

    resolved, unresolved = get_task_lists(submission_dir)
    resolved_set = set(resolved)

    print(f"\nAgent: {agent_name}")
    print(f"  Found {len(resolved)} resolved, {len(unresolved)} unresolved tasks")

    # Sample tasks (balanced between resolved and unresolved)
    random.seed(seed)
    n_each = num_tasks // 2
    sampled_resolved = random.sample(resolved, min(n_each, len(resolved)))
    sampled_unresolved = random.sample(unresolved, min(n_each, len(unresolved)))
    task_ids = sampled_resolved + sampled_unresolved

    print(f"  Sampled {len(task_ids)} tasks (seed={seed})")
    print(f"  First 5: {task_ids[:5]}")

    # Step 1: Upload filtered trajectories
    task_to_run = await upload_filtered_trajectories(
        client=client,
        submission_dir=submission_dir,
        agent_name=agent_name,
        task_ids=task_ids,
        resolved_set=resolved_set,
    )

    # Save run IDs for potential re-grading later
    run_ids_path = output_dir / f"{agent_name}_run_ids.json"
    with open(run_ids_path, 'w') as f:
        json.dump(task_to_run, f, indent=2)
    print(f"\n  Saved run IDs to {run_ids_path}")

    # Step 2: Grade all runs
    output_path = output_dir / f"{agent_name}_behavior_grades.csv"
    df = await grade_runs_batch(
        client=client,
        task_to_run=task_to_run,
        resolved_set=resolved_set,
        agent_name=agent_name,
        output_path=output_path,
    )

    print(f"\n  Saved grades to {output_path}")
    return df


async def grade_existing_runs(
    client,
    run_ids: list[str],
    output_path: Path,
) -> pd.DataFrame:
    """Grade already-uploaded runs by their IDs."""
    all_results = []

    print(f"\nGrading {len(run_ids)} existing runs...")

    for i, run_id in enumerate(run_ids):
        print(f"  [{i+1}/{len(run_ids)}] {run_id}...", end=" ", flush=True)

        try:
            results = await client.investigate(
                run_id=run_id,
                plan=GradingPlan(
                    name="agent-behavior",
                    prompt=AGENT_BEHAVIOR_GRADING_PROMPT,
                ),
                limit=1,
            )

            if results.results:
                result_data = results.results[0].data
                result_data["run_id"] = run_id
                all_results.append(result_data)
                loc = result_data.get('localization_strategy', '?')
                prec = result_data.get('edit_precision', '?')
                print(f"loc={loc}, precision={prec}")
            else:
                print("NO RESULTS")
                all_results.append({"run_id": run_id, "error": "No grading results"})

            # Save incrementally
            df = pd.DataFrame(all_results)
            df.to_csv(output_path, index=False)

        except Exception as e:
            print(f"ERROR - {e}")
            all_results.append({"run_id": run_id, "error": str(e)})

    return pd.DataFrame(all_results)


def print_summary(df: pd.DataFrame, agent_name: str = "Agent"):
    """Print summary statistics for graded features."""
    feature_cols = [
        'localization_strategy', 'hypothesis_testing', 'incremental_vs_big_bang',
        'error_recovery', 'verification_approach', 'exploration_depth',
        'edit_precision', 'iteration_count', 'test_creation', 'context_gathering'
    ]

    available = [c for c in feature_cols if c in df.columns]
    if not available:
        print("No feature data available")
        return

    print(f"\n{'='*60}")
    print(f"BEHAVIORAL PROFILE: {agent_name}")
    print(f"{'='*60}")
    print(f"Tasks graded: {len(df)}")

    if 'resolved' in df.columns:
        print(f"Resolution rate: {df['resolved'].mean()*100:.1f}%")

    print(f"\nFeature Means (higher = more of that behavior):")
    print("-" * 40)
    for col in available:
        mean = df[col].mean()
        std = df[col].std()
        print(f"  {col:25s}: {mean:.2f} (std={std:.2f})")


async def main():
    parser = argparse.ArgumentParser(
        description='Batch grade agent behavioral signatures using Lunette',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--agent', type=str, help='Single agent to process')
    parser.add_argument('--agents', type=str, help='Comma-separated list of agents')
    parser.add_argument('--num_tasks', type=int, default=50, help='Tasks per agent (default: 50)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling')
    parser.add_argument('--output_dir', type=str, default='chris_output/behavior_grades',
                        help='Output directory')
    parser.add_argument('--grade_runs', type=str, help='Comma-separated run IDs to grade (skip upload)')
    parser.add_argument('--grade_from_file', type=str, help='JSON file with task_id->run_id mapping')

    args = parser.parse_args()

    if not LUNETTE_AVAILABLE:
        print("Error: lunette SDK not installed. Run: pip install lunette-sdk")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments_dir = Path(__file__).resolve().parents[1] / 'experiments'
    verified_dir = experiments_dir / 'evaluation' / 'verified'

    async with LunetteClient() as client:

        if args.grade_runs:
            # Grade existing runs by ID
            run_ids = [r.strip() for r in args.grade_runs.split(',')]
            output_path = output_dir / 'graded_runs.csv'
            df = await grade_existing_runs(client, run_ids, output_path)
            print_summary(df)

        elif args.grade_from_file:
            # Grade runs from a saved mapping file
            with open(args.grade_from_file) as f:
                task_to_run = json.load(f)
            output_path = output_dir / 'graded_from_file.csv'
            df = await grade_runs_batch(
                client=client,
                task_to_run=task_to_run,
                resolved_set=set(),  # Unknown without original data
                agent_name="unknown",
                output_path=output_path,
            )
            print_summary(df)

        elif args.agent:
            # Process single agent
            submission_dir = verified_dir / args.agent
            if not submission_dir.exists():
                print(f"Error: Agent not found: {submission_dir}")
                return
            if not (submission_dir / 'trajs').exists():
                print(f"Error: No trajectories for {args.agent}")
                print(f"Download with: cd experiments && python -m analysis.download_logs evaluation/verified/{args.agent} --only_trajs")
                return

            df = await process_agent(
                client=client,
                agent_name=args.agent,
                submission_dir=submission_dir,
                num_tasks=args.num_tasks,
                output_dir=output_dir,
                seed=args.seed,
            )
            print_summary(df, args.agent)

        elif args.agents:
            # Process multiple agents
            agent_names = [a.strip() for a in args.agents.split(',')]
            all_dfs = []

            for agent_name in agent_names:
                submission_dir = verified_dir / agent_name
                if not (submission_dir / 'trajs').exists():
                    print(f"\nSkipping {agent_name} - no trajectories")
                    continue

                df = await process_agent(
                    client=client,
                    agent_name=agent_name,
                    submission_dir=submission_dir,
                    num_tasks=args.num_tasks,
                    output_dir=output_dir,
                    seed=args.seed,
                )
                all_dfs.append(df)
                print_summary(df, agent_name)

            # Save combined results
            if all_dfs:
                combined = pd.concat(all_dfs, ignore_index=True)
                combined.to_csv(output_dir / 'all_agents_behavior_grades.csv', index=False)
                print(f"\nSaved combined results to {output_dir / 'all_agents_behavior_grades.csv'}")

        else:
            parser.print_help()


if __name__ == '__main__':
    asyncio.run(main())

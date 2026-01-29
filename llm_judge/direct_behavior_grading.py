"""
Direct LLM-as-a-judge grading of agent behavioral signatures.

This script grades filtered trajectories directly using the Anthropic API.

Usage:
    python llm_judge/direct_behavior_grading.py --num_tasks 20
    python llm_judge/direct_behavior_grading.py --agents agent1,agent2 --num_tasks 20
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# Add parent dir for local imports
sys.path.insert(0, str(Path(__file__).parent))
from trajectory_filter import filter_trajectory, load_trajectory

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Warning: anthropic not installed. Run: pip install anthropic")


# Same grading prompt as agent_behavior_grading.py
GRADING_PROMPT = """You are analyzing a coding agent's trajectory to identify behavioral patterns.

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


def format_trajectory_for_grading(filtered_traj: dict, max_chars: int = 30000) -> str:
    """Format a filtered trajectory as text for the LLM."""
    steps = filtered_traj.get('trajectory', [])

    lines = []
    lines.append(f"=== FILTERED TRAJECTORY ({len(steps)} steps) ===\n")

    total_chars = 0
    for i, step in enumerate(steps):
        action = step.get('action', '')
        observation = step.get('observation', '')

        # Truncate very long observations
        if len(observation) > 2000:
            observation = observation[:2000] + f"\n... [truncated, {len(observation)-2000} more chars]"

        step_text = f"\n--- Step {i+1} ---\nACTION:\n{action}\n\nRESULT:\n{observation}\n"

        if total_chars + len(step_text) > max_chars:
            lines.append(f"\n... [trajectory truncated at step {i+1}/{len(steps)}]")
            break

        lines.append(step_text)
        total_chars += len(step_text)

    return ''.join(lines)


def grade_trajectory(client: anthropic.Anthropic, trajectory_text: str, model: str = "claude-sonnet-4-20250514") -> dict:
    """Grade a single trajectory using the Anthropic API."""

    message = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"{GRADING_PROMPT}\n\n## TRAJECTORY TO ANALYZE:\n\n{trajectory_text}"
            }
        ]
    )

    response_text = message.content[0].text.strip()

    # Parse JSON from response
    try:
        # Try to find JSON in the response
        json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            return {"error": "No JSON found in response", "raw_response": response_text[:500]}
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {e}", "raw_response": response_text[:500]}


def grade_agent(
    client: anthropic.Anthropic,
    agent_name: str,
    submission_dir: Path,
    num_tasks: int,
    seed: int,
    model: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Grade trajectories for a single agent."""

    trajs_dir = submission_dir / 'trajs'
    resolved, unresolved = get_task_lists(submission_dir)
    resolved_set = set(resolved)

    print(f"\n{'='*60}")
    print(f"Agent: {agent_name}")
    print(f"{'='*60}")
    print(f"Found {len(resolved)} resolved, {len(unresolved)} unresolved tasks")

    # Sample tasks (balanced)
    random.seed(seed)
    n_each = num_tasks // 2
    sampled_resolved = random.sample(resolved, min(n_each, len(resolved)))
    sampled_unresolved = random.sample(unresolved, min(n_each, len(unresolved)))
    task_ids = sampled_resolved + sampled_unresolved

    print(f"Sampled {len(task_ids)} tasks (seed={seed})")
    print(f"Tasks: {task_ids}")

    results = []

    for i, task_id in enumerate(task_ids):
        print(f"\n[{i+1}/{len(task_ids)}] {task_id}...", end=" ", flush=True)

        traj_path = trajs_dir / f"{task_id}.traj"
        if not traj_path.exists():
            print("NOT FOUND")
            continue

        try:
            # Load and filter
            raw_traj = load_trajectory(traj_path)
            filtered_traj = filter_trajectory(raw_traj, redact_models=True, keep_thoughts=False)

            # Format for grading
            traj_text = format_trajectory_for_grading(filtered_traj)

            # Grade
            grades = grade_trajectory(client, traj_text, model=model)
            grades['task_id'] = task_id
            grades['agent'] = agent_name
            grades['resolved'] = task_id in resolved_set
            grades['original_steps'] = filtered_traj.get('_original_steps', 0)
            grades['filtered_steps'] = filtered_traj.get('_filtered_steps', 0)

            results.append(grades)

            if 'error' not in grades:
                print(f"loc={grades.get('localization_strategy')}, prec={grades.get('edit_precision')}")
            else:
                print(f"ERROR: {grades.get('error', 'unknown')[:50]}")

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                'task_id': task_id,
                'agent': agent_name,
                'error': str(e)
            })

    return pd.DataFrame(results), task_ids


def compute_variance_analysis(all_results: pd.DataFrame) -> pd.DataFrame:
    """Compute within-agent and between-agent variance for each feature."""

    feature_cols = [
        'localization_strategy', 'hypothesis_testing', 'incremental_vs_big_bang',
        'error_recovery', 'verification_approach', 'exploration_depth',
        'edit_precision', 'iteration_count', 'test_creation', 'context_gathering'
    ]

    available_features = [c for c in feature_cols if c in all_results.columns]

    variance_data = []

    for feature in available_features:
        # Convert to numeric, coercing errors to NaN
        all_results[feature] = pd.to_numeric(all_results[feature], errors='coerce')

        # Within-agent variance (average variance within each agent)
        within_var = all_results.groupby('agent')[feature].var().mean()

        # Between-agent variance (variance of agent means)
        agent_means = all_results.groupby('agent')[feature].mean()
        between_var = agent_means.var()

        # Overall stats
        overall_mean = all_results[feature].mean()
        overall_std = all_results[feature].std()

        # Discrimination ratio: high between / low within = good discriminator
        discrimination = between_var / within_var if within_var > 0 else float('inf')

        variance_data.append({
            'feature': feature,
            'overall_mean': overall_mean,
            'overall_std': overall_std,
            'within_agent_var': within_var,
            'between_agent_var': between_var,
            'discrimination_ratio': discrimination,
        })

    return pd.DataFrame(variance_data).sort_values('discrimination_ratio', ascending=False)


def print_agent_profiles(all_results: pd.DataFrame):
    """Print mean feature values for each agent."""

    feature_cols = [
        'localization_strategy', 'hypothesis_testing', 'incremental_vs_big_bang',
        'error_recovery', 'verification_approach', 'exploration_depth',
        'edit_precision', 'iteration_count', 'test_creation', 'context_gathering'
    ]

    available_features = [c for c in feature_cols if c in all_results.columns]

    print("\n" + "="*80)
    print("AGENT BEHAVIORAL PROFILES (mean scores)")
    print("="*80)

    # Convert to numeric
    for f in available_features:
        all_results[f] = pd.to_numeric(all_results[f], errors='coerce')

    profiles = all_results.groupby('agent')[available_features].mean()
    print(profiles.round(2).to_string())


def main():
    parser = argparse.ArgumentParser(description='Direct LLM grading of agent behavioral signatures')
    parser.add_argument('--agents', type=str, default=None,
                        help='Comma-separated agent names (default: all with trajectories)')
    parser.add_argument('--num_tasks', type=int, default=20,
                        help='Tasks to grade per agent (default: 20)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sampling')
    parser.add_argument('--model', type=str, default='claude-sonnet-4-20250514',
                        help='Model to use for grading')
    parser.add_argument('--output_dir', type=str, default='chris_output/direct_grading',
                        help='Output directory')
    args = parser.parse_args()

    if not ANTHROPIC_AVAILABLE:
        print("Error: anthropic package not installed")
        return

    # Check API key
    if not os.environ.get('ANTHROPIC_API_KEY'):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        return

    client = anthropic.Anthropic()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments_dir = Path(__file__).resolve().parents[1] / 'experiments'
    verified_dir = experiments_dir / 'evaluation' / 'verified'

    # Find agents with trajectories
    if args.agents:
        agent_names = [a.strip() for a in args.agents.split(',')]
    else:
        agent_names = []
        for d in verified_dir.iterdir():
            trajs_dir = d / 'trajs'
            if trajs_dir.exists() and list(trajs_dir.glob('*.traj')):
                agent_names.append(d.name)
        agent_names = sorted(agent_names)

    print(f"Agents to grade: {agent_names}")
    print(f"Tasks per agent: {args.num_tasks}")
    print(f"Model: {args.model}")

    all_results = []
    all_task_ids = {}

    for agent_name in agent_names:
        submission_dir = verified_dir / agent_name
        if not (submission_dir / 'trajs').exists():
            print(f"\nSkipping {agent_name} - no trajectories")
            continue

        df, task_ids = grade_agent(
            client=client,
            agent_name=agent_name,
            submission_dir=submission_dir,
            num_tasks=args.num_tasks,
            seed=args.seed,
            model=args.model,
        )

        all_results.append(df)
        all_task_ids[agent_name] = task_ids

        # Save individual agent results
        df.to_csv(output_dir / f'{agent_name}_grades.csv', index=False)
        print(f"\nSaved to {output_dir / f'{agent_name}_grades.csv'}")

    if not all_results:
        print("No results to analyze")
        return

    # Combine all results
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(output_dir / 'all_agents_grades.csv', index=False)

    # Save task ID mapping
    with open(output_dir / 'sampled_task_ids.json', 'w') as f:
        json.dump(all_task_ids, f, indent=2)

    # Print agent profiles
    print_agent_profiles(combined)

    # Compute and print variance analysis
    variance_df = compute_variance_analysis(combined)
    variance_df.to_csv(output_dir / 'variance_analysis.csv', index=False)

    print("\n" + "="*80)
    print("FEATURE DISCRIMINATION ANALYSIS")
    print("(high discrimination = low within-agent variance, high between-agent variance)")
    print("="*80)
    print(variance_df.round(3).to_string(index=False))

    print(f"\n\nAll results saved to {output_dir}/")
    print(f"  - all_agents_grades.csv: Combined grading results")
    print(f"  - sampled_task_ids.json: Which tasks were graded per agent")
    print(f"  - variance_analysis.csv: Feature discrimination analysis")


if __name__ == '__main__':
    main()

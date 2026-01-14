"""Batch grading of uploaded Lunette trajectories for difficulty prediction.

This script grades a sample of tasks across multiple agents using the same
fixed evaluation prompt from lunette_analysis.py, but operates on already-
uploaded trajectories rather than uploading new ones.

Usage:
    # Grade 50 tasks across 3 agents
    python llm_judge/lunette_batch_grading.py --n_tasks 50 --n_agents 3

    # Grade specific agents
    python llm_judge/lunette_batch_grading.py --agents 20240620_sweagent_claude3.5sonnet --n_tasks 50

    # Dry run to see execution plan
    python llm_judge/lunette_batch_grading.py --dry_run
"""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from scipy import stats

from lunette import LunetteClient
from lunette.analysis import GradingPlan


# Same grading prompt as lunette_analysis.py
TASK_GRADING_PROMPT = """You are analyzing a SWE-bench coding task to predict its difficulty.

## Your Goals
1. Evaluate features of the TASK ITSELF that predict difficulty
2. Note trajectory-based signals that suggest the task is hard

## Task-Intrinsic Features (about the problem, not the agent)

Evaluate each feature based on the problem statement and gold patch:

1. **fix_in_description** (0-3): Does the problem statement contain or suggest the fix?
   - 0: No hint at the solution
   - 1: Vague hint or direction
   - 2: Clear description of what needs to change
   - 3: Exact code fix provided in the description

2. **problem_clarity** (1-5): How clear and well-specified is the problem?
   - 1: Very vague, unclear what's wrong
   - 3: Reasonably clear but some ambiguity
   - 5: Crystal clear with reproduction steps and expected behavior

3. **error_message_provided** (0/1): Does the problem include an error message or traceback?

4. **reproduction_steps** (0/1): Are concrete reproduction steps provided?

5. **fix_locality** (1-3): How localized is the fix?
   - 1: Single location, few lines changed
   - 2: Multiple locations in same file, or moderate changes
   - 3: Multiple files or significant changes

6. **domain_knowledge_required** (1-5): How much specialized knowledge is needed?
   - 1: Basic Python, obvious fix
   - 3: Framework-specific knowledge (Django, pytest, etc.)
   - 5: Obscure APIs, protocols, or very specialized knowledge

7. **fix_complexity** (1-5): How complex is the actual fix?
   - 1: Trivial (add parameter, change value, simple one-liner)
   - 3: Moderate (requires understanding context)
   - 5: Very complex (architectural changes, subtle edge cases)

## Trajectory-Based Difficulty Signals

Observe the agent's trajectory for signals that indicate task difficulty:

8. **agent_declared_success_wrongly** (0/1): Did the agent claim to have solved it but was wrong?
   - This suggests the task has subtle correctness requirements

9. **agent_looping** (0/1): Was the agent stuck in a loop or repeating similar actions?
   - This suggests the task lacks clear progress indicators

10. **agent_expressed_uncertainty** (0/1): Did the agent express confusion or doubt?
    - This suggests the task is ambiguous or requires non-obvious reasoning

11. **agent_wrong_file_focus** (0/1): Did the agent spend significant time on irrelevant files?
    - This suggests the task requires difficult code localization

12. **agent_gave_up_early** (0/1): Did the agent stop trying before exhausting options?
    - This suggests the task seemed intractable to the agent

Respond with ONLY a JSON object:
{
    "fix_in_description": <0-3>,
    "problem_clarity": <1-5>,
    "error_message_provided": <0 or 1>,
    "reproduction_steps": <0 or 1>,
    "fix_locality": <1-3>,
    "domain_knowledge_required": <1-5>,
    "fix_complexity": <1-5>,
    "agent_declared_success_wrongly": <0 or 1>,
    "agent_looping": <0 or 1>,
    "agent_expressed_uncertainty": <0 or 1>,
    "agent_wrong_file_focus": <0 or 1>,
    "agent_gave_up_early": <0 or 1>,
    "reasoning": "<2-3 sentence explanation of key difficulty factors>"
}
"""


def load_upload_tracking(agent_dir: Path) -> Optional[Dict]:
    """Load upload tracking file for an agent."""
    tracking_file = agent_dir / "_lunette_uploads.json"
    if not tracking_file.exists():
        return None

    with open(tracking_file) as f:
        return json.load(f)


def get_available_agents(trajectories_dir: Path) -> List[str]:
    """Get list of agents with uploaded trajectories."""
    agents = []
    for agent_dir in sorted(trajectories_dir.iterdir()):
        if not agent_dir.is_dir() or agent_dir.name.startswith("_"):
            continue
        if (agent_dir / "_lunette_uploads.json").exists():
            agents.append(agent_dir.name)
    return agents


async def build_task_to_run_mapping(
    client: LunetteClient,
    run_ids: List[str],
) -> Dict[str, str]:
    """Build mapping from task_id to run_id by querying Lunette API.

    Args:
        client: LunetteClient instance.
        run_ids: List of run IDs to check.

    Returns:
        Dict mapping task_id -> run_id.
    """
    task_to_run = {}

    # Query each run to get its task list
    for run_id in run_ids:
        try:
            run = await client.get_run(run_id)
            # run.trajectories is a list of Trajectory objects
            # Each trajectory has a 'sample' field which is the task_id
            for traj in run.trajectories:
                task_to_run[traj.sample] = run_id
        except Exception as e:
            print(f"  Warning: Failed to fetch run {run_id[:16]}...: {e}")
            continue

    return task_to_run


async def grade_trajectory(
    client: LunetteClient,
    run_id: str,
    task_id: str,
    trajectory_id: Optional[str] = None,
) -> Dict:
    """Grade a single trajectory.

    Args:
        client: LunetteClient instance.
        run_id: Lunette run ID.
        task_id: SWE-bench task ID.
        trajectory_id: Optional specific trajectory ID.

    Returns:
        Dict with grading results.
    """
    try:
        results = await client.investigate(
            run_id=run_id,
            plan=GradingPlan(
                name="task-difficulty",
                prompt=TASK_GRADING_PROMPT,
            ),
            limit=1,
        )

        if not results.results:
            return {"task_id": task_id, "error": "No results returned"}

        result_data = results.results[0].data
        result_data["task_id"] = task_id
        result_data["run_id"] = run_id
        return result_data

    except Exception as e:
        return {"task_id": task_id, "run_id": run_id, "error": str(e)}


async def main():
    parser = argparse.ArgumentParser(
        description="Batch grade Lunette trajectories for difficulty prediction"
    )
    parser.add_argument(
        "--n_tasks",
        type=int,
        default=50,
        help="Number of tasks to grade per agent",
    )
    parser.add_argument(
        "--n_agents",
        type=int,
        default=3,
        help="Number of agents to grade",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=None,
        help="Specific agent names to grade (default: random sample)",
    )
    parser.add_argument(
        "--items_path",
        type=str,
        default="clean_data/swebench_verified_20250930_full/1d/items.csv",
        help="Path to IRT items.csv",
    )
    parser.add_argument(
        "--trajectories_dir",
        type=str,
        default="trajectory_data/unified_trajs",
        help="Base directory containing uploaded trajectories",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="chris_output/lunette_grading",
        help="Output directory for results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show execution plan without running grading",
    )

    args = parser.parse_args()

    trajectories_dir = Path(args.trajectories_dir)
    items_path = Path(args.items_path)

    # Load IRT items
    print(f"Loading IRT items from {items_path}...")
    items_df = pd.read_csv(items_path, index_col=0)
    print(f"  Loaded {len(items_df)} tasks")

    # Get available agents
    available_agents = get_available_agents(trajectories_dir)
    print(f"\nFound {len(available_agents)} agents with uploaded trajectories")

    # Select agents to grade
    if args.agents:
        selected_agents = [a for a in args.agents if a in available_agents]
        if not selected_agents:
            print(f"Error: None of the specified agents found in {trajectories_dir}")
            return
    else:
        import random
        random.seed(args.seed)
        selected_agents = random.sample(
            available_agents,
            min(args.n_agents, len(available_agents)),
        )

    print(f"\nSelected {len(selected_agents)} agents:")
    for agent in selected_agents:
        print(f"  - {agent}")

    # Build trajectory-to-run mapping (use stored mapping if available)
    print("\nBuilding trajectory-to-run mapping...")

    agent_mappings = {}
    agents_need_api = []

    # First pass: check for stored mappings
    for agent in selected_agents:
        agent_dir = trajectories_dir / agent
        upload_data = load_upload_tracking(agent_dir)

        if not upload_data:
            print(f"  Warning: No upload data for {agent}")
            continue

        run_ids = upload_data.get("run_ids", [upload_data.get("run_id")])
        trajectories = upload_data.get("trajectories", [])

        # Handle case where run_id might be None
        if not run_ids or run_ids[0] is None:
            print(f"  Warning: No valid run IDs for {agent}")
            continue

        # Check if we have a stored task-to-run mapping
        if "task_to_run_map" in upload_data:
            task_to_run = upload_data["task_to_run_map"]
            updated_at = upload_data.get("task_to_run_map_updated_at", "unknown")
            print(f"  {agent}: Using stored mapping ({len(task_to_run)} tasks, updated {updated_at[:10]})")

            agent_mappings[agent] = {
                "upload_data": upload_data,
                "task_to_run": task_to_run,
            }
        else:
            print(f"  {agent}: No stored mapping, will query Lunette API")
            agents_need_api.append((agent, agent_dir, upload_data, run_ids))

    # Second pass: query API for agents without stored mappings
    if agents_need_api:
        print(f"\nQuerying Lunette API for {len(agents_need_api)} agent(s)...")
        print("(Tip: Run 'python trajectory_upload/lunette_augment_mappings.py' to pre-compute all mappings)")

        async with LunetteClient() as client:
            for agent, agent_dir, upload_data, run_ids in agents_need_api:
                print(f"  {agent}: Querying {len(run_ids)} run(s)...")
                task_to_run = await build_task_to_run_mapping(client, run_ids)

                agent_mappings[agent] = {
                    "upload_data": upload_data,
                    "task_to_run": task_to_run,
                }

                print(f"    Mapped {len(task_to_run)} tasks across {len(run_ids)} run(s)")

    # Build list of trajectories to grade
    trajectories_to_grade = []

    for agent in selected_agents:
        if agent not in agent_mappings:
            continue

        upload_data = agent_mappings[agent]["upload_data"]
        task_to_run = agent_mappings[agent]["task_to_run"]
        trajectories = upload_data.get("trajectories", [])

        # Filter to tasks in IRT data
        valid_trajs = [
            t for t in trajectories
            if t["task_id"] in items_df.index
        ]

        # Sample n_tasks
        import random
        random.seed(args.seed + hash(agent))
        sampled_trajs = random.sample(
            valid_trajs,
            min(args.n_tasks, len(valid_trajs)),
        )

        for traj in sampled_trajs:
            task_id = traj["task_id"]
            trajectory_id = traj.get("trajectory_id")

            # Look up which run this task belongs to
            run_id = task_to_run.get(task_id)

            if not run_id:
                print(f"  Warning: No run_id found for {task_id}")
                continue

            trajectories_to_grade.append({
                "agent": agent,
                "task_id": task_id,
                "run_id": run_id,
                "trajectory_id": trajectory_id,
                "difficulty": items_df.loc[task_id, "b"],
                "resolved": traj.get("resolved", False),
            })

    print(f"\n=== Grading Plan ===")
    print(f"Total trajectories to grade: {len(trajectories_to_grade)}")
    print(f"Unique tasks: {len(set(t['task_id'] for t in trajectories_to_grade))}")

    # Show run distribution
    run_distribution = {}
    for traj in trajectories_to_grade:
        run_id = traj["run_id"]
        run_distribution[run_id] = run_distribution.get(run_id, 0) + 1

    if len(run_distribution) > 1:
        print(f"Trajectories distributed across {len(run_distribution)} run(s):")
        for run_id, count in sorted(run_distribution.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  - {run_id[:16]}...: {count} trajectories")

    if args.dry_run:
        print("\nDRY RUN - not executing grading")
        print(f"\nEstimated cost (assuming ~$0.50 per grading):")
        print(f"  ${len(trajectories_to_grade) * 0.50:.2f}")
        return

    # Grade trajectories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"grading_results_{timestamp}.csv"

    print(f"\n=== Grading Trajectories ===")
    print(f"Output: {output_file}")

    all_results = []

    async with LunetteClient() as client:
        for i, traj_info in enumerate(trajectories_to_grade):
            print(f"\n[{i+1}/{len(trajectories_to_grade)}] {traj_info['agent']} / {traj_info['task_id']}")

            result = await grade_trajectory(
                client=client,
                run_id=traj_info["run_id"],
                task_id=traj_info["task_id"],
                trajectory_id=traj_info["trajectory_id"],
            )

            # Add metadata
            result["agent"] = traj_info["agent"]
            result["difficulty"] = traj_info["difficulty"]
            result["resolved"] = traj_info["resolved"]

            all_results.append(result)

            # Save incrementally
            df = pd.DataFrame(all_results)
            df.to_csv(output_file, index=False)

            if "error" not in result:
                print(f"  fix_complexity={result.get('fix_complexity')}, "
                      f"domain_knowledge={result.get('domain_knowledge_required')}")
            else:
                print(f"  ERROR: {result['error']}")

    # Compute correlations and discrimination
    df = pd.DataFrame(all_results)

    # Filter to valid results
    valid_df = df[~df["error"].notna()].copy()

    feature_cols = [
        "fix_in_description", "problem_clarity", "error_message_provided",
        "reproduction_steps", "fix_locality", "domain_knowledge_required",
        "fix_complexity", "agent_declared_success_wrongly", "agent_looping",
        "agent_expressed_uncertainty", "agent_wrong_file_focus", "agent_gave_up_early"
    ]

    if len(valid_df) > 5:
        # === CORRELATIONS WITH IRT DIFFICULTY ===
        print(f"\n=== CORRELATIONS WITH IRT DIFFICULTY (n={len(valid_df)}) ===")

        correlations = []
        for col in feature_cols:
            if col in valid_df.columns and valid_df[col].notna().sum() > 2:
                # Compute correlation and p-value
                r, p = stats.pearsonr(valid_df["difficulty"], valid_df[col])
                n = valid_df[col].notna().sum()
                correlations.append({
                    "feature": col,
                    "correlation": r,
                    "p_value": p,
                    "n": n,
                })

        corr_df = pd.DataFrame(correlations)
        corr_df = corr_df.sort_values("correlation", key=abs, ascending=False)

        for _, row in corr_df.iterrows():
            sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
            print(f"  {row['feature']:35s}: r = {row['correlation']:+.3f} (p={row['p_value']:.3f}) {sig}")

        # Save correlation summary
        corr_output = output_dir / f"correlations_{timestamp}.csv"
        corr_df.to_csv(corr_output, index=False)
        print(f"\nCorrelation summary saved to: {corr_output}")

    # === DISCRIMINATION ANALYSIS (if multiple agents) ===
    unique_agents = valid_df["agent"].nunique()
    if unique_agents >= 2 and len(valid_df) > 10:
        print(f"\n=== FEATURE DISCRIMINATION ANALYSIS (n_agents={unique_agents}) ===")
        print("(high discrimination = low within-agent variance, high between-agent variance)")

        discrimination_data = []

        for col in feature_cols:
            if col not in valid_df.columns:
                continue

            # Convert to numeric
            valid_df[col] = pd.to_numeric(valid_df[col], errors='coerce')

            # Filter to non-null values
            feature_data = valid_df[["agent", col]].dropna()
            if len(feature_data) < 5:
                continue

            # Within-agent variance (average variance within each agent)
            within_var = feature_data.groupby('agent')[col].var().mean()

            # Between-agent variance (variance of agent means)
            agent_means = feature_data.groupby('agent')[col].mean()
            between_var = agent_means.var()

            # Overall stats
            overall_mean = feature_data[col].mean()
            overall_std = feature_data[col].std()

            # Discrimination ratio: high between / low within = good discriminator
            discrimination = between_var / within_var if within_var > 0 else float('inf')

            discrimination_data.append({
                "feature": col,
                "overall_mean": overall_mean,
                "overall_std": overall_std,
                "within_agent_var": within_var,
                "between_agent_var": between_var,
                "discrimination_ratio": discrimination,
            })

        disc_df = pd.DataFrame(discrimination_data)
        disc_df = disc_df.sort_values("discrimination_ratio", ascending=False)

        for _, row in disc_df.iterrows():
            print(f"  {row['feature']:35s}: ratio = {row['discrimination_ratio']:6.3f} "
                  f"(within={row['within_agent_var']:.3f}, between={row['between_agent_var']:.3f})")

        # Save discrimination analysis
        disc_output = output_dir / f"discrimination_{timestamp}.csv"
        disc_df.to_csv(disc_output, index=False)
        print(f"\nDiscrimination summary saved to: {disc_output}")

    print(f"\nGrading complete. Results saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())

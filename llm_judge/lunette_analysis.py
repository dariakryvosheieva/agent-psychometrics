"""
Lunette-based task difficulty analysis for SWE-bench.

This script uses Lunette's GradingPlan to analyze SWE-bench tasks and extract
features that predict IRT difficulty. It combines:
1. Task-intrinsic features (same as llm_judge.py) - about the problem itself
2. Trajectory-based signals - failure modes that indicate difficulty

The agent trajectory is used as a vehicle to get into Lunette's sandbox,
but the primary focus is grading the task difficulty, not agent performance.

Usage:
    python llm_judge/lunette_analysis.py --run_id <lunette-run-id>
    python llm_judge/lunette_analysis.py --run_id <id> --output_path results.csv
"""

import argparse
import asyncio
import json
from pathlib import Path

import pandas as pd

from lunette import LunetteClient
from lunette.analysis import GradingPlan


# Grading prompt that extracts task features (matching llm_judge.py)
# plus trajectory-based failure mode signals
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


async def grade_run(client: LunetteClient, run_id: str) -> dict:
    """Grade a Lunette run to extract task difficulty features."""

    print(f"Grading run: {run_id}")

    results = await client.investigate(
        run_id=run_id,
        plan=GradingPlan(
            name="task-difficulty",
            prompt=TASK_GRADING_PROMPT,
        ),
        limit=1,
    )

    if not results.results:
        return {"error": "No results returned", "run_id": run_id}

    # Extract the grading data
    result_data = results.results[0].data
    result_data["run_id"] = run_id

    return result_data


async def grade_multiple_runs(run_ids: list[str], output_path: Path) -> pd.DataFrame:
    """Grade multiple Lunette runs and save results."""

    all_results = []

    async with LunetteClient() as client:
        for i, run_id in enumerate(run_ids):
            print(f"\n[{i+1}/{len(run_ids)}] ", end="")

            try:
                result = await grade_run(client, run_id)
                all_results.append(result)

                # Save incrementally
                df = pd.DataFrame(all_results)
                df.to_csv(output_path, index=False)

                print(f"  -> fix_complexity={result.get('fix_complexity')}, "
                      f"domain_knowledge={result.get('domain_knowledge_required')}")

            except Exception as e:
                print(f"  -> Error: {e}")
                all_results.append({"run_id": run_id, "error": str(e)})

    return pd.DataFrame(all_results)


async def list_available_runs(client: LunetteClient) -> list[dict]:
    """List available runs from Lunette API."""
    import httpx

    # Load API key
    config_path = Path.home() / ".lunette" / "config.json"
    with open(config_path) as f:
        api_key = json.load(f)["api_key"]

    async with httpx.AsyncClient(
        base_url="https://lunette.dev/api",
        headers={"X-API-Key": api_key},
        timeout=30
    ) as http_client:
        r = await http_client.get("/runs/")
        runs = r.json()

    # Filter to SWE-bench runs (not investigations)
    swebench_runs = [
        r for r in runs
        if "swe" in r.get("task", "").lower()
        and r.get("source_run_id") is None  # Not an investigation
    ]

    return swebench_runs


async def main():
    parser = argparse.ArgumentParser(
        description='Grade SWE-bench tasks using Lunette for difficulty prediction'
    )
    parser.add_argument('--run_id', type=str, default=None,
                        help='Specific Lunette run ID to grade')
    parser.add_argument('--run_ids', type=str, default=None,
                        help='Comma-separated list of run IDs')
    parser.add_argument('--list_runs', action='store_true',
                        help='List available SWE-bench runs')
    parser.add_argument('--grade_all', action='store_true',
                        help='Grade all available SWE-bench runs')
    parser.add_argument('--output_path', type=str,
                        default='chris_output/lunette/task_features.csv',
                        help='Output path for results')
    args = parser.parse_args()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    async with LunetteClient() as client:

        if args.list_runs:
            runs = await list_available_runs(client)
            print(f"\nAvailable SWE-bench runs ({len(runs)}):")
            for run in runs:
                print(f"  {run['id']}: {run.get('model', 'unknown')} on {run.get('task', 'unknown')}")
            return

        if args.run_id:
            result = await grade_run(client, args.run_id)
            print(f"\nResults:")
            for k, v in result.items():
                if k != "reasoning":
                    print(f"  {k}: {v}")
            print(f"\nReasoning: {result.get('reasoning', 'N/A')}")

            # Save single result
            pd.DataFrame([result]).to_csv(output_path, index=False)
            print(f"\nSaved to {output_path}")

        elif args.run_ids:
            run_ids = [r.strip() for r in args.run_ids.split(",")]
            df = await grade_multiple_runs(run_ids, output_path)
            print(f"\nGraded {len(df)} runs, saved to {output_path}")

        elif args.grade_all:
            runs = await list_available_runs(client)
            run_ids = [r["id"] for r in runs]
            print(f"\nGrading {len(run_ids)} SWE-bench runs...")
            df = await grade_multiple_runs(run_ids, output_path)
            print(f"\nGraded {len(df)} runs, saved to {output_path}")

        else:
            print("Please specify --run_id, --run_ids, --list_runs, or --grade_all")
            print("\nExample usage:")
            print("  python llm_judge/lunette_analysis.py --list_runs")
            print("  python llm_judge/lunette_analysis.py --run_id abc123")
            print("  python llm_judge/lunette_analysis.py --grade_all")


if __name__ == "__main__":
    asyncio.run(main())

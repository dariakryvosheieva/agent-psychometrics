"""Test V3 prompt on sample tasks to check if it better predicts residuals.

This script:
1. Takes a few tasks with large residuals (both positive and negative)
2. Runs the V3 prompt to extract features
3. Analyzes whether the new features could predict the residual direction
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

from experiment_b.llm_judge.features_v3 import (
    RESIDUAL_AWARE_PROMPT,
    LLMJudgeV3Features,
    LLM_JUDGE_V3_FEATURE_NAMES,
)

# Expected correlations with residual:
# - effort_to_solution_ratio: POSITIVE (high effort for small fix = harder than expected)
# - problem_text_accuracy: POSITIVE (misleading text = harder than expected)
# - location_discoverability: POSITIVE (hard to find = harder)
# - solution_path_clarity: POSITIVE (unclear path = harder)


def load_trajectory(task_id: str, agent: str, trajectories_dir: Path) -> Optional[Dict]:
    """Load a trajectory from the unified format."""
    traj_path = trajectories_dir / agent / f"{task_id}.json"
    if not traj_path.exists():
        return None

    with open(traj_path) as f:
        return json.load(f)


def format_trajectory_for_prompt(traj: Dict, max_chars: int = 40000) -> str:
    """Format trajectory messages for the prompt."""
    messages = traj.get("messages", [])
    if not messages:
        return "(No messages in trajectory)"

    lines = []
    total_chars = 0

    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        # Truncate long messages
        if len(content) > 3000:
            content = content[:1500] + "\n...[truncated]...\n" + content[-1500:]

        line = f"[{role.upper()}]: {content}"

        if total_chars + len(line) > max_chars:
            lines.append(f"\n...[{len(messages) - i} more messages truncated]...")
            break

        lines.append(line)
        total_chars += len(line)

    return "\n\n".join(lines)


def call_anthropic(prompt: str, model: str = "claude-sonnet-4-20250514") -> str:
    """Call Anthropic API."""
    if not HAS_ANTHROPIC:
        raise ImportError("anthropic not installed")

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def parse_response(response_text: str) -> Optional[Dict]:
    """Parse LLM response to extract JSON."""
    # Try to find JSON in response
    start = response_text.find("{")
    end = response_text.rfind("}") + 1

    if start == -1 or end == 0:
        return None

    try:
        return json.loads(response_text[start:end])
    except json.JSONDecodeError:
        return None


def extract_v3_features(
    task_id: str,
    agent: str,
    trajectories_dir: Path,
    swebench_data: Dict,
    model: str = "claude-sonnet-4-20250514",
) -> Optional[Dict]:
    """Extract V3 features for a task-agent pair."""
    # Load trajectory
    traj = load_trajectory(task_id, agent, trajectories_dir)
    if traj is None:
        print(f"  No trajectory found for {agent}")
        return None

    # Get SWE-bench metadata
    meta = swebench_data.get(task_id, {})
    if not meta:
        print(f"  No SWE-bench metadata for {task_id}")
        return None

    # Format prompt
    problem = meta.get("problem_statement", "")
    patch = meta.get("patch", "")
    hints = meta.get("hints_text", "")

    prompt = RESIDUAL_AWARE_PROMPT.format(
        instance_id=task_id,
        repo=meta.get("repo", "unknown"),
        version=meta.get("version", "unknown"),
        problem_len=len(problem),
        problem_statement=problem[:8000] if len(problem) > 8000 else problem,
        patch_len=len(patch),
        patch=patch[:6000] if len(patch) > 6000 else patch,
        fail_to_pass=meta.get("FAIL_TO_PASS", "[]"),
        hints_section=f"**Hints:** {hints}" if hints else "",
        trajectory_text=format_trajectory_for_prompt(traj),
        resolved_status="RESOLVED" if traj.get("resolved") else "UNRESOLVED",
    )

    # Call LLM
    response = call_anthropic(prompt, model)

    # Parse response
    features = parse_response(response)
    if features is None:
        print(f"  Failed to parse response")
        return None

    return features


def main():
    parser = argparse.ArgumentParser(description="Test V3 prompt on sample tasks")
    parser.add_argument(
        "--n_tasks",
        type=int,
        default=5,
        help="Number of tasks to test per category",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Model to use",
    )
    args = parser.parse_args()

    # Load SWE-bench data
    print("Loading SWE-bench data...")
    try:
        from datasets import load_dataset
        ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
        swebench_data = {ex["instance_id"]: ex for ex in ds}
        print(f"Loaded {len(swebench_data)} tasks")
    except Exception as e:
        print(f"Failed to load SWE-bench: {e}")
        return

    # Load residual analysis
    analysis_path = ROOT / "chris_output/experiment_b/llm_judge_residual_analysis.json"
    with open(analysis_path) as f:
        analysis = json.load(f)

    train_tasks = analysis["train_tasks"]

    # Get tasks with large residuals
    import pandas as pd
    df = pd.DataFrame(train_tasks)

    # Sort by residual
    large_positive = df[df["actual_residual"] > 1.5].sort_values("actual_residual", ascending=False)
    large_negative = df[df["actual_residual"] < -1.0].sort_values("actual_residual")

    print(f"\nTasks with large POSITIVE residual (prior underestimated): {len(large_positive)}")
    print(f"Tasks with large NEGATIVE residual (prior overestimated): {len(large_negative)}")

    trajectories_dir = ROOT / "trajectory_data/unified_trajs"
    agent = "20240402_sweagent_gpt4"  # Use a common agent

    results = []

    # Test on positive residual tasks (embedding UNDERESTIMATED difficulty)
    print(f"\n{'='*80}")
    print("Testing on tasks where EMBEDDING UNDERESTIMATED difficulty")
    print("(We expect: hidden_complexity > 0, solution_simpler < 0)")
    print("="*80)

    for _, row in large_positive.head(args.n_tasks).iterrows():
        task_id = row["task_id"]
        print(f"\n{task_id}:")
        print(f"  Actual residual: {row['actual_residual']:.3f}")

        features = extract_v3_features(task_id, agent, trajectories_dir, swebench_data, args.model)
        if features:
            print(f"  hidden_complexity: {features.get('trajectory_revealed_hidden_complexity', 'N/A')}")
            print(f"  solution_simpler: {features.get('solution_simpler_than_problem_suggests', 'N/A')}")
            print(f"  api_traps: {features.get('api_interaction_traps', 'N/A')}")
            print(f"  reasoning: {features.get('reasoning', 'N/A')[:150]}...")

            results.append({
                "task_id": task_id,
                "actual_residual": row["actual_residual"],
                "category": "positive",
                **features,
            })

    # Test on negative residual tasks (embedding OVERESTIMATED difficulty)
    print(f"\n{'='*80}")
    print("Testing on tasks where EMBEDDING OVERESTIMATED difficulty")
    print("(We expect: hidden_complexity < 0, solution_simpler > 0)")
    print("="*80)

    for _, row in large_negative.head(args.n_tasks).iterrows():
        task_id = row["task_id"]
        print(f"\n{task_id}:")
        print(f"  Actual residual: {row['actual_residual']:.3f}")

        features = extract_v3_features(task_id, agent, trajectories_dir, swebench_data, args.model)
        if features:
            print(f"  hidden_complexity: {features.get('trajectory_revealed_hidden_complexity', 'N/A')}")
            print(f"  solution_simpler: {features.get('solution_simpler_than_problem_suggests', 'N/A')}")
            print(f"  api_traps: {features.get('api_interaction_traps', 'N/A')}")
            print(f"  reasoning: {features.get('reasoning', 'N/A')[:150]}...")

            results.append({
                "task_id": task_id,
                "actual_residual": row["actual_residual"],
                "category": "negative",
                **features,
            })

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print("="*80)

    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        from scipy import stats

        print("\nCorrelation with actual residual (POSITIVE = feature predicts harder than expected):")
        key_features = [
            "effort_to_solution_ratio",
            "problem_text_accuracy",
            "location_discoverability",
            "solution_path_clarity",
            "api_surprise",
            "error_misdirection",
        ]
        for col in key_features:
            if col in results_df.columns:
                r, p = stats.pearsonr(results_df["actual_residual"], results_df[col])
                expected = "CORRECT" if r > 0 else "WRONG DIRECTION"
                sig = "*" if p < 0.05 else ""
                print(f"  {col:30}: r={r:+.3f}{sig} ({expected})")

        # Check if features distinguish the two categories
        pos = results_df[results_df["category"] == "positive"]
        neg = results_df[results_df["category"] == "negative"]

        print("\nMean features by category (should be higher for positive residual):")
        for col in key_features:
            if col in results_df.columns:
                pos_mean = pos[col].mean() if len(pos) > 0 else None
                neg_mean = neg[col].mean() if len(neg) > 0 else None
                diff = pos_mean - neg_mean if pos_mean and neg_mean else 0
                direction = "✓" if diff > 0 else "✗"
                print(f"  {col}:")
                print(f"    Positive residual (harder): {pos_mean:.3f}")
                print(f"    Negative residual (easier): {neg_mean:.3f}")
                print(f"    Difference: {diff:+.3f} {direction}")

    # Save results
    output_path = ROOT / "chris_output/experiment_b/v3_prompt_test_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

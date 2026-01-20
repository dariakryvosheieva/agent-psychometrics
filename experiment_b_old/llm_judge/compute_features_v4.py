"""Compute V4 LLM judge features for experiment B.

This script extracts V4 features (effort_to_solution_ratio, problem_text_accuracy,
error_misdirection, fix_complexity_vs_description, exploration_efficiency) for
all task-agent combinations needed in experiment B.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

from experiment_b.llm_judge.features_v4 import V4_PROMPT, LLMJudgeV4Features


def load_swebench_metadata() -> Dict[str, dict]:
    """Load SWE-bench Verified metadata from HuggingFace."""
    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    return {
        item["instance_id"]: {
            "repo": item["repo"],
            "problem_statement": item["problem_statement"],
            "patch": item["patch"],
            "hints_text": item.get("hints_text", ""),
            "FAIL_TO_PASS": item.get("FAIL_TO_PASS", []),
        }
        for item in ds
    }


def load_trajectory(task_id: str, agent: str, trajectories_dir: Path) -> Optional[Dict]:
    """Load trajectory from unified format."""
    traj_path = trajectories_dir / agent / f"{task_id}.json"
    if not traj_path.exists():
        return None
    with open(traj_path) as f:
        return json.load(f)


def format_trajectory_for_prompt(traj: Dict, max_chars: int = 40000) -> str:
    """Format trajectory messages for prompt."""
    messages = traj.get("messages", [])
    if not messages:
        return "(No messages in trajectory)"

    lines = []
    total_chars = 0

    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

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
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def parse_response(response_text: str) -> Optional[Dict]:
    """Parse LLM response to extract JSON."""
    start = response_text.find("{")
    end = response_text.rfind("}") + 1

    if start == -1 or end == 0:
        return None

    try:
        return json.loads(response_text[start:end])
    except json.JSONDecodeError:
        return None


def extract_v4_features(
    task_id: str,
    agent: str,
    trajectories_dir: Path,
    swebench_data: Dict,
    model: str = "claude-sonnet-4-20250514",
) -> Optional[Dict]:
    """Extract V4 features for a task-agent pair."""
    traj = load_trajectory(task_id, agent, trajectories_dir)
    if traj is None:
        return None

    meta = swebench_data.get(task_id, {})
    if not meta:
        return None

    problem = meta.get("problem_statement", "")
    patch = meta.get("patch", "")
    hints = meta.get("hints_text", "")

    prompt = V4_PROMPT.format(
        instance_id=task_id,
        repo=meta.get("repo", "unknown"),
        problem_len=len(problem),
        problem_statement=problem[:8000] if len(problem) > 8000 else problem,
        patch_len=len(patch),
        patch=patch[:6000] if len(patch) > 6000 else patch,
        hints_section=f"**Hints:** {hints}" if hints else "",
        trajectory_text=format_trajectory_for_prompt(traj),
        resolved_status="RESOLVED" if traj.get("resolved") else "UNRESOLVED",
    )

    try:
        response = call_anthropic(prompt, model)
        features = parse_response(response)
        if features and "effort_to_solution_ratio" in features:
            features["_model"] = model
            features["_provider"] = "anthropic"
            return features
    except Exception as e:
        print(f"    Error: {e}")

    return None


def main():
    parser = argparse.ArgumentParser(description="Compute V4 LLM judge features")
    parser.add_argument("--dry_run", action="store_true", help="Show what would be computed")
    parser.add_argument("--limit", type=int, default=None, help="Limit tasks per agent")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514", help="Model to use")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="chris_output/experiment_b/llm_judge_v4_features",
        help="Output directory",
    )
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    trajectories_dir = ROOT / "trajectory_data/unified_trajs"

    # Load experiment B results to get task/agent splits
    results_path = ROOT / "chris_output/experiment_b/embedding_llm_judge_fixed/experiment_b_results.json"
    with open(results_path) as f:
        results = json.load(f)

    m1_agents = results["split"]["m1_agents"]
    m2_agents = results["split"]["m2_agents"]
    d_train_tasks = results["split"]["d_train_tasks"]
    d_valid_tasks = results["split"]["d_valid_tasks"]

    # Build list of (agent, task) pairs to compute
    pairs_to_compute = []

    # M1 agents × D_train tasks
    for agent in m1_agents:
        for task_id in d_train_tasks:
            pairs_to_compute.append((agent, task_id, "train"))

    # M2 agents × D_valid tasks
    for agent in m2_agents:
        for task_id in d_valid_tasks:
            pairs_to_compute.append((agent, task_id, "valid"))

    print(f"Total pairs to compute: {len(pairs_to_compute)}")
    print(f"  M1 agents × D_train: {len(m1_agents)} × {len(d_train_tasks)} = {len(m1_agents) * len(d_train_tasks)}")
    print(f"  M2 agents × D_valid: {len(m2_agents)} × {len(d_valid_tasks)} = {len(m2_agents) * len(d_valid_tasks)}")

    if args.dry_run:
        print("\nDry run - would compute features for:")
        for agent, task_id, split in pairs_to_compute[:10]:
            print(f"  {agent} × {task_id} ({split})")
        print(f"  ... and {len(pairs_to_compute) - 10} more")
        return

    # Load SWE-bench metadata
    print("\nLoading SWE-bench metadata...")
    swebench_data = load_swebench_metadata()
    print(f"Loaded {len(swebench_data)} tasks")

    # Filter to pairs that have trajectories and don't already have features
    pairs_filtered = []
    for agent, task_id, split in pairs_to_compute:
        traj_path = trajectories_dir / agent / f"{task_id}.json"
        feat_path = output_dir / agent / f"{task_id}.json"

        if not traj_path.exists():
            continue
        if feat_path.exists():
            continue

        pairs_filtered.append((agent, task_id, split))

    print(f"Pairs to compute (after filtering): {len(pairs_filtered)}")

    if args.limit:
        pairs_filtered = pairs_filtered[:args.limit]
        print(f"Limited to: {len(pairs_filtered)}")

    # Compute features
    computed = 0
    errors = 0

    for i, (agent, task_id, split) in enumerate(pairs_filtered):
        print(f"\n[{i+1}/{len(pairs_filtered)}] {agent} × {task_id}")

        features = extract_v4_features(
            task_id, agent, trajectories_dir, swebench_data, args.model
        )

        if features:
            # Save features
            agent_dir = output_dir / agent
            agent_dir.mkdir(parents=True, exist_ok=True)
            feat_path = agent_dir / f"{task_id}.json"
            with open(feat_path, "w") as f:
                json.dump(features, f, indent=2)
            computed += 1
            print(f"  -> Saved: effort={features.get('effort_to_solution_ratio')}, "
                  f"problem={features.get('problem_text_accuracy')}")
        else:
            errors += 1
            print(f"  -> Failed")

        # Rate limiting
        time.sleep(0.5)

    print(f"\n\nDone! Computed: {computed}, Errors: {errors}")


if __name__ == "__main__":
    main()

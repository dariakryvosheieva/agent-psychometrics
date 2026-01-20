"""Compute V4 features specifically for validation tasks."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import anthropic
except ImportError:
    print("anthropic not installed")
    sys.exit(1)

from experiment_b.llm_judge.features_v4 import V4_PROMPT


def load_swebench_metadata():
    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    return {item["instance_id"]: item for item in ds}


def load_trajectory(task_id, agent, trajectories_dir):
    traj_path = trajectories_dir / agent / f"{task_id}.json"
    if not traj_path.exists():
        return None
    with open(traj_path) as f:
        return json.load(f)


def format_trajectory(traj, max_chars=40000):
    messages = traj.get("messages", [])
    if not messages:
        return "(No messages)"

    lines = []
    total = 0
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if len(content) > 3000:
            content = content[:1500] + "\n...[truncated]...\n" + content[-1500:]
        line = f"[{role.upper()}]: {content}"
        if total + len(line) > max_chars:
            lines.append(f"\n...[{len(messages) - i} more messages truncated]...")
            break
        lines.append(line)
        total += len(line)
    return "\n\n".join(lines)


def call_anthropic(prompt, model="claude-sonnet-4-20250514"):
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def parse_response(text):
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return None


def main():
    # Load results to get M2 agents and D_valid tasks
    with open(ROOT / "chris_output/experiment_b/embedding_llm_judge_v4/experiment_b_results.json") as f:
        results = json.load(f)

    m2_agents = results["split"]["m2_agents"]
    d_valid = results["split"]["d_valid_tasks"]

    print(f"Computing V4 features for validation:")
    print(f"  M2 agents: {len(m2_agents)}")
    print(f"  D_valid tasks: {len(d_valid)}")

    trajectories_dir = ROOT / "trajectory_data/unified_trajs"
    output_dir = ROOT / "chris_output/experiment_b/llm_judge_v4_features"

    print("\nLoading SWE-bench metadata...")
    swebench = load_swebench_metadata()

    # Build pairs - one agent per task for speed
    pairs = []
    for task_id in d_valid:
        # Find first M2 agent with trajectory for this task
        for agent in m2_agents:
            traj_path = trajectories_dir / agent / f"{task_id}.json"
            feat_path = output_dir / agent / f"{task_id}.json"
            if traj_path.exists() and not feat_path.exists():
                pairs.append((agent, task_id))
                break

    print(f"Pairs to compute: {len(pairs)}")

    computed = 0
    errors = 0

    for i, (agent, task_id) in enumerate(pairs):
        print(f"\n[{i+1}/{len(pairs)}] {agent} × {task_id}")

        traj = load_trajectory(task_id, agent, trajectories_dir)
        if not traj:
            print("  -> No trajectory")
            errors += 1
            continue

        meta = swebench.get(task_id, {})
        if not meta:
            print("  -> No metadata")
            errors += 1
            continue

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
            trajectory_text=format_trajectory(traj),
            resolved_status="RESOLVED" if traj.get("resolved") else "UNRESOLVED",
        )

        try:
            response = call_anthropic(prompt)
            features = parse_response(response)
            if features and "effort_to_solution_ratio" in features:
                features["_model"] = "claude-sonnet-4-20250514"
                features["_provider"] = "anthropic"

                agent_dir = output_dir / agent
                agent_dir.mkdir(parents=True, exist_ok=True)
                with open(agent_dir / f"{task_id}.json", "w") as f:
                    json.dump(features, f, indent=2)
                computed += 1
                print(f"  -> Saved: effort={features.get('effort_to_solution_ratio')}")
            else:
                print("  -> Failed to parse")
                errors += 1
        except Exception as e:
            print(f"  -> Error: {e}")
            errors += 1

        time.sleep(0.5)

    print(f"\n\nDone! Computed: {computed}, Errors: {errors}")


if __name__ == "__main__":
    main()

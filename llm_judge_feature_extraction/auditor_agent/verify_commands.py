"""Verify that the auditor agent correctly sees command outputs.

This script:
1. Runs specific commands manually in a SWE-bench Docker container
2. Runs the auditor verification task which asks the agent to run the same commands
3. Compares the results to verify the agent sees correct outputs

Usage:
    # Run verification on a specific instance
    python -m llm_judge_feature_extraction.auditor_agent.verify_commands --instance_id django__django-11099

    # Just run manual commands (no agent)
    python -m llm_judge_feature_extraction.auditor_agent.verify_commands --instance_id django__django-11099 --manual_only
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from llm_judge_feature_extraction.auditor_agent.sandbox_utils import get_swebench_image_name


def run_command_in_docker(instance_id: str, command: str) -> tuple[bool, str, str]:
    """Run a command in a SWE-bench Docker container.

    Args:
        instance_id: SWE-bench instance ID (e.g., django__django-11099)
        command: Bash command to run

    Returns:
        Tuple of (success, stdout, stderr)
    """
    image_name = get_swebench_image_name(instance_id)

    docker_cmd = [
        "docker", "run", "--rm",
        "-w", "/testbed",
        image_name,
        "bash", "-c", command,
    ]

    result = subprocess.run(docker_cmd, capture_output=True, text=True)

    return result.returncode == 0, result.stdout, result.stderr


def run_manual_verification(instance_id: str) -> dict:
    """Run verification commands manually in Docker.

    Returns dict with command outputs that should match agent's output.
    """
    print(f"\n=== Running manual verification for {instance_id} ===")

    results = {}

    # Command 1: List first 5 items in /testbed
    print("  Running: ls /testbed | head -5")
    success, stdout, stderr = run_command_in_docker(
        instance_id, "ls /testbed | head -5"
    )
    items = [line.strip() for line in stdout.strip().split("\n") if line.strip()]
    results["first_five_items"] = items
    print(f"    Result: {items}")

    # Command 2: Count Python files
    print("  Running: find /testbed -name '*.py' | wc -l")
    success, stdout, stderr = run_command_in_docker(
        instance_id, "find /testbed -name '*.py' | wc -l"
    )
    try:
        count = int(stdout.strip())
    except ValueError:
        count = -1
    results["python_file_count"] = count
    print(f"    Result: {count}")

    # Command 3: Count git commits
    print("  Running: cd /testbed && git rev-list --count HEAD")
    success, stdout, stderr = run_command_in_docker(
        instance_id, "cd /testbed && git rev-list --count HEAD"
    )
    try:
        count = int(stdout.strip())
    except ValueError:
        count = -1
    results["git_commit_count"] = count
    print(f"    Result: {count}")

    return results


def run_agent_verification(instance_id: str, model: str = "anthropic/claude-opus-4-6") -> dict | None:
    """Run the auditor verification task and extract results.

    Returns dict with agent's reported values, or None if failed.
    """
    print(f"\n=== Running agent verification for {instance_id} ===")
    print(f"    Model: {model}")

    # Run inspect eval with verification task
    cmd = [
        "inspect", "eval",
        "llm_judge_feature_extraction/auditor_agent/inspect_tasks.py@auditor_verification",
        f"--model={model}",
        f"--sample-id={instance_id}",
        "--log-dir=chris_output/auditor_verification",
    ]

    print(f"    Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=_project_root, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"    ERROR: inspect eval failed")
        print(f"    stderr: {result.stderr}")
        return None

    # Read the log file to extract agent's completion
    from inspect_ai.log import read_eval_log

    log_dir = _project_root / "chris_output" / "auditor_verification"
    log_files = list(log_dir.rglob("*.eval"))

    if not log_files:
        print("    ERROR: No log files found")
        return None

    # Get most recent log
    log_path = sorted(log_files)[-1]
    print(f"    Reading log: {log_path}")

    try:
        log = read_eval_log(str(log_path))
        if not log.samples:
            print("    ERROR: No samples in log")
            return None

        sample = log.samples[0]

        # Look for JSON in assistant messages only (skip system/user/tool messages)
        assistant_text = ""
        if sample.messages:
            for msg in sample.messages:
                if msg.role != "assistant":
                    continue
                if hasattr(msg, 'content'):
                    content = msg.content
                    if isinstance(content, str):
                        assistant_text += content + "\n"
                    elif isinstance(content, list):
                        for item in content:
                            if hasattr(item, 'text'):
                                assistant_text += item.text + "\n"

        # Also check completion
        completion = sample.output.completion if sample.output else ""
        assistant_text += completion

        print(f"    Assistant text to search: {len(assistant_text)} chars")

        # Extract JSON from code blocks in assistant text
        code_match = re.search(r'```json\s*(\{[^`]+\})\s*```', assistant_text, re.DOTALL)
        if code_match:
            try:
                agent_results = json.loads(code_match.group(1))
                print(f"    Extracted JSON: {agent_results}")
                return agent_results
            except json.JSONDecodeError as e:
                print(f"    WARNING: Could not parse JSON from code block: {e}")

        # Fallback: try to find raw JSON object
        # Match a complete JSON object with our expected keys
        json_match = re.search(
            r'\{\s*"first_five_items"\s*:\s*\[[^\]]*\]\s*,\s*"python_file_count"\s*:\s*\d+\s*,\s*"git_commit_count"\s*:\s*\d+\s*\}',
            assistant_text,
            re.DOTALL
        )
        if json_match:
            try:
                agent_results = json.loads(json_match.group())
                print(f"    Extracted JSON: {agent_results}")
                return agent_results
            except json.JSONDecodeError as e:
                print(f"    ERROR: Could not parse JSON: {e}")
                return None

        print("    ERROR: No JSON found in assistant messages")
        print(f"    Text preview: {assistant_text[:1000]}")
        return None

    except Exception as e:
        print(f"    ERROR: Could not read log: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_results(manual: dict, agent: dict | None) -> bool:
    """Compare manual and agent results."""
    print("\n=== Comparison ===")

    if agent is None:
        print("  FAIL: Agent verification failed")
        return False

    all_match = True

    # Compare first_five_items (may have slight differences in ordering)
    manual_items = set(manual.get("first_five_items", []))
    agent_items = set(agent.get("first_five_items", []))
    if manual_items == agent_items:
        print(f"  OK: first_five_items match: {manual_items}")
    else:
        print(f"  WARN: first_five_items differ:")
        print(f"    Manual: {manual_items}")
        print(f"    Agent:  {agent_items}")
        # Don't fail on this - ordering might differ

    # Compare python_file_count
    manual_count = manual.get("python_file_count", -1)
    agent_count = agent.get("python_file_count", -1)
    if manual_count == agent_count:
        print(f"  OK: python_file_count match: {manual_count}")
    else:
        print(f"  FAIL: python_file_count differ:")
        print(f"    Manual: {manual_count}")
        print(f"    Agent:  {agent_count}")
        all_match = False

    # Compare git_commit_count
    manual_count = manual.get("git_commit_count", -1)
    agent_count = agent.get("git_commit_count", -1)
    if manual_count == agent_count:
        print(f"  OK: git_commit_count match: {manual_count}")
    else:
        print(f"  FAIL: git_commit_count differ:")
        print(f"    Manual: {manual_count}")
        print(f"    Agent:  {agent_count}")
        all_match = False

    if all_match:
        print("\n  SUCCESS: Agent correctly sees command outputs!")
    else:
        print("\n  FAILURE: Agent outputs don't match manual verification")

    return all_match


def main():
    parser = argparse.ArgumentParser(
        description="Verify auditor agent sees correct command outputs"
    )
    parser.add_argument(
        "--instance_id",
        type=str,
        default="django__django-11099",
        help="SWE-bench instance ID to verify (default: django__django-11099)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="anthropic/claude-opus-4-6",
        help="Model to use for agent (default: anthropic/claude-opus-4-6)",
    )
    parser.add_argument(
        "--manual_only",
        action="store_true",
        help="Only run manual verification (skip agent)",
    )

    args = parser.parse_args()

    # Run manual verification
    manual_results = run_manual_verification(args.instance_id)

    if args.manual_only:
        print("\n=== Manual verification complete ===")
        print(f"Results: {json.dumps(manual_results, indent=2)}")
        return

    # Run agent verification
    agent_results = run_agent_verification(args.instance_id, args.model)

    # Compare
    success = compare_results(manual_results, agent_results)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

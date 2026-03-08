#!/usr/bin/env python3
"""Validate trajectory files in unified_trajs directory for consistency."""

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

UNIFIED_TRAJS_DIR = Path(__file__).parent / "unified_trajs"


def analyze_message_content(messages: list) -> dict[str, Any]:
    """Analyze message content patterns to detect format variations."""
    analysis = {
        "has_thought_response_pattern": False,
        "has_tool_calls": False,
        "has_code_blocks": False,
        "has_bash_commands": False,
        "has_file_edits": False,
        "content_types": set(),
        "alternation_pattern": [],  # Track role alternation
    }

    for msg in messages:
        if not isinstance(msg, dict):
            continue

        content = msg.get("content", "")
        role = msg.get("role", "")
        analysis["alternation_pattern"].append(role)

        if not isinstance(content, str):
            continue

        # Check for common patterns
        if "THOUGHT" in content and "RESPONSE" in content:
            analysis["has_thought_response_pattern"] = True
            analysis["content_types"].add("thought_response")

        if "```" in content:
            analysis["has_code_blocks"] = True
            if "```python" in content or "```bash" in content:
                analysis["content_types"].add("code_block")

        if re.search(r'\$ |bash|cd |ls |grep |find |cat |echo ', content):
            analysis["has_bash_commands"] = True
            analysis["content_types"].add("bash_command")

        if re.search(r'edit_file|create_file|write_file|open\s+<path>', content, re.IGNORECASE):
            analysis["has_file_edits"] = True
            analysis["content_types"].add("file_edit")

    return analysis


def validate_trajectory(filepath: Path) -> dict[str, Any]:
    """Validate a single trajectory file and return validation results."""
    result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {}
    }

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        result["valid"] = False
        result["errors"].append(f"JSON parse error: {e}")
        return result
    except Exception as e:
        result["valid"] = False
        result["errors"].append(f"File read error: {e}")
        return result

    # Check required top-level fields
    required_fields = ["task_id", "agent", "messages"]
    for field in required_fields:
        if field not in data:
            result["valid"] = False
            result["errors"].append(f"Missing required field: {field}")

    # Check task_id matches filename
    expected_task_id = filepath.stem
    if data.get("task_id") != expected_task_id:
        result["warnings"].append(f"task_id mismatch: file={expected_task_id}, content={data.get('task_id')}")

    # Check messages structure
    messages = data.get("messages", [])
    if not isinstance(messages, list):
        result["valid"] = False
        result["errors"].append(f"messages is not a list: {type(messages)}")
        return result

    result["stats"]["num_messages"] = len(messages)

    if len(messages) == 0:
        result["valid"] = False
        result["errors"].append("Empty messages array")
        return result

    if len(messages) == 1:
        result["warnings"].append("Only 1 message (no back-and-forth)")

    # Count roles
    role_counts = defaultdict(int)
    valid_roles = {"user", "assistant", "system"}

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            result["valid"] = False
            result["errors"].append(f"Message {i} is not a dict: {type(msg)}")
            continue

        role = msg.get("role")
        if role is None:
            result["valid"] = False
            result["errors"].append(f"Message {i} missing 'role' field")
        elif role not in valid_roles:
            result["warnings"].append(f"Message {i} has unexpected role: {role}")
        else:
            role_counts[role] += 1

        content = msg.get("content")
        if content is None:
            result["warnings"].append(f"Message {i} has no 'content' field")
        elif not isinstance(content, str):
            result["warnings"].append(f"Message {i} content is not a string: {type(content)}")

    result["stats"]["role_counts"] = dict(role_counts)

    # Check for back-and-forth pattern (assistant should respond to user/system)
    assistant_count = role_counts.get("assistant", 0)
    user_system_count = role_counts.get("user", 0) + role_counts.get("system", 0)

    if assistant_count == 0:
        result["warnings"].append("No assistant messages found")

    # Check resolved field
    if "resolved" in data:
        if not isinstance(data["resolved"], bool):
            result["warnings"].append(f"'resolved' is not boolean: {type(data['resolved'])}")
        result["stats"]["resolved"] = data.get("resolved")
    else:
        result["warnings"].append("Missing 'resolved' field")

    # Analyze content patterns
    content_analysis = analyze_message_content(messages)
    result["stats"]["content_analysis"] = content_analysis

    # Check for proper back-and-forth alternation
    pattern = content_analysis["alternation_pattern"]
    if len(pattern) > 1:
        # Check if there's at least some alternation
        unique_roles = set(pattern)
        if len(unique_roles) == 1:
            result["warnings"].append(f"All messages have same role: {pattern[0]}")

    return result


def validate_agent_directory(agent_dir: Path) -> dict[str, Any]:
    """Validate all trajectory files in an agent directory."""
    result = {
        "agent": agent_dir.name,
        "valid": True,
        "total_files": 0,
        "valid_files": 0,
        "invalid_files": 0,
        "task_ids": set(),
        "duplicate_tasks": [],
        "errors": [],
        "warnings": [],
        "file_results": {}
    }

    json_files = list(agent_dir.glob("*.json"))
    # Filter out metadata files
    json_files = [f for f in json_files if not f.name.startswith("_")]

    result["total_files"] = len(json_files)

    for filepath in json_files:
        file_result = validate_trajectory(filepath)
        result["file_results"][filepath.name] = file_result

        if file_result["valid"]:
            result["valid_files"] += 1
        else:
            result["invalid_files"] += 1
            result["valid"] = False

        # Track task IDs for duplicate detection
        task_id = filepath.stem
        if task_id in result["task_ids"]:
            result["duplicate_tasks"].append(task_id)
        result["task_ids"].add(task_id)

    return result


def main():
    """Run validation across all agents."""
    print("=" * 80)
    print("TRAJECTORY VALIDATION REPORT")
    print("=" * 80)

    agent_dirs = sorted([
        d for d in UNIFIED_TRAJS_DIR.iterdir()
        if d.is_dir() and not d.name.startswith("_")
    ])

    print(f"\nFound {len(agent_dirs)} agent directories\n")

    all_task_ids = set()
    global_stats = {
        "total_agents": len(agent_dirs),
        "valid_agents": 0,
        "total_trajectories": 0,
        "valid_trajectories": 0,
        "invalid_trajectories": 0,
        "agents_with_errors": [],
        "agents_with_warnings": [],
        "message_count_distribution": defaultdict(int),
        "role_distribution": defaultdict(int),
        "agents_by_task_count": defaultdict(list),
        "format_patterns": defaultdict(list),  # Track which agents use which patterns
        "agents_all_user_messages": [],  # Agents where all messages are from user role
    }

    for agent_dir in agent_dirs:
        print(f"\nValidating: {agent_dir.name}")
        result = validate_agent_directory(agent_dir)

        global_stats["total_trajectories"] += result["total_files"]
        global_stats["valid_trajectories"] += result["valid_files"]
        global_stats["invalid_trajectories"] += result["invalid_files"]
        global_stats["agents_by_task_count"][result["total_files"]].append(agent_dir.name)

        if result["valid"]:
            global_stats["valid_agents"] += 1

        # Collect errors
        errors_found = []
        warnings_found = []

        for fname, fresult in result["file_results"].items():
            if fresult["errors"]:
                errors_found.append((fname, fresult["errors"]))
            if fresult["warnings"]:
                warnings_found.append((fname, fresult["warnings"]))

            # Aggregate message counts
            if "num_messages" in fresult["stats"]:
                global_stats["message_count_distribution"][fresult["stats"]["num_messages"]] += 1

            # Aggregate role counts
            for role, count in fresult["stats"].get("role_counts", {}).items():
                global_stats["role_distribution"][role] += count

            # Track format patterns
            if "content_analysis" in fresult["stats"]:
                analysis = fresult["stats"]["content_analysis"]
                if analysis.get("has_thought_response_pattern"):
                    global_stats["format_patterns"]["thought_response"].append(agent_dir.name)
                if analysis.get("has_code_blocks"):
                    global_stats["format_patterns"]["code_blocks"].append(agent_dir.name)

        # Check if agent has mostly user-only messages (no proper assistant turns)
        agent_roles = result["file_results"]
        user_only_count = sum(
            1 for fr in agent_roles.values()
            if fr["stats"].get("role_counts", {}).get("assistant", 0) == 0
        )
        if user_only_count > result["total_files"] * 0.5:
            global_stats["agents_all_user_messages"].append(
                (agent_dir.name, user_only_count, result["total_files"])
            )

        # Print summary for this agent
        print(f"  Files: {result['total_files']} | Valid: {result['valid_files']} | Invalid: {result['invalid_files']}")

        if errors_found:
            global_stats["agents_with_errors"].append(agent_dir.name)
            print(f"  ERRORS ({len(errors_found)} files):")
            for fname, errs in errors_found[:5]:  # Show first 5
                print(f"    - {fname}: {errs}")
            if len(errors_found) > 5:
                print(f"    ... and {len(errors_found) - 5} more")

        if warnings_found:
            global_stats["agents_with_warnings"].append(agent_dir.name)
            # Only show warning counts, not details
            warning_types = defaultdict(int)
            for fname, warns in warnings_found:
                for w in warns:
                    # Extract warning type
                    if "mismatch" in w:
                        warning_types["task_id mismatch"] += 1
                    elif "Only 1 message" in w:
                        warning_types["single message"] += 1
                    elif "No assistant" in w:
                        warning_types["no assistant messages"] += 1
                    elif "Missing 'resolved'" in w:
                        warning_types["missing resolved"] += 1
                    else:
                        warning_types["other"] += 1
            print(f"  Warnings: {dict(warning_types)}")

        # Track all task IDs
        all_task_ids.update(result["task_ids"])

        if result["duplicate_tasks"]:
            print(f"  DUPLICATE TASKS: {result['duplicate_tasks']}")

    # Print global summary
    print("\n" + "=" * 80)
    print("GLOBAL SUMMARY")
    print("=" * 80)

    print(f"\nAgents: {global_stats['valid_agents']}/{global_stats['total_agents']} fully valid")
    print(f"Trajectories: {global_stats['valid_trajectories']}/{global_stats['total_trajectories']} valid")
    print(f"Unique task IDs across all agents: {len(all_task_ids)}")

    print(f"\nAgents with errors: {len(global_stats['agents_with_errors'])}")
    if global_stats['agents_with_errors']:
        for a in global_stats['agents_with_errors']:
            print(f"  - {a}")

    print(f"\nTask count distribution:")
    for count in sorted(global_stats["agents_by_task_count"].keys()):
        agents = global_stats["agents_by_task_count"][count]
        print(f"  {count} tasks: {len(agents)} agents")
        if count < 450 or count > 505:
            for a in agents:
                print(f"    - {a}")

    print(f"\nMessage count distribution (sample):")
    msg_counts = sorted(global_stats["message_count_distribution"].items())
    # Show min, max, and some middle values
    if msg_counts:
        print(f"  Min messages: {msg_counts[0][0]} ({msg_counts[0][1]} trajectories)")
        print(f"  Max messages: {msg_counts[-1][0]} ({msg_counts[-1][1]} trajectories)")

        # Calculate average
        total_msgs = sum(k * v for k, v in msg_counts)
        total_trajs = sum(v for _, v in msg_counts)
        print(f"  Average messages: {total_msgs / total_trajs:.1f}")

        # Show trajectories with very few messages
        few_messages = [(k, v) for k, v in msg_counts if k <= 3]
        if few_messages:
            print(f"\n  Trajectories with <=3 messages:")
            for count, num in few_messages:
                print(f"    {count} messages: {num} trajectories")

    print(f"\nTotal role distribution:")
    for role, count in sorted(global_stats["role_distribution"].items()):
        print(f"  {role}: {count:,} messages")

    print(f"\n" + "=" * 80)
    print("FORMAT VARIATION ANALYSIS")
    print("=" * 80)

    # Agents with non-standard role patterns
    if global_stats["agents_all_user_messages"]:
        print(f"\nAgents with >50% trajectories having NO assistant messages:")
        print("(These use non-standard formats where agent output is in 'user' role)")
        for agent, count, total in global_stats["agents_all_user_messages"]:
            print(f"  - {agent}: {count}/{total} trajectories")

    # Deduplicate format patterns
    thought_response_agents = set(global_stats["format_patterns"]["thought_response"])
    if thought_response_agents:
        print(f"\nAgents using THOUGHT/RESPONSE pattern in content:")
        for agent in sorted(thought_response_agents):
            print(f"  - {agent}")

    print(f"\n" + "=" * 80)
    print("SEMANTIC CONSISTENCY SUMMARY")
    print("=" * 80)

    print("""
All trajectories share the same core structure:
  - task_id: SWE-bench task identifier
  - agent: Agent name/identifier
  - resolved: Boolean indicating success
  - messages: Array of conversation turns

However, there are semantic variations in how agents represent their actions:

1. STANDARD FORMAT (user/assistant alternation):
   - User messages contain task description or environment output
   - Assistant messages contain agent's reasoning and actions
   - Most agents use this format

2. DEVLO-STYLE FORMAT (all user messages):
   - All content is in 'user' role messages
   - Agent reasoning marked with THOUGHT/RESPONSE sections
   - Environment output interleaved with agent output
   - Used by: devlo, learn_by_interact, and similar agents

3. ARTEMIS-STYLE FORMAT (log-based):
   - Content includes full log entries with timestamps
   - Messages may not follow strict user/assistant alternation
   - Used by: artemis_agent_v2

4. OPENHANDS FORMAT:
   - Standard user/assistant format
   - Often includes JSON-structured content
   - Used by: OpenHands variants

These format variations are acceptable for the unified format as long as:
  ✓ All required fields are present (task_id, agent, messages, resolved)
  ✓ Messages array contains the full conversation/trajectory
  ✓ The trajectory captures the agent's problem-solving process
""")


if __name__ == "__main__":
    main()

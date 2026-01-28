"""Extract trajectory-level features from agent trajectories.

Features extracted:
- assistant_char_count: Total characters in assistant messages
- n_assistant_messages: Number of assistant messages
- total_messages: Total number of messages in trajectory

Filtering applied:
- SWE-bench Verified: 44 agents with complete coverage (500 tasks, 0 broken)
  Excluded: agents with incomplete task coverage or >0 broken trajectories
  (e.g., 20250203_openhands_4x_scaled has 22% broken, 20250519_devlo has 50% broken)
- SWE-bench Pro: 10 agents after excluding 4 with >10% broken trajectories
  (gpt_5, gpt_5_codex, gpt_5_high, gpt_oss - all have 50-90% broken)
  Restricted to 398 tasks common across all 10 remaining agents.
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


# SWE-bench Verified: 44 agents with complete coverage (500 tasks, 0 broken trajectories)
COMPLETE_VERIFIED_AGENTS: Set[str] = {
    "20240612_MASAI_gpt4o",
    "20240620_sweagent_claude3.5sonnet",
    "20240721_amazon-q-developer-agent-20240719-dev",
    "20240820_honeycomb",
    "20241002_lingma-agent_lingma-swe-gpt-72b",
    "20241028_agentless-1.5_gpt4o",
    "20241029_OpenHands-CodeAct-2.1-sonnet-20241022",
    "20241106_navie-2-gpt4o-sonnet",
    "20241108_autocoderover-v2.0-claude-3-5-sonnet-20241022",
    "20241113_nebius-search-open-weight-models-11-24",
    "20241125_marscode-agent-dev",
    "20241202_agentless-1.5_claude-3.5-sonnet-20241022",
    "20241212_epam-ai-run-claude-3-5-sonnet",
    "20250117_wandb_programmer_o1_crosscheck5",
    "20250226_swerl_llama3_70b",
    "20250228_epam-ai-run-claude-3-5-sonnet",
    "20250410_cortexa",
    "20250415_openhands",
    "20250503_patchpilot-v1.1-o4-mini",
    "20250511_sweagent_lm_32b",
    "20250515_Refact_Agent",
    "20250516_cortexa_o3",
    "20250519_trae",
    "20250520_openhands_devstral_small",
    "20250522_sweagent_claude-4-sonnet-20250514",
    "20250524_openhands_claude_4_sonnet",
    "20250527_amazon.nova-premier-v1.0",
    "20250528_patchpilot_Co-PatcheR",
    "20250603_Refact_Agent_claude-4-sonnet",
    "20250611_moatless_claude-4-sonnet-20250514",
    "20250616_Skywork-SWE-32B",
    "20250616_Skywork-SWE-32B+TTS_Bo8",
    "20250627_agentless_MCTS-Refine-7B",
    "20250710_bloop",
    "20250716_openhands_kimi_k2",
    "20250728_zai_glm4-5",
    "20250804_codesweep_sweagent_kimi_k2_instruct",
    "20250804_epam-ai-run-claude-4-sonnet",
    "20250807_openhands_gpt5",
    "20250901_entroPO_R2E_QwenCoder30BA3B",
    "20250901_entroPO_R2E_QwenCoder30BA3B_tts",
    "20250928_trae_doubao_seed_code",
    "20251110_frogboss-32b",
    "20251110_frogmini-14b",
}

# SWE-bench Pro: Agents to exclude (>10% broken trajectories)
EXCLUDED_PRO_AGENTS: Set[str] = {
    "gpt_5___10132025",
    "gpt_5_codex____debug_oct22",
    "gpt_5_high___paper",
    "gpt_oss___paper",
}


def get_common_pro_tasks(trajectory_dir: Path) -> Set[str]:
    """Compute the intersection of tasks across all non-excluded Pro agents.

    Returns 398 tasks that all 10 remaining agents have trajectories for.
    """
    agent_tasks = {}
    for agent_dir in sorted(trajectory_dir.iterdir()):
        if not agent_dir.is_dir() or agent_dir.name.startswith("."):
            continue
        if agent_dir.name in EXCLUDED_PRO_AGENTS:
            continue

        task_ids = set()
        for traj_file in agent_dir.glob("*.json"):
            if traj_file.name.startswith("_"):
                continue
            match = re.match(r"instance_(.+)\.json", traj_file.name)
            if match:
                task_ids.add(match.group(1))

        agent_tasks[agent_dir.name] = task_ids

    # Compute intersection
    if not agent_tasks:
        return set()

    common_tasks = set.intersection(*agent_tasks.values())
    return common_tasks


def count_assistant_chars(messages: List[Dict]) -> Tuple[int, int, int]:
    """Count characters in assistant messages.

    Args:
        messages: List of message dicts with 'role' and 'content' keys

    Returns:
        Tuple of (assistant_char_count, n_assistant_messages, total_messages)
    """
    assistant_chars = 0
    n_assistant = 0
    total = len(messages)

    for msg in messages:
        role = msg.get("role", "")
        if role == "assistant":
            content = msg.get("content", "")
            # Handle potential non-string content
            if content is None:
                continue
            if isinstance(content, list):
                # Some formats store content as list of blocks
                content = "".join(str(item) for item in content)
            elif not isinstance(content, str):
                content = str(content)

            assistant_chars += len(content)
            n_assistant += 1

    return assistant_chars, n_assistant, total


def extract_char_counts_verified(
    trajectory_dir: Path,
    verbose: bool = True,
    filter_complete_agents: bool = True,
) -> pd.DataFrame:
    """Extract assistant character counts from SWE-bench Verified unified trajectories.

    Args:
        trajectory_dir: Path to unified_trajs/ directory
        verbose: Print progress and statistics
        filter_complete_agents: If True, only include agents in COMPLETE_VERIFIED_AGENTS

    Returns:
        DataFrame with columns: agent, task_id, assistant_char_count, n_assistant_messages, total_messages, resolved
    """
    trajectory_dir = Path(trajectory_dir)
    if not trajectory_dir.exists():
        raise FileNotFoundError(f"Trajectory directory not found: {trajectory_dir}")

    results = []
    role_counter = Counter()
    content_type_counter = Counter()
    n_files = 0
    n_skipped = 0
    n_skipped_agent = 0
    n_errors = 0

    # Iterate over agent directories
    agent_dirs = sorted([d for d in trajectory_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])

    if verbose:
        if filter_complete_agents:
            print(f"Processing agents from {trajectory_dir} (filtering to {len(COMPLETE_VERIFIED_AGENTS)} complete agents)")
        else:
            print(f"Processing {len(agent_dirs)} agents from {trajectory_dir}")

    for agent_dir in agent_dirs:
        agent = agent_dir.name

        # Filter to complete agents if requested
        if filter_complete_agents and agent not in COMPLETE_VERIFIED_AGENTS:
            n_skipped_agent += 1
            continue

        # Iterate over trajectory files, skipping special files starting with _
        for traj_file in sorted(agent_dir.glob("*.json")):
            # Skip special files like _upload_summary.json, _lunette_uploads.json
            if traj_file.name.startswith("_"):
                n_skipped += 1
                continue

            n_files += 1
            task_id = traj_file.stem

            try:
                with open(traj_file) as f:
                    data = json.load(f)

                messages = data.get("messages", [])
                resolved = data.get("resolved", None)

                # Track role distribution
                for msg in messages:
                    role_counter[msg.get("role", "MISSING")] += 1
                    content_type_counter[type(msg.get("content")).__name__] += 1

                # Extract features
                char_count, n_assistant, total = count_assistant_chars(messages)

                results.append(
                    {
                        "agent": agent,
                        "task_id": task_id,
                        "assistant_char_count": char_count,
                        "n_assistant_messages": n_assistant,
                        "total_messages": total,
                        "resolved": resolved,
                    }
                )

            except (json.JSONDecodeError, IOError) as e:
                n_errors += 1
                if verbose:
                    print(f"Error loading {traj_file}: {e}")

    df = pd.DataFrame(results)

    if verbose:
        print(f"\n=== SWE-bench Verified Extraction Summary ===")
        print(f"Agents skipped (not in complete set): {n_skipped_agent}")
        print(f"Files processed: {n_files}")
        print(f"Files skipped (special): {n_skipped}")
        print(f"Errors: {n_errors}")
        print(f"Rows in DataFrame: {len(df)}")
        print(f"\nRole distribution: {dict(role_counter)}")
        print(f"Content types: {dict(content_type_counter)}")

        # Sanity checks
        print(f"\n=== Sanity Checks ===")
        zero_count = (df["assistant_char_count"] == 0).sum()
        print(f"Trajectories with 0 assistant chars: {zero_count}")
        if zero_count > 0:
            print("WARNING: Some trajectories have 0 assistant characters!")
            print(df[df["assistant_char_count"] == 0].head())

        print(f"\nChar count stats:")
        print(df["assistant_char_count"].describe())

    return df


def extract_char_counts_pro(
    trajectory_dir: Path,
    verbose: bool = True,
    exclude_broken_agents: bool = True,
    filter_to_common_tasks: bool = True,
) -> pd.DataFrame:
    """Extract assistant character counts from SWE-bench Pro trajectories.

    Args:
        trajectory_dir: Path to swebench_pro/ directory
        verbose: Print progress and statistics
        exclude_broken_agents: If True, exclude agents in EXCLUDED_PRO_AGENTS
        filter_to_common_tasks: If True, only include the 398 tasks common to all 10 agents

    Returns:
        DataFrame with columns: agent, task_id, assistant_char_count, n_assistant_messages, total_messages
    """
    trajectory_dir = Path(trajectory_dir)
    if not trajectory_dir.exists():
        raise FileNotFoundError(f"Trajectory directory not found: {trajectory_dir}")

    # Compute common tasks if filtering
    common_tasks: Optional[Set[str]] = None
    if filter_to_common_tasks:
        common_tasks = get_common_pro_tasks(trajectory_dir)
        if verbose:
            print(f"Filtering to {len(common_tasks)} common tasks across all non-excluded agents")

    results = []
    role_counter = Counter()
    content_type_counter = Counter()
    n_files = 0
    n_skipped = 0
    n_skipped_agent = 0
    n_skipped_task = 0
    n_errors = 0

    # Iterate over agent directories
    agent_dirs = sorted([d for d in trajectory_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])

    if verbose:
        if exclude_broken_agents:
            print(f"Processing agents from {trajectory_dir} (excluding {len(EXCLUDED_PRO_AGENTS)} broken agents)")
        else:
            print(f"Processing {len(agent_dirs)} agents from {trajectory_dir}")

    for agent_dir in agent_dirs:
        agent = agent_dir.name

        # Skip excluded agents
        if exclude_broken_agents and agent in EXCLUDED_PRO_AGENTS:
            n_skipped_agent += 1
            continue

        # Iterate over trajectory files
        for traj_file in sorted(agent_dir.glob("*.json")):
            # Skip special files starting with _
            if traj_file.name.startswith("_"):
                n_skipped += 1
                continue

            # Extract task_id from filename: instance_{task_id}.json
            match = re.match(r"instance_(.+)\.json", traj_file.name)
            if not match:
                n_errors += 1
                if verbose:
                    print(f"Could not parse task_id from filename: {traj_file.name}")
                continue
            task_id = match.group(1)

            # Skip tasks not in common set
            if common_tasks is not None and task_id not in common_tasks:
                n_skipped_task += 1
                continue

            n_files += 1

            try:
                with open(traj_file) as f:
                    data = json.load(f)

                # Navigate to messages: data[0]["transcripts"][0]["messages"]
                messages = data[0]["transcripts"][0]["messages"]

                # Track role distribution
                for msg in messages:
                    role_counter[msg.get("role", "MISSING")] += 1
                    content_type_counter[type(msg.get("content")).__name__] += 1

                # Extract features
                char_count, n_assistant, total = count_assistant_chars(messages)

                results.append(
                    {
                        "agent": agent,
                        "task_id": task_id,
                        "assistant_char_count": char_count,
                        "n_assistant_messages": n_assistant,
                        "total_messages": total,
                    }
                )

            except (json.JSONDecodeError, IOError, IndexError, KeyError, TypeError) as e:
                n_errors += 1
                if verbose:
                    print(f"Error loading {traj_file}: {e}")

    df = pd.DataFrame(results)

    if verbose:
        print(f"\n=== SWE-bench Pro Extraction Summary ===")
        print(f"Agents skipped (broken): {n_skipped_agent}")
        print(f"Tasks skipped (not in common set): {n_skipped_task}")
        print(f"Files processed: {n_files}")
        print(f"Files skipped (special): {n_skipped}")
        print(f"Errors: {n_errors}")
        print(f"Rows in DataFrame: {len(df)}")
        print(f"\nRole distribution: {dict(role_counter)}")
        print(f"Content types: {dict(content_type_counter)}")

        # Sanity checks
        print(f"\n=== Sanity Checks ===")
        zero_count = (df["assistant_char_count"] == 0).sum()
        print(f"Trajectories with 0 assistant chars: {zero_count}")
        if zero_count > 0:
            print("WARNING: Some trajectories have 0 assistant characters!")
            print(df[df["assistant_char_count"] == 0].head())

        print(f"\nChar count stats:")
        print(df["assistant_char_count"].describe())

    return df


def verify_extraction(df: pd.DataFrame, trajectory_dir: Path, n_samples: int = 5, is_pro: bool = False) -> None:
    """Manually verify extraction by checking random samples.

    Args:
        df: DataFrame with extracted features
        trajectory_dir: Path to trajectory directory
        n_samples: Number of random samples to verify
        is_pro: Whether this is SWE-bench Pro format
    """
    print(f"\n=== Manual Verification ({n_samples} samples) ===")

    # Sample random rows
    sample_rows = df.sample(n=min(n_samples, len(df)), random_state=42)

    for _, row in sample_rows.iterrows():
        agent = row["agent"]
        task_id = row["task_id"]
        extracted_count = row["assistant_char_count"]

        # Load trajectory and manually count
        if is_pro:
            traj_file = trajectory_dir / agent / f"instance_{task_id}.json"
        else:
            traj_file = trajectory_dir / agent / f"{task_id}.json"

        with open(traj_file) as f:
            data = json.load(f)

        if is_pro:
            messages = data[0]["transcripts"][0]["messages"]
        else:
            messages = data.get("messages", [])

        manual_count = 0
        for msg in messages:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if content:
                    manual_count += len(content)

        match = "OK" if extracted_count == manual_count else "MISMATCH"
        print(f"  {agent}/{task_id}: extracted={extracted_count}, manual={manual_count} [{match}]")


def main():
    """Main entry point for trajectory feature extraction."""
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "chris_output" / "tensor_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract SWE-bench Verified
    print("=" * 60)
    print("Extracting SWE-bench Verified trajectories")
    print("=" * 60)
    verified_dir = base_dir / "trajectory_data" / "unified_trajs"
    df_verified = extract_char_counts_verified(verified_dir)
    verify_extraction(df_verified, verified_dir, n_samples=5, is_pro=False)

    # Save
    verified_output = output_dir / "swebench_verified_char_counts.csv"
    df_verified.to_csv(verified_output, index=False)
    print(f"\nSaved to {verified_output}")

    # Extract SWE-bench Pro
    print("\n" + "=" * 60)
    print("Extracting SWE-bench Pro trajectories")
    print("=" * 60)
    pro_dir = base_dir / "trajectory_data" / "swebench_pro"
    df_pro = extract_char_counts_pro(pro_dir)
    verify_extraction(df_pro, pro_dir, n_samples=5, is_pro=True)

    # Save
    pro_output = output_dir / "swebench_pro_char_counts.csv"
    df_pro.to_csv(pro_output, index=False)
    print(f"\nSaved to {pro_output}")

    # Final sanity checks
    print("\n" + "=" * 60)
    print("Final Sanity Checks")
    print("=" * 60)

    # Check for zero counts
    verified_zeros = (df_verified["assistant_char_count"] == 0).sum()
    pro_zeros = (df_pro["assistant_char_count"] == 0).sum()

    if verified_zeros > 0 or pro_zeros > 0:
        print(f"WARNING: Found trajectories with 0 assistant chars!")
        print(f"  Verified: {verified_zeros}")
        print(f"  Pro: {pro_zeros}")
    else:
        print("OK: No trajectories with 0 assistant characters")

    # Coverage check
    print(f"\nCoverage:")
    print(f"  Verified: {df_verified['agent'].nunique()} agents, {df_verified['task_id'].nunique()} tasks, {len(df_verified)} total")
    print(f"  Pro: {df_pro['agent'].nunique()} agents, {df_pro['task_id'].nunique()} tasks, {len(df_pro)} total")


if __name__ == "__main__":
    main()

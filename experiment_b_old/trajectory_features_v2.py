"""Extract mechanical/deterministic features from agent trajectories.

This module extracts ONLY features that can be computed deterministically
from trajectory JSON without any semantic understanding:

1. syntax_error_count - Count of syntax errors encountered
2. test_run_count - Number of test commands executed
3. traceback_count - Number of Python tracebacks
4. unique_files_edited - Number of distinct files edited
5. total_commands - Total shell commands executed
6. edit_attempts - Number of edit/write operations

These are "mechanical" features - just counting patterns in text.
Semantic features (error misdirection, bug reproduction quality, etc.)
should use LLM judge v7 instead.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np


# Feature names for the mechanical feature vector
MECHANICAL_FEATURE_NAMES = [
    "syntax_error_count",    # Count of syntax errors
    "test_run_count",        # Number of test executions
    "traceback_count",       # Number of tracebacks
    "unique_files_edited",   # Number of files edited
    "total_commands",        # Total commands run
    "edit_attempts",         # Number of edit operations
]

# Keep old name for backwards compatibility
EXECUTION_FEATURE_NAMES = MECHANICAL_FEATURE_NAMES


@dataclass
class MechanicalFeatures:
    """Mechanical features extracted from a single trajectory."""

    syntax_error_count: int
    test_run_count: int
    traceback_count: int
    unique_files_edited: Set[str]
    total_commands: int
    edit_attempts: int


def parse_gold_patch_files(patch: str) -> Set[str]:
    """Extract file paths from a unified diff patch.

    Args:
        patch: Unified diff string (--- a/file.py, +++ b/file.py format)

    Returns:
        Set of file paths modified by the patch
    """
    files = set()
    # Match both "--- a/path/to/file.py" and "+++ b/path/to/file.py"
    for match in re.finditer(r'^(?:---|\+\+\+) [ab]/(.+)$', patch, re.MULTILINE):
        filepath = match.group(1)
        # Normalize path (remove leading ./)
        if filepath.startswith('./'):
            filepath = filepath[2:]
        files.add(filepath)
    return files


def count_tracebacks(content: str) -> int:
    """Count Python tracebacks in content.

    Args:
        content: Message content

    Returns:
        Number of tracebacks found
    """
    return len(re.findall(r'Traceback \(most recent call last\):', content))


def count_syntax_errors(content: str) -> int:
    """Count syntax errors in content.

    Args:
        content: Full trajectory content

    Returns:
        Count of syntax errors
    """
    patterns = [
        r'SyntaxError:',
        r'IndentationError:',
        r'unexpected indent',
        r'invalid syntax',
        r'Your proposed edit has introduced new syntax error',
    ]
    count = 0
    for pattern in patterns:
        count += len(re.findall(pattern, content, re.IGNORECASE))
    return count


def count_test_runs(content: str) -> int:
    """Count test execution commands.

    Args:
        content: Full trajectory content

    Returns:
        Count of test runs
    """
    patterns = [
        r'pytest\s',
        r'python\s+-m\s+pytest',
        r'python\s+.*test.*\.py',
        r'./manage\.py\s+test',
        r'tox\s',
        r'nosetests',
        r'python\s+-m\s+unittest',
    ]
    count = 0
    for pattern in patterns:
        count += len(re.findall(pattern, content, re.IGNORECASE))
    return count


def detect_edited_files(content: str) -> Set[str]:
    """Extract files that the agent edited.

    Args:
        content: Full trajectory content

    Returns:
        Set of file paths that were edited
    """
    files = set()

    patterns = [
        r'\[File: ([^\]]+\.py)',  # [File: /repo/path/file.py ...]
        r'open\s+([^\s]+\.py)',   # open path/to/file.py
        r'File updated.*?([^\s/]+\.py)',  # File updated ... file.py
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, content, re.IGNORECASE):
            filepath = match.group(1)
            # Just keep the filename for simplicity
            if '/' in filepath:
                filepath = filepath.split('/')[-1]
            files.add(filepath)

    return files


def count_commands(content: str) -> int:
    """Count shell commands executed.

    Args:
        content: Full trajectory content

    Returns:
        Count of commands
    """
    # Look for command patterns in agent messages
    patterns = [
        r'<command>',
        r'\$ ',  # Shell prompt
        r'bash-\$',
    ]
    count = 0
    for pattern in patterns:
        count += len(re.findall(pattern, content))
    return count


def count_edit_attempts(content: str) -> int:
    """Count edit/write operations.

    Args:
        content: Full trajectory content

    Returns:
        Count of edit attempts
    """
    patterns = [
        r'edit\s+\d+:\d+',  # edit 10:20
        r'<edit>',
        r'File updated',
        r'create\s+\w+\.py',  # create file.py
    ]
    count = 0
    for pattern in patterns:
        count += len(re.findall(pattern, content, re.IGNORECASE))
    return count


def extract_mechanical_features(trajectory: dict) -> MechanicalFeatures:
    """Extract mechanical features from a single trajectory.

    Args:
        trajectory: Loaded trajectory JSON

    Returns:
        MechanicalFeatures dataclass
    """
    messages = trajectory.get("messages", [])

    # Concatenate all message content
    full_content = "\n".join(m.get("content", "") for m in messages)

    # Extract user messages (contain command outputs)
    user_content = "\n".join(
        m.get("content", "") for m in messages if m.get("role") == "user"
    )

    return MechanicalFeatures(
        syntax_error_count=count_syntax_errors(full_content),
        test_run_count=count_test_runs(full_content),
        traceback_count=count_tracebacks(user_content),
        unique_files_edited=detect_edited_files(full_content),
        total_commands=count_commands(full_content),
        edit_attempts=count_edit_attempts(full_content),
    )


# Backwards compatibility alias
extract_execution_features = extract_mechanical_features


def aggregate_mechanical_features(
    agent_features: Dict[str, MechanicalFeatures],
) -> np.ndarray:
    """Aggregate mechanical features across agents for a single task.

    Args:
        agent_features: Dict mapping agent -> MechanicalFeatures

    Returns:
        Feature vector of length len(MECHANICAL_FEATURE_NAMES)
    """
    if not agent_features:
        return np.zeros(len(MECHANICAL_FEATURE_NAMES))

    # Compute averages, normalized to roughly 0-1 range
    syntax_errors = np.mean([f.syntax_error_count for f in agent_features.values()])
    test_runs = np.mean([f.test_run_count for f in agent_features.values()])
    tracebacks = np.mean([f.traceback_count for f in agent_features.values()])
    files_edited = np.mean([len(f.unique_files_edited) for f in agent_features.values()])
    commands = np.mean([f.total_commands for f in agent_features.values()])
    edits = np.mean([f.edit_attempts for f in agent_features.values()])

    return np.array([
        min(1.0, syntax_errors / 10),   # Normalize by 10
        min(1.0, test_runs / 20),       # Normalize by 20
        min(1.0, tracebacks / 10),      # Normalize by 10
        min(1.0, files_edited / 10),    # Normalize by 10
        min(1.0, commands / 50),        # Normalize by 50
        min(1.0, edits / 20),           # Normalize by 20
    ])


# Backwards compatibility alias
aggregate_execution_features = aggregate_mechanical_features


# Cross-agent feature names
CROSS_AGENT_FEATURE_NAMES = [
    "edit_location_entropy",      # Shannon entropy of edit locations across agents
    "edit_location_agreement",    # Fraction of agents editing same files
    "syntax_error_std",           # Std dev of syntax errors across agents
    "traceback_std",              # Std dev of tracebacks across agents
    "test_run_std",               # Std dev of test runs across agents
    "resolved_rate",              # Fraction of agents that resolved the task
]


def compute_cross_agent_features(
    agent_features: Dict[str, MechanicalFeatures],
    agent_resolved: Optional[Dict[str, bool]] = None,
) -> Dict[str, float]:
    """Compute features that look across multiple agents on the same task.

    Args:
        agent_features: Dict mapping agent -> MechanicalFeatures
        agent_resolved: Optional dict mapping agent -> whether they resolved

    Returns:
        Dict of cross-agent feature values
    """
    if len(agent_features) < 2:
        return {name: 0.0 for name in CROSS_AGENT_FEATURE_NAMES}

    # Collect edit locations from all agents
    all_edit_locations = []
    for features in agent_features.values():
        all_edit_locations.append(frozenset(features.unique_files_edited))

    # Edit location entropy - how much do agents disagree on where to edit?
    file_counts: Dict[str, int] = {}
    for edit_set in all_edit_locations:
        for f in edit_set:
            file_counts[f] = file_counts.get(f, 0) + 1

    total_edits = sum(file_counts.values())
    if total_edits > 0:
        probs = np.array(list(file_counts.values())) / total_edits
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        # Normalize by log2(n_unique_files) to get 0-1 range
        max_entropy = np.log2(len(file_counts)) if len(file_counts) > 1 else 1
        edit_location_entropy = entropy / max_entropy if max_entropy > 0 else 0
    else:
        edit_location_entropy = 0.0

    # Edit location agreement - what fraction of agents edit the most common file?
    if file_counts:
        max_count = max(file_counts.values())
        edit_location_agreement = max_count / len(agent_features)
    else:
        edit_location_agreement = 0.0

    # Standard deviations of mechanical features
    syntax_errors = [f.syntax_error_count for f in agent_features.values()]
    tracebacks = [f.traceback_count for f in agent_features.values()]
    test_runs = [f.test_run_count for f in agent_features.values()]

    syntax_error_std = np.std(syntax_errors) if len(syntax_errors) > 1 else 0.0
    traceback_std = np.std(tracebacks) if len(tracebacks) > 1 else 0.0
    test_run_std = np.std(test_runs) if len(test_runs) > 1 else 0.0

    # Resolved rate
    if agent_resolved:
        resolved_rate = np.mean([1 if v else 0 for v in agent_resolved.values()])
    else:
        resolved_rate = 0.0

    return {
        "edit_location_entropy": float(edit_location_entropy),
        "edit_location_agreement": float(edit_location_agreement),
        "syntax_error_std": float(syntax_error_std),
        "traceback_std": float(traceback_std),
        "test_run_std": float(test_run_std),
        "resolved_rate": float(resolved_rate),
    }


def load_and_extract_for_task(
    task_id: str,
    agents: List[str],
    trajectories_dir: Path,
) -> Dict[str, MechanicalFeatures]:
    """Load trajectories and extract features for a task.

    Args:
        task_id: Task instance ID
        agents: List of agent names
        trajectories_dir: Base directory for trajectories

    Returns:
        Dict mapping agent -> MechanicalFeatures
    """
    agent_features = {}

    for agent in agents:
        traj_path = trajectories_dir / agent / f"{task_id}.json"
        if not traj_path.exists():
            continue

        try:
            with open(traj_path) as f:
                trajectory = json.load(f)
            features = extract_mechanical_features(trajectory)
            agent_features[agent] = features
        except (json.JSONDecodeError, IOError, KeyError):
            continue

    return agent_features

"""Utilities for sampling binary responses from binomial data.

This module provides functions to convert binomial (k successes out of n trials)
response data to binary (0/1) responses by sampling one outcome per pair.
"""

import json
from pathlib import Path
from typing import Dict

import numpy as np


def sample_binary_from_binomial(
    binomial_responses: Dict[str, Dict[str, Dict[str, int]]],
    seed: int = 0,
) -> Dict[str, Dict[str, int]]:
    """Sample one binary outcome per (agent, task) pair from binomial data.

    For each pair with k successes out of n trials:
    - P(success=1) = k/n
    - P(success=0) = (n-k)/n

    Args:
        binomial_responses: {agent_id: {task_id: {"successes": k, "trials": n}}}
        seed: Random seed for reproducibility

    Returns:
        Binary responses: {agent_id: {task_id: 0 or 1}}
    """
    rng = np.random.default_rng(seed)
    binary_responses = {}

    for agent_id, tasks in binomial_responses.items():
        binary_responses[agent_id] = {}
        for task_id, response in tasks.items():
            k = response["successes"]
            n = response["trials"]

            if n == 0:
                # No trials - treat as failure
                binary_responses[agent_id][task_id] = 0
            else:
                # Sample with P(1) = k/n
                prob_success = k / n
                binary_responses[agent_id][task_id] = int(rng.random() < prob_success)

    return binary_responses


def load_binomial_responses(responses_path: Path) -> Dict[str, Dict[str, Dict[str, int]]]:
    """Load binomial response matrix from JSONL.

    Args:
        responses_path: Path to JSONL file with binomial responses

    Returns:
        {agent_id: {task_id: {"successes": k, "trials": n}}}
    """
    responses = {}
    with open(responses_path) as f:
        for line in f:
            record = json.loads(line)
            agent_id = record["subject_id"]
            responses[agent_id] = record["responses"]
    return responses


def save_binary_responses(
    binary_responses: Dict[str, Dict[str, int]],
    output_path: Path,
) -> None:
    """Save binary responses in JSONL format for IRT training.

    Args:
        binary_responses: {agent_id: {task_id: 0 or 1}}
        output_path: Path to write JSONL
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for agent_id, tasks in binary_responses.items():
            record = {
                "subject_id": agent_id,
                "responses": tasks,
            }
            f.write(json.dumps(record) + "\n")

"""Dataset for SAD-IRT training with trajectory data."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class TrajectoryIRTDataset(Dataset):
    """Dataset for SAD-IRT that combines trajectories with IRT response data.

    Each sample contains:
    - agent_id: Index of the agent
    - task_id: Index of the task
    - input_text: [PROBLEM] + problem_statement + [SOLUTION] + gold_patch + [TRAJECTORY] + trajectory_text
    - response: Binary (0/1) indicating if agent solved the task
    """

    def __init__(
        self,
        response_matrix_path: str,
        trajectory_dir: str,
        tokenizer: AutoTokenizer,
        max_length: int = 8192,
        agent_ids: Optional[List[str]] = None,
        task_ids: Optional[List[str]] = None,
        pairs: Optional[List[Tuple[int, int]]] = None,
        swebench_dataset: str = "princeton-nlp/SWE-bench_Verified",
        use_summaries: bool = True,
    ):
        """Initialize dataset.

        Args:
            response_matrix_path: Path to JSONL file with response matrix
            trajectory_dir: Path to directory with trajectory JSON files
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length (truncate trajectory suffix if needed)
            agent_ids: Optional list of agent IDs to include (for filtering)
            task_ids: Optional list of task IDs to include (for filtering)
            pairs: Optional list of (agent_idx, task_idx) pairs to include
            swebench_dataset: HuggingFace dataset name for problem/solution text
            use_summaries: If True (default), load from trajectory summaries instead of full trajectories
        """
        self.trajectory_dir = Path(trajectory_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_summaries = use_summaries

        # Load SWE-bench data for problem statements and patches
        self.task_data = self._load_swebench_data(swebench_dataset)

        # Load response matrix
        self.response_matrix, self.all_agent_ids, self.all_task_ids = self._load_response_matrix(
            response_matrix_path
        )

        # Filter agents/tasks if specified
        if agent_ids is not None:
            self.agent_ids = [a for a in agent_ids if a in self.all_agent_ids]
        else:
            self.agent_ids = self.all_agent_ids

        if task_ids is not None:
            self.task_ids = [t for t in task_ids if t in self.all_task_ids]
        else:
            self.task_ids = self.all_task_ids

        # Create index mappings
        self.agent_to_idx = {a: i for i, a in enumerate(self.agent_ids)}
        self.task_to_idx = {t: i for i, t in enumerate(self.task_ids)}

        # Build samples (agent_idx, task_idx, response) for pairs with trajectories
        self.samples = self._build_samples(pairs)

        logger.info(
            f"Dataset initialized: {len(self.samples)} samples, "
            f"{len(self.agent_ids)} agents, {len(self.task_ids)} tasks"
        )

    def _load_swebench_data(self, dataset_name: str) -> Dict[str, dict]:
        """Load SWE-bench dataset for problem statements and patches."""
        try:
            from datasets import load_dataset

            ds = load_dataset(dataset_name, split="test")
            return {ex["instance_id"]: ex for ex in ds}
        except Exception as e:
            logger.warning(f"Could not load SWE-bench dataset: {e}")
            return {}

    def _load_response_matrix(
        self, path: str
    ) -> Tuple[Dict[str, Dict[str, int]], List[str], List[str]]:
        """Load response matrix from JSONL file.

        Returns:
            response_matrix: Dict[agent_id, Dict[task_id, response]]
            agent_ids: List of all agent IDs
            task_ids: List of all task IDs
        """
        response_matrix = {}
        all_task_ids = set()

        with open(path, "r") as f:
            for line in f:
                data = json.loads(line)
                agent_id = data["subject_id"]
                responses = data["responses"]
                response_matrix[agent_id] = responses
                all_task_ids.update(responses.keys())

        agent_ids = list(response_matrix.keys())
        task_ids = sorted(all_task_ids)

        return response_matrix, agent_ids, task_ids

    def _build_samples(
        self, pairs: Optional[List[Tuple[int, int]]] = None
    ) -> List[Tuple[int, int, int]]:
        """Build list of (agent_idx, task_idx, response) samples.

        Only includes pairs where trajectory exists.
        """
        samples = []
        missing_trajectories = 0
        total_pairs = 0

        if pairs is not None:
            # Use specified pairs
            for agent_idx, task_idx in pairs:
                agent_id = self.agent_ids[agent_idx]
                task_id = self.task_ids[task_idx]
                total_pairs += 1

                # Check if trajectory exists
                traj_path = self.trajectory_dir / agent_id / f"{task_id}.json"
                if not traj_path.exists():
                    missing_trajectories += 1
                    continue

                # Get response
                if agent_id in self.response_matrix and task_id in self.response_matrix[agent_id]:
                    response = self.response_matrix[agent_id][task_id]
                    samples.append((agent_idx, task_idx, response))
        else:
            # Use all agent-task pairs
            for agent_id in self.agent_ids:
                agent_idx = self.agent_to_idx[agent_id]
                for task_id in self.task_ids:
                    task_idx = self.task_to_idx[task_id]
                    total_pairs += 1

                    # Check if trajectory exists
                    traj_path = self.trajectory_dir / agent_id / f"{task_id}.json"
                    if not traj_path.exists():
                        missing_trajectories += 1
                        continue

                    # Get response
                    if agent_id in self.response_matrix and task_id in self.response_matrix[agent_id]:
                        response = self.response_matrix[agent_id][task_id]
                        samples.append((agent_idx, task_idx, response))

        logger.info(
            f"Built {len(samples)} samples from {total_pairs} pairs. "
            f"Dropped {missing_trajectories} pairs due to missing trajectories "
            f"({missing_trajectories / total_pairs * 100:.1f}%)"
        )

        return samples

    def _load_trajectory(self, agent_id: str, task_id: str) -> str:
        """Load trajectory text from JSON file.

        If use_summaries is True, loads the 'summary' field from summary JSON.
        Otherwise, loads full trajectory messages.
        """
        traj_path = self.trajectory_dir / agent_id / f"{task_id}.json"

        with open(traj_path, "r") as f:
            data = json.load(f)

        if self.use_summaries:
            # Summary format: {"summary": "...", "task_id": "...", "agent": "...", ...}
            return data.get("summary", "")
        else:
            # Full trajectory format: {"messages": [{"role": "...", "content": "..."}, ...]}
            messages = data.get("messages", [])
            trajectory_parts = []
            for msg in messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                trajectory_parts.append(f"[{role.upper()}]\n{content}")
            return "\n\n".join(trajectory_parts)

    def _format_input(self, task_id: str, trajectory_text: str) -> str:
        """Format input text with problem, solution, and trajectory.

        Format: [PROBLEM]\n{problem}\n[SOLUTION]\n{patch}\n[TRAJECTORY]\n{trajectory}
        """
        problem = ""
        patch = ""

        if task_id in self.task_data:
            task = self.task_data[task_id]
            problem = task.get("problem_statement", "")
            patch = task.get("patch", "")

        return f"[PROBLEM]\n{problem}\n\n[SOLUTION]\n{patch}\n\n[TRAJECTORY]\n{trajectory_text}"

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        agent_idx, task_idx, response = self.samples[idx]

        agent_id = self.agent_ids[agent_idx]
        task_id = self.task_ids[task_idx]

        # Load trajectory
        trajectory_text = self._load_trajectory(agent_id, task_id)

        # Format input
        input_text = self._format_input(task_id, trajectory_text)

        # Tokenize with truncation from the left (keep trajectory suffix)
        # First tokenize without truncation to check length
        tokens = self.tokenizer(
            input_text,
            truncation=False,
            return_tensors="pt",
        )

        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)

        # If too long, truncate from the beginning (keep suffix)
        if len(input_ids) > self.max_length:
            input_ids = input_ids[-self.max_length :]
            attention_mask = attention_mask[-self.max_length :]

        # Pad to max_length
        pad_length = self.max_length - len(input_ids)
        if pad_length > 0:
            input_ids = torch.cat(
                [torch.full((pad_length,), self.tokenizer.pad_token_id), input_ids]
            )
            attention_mask = torch.cat([torch.zeros(pad_length), attention_mask])

        return {
            "agent_idx": torch.tensor(agent_idx, dtype=torch.long),
            "task_idx": torch.tensor(task_idx, dtype=torch.long),
            "input_ids": input_ids.long(),
            "attention_mask": attention_mask.long(),
            "response": torch.tensor(response, dtype=torch.float),
        }

    @property
    def num_agents(self) -> int:
        return len(self.agent_ids)

    @property
    def num_tasks(self) -> int:
        return len(self.task_ids)


def create_train_test_split(
    dataset: TrajectoryIRTDataset,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Split dataset into train/test by (agent, task) pairs.

    Both agent and task will appear in both train and test sets,
    but specific (agent, task) pairs are held out.
    """
    rng = np.random.RandomState(seed)

    # Get all pairs
    all_pairs = [(s[0], s[1]) for s in dataset.samples]

    # Shuffle and split
    indices = rng.permutation(len(all_pairs))
    n_test = int(len(all_pairs) * test_fraction)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    train_pairs = [all_pairs[i] for i in train_indices]
    test_pairs = [all_pairs[i] for i in test_indices]

    logger.info(f"Train/test split: {len(train_pairs)} train, {len(test_pairs)} test pairs")

    return train_pairs, test_pairs

"""Base class for dataset configurations in Experiment B."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _parse_date(date_str: str) -> datetime:
    """Parse YYYYMMDD date string to datetime."""
    return datetime.strptime(date_str, "%Y%m%d")


@dataclass
class DatasetConfig(ABC):
    """Abstract base class for dataset-specific configurations.

    Each dataset must implement:
    - get_agent_dates(): Return mapping of agent_id -> date string (YYYYMMDD)
    - name property: Human-readable dataset name

    Attributes:
        responses_path: Path to response matrix JSONL
        oracle_irt_path: Path to oracle IRT items.csv (all agents)
        oracle_abilities_path: Path to oracle IRT abilities.csv (all agents)
        baseline_irt_path: Path to baseline IRT items.csv (pre-frontier only), optional
        embeddings_path: Path to pre-computed embeddings .npz file
        llm_judge_path: Path to LLM judge features CSV
        cutoff_date: Frontier cutoff date in YYYYMMDD format
        output_dir: Directory for output files
    """

    # Core data paths (must be set by subclass)
    responses_path: Path = field(default_factory=Path)
    oracle_irt_path: Path = field(default_factory=Path)
    oracle_abilities_path: Path = field(default_factory=Path)

    # Optional paths
    baseline_irt_path: Optional[Path] = None
    embeddings_path: Optional[Path] = None
    llm_judge_path: Optional[Path] = None

    # Frontier split settings
    cutoff_date: str = ""  # YYYYMMDD format
    pre_threshold: float = 0.1  # Max pass rate for pre-frontier
    post_threshold: float = 0.1  # Min pass rate for post-frontier

    # Anchor task settings (for scale alignment)
    anchor_min_pass_rate: float = 0.10
    anchor_max_pass_rate: float = 0.90

    # IRT-based frontier definition threshold
    # Tasks where no pre-frontier agent has >= this probability are "frontier"
    irt_solve_probability: float = 0.5

    # Output
    output_dir: Path = field(default_factory=Path)

    # Cached data (lazy-loaded, not included in repr)
    _responses: Optional[Dict[str, Dict[str, int]]] = field(default=None, repr=False)
    _all_agents: Optional[List[str]] = field(default=None, repr=False)
    _all_task_ids: Optional[List[str]] = field(default=None, repr=False)
    _agent_dates: Optional[Dict[str, str]] = field(default=None, repr=False)

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable dataset name."""
        ...

    @abstractmethod
    def get_agent_dates(self, agents: List[str]) -> Dict[str, str]:
        """Get date mapping for agents.

        Args:
            agents: List of agent IDs from response matrix

        Returns:
            Dict mapping agent_id -> date string in YYYYMMDD format
            Agents without valid dates should be omitted from the dict.
        """
        ...

    @property
    def llm_judge_feature_cols(self) -> List[str]:
        """Feature columns for LLM judge predictor.

        Override in subclass if the dataset has different feature columns.
        Default returns SWE-bench feature columns.
        """
        return [
            "fix_in_description",
            "problem_clarity",
            "error_message_provided",
            "reproduction_steps",
            "fix_locality",
            "domain_knowledge_required",
            "fix_complexity",
            "logical_reasoning_required",
            "atypicality",
        ]

    def validate_paths(self) -> List[str]:
        """Check which required paths exist.

        Returns:
            List of error messages for missing required files
        """
        errors = []
        required = [
            (self.responses_path, "Response matrix"),
            (self.oracle_irt_path, "Oracle IRT items"),
            (self.oracle_abilities_path, "Oracle IRT abilities"),
        ]
        for path, name in required:
            if not path.exists():
                errors.append(f"{name} not found: {path}")
        return errors

    # =========================================================================
    # Lazy-loaded data properties (cached on first access)
    # =========================================================================

    @property
    def responses(self) -> Dict[str, Dict[str, int]]:
        """Load and cache response matrix.

        Returns:
            Dict mapping agent_id -> task_id -> 0|1
        """
        if self._responses is None:
            # Import here to avoid circular imports
            from experiment_b.shared.evaluation import load_responses_dict
            self._responses = load_responses_dict(self.responses_path)
        return self._responses

    @property
    def all_agents(self) -> List[str]:
        """Get all agent IDs from response matrix (cached)."""
        if self._all_agents is None:
            self._all_agents = list(self.responses.keys())
        return self._all_agents

    @property
    def all_task_ids(self) -> List[str]:
        """Get all task IDs from response matrix (cached).

        Returns union of all tasks across all agents, sorted.
        """
        if self._all_task_ids is None:
            tasks: set[str] = set()
            for agent_responses in self.responses.values():
                tasks.update(agent_responses.keys())
            self._all_task_ids = sorted(tasks)
        return self._all_task_ids

    @property
    def agent_dates(self) -> Dict[str, str]:
        """Get dates for all agents (cached).

        Returns:
            Dict mapping agent_id -> date string (YYYYMMDD)
        """
        if self._agent_dates is None:
            self._agent_dates = self.get_agent_dates(self.all_agents)
        return self._agent_dates

    @property
    def last_agent_date(self) -> Optional[str]:
        """Get the latest agent date as YYYY-MM-DD string."""
        if not self.agent_dates:
            return None
        all_dates = [_parse_date(d) for d in self.agent_dates.values()]
        return max(all_dates).strftime("%Y-%m-%d")

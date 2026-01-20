"""Base class for dataset configurations in Experiment B."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


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

    # Output
    output_dir: Path = field(default_factory=Path)

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

"""Configuration for Experiment A."""

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ExperimentAConfig:
    """Configuration for Experiment A: Prior Validation (IRT AUC).

    Attributes:
        abilities_path: Path to 1PL abilities.csv (agent theta values)
        items_path: Path to 1PL items.csv (ground truth difficulty b)
        responses_path: Path to response matrix JSONL
        output_dir: Directory for output files
        test_fraction: Fraction of tasks to hold out for testing
        split_seed: Random seed for deterministic train/test splits
        embeddings_path: Path to pre-computed embeddings .npz file
        ridge_alpha: Ridge regression regularization parameter
        llm_judge_features_path: Path to LLM judge features CSV file
        llm_judge_ridge_alpha: Ridge alpha for LLM judge predictor
        llm_judge_max_features: Maximum number of features to select (None = no limit)
    """

    # Data paths
    abilities_path: Path = Path("clean_data/swebench_verified_20251120_full/1d_1pl/abilities.csv")
    items_path: Path = Path("clean_data/swebench_verified_20251120_full/1d_1pl/items.csv")
    responses_path: Path = Path("clean_data/swebench_verified/swebench_verified_20251120_full.jsonl")
    output_dir: Path = Path("chris_output/experiment_a")

    # Train/test splitting
    test_fraction: float = 0.2
    split_seed: int = 0

    # Embedding predictor config
    embeddings_path: Optional[Path] = None  # Required for EmbeddingPredictor
    ridge_alphas: List[float] = field(
        default_factory=lambda: [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
    )

    # LLM Judge predictor config
    llm_judge_features_path: Optional[Path] = None  # Required for LLMJudgePredictor
    llm_judge_ridge_alphas: List[float] = field(
        default_factory=lambda: [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    )
    llm_judge_max_features: Optional[int] = None  # None = use all features

    # Task filtering
    exclude_unsolved: bool = False  # Exclude tasks no agent solved

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentAConfig":
        """Create config from dict, converting strings to Paths."""
        path_fields = {
            "abilities_path", "items_path", "responses_path",
            "output_dir", "embeddings_path", "llm_judge_features_path"
        }
        converted = {}
        for k, v in d.items():
            if k in path_fields and isinstance(v, str):
                converted[k] = Path(v) if v else None
            elif k in path_fields and v is None:
                converted[k] = None
            else:
                converted[k] = v
        return cls(**converted)

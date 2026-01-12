"""Configuration for Experiment A."""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional


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
    """

    # Data paths
    abilities_path: Path = Path("clean_data/swebench_verified_20251115_full/1d_1pl/abilities.csv")
    items_path: Path = Path("clean_data/swebench_verified_20251115_full/1d_1pl/items.csv")
    responses_path: Path = Path("chris_output/clean_data/swebench_verified/swebench_verified_20251115_full.jsonl")
    output_dir: Path = Path("chris_output/experiment_a")

    # Train/test splitting
    test_fraction: float = 0.2
    split_seed: int = 0

    # Embedding predictor config
    embeddings_path: Optional[Path] = None  # Required for EmbeddingPredictor
    ridge_alpha: float = 10000.0

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
        path_fields = {"abilities_path", "items_path", "responses_path", "output_dir", "embeddings_path"}
        converted = {}
        for k, v in d.items():
            if k in path_fields and isinstance(v, str):
                converted[k] = Path(v) if v else None
            elif k in path_fields and v is None:
                converted[k] = None
            else:
                converted[k] = v
        return cls(**converted)

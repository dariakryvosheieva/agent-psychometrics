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
        lunette_features_path: Path to Lunette features CSV file
        lunette_ridge_alpha: Ridge alpha for Lunette predictor
        lunette_feature_selection: Feature selection method ("lasso_cv" or "select_k_best")
        lunette_max_features: Maximum number of features to select
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
    ridge_alpha: float = 10000.0

    # Lunette predictor config
    lunette_features_path: Optional[Path] = None  # Required for LunettePredictor
    lunette_ridge_alpha: float = 1.0
    lunette_feature_selection: str = "lasso_cv"  # "lasso_cv" or "select_k_best"
    lunette_max_features: int = 10

    # LLM Judge predictor config
    llm_judge_features_path: Optional[Path] = None  # Required for LLMJudgePredictor
    llm_judge_ridge_alpha: float = 1.0
    llm_judge_max_features: Optional[int] = None  # None = use all 9 features

    # Embedding Similarity predictor config
    embedding_similarity_ridge_alpha: float = 1.0

    # MLE Embedding predictor config (Truong et al. 2025 approach)
    use_mle_embedding: bool = False  # Whether to run MLE embedding predictor
    mle_lr: float = 0.1  # L-BFGS learning rate
    mle_max_iter: int = 100  # Max L-BFGS iterations
    mle_l2_lambda: float = 0.15  # L2 regularization strength (tuned)
    mle_use_mc_abilities: bool = False  # MC marginalization over abilities
    mle_n_mc_samples: int = 100  # Number of MC samples for ability marginalization

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
            "output_dir", "embeddings_path", "lunette_features_path",
            "llm_judge_features_path"
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

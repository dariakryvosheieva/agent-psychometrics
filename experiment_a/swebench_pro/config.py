"""Configuration for Experiment A on SWE-bench Pro."""

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class SWEBenchProConfig:
    """Configuration for Experiment A: Prior Validation (IRT AUC) on SWE-bench Pro.

    Attributes:
        abilities_path: Path to 1PL abilities.csv (agent theta values)
        items_path: Path to 1PL items.csv (ground truth difficulty b)
        responses_path: Path to response matrix JSONL
        output_dir: Directory for output files
        test_fraction: Fraction of tasks to hold out for testing
        split_seed: Random seed for deterministic train/test splits
        embeddings_path: Path to pre-computed embeddings .npz file
        ridge_alphas: Ridge regression regularization parameters
        llm_judge_features_path: Path to LLM judge features CSV file
        llm_judge_max_features: Maximum number of features to select (None = no limit)
    """

    # Data paths
    abilities_path: Path = Path("chris_output/swebench_pro_irt/1d/abilities.csv")
    items_path: Path = Path("chris_output/swebench_pro_irt/1d/items.csv")
    responses_path: Path = Path("out/chris_irt/swebench_pro.jsonl")
    output_dir: Path = Path("chris_output/experiment_a_swebench_pro")

    # Train/test splitting
    test_fraction: float = 0.2
    split_seed: int = 0

    # Embedding predictor config
    embeddings_path: Optional[Path] = Path(
        "embeddings/"
        "embeddings__deepseek-ai__DeepSeek-R1-Distill-Qwen-32B__pool-lasttoken__qs-sol-instr__"
        "__solcap_nocapqs_sol_no_tests_instr_nocap_b7008f2d__idnorm_instance-v1__"
        "ScaleAI_SWE-bench_Pro__test__maxlen8192.npz"
    )

    # Ridge alphas for ALL feature-based predictors (unified, no special casing)
    ridge_alphas: List[float] = field(
        default_factory=lambda: [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
    )

    # LLM Judge predictor config
    # Default: 8 unified LLM features (same across all datasets except SWE-bench Verified)
    llm_judge_features_path: Optional[Path] = Path(
        "chris_output/llm_judge_features/experiment_a_defaults/swebench_pro.csv"
    )
    llm_judge_max_features: Optional[int] = None  # None = use all features

    # Task filtering
    exclude_unsolved: bool = False  # Exclude tasks no agent solved

    # Alpha selection method for grouped ridge
    expand_grouped_ridge: bool = False  # If True, use AUC-based alpha selection

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SWEBenchProConfig":
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

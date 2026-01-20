"""Configuration for Experiment B: Frontier Task Difficulty Prediction."""

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ExperimentBConfig:
    """Configuration for Experiment B: Frontier Task Difficulty Prediction.

    Attributes:
        responses_path: Path to response matrix JSONL
        oracle_irt_path: Path to oracle IRT items.csv (all agents)
        oracle_abilities_path: Path to oracle IRT abilities.csv (all agents)
        baseline_irt_path: Path to baseline IRT items.csv (pre-frontier only)
        embeddings_path: Path to pre-computed embeddings .npz file (any backbone)
        llm_judge_path: Path to LLM judge features CSV file
        cutoff_date: Frontier cutoff date in YYYYMMDD format
        pre_threshold: Max pass rate for pre-frontier (frontier task criterion)
        post_threshold: Min pass rate for post-frontier (frontier task criterion)
        anchor_min_pass_rate: Min pass rate for anchor tasks (scale alignment)
        anchor_max_pass_rate: Max pass rate for anchor tasks (scale alignment)
        alignment_method: Scale alignment method ("constant" or "affine")
        output_dir: Directory for output files
        seed: Random seed for reproducibility
    """

    # Data paths
    responses_path: Path = Path("clean_data/swebench_verified/swebench_verified_20251120_full.jsonl")
    oracle_irt_path: Path = Path("clean_data/swebench_verified_20251120_full/1d/items.csv")
    oracle_abilities_path: Path = Path("clean_data/swebench_verified_20251120_full/1d/abilities.csv")
    baseline_irt_path: Path = Path("chris_output/sad_irt/baseline_irt/items.csv")

    # Feature paths (for predictors)
    embeddings_path: Optional[Path] = Path(
        "out/prior_qwen3vl8b/embeddings__Qwen__Qwen3-VL-8B-Instruct__pool-lasttoken__"
        "qs-sol-instr__qs_sol_instr_b7008f2d__idnorm_instance-v1__"
        "princeton-nlp_SWE-bench_Verified__test__n500__maxlen8192__seed0.npz"
    )
    llm_judge_path: Optional[Path] = Path("chris_output/experiment_a/llm_judge_features/llm_judge_features.csv")

    # SAD-IRT beta values (optional, for comparing with SAD-IRT results)
    sad_irt_beta_dir: Optional[Path] = Path("chris_output/sad_irt_beta_values")

    # Frontier split settings
    cutoff_date: str = "20250807"  # gpt-5-mini release date
    pre_threshold: float = 0.1     # Max pass rate for pre-frontier (10%)
    post_threshold: float = 0.1    # Min pass rate for post-frontier (10%)

    # Anchor task settings (for scale alignment)
    anchor_min_pass_rate: float = 0.10
    anchor_max_pass_rate: float = 0.90

    # Alignment method
    alignment_method: str = "affine"  # "constant" or "affine"

    # Ridge regression settings
    ridge_alphas: List[float] = field(
        default_factory=lambda: [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
    )

    # Output
    output_dir: Path = Path("chris_output/experiment_b")
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentBConfig":
        """Create config from dict, converting strings to Paths."""
        path_fields = {
            "responses_path", "oracle_irt_path", "oracle_abilities_path",
            "baseline_irt_path", "embeddings_path", "llm_judge_path",
            "sad_irt_beta_dir", "output_dir"
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

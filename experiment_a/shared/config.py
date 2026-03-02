"""Unified configuration for Experiment A across all datasets.

All datasets share the same config structure. Per-dataset defaults are stored
in DATASET_DEFAULTS and accessed via ExperimentAConfig.for_dataset().

TerminalBenchConfig extends ExperimentAConfig with binary/binomial mode support.
"""

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# Per-dataset defaults. Used by ExperimentAConfig.for_dataset() and build_spec().
# Each entry contains config field defaults plus experiment spec metadata:
#   - display_name: Human-readable name for output (e.g., "SWE-bench Verified")
#   - is_binomial: Whether responses are binomial (True) or binary (False)
# The irt_cache_dir is derived from output_dir / "irt_splits".
DATASET_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "swebench": {
        "display_name": "SWE-bench Verified",
        "is_binomial": False,
        "abilities_path": Path("clean_data/swebench_verified_20251120_full/1d_1pl/abilities.csv"),
        "items_path": Path("clean_data/swebench_verified_20251120_full/1d_1pl/items.csv"),
        "responses_path": Path("clean_data/swebench_verified/swebench_verified_20251120_full.jsonl"),
        "output_dir": Path("chris_output/experiment_a"),
        "embeddings_path": Path(
            "embeddings/"
            "embeddings__deepseek-ai__DeepSeek-R1-Distill-Qwen-32B__pool-lasttoken__qs-sol-instr__"
            "__solcap_nocapqs_sol_no_tests_instr_nocap_b7008f2d__idnorm_instance-v1__"
            "princeton-nlp_SWE-bench_Verified__test__maxlen8192.npz"
        ),
        "llm_judge_features_path": Path(
            "chris_output/llm_judge_features/experiment_a_defaults/swebench.csv"
        ),
    },
    "gso": {
        "display_name": "GSO",
        "is_binomial": False,
        "abilities_path": Path("chris_output/gso_irt/1d_1pl/abilities.csv"),
        "items_path": Path("chris_output/gso_irt/1d_1pl/items.csv"),
        "responses_path": Path("out/chris_irt/gso.jsonl"),
        "output_dir": Path("chris_output/experiment_a_gso"),
        "embeddings_path": Path(
            "embeddings/"
            "embeddings__deepseek-ai__DeepSeek-R1-Distill-Qwen-32B__pool-lasttoken__qs-sol-instr__"
            "__solcap_nocapqs_sol_no_tests_instr_nocap_b7008f2d__idnorm_instance-v1__"
            "gso-bench_gso__test__maxlen8192.npz"
        ),
        "llm_judge_features_path": Path(
            "chris_output/llm_judge_features/experiment_a_defaults/gso.csv"
        ),
    },
    "swebench_pro": {
        "display_name": "SWE-bench Pro",
        "is_binomial": False,
        "abilities_path": Path("chris_output/swebench_pro_irt/1d/abilities.csv"),
        "items_path": Path("chris_output/swebench_pro_irt/1d/items.csv"),
        "responses_path": Path("out/chris_irt/swebench_pro.jsonl"),
        "output_dir": Path("chris_output/experiment_a_swebench_pro"),
        "embeddings_path": Path(
            "embeddings/"
            "embeddings__deepseek-ai__DeepSeek-R1-Distill-Qwen-32B__pool-lasttoken__qs-sol-instr__"
            "__solcap_nocapqs_sol_no_tests_instr_nocap_b7008f2d__idnorm_instance-v1__"
            "ScaleAI_SWE-bench_Pro__test__maxlen8192.npz"
        ),
        "llm_judge_features_path": Path(
            "chris_output/llm_judge_features/experiment_a_defaults/swebench_pro.csv"
        ),
    },
    "terminalbench": {
        "display_name": "TerminalBench",
        "is_binomial": True,
        "abilities_path": Path("chris_output/terminal_bench_2.0_binomial_1pl/1d/abilities.csv"),
        "items_path": Path("chris_output/terminal_bench_2.0_binomial_1pl/1d/items.csv"),
        "responses_path": Path("data/terminal_bench/terminal_bench_2.0_raw.jsonl"),
        "output_dir": Path("chris_output/experiment_a_terminalbench"),
        "embeddings_path": Path(
            "embeddings/"
            "embeddings__deepseek-ai__DeepSeek-R1-Distill-Qwen-32B__pool-lasttoken__qs-sol-instr__"
            "__solcap_nocapqs_sol_no_tests_instr_nocap_b7008f2d__idnorm_instance-v2__"
            "json_terminal_bench_tasks.jsonl__test__maxlen8192.npz"
        ),
        "llm_judge_features_path": Path(
            "chris_output/llm_judge_features/experiment_a_defaults/terminalbench.csv"
        ),
    },
}

# Fields in DATASET_DEFAULTS that are spec metadata, not config fields
_SPEC_FIELDS = {"display_name", "is_binomial"}

# TerminalBench binary mode paths (collapsed: any success out of 5 = 1)
_TB_BINARY_DEFAULTS = {
    "abilities_path": Path("chris_output/terminal_bench_2.0/1d_1pl/abilities.csv"),
    "items_path": Path("chris_output/terminal_bench_2.0/1d_1pl/items.csv"),
    "responses_path": Path("data/terminal_bench/terminal_bench_2.0.jsonl"),
    "output_dir": Path("chris_output/experiment_a_terminalbench_binary"),
}

_PATH_FIELDS = {
    "abilities_path", "items_path", "responses_path",
    "output_dir", "embeddings_path", "llm_judge_features_path",
}


@dataclass
class ExperimentAConfig:
    """Unified configuration for Experiment A across all datasets.

    Use ExperimentAConfig.for_dataset("swebench") to create a config with
    dataset-specific defaults. Fields can be overridden via constructor kwargs.
    """

    abilities_path: Path = Path("")
    items_path: Path = Path("")
    responses_path: Path = Path("")
    output_dir: Path = Path("")
    test_fraction: float = 0.2
    split_seed: int = 0
    embeddings_path: Optional[Path] = None
    ridge_alphas: List[float] = field(
        default_factory=lambda: [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
    )
    llm_judge_features_path: Optional[Path] = None
    exclude_unsolved: bool = False

    @classmethod
    def for_dataset(cls, dataset: str, **overrides) -> "ExperimentAConfig":
        """Create a config with dataset-specific defaults.

        Args:
            dataset: One of "swebench", "gso", "swebench_pro", "terminalbench".
            **overrides: Override any field value.
        """
        if dataset not in DATASET_DEFAULTS:
            raise ValueError(
                f"Unknown dataset: {dataset}. Valid: {list(DATASET_DEFAULTS.keys())}"
            )
        # Filter out spec metadata fields — only pass config fields
        defaults = {
            k: v for k, v in DATASET_DEFAULTS[dataset].items()
            if k not in _SPEC_FIELDS
        }
        defaults.update(overrides)
        return cls(**defaults)

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
        converted = {}
        for k, v in d.items():
            if k in _PATH_FIELDS and isinstance(v, str):
                converted[k] = Path(v) if v else None
            elif k in _PATH_FIELDS and v is None:
                converted[k] = None
            else:
                converted[k] = v
        return cls(**converted)


def build_spec(dataset: str, root: Path) -> "ExperimentSpec":
    """Build an ExperimentSpec from the dataset registry.

    Args:
        dataset: Dataset short name (e.g., "swebench", "gso").
        root: Project root directory (for resolving irt_cache_dir).

    Returns:
        ExperimentSpec for the dataset.
    """
    from experiment_a.shared.pipeline import ExperimentSpec

    if dataset not in DATASET_DEFAULTS:
        raise ValueError(
            f"Unknown dataset: {dataset}. Valid: {list(DATASET_DEFAULTS.keys())}"
        )
    defaults = DATASET_DEFAULTS[dataset]
    return ExperimentSpec(
        name=defaults["display_name"],
        is_binomial=defaults["is_binomial"],
        irt_cache_dir=root / defaults["output_dir"] / "irt_splits",
    )


@dataclass
class TerminalBenchConfig(ExperimentAConfig):
    """TerminalBench-specific config with binary/binomial mode support.

    Extends ExperimentAConfig with:
    - use_binary: Switch between binomial (default) and binary modes
    - repo_path: Path to cloned terminal-bench repo (for task.yaml + solution.sh)
    """

    use_binary: bool = False
    repo_path: Path = Path("terminal-bench")

    def __post_init__(self):
        """Switch data paths when use_binary=True."""
        if self.use_binary:
            tb_defaults = DATASET_DEFAULTS["terminalbench"]
            for field_name, binary_path in _TB_BINARY_DEFAULTS.items():
                if getattr(self, field_name) == tb_defaults[field_name]:
                    setattr(self, field_name, binary_path)

    @classmethod
    def for_dataset(cls, dataset: str = "terminalbench", **overrides) -> "TerminalBenchConfig":
        """Create a TerminalBench config with defaults."""
        defaults = {
            k: v for k, v in DATASET_DEFAULTS.get(dataset, DATASET_DEFAULTS["terminalbench"]).items()
            if k not in _SPEC_FIELDS
        }
        defaults.update(overrides)
        return cls(**defaults)

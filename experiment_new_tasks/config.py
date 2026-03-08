"""Unified configuration for Experiment A across all datasets.

All datasets share the same config structure. Per-dataset defaults are stored
in DATASET_DEFAULTS and accessed via ExperimentAConfig.for_dataset().
"""

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# Per-dataset defaults. Used by ExperimentAConfig.for_dataset().
# The irt_cache_dir is derived from output_dir / "irt_splits".
DATASET_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "swebench_verified": {
        "display_name": "SWE-bench Verified",
        "abilities_path": Path("data/swebench_verified/irt/1d_1pl/abilities.csv"),
        "items_path": Path("data/swebench_verified/irt/1d_1pl/items.csv"),
        "responses_path": Path("data/swebench_verified/responses.jsonl"),
        "output_dir": Path("chris_output/experiment_a"),
        "embeddings_path": Path(
            "embeddings/"
            "embeddings__deepseek-ai__DeepSeek-R1-Distill-Qwen-32B__pool-lasttoken__qs-sol-instr__"
            "__solcap_nocapqs_sol_no_tests_instr_nocap_b7008f2d__idnorm_instance-v1__"
            "princeton-nlp_SWE-bench_Verified__test__maxlen8192.npz"
        ),
        "llm_judge_features_path": Path(
            "llm_judge_features/defaults/swebench_verified/llm_judge_features.csv"
        ),
        "judge_ablation_paths": {
            "1_problem_15": Path("llm_judge_features/information_ablation/swebench_verified/1_problem_15.csv"),
            "2_problem_auditor_15": Path("llm_judge_features/information_ablation/swebench_verified/2_problem_auditor_15.csv"),
            "3_problem_auditor_test_15": Path("llm_judge_features/information_ablation/swebench_verified/3_problem_auditor_test_15.csv"),
            "4_full_15": Path("llm_judge_features/information_ablation/swebench_verified/4_full_15.csv"),
        },
    },
    "gso": {
        "display_name": "GSO",
        "abilities_path": Path("data/gso/irt/1d_1pl/abilities.csv"),
        "items_path": Path("data/gso/irt/1d_1pl/items.csv"),
        "responses_path": Path("data/gso/responses.jsonl"),
        "output_dir": Path("chris_output/experiment_a_gso"),
        "embeddings_path": Path(
            "embeddings/"
            "embeddings__deepseek-ai__DeepSeek-R1-Distill-Qwen-32B__pool-lasttoken__qs-sol-instr__"
            "__solcap_nocapqs_sol_no_tests_instr_nocap_b7008f2d__idnorm_instance-v1__"
            "gso-bench_gso__test__maxlen8192.npz"
        ),
        "llm_judge_features_path": Path(
            "llm_judge_features/defaults/gso/llm_judge_features.csv"
        ),
        "judge_ablation_paths": {
            "1_problem_15": Path("llm_judge_features/information_ablation/gso/1_problem_15.csv"),
            "2_problem_auditor_15": Path("llm_judge_features/information_ablation/gso/2_problem_auditor_15.csv"),
            "3_problem_auditor_test_15": Path("llm_judge_features/information_ablation/gso/3_problem_auditor_test_15.csv"),
            "4_full_15": Path("llm_judge_features/information_ablation/gso/4_full_15.csv"),
        },
    },
    "swebench_pro": {
        "display_name": "SWE-bench Pro",
        "abilities_path": Path("data/swebench_pro/irt/1d_1pl/abilities.csv"),
        "items_path": Path("data/swebench_pro/irt/1d_1pl/items.csv"),
        "responses_path": Path("data/swebench_pro/responses.jsonl"),
        "output_dir": Path("chris_output/experiment_a_swebench_pro"),
        "embeddings_path": Path(
            "embeddings/"
            "embeddings__deepseek-ai__DeepSeek-R1-Distill-Qwen-32B__pool-lasttoken__qs-sol-instr__"
            "__solcap_nocapqs_sol_no_tests_instr_nocap_b7008f2d__idnorm_instance-v1__"
            "ScaleAI_SWE-bench_Pro__test__maxlen8192.npz"
        ),
        "llm_judge_features_path": Path(
            "llm_judge_features/defaults/swebench_pro/llm_judge_features.csv"
        ),
        "judge_ablation_paths": {
            "1_problem_15": Path("llm_judge_features/information_ablation/swebench_pro/1_problem_15.csv"),
            "2_problem_auditor_15": Path("llm_judge_features/information_ablation/swebench_pro/2_problem_auditor_15.csv"),
            "3_problem_auditor_test_15": Path("llm_judge_features/information_ablation/swebench_pro/3_problem_auditor_test_15.csv"),
            "4_full_15": Path("llm_judge_features/information_ablation/swebench_pro/4_full_15.csv"),
        },
    },
    "terminalbench": {
        "display_name": "TerminalBench",
        "abilities_path": Path("data/terminalbench/irt/1d_1pl/abilities.csv"),
        "items_path": Path("data/terminalbench/irt/1d_1pl/items.csv"),
        "responses_path": Path("data/terminalbench/responses.jsonl"),
        "output_dir": Path("chris_output/experiment_a_terminalbench"),
        "embeddings_path": Path(
            "embeddings/"
            "embeddings__deepseek-ai__DeepSeek-R1-Distill-Qwen-32B__pool-lasttoken__qs-sol-instr__"
            "__solcap_nocapqs_sol_no_tests_instr_nocap_b7008f2d__idnorm_instance-v2__"
            "json_terminal_bench_tasks.jsonl__test__maxlen8192.npz"
        ),
        "llm_judge_features_path": Path(
            "llm_judge_features/defaults/terminalbench/llm_judge_features.csv"
        ),
        "judge_ablation_paths": {
            "1_problem_15": Path("llm_judge_features/information_ablation/terminalbench/1_problem_15.csv"),
            "2_problem_auditor_15": Path("llm_judge_features/information_ablation/terminalbench/2_problem_auditor_15.csv"),
            "3_problem_auditor_test_15": Path("llm_judge_features/information_ablation/terminalbench/3_problem_auditor_test_15.csv"),
            "4_full_15": Path("llm_judge_features/information_ablation/terminalbench/4_full_15.csv"),
        },
    },
}

_PATH_FIELDS = {
    "abilities_path", "items_path", "responses_path",
    "output_dir", "embeddings_path", "llm_judge_features_path",
}

_DICT_OF_PATHS_FIELDS = {"judge_ablation_paths"}


@dataclass
class ExperimentAConfig:
    """Unified configuration for Experiment A across all datasets.

    Use ExperimentAConfig.for_dataset("swebench_verified") to create a config with
    dataset-specific defaults. Fields can be overridden via constructor kwargs.
    """

    display_name: str = ""
    abilities_path: Path = Path("")
    items_path: Path = Path("")
    responses_path: Path = Path("")
    output_dir: Path = Path("")
    split_seed: int = 0
    embeddings_path: Optional[Path] = None
    ridge_alphas: List[float] = field(
        default_factory=lambda: [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
    )
    llm_judge_features_path: Optional[Path] = None
    judge_ablation_paths: Dict[str, Path] = field(default_factory=dict)
    exclude_unsolved: bool = False

    @property
    def irt_cache_dir(self) -> Path:
        """Directory for caching fold-specific IRT models."""
        return self.output_dir / "irt_splits"

    @classmethod
    def for_dataset(cls, dataset: str, **overrides) -> "ExperimentAConfig":
        """Create a config with dataset-specific defaults.

        Args:
            dataset: One of "swebench_verified", "gso", "swebench_pro", "terminalbench".
            **overrides: Override any field value.
        """
        if dataset not in DATASET_DEFAULTS:
            raise ValueError(
                f"Unknown dataset: {dataset}. Valid: {list(DATASET_DEFAULTS.keys())}"
            )
        defaults = dict(DATASET_DEFAULTS[dataset])
        defaults.update(overrides)
        return cls(**defaults)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
            elif k in _DICT_OF_PATHS_FIELDS and isinstance(v, dict):
                d[k] = {dk: str(dv) for dk, dv in v.items()}
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
            elif k in _DICT_OF_PATHS_FIELDS and isinstance(v, dict):
                converted[k] = {dk: Path(dv) for dk, dv in v.items()}
            else:
                converted[k] = v
        return cls(**converted)



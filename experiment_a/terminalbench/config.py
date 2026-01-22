"""Configuration for Experiment A on TerminalBench."""

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Optional


# Binary mode paths (collapsed: any success out of 5 = 1)
_BINARY_ABILITIES_PATH = Path("chris_output/terminal_bench_2.0/1d_1pl/abilities.csv")
_BINARY_ITEMS_PATH = Path("chris_output/terminal_bench_2.0/1d_1pl/items.csv")
_BINARY_RESPONSES_PATH = Path("data/terminal_bench/terminal_bench_2.0.jsonl")

# Binomial mode paths (default: k successes out of n trials)
_BINOMIAL_ABILITIES_PATH = Path("chris_output/terminal_bench_2.0_binomial_1pl/1d/abilities.csv")
_BINOMIAL_ITEMS_PATH = Path("chris_output/terminal_bench_2.0_binomial_1pl/1d/items.csv")
_BINOMIAL_RESPONSES_PATH = Path("data/terminal_bench/terminal_bench_2.0_raw.jsonl")


@dataclass
class TerminalBenchConfig:
    """Configuration for Experiment A on TerminalBench.

    Supports two modes controlled by use_binary:
    - use_binary=True (default): Collapsed binary data where any success = 1
    - use_binary=False: Binomial data with k successes out of n trials

    Attributes:
        use_binary: If True (default), use collapsed binary data (any success = 1).
                   If False, use binomial data (k/n successes).
        abilities_path: Path to 1PL abilities.csv (agent theta values)
        items_path: Path to 1PL items.csv (ground truth difficulty b)
        responses_path: Path to response matrix JSONL
        repo_path: Path to cloned terminal-bench repo (for task.yaml + solution.sh)
        output_dir: Directory for output files
        test_fraction: Fraction of tasks to hold out for testing
        split_seed: Random seed for deterministic train/test splits
        embeddings_path: Path to pre-computed embeddings .npz file
        ridge_alpha: Ridge regression regularization parameter
        llm_judge_features_path: Path to LLM judge features CSV file
        llm_judge_ridge_alpha: Ridge alpha for LLM judge predictor
    """

    # Binary vs binomial mode
    # Binary (default): Collapsed any success out of 5 = 1
    # Binomial: Full k/n successes information
    use_binary: bool = True

    # Data paths - defaults are for binary mode (the default)
    # __post_init__ switches to binomial paths when use_binary=False
    abilities_path: Path = field(default_factory=lambda: _BINARY_ABILITIES_PATH)
    items_path: Path = field(default_factory=lambda: _BINARY_ITEMS_PATH)
    responses_path: Path = field(default_factory=lambda: _BINARY_RESPONSES_PATH)
    repo_path: Path = Path("terminal-bench")  # Cloned terminal-bench repo
    output_dir: Path = Path("chris_output/experiment_a_terminalbench_binary")

    # Train/test splitting
    test_fraction: float = 0.2
    split_seed: int = 0

    # Embedding predictor config
    embeddings_path: Optional[Path] = Path(
        "chris_output/experiment_a_terminalbench/embeddings/"
        "embeddings__deepseek-ai__DeepSeek-R1-Distill-Qwen-32B__pool-lasttoken__maxlen8192.npz"
    )

    # Ridge alphas for ALL feature-based predictors (unified, no special casing)
    ridge_alphas: tuple = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0)

    # LLM Judge predictor config
    llm_judge_features_path: Optional[Path] = Path(
        "chris_output/experiment_a_terminalbench/llm_judge_features/llm_judge_features.csv"
    )
    llm_judge_max_features: Optional[int] = None  # None = use all features

    # Task filtering
    exclude_unsolved: bool = False  # Exclude tasks no agent solved

    def __post_init__(self):
        """Switch data paths based on use_binary mode.

        Default paths are for binary mode. When use_binary=False,
        switch them to the binomial data paths.
        """
        if not self.use_binary:
            # Switch from binary (default) to binomial paths
            # Only switch if paths are at default binary values (allows CLI overrides)
            if self.abilities_path == _BINARY_ABILITIES_PATH:
                self.abilities_path = _BINOMIAL_ABILITIES_PATH
            if self.items_path == _BINARY_ITEMS_PATH:
                self.items_path = _BINOMIAL_ITEMS_PATH
            if self.responses_path == _BINARY_RESPONSES_PATH:
                self.responses_path = _BINOMIAL_RESPONSES_PATH
            # Also update output_dir to keep results separate
            if self.output_dir == Path("chris_output/experiment_a_terminalbench_binary"):
                self.output_dir = Path("chris_output/experiment_a_terminalbench")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
            elif isinstance(v, tuple):
                d[k] = list(v)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TerminalBenchConfig":
        """Create config from dict, converting strings to Paths."""
        path_fields = {
            "abilities_path", "items_path", "responses_path",
            "repo_path", "output_dir", "embeddings_path",
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

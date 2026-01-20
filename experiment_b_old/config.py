"""Configuration for Experiment B."""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

# Regression mode for posterior model
# - "residual": posterior = prior + psi * traj_features (current behavior)
# - "direct_with_prior": posterior = model(traj_features, prior_pred)
# - "direct_with_prior_features": posterior = model(traj_features, prior_input_features)
RegressionMode = Literal["residual", "direct_with_prior", "direct_with_prior_features"]


@dataclass
class ExperimentConfig:
    """Configuration for Experiment B."""

    # Data paths (use 1PL model for consistency with evaluation formula)
    items_path: Path = Path("clean_data/swebench_verified_20251120_full/1d_1pl/items.csv")
    responses_path: Path = Path("clean_data/swebench_verified/swebench_verified_20251120_full.jsonl")
    trajectories_dir: Path = Path("trajectory_data/unified_trajs")
    lunette_features_dir: Path = Path("chris_output/experiment_b/lunette_features")
    llm_judge_features_dir: Path = Path("chris_output/experiment_b/llm_judge_features")
    llm_judge_v4_features_dir: Path = Path("chris_output/experiment_b/llm_judge_v4_features")
    llm_judge_v5_features_dir: Path = Path("chris_output/experiment_b/llm_judge_v5_features")
    llm_judge_v5_single_features_dir: Path = Path("chris_output/experiment_b/llm_judge_v5_single_features")
    execution_features_dir: Path = Path("chris_output/experiment_b/execution_features")
    llm_judge_v6_features_dir: Path = Path("chris_output/experiment_b/llm_judge_v6_features")
    llm_judge_v7_features_dir: Path = Path("chris_output/experiment_b/llm_judge_v7_features")
    trajectory_embeddings_dir: Path = Path("chris_output/experiment_b/trajectory_embeddings")
    test_progression_features_dir: Path = Path("chris_output/experiment_b/test_progression_features")
    critic_features_dir: Path = Path("chris_output/experiment_b/critic_rewards")
    output_dir: Path = Path("chris_output/experiment_b")

    # Agent splitting
    m1_fraction: float = 0.4  # Oldest 40%
    m2_fraction: float = 0.4  # Middle 40%
    # M3 = remaining 20%

    # Task selection
    weak_threshold: float = 0.2  # Max pass rate for "hard" tasks
    strong_min_improvement: float = 0.1  # Min improvement for strong group

    # Model parameters
    # NOTE: For embedding prior, alpha=10000 gives r≈0.63 on held-out test (proper regularization)
    #       alpha=1 causes overfitting (r=0.9999 on train, memorizes data)
    prior_alpha: float = 10000.0  # Ridge alpha for embedding prior
    posterior_alpha: Union[float, str] = "cv"  # Ridge alpha for psi ("cv" for cross-validation)

    # Feature source options:
    # - "simple": Basic message stats (count, chars, resolved_rate)
    # - "lunette": Lunette API features
    # - "llm_judge": Direct LLM API (v1)
    # - "llm_judge_v4": V4 LLM features
    # - "llm_judge_v5": V5 LLM features (4 features)
    # - "llm_judge_v5_single": Single location_vs_fix_alignment feature
    # - "execution": Deterministic execution features (v2) - error misdirection, edit entropy, etc.
    # - "discoverability": LLM judge v6 solution discoverability
    # - "combined_v2": execution + discoverability combined
    # - "test_progression": Test pass rate progression features
    # - "critic_model": OpenHands critic model per-step reward features
    feature_source: Literal["simple", "lunette", "llm_judge", "llm_judge_v4", "llm_judge_v5", "llm_judge_v5_single", "execution", "discoverability", "combined_v2", "llm_judge_v7", "mechanical_v7", "embedding", "test_progression", "critic_model"] = "simple"

    # Embedding posterior configuration (when feature_source="embedding")
    # Content type: how much trajectory information to include
    embedding_content_type: Literal["full", "condensed", "failure_focused", "no_solution"] = "full"
    # Instruction type: what question to ask at end of input
    embedding_instruction_type: Literal["difficulty", "residual", "progress", "closeness"] = "difficulty"
    # Aggregation: how to combine embeddings across agents
    embedding_aggregation: Literal["mean_only", "mean_std", "weighted", "all_stats"] = "mean_std"

    # Prior source: "heuristic" (repo, text length) or "embedding" (Daria's embeddings)
    prior_source: Literal["heuristic", "embedding"] = "heuristic"

    # Path to embeddings file (required if prior_source="embedding")
    embeddings_path: Optional[Path] = None

    # Prior-only mode (no trajectory correction)
    prior_only: bool = False

    # Regression mode for posterior model
    regression_mode: RegressionMode = "residual"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        """Create config from dict, converting strings to Paths."""
        path_fields = {"items_path", "responses_path", "trajectories_dir", "lunette_features_dir", "llm_judge_features_dir", "llm_judge_v4_features_dir", "llm_judge_v5_features_dir", "llm_judge_v5_single_features_dir", "execution_features_dir", "llm_judge_v6_features_dir", "llm_judge_v7_features_dir", "trajectory_embeddings_dir", "test_progression_features_dir", "critic_features_dir", "output_dir", "embeddings_path"}
        converted = {}
        for k, v in d.items():
            if k in path_fields and isinstance(v, str):
                converted[k] = Path(v)
            else:
                converted[k] = v
        return cls(**converted)

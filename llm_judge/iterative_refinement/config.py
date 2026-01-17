"""Configuration for iterative prompt refinement."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class IterativeRefinementConfig:
    """Configuration for iterative prompt refinement.

    Attributes:
        model: LLM model to use (default: gpt-5.2 for cost efficiency + auto caching)
        max_iterations: Fixed number of refinement iterations
        quick_eval_tasks: Number of stratified tasks for quick evaluation
        high_residual_tasks: Number of high-residual tasks to analyze per iteration
        correlation_threshold: Optional early stopping if r exceeds this
        entropy_threshold: Minimum entropy for a feature to be considered useful
        redundancy_threshold: Max correlation between features before flagging redundant
        output_dir: Directory for results and prompt versions
        items_path: Path to IRT items.csv with ground truth difficulties
        api_delay: Delay between API calls in seconds
    """

    # LLM settings - GPT-5.2 for cost efficiency and automatic caching
    model: str = "gpt-5.2"
    api_delay: float = 0.1  # Faster since we're batching

    # Iteration parameters
    max_iterations: int = 5
    quick_eval_tasks: int = 30
    high_residual_tasks: int = 20

    # Quality thresholds
    correlation_threshold: Optional[float] = None  # No early stopping by default
    entropy_threshold: float = 1.0  # Min entropy for useful feature
    redundancy_threshold: float = 0.9  # Max inter-feature correlation

    # Stratification for quick eval (tasks per difficulty tercile)
    tasks_per_tercile: int = 10  # 10 easy + 10 medium + 10 hard = 30

    # Paths
    output_dir: Path = field(
        default_factory=lambda: Path("chris_output/iterative_refinement")
    )
    items_path: Path = field(
        default_factory=lambda: Path(
            "clean_data/swebench_verified_20251120_full/1d/items.csv"
        )
    )

    # Initial feature definitions (from experiment_a/llm_judge_prompt.py)
    initial_features: List[str] = field(
        default_factory=lambda: [
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
    )

    def __post_init__(self):
        """Convert string paths to Path objects and create directories."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.items_path, str):
            self.items_path = Path(self.items_path)

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "prompt_versions").mkdir(exist_ok=True)
        (self.output_dir / "evaluations").mkdir(exist_ok=True)

    @property
    def prompt_versions_dir(self) -> Path:
        """Directory for storing prompt version JSON files."""
        return self.output_dir / "prompt_versions"

    @property
    def evaluations_dir(self) -> Path:
        """Directory for storing evaluation results."""
        return self.output_dir / "evaluations"

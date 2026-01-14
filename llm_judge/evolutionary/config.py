"""Configuration for evolutionary feature discovery."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class EvolutionConfig:
    """Configuration for evolutionary feature discovery."""

    # LLM settings
    # Default to Sonnet for development/testing; use Opus 4.5 for full runs
    model: str = "claude-sonnet-4-20250514"
    api_delay: float = 0.5
    max_retries: int = 3

    # Evolution parameters (conservative defaults for testing)
    initial_features: int = 10
    top_k: int = 5
    max_generations: int = 5
    plateau_threshold: float = 0.01
    plateau_patience: int = 3

    # Evaluation settings (conservative defaults for testing)
    tasks_per_eval: int = 50
    difficulty_percentile: int = 20  # Top/bottom percentile for extremes

    # Selection settings
    redundancy_threshold: float = 0.8  # Inter-feature correlation threshold
    diversity_threshold: float = 0.85  # Embedding cosine similarity threshold

    # Mutation operator weights (PromptBreeder-inspired)
    mutation_weights: dict = field(default_factory=lambda: {
        "direct_mutation": 0.35,     # Mutate feature prompt directly
        "eda_mutation": 0.25,        # Estimation of Distribution (crossover)
        "hypermutation": 0.25,       # Mutate the mutation prompt itself
        "zero_order": 0.15,          # Generate from scratch with context
    })

    # Thinking styles for diverse initialization (from PromptBreeder)
    thinking_styles: List[str] = field(default_factory=lambda: [
        "Analyze step by step.",
        "Consider edge cases first.",
        "Think about what distinguishes experts from novices.",
        "Focus on structural complexity.",
        "Consider the information available to the solver.",
        "Think about prerequisite knowledge.",
        "Consider common failure modes.",
        "Think about cognitive load.",
    ])

    # Paths
    output_dir: Path = field(default_factory=lambda: Path("llm_judge/evolutionary_results"))
    items_path: Path = field(default_factory=lambda: Path(
        "clean_data/swebench_verified_20250930_full/1d/items.csv"
    ))

    def __post_init__(self):
        """Convert string paths to Path objects."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.items_path, str):
            self.items_path = Path(self.items_path)

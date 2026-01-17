"""Configuration for SAD-IRT experiment."""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class SADIRTConfig:
    """Configuration for SAD-IRT training and evaluation."""

    # Mode
    mode: str = "full_auc"  # "full_auc" (Part 2) or "calibration" (Part 1)

    # Model
    model_name: str = "Qwen/Qwen3-0.6B"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: list = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Data
    response_matrix_path: str = "clean_data/swebench_verified/swebench_verified_20251120_full.jsonl"
    trajectory_dir: str = "chris_output/trajectory_summaries_api"
    swebench_dataset: str = "princeton-nlp/SWE-bench_Verified"
    max_length: int = 1024  # Summary-only input; avg ~445 tokens, max 771
    test_fraction: float = 0.2
    hard_threshold: float = 0.2  # Tasks with pass rate <= this are "hard"

    # Training
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    epochs: int = 3
    learning_rate_encoder: float = 1e-4
    learning_rate_embeddings: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # Evaluation
    eval_steps: int = 100
    save_steps: int = 500
    logging_steps: int = 10

    # Output
    output_dir: str = "chris_output/sad_irt"
    seed: int = 42

    # Ablations
    freeze_irt: bool = False  # If True, freeze θ/β and only train ψ predictor
    psi_normalization: Optional[str] = None  # "batchnorm", "center", or "none" (None = auto)

    # Debug
    dry_run: bool = False
    max_samples: Optional[int] = None  # Limit samples for testing
    smoke_test: bool = False  # Just check code paths, no real training
    overfit_test: bool = False  # Test overfitting on small batch
    debug_gradients: bool = False  # Enable verbose gradient logging

    # Resumption
    resume_from: Optional[str] = None  # Path to checkpoint to resume from

    def __post_init__(self):
        """Validate configuration."""
        assert self.mode in ["full_auc", "calibration"], f"Invalid mode: {self.mode}"
        assert 0 < self.test_fraction < 1, "test_fraction must be between 0 and 1"
        assert 0 < self.hard_threshold <= 1, "hard_threshold must be between 0 and 1"

    @property
    def effective_batch_size(self) -> int:
        """Return effective batch size after gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps

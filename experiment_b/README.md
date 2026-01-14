# Experiment B: Posterior Difficulty Prediction

Uses failing agent trajectories to improve IRT difficulty prediction for hard tasks.

## Overview

**Goal**: Prove that failure trajectories from "weak" models contain signal about the solvability for "stronger" models.

**Key Insight**: Trajectory features from weak models (that fail on a task) provide signal about task difficulty beyond what can be inferred from static task features alone.

This corresponds to **Section 3.2** in the [research proposal](../chris%20proposal.md).

## Model Architecture

```
posterior_difficulty_i = prior(x_i) + psi^T * avg_j[f(tau_ij)]

Loss = MSE(predicted, ground_truth_b) + lambda * ||psi||^2
```

Where:
- `prior(x_i)` = linear model on task features (from Experiment A)
- `psi` = single set of learned weights (same for all trajectories)
- `f(tau_ij)` = feature vector from trajectory of agent j on task i
- `avg_j` = average across all weak model trajectories for task i

## Quick Start

```bash
source .venv/bin/activate

# Run with defaults (simple trajectory features)
python -m experiment_b.train_evaluate

# Run with LLM judge features
python -m experiment_b.train_evaluate --feature_source llm_judge

# Adjust pass rate threshold
python -m experiment_b.train_evaluate --weak_threshold 0.1

# Dry run (show config without running)
python -m experiment_b.train_evaluate --dry_run
```

## Data Splitting Strategy

### Agent Splitting (by submission date)

- Parse YYYYMMDD prefix from agent names
- **M1 (oldest 40%)**: Used for training posterior on D_train
- **M2 (middle 40%)**: Used for evaluating posterior on D_valid
- **M3 (newest 20%)**: Held out for future testing

### Task Splitting (by empirical pass rate)

- **D_train**: Tasks with ≤20% pass rate among M1, but >30% among M2 (tasks that got easier)
- **D_valid**: Tasks with ≤20% pass rate among M2, but >30% among M3 (disjoint from D_train)
- Threshold (20%) is configurable via `--weak_threshold`

## Results (2026-01-13)

**Primary metric: AUC-ROC** - Uses IRT formula P(success) = sigmoid(θ - β) to predict agent-task outcomes.

| Feature Source | D_train AUC | D_valid AUC | ΔAUC | Notes |
|----------------|-------------|-------------|------|-------|
| Prior only (embedding) | 0.6830 | 0.7383 | — | Baseline |
| + Simple features | 0.7076 | **0.7444** | **+0.0062** | Best result |
| + LLM judge features | 0.6929 | 0.7215 | -0.0168 | Overfits |
| + Lunette features | 0.6846 | 0.7383 | +0.0000 | Too few |
| + Trajectory embeddings | — | 0.7384 | +0.0002 | No improvement |

**Key finding:** Simple trajectory features (message count, chars, resolve rate) provide a small but positive improvement. LLM judge features overfit and hurt validation. Trajectory embeddings provide no signal beyond the prior.

### Embedding PCA Ablation (2026-01-13)

Tested trajectory embeddings with PCA dimensionality reduction and varying ridge alphas. Key findings:

| Aggregation | PCA Dims | Best Alpha | D_valid AUC | ΔAUC |
|-------------|----------|------------|-------------|------|
| mean_only | None | 1M | 0.7384 | +0.0001 |
| mean_only | 25 | 1M | 0.7384 | +0.0002 |
| mean_std | None | 1M | 0.7383 | 0.0000 |

**Conclusion:** Embeddings contain no useful signal for difficulty residual prediction:
- Only alpha=1M (extreme regularization) preserves prior performance
- Lower alphas (100-100k) hurt AUC by -0.7% to -5%
- PCA doesn't help—the correction term is zeroed out regardless of dimensionality
- The feature/sample ratio (4096-8192 dims / 119 samples) is too extreme

## Feature Sources

### 1. Simple Features (Default)

5 basic trajectory statistics averaged across agents:

| Feature | Description |
|---------|-------------|
| `avg_message_count` | Average number of messages in trajectory |
| `avg_total_chars` | Average total character count |
| `avg_assistant_ratio` | Ratio of assistant messages to total |
| `avg_message_length` | Average characters per message |
| `resolved_rate` | Fraction of trajectories that resolved the task |

### 2. LLM Judge Features

Pre-compute with:
```bash
python -m experiment_b.llm_judge.compute_features_v1 --dry_run  # See what would be computed
python -m experiment_b.llm_judge.compute_features_v7 --limit 50  # Compute v7 features
```

**Important: Feature Version Coverage**

| Version | Tasks | Agents | Notes |
|---------|-------|--------|-------|
| v5_single | **143** | 1 | Best coverage, single agent (location_vs_fix_alignment) |
| v4 | 143 | 5 | Multi-agent features |
| v1 | 126 | 3 | Original features |
| v7 | 26 | 31 | Most agents but fewest tasks - **insufficient for D_valid** |

**Recommendation:** Use `--feature_source llm_judge_v5_single` for best task coverage. v7 only has 26 tasks with features, which means D_valid tasks likely have no features and fall back to prior-only predictions.

Features (v7):
- **Primary**: `llm_judge_difficulty_score` (0-1)
- **Competencies (1-4)**: backtracking_exploration, task_decomposition, observation_reading, self_verification
- **Failure modes (0-1)**: localization_failure, strategy_defect, implementation_defect, incomplete_repair, verification_failure
- **Trajectory signals (0-1)**: agent_looping, agent_gave_up_early, agent_wrong_focus, context_overflow

### 3. Lunette Features

Uses Lunette API for grading. Requires trajectories to be uploaded first (see [LUNETTE.md](../lunette_utils/LUNETTE.md)).

### 4. Embedding Features (Experimental)

Uses VLM trajectory embeddings instead of hand-crafted features. See [embeddings/README.md](embeddings/README.md) for full documentation.

```bash
# Compute embeddings on cluster (GPU required)
sbatch scripts/embedding/compute_embeddings_multi_gpu.sh

# Train and evaluate (CPU)
python -m experiment_b.embeddings.train_evaluate \
    --embeddings_dir chris_output/experiment_b/trajectory_embeddings/full_difficulty
```

## Module Structure

```
experiment_b/
├── __init__.py
├── README.md                      # This file
├── config.py                      # ExperimentConfig dataclass
├── data_splits.py                 # Agent/task splitting logic
├── prior_model.py                 # Prior models (heuristic + embedding)
├── posterior_model.py             # Prior + trajectory correction
├── train_evaluate.py              # Main training pipeline
├── trajectory_features.py         # Simple 5-feature extraction
├── trajectory_features_v2.py      # Extended simple features
│
├── embeddings/                    # VLM embedding approach
│   ├── README.md                  # Embedding documentation
│   ├── compute_embeddings.py      # GPU embedding extraction
│   ├── aggregator.py              # Multi-trajectory aggregation
│   ├── posterior_model.py         # Ridge regression on embeddings
│   ├── train_evaluate.py          # Embedding training pipeline
│   └── pca_ablation.py            # PCA ablation study
│
├── llm_judge/                     # LLM-as-judge approach
│   ├── features_v1.py ... v7.py   # Feature extraction versions
│   ├── compute_features_*.py      # Pre-compute scripts
│   └── analyze_residuals.py       # Analysis scripts
│
└── lunette/                       # Lunette grading approach
    ├── features.py                # Lunette feature extraction
    ├── compute_features.py        # Pre-compute with Lunette API
    └── structured_output.py       # Structured output utilities
```

## Regression Modes

Three approaches for combining trajectory features with the prior:

```bash
# Compare all modes
python -m experiment_b.train_evaluate --feature_source llm_judge_v5_single --compare_modes

# Run specific mode
python -m experiment_b.train_evaluate --regression_mode direct_with_prior
```

| Mode | Formula | Features | Target |
|------|---------|----------|--------|
| `residual` (default) | `prior + psi * features` | trajectory only | ground_truth - prior |
| `direct_with_prior` | `model(features, prior)` | trajectory + prior prediction | ground_truth |
| `direct_with_prior_features` | `model(features, embeddings)` | trajectory + prior input features | ground_truth |

**Results (llm_judge_v5_single, 2026-01-14, with RidgeCV alpha selection):**

| Mode | Best Alpha | D_train AUC | D_valid AUC | ΔAUC vs Prior |
|------|------------|-------------|-------------|---------------|
| Prior only | — | 0.6823 | **0.7362** | — |
| residual | 1e+06 | 0.6820 | **0.7362** | +0.0000 |
| direct_with_prior | 1e+06 | 0.7350 | 0.6911 | -0.0451 |
| direct_with_prior_features | 1e+06 | 0.7353 | 0.6880 | -0.0481 |

**Key finding:** Even with RidgeCV selecting optimal regularization (alpha=1e+06), direct regression modes underperform. The residual mode perfectly preserves the prior's AUC. Direct modes fail because they try to predict absolute difficulty from scratch rather than learning a correction—with 117 training samples, they can't generalize even with strong regularization. **Use `residual` mode**.

## Configuration

```python
@dataclass
class ExperimentConfig:
    # Data paths (use 1PL model for consistency with evaluation formula)
    items_path: Path = Path("clean_data/swebench_verified_20251120_full/1d_1pl/items.csv")
    responses_path: Path = Path("clean_data/swebench_verified/swebench_verified_20251120_full.jsonl")
    trajectories_dir: Path = Path("trajectory_data/unified_trajs")
    output_dir: Path = Path("chris_output/experiment_b")

    # Agent splitting
    m1_fraction: float = 0.4  # Oldest 40%
    m2_fraction: float = 0.4  # Middle 40%
    # M3 = remaining 20%

    # Task selection
    weak_threshold: float = 0.2  # Max pass rate for "hard" tasks
    strong_min_improvement: float = 0.1  # Min improvement for strong group

    # Model parameters
    prior_alpha: float = 1.0  # Ridge alpha for prior
    posterior_alpha: float = 1.0  # Ridge alpha for psi
```

## Output

Results saved to `chris_output/experiment_b/experiment_b_results.json`:

```json
{
  "split": {
    "m1_agents": ["agent1", ...],
    "m2_agents": ["agent2", ...],
    "m3_agents": ["agent3", ...],
    "d_train_tasks": ["task1", ...],
    "d_valid_tasks": ["task2", ...]
  },
  "prior_train": {"pearson_r": 0.031, "mse": 1.23, "n": 45},
  "posterior_train": {"pearson_r": 0.114, "mse": 1.18, "n": 45},
  "prior_valid": {"pearson_r": -0.153, "mse": 1.45, "n": 38},
  "posterior_valid": {"pearson_r": -0.087, "mse": 1.41, "n": 38},
  "psi_coefficients": {...}
}
```

## Restriction of Range Issue

**Important**: Low correlation on D_train/D_valid is misleading due to "restriction of range."

D_train is a biased subset with much lower variance in difficulty:

| Dataset | Mean b | Std b | Range |
|---------|--------|-------|-------|
| All 500 tasks | 0.21 | 2.13 | [-3.78, 5.16] |
| D_train (119) | 0.91 | 0.81 | [-0.43, 3.58] |

**Recommendation**: Use AUC-ROC instead of correlation/RMSE for evaluation.

## References

- [SWE-Bench Failure Analysis](https://arxiv.org/pdf/2509.13941) - Taxonomy of failure modes
- [AgentDiagnose](https://aclanthology.org/2025.emnlp-demos.15.pdf) - Trajectory quality judging

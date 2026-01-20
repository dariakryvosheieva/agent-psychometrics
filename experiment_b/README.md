# Experiment B: Frontier Task Difficulty Prediction

Predict difficulty of **frontier tasks** (tasks only solvable by newer models) using methods that do NOT have access to post-frontier agents. Evaluate using ROC-AUC after projecting predicted difficulties onto the oracle IRT scale.

## Overview

**Research Question**: Can we predict the difficulty of tasks that are currently beyond the capability of existing models, using only information available before those models were released?

**Setting**:
- **Date-based split**: Pre-frontier vs Post-frontier agents (by release date)
- **Frontier tasks**: Tasks with ≤10% pass rate among pre-frontier agents, but >10% among post-frontier agents
- **No data leakage**: Predictions made using only pre-frontier information; post-frontier agents only used for evaluation

## Quick Start

```bash
source .venv/bin/activate

# Run on SWE-bench (default)
python -m experiment_b.compare_methods

# Run on TerminalBench
python -m experiment_b.compare_methods --dataset terminalbench

# Save results to CSV
python -m experiment_b.compare_methods --output_csv results.csv
```

## Results (2026-01-20)

### SWE-bench Verified

**Data**: 40 frontier tasks, 19 post-frontier eval agents
**Cutoff**: 2025-08-07 (gpt-5-mini release)

| Method | ROC-AUC | Spearman ρ | p-value |
|--------|---------|------------|---------|
| Oracle (upper bound) | 0.7716 | 1.0000 | <0.0001 |
| **Feature-IRT (Embedding)** | **0.7074** | 0.1850 | 0.2531 |
| Baseline IRT | 0.6978 | 0.3336 | 0.0354 |
| Feature-IRT (LLM Judge) | 0.6841 | -0.1236 | 0.4472 |
| LLM Judge + Ridge | 0.6835 | -0.0921 | 0.5718 |
| Embedding + Ridge | 0.6655 | -0.1629 | 0.3154 |

### TerminalBench

**Data**: 18 frontier tasks, 31 post-frontier eval agents
**Cutoff**: 2025-11-17

| Method | ROC-AUC | Spearman ρ | p-value |
|--------|---------|------------|---------|
| Oracle (upper bound) | 0.8130 | 1.0000 | <0.0001 |
| **Feature-IRT (Embedding)** | **0.7378** | **0.5397** | 0.0208 |
| LLM Judge + Ridge | 0.7320 | -0.1547 | 0.5399 |
| Feature-IRT (LLM Judge) | 0.7314 | 0.0630 | 0.8040 |
| Baseline IRT | 0.6988 | 0.5253 | 0.0252 |
| Embedding + Ridge | 0.6007 | -0.2632 | 0.2914 |

**Key observations**:
- **Feature-IRT (Embedding) is the best method** on both benchmarks
- On SWE-bench: +1% AUC over Baseline IRT (0.7074 vs 0.6978)
- On TerminalBench: +4% AUC over Baseline IRT (0.7378 vs 0.6988) with best Spearman correlation (0.54)
- Ridge-based predictors have poor Spearman correlation on frontier tasks, suggesting they don't generalize well
- Feature-IRT learns task difficulties jointly with agent abilities, leveraging response patterns

## Methods Compared

| Method | Description | Training Data |
|--------|-------------|---------------|
| **Oracle** | True IRT difficulties (upper bound) | All agents |
| **Baseline IRT** | IRT trained on pre-frontier agents only | Pre-frontier responses |
| **Feature-IRT** | Joint IRT + feature learning (see below) | All tasks + pre-frontier responses |
| **Embedding + Ridge** | Task embeddings → Ridge regression | Non-frontier tasks + baseline IRT β |
| **LLM Judge + Ridge** | LLM semantic features → Ridge regression | Non-frontier tasks + baseline IRT β |

### Feature-IRT (New)

Feature-IRT learns task difficulties as a linear function of features, jointly with agent abilities:

```
b_i = w^T f_i + bias + r_i    (task difficulty)
θ_j learned jointly           (agent ability)
P(success) = sigmoid(θ_j - b_i)
```

**Key differences from Ridge predictors**:
- Learns from response patterns (IRT likelihood), not just baseline IRT difficulties
- Trains on ALL tasks (including frontier), since it uses pre-frontier agent responses
- Per-task residuals (r_i) with strong L2 regularization encourage feature-based predictions
- Ridge warm-start initialization for feature weights

**Hyperparameters** (grid search available via `--grid_search`):
- `l2_weight`: Regularization on feature weights (default: 0.01)
- `l2_residual`: Regularization on per-task residuals (default: 10.0)
- `use_residuals`: Whether to include per-task residuals (default: True)

## Evaluation Methodology

### 1. Spearman Correlation
- Compare predicted vs oracle difficulty ranking on frontier tasks
- No scale alignment needed (rank-based metric)

### 2. ROC-AUC with Scale Alignment
1. **Identify anchor tasks**: Tasks with 10-90% pass rate in BOTH pre- and post-frontier groups
2. **Fit affine transformation**: `oracle_β = slope × predicted_β + intercept` on anchors
3. **Compute probabilities**: For each (post-frontier agent, frontier task): `P(success) = sigmoid(θ_oracle - β_shifted)`
4. **Calculate ROC-AUC**: Compare predicted probabilities to actual responses

## Data Leakage Constraints

**Critical**: Oracle data and post-frontier agent data must NEVER be exposed during training.

- **Training ground truth**: Always from baseline IRT (pre-frontier agents only)
- **Oracle IRT**: Used ONLY for evaluation metrics
- **`--train_on_all_tasks` flag**: Includes frontier tasks in training but still uses baseline IRT difficulties as ground truth (poorly calibrated for frontier tasks since pre-frontier agents rarely solve them)

This constraint ensures a realistic simulation of predicting difficulty for tasks beyond current model capabilities.

## Architecture

Uses the shared `shared/` infrastructure for predictors:

```python
from shared.feature_source import EmbeddingFeatureSource, CSVFeatureSource
from shared.feature_predictor import FeatureBasedPredictor
from experiment_b.shared.feature_irt_predictor import FeatureIRTPredictor

# Ridge-based predictor
source = EmbeddingFeatureSource(embeddings_path)
predictor = FeatureBasedPredictor(source)

# Feature-IRT predictor (joint learning)
predictor = FeatureIRTPredictor(source, use_residuals=True)
predictor.fit(task_ids, ground_truth_b, responses)  # responses = pre-frontier only
```

### Directory Structure

```
experiment_b/
├── compare_methods.py        # Main entry point
├── shared/
│   ├── data_splits.py        # Agent/task splitting utilities
│   ├── evaluate.py           # Evaluation metrics (Spearman, AUC, alignment)
│   ├── baseline_irt.py       # Baseline IRT training with caching
│   └── feature_irt_predictor.py  # Feature-IRT predictor (NEW)
└── datasets/                 # Dataset-specific configurations
    ├── base.py               # DatasetConfig base class
    ├── swebench.py           # SWE-bench config
    └── terminalbench.py      # TerminalBench config
```

### Key Functions

**Data Splitting** (`data_splits.py`):
- `split_agents_by_dates()`: Split by date cutoff
- `identify_frontier_tasks()`: Tasks hard for pre-frontier, solved by post-frontier
- `identify_nontrivial_tasks()`: Anchor tasks (10-90% pass rate)

**Evaluation** (`evaluate.py`):
- `compute_frontier_difficulty_metrics()`: Spearman correlation
- `compute_scale_offset()`: Fit alignment transformation
- `compute_frontier_auc()`: ROC-AUC on frontier tasks

## Data Paths

### SWE-bench

| File | Purpose |
|------|---------|
| `clean_data/swebench_verified_20251120_full/1d_1pl/items.csv` | Oracle IRT difficulties |
| `clean_data/swebench_verified/swebench_verified_20251120_full.jsonl` | Response matrix |
| `chris_output/experiment_a/embeddings/` | Task embeddings |
| `chris_output/experiment_a/llm_judge_features/` | LLM judge features |

### TerminalBench

| File | Purpose |
|------|---------|
| `chris_output/terminal_bench_2.0/1d/items.csv` | Oracle IRT difficulties |
| `data/terminal_bench/terminal_bench_2.0.jsonl` | Response matrix |
| `chris_output/experiment_a_terminalbench/embeddings/` | Task embeddings |
| `chris_output/experiment_a_terminalbench/llm_judge_features/` | LLM judge features (4 pre-selected) |

## Configuration

Dataset configs in `datasets/`:

```python
@dataclass
class DatasetConfig:
    responses_path: Path
    oracle_irt_path: Path
    embeddings_path: Path
    llm_judge_path: Path
    cutoff_date: str
    pre_threshold: float = 0.1   # Max pass rate for pre-frontier
    post_threshold: float = 0.1  # Min pass rate for post-frontier

    @property
    def llm_judge_feature_cols(self) -> List[str]:
        # SWE-bench: 9 features
        # TerminalBench: 4 pre-selected features
```

## Command Line Options

```
--dataset             Dataset to use: swebench (default) or terminalbench
--output_csv          Save results to CSV file
--train_on_all_tasks  Include frontier tasks in training (still uses baseline IRT)
--grid_search         Run grid search over Feature-IRT hyperparameters
--verbose             Show alignment parameters and training progress
```

## Caches

| Cache | Location | When to Clear |
|-------|----------|---------------|
| **Baseline IRT** | `chris_output/experiment_b/{dataset}/baseline_irt/` | Auto-invalidated when training data changes |
| **Embeddings** | `chris_output/experiment_a{_terminalbench}/embeddings/` | When changing backbone |
| **LLM Features** | `chris_output/experiment_a{_terminalbench}/llm_judge_features/` | When re-extracting |

## Related Experiments

- **Experiment A**: Prior validation - tests how well static task features predict difficulty on held-out tasks
- **Experiment SAD-IRT**: Uses trajectory information for frontier difficulty prediction

## References

- [IRT Models Documentation](../docs/IRT_MODELS.md)
- [Research Proposal](../chris%20proposal.md) - Section 3.2

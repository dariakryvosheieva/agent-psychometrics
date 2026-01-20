# Experiment B: Frontier Task Difficulty Prediction

Predict difficulty of **frontier tasks** (tasks only solvable by newer models) using any method WITHOUT access to held-out post-frontier agents. Evaluate using ROC-AUC after projecting predicted difficulties onto the oracle IRT scale.

## Overview

**Research Question**: Can we predict the difficulty of tasks that are currently beyond the capability of existing models, using only information available before those models were released?

**Setting**:
- **Date-based split**: Pre-frontier (< 2025-08-07) vs Post-frontier (>= 2025-08-07)
- **Frontier tasks**: Tasks with ≤10% pass rate among pre-frontier agents, but >10% among post-frontier agents
- **No data leakage**: Predictions made using only pre-frontier information; post-frontier agents only used for evaluation

## Quick Start

```bash
source .venv/bin/activate

# Run comparison with default settings
python -m experiment_b.compare_methods

# Compare training on pre-frontier tasks only vs all tasks
python -m experiment_b.compare_methods --train_on_all_tasks

# Run on TerminalBench
python -m experiment_b.compare_methods --dataset terminalbench

# Save results to CSV
python -m experiment_b.compare_methods --output_csv chris_output/experiment_b_results.csv

# Show alignment parameters
python -m experiment_b.compare_methods --verbose
```

## Methods Compared

| Method | Description | Training Data |
|--------|-------------|---------------|
| **Oracle** | True IRT difficulties (upper bound) | All agents |
| **Baseline IRT** | IRT trained on pre-frontier agents only | Pre-frontier responses |
| **Embedding** | Task embeddings → difficulty via Ridge | Non-frontier tasks + baseline IRT β |
| **LLM Judge** | LLM-extracted semantic features via Lasso/Ridge | Non-frontier tasks + baseline IRT β |

## Evaluation Methodology

### 1. Spearman Correlation
- Compare predicted vs oracle difficulty ranking on frontier tasks
- No scale alignment needed (rank-based metric)

### 2. ROC-AUC with Scale Alignment
1. **Identify anchor tasks**: Tasks with 10-90% pass rate in BOTH pre- and post-frontier groups
2. **Fit affine transformation**: `oracle_β = slope × predicted_β + intercept` on anchors
3. **Compute probabilities**: For each (post-frontier agent, frontier task) pair:
   - `P(success) = sigmoid(θ_oracle - β_shifted)`
4. **Calculate ROC-AUC**: Compare predicted probabilities to actual responses

**Note**: Scale alignment uses oracle information and is ONLY for evaluation. In production, you would not have access to oracle difficulties.

## Data Leakage Constraints

**Critical**: Oracle data and held-out (post-frontier) agent data must NEVER be exposed during training.

- **Training ground truth**: Always from baseline IRT (pre-frontier agents only)
- **Oracle IRT**: Used ONLY for evaluation metrics
- **`--train_on_all_tasks` flag**: Includes frontier tasks in training but still uses baseline IRT difficulties as ground truth (these are poorly calibrated for frontier tasks since pre-frontier agents rarely solve them)

This constraint ensures a realistic simulation of predicting difficulty for tasks that are beyond current model capabilities.

## Data Splits

```
Pre-frontier agents (< 2025-08-07):  ~107 agents
Post-frontier agents (>= 2025-08-07): ~24 agents
Frontier tasks:                       ~40 tasks
Anchor tasks (10-90% pass rate):      ~200 tasks
```

The cutoff date (2025-08-07) corresponds to the gpt-5-mini release, which significantly improved agent capabilities.

## Directory Structure

```
experiment_b/
├── __init__.py           # Module docstring
├── README.md             # This file
├── config.py             # ExperimentBConfig dataclass
├── data_splits.py        # Agent/task splitting utilities
├── evaluate.py           # Evaluation metrics (Spearman, AUC, alignment)
├── compare_methods.py    # Main entry point for comparing methods
└── datasets/             # Dataset-specific configurations
    ├── __init__.py       # Dataset registry and factory
    ├── base.py           # DatasetConfig base class
    ├── swebench.py       # SWE-bench Verified config
    └── terminalbench.py  # TerminalBench config
```

## Configuration

All settings in `config.py`:

```python
@dataclass
class ExperimentBConfig:
    # Data paths
    responses_path: Path = "clean_data/swebench_verified/swebench_verified_20251120_full.jsonl"
    oracle_irt_path: Path = "clean_data/swebench_verified_20251120_full/1d/items.csv"
    baseline_irt_path: Path = "chris_output/sad_irt/baseline_irt/items.csv"
    embeddings_path: Path = "..."  # Any backbone's embeddings
    llm_judge_path: Path = "..."   # LLM judge features CSV

    # Frontier split
    cutoff_date: str = "20250807"
    pre_threshold: float = 0.1   # Max pass rate for pre-frontier
    post_threshold: float = 0.1  # Min pass rate for post-frontier

    # Alignment
    alignment_method: str = "affine"  # or "constant"
```

## Multi-Dataset Support

This experiment supports multiple datasets via the `--dataset` flag:

```bash
# SWE-bench Verified (default)
python -m experiment_b.compare_methods --dataset swebench

# TerminalBench
python -m experiment_b.compare_methods --dataset terminalbench
```

Default embeddings (DeepSeek-R1-Distill-Qwen-32B):
- SWE-bench: `chris_output/experiment_a/embeddings/embeddings__deepseek-ai__DeepSeek-R1-Distill-Qwen-32B__merged.npz`
- TerminalBench: `chris_output/experiment_a_terminalbench/embeddings/embeddings__deepseek-ai__DeepSeek-R1-Distill-Qwen-32B__pool-lasttoken__maxlen8192.npz`

## Results: SWE-bench Verified

**Data**: 40 frontier tasks, 19 post-frontier eval agents

| Method | ROC-AUC | Spearman ρ | p-value |
|--------|---------|------------|---------|
| Oracle | 0.7716 | 1.0000 | <0.0001 |
| Baseline IRT | 0.6978 | 0.3336 | 0.0354 |
| LLM Judge | 0.6835 | -0.0921 | 0.5718 |
| Embedding | 0.6655 | -0.1629 | 0.3154 |

**Key observations**:
- Baseline IRT (trained only on pre-frontier agents) already achieves reasonable AUC
- Embedding and LLM Judge predictors underperform baseline IRT on frontier tasks
- Spearman correlation is significant for IRT-based methods but not for feature-based predictors

## Related Experiments

- **Experiment A**: Prior validation - tests how well static task features predict difficulty
- **Experiment SAD-IRT**: SAD-IRT model for frontier difficulty prediction using trajectory information

## References

- [IRT Models Documentation](../docs/IRT_MODELS.md)
- [Data Pipeline](../docs/DATA_PIPELINE.md)
- [Research Proposal](../chris%20proposal.md) - Section 3.2

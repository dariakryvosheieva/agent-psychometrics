# Frontier Task Difficulty Prediction (Appendix H)

Predict difficulty of **frontier tasks** (tasks only solvable by newer models) using methods that do NOT have access to post-frontier agents.

## Overview

**Research Question**: Can we predict the difficulty of tasks that are currently beyond the capability of existing models, using only information available before those models were released?

**Setting**:
- **Date-based split**: Pre-frontier vs post-frontier agents (by release date)
- **No data leakage**: Predictions made using only pre-frontier information; post-frontier agents only used for evaluation

## Evaluation Metrics

**Primary: Mean Per-Agent AUC** (scale-free). For each post-frontier agent, compute AUC on frontier tasks using predicted difficulty ranking, then average across all agents with response variance. No oracle abilities or scale alignment needed.

**Secondary: Pooled ROC-AUC**. Pools all (agent, task) pairs; requires fitting an affine transformation to align predicted difficulties to the oracle scale.

## Frontier Task Definitions

1. **Zero pre-frontier** (`zero_pre`) [default]: Tasks with 0% pre-frontier pass rate AND >0% post-frontier pass rate
2. **Pass-rate based** (`passrate`): Tasks with <=10% pre-frontier AND >10% post-frontier pass rate
3. **Pre-only** (`pre_only`): Tasks with <=X% pre-frontier pass rate (no post-frontier filter)
4. **IRT-based** (`irt`): Tasks where no pre-frontier agent has >=50% solve probability under IRT

## Quick Start

```bash
source .venv/bin/activate

# Run on SWE-bench Verified (default)
python -m experiment_appendix_h_hard_tasks.compare_methods

# Run on other datasets
python -m experiment_appendix_h_hard_tasks.compare_methods --dataset swebench_pro
python -m experiment_appendix_h_hard_tasks.compare_methods --dataset terminalbench
python -m experiment_appendix_h_hard_tasks.compare_methods --dataset gso

# Enable date forecasting
python -m experiment_appendix_h_hard_tasks.compare_methods --forecast_dates

# Run threshold sweep analysis
python -m experiment_appendix_h_hard_tasks.threshold_sweep --datasets swebench

# Save results to CSV
python -m experiment_appendix_h_hard_tasks.compare_methods --output_csv results.csv
```

## Results

All results use the **zero_pre** frontier definition and report **Mean Per-Agent AUC +/- SEM**.

### SWE-bench Verified

**Cutoff**: 2025-05-01 | **Pre-frontier agents**: 76 | **Post-frontier agents**: 55 | **Frontier tasks**: 34

| Method | Mean AUC +/- SEM | ROC-AUC |
|--------|------------------|---------|
| Oracle (upper bound) | **0.7322 +/- 0.039** | 0.8399 |
| Trajectory + Ridge | 0.5442 +/- 0.033 | 0.7157 |
| Grouped Ridge (all features) | 0.5256 +/- 0.032 | 0.7041 |
| Baseline IRT (pre-frontier only) | 0.4854 +/- 0.031 | 0.7503 |
| Embedding + Ridge | 0.4716 +/- 0.039 | 0.6868 |
| LLM Judge + Ridge | 0.4665 +/- 0.034 | 0.6898 |

### SWE-bench Pro

**Cutoff**: 2025-06-01 | **Pre-frontier agents**: 5 | **Post-frontier agents**: 9 | **Frontier tasks**: 114

| Method | Mean AUC +/- SEM | ROC-AUC |
|--------|------------------|---------|
| Oracle (upper bound) | **0.7377 +/- 0.029** | 0.8000 |
| Embedding + Ridge | 0.5896 +/- 0.029 | 0.7132 |
| Grouped Ridge | 0.5757 +/- 0.027 | 0.7103 |
| LLM Judge + Ridge | 0.5419 +/- 0.024 | 0.7018 |
| Baseline IRT | 0.5055 +/- 0.017 | 0.6958 |

### TerminalBench

**Cutoff**: 2025-09-01 | **Pre-frontier agents**: 37 | **Post-frontier agents**: 46 | **Frontier tasks**: 11

| Method | Mean AUC +/- SEM | ROC-AUC |
|--------|------------------|---------|
| Oracle (upper bound) | **0.7614 +/- 0.041** | 0.8348 |
| Baseline IRT | 0.5910 +/- 0.051 | 0.7681 |
| LLM Judge + Ridge | 0.5318 +/- 0.046 | 0.7539 |
| Embedding + Ridge | 0.4253 +/- 0.041 | 0.6676 |

### GSO

**Cutoff**: 2025-08-15 | **Pre-frontier agents**: 8 | **Post-frontier agents**: 6 | **Frontier tasks**: 33

| Method | Mean AUC +/- SEM | ROC-AUC |
|--------|------------------|---------|
| Oracle (upper bound) | **0.7319 +/- 0.043** | 0.8043 |
| Grouped Ridge | 0.5886 +/- 0.128 | 0.7351 |
| LLM Judge + Ridge | 0.5787 +/- 0.114 | 0.7334 |
| Embedding + Ridge | 0.5678 +/- 0.150 | 0.7023 |
| Baseline IRT | 0.5206 +/- 0.047 | 0.7155 |

## Methods Compared

| Method | Description | Training Data |
|--------|-------------|---------------|
| **Oracle** | True IRT difficulties (upper bound) | All agents |
| **Baseline IRT** | IRT trained on pre-frontier agents only | Pre-frontier responses |
| **Feature-IRT** | Joint IRT + feature learning | All tasks + pre-frontier responses |
| **Embedding + Ridge** | Task embeddings -> Ridge regression | Non-frontier tasks + baseline IRT |
| **LLM Judge + Ridge** | LLM semantic features -> Ridge regression | Non-frontier tasks + baseline IRT |

### Feature-IRT

Learns task difficulties as a linear function of features, jointly with agent abilities:

```
b_i = w^T f_i + bias + r_i    (task difficulty)
theta_j learned jointly        (agent ability)
P(success) = sigmoid(theta_j - b_i)
```

Key differences from Ridge predictors:
- Learns from response patterns (IRT likelihood), not just baseline IRT difficulties
- Trains on all tasks (including frontier), since it uses pre-frontier agent responses
- Per-task residuals (r_i) with strong L2 regularization encourage feature-based predictions

## Date Forecasting

Predicts **when** tasks will become solvable with 50% probability. From IRT, `P(success) = 0.5` when `theta = beta`. Combined with the finding that frontier ability grows linearly over time, we invert the ability-over-time relationship to predict when a task with difficulty beta will become solvable.

**Methodology** (per-method, avoids data leakage):
1. Fit ability-over-time regression using each method's own IRT abilities (pre-frontier only)
2. Predict solvability date by inverting: `days = (beta - intercept) / slope`
3. Evaluate against ground truth (Oracle IRT's first-capable-date)

Only methods with their own IRT model (Oracle, Baseline IRT, Feature-IRT) can produce date forecasts. Ridge-based methods do not learn IRT abilities and cannot forecast dates.

```bash
# Run with date forecasting
python -m experiment_appendix_h_hard_tasks.compare_methods --forecast_dates
```

## Data Leakage Constraints

Oracle data and post-frontier agent data are NEVER exposed during training. Training ground truth always comes from baseline IRT (pre-frontier agents only). Oracle IRT is used only for evaluation metrics.

## Directory Structure

```
experiment_appendix_h_hard_tasks/
├── compare_methods.py        # Main evaluation script
├── threshold_sweep.py        # Threshold sweep analysis
├── trajectory_data/          # Downloaded trajectories
├── trajectory_summarization_api/  # Trajectory summarization via OpenAI API
├── trajectory_upload/        # Trajectory conversion and upload
├── shared/                   # Shared utilities
│   ├── config_base.py        # DatasetConfig base class
│   ├── data_preparation.py   # Data loading, frontier identification
│   ├── evaluation.py         # ROC-AUC, scale alignment
│   ├── prediction_methods.py # Predictor classes and method collection
│   └── date_forecasting.py   # Date prediction utilities
├── swebench/config.py        # SWE-bench Verified configuration
├── swebench_pro/config.py    # SWE-bench Pro configuration
├── terminalbench/config.py   # TerminalBench configuration
└── gso/config.py             # GSO configuration
```

## Data Paths

### SWE-bench Verified

| File | Purpose |
|------|---------|
| `data/swebench_verified/irt/1d_1pl/items.csv` | Oracle IRT difficulties |
| `data/swebench_verified/responses.jsonl` | Response matrix |
| `embeddings/embeddings__...__SWE-bench_Verified__test__maxlen8192.npz` | Task embeddings |
| `llm_judge_features/defaults/swebench_verified/llm_judge_features.csv` | LLM judge features |

### SWE-bench Pro

| File | Purpose |
|------|---------|
| `data/swebench_pro/irt/1d_1pl/items.csv` | Oracle IRT difficulties |
| `data/swebench_pro/responses.jsonl` | Response matrix |
| `data/swebench_pro/agent_dates.json` | Agent release dates |

### TerminalBench

| File | Purpose |
|------|---------|
| `data/terminalbench/irt/1d_1pl/items.csv` | Oracle IRT difficulties |
| `data/terminalbench/responses.jsonl` | Response matrix |
| `data/terminalbench/model_release_dates.json` | Model release dates |

### GSO

| File | Purpose |
|------|---------|
| `data/gso/irt/1d_1pl/items.csv` | Oracle IRT difficulties |
| `data/gso/responses.jsonl` | Response matrix |
| `data/gso/agent_dates.json` | Agent release dates |

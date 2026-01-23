# Experiment A: Prior Validation (IRT AUC)

Evaluates how well a difficulty predictor can predict agent success on held-out tasks using the 1PL IRT model.

## Overview

**Goal**: Validate that predicted task difficulties are useful for forecasting agent performance without running agents on new tasks.

**Core Idea**: Given a predicted difficulty β̂_i and known agent ability θ_j, compute:

```
P(success) = sigmoid(θ_j - β̂_i)
```

Then measure AUC by comparing these predicted probabilities to actual binary outcomes.

## Quick Start

```bash
source .venv/bin/activate

# Run SWE-bench Verified experiment
python -m experiment_a.swebench.train_evaluate

# Run SWE-bench Pro experiment
python -m experiment_a.swebench_pro.train_evaluate

# Run TerminalBench experiment (binomial mode - default, k/n successes)
python -m experiment_a.terminalbench.train_evaluate

# Run TerminalBench experiment (binary mode - any success = 1)
python -m experiment_a.terminalbench.train_evaluate --binary

# Dry run to check config
python -m experiment_a.swebench.train_evaluate --dry_run
```

## Results (2026-01-22)

### SWE-bench Verified (5-Fold Cross-Validation)

**Data**: 500 tasks, 131 agents

| Method | Mean AUC | Std |
|--------|----------|-----|
| Oracle (true b) | 0.9441 | 0.0045 |
| Embedding | 0.8269 | 0.0070 |
| LLM Judge | 0.8230 | 0.0127 |
| Constant (mean b) | 0.7149 | 0.0108 |
| Agent-only | 0.7150 | 0.0109 |

### SWE-bench Pro (5-Fold Cross-Validation)

**Data**: 730 tasks, 14 agents

| Method | Mean AUC | Std |
|--------|----------|-----|
| Oracle (true b) | 0.9180 | 0.0160 |
| Embedding | 0.7364 | 0.0166 |
| LLM Judge | 0.7310 | 0.0103 |
| Agent-only | 0.6568 | 0.0127 |
| Constant (mean b) | 0.6563 | 0.0123 |

**Note**: SWE-bench Pro shows lower predictor AUCs (~0.73) compared to SWE-bench Verified (~0.83). This may be due to having fewer agents (14 vs 131) for IRT training, or inherently harder-to-predict task difficulty in the Pro dataset.

### TerminalBench (5-Fold Cross-Validation)

TerminalBench supports two data modes:
- **Binomial** (default): Models k successes out of 5 trials per agent-task pair
- **Binary** (`--binary`): Collapses to any success = 1 (single observation per pair)

#### Binomial Mode (Default)

**Data**: 89 tasks, 83 agents, 5 trials each

| Method | Mean AUC | Std | Pass Rate MSE |
|--------|----------|-----|---------------|
| Oracle (true b) | 0.9040 | 0.0109 | 0.0540 |
| Embedding | 0.7817 | 0.0429 | - |
| LLM Judge | 0.7738 | 0.0256 | - |
| Constant (mean b) | 0.7036 | 0.0123 | 0.1490 |
| Agent-only | 0.7039 | 0.0125 | 0.1404 |

#### Binary Mode (`--binary`)

**Data**: 88 tasks, 83 agents (any success = 1)

| Method | Mean AUC | Std |
|--------|----------|-----|
| Oracle (true b) | 0.9319 | 0.0104 |
| Embedding | 0.7779 | 0.0505 |
| LLM Judge | 0.7734 | 0.0311 |
| Constant (mean b) | 0.6904 | 0.0163 |
| Agent-only | 0.6904 | 0.0167 |

**Summary**: Binomial mode (default) preserves more information about task difficulty gradations and shows slightly better predictor AUCs when evaluated fairly. See "Fair Comparison" below for details.

#### Fair Comparison: Training vs Evaluation Methods (5-Fold CV)

To fairly compare binomial vs binary training, we hold the evaluation method constant:

**Multi-attempt AUC Evaluation** (expand to 5 observations per pair):

| Training Method | Oracle | Embedding | LLM Judge | Constant | Agent-only |
|-----------------|--------|-----------|-----------|----------|------------|
| Binomial (k/n)  | 0.9040 | 0.7817    | 0.7738    | 0.7036   | 0.7039     |
| Binary (any success) | 0.8981 | 0.7761 | 0.7712    | 0.6904   | 0.6904     |

**Binary AUC Evaluation** (any_success = k > 0):

| Training Method | Oracle | Embedding | LLM Judge | Constant | Agent-only |
|-----------------|--------|-----------|-----------|----------|------------|
| Binomial (k/n)  | 0.9253 | 0.7800    | 0.7714    | 0.7153   | 0.7153     |
| Binary (any success) | 0.9319 | 0.7779 | 0.7734    | 0.6904   | 0.6904     |

**Key findings**:
- When evaluated with the **same metric**, binomial and binary training yield very similar predictor AUCs
- The apparent advantage of binary training (0.9319 vs 0.9037 Oracle AUC) in earlier comparisons was largely due to using different evaluation methods
- Binomial training shows a slight edge (~0.5-1% higher AUC) when evaluation is held constant

## Evaluation Protocol

1. **Split tasks** (not agents) into train/test sets using deterministic hash-based splitting
2. **Train IRT on train tasks only** to get uncontaminated ground truth difficulties
3. **Train difficulty predictor** on train tasks using train-only IRT difficulties as targets
4. **Predict difficulty** for test tasks
5. **Compute IRT probabilities**: For each (agent, task) pair, compute P(success) = sigmoid(θ - β̂)
6. **Calculate AUC**: Compare predicted probabilities to actual outcomes

### Data Leakage Prevention

The IRT model provides ground truth difficulties (β) used as training targets. To avoid data leakage, we train **two separate IRT models**:

1. **IRT^train (Train-only IRT)**: Trained on train tasks only - provides uncontaminated ground truth
2. **IRT^full (Full IRT)**: Trained on all tasks - **used ONLY for oracle baseline**

## Architecture

The experiment uses a unified CVPredictor protocol that all methods implement, enabling consistent k-fold cross-validation across both SWE-bench (binary) and TerminalBench (binomial) datasets.

### CVPredictor Protocol

All methods implement the same interface:

```python
class CVPredictor(Protocol):
    def fit(self, data: ExperimentData, train_task_ids: List[str]) -> None: ...
    def predict_probability(self, data: ExperimentData, agent_id: str, task_id: str) -> float: ...
```

### Shared Infrastructure

**`experiment_ab_shared/`** - Core abstractions shared with Experiment B:

| File | Purpose |
|------|---------|
| `dataset.py` | `ExperimentData` ABC with `BinaryExperimentData`, `BinomialExperimentData` |
| `feature_source.py` | `TaskFeatureSource` ABC with `EmbeddingFeatureSource`, `CSVFeatureSource` |
| `feature_predictor.py` | `FeatureBasedPredictor` (StandardScaler → RidgeCV) |
| `predictor_base.py` | `DifficultyPredictorBase` ABC |
| `evaluator.py` | `compute_auc()`, `compute_irt_probability()` |

**`experiment_a/shared/`** - Experiment A orchestration:

| File | Purpose |
|------|---------|
| `pipeline.py` | `ExperimentSpec`, `CVPredictorConfig`, `run_experiment_main()` |
| `cross_validation.py` | `CVPredictor` protocol, `run_cv()`, `k_fold_split_tasks()` |
| `baselines.py` | `OraclePredictor`, `ConstantPredictor`, `AgentOnlyPredictor`, `DifficultyPredictorAdapter`, `FeatureIRTCVPredictor` |

### SWE-bench Verified (`experiment_a/swebench/`)

| File | Purpose |
|------|---------|
| `train_evaluate.py` | Main entry point |
| `config.py` | `ExperimentAConfig` with default paths |
| `generate_embeddings.py` | Generate task embeddings |
| `compute_llm_judge_features.py` | Extract LLM semantic features |

### SWE-bench Pro (`experiment_a/swebench_pro/`)

| File | Purpose |
|------|---------|
| `train_evaluate.py` | Main entry point |
| `config.py` | `SWEBenchProConfig` with SWE-bench Pro paths |

### TerminalBench Specific (`experiment_a/terminalbench/`)

| File | Purpose |
|------|---------|
| `train_evaluate.py` | Main entry with `is_binomial=True` |
| `config.py` | `TerminalBenchConfig` with TerminalBench paths |
| `data_loader.py` | Load task data from terminal-bench repo |
| `binomial_metrics.py` | Pass rate MSE for binomial responses |
| `sampling.py` | Stratified train/test splitting |

## Methods

### Ridge Regression (Embedding, LLM Judge)

Standard approach: train Ridge regression to predict IRT difficulty from task features.

```
β̂_i = w^T f_i + bias
```

Uses ground truth difficulties from train-only IRT as targets. The adapter then uses IRT-trained abilities for probability computation.

### Feature-IRT (Joint Learning)

Jointly learns feature weights and agent abilities by maximizing IRT log-likelihood:

```
minimize: -Σ_ij log P(y_ij | θ_j, β_i) + λ_w ||w||² + λ_θ mean(θ)²

where β_i = w^T f_i + bias  (task difficulty from features)
      θ_j learned jointly    (agent ability)
```

Key differences from Ridge:
- Learns from response patterns (IRT likelihood), not frozen IRT difficulties
- Agent abilities are jointly optimized with feature weights
- Supports both Bernoulli (binary) and Binomial (multi-trial) likelihoods
- Uses internal 3-fold CV to select `l2_weight` from `[0.01, 0.1, 1.0, 10.0]` (similar to RidgeCV)

**Note**: In Experiment A (task holdout), Feature-IRT performs similarly to Ridge because it must generalize to unseen test tasks using only feature weights. This is unlike Experiment B (agent holdout) where Feature-IRT can leverage jointly-learned abilities across all tasks.

## Feature Sources

### 1. Embeddings (DeepSeek-R1-Distill-Qwen-32B)

Pre-computed embeddings are configured by default:
- SWE-bench Verified: `chris_output/experiment_a/embeddings/embeddings__deepseek-ai__DeepSeek-R1-Distill-Qwen-32B__merged.npz`
- SWE-bench Pro: `out/swebench_pro/embeddings__deepseek-ai__DeepSeek-R1-Distill-Qwen-32B__...npz`
- TerminalBench: `chris_output/experiment_a_terminalbench/embeddings/embeddings__deepseek-ai__DeepSeek-R1-Distill-Qwen-32B__pool-lasttoken__maxlen8192.npz`

To generate new embeddings:
```bash
python -m experiment_a.generate_embeddings --backbone "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
```

### 2. LLM Judge Features

Semantic features extracted via LLM structured output:

**SWE-bench Verified (10 features)**:
- fix_in_description, problem_clarity, error_message_provided, reproduction_steps
- fix_locality, domain_knowledge_required, fix_complexity, logical_reasoning_required, atypicality, integration_complexity

**SWE-bench Pro (8 features, auto-detected from v5 CSV)**:
- LLM features: fix_complexity, verification_difficulty, standard_pattern_available, integration_complexity
- Deterministic features: num_files_modified, num_hunks, num_lines_changed, log_lines_changed

**TerminalBench (9 features, auto-detected)**:
- solution_in_instruction, task_clarity, solution_size, domain_knowledge_required
- task_complexity, logical_reasoning_required, atypicality, tooling_complexity, log_lines

To extract features:
```bash
# SWE-bench features
python -m experiment_ab_shared.llm_judge extract --dataset swebench --dry-run
python -m experiment_ab_shared.llm_judge extract --dataset swebench

# TerminalBench features
python -m experiment_ab_shared.llm_judge extract --dataset terminalbench --dry-run
python -m experiment_ab_shared.llm_judge extract --dataset terminalbench

# Options
python -m experiment_ab_shared.llm_judge extract --dataset swebench --limit 50  # Process first 50 tasks
python -m experiment_ab_shared.llm_judge extract --dataset swebench --provider openai  # Use OpenAI
python -m experiment_ab_shared.llm_judge extract --dataset swebench --model claude-sonnet-4-20250514  # Use specific model

# Aggregate existing JSON files to CSV
python -m experiment_ab_shared.llm_judge aggregate --dataset swebench
```

## Data Paths

### SWE-bench Verified

| File | Purpose |
|------|---------|
| `clean_data/swebench_verified_20251120_full/1d_1pl/abilities.csv` | Oracle IRT abilities |
| `clean_data/swebench_verified_20251120_full/1d_1pl/items.csv` | Oracle IRT difficulties |
| `clean_data/swebench_verified/swebench_verified_20251120_full.jsonl` | Response matrix |
| `chris_output/experiment_a/irt_splits/` | Fold-specific IRT models (cached) |

### SWE-bench Pro

| File | Purpose |
|------|---------|
| `chris_output/swebench_pro_irt/1d/abilities.csv` | Oracle IRT abilities |
| `chris_output/swebench_pro_irt/1d/items.csv` | Oracle IRT difficulties |
| `out/chris_irt/swebench_pro.jsonl` | Response matrix |
| `chris_output/experiment_a_swebench_pro/irt_splits/` | Fold-specific IRT models (cached) |
| `chris_output/experiment_a_swebench_pro/llm_judge_features_v5/` | LLM Judge features (v5) |

### TerminalBench

| File | Purpose |
|------|---------|
| `chris_output/terminal_bench_2.0_binomial_1pl/1d/abilities.csv` | Oracle IRT abilities |
| `chris_output/terminal_bench_2.0_binomial_1pl/1d/items.csv` | Oracle IRT difficulties |
| `data/terminal_bench/terminal_bench_2.0_raw.jsonl` | Response matrix (binomial) |
| `terminal-bench/tasks/{task_id}/` | Task instructions and solutions |

## Command Line Options

```
--k_folds             Number of folds for cross-validation (default: 5)
--split_seed          Random seed for train/test split (default: 0)
--embeddings_path     Override default embeddings path
--llm_judge_features_path  Override default LLM features path
--output_dir          Output directory
--dry_run             Show configuration without running
--exclude_unsolved    Exclude tasks no agent solved
--include_feature_irt Include Feature-IRT joint learning methods (off by default)
```

## Output

Results saved to `chris_output/experiment_a/experiment_a_cv5_results.json`:

```json
{
  "config": {...},
  "data_summary": {"n_agents": 130, "n_tasks_total": 500},
  "oracle": {"mean_auc": 0.9441, "std": 0.0045},
  "embedding_predictor": {"mean_auc": 0.8269, "std": 0.0070},
  "llm_judge_predictor": {"mean_auc": 0.8227, "std": 0.0118},
  ...
}
```

## Caches

| Cache | Location | When to Clear |
|-------|----------|---------------|
| **IRT Split Models** | `chris_output/experiment_a/irt_splits/` | When changing split parameters |
| **Embeddings** | `chris_output/experiment_a/embeddings/` | When changing backbone |
| **LLM Judge Features** | `chris_output/experiment_a/llm_judge_features/` | When re-extracting |

## References

- IRT formula: `P = sigmoid(theta - beta)` matches py_irt's 1PL implementation
- [Research Proposal](../chris%20proposal.md) - Section 3.1

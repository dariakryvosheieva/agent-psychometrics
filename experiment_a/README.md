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

# Run ALL datasets and get a summary table
python -m experiment_a.run_all_datasets --unified_judge

# Run specific datasets only
python -m experiment_a.run_all_datasets --unified_judge --datasets gso terminalbench

# Export results to CSV
python -m experiment_a.run_all_datasets --unified_judge --output results.csv

# Run individual datasets
python -m experiment_a.swebench.train_evaluate
python -m experiment_a.swebench_pro.train_evaluate
python -m experiment_a.gso.train_evaluate --exclude_unsolved
python -m experiment_a.terminalbench.train_evaluate

# Dry run to check config
python -m experiment_a.swebench.train_evaluate --dry_run
```

## Results (2026-01-24)

### Summary Table (Unified Judge Features)

Run with: `python -m experiment_a.run_all_datasets --unified_judge`

| Dataset | Oracle | Grouped Ridge | Stacked (Emb→LLM) | Embedding | LLM Judge | Baseline |
|---------|--------|---------------|-------------------|-----------|-----------|----------|
| SWE-bench Verified | 0.9441 | 0.8280 | **0.8296** | 0.8230 | 0.8163 | 0.7146 |
| GSO | 0.8516 | **0.7496** | 0.7367 | 0.7332 | 0.7333 | 0.7262 |
| TerminalBench | 0.8995 | **0.8006** | 0.7994 | 0.7905 | 0.7700 | 0.7076 |
| SWE-bench Pro | 0.9183 | 0.7443 | **0.7444** | 0.7366 | 0.7212 | 0.6567 |

**Key findings**:
- **Grouped Ridge vs Embedding**: Grouped Ridge consistently outperforms Embedding alone (+0.5% to +1.6%)
- **Grouped Ridge vs Stacked**: Mixed results - Stacked wins on SWE-bench Verified/Pro by tiny margins (0.1-0.2%), Grouped Ridge wins on GSO/TerminalBench (+1.0-1.3%)
- **Combining features helps**: All combination methods (Grouped Ridge, Stacked) outperform single sources

### SWE-bench Verified (5-Fold Cross-Validation)

**Data**: 500 tasks, 131 agents

| Method | Mean AUC | Std |
|--------|----------|-----|
| Oracle (true b) | 0.9441 | 0.0085 |
| Stacked (Emb → LLM) | 0.8296 | 0.0175 |
| Grouped Ridge (Emb + LLM) | 0.8280 | 0.0185 |
| Stacked (LLM → Emb) | 0.8242 | 0.0155 |
| Embedding | 0.8230 | 0.0193 |
| LLM Judge | 0.8163 | 0.0093 |
| Constant (mean b) | 0.7146 | 0.0083 |

**Note**: On this large dataset, both combination methods work well. Stacked has a slight edge (+0.16%) over Grouped Ridge.

### SWE-bench Pro (5-Fold Cross-Validation)

**Data**: 730 tasks, 14 agents

| Method | Mean AUC | Std |
|--------|----------|-----|
| Oracle (true b) | 0.9183 | 0.0074 |
| Stacked (Emb → LLM) | 0.7444 | 0.0295 |
| Grouped Ridge (Emb + LLM) | 0.7443 | 0.0261 |
| Stacked (LLM → Emb) | 0.7434 | 0.0235 |
| Embedding | 0.7366 | 0.0281 |
| LLM Judge | 0.7212 | 0.0224 |
| Constant (mean b) | 0.6567 | 0.0072 |

**Note**: SWE-bench Pro shows lower predictor AUCs (~0.73-0.75) compared to SWE-bench Verified (~0.82-0.83). This may be due to having fewer agents (14 vs 131) for IRT training.

### GSO (5-Fold Cross-Validation)

**Data**: 57 tasks (excluding zero-solve), 14 agents (performance optimization benchmark)

| Method | Mean AUC | Std |
|--------|----------|-----|
| Oracle (true b) | 0.8516 | 0.0453 |
| Grouped Ridge (Emb + LLM) | 0.7496 | 0.0626 |
| Stacked (Emb → LLM) | 0.7367 | 0.0659 |
| Stacked (LLM → Emb) | 0.7351 | 0.0651 |
| LLM Judge | 0.7333 | 0.0783 |
| Embedding | 0.7332 | 0.0603 |
| Constant (mean b) | 0.7262 | 0.0709 |

**Note**: GSO uses `--exclude_unsolved` to match Daria's setup (excluding 45 zero-solve tasks). On this dataset, Grouped Ridge significantly outperforms Stacked (+1.3%).

### TerminalBench (5-Fold Cross-Validation)

TerminalBench supports two data modes:
- **Binomial** (default): Models k successes out of 5 trials per agent-task pair
- **Binary** (`--binary`): Collapses to any success = 1 (single observation per pair)

#### Binomial Mode (Default)

**Data**: 88 tasks, 83 agents, 5 trials each

| Method | Mean AUC | Std | Pass Rate MSE |
|--------|----------|-----|---------------|
| Oracle (true b) | 0.8995 | 0.0224 | 0.0533 |
| Grouped Ridge (Emb + LLM) | 0.8103 | 0.0302 | 0.1076 |
| Embedding | 0.7905 | 0.0172 | 0.1188 |
| LLM Judge | 0.7663 | 0.0165 | 0.1307 |
| Constant (mean b) | 0.7076 | 0.0172 | 0.1510 |
| Agent-only | 0.7078 | 0.0174 | 0.1423 |

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
| `feature_predictor.py` | `FeatureBasedPredictor`, `GroupedRidgePredictor`, `StackedResidualPredictor` |
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

### Stacked Residual (Emb → LLM)

Two-stage predictor where the second model corrects errors from the first:

```
Stage 1: β̂_base = Ridge(embeddings)           # Base prediction from embeddings
Stage 2: β̂_residual = Ridge(llm_features)     # Predict residuals (β_true - β̂_base)
Final:   β̂ = β̂_base + β̂_residual              # Combined prediction
```

Key differences from Grouped Ridge:
- **Sequential, not joint**: LLM features specifically learn to correct embedding errors
- **No feature space competition**: Each model operates on its own feature space
- **Works best when sources are complementary**: Shows improvement on GSO (+0.4% over Embedding alone) but not on SWE-bench

**When to use**: Stacked (Emb → LLM) is recommended for smaller datasets (like GSO) where it outperforms Grouped Ridge. For larger datasets (like SWE-bench), Grouped Ridge performs slightly better.

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

### Unified Features (Experimental)

A standardized 9-feature set is available for fair cross-dataset comparison. All datasets share 8 core features with one dataset-specific feature:

**Core features (all datasets)**:
- solution_hint (0-3), problem_clarity (1-5), solution_complexity (1-5)
- domain_knowledge_required (1-5), logical_reasoning_required (1-5), atypicality (1-5)
- verification_difficulty (1-5), standard_pattern_available (0-1)

**Dataset-specific**:
- Code datasets (SWE-bench, SWE-bench Pro, GSO): `integration_complexity` (1-5)
- TerminalBench: `tooling_complexity` (1-5)

**Unified feature paths**:
- `chris_output/llm_judge_features/swebench_unified/llm_judge_features.csv`
- `chris_output/llm_judge_features/swebench_pro_unified/llm_judge_features.csv`
- `chris_output/llm_judge_features/terminalbench_unified/llm_judge_features.csv`
- `chris_output/llm_judge_features/gso_unified/llm_judge_features.csv`

To use unified features:
```bash
python -m experiment_a.swebench.train_evaluate --llm_judge_features_path chris_output/llm_judge_features/swebench_unified/llm_judge_features.csv
```

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

## Stacked Predictor Coefficient Analysis

Analyze LLM judge feature coefficients from the stacked predictor across all datasets:

```bash
# Run analysis across all 4 datasets
python -m experiment_a.analyze_stacked_coefficients

# Run for single dataset (faster for testing)
python -m experiment_a.analyze_stacked_coefficients --dataset swebench

# Custom output path
python -m experiment_a.analyze_stacked_coefficients --output_path my_results.json
```

This script:
1. Runs 5-fold CV with only 6 methods: Oracle, Embedding, LLM Judge, Stacked (Emb→LLM), Constant, Agent-only
2. Uses **unified LLM judge features** for consistent cross-dataset comparison
3. Extracts coefficients from the stacked predictor's residual stage (LLM judge)
4. Computes contribution analysis (embedding % vs LLM judge % of prediction variance)

**Output includes:**
- Per-dataset AUC tables
- LLM judge coefficients ranked by magnitude
- Contribution analysis (train and test sets)
- Cross-dataset feature importance comparison

Results saved to `chris_output/stacked_coefficient_analysis.json`.

### Example Output

```
Feature Importance Ranking (by |coefficient|):
Feature                   SWE  Pro  GSO  Term  Avg
--------------------------------------------------
verification_difficulty     5    7    1    1   3.5
problem_clarity             1    6    6    8   5.2
solution_complexity         8    1    4    9   5.5
...

Contribution Summary (Test Set):
                    SWE      Pro      GSO     Term
--------------------------------------------------
Embedding %       73.1%    69.2%    65.3%    58.8%
LLM Judge %        4.9%    11.9%    16.4%    17.4%
```

## Standalone LLM Judge Coefficient Analysis

Compare LLM judge coefficients between **standalone** (direct prediction) and **residual** (error correction) forms:

```bash
# Run analysis across all 4 datasets
python -m experiment_a.analyze_llm_standalone_coefficients

# Run for single dataset (faster for testing)
python -m experiment_a.analyze_llm_standalone_coefficients --dataset swebench
```

This script:
1. Extracts coefficients from standalone LLM Judge Ridge predictor
2. Compares rankings with residual form (from stacked predictor)
3. Shows which features matter more for direct prediction vs. error correction

**Key differences:**
- **Standalone**: LLM Judge directly predicts task difficulty (β)
- **Residual** (stacked): LLM Judge predicts errors from embedding predictions

Results saved to `chris_output/llm_standalone_coefficient_analysis.json`.

### Example Output

```
Feature Importance Ranking (by |coefficient|):
                             ---- Standalone ----     ---- Residual ----
Feature                    SWE  Pro  GSO Term  Avg    SWE  Pro  GSO Term  Avg
-----------------------------------------------------------------------------
verification_difficulty      7    3    1    1   3.0     5    7    1    1   3.5
solution_complexity          3    2    3    7   3.8     8    1    4    9   5.5
...

Features with largest ranking changes:
Higher in Standalone: log_lines_changed (-4.2), solution_complexity (-1.8)
Higher in Residual:   problem_clarity (+2.8), num_hunks (+2.2)
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

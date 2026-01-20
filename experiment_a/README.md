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

# Run with default settings (uses pre-configured embeddings and LLM judge features)
python -m experiment_a.train_evaluate

# Run on TerminalBench (binomial responses)
python -m experiment_a_terminalbench.train_evaluate

# Dry run to check config
python -m experiment_a.train_evaluate --dry_run
```

## Results (2026-01-20)

### SWE-bench Verified (5-Fold Cross-Validation)

**Data**: 500 tasks, 130 agents

| Method | Mean AUC | Std |
|--------|----------|-----|
| Oracle (true b) | 0.9441 | 0.0045 |
| Embedding | 0.8269 | 0.0070 |
| LLM Judge | 0.8227 | 0.0118 |
| Constant (mean b) | 0.7149 | 0.0108 |
| Agent-only | 0.7150 | 0.0109 |

### TerminalBench (5-Fold Cross-Validation)

**Data**: 89 tasks, 83 agents (binomial responses)

| Method | Mean AUC | Std | Pass Rate MSE |
|--------|----------|-----|---------------|
| Oracle (true b) | 0.9037 | 0.0109 | 0.0540 |
| LLM Judge | 0.7841 | 0.0278 | 0.1212 |
| Embedding | 0.7829 | 0.0402 | 0.1240 |
| Constant (mean b) | 0.7036 | 0.0123 | 0.1490 |
| Agent-only | 0.7039 | 0.0125 | 0.1404 |

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

The experiment uses a unified framework in `experiment_a_common/` that supports both SWE-bench (binary) and TerminalBench (binomial) datasets.

### Shared Infrastructure (`experiment_a_common/`)

| File | Purpose |
|------|---------|
| `feature_source.py` | `TaskFeatureSource` ABC with `EmbeddingFeatureSource`, `CSVFeatureSource` |
| `feature_predictor.py` | `FeatureBasedPredictor` (StandardScaler → RidgeCV) |
| `predictor_base.py` | `DifficultyPredictorBase` ABC, `ConstantPredictor`, `GroundTruthPredictor` |
| `dataset.py` | `ExperimentData` ABC with `BinaryExperimentData`, `BinomialExperimentData` |
| `evaluator.py` | `compute_auc()`, `PredictorConfig`, evaluation pipeline |
| `cross_validation.py` | k-fold CV utilities |
| `pipeline.py` | `ExperimentSpec`, `run_experiment_main()` orchestration |

### SWE-bench Specific (`experiment_a/`)

| File | Purpose |
|------|---------|
| `train_evaluate.py` | Main entry point (thin wrapper, ~40 LOC) |
| `config.py` | `ExperimentAConfig` with default paths |
| `generate_embeddings.py` | Generate task embeddings |
| `compute_llm_judge_features.py` | Extract LLM semantic features |

### TerminalBench Specific (`experiment_a_terminalbench/`)

| File | Purpose |
|------|---------|
| `train_evaluate.py` | Main entry with `is_binomial=True` |
| `config.py` | `TerminalBenchConfig` with TerminalBench paths |
| `data_loader.py` | Load task data from terminal-bench repo |

## Feature Sources

### 1. Embeddings (DeepSeek-R1-Distill-Qwen-32B)

Pre-computed embeddings are configured by default:
- SWE-bench: `chris_output/experiment_a/embeddings/embeddings__deepseek-ai__DeepSeek-R1-Distill-Qwen-32B__merged.npz`
- TerminalBench: `chris_output/experiment_a_terminalbench/embeddings/embeddings__deepseek-ai__DeepSeek-R1-Distill-Qwen-32B__pool-lasttoken__maxlen8192.npz`

To generate new embeddings:
```bash
python -m experiment_a.generate_embeddings --backbone "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
```

### 2. LLM Judge Features

Semantic features extracted via LLM structured output:

**SWE-bench (9 features)**:
- fix_in_description, problem_clarity, error_message_provided, reproduction_steps
- fix_locality, domain_knowledge_required, fix_complexity, logical_reasoning_required, atypicality

**TerminalBench (4 pre-selected features)**:
- task_clarity, domain_knowledge_required, task_complexity, atypicality

To extract features:
```bash
python -m experiment_a.compute_llm_judge_features --dry_run
python -m experiment_a.compute_llm_judge_features
```

## Data Paths

### SWE-bench

| File | Purpose |
|------|---------|
| `clean_data/swebench_verified_20251120_full/1d_1pl/abilities.csv` | Oracle IRT abilities |
| `clean_data/swebench_verified_20251120_full/1d_1pl/items.csv` | Oracle IRT difficulties |
| `clean_data/swebench_verified/swebench_verified_20251120_full.jsonl` | Response matrix |
| `chris_output/experiment_a/irt_splits/` | Fold-specific IRT models (cached) |

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
--single_holdout      Use single 20% holdout instead of CV
--test_fraction       Fraction of tasks for test set (default: 0.2)
--split_seed          Random seed for train/test split (default: 0)
--embeddings_path     Override default embeddings path
--llm_judge_features_path  Override default LLM features path
--output_dir          Output directory
--dry_run             Show configuration without running
--exclude_unsolved    Exclude tasks no agent solved
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

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

# Run ALL datasets and get a summary table (uses unified judge features by default)
python -m experiment_a.run_all_datasets

# Run specific datasets only
python -m experiment_a.run_all_datasets --datasets gso terminalbench

# Use dataset-specific features instead of unified
python -m experiment_a.run_all_datasets --no-unified_judge

# Export results to CSV
python -m experiment_a.run_all_datasets --output results.csv

# Run Feature-IRT variant (joint training instead of Ridge)
python -m experiment_a.run_feature_irt
python -m experiment_a.run_feature_irt --datasets swebench gso
python -m experiment_a.run_feature_irt --sequential

# Run individual datasets
python -m experiment_a.swebench.train_evaluate
python -m experiment_a.swebench_pro.train_evaluate
python -m experiment_a.gso.train_evaluate --exclude_unsolved
python -m experiment_a.terminalbench.train_evaluate

# Dry run to check config
python -m experiment_a.swebench.train_evaluate --dry_run
```

## Results (2026-01-30)

### Summary Table (Unified Judge + Auditor + Test, Default)

Run with: `python -m experiment_a.run_all_datasets`

| Dataset | Oracle | Grouped Ridge (Emb+LLM) | LLM Judge | Embedding | Baseline |
|---------|--------|-------------------------|-----------|-----------|----------|
| SWE-bench Verified | 0.9441 | **0.8415** | 0.8336 | 0.8230 | 0.7146 |
| GSO | 0.8516 | **0.7496** | 0.7333 | 0.7332 | 0.7262 |
| TerminalBench | 0.8995 | **0.8006** | 0.7700 | 0.7905 | 0.7076 |
| SWE-bench Pro | 0.9183 | **0.7443** | 0.7212 | 0.7366 | 0.6567 |

**Key findings**:
- **Grouped Ridge (Emb+LLM) is best**: Outperforms single sources on all datasets
- **Test patch features help**: Adding 3 test quality features improved SWE-bench Grouped Ridge from 0.8331 → 0.8415 (+1.0%)
- **Combining features helps**: Grouped Ridge outperforms single sources (+0.5% to +1.8%)

### Feature-IRT Results (Joint Training)

Run with: `python -m experiment_a.run_feature_irt`

| Dataset | Oracle | Feature-IRT (Emb+LLM) | Feature-IRT (LLM) | Feature-IRT (Emb) | Baseline |
|---------|--------|----------------------|-------------------|-------------------|----------|
| SWE-bench Verified | 0.9441 | **0.8394** | 0.8371 | 0.8237 | 0.7146 |
| GSO | 0.8516 | 0.7454 | 0.7345 | **0.7470** | 0.7262 |
| TerminalBench | 0.8995 | 0.7857 | 0.7742 | **0.7976** | 0.7076 |
| SWE-bench Pro | 0.9183 | 0.7240 | 0.7114 | **0.7560** | 0.6567 |

**Key findings**:
- **Feature-IRT performs similarly to Ridge** in Experiment A (task holdout) because it must generalize to unseen test tasks using only feature weights
- **Embedding alone often outperforms combined** in Feature-IRT, suggesting the joint optimization may overfit to LLM features on some datasets
- **Per-source regularization**: Feature-IRT uses per-source L2 grids (Emb: [100, 1000, 10000], LLM: [0.01, 0.1, 1, 10]) with internal CV for selection

### SWE-bench Verified (5-Fold Cross-Validation)

**Data**: 500 tasks, 131 agents

| Method | Mean AUC | Std |
|--------|----------|-----|
| Oracle (true b) | 0.9441 | 0.0085 |
| Grouped Ridge (Emb + LLM) | 0.8415 | 0.0197 |
| Stacked (Emb → LLM) | 0.8383 | 0.0188 |
| LLM Judge (15 features) | 0.8336 | 0.0211 |
| Embedding | 0.8230 | 0.0193 |
| Constant (mean b) | 0.7146 | 0.0083 |

**Note**: LLM Judge includes 9 unified semantic features + 3 auditor features + 3 test quality features = 15 features total. See ablation study below for feature contribution breakdown.

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
| Stacked (Emb → LLM) | 0.7507 | 0.0677 |
| Grouped Ridge (Emb + LLM) | 0.7496 | 0.0626 |
| Stacked (LLM → Emb) | 0.7433 | 0.0706 |
| LLM Judge | 0.7333 | 0.0783 |
| Embedding | 0.7332 | 0.0603 |
| Constant (mean b) | 0.7262 | 0.0709 |

**Note**: GSO uses `--exclude_unsolved` to match Daria's setup (excluding 45 zero-solve tasks). Stacked slightly outperforms Grouped Ridge (+0.1%).

### TerminalBench (5-Fold Cross-Validation)

TerminalBench supports two data modes:
- **Binomial** (default): Models k successes out of 5 trials per agent-task pair
- **Binary** (`--binary`): Collapses to any success = 1 (single observation per pair)

#### Binomial Mode (Default)

**Data**: 88 tasks, 83 agents, 5 trials each

| Method | Mean AUC | Std | Pass Rate MSE |
|--------|----------|-----|---------------|
| Oracle (true b) | 0.8995 | 0.0224 | 0.0533 |
| Stacked (Emb → LLM) | 0.8044 | 0.0285 | 0.1121 |
| Stacked (LLM → Emb) | 0.8043 | 0.0239 | 0.1144 |
| Grouped Ridge (Emb + LLM) | 0.8006 | 0.0210 | 0.1143 |
| Embedding | 0.7905 | 0.0172 | 0.1188 |
| LLM Judge | 0.7700 | 0.0165 | 0.1307 |
| Constant (mean b) | 0.7076 | 0.0172 | 0.1510 |

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
| `feature_predictor.py` | `FeatureBasedPredictor`, `GroupedRidgePredictor`, `StackedResidualPredictor`, `DecisionTreePredictor`, `RandomForestPredictor` |
| `predictor_base.py` | `DifficultyPredictorBase` ABC |
| `evaluator.py` | `compute_auc()`, `compute_irt_probability()` |

**`experiment_a/shared/`** - Experiment A orchestration:

| File | Purpose |
|------|---------|
| `pipeline.py` | `ExperimentSpec`, `CVPredictorConfig`, `run_experiment_main()` |
| `cross_validation.py` | `CVPredictor` protocol, `run_cv()`, `k_fold_split_tasks()` |
| `baselines.py` | `OraclePredictor`, `ConstantPredictor`, `AgentOnlyPredictor`, `DifficultyPredictorAdapter`, `FeatureIRTCVPredictor` (with per-source L2 regularization) |
| `mlp_predictor.py` | `MLPPredictor` - direct success prediction via neural network |

### Multi-Dataset Scripts (`experiment_a/`)

| File | Purpose |
|------|---------|
| `run_all_datasets.py` | Run Ridge-based predictors on all datasets (default) |
| `run_feature_irt.py` | Run Feature-IRT (joint training) on all datasets |

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
- Uses internal 3-fold CV to select L2 weights (similar to RidgeCV)

**Per-source regularization**: When using `GroupedFeatureSource` (combined embeddings + LLM), applies different L2 penalties per source:
- High-dim embeddings: `l2_emb ∈ [100, 1000, 10000]` (strong regularization)
- Low-dim LLM features: `l2_llm ∈ [0.01, 0.1, 1, 10]` (weak regularization)
- Loss: `weight_reg = l2_emb * ||w_emb||² + l2_llm * ||w_llm||²`
- Internal CV selects best (l2_emb, l2_llm) combination

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

### Decision Tree and Random Forest (Experimental)

Tree-based methods for difficulty prediction. Both train on LLM judge features to predict IRT difficulty.

**Note:** These methods are **off by default** (`--no-trees`) since they don't consistently outperform Ridge regression. Use `--trees` to include them.

**Decision Tree** (`LLM Judge (Tree)`):
- Uses cross-validated hyperparameter search over `max_depth` values: [3, 5, 7, 10]
- Simple, interpretable model but prone to overfitting

**Random Forest** (`LLM Judge (RF)`):
- Ensemble of 50 decision trees with fixed hyperparameters (no grid search for speed)
- `max_depth=5`, `min_samples_split=5`, `min_samples_leaf=2`
- More robust than single Decision Tree due to bagging and averaging
- Uses `oob_score=True` for out-of-bag generalization estimate

**Performance comparison** (SWE-bench Verified):
| Method | Mean AUC | Std |
|--------|----------|-----|
| LLM Judge (Ridge) | 0.8163 | 0.0093 |
| LLM Judge (RF) | 0.8145 | 0.0089 |
| LLM Judge (Tree) | 0.7961 | 0.0151 |

Random Forest performs comparably to Ridge regression while being more robust than a single Decision Tree.

### MLP Predictor (Direct Success Prediction)

Unlike the difficulty-based predictors above, the MLP directly predicts P(success) for (agent, task) pairs without going through IRT difficulty:

```
Input: [agent_one_hot | task_features]
  → Linear(hidden_size) → ReLU → Linear(1) → Sigmoid
Output: P(success) ∈ [0, 1]
```

**Key differences from difficulty-based methods:**
- **No IRT formula**: Directly learns P(success), not task difficulty β
- **Agent-aware**: Uses one-hot agent encoding, so the model sees which agent is being evaluated
- **End-to-end**: Trained with binary cross-entropy on actual success/failure labels

**Architecture:**
- Input: `[agent_one_hot (n_agents) | scaled_task_features (feature_dim)]`
- Hidden layer: 64 units (32 for low-dim features like LLM Judge)
- Training: Adam with weight_decay=0.01 (L2 regularization), 200 epochs

**Three variants:**
- `MLP (Embedding)`: Uses task embeddings only
- `MLP (LLM Judge)`: Uses LLM judge features only
- `MLP (Grouped)`: Uses combined embedding + LLM judge features

**Training loss tracking:** The MLP predictor tracks loss per iteration for convergence verification. Use `plot_mlp_training_loss.py` to visualize:

```bash
python -m experiment_a.plot_mlp_training_loss --losses_json path/to/losses.json
```

## Feature Sources

### 1. Embeddings (DeepSeek-R1-Distill-Qwen-32B)

Pre-computed embeddings are stored in the unified `embeddings/` directory at the repository root:
- SWE-bench Verified: `embeddings/embeddings__...__princeton-nlp_SWE-bench_Verified__test__maxlen8192.npz`
- SWE-bench Pro: `embeddings/embeddings__...__ScaleAI_SWE-bench_Pro__test__maxlen8192.npz`
- TerminalBench: `embeddings/embeddings__...__idnorm_instance-v2__json_terminal_bench_tasks.jsonl__test__maxlen8192.npz`
- GSO: `embeddings/embeddings__...__gso-bench_gso__test__maxlen8192.npz`

Note: The `embeddings/` directory is git-ignored. Embeddings must be generated or copied separately.

To generate new embeddings:
```bash
python -m experiment_a.generate_embeddings --backbone "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
```

### 2. Environment Features (NEW)

Deterministic features extracted from SWE-bench task environments by running bash commands inside Docker containers. These features capture repository structure, codebase statistics, and git history.

**Location**: `experiment_a/env_features/`

**34 features extracted**:
- **File system**: `env_total_files`, `env_total_dirs`, `env_dir_depth_max`
- **Python files**: `env_python_files`, `env_python_loc`, `env_init_files`
- **Test infrastructure**: `env_test_files`, `env_test_dirs`, `env_conftest_files`, `env_has_pytest_ini`, `env_has_tox`
- **Build config**: `env_has_setup_py`, `env_has_setup_cfg`, `env_has_pyproject`, `env_has_makefile`, `env_has_dockerfile`
- **Dependencies**: `env_requirements_count`, `env_has_requirements`
- **Git stats**: `env_git_commits_total`, `env_git_branches`, `env_git_tags`, `env_git_contributors`
- **Documentation**: `env_doc_files`, `env_has_readme`, `env_has_docs_dir`, `env_sphinx_conf`
- **Code complexity proxies**: `env_class_count`, `env_function_count`, `env_import_count`, `env_todo_count`
- **Other files**: `env_json_files`, `env_yaml_files`, `env_config_files`, `env_shell_scripts`

**Usage**:
```bash
# Run full extraction with batching (processes 10 tasks at a time, cleans up Docker between batches)
python -m experiment_a.env_features.run_extraction \
    --batch_size 10 \
    --max_connections 10 \
    --output_dir chris_output/env_features/swebench_verified/

# Resume interrupted extraction (automatically skips completed samples)
python -m experiment_a.env_features.run_extraction --output_dir chris_output/env_features/swebench_verified/

# Just aggregate existing logs to CSV
python -m experiment_a.env_features.run_extraction --aggregate_only
```

**Output**: `chris_output/env_features/swebench_verified/env_features.csv`

**Architecture**:
- Uses Inspect AI framework with deterministic solver (no LLM calls)
- Each task runs in its own Docker container (`swebench/sweb.eval.x86_64.{instance_id}`)
- Batched processing with Docker cleanup between batches to prevent memory overflow
- Supports resuming interrupted runs

See [experiment_a/env_features/README.md](env_features/README.md) for detailed documentation.

### 3. Auditor Agent Features (NEW)

Semantic features extracted by an LLM auditor agent (Claude Opus 4.5) that explores SWE-bench task environments via bash shell access. These features capture aspects that require environment interaction to assess.

**Location**: `experiment_a/auditor_agent/`

**3 features extracted (1-5 scale)**:
- **entry_point_clarity**: How easy is it to find where the bug manifests? (1=unclear → 5=obvious from problem statement)
- **change_blast_radius**: How many components would be affected by changes? (1=isolated → 5=highly coupled)
- **fix_localization**: How easily can the fix location be determined? (1=requires deep exploration → 5=immediately clear)

**Correlation with IRT difficulty (β)**:
- entry_point_clarity: -0.392 (clearer entry point → easier task)
- fix_localization: -0.389 (easier to localize → easier task)
- change_blast_radius: +0.351 (larger blast radius → harder task)

**Usage**:
```bash
# Run auditor on all SWE-bench Verified tasks
python -m experiment_a.auditor_agent.run_auditor --output_dir chris_output/auditor_pilot/

# Parse outputs to CSV
python -m experiment_a.auditor_agent.parse_outputs
```

**Output**: `chris_output/auditor_pilot/v3_features_top3.csv`

**Integration**: Auditor features are included in the default LLM Judge feature set via coefficient-based selection.

See [experiment_a/auditor_agent/README.md](auditor_agent/README.md) for detailed documentation.

### 4. LLM Judge Features

Semantic features extracted via LLM structured output:

**SWE-bench Verified (15 features, default)**:

The default feature set is `chris_output/llm_judge_features/swebench_ablation_controlled_v3/4_full_15.csv`, which contains the top 15 features selected by Ridge coefficient magnitude from a pool of 23 features:
- **Problem (7)**: solution_hint, problem_clarity, domain_knowledge_required, logical_reasoning_required, atypicality, verification_difficulty, standard_pattern_available
- **Problem Extended (8)**: error_specificity, reproduction_clarity, expected_behavior_clarity, debugging_complexity, codebase_scope, information_completeness, similar_issue_likelihood, backwards_compatibility_risk
- **Auditor (3)**: entry_point_clarity, change_blast_radius, fix_localization
- **Test Quality (3)**: test_comprehensiveness, test_assertion_complexity, test_edge_case_coverage
- **Solution (2)**: solution_complexity, integration_complexity

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

**Unified feature paths** (9 features: 8 core + 1 dataset-specific):
- `chris_output/llm_judge_features/swebench_unified/llm_judge_features.csv`
- `chris_output/llm_judge_features/swebench_pro_unified/llm_judge_features.csv`
- `chris_output/llm_judge_features/terminalbench_unified/llm_judge_features.csv`
- `chris_output/llm_judge_features/gso_unified/llm_judge_features.csv`

**Core-only feature paths** (8 features: exactly the same across all datasets):
- `chris_output/llm_judge_features/swebench_unified_core/llm_judge_features.csv`
- `chris_output/llm_judge_features/swebench_pro_unified_core/llm_judge_features.csv`
- `chris_output/llm_judge_features/terminalbench_unified_core/llm_judge_features.csv`
- `chris_output/llm_judge_features/gso_unified_core/llm_judge_features.csv`

Usage:
```bash
# run_all_datasets uses unified features (9 features) by default
python -m experiment_a.run_all_datasets

# Use core-only features (8 features, identical across datasets)
python -m experiment_a.run_all_datasets --unified_judge_suffix _core

# For individual dataset scripts, specify the path explicitly
python -m experiment_a.swebench.train_evaluate --llm_judge_features_path chris_output/llm_judge_features/swebench_unified/llm_judge_features.csv
```

#### Unified vs Dataset-Specific Features Comparison

Both approaches yield comparable results (within ~0.005 AUC noise). Dataset-specific features have a slight edge on some benchmarks:

| Dataset | Grouped Ridge (Unified) | Grouped Ridge (Default) | Stacked (Unified) | Stacked (Default) |
|---------|------------------------|------------------------|-------------------|-------------------|
| SWE-bench Verified | 0.8280 | **0.8309** | 0.8296 | 0.8278 |
| GSO | 0.7496 | **0.7500** | 0.7507 | **0.7509** |
| TerminalBench | 0.8006 | **0.8062** | 0.8044 | 0.8034 |
| SWE-bench Pro | 0.7443 | **0.7477** | 0.7444 | **0.7493** |

**Recommendation**: Unified features are now the default for consistency across datasets. Use `--no-unified_judge` if you need dataset-specific features for slightly better per-dataset performance.

### Ablation Study: Solution Information Contribution

To measure how much the solution patch contributes to feature quality, we ran ablation experiments removing solution information from both embeddings and LLM judge features.

#### Embedding Ablation (SWE-bench Verified)

Embeddings are generated with a prompt containing `question_statement + solution + instruction`. The "no solution" variant removes the solution from the prompt.

| Embedding Variant | Source Alone | Grouped Ridge |
|-------------------|--------------|---------------|
| Without Solution | 0.7506 | 0.7800 |
| **With Solution** | **0.8230** | **0.8436** |

**Note**: Grouped Ridge pairs each embedding variant with the corresponding LLM features:
- No-solution embedding: paired with Problem Only LLM features
- With-solution embedding: paired with Full LLM features

**Cross-dataset embedding ablation (Source Alone):**

| Dataset | With Solution | No Solution | Δ AUC |
|---------|---------------|-------------|-------|
| SWE-bench Verified | 0.823 | 0.751 | -0.072 (-8.7%) |
| SWE-bench Pro | 0.747 | 0.728 | -0.019 (-2.5%) |
| TerminalBench | 0.804 | 0.784 | -0.020 (-2.5%) |
| GSO | 0.728 | 0.656 | -0.072 (-9.9%) |

**Key finding**: Solution information in the embedding prompt contributes +2.5% to +9.9% AUC improvement, with the largest impact on GSO and SWE-bench Verified.

#### LLM Judge Ablation (SWE-bench Verified)

To measure the contribution of different information sources, we ran ablation experiments progressively adding features from different affordances. Feature count is held constant at 15 via Ridge coefficient-based selection to isolate the value of each information source:

| Method | # Features | LLM Judge AUC | Grouped Ridge AUC |
|--------|------------|---------------|-------------------|
| Problem Only | 15 | 0.7821 ± 0.0164 | 0.7800 ± 0.0241 |
| Problem + Auditor | 15 | 0.8015 ± 0.0167 | 0.7967 ± 0.0240 |
| Problem + Auditor + Test | 15 | 0.8225 ± 0.0230 | 0.8174 ± 0.0235 |
| **Full** | 15 | **0.8363 ± 0.0205** | **0.8436 ± 0.0216** |

**Note**: Grouped Ridge pairs each LLM ablation level with the corresponding embedding variant:
- Problem Only, +Auditor, +Test: paired with no-solution embedding (neither has solution access)
- Full: paired with with-solution embedding (both have solution access)

**Information sources**:
- **Problem Only**: Features derived from the problem statement alone (7 original + 8 extended)
- **+ Auditor**: 3 features requiring shell access to task environment (entry_point_clarity, change_blast_radius, fix_localization)
- **+ Test**: 3 features analyzing the test patch diff (test_comprehensiveness, test_assertion_complexity, test_edge_case_coverage)
- **+ Solution (Full)**: 2 features requiring the gold solution patch (solution_complexity, integration_complexity)

**Feature selection method**: Ridge regression on all available features → rank by |coefficient| → keep top 15.

**Key finding**: Even with the same number of features, adding affordances provides significant improvement in LLM Judge AUC:
- +Auditor: +1.9% (0.782 → 0.801)
- +Test: +2.1% (0.801 → 0.823)
- +Solution: +1.4% (0.823 → 0.836)

**Note**: Test features are extracted WITHOUT access to the solution patch, ensuring clean ablation. The Grouped Ridge column uses matching embedding variants (no-solution for levels 1-3, with-solution for Full).

All affordance-specific features (auditor, test, solution) ranked in the top 15 by coefficient magnitude, confirming they capture information not available from the problem statement alone.

**Full model coefficients (23 features, ranked by |coefficient|)**:

| Rank | Feature | Coefficient | Source |
|------|---------|-------------|--------|
| 1 | solution_complexity | +0.566 | Solution |
| 2 | integration_complexity | +0.450 | Solution |
| 3 | test_comprehensiveness | +0.386 | Test |
| 4 | test_edge_case_coverage | +0.334 | Test |
| 5 | information_completeness | +0.332 | Problem (ext) |
| 6 | fix_localization | -0.311 | Auditor |
| 7 | entry_point_clarity | -0.303 | Auditor |
| 8 | atypicality | +0.238 | Problem (orig) |
| 9 | similar_issue_likelihood | +0.189 | Problem (ext) |
| 10 | codebase_scope | +0.189 | Problem (ext) |
| 11 | problem_clarity | -0.187 | Problem (orig) |
| 12 | change_blast_radius | +0.159 | Auditor |
| 13 | error_specificity | -0.140 | Problem (ext) |
| 14 | solution_hint | -0.104 | Problem (orig) |
| 15 | debugging_complexity | +0.099 | Problem (ext) |

To run the ablation study:
```bash
# Run information source ablation (SWE-bench Verified)
python -m experiment_a.run_information_ablation

# Rebuild ablation CSVs (if feature sources change)
python -m experiment_a.run_information_ablation --rebuild_csvs
```

To run the ablation across all datasets:
```bash
# Run all datasets with judge ablation
python -m experiment_a.run_all_datasets --judge_ablation --sequential
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
--unified_judge_suffix    Suffix to append to unified judge paths (e.g., '_core')
--output_dir          Output directory
--dry_run             Show configuration without running
--exclude_unsolved    Exclude tasks no agent solved
--include_feature_irt Include Feature-IRT joint learning methods (off by default)
--mlp / --no-mlp      Include/exclude MLP predictors (default: --no-mlp). Use --mlp to include PyTorch-based MLP predictors.
--trees / --no-trees  Include/exclude tree-based predictors (default: --no-trees). Use --trees to include Decision Tree and Random Forest.
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

# Use custom LLM features (e.g., ablation CSV) with LaTeX and plot output
python -m experiment_a.analyze_llm_standalone_coefficients --dataset swebench \
    --llm_path chris_output/llm_judge_features/swebench_ablation_controlled_v3/4_full_15.csv \
    --latex --plot
```

This script:
1. Extracts coefficients from standalone LLM Judge Ridge predictor
2. Compares rankings with residual form (from stacked predictor)
3. Shows which features matter more for direct prediction vs. error correction
4. Optionally generates LaTeX tables (`--latex`) and bar graphs (`--plot`) by feature source

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

## Grouped Ridge Coefficient Analysis

Analyze LLM judge feature coefficients from the **Grouped Ridge** predictor (joint embedding + LLM fitting):

```bash
# Run analysis across all 4 datasets
python -m experiment_a.analyze_grouped_ridge_coefficients

# Run for single dataset (faster for testing)
python -m experiment_a.analyze_grouped_ridge_coefficients --dataset swebench
```

This script:
1. Uses the existing CV infrastructure with a diagnostics callback
2. Extracts coefficients from `GroupedRidgePredictor.get_detailed_diagnostics()`
3. Reports contribution analysis (L2 norm squared) and selected alphas per source

**Contribution is measured by L2 norm squared:**
```
Embedding % = ||w_emb||² / (||w_emb||² + ||w_llm||²)
```

Results saved to `chris_output/grouped_ridge_coefficient_analysis.json`.

### Results (2026-01-25)

**Feature Importance Ranking (by |coefficient|):**

| Feature | SWE | Pro | GSO | Term | Avg Rank |
|---------|-----|-----|-----|------|----------|
| verification_difficulty | 5 | 4 | 1 | 1 | **2.8** |
| solution_complexity | 3 | 1 | 4 | 8 | **4.0** |
| logical_reasoning_required | 6 | 3 | 2 | 7 | **4.5** |
| atypicality | 4 | 8 | 6 | 4 | 5.5 |
| solution_hint | 12 | 5 | 3 | 6 | 6.5 |
| standard_pattern_available | 8 | 9 | 7 | 2 | 6.5 |

**Contribution Summary (L2 norm squared):**

| Dataset | Embedding % | LLM Judge % | Emb α | LLM α |
|---------|-------------|-------------|-------|-------|
| SWE-bench | 86.8% | 13.2% | 10000 | 640 |
| SWE-bench Pro | 84.4% | 15.6% | 10000 | 1000 |
| GSO | 76.5% | 23.5% | 6400 | 2260 |
| TerminalBench | 91.0% | 9.0% | 800 | 44 |

**Key findings:**
- **verification_difficulty** is the most consistently important LLM feature (avg rank 2.8)
- LLM Judge contributes 9-24% of coefficient magnitude, with GSO showing highest contribution
- Selected alphas vary significantly: higher embedding alpha for larger datasets

## Full Feature-IRT (Experimental - Not for Comparison)

The `--include_feature_irt` flag enables Full Feature-IRT predictors that train on ALL tasks (like Oracle). These were added to **sanity check trajectory features** and verify the Feature-IRT implementation, **not** as proper comparison methods.

**Why these shouldn't be used for comparison:**

1. **Trains on test data**: Unlike other predictors, Full Feature-IRT trains on ALL tasks including test tasks, making it an unfair comparison with methods that only see train tasks.

2. **Matches Oracle by design**: With low residual regularization, the per-task difficulty latents can learn task-specific information, effectively recovering Oracle-level predictions (AUC ~0.944).

3. **Different evaluation setup**: These predictors test whether features add value *given full response data*, which is a different question than Experiment A's goal of predicting difficulty *without* running agents.

**When to use:**
- Verifying feature source implementations work correctly
- Sanity checking new feature sources (e.g., trajectory features)
- Understanding variance decomposition between features and difficulty latents

**Analysis script:**
```bash
# Analyze Feature-IRT decomposition (feature vs residual contributions)
python -m experiment_a.analyze_feature_irt
```

**Do NOT include in results tables or comparison figures.**

## Caches

| Cache | Location | When to Clear |
|-------|----------|---------------|
| **IRT Split Models** | `chris_output/experiment_a/irt_splits/` | When changing split parameters |
| **Embeddings** | `chris_output/experiment_a/embeddings/` | When changing backbone |
| **LLM Judge Features** | `chris_output/experiment_a/llm_judge_features/` | When re-extracting |

## References

- IRT formula: `P = sigmoid(theta - beta)` matches py_irt's 1PL implementation
- [Research Proposal](../chris%20proposal.md) - Section 3.1

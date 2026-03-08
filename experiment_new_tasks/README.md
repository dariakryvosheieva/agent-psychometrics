# Experiment New Tasks: Prior Validation (IRT AUC)

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
python -m experiment_new_tasks.run_all_datasets

# Run specific datasets only
python -m experiment_new_tasks.run_all_datasets --datasets gso terminalbench

# Disable LLM judge features (embedding only)
python -m experiment_new_tasks.run_all_datasets --no-unified_judge

# Export results to CSV
python -m experiment_new_tasks.run_all_datasets --output results.csv

# Run Feature-IRT variant (joint training instead of Ridge)
python -m experiment_new_tasks.run_all_datasets --feature_irt
python -m experiment_new_tasks.run_all_datasets --feature_irt --datasets swebench_verified gso

# Run a single dataset
python -m experiment_new_tasks.run_all_datasets --datasets terminalbench
```

## Results (2026-03-08)

### Summary Table (Default Ridge)

Run with: `python -m experiment_new_tasks.run_all_datasets`

**LLM features**: v7 Opus 4.6, unified 15 features across all datasets (selected from 28 features = 20 judge + 8 auditor via greedy forward selection maximizing cross-dataset mean AUC). See `llm_judge_features/README.md` for details.

| Dataset | Tasks | Agents | Oracle | Grouped Ridge (Emb+LLM) | LLM Judge | Embedding | Baseline |
|---------|-------|--------|--------|-------------------------|-----------|-----------|----------|
| SWE-bench Verified | 500 | 134 | 0.9447 | **0.8427** | 0.8415 | 0.8244 | 0.7175 |
| GSO | 102 | 15 | 0.9139 | **0.7944** | 0.7844 | 0.7581 | 0.7130 |
| TerminalBench | 89 | 112 | 0.9317 | **0.8209** | 0.8059 | 0.8174 | 0.7338 |
| SWE-bench Pro | 730 | 14 | 0.9183 | **0.7632** | 0.7421 | 0.7550 | 0.6558 |

**Key findings**:
- **Grouped Ridge (Emb+LLM) is best**: Outperforms single sources on all datasets
- **v7 Opus 4.6 solution dominates**: Best mean AUC across all 6 provider/info-level combinations tested
- **Unified 15 features**: Same feature set used across all datasets (mean AUC 0.7935 vs 0.7990 with per-dataset top-15, a -0.0055 gap)

### Feature-IRT Results (Joint Training)

Run with: `python -m experiment_new_tasks.run_all_datasets --feature_irt`

| Dataset | Oracle | Feature-IRT (Emb+LLM) | Feature-IRT (LLM) | Feature-IRT (Emb) | Baseline |
|---------|--------|----------------------|-------------------|-------------------|----------|
| SWE-bench Verified | 0.9447 | **0.8389** | 0.8370 | 0.8243 | 0.7174 |
| GSO | 0.9139 | 0.7407 | 0.7149 | **0.7571** | 0.7130 |
| TerminalBench | 0.9317 | — | — | — | 0.7338 |
| SWE-bench Pro | 0.9183 | 0.7236 | 0.7112 | **0.7555** | 0.6567 |

**Key findings**:
- **Feature-IRT performs similarly to Ridge** in Experiment New Tasks (task holdout) because it must generalize to unseen test tasks using only feature weights
- **Embedding alone often outperforms combined** in Feature-IRT (3/4 datasets), suggesting the joint optimization may overfit to LLM features
- **Per-source regularization**: Feature-IRT uses per-source L2 grids (Emb: [100, 1000, 10000], LLM: [0.01, 0.1, 1, 10]) with internal CV for selection

## Evaluation Protocol

1. **Split tasks** (not agents) into train/test sets using k-fold cross-validation (sklearn)
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

The experiment uses a unified CVPredictor protocol that all methods implement, enabling consistent k-fold cross-validation across all datasets.

### CVPredictor Protocol

All methods implement the same interface:

```python
class CVPredictor(Protocol):
    def fit(self, data: ExperimentData, train_task_ids: List[str]) -> None: ...
    def predict_probability(self, data: ExperimentData, agent_id: str, task_id: str) -> float: ...
```

### Shared Infrastructure

**`experiment_new_tasks/`** - Core abstractions shared with Experiment B:

| File | Purpose |
|------|---------|
| `dataset.py` | `ExperimentData`, `load_dataset_for_fold()` |
| `feature_source.py` | `TaskFeatureSource` ABC with `EmbeddingFeatureSource`, `CSVFeatureSource`, `GroupedFeatureSource` |
| `feature_predictor.py` | `DifficultyPredictorBase` ABC, `FeatureBasedPredictor`, `GroupedRidgePredictor` |
| `evaluator.py` | `compute_irt_probability()`, `convert_numpy()` |

**`experiment_new_tasks/`** - Experiment New Tasks orchestration:

| File | Purpose |
|------|---------|
| `config.py` | `ExperimentAConfig`, `DATASET_DEFAULTS` registry |
| `pipeline.py` | `CVPredictorConfig`, `cross_validate_all_predictors()`, `build_cv_predictors()` |
| `cross_validation.py` | `CVPredictor` protocol, `evaluate_predictor_cv()`, `k_fold_split_tasks()` |
| `difficulty_predictors.py` | `OraclePredictor`, `ConstantPredictor`, `DifficultyPredictorAdapter` |
| `feature_irt.py` | `JointTrainingCVPredictor` (with per-source L2 regularization), `feature_irt_predictor_factory()` |
| `coefficient_analysis.py` | `extract_llm_coefficients()`, `print_coefficient_table()`, `save_coefficient_bar_chart()` |

### Multi-Dataset Scripts (`experiment_new_tasks/`)

| File | Purpose |
|------|---------|
| `run_all_datasets.py` | Run all datasets with Ridge (default) or Feature-IRT (`--feature_irt`) |
| `run_information_ablation.py` | Run information level ablation across all datasets (v8 features) |

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
- Uses internal 3-fold CV to select L2 weights (similar to RidgeCV)

**Per-source regularization**: When using `GroupedFeatureSource` (combined embeddings + LLM), applies different L2 penalties per source:
- High-dim embeddings: `l2_emb ∈ [100, 1000, 10000]` (strong regularization)
- Low-dim LLM features: `l2_llm ∈ [0.01, 0.1, 1, 10]` (weak regularization)
- Loss: `weight_reg = l2_emb * ||w_emb||² + l2_llm * ||w_llm||²`
- Internal CV selects best (l2_emb, l2_llm) combination

**Note**: In Experiment New Tasks (task holdout), Feature-IRT performs similarly to Ridge because it must generalize to unseen test tasks using only feature weights. This is unlike Experiment B (agent holdout) where Feature-IRT can leverage jointly-learned abilities across all tasks.

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
python -m experiment_new_tasks.generate_embeddings --backbone "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
```

### 2. Environment Features (NEW)

Deterministic features extracted from SWE-bench task environments by running bash commands inside Docker containers. These features capture repository structure, codebase statistics, and git history.

**Location**: `experiment_new_tasks/env_features/`

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
python -m experiment_new_tasks.env_features.run_extraction \
    --batch_size 10 \
    --max_connections 10 \
    --output_dir chris_output/env_features/swebench_verified/

# Resume interrupted extraction (automatically skips completed samples)
python -m experiment_new_tasks.env_features.run_extraction --output_dir chris_output/env_features/swebench_verified/

# Just aggregate existing logs to CSV
python -m experiment_new_tasks.env_features.run_extraction --aggregate_only
```

**Output**: `chris_output/env_features/swebench_verified/env_features.csv`

**Architecture**:
- Uses Inspect AI framework with deterministic solver (no LLM calls)
- Each task runs in its own Docker container (`swebench/sweb.eval.x86_64.{instance_id}`)
- Batched processing with Docker cleanup between batches to prevent memory overflow
- Supports resuming interrupted runs

See [experiment_new_tasks/env_features/README.md](env_features/README.md) for detailed documentation.

### 3. Auditor Agent Features (NEW)

Semantic features extracted by an LLM auditor agent (Claude Opus 4.5) that explores SWE-bench task environments via bash shell access. These features capture aspects that require environment interaction to assess.

**Location**: `experiment_new_tasks/auditor_agent/`

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
python -m experiment_new_tasks.auditor_agent.run_auditor --output_dir chris_output/auditor_pilot/

# Parse outputs to CSV
python -m experiment_new_tasks.auditor_agent.parse_outputs
```

**Output**: `chris_output/auditor_pilot/v3_features_top3.csv`

**Integration**: Auditor features are included in the default LLM Judge feature set via coefficient-based selection.

See [experiment_new_tasks/auditor_agent/README.md](auditor_agent/README.md) for detailed documentation.

### 4. LLM Judge Features

Semantic features extracted via LLM structured output:

**All datasets (20 features, v5 — Anthropic Sonnet 4.6, solution-level override)**:

Default feature set: `chris_output/llm_judge_features/v5_anthropic_solution/{dataset}/llm_judge_features.csv`

20 features extracted with solution-level context:
- solution_hint, problem_clarity, domain_knowledge_required, logical_reasoning_required
- atypicality, verification_difficulty, standard_pattern_available, error_specificity
- reproduction_clarity, expected_behavior_clarity, debugging_complexity, codebase_scope
- information_completeness, similar_issue_likelihood, side_effect_risk
- test_comprehensiveness, test_assertion_complexity, test_edge_case_coverage
- solution_complexity, integration_complexity

Note: SWE-bench Verified's v5 CSV also includes 3 auditor features (entry_point_clarity, change_blast_radius, fix_localization) for 23 total.

Old defaults (9-15 features) are archived in `chris_output/llm_judge_features/experiment_a_old_defaults/`.

### Unified Features (Default)

The default feature set uses **15 unified features** identical across all 4 datasets (10 Problem + 1 Test + 1 Solution + 3 Auditor):
- **Problem (10)**: atypicality, codebase_scope, debugging_complexity, domain_knowledge_required, error_specificity, logical_reasoning_required, side_effect_risk, similar_issue_likelihood, solution_hint, verification_difficulty
- **Test (1)**: test_edge_case_coverage
- **Solution (1)**: solution_complexity
- **Auditor (3)**: codebase_scale, fix_localization, implementation_language_complexity

**Feature paths**: `llm_judge_features/defaults/{dataset}/llm_judge_features.csv`

### LLM Judge Feature Variants (v2, v3, v5, v6)

We tested whether extracting all 20 non-auditor features (instead of the 9/15 defaults) and overriding info levels would improve results. The variants test different LLM providers and info levels in a 2×2 design.

| Variant | Features | Info Level | Model | Provider |
|---------|----------|-----------|-------|----------|
| Defaults | 9–15 per dataset | Natural | Claude Opus 4.5 | Anthropic |
| v2 | 20 | Natural | GPT 5.4 | OpenAI |
| v3 | 20 | SOLUTION override | GPT 5.4 | OpenAI |
| v6 | 20 | Natural | Claude Sonnet 4.6 | Anthropic |
| v5 | 20 | SOLUTION override | Claude Sonnet 4.6 | Anthropic |

Run with:
```bash
python -m experiment_new_tasks.run_all_datasets \
  --llm_judge_features_path "chris_output/llm_judge_features/v2_full_20features/{dataset}/llm_judge_features.csv"

python -m experiment_new_tasks.run_all_datasets \
  --llm_judge_features_path "chris_output/llm_judge_features/v3_solution_level/{dataset}/llm_judge_features.csv"

python -m experiment_new_tasks.run_all_datasets \
  --llm_judge_features_path "chris_output/llm_judge_features/v6_anthropic_natural/{dataset}/llm_judge_features.csv"

python -m experiment_new_tasks.run_all_datasets \
  --llm_judge_features_path "chris_output/llm_judge_features/v5_anthropic_solution/{dataset}/llm_judge_features.csv"
```

**Results (Grouped Ridge AUC — Emb+LLM combined)**:

| Dataset | Defaults | v2 (GPT, nat) | v3 (GPT, sol) | v6 (Anth, nat) | v5 (Anth, sol) |
|---------|----------|---------------|---------------|----------------|----------------|
| SWE-bench Verified | **0.8445** | 0.8356 | 0.8387 | 0.8374 | 0.8414 |
| GSO | **0.7642** | 0.7501 | 0.7123 | 0.7597 | 0.7500 |
| TerminalBench | 0.8065 | **0.8354** | 0.8348 | 0.8149 | 0.8310 |
| SWE-bench Pro | 0.7557 | **0.7674** | 0.7672 | 0.7652 | 0.7658 |

**Results (LLM Judge AUC — LLM features only)**:

| Dataset | Defaults | v2 (GPT, nat) | v3 (GPT, sol) | v6 (Anth, nat) | v5 (Anth, sol) |
|---------|----------|---------------|---------------|----------------|----------------|
| SWE-bench Verified | 0.8372 | 0.8292 | 0.8322 | 0.8343 | **0.8403** |
| GSO | **0.7642** | 0.7337 | 0.7356 | 0.7439 | 0.7462 |
| TerminalBench | 0.7978 | **0.8138** | 0.8001 | 0.7978 | 0.8108 |
| SWE-bench Pro | 0.7212 | **0.7370** | 0.7362 | 0.7358 | 0.7336 |

**Key findings**:
- **Model choice matters more than info level**: Anthropic consistently outperforms GPT across both info levels (compare v6 vs v2, v5 vs v3)
- **Solution-level helps LLM Judge** on SWE-bench Verified (v5 0.8403 > v6 0.8343) and GSO (v5 0.7462 > v6 0.7439), but **natural level helps Grouped on GSO** (v6 0.7597 > v5 0.7500) — likely because embeddings already encode solution info, so natural-level LLM features add orthogonal signal
- **TerminalBench and SWE-bench Pro prefer GPT** with natural levels (v2 best) — possibly due to code-heavy tasks
- **GSO is structurally different**: no `problem_statement` field exists — the `prob_script` (benchmark) IS the problem. PROBLEM-level extraction for GSO sees only a function name, making 15/20 features essentially meaningless at that level
- **Scale text changes (optimization-specific rubrics) did not help**: we tested adding GSO-specific wording to 8 features — correlation with v1 defaults was unchanged (n=10 pilot)
- **Remaining gap on GSO** (best 0.7597 vs 0.7642 defaults) likely due to the original 8 GSO-specific features being better tailored than the generic 20-feature set

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

#### LLM Judge Information Level Ablation (All Datasets, v8 features)

Measures the contribution of each information level by progressively adding features. Uses v8 features extracted at natural info levels (no leakage). Feature count is held constant at 15 via Ridge coefficient-based selection.

**LLM Judge AUC by Information Level**:

| Info Level | SWE-bench Verified | GSO | TerminalBench | SWE-bench Pro |
|---|---|---|---|---|
| Baseline | 0.7175 | 0.7130 | 0.7338 | 0.6558 |
| Problem | 0.7873 | 0.7276 | 0.7984 | 0.7175 |
| + Auditor | 0.7984 | 0.7296 | 0.8076 | 0.7365 |
| + Test | 0.8215 | 0.7250 | 0.8059 | 0.7489 |
| + Solution (Full) | 0.8435 | 0.7343 | 0.8086 | 0.7562 |
| Oracle | 0.9447 | 0.9139 | 0.9317 | 0.9183 |

**Information sources** (28 features total across 4 levels):
- **Problem** (15 features): Derived from problem statement alone
- **+ Auditor** (8 features): Environment exploration via Docker (no tests/solution access)
- **+ Test** (3 features): Test/evaluation artifact analysis (no solution access)
- **+ Solution** (2 features): Gold solution patch analysis

**Feature selection**: Ridge regression (alpha=1.0) on all available features at each level → rank by |coefficient| → keep top 15.

**Key findings**:
- **SWE-bench Verified** shows clear monotonic improvement: each level adds ~1-2% AUC
- **SWE-bench Pro** similarly monotonic, with auditor providing the largest single jump (+1.9%)
- **TerminalBench** captures most signal at Problem level; additional levels add marginal value
- **GSO** is noisy (102 tasks, std ~0.05) — all levels are within noise of each other

To run:
```bash
# Run information level ablation (all datasets, parallel)
python -m experiment_new_tasks.run_information_ablation

# Rebuild ablation CSVs
python -m experiment_new_tasks.run_information_ablation --rebuild_csvs

# Run specific datasets
python -m experiment_new_tasks.run_information_ablation --datasets swebench_verified gso
```

To extract features:
```bash
# SWE-bench features
python -m llm_judge_feature_extraction extract --dataset swebench_verified --dry-run
python -m llm_judge_feature_extraction extract --dataset swebench_verified

# TerminalBench features
python -m llm_judge_feature_extraction extract --dataset terminalbench --dry-run
python -m llm_judge_feature_extraction extract --dataset terminalbench

# Options
python -m llm_judge_feature_extraction extract --dataset swebench_verified --limit 50  # Process first 50 tasks
python -m llm_judge_feature_extraction extract --dataset swebench_verified --provider openai  # Use OpenAI
python -m llm_judge_feature_extraction extract --dataset swebench_verified --model claude-sonnet-4-20250514  # Use specific model
python -m llm_judge_feature_extraction extract --dataset gso --info-level-override solution --all  # All features at solution level

# Aggregate existing JSON files to CSV
python -m llm_judge_feature_extraction aggregate --dataset swebench_verified
```

## Data Paths

All datasets follow the same layout under `data/{dataset}/`:

| File | Purpose |
|------|---------|
| `data/{dataset}/responses.jsonl` | Binary response matrix |
| `data/{dataset}/irt/1d_1pl/abilities.csv` | Oracle IRT abilities |
| `data/{dataset}/irt/1d_1pl/items.csv` | Oracle IRT difficulties |

Dataset-specific auxiliary files:
- **SWE-bench Pro**: `data/swebench_pro/agent_dates.json`, `data/swebench_pro/swe-bench-pro.csv`
- **GSO**: `data/gso/agent_dates.json`
- **TerminalBench**: `data/terminalbench/model_release_dates.json`, `data/terminalbench/tasks.jsonl`, `data/terminalbench/meta.json`

Fold-specific IRT models (cached): `chris_output/experiment_a_{dataset}/irt_splits/`

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
--coefficients        Extract and display LLM Judge Ridge coefficients
```

## Output

Results saved to `chris_output/experiment_new_tasks/experiment_a_cv5_results.json`:

```json
{
  "config": {...},
  "k_folds": 5,
  "cv_results": {
    "oracle": {"mean_auc": 0.9447, "std_auc": 0.0085, ...},
    "grouped": {"mean_auc": 0.8445, "std_auc": 0.0208, ...},
    "llm_judge": {"mean_auc": 0.8372, "std_auc": 0.0204, ...},
    "embedding": {"mean_auc": 0.8242, "std_auc": 0.0193, ...},
    "constant_baseline": {"mean_auc": 0.7174, "std_auc": 0.0082, ...}
  }
}
```

## Coefficient Analysis

Extract and display LLM Judge Ridge coefficients (Table 10 / Figure 3 in the paper):

```bash
# Run with coefficient analysis for a specific dataset
python -m experiment_new_tasks.run_all_datasets --datasets swebench_verified --coefficients --sequential
```

The `--coefficients` flag extracts per-fold Ridge coefficients from the LLM Judge predictor and prints:
- Feature rankings by |coefficient| magnitude
- Source-level summary (Problem Statement, Environment, Test Patch, Solution Patch)
- Bar chart saved to the output directory

## Caches

| Cache | Location | When to Clear |
|-------|----------|---------------|
| **IRT Split Models** | `chris_output/experiment_new_tasks/irt_splits/` | When changing split parameters |
| **Embeddings** | `chris_output/experiment_new_tasks/embeddings/` | When changing backbone |
| **LLM Judge Features** | `chris_output/experiment_new_tasks/llm_judge_features/` | When re-extracting |

## References

- IRT formula: `P = sigmoid(theta - beta)` matches py_irt's 1PL implementation
- [Research Proposal](../chris%20proposal.md) - Section 3.1

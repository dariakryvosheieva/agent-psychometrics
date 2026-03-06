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

# Disable LLM judge features (embedding only)
python -m experiment_a.run_all_datasets --no-unified_judge

# Export results to CSV
python -m experiment_a.run_all_datasets --output results.csv

# Run Feature-IRT variant (joint training instead of Ridge)
python -m experiment_a.run_all_datasets --feature_irt
python -m experiment_a.run_all_datasets --feature_irt --datasets swebench_verified gso

# Run a single dataset
python -m experiment_a.run_all_datasets --datasets terminalbench
```

## Results (2026-03-04)

### Summary Table (Default Ridge)

Run with: `python -m experiment_a.run_all_datasets`

| Dataset | Tasks | Agents | Oracle | Grouped Ridge (Emb+LLM) | LLM Judge | Embedding | Baseline |
|---------|-------|--------|--------|-------------------------|-----------|-----------|----------|
| SWE-bench Verified | 500 | 134 | 0.9447 | **0.8445** | 0.8372 | 0.8242 | 0.7174 |
| GSO | 102 | 15 | 0.9139 | **0.7642** | 0.7562 | 0.7581 | 0.7130 |
| TerminalBench | 89 | 112 | 0.9317 | **0.8065** | 0.7835 | 0.8174 | 0.7338 |
| SWE-bench Pro | 730 | 14 | 0.9183 | **0.7557** | 0.7089 | 0.7550 | 0.6558 |

**LLM features**: SWE-bench Verified uses 15 features (9 semantic + 3 auditor + 3 test); all other datasets use 9 features (8 core + codebase_scope). See Feature Sources section for details.

**Key findings**:
- **Grouped Ridge (Emb+LLM) is best**: Outperforms single sources on all datasets
- **Combining features helps**: Grouped Ridge outperforms single sources on all datasets

### Feature-IRT Results (Joint Training)

Run with: `python -m experiment_a.run_all_datasets --feature_irt`

| Dataset | Oracle | Feature-IRT (Emb+LLM) | Feature-IRT (LLM) | Feature-IRT (Emb) | Baseline |
|---------|--------|----------------------|-------------------|-------------------|----------|
| SWE-bench Verified | 0.9447 | **0.8389** | 0.8370 | 0.8243 | 0.7174 |
| GSO | 0.9139 | 0.7407 | 0.7149 | **0.7571** | 0.7130 |
| TerminalBench | 0.9317 | — | — | — | 0.7338 |
| SWE-bench Pro | 0.9183 | 0.7236 | 0.7112 | **0.7555** | 0.6567 |

**Key findings**:
- **Feature-IRT performs similarly to Ridge** in Experiment A (task holdout) because it must generalize to unseen test tasks using only feature weights
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

**`experiment_ab_shared/`** - Core abstractions shared with Experiment B:

| File | Purpose |
|------|---------|
| `dataset.py` | `ExperimentData`, `load_dataset_for_fold()` |
| `feature_source.py` | `TaskFeatureSource` ABC with `EmbeddingFeatureSource`, `CSVFeatureSource`, `GroupedFeatureSource` |
| `feature_predictor.py` | `DifficultyPredictorBase` ABC, `FeatureBasedPredictor`, `GroupedRidgePredictor` |
| `evaluator.py` | `compute_irt_probability()`, `convert_numpy()` |

**`experiment_a/`** - Experiment A orchestration:

| File | Purpose |
|------|---------|
| `config.py` | `ExperimentAConfig`, `DATASET_DEFAULTS` registry |
| `pipeline.py` | `CVPredictorConfig`, `cross_validate_all_predictors()`, `build_cv_predictors()` |
| `cross_validation.py` | `CVPredictor` protocol, `evaluate_predictor_cv()`, `k_fold_split_tasks()` |
| `difficulty_predictors.py` | `OraclePredictor`, `ConstantPredictor`, `DifficultyPredictorAdapter` |
| `feature_irt.py` | `JointTrainingCVPredictor` (with per-source L2 regularization), `feature_irt_predictor_factory()` |
| `coefficient_analysis.py` | `extract_llm_coefficients()`, `print_coefficient_table()`, `save_coefficient_bar_chart()` |

### Multi-Dataset Scripts (`experiment_a/`)

| File | Purpose |
|------|---------|
| `run_all_datasets.py` | Run all datasets with Ridge (default) or Feature-IRT (`--feature_irt`) |
| `run_information_ablation.py` | Run information source ablation study (SWE-bench Verified) |

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

**Note**: In Experiment A (task holdout), Feature-IRT performs similarly to Ridge because it must generalize to unseen test tasks using only feature weights. This is unlike Experiment B (agent holdout) where Feature-IRT can leverage jointly-learned abilities across all tasks.

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

**SWE-bench Verified (15 features)**:

Default feature set: `chris_output/llm_judge_features/experiment_a_defaults/swebench_verified.csv`

15 features selected by Ridge coefficient magnitude:
- **Problem**: problem_clarity, atypicality, logical_reasoning_required, codebase_scope, information_completeness, similar_issue_likelihood, error_specificity, reproduction_clarity
- **Auditor (3)**: entry_point_clarity, change_blast_radius, fix_localization
- **Test (2)**: test_comprehensiveness, test_edge_case_coverage
- **Solution (2)**: solution_complexity, integration_complexity

**GSO, TerminalBench, SWE-bench Pro (9 features)**:

Default feature set: `chris_output/llm_judge_features/experiment_a_defaults/{dataset}.csv`

8 core features + codebase_scope:
- solution_hint, problem_clarity, solution_complexity, domain_knowledge_required
- logical_reasoning_required, atypicality, verification_difficulty, standard_pattern_available, codebase_scope

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

The current defaults use dataset-specific feature sets stored in `experiment_a_defaults/`. To use unified features instead, specify the path explicitly:
```bash
python -m experiment_a.run_all_datasets --unified_judge_suffix _core
```

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

#### LLM Judge Ablation (SWE-bench Verified)

To measure the contribution of different information sources, we ran ablation experiments progressively adding features from different affordances. Feature count is held constant at 15 via Ridge coefficient-based selection to isolate the value of each information source:

| Method | # Features | LLM Judge AUC | Grouped Ridge AUC |
|--------|------------|---------------|-------------------|
| Problem Only | 15 | 0.7821 ± 0.0164 | 0.7800 ± 0.0241 |
| + Auditor | 15 | 0.8015 ± 0.0167 | 0.7967 ± 0.0240 |
| + Test | 15 | 0.8225 ± 0.0230 | 0.8174 ± 0.0235 |
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
python -m experiment_ab_shared.llm_judge extract --dataset swebench_verified --dry-run
python -m experiment_ab_shared.llm_judge extract --dataset swebench_verified

# TerminalBench features
python -m experiment_ab_shared.llm_judge extract --dataset terminalbench --dry-run
python -m experiment_ab_shared.llm_judge extract --dataset terminalbench

# Options
python -m experiment_ab_shared.llm_judge extract --dataset swebench_verified --limit 50  # Process first 50 tasks
python -m experiment_ab_shared.llm_judge extract --dataset swebench_verified --provider openai  # Use OpenAI
python -m experiment_ab_shared.llm_judge extract --dataset swebench_verified --model claude-sonnet-4-20250514  # Use specific model

# Aggregate existing JSON files to CSV
python -m experiment_ab_shared.llm_judge aggregate --dataset swebench_verified
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

Results saved to `chris_output/experiment_a/experiment_a_cv5_results.json`:

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
python -m experiment_a.run_all_datasets --datasets swebench_verified --coefficients --sequential
```

The `--coefficients` flag extracts per-fold Ridge coefficients from the LLM Judge predictor and prints:
- Feature rankings by |coefficient| magnitude
- Source-level summary (Problem Statement, Environment, Test Patch, Solution Patch)
- Bar chart saved to the output directory

## Caches

| Cache | Location | When to Clear |
|-------|----------|---------------|
| **IRT Split Models** | `chris_output/experiment_a/irt_splits/` | When changing split parameters |
| **Embeddings** | `chris_output/experiment_a/embeddings/` | When changing backbone |
| **LLM Judge Features** | `chris_output/experiment_a/llm_judge_features/` | When re-extracting |

## References

- IRT formula: `P = sigmoid(theta - beta)` matches py_irt's 1PL implementation
- [Research Proposal](../chris%20proposal.md) - Section 3.1

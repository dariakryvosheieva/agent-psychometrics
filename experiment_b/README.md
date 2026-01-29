# Experiment B: Frontier Task Difficulty Prediction

Predict difficulty of **frontier tasks** (tasks only solvable by newer models) using methods that do NOT have access to post-frontier agents.

## Overview

**Research Question**: Can we predict the difficulty of tasks that are currently beyond the capability of existing models, using only information available before those models were released?

**Setting**:
- **Date-based split**: Pre-frontier vs Post-frontier agents (by release date)
- **No data leakage**: Predictions made using only pre-frontier information; post-frontier agents only used for evaluation

## Evaluation Metrics

### Primary: Mean Per-Agent AUC (Scale-Free)

The primary metric is **Mean Per-Agent AUC**, a scale-free metric that requires no oracle data or scale alignment:

1. For each post-frontier agent, compute AUC on frontier tasks using predicted difficulty ranking
2. Average across all agents with response variance (at least one success on frontier tasks)
3. Report mean ± SEM (standard error of the mean)

**Key insight**: For a fixed agent, `AUC(y_true, sigmoid(θ - β̂)) = AUC(y_true, -β̂)` because sigmoid is monotonic. We don't need to know agent abilities—just whether harder-predicted tasks have lower success rates.

**Advantages over pooled ROC-AUC**:
- No scale alignment needed (completely scale-free)
- No oracle abilities needed
- Each agent is an independent measurement
- Directly grounded in response data

### Secondary: Pooled ROC-AUC (Requires Scale Alignment)

The pooled ROC-AUC pools all (agent, task) pairs and requires fitting an affine transformation to align predicted difficulties to the oracle scale. This metric is still reported for comparison but is less principled.

## Frontier Task Definitions

Four definitions of "frontier task" are supported:

1. **Zero pre-frontier** (`zero_pre`) **[Recommended]**: Tasks with 0% pre-frontier pass rate AND >0% post-frontier pass rate. Most principled—identifies tasks that became solvable with frontier agents.
2. **Pass-rate based** (`passrate`): Tasks with ≤10% pre-frontier pass rate AND >10% post-frontier pass rate
3. **Pre-only** (`pre_only`): Tasks with ≤X% pre-frontier pass rate (no post-frontier filter). Uses `--pre_threshold` argument. Useful for analyzing how prediction difficulty scales with available training signal.
4. **IRT-based** (`irt`): Tasks where NO pre-frontier agent has ≥50% solve probability under IRT

## Quick Start

```bash
source .venv/bin/activate

# Run on SWE-bench with default settings (zero_pre frontier definition, no date forecasting)
python -m experiment_b.compare_methods

# Run on all datasets
python -m experiment_b.compare_methods --dataset swebench_pro
python -m experiment_b.compare_methods --dataset terminalbench
python -m experiment_b.compare_methods --dataset gso

# Run with multiple frontier definitions
python -m experiment_b.compare_methods --frontier_definitions passrate irt

# Run threshold sweep analysis (0% to 30% pre-frontier threshold)
python -m experiment_b.threshold_sweep --datasets swebench

# Enable date forecasting (slower)
python -m experiment_b.compare_methods --forecast_dates

# Save results to CSV
python -m experiment_b.compare_methods --output_csv results.csv
```

## Results (2026-01-23)

All results use the **zero_pre** frontier definition (0% pre-frontier, >0% post-frontier) and report **Mean Per-Agent AUC ± SEM**.

### SWE-bench Verified

**Cutoff**: 2025-05-01 | **Pre-frontier agents**: 76 | **Post-frontier agents**: 55 | **Frontier tasks**: 34 | **Eval agents with variance**: 41

| Method | Mean AUC ± SEM | ROC-AUC |
|--------|----------------|---------|
| SAD-IRT (best) | **0.7592 ± 0.022** | 0.7794 |
| Oracle (upper bound) | 0.7322 ± 0.039 | 0.8399 |
| Trajectory + Ridge | 0.5442 ± 0.033 | 0.7157 |
| Grouped Ridge (all features) | 0.5256 ± 0.032 | 0.7041 |
| Feature-IRT (Trajectory) | 0.5250 ± 0.034 | 0.7017 |
| Baseline IRT (pre-frontier only) | 0.4854 ± 0.031 | 0.7503 |
| Embedding + Ridge | 0.4716 ± 0.039 | 0.6868 |
| LLM Judge + Ridge | 0.4665 ± 0.034 | 0.6898 |

### SWE-bench Pro

**Cutoff**: 2025-06-01 | **Pre-frontier agents**: 5 | **Post-frontier agents**: 9 | **Frontier tasks**: 114 | **Eval agents with variance**: 9

| Method | Mean AUC ± SEM | ROC-AUC |
|--------|----------------|---------|
| Oracle (upper bound) | **0.7377 ± 0.029** | 0.8000 |
| Embedding + Ridge | 0.5896 ± 0.029 | 0.7132 |
| Grouped Ridge (Embedding + LLM Judge) | 0.5757 ± 0.027 | 0.7103 |
| Feature-IRT (Embedding) | 0.5756 ± 0.027 | 0.6652 |
| Feature-IRT (LLM Judge) | 0.5444 ± 0.025 | 0.7013 |
| LLM Judge + Ridge | 0.5419 ± 0.024 | 0.7018 |
| Baseline IRT (pre-frontier only) | 0.5055 ± 0.017 | 0.6958 |

**Note**: SAD-IRT not available (trained on SWE-bench tasks only). Cutoff moved earlier (from 2025-09-01) to get more post-frontier agents for reliable evaluation.

### TerminalBench

**Cutoff**: 2025-09-01 | **Pre-frontier agents**: 37 | **Post-frontier agents**: 46 | **Frontier tasks**: 11 | **Eval agents with variance**: 25

| Method | Mean AUC ± SEM | ROC-AUC |
|--------|----------------|---------|
| Oracle (upper bound) | **0.7614 ± 0.041** | 0.8348 |
| Baseline IRT (pre-frontier only) | 0.5910 ± 0.051 | 0.7681 |
| LLM Judge + Ridge | 0.5318 ± 0.046 | 0.7539 |
| Feature-IRT (LLM Judge) | 0.4716 ± 0.040 | 0.7502 |
| Embedding + Ridge | 0.4253 ± 0.041 | 0.6676 |
| Feature-IRT (Embedding) | 0.4189 ± 0.043 | 0.6937 |

**Note**: SAD-IRT not available (trained on SWE-bench tasks only). Earlier cutoff (moved from 2025-11-01 to 2025-09-01) provides more frontier tasks and much tighter confidence intervals.

### GSO (Software Optimization Benchmark)

**Cutoff**: 2025-08-15 | **Pre-frontier agents**: 8 | **Post-frontier agents**: 6 | **Frontier tasks**: 33 | **Eval agents with variance**: 6

| Method | Mean AUC ± SEM | ROC-AUC |
|--------|----------------|---------|
| Oracle (upper bound) | **0.7319 ± 0.043** | 0.8043 |
| Feature-IRT (Embedding) | 0.6282 ± 0.171 | 0.6890 |
| Grouped Ridge (Embedding + LLM Judge) | 0.5886 ± 0.128 | 0.7351 |
| LLM Judge + Ridge | 0.5787 ± 0.114 | 0.7334 |
| Embedding + Ridge | 0.5678 ± 0.150 | 0.7023 |
| Baseline IRT (pre-frontier only) | 0.5206 ± 0.047 | 0.7155 |

**Note**: SAD-IRT not available (trained on SWE-bench tasks only).

### Key Observations

- **SAD-IRT vs Oracle**: SAD-IRT (0.759 ± 0.022) slightly outperforms Oracle (0.732 ± 0.039) on SWE-bench, but the difference is **not statistically significant** (combined SE ≈ 0.045, p > 0.5). Trajectory features may capture difficulty signal that outcomes don't, but more data is needed to confirm.
- **Baseline IRT on zero_pre tasks**: For tasks with 0% pre-frontier success, Baseline IRT has no training signal and produces near-random predictions (AUC ≈ 0.5). This is expected behavior, not a bug.
- **Mean AUC and ROC-AUC can give different rankings**: This is because Mean AUC is scale-free while ROC-AUC requires scale alignment
- **Error bars are informative**: Small eval agent counts lead to high variance (e.g., SWE-bench Pro with 3 agents had SEM ± 0.18)
- **Agent filtering**: Only agents with at least one frontier task success are included in Mean AUC computation

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

**Hyperparameters** (grid search over l2_weight and l2_residual):
- `l2_weight`: Regularization on feature weights (grid: 0.001 to 10.0)
- `l2_residual`: Regularization on per-task residuals (grid: 0.001 to 10.0)
- `use_residuals`: Whether to include per-task residuals (default: True)

### Baseline-Init Feature-IRT

A variant of Feature-IRT that initializes from Baseline IRT values, then learns feature-based corrections:

```
b_i = w^T f_i + r_i           (task difficulty)
θ_j initialized from baseline (agent ability)

Initialization:
- r_i starts at baseline IRT difficulty (b_baseline)
- θ_j starts at baseline IRT ability (θ_baseline)
- w starts at zero (features contribute nothing initially)
```

**Key insight**: When the frontier threshold is high (e.g., 20-30%), Baseline IRT already has strong signal from pre-frontier agents. Initializing from baseline and learning residual corrections helps preserve this signal while allowing features to add information.

**Diagnostics** (via `get_baseline_init_diagnostics()`):
- Weight norm: How much feature weights deviate from zero
- Difficulty drift: Mean |b_final - b_baseline|
- Ability drift: Mean |θ_final - θ_baseline|
- Feature contribution: |w^T f| / (|w^T f| + |r|)

### Hyperparameter Selection via Held-Out Response AUC

Feature-IRT hyperparameters (`l2_weight`, `l2_residual`, `l2_ability`) can be selected automatically using cross-validation on held-out response pairs. This avoids manual tuning and adapts to each dataset.

**How it works:**
1. Randomly hold out 20% of (agent, task) response pairs for validation
2. Ensure all agents and tasks appear in training set (stratified split)
3. Grid search: For each hyperparam combination, train on 80%, compute AUC on 20%
4. Select hyperparams that maximize validation AUC
5. Final model trained on full dataset with best hyperparams

**Usage:**
```bash
# Enable CV-based hyperparameter selection
python -m experiment_b.threshold_sweep --datasets swebench --use_cv_hyperparams
```

**Grid** (same for all L2 parameters): `[0.001, 0.01, 0.1, 1.0, 10.0, 100.0]`

### Grouped Feature Support

Feature-IRT supports combining multiple feature sources (e.g., Embedding + Trajectory) with per-source regularization. This is important because:
- Embeddings are 5120-dim and need stronger regularization
- Trajectory features are 20-dim and need weaker regularization

**How it works:**
1. Per-source StandardScaler (fit separately per source to prevent high-dim features from dominating)
2. Group scaling: Features scaled by `1/sqrt(alpha)` per source
3. This is mathematically equivalent to different L2 penalties per source with `l2_weight=1.0`

**Usage:**
```bash
# Test all feature configurations (Embedding, Trajectory, Embedding+Trajectory)
python -m experiment_b.threshold_sweep --datasets swebench --test_all_feature_configs
```

For grouped sources, the hyperparameter grid includes per-source alphas instead of a single `l2_weight`:
- `alpha_Embedding`: Regularization for embedding features
- `alpha_Trajectory`: Regularization for trajectory features
- `l2_residual`, `l2_ability`: Same as single-source

## Evaluation Methodology

### Mean Per-Agent AUC (Primary, Scale-Free)

1. **Filter eval agents**: Remove agents that fail ALL frontier tasks (they provide no ranking information)
2. **For each remaining agent**:
   - Rank frontier tasks by predicted difficulty (higher β = harder = lower expected success)
   - Compute AUC: do harder-predicted tasks have lower success rates?
   - Note: This is equivalent to `AUC(y_true, -predicted_β)` since AUC only depends on ranking
3. **Average across agents**: Compute mean and standard error of the mean (SEM)

**Why this works**: For any fixed agent ability θ, `sigmoid(θ - β)` is a monotonic function of `-β`. Therefore, ranking tasks by `sigmoid(θ - β)` is equivalent to ranking by `-β`. We never need to know θ!

### ROC-AUC with Scale Alignment (Secondary)
1. **Identify anchor tasks**: Tasks with 10-90% pass rate in BOTH pre- and post-frontier groups
2. **Fit affine transformation**: `oracle_β = slope × predicted_β + intercept` on anchors
3. **Compute probabilities**: For each (post-frontier agent, frontier task): `P(success) = sigmoid(θ_oracle - β_shifted)`
4. **Calculate ROC-AUC**: Compare predicted probabilities to actual responses

**Note**: This requires oracle abilities and scale alignment, making it less principled than Mean Per-Agent AUC.

### 3. Date Forecasting

Predict **when** tasks will become solvable with 50% probability. Enabled by default (disable with `--no_forecast_dates`).

**Key insight**: From IRT, `P(success) = sigmoid(θ - β) = 0.5` when `θ = β`. Combined with Experiment D's finding that frontier ability grows linearly over time, we can invert the ability-over-time relationship to predict when a task with difficulty β will become solvable.

**Methodology** (per-method, avoids data leakage):

1. **Fit ability-over-time regression** using each method's own IRT abilities:
   - Group pre-frontier agents by date, compute cumulative max ability (frontier trajectory)
   - Fit linear model: `frontier_θ = slope × days + intercept`
   - This uses ONLY pre-frontier information from that method's own IRT model

2. **Predict solvability date** by inverting the regression:
   - For a task with predicted difficulty β: `days = (β - intercept) / slope`
   - Convert days to calendar date

3. **Evaluate against ground truth**:
   - Ground truth: Oracle IRT's first-capable-date (earliest agent with θ >= β_oracle)
   - Metrics: MAE (days), Pearson r

**Which methods support date forecasting?**

Only methods that learn their own IRT model with abilities can produce date forecasts:

| Method | Has Own IRT? | Date Forecast |
|--------|--------------|---------------|
| Oracle | ✓ (oracle abilities) | ✓ |
| Baseline IRT | ✓ (pre-frontier abilities) | ✓ |
| Feature-IRT | ✓ (learned abilities) | ✓ |
| SAD-IRT | ✓ (learned abilities) | ✓ |
| Embedding + Ridge | ✗ (no IRT) | N/A |
| LLM Judge + Ridge | ✗ (no IRT) | N/A |

```bash
# Run with date forecasting (default)
python -m experiment_b.compare_methods

# Disable date forecasting (faster)
python -m experiment_b.compare_methods --no_forecast_dates
```

### 4. Oracle MAE (Difficulty Prediction Quality)

In addition to the end-to-end MAE (which uses each method's ability-over-time regression), we report **Oracle MAE** to isolate difficulty prediction quality from ability modeling error.

**How it works:**
- For each method's predicted difficulty β (after scale alignment to Oracle):
- Directly look up the earliest Oracle agent with ability θ ≥ β
- Compare this date to ground truth (no regression involved)

**Interpretation:**
- **Oracle MAE = 0** for the Oracle method (sanity check - predicted β matches Oracle β exactly)
- **Low Oracle MAE, high regular MAE**: Difficulty predictions are good, but ability-over-time regression adds error
- **Both MAEs high**: Difficulty predictions themselves are poor

**Example output:**
```
Method                                      ROC-AUC   MAE (days)  Oracle MAE†
----------------------------------------------------------------------------
Oracle (upper bound)                         0.8439         22.1          0.0
SAD-IRT (best)                               0.8036         92.4         53.1
Feature-IRT (Embedding)                      0.7747         53.1         44.9
Baseline IRT (pre-frontier only)             0.7600        103.6         43.8

† Oracle MAE: Earliest Oracle agent with θ ≥ predicted β (bypasses regression)
```

### 5. Frontier Model Intervals

The experiment also reports statistics on how frequently new frontier-capability models appear:

```
Frontier model intervals: mean=24.0, median=13.5, range=1-175 days (31 jumps)
```

This helps contextualize prediction errors relative to how often frontier ability actually advances.

## Data Leakage Constraints

**Critical**: Oracle data and post-frontier agent data must NEVER be exposed during training.

- **Training ground truth**: Always from baseline IRT (pre-frontier agents only)
- **Oracle IRT**: Used ONLY for evaluation metrics (ROC-AUC alignment, date forecasting ground truth)
- **All methods train on ALL tasks**: Ground truth difficulties come from baseline IRT, which is trained only on pre-frontier agents

This constraint ensures a realistic simulation of predicting difficulty for tasks beyond current model capabilities.

## Architecture

Uses the shared `experiment_ab_shared/` infrastructure for predictors:

```python
from experiment_ab_shared.feature_source import EmbeddingFeatureSource, CSVFeatureSource
from experiment_ab_shared.feature_predictor import FeatureBasedPredictor
from experiment_b.shared.prediction_methods import FeatureIRTPredictor

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
├── compare_methods.py        # Main evaluation script
├── threshold_sweep.py        # Threshold sweep analysis
├── shared/
│   ├── config_base.py        # DatasetConfig base class with lazy-loaded properties
│   ├── data_preparation.py   # Data loading, frontier ID, baseline IRT
│   ├── evaluation.py         # ROC-AUC, scale alignment metrics
│   ├── prediction_methods.py # FeatureIRTPredictor + method collection
│   ├── frontier_evaluation.py# Date forecasting setup, evaluation loop
│   ├── output_formatting.py  # Print tables, save CSV
│   ├── diagnostics.py        # Feature-IRT diagnostic plots
│   └── date_forecasting.py   # Date prediction utilities
├── swebench/
│   └── config.py             # SWE-bench Verified dataset configuration
├── swebench_pro/
│   └── config.py             # SWE-bench Pro dataset configuration
└── terminalbench/
    └── config.py             # TerminalBench dataset configuration
```

### Code Organization

**Design Principle**: Adding a new prediction method should only require changes in `prediction_methods.py`. The method is then automatically evaluated through the whole pipeline.

**Module Responsibilities**:

| Module | Purpose |
|--------|---------|
| `data_preparation.py` | Load IRT models, split agents, identify frontier tasks |
| `evaluation.py` | Compute ROC-AUC, scale alignment |
| `prediction_methods.py` | All predictor classes and collection functions |
| `frontier_evaluation.py` | Date forecasting setup, multi-definition evaluation |
| `output_formatting.py` | Print comparison tables, save CSV |
| `diagnostics.py` | Grid search heatmaps, loss curves |
| `date_forecasting.py` | Date prediction utilities |
| `config_base.py` | Dataset config with lazy-loaded properties |

**Key Dataclasses**:
- `ExperimentData`: Runtime-computed values (IRT models, agent splits, frontier tasks)
- `FeatureIRTResults`: Predictions, abilities, and diagnostics from Feature-IRT
- `DateForecastingData`: Date models, ground truth, and evaluation metadata

**Key Functions**:
- `load_and_prepare_data()`: Load IRT models and compute experiment-specific splits
- `build_feature_sources()`: Create feature source objects for embeddings/LLM judge
- `collect_ridge_predictions()`: Train Ridge regressors on feature sources
- `collect_feature_irt_predictions()`: Run Feature-IRT with optional grid search
- `collect_sad_irt_predictions()`: Load all SAD-IRT runs as separate methods
- `setup_date_forecasting()`: Prepare date models and ground truth
- `evaluate_all_frontier_definitions()`: Evaluate all methods across frontier definitions

## Data Paths

### SWE-bench Verified

| File | Purpose |
|------|---------|
| `clean_data/swebench_verified_20251120_full/1d_1pl/items.csv` | Oracle IRT difficulties |
| `clean_data/swebench_verified/swebench_verified_20251120_full.jsonl` | Response matrix |
| `chris_output/experiment_a/embeddings/` | Task embeddings |
| `chris_output/experiment_a/llm_judge_features/` | LLM judge features |

### SWE-bench Pro

| File | Purpose |
|------|---------|
| `chris_output/swebench_pro_irt/1d/items.csv` | Oracle IRT difficulties |
| `out/chris_irt/swebench_pro.jsonl` | Response matrix (730 tasks, 14 agents) |
| `out/swebench_pro/embeddings__deepseek-ai__...npz` | Task embeddings |
| `chris_output/experiment_a_swebench_pro/llm_judge_features/llm_judge_features.csv` | LLM judge v5 features (8 features) |
| `data/swebench_pro_agent_dates.json` | Agent release dates (public announcements) |

### TerminalBench

| File | Purpose |
|------|---------|
| `chris_output/terminal_bench_2.0/1d/items.csv` | Oracle IRT difficulties |
| `data/terminal_bench/terminal_bench_2.0.jsonl` | Response matrix |
| `chris_output/experiment_a_terminalbench/embeddings/` | Task embeddings |
| `chris_output/experiment_a_terminalbench/llm_judge_features/` | LLM judge features (4 pre-selected) |

## Configuration

Dataset configs in `swebench/config.py`, `swebench_pro/config.py`, and `terminalbench/config.py`:

```python
@dataclass
class DatasetConfig(ABC):
    # Core paths
    responses_path: Path
    oracle_irt_path: Path
    oracle_abilities_path: Path
    embeddings_path: Path
    llm_judge_path: Path

    # Frontier split settings
    cutoff_date: str
    pre_threshold: float = 0.1   # Max pass rate for pre-frontier
    post_threshold: float = 0.1  # Min pass rate for post-frontier

    # Lazy-loaded cached properties (computed on first access)
    @property
    def responses(self) -> Dict[str, Dict[str, int]]: ...
    @property
    def all_agents(self) -> List[str]: ...
    @property
    def all_task_ids(self) -> List[str]: ...
    @property
    def agent_dates(self) -> Dict[str, str]: ...
    @property
    def last_agent_date(self) -> Optional[str]: ...

    # Abstract methods (implemented per dataset)
    @abstractmethod
    def get_agent_dates(self, agents: List[str]) -> Dict[str, str]: ...
    @property
    @abstractmethod
    def name(self) -> str: ...
    @property
    def llm_judge_feature_cols(self) -> List[str]:
        # SWE-bench: 9 features, TerminalBench: 4 pre-selected features
```

## Command Line Options

```
--dataset              Dataset to use: swebench (default), swebench_pro, terminalbench, or gso
--frontier_definitions Space-separated list: 'zero_pre' 'passrate' 'irt' (default: zero_pre)
--forecast_dates       Enable date forecasting evaluation (disabled by default)
--output_csv           Save results to CSV file
--grid_search          Run grid search over Feature-IRT hyperparameters
--verbose              Show alignment parameters and training progress
--cutoff_date          Override default cutoff date (YYYYMMDD format)
--filter_bottom_percentile  Remove bottom X% of eval agents by frontier success rate (0.0-1.0)
--min_oracle_ability   [Research] Minimum oracle theta for eval agents (may bias results)
```

## Evaluation Agent Filtering

Post-frontier agents are defined by date (release date >= cutoff), but not all are capable frontier solvers. Low-performing agents add noise to ROC-AUC evaluation. Use `--filter_bottom_percentile` to remove the bottom X% of agents by their success rate on frontier tasks.

```bash
# Remove bottom 20% of agents by frontier success rate
python -m experiment_b.compare_methods --filter_bottom_percentile 0.2
```

**How it works:**
1. Compute success rate on frontier tasks for each post-frontier agent
2. Find the Xth percentile threshold (e.g., 20th percentile)
3. Remove agents below this threshold

**Example output with filtering:**
```
Data Summary:
  - Pre-frontier agents: 76
  - Post-frontier agents: 55 (filtered to 44)
    └─ Removed bottom 20% (success rate < 0.060)
  - Frontier tasks: 47
```

**Default behavior:** No filtering (`--filter_bottom_percentile 0.0`). To change the default, edit `DEFAULT_FILTER_BOTTOM_PERCENTILE` in `compare_methods.py`.

## Caches

| Cache | Location | When to Clear |
|-------|----------|---------------|
| **Baseline IRT** | `chris_output/experiment_b/{dataset}/baseline_irt/` | Auto-invalidated when training data changes |
| **Embeddings** | `chris_output/experiment_a{_terminalbench}/embeddings/` | When changing backbone |
| **LLM Features** | `chris_output/experiment_a{_terminalbench}/llm_judge_features/` | When re-extracting |

## Threshold Sweep Analysis

The `threshold_sweep.py` script analyzes how method performance changes as the frontier task definition threshold varies from 0% to 30%.

```bash
# Run threshold sweep on all datasets
python -m experiment_b.threshold_sweep

# Run on specific datasets
python -m experiment_b.threshold_sweep --datasets swebench terminalbench

# Custom thresholds
python -m experiment_b.threshold_sweep --thresholds 0.0 0.1 0.2 0.3

# Enable date forecasting for all datasets (by default, only swebench)
python -m experiment_b.threshold_sweep --date_forecast_all

# Test all feature configurations (Embedding, Trajectory, combined)
python -m experiment_b.threshold_sweep --datasets swebench --test_all_feature_configs

# Use CV-based hyperparameter selection (slower but may find better params)
python -m experiment_b.threshold_sweep --datasets swebench --use_cv_hyperparams

# Combine both flags for comprehensive analysis
python -m experiment_b.threshold_sweep --datasets swebench --test_all_feature_configs --use_cv_hyperparams
```

### Output Files

Results are saved to `chris_output/threshold_sweep/`:

| File | Description |
|------|-------------|
| `threshold_sweep_{dataset}.csv` | AUC and MAE metrics per threshold/method |
| `threshold_sweep_{dataset}.png` | Mean Per-Agent AUC plot |
| `predicted_vs_oracle_{dataset}.png` | Scatter plot of Baseline-Init Feature-IRT predicted vs Oracle difficulty for zero_pre frontier tasks |
| `date_forecast_{dataset}.png` | Date forecast MAE plot (swebench only by default) |
| `ability_vs_date_{dataset}.png` | Ability-over-time linear fit visualization (swebench only by default) |

### Date Forecasting

Date forecasting predicts when tasks will become solvable based on the linear relationship between frontier ability and time. This is only enabled for SWE-bench by default because other datasets have insufficient agent date diversity for meaningful ability-over-time regression.

The ability-vs-date plot shows:
- Scatter plot of agent abilities vs release date
- Red stars marking "frontier points" where cumulative max ability improved
- Linear fit line with R² (requires ≥2 frontier points)

Use `--date_forecast_all` to enable date forecasting for all datasets (may fail or produce poor fits for datasets with limited agent diversity).

### Key Options

**Fixed agent set**: The sweep uses a fixed set of evaluation agents (those with variance at threshold=0%) across all thresholds for consistent comparison. Without this, weaker agents joining at higher thresholds would cause misleading AUC drops.

**Post-frontier Oracle**: By default, Oracle IRT is trained on all agents. Use `--post_frontier_oracle` to train Oracle on post-frontier agents only:

```bash
python -m experiment_b.threshold_sweep --datasets gso --post_frontier_oracle
```

**Parallel execution**: Datasets are processed in parallel automatically (one process per dataset).

**Key insight**: At low thresholds (0-5%), Baseline IRT has little signal on frontier tasks. As threshold increases, Baseline IRT catches up to Oracle. The Baseline-Init Feature-IRT can sometimes improve over Baseline IRT by leveraging task features.

## Negative Results: Ordered Logit IRT

We attempted to use trajectory rubric scores as an **auxiliary objective** (output) rather than input features. The key insight was that rubric scores are generated by latent task difficulty, so we could invert the model to estimate difficulty.

### Approach

**Three-phase training:**
1. **Baseline IRT** (existing): Get θ_j (ability) and β_i (difficulty) from pre-frontier agents
2. **Learn ordered logit parameters**: For each rubric item k, fit `rubric_ijk ~ OrderedLogit(λ_k × η_ij + δ_k)` where `η_ij = θ_j - β_i`
3. **MAP estimation**: For frontier tasks, fix all parameters except β_i and estimate via MAP with prior `β_i ~ N(mean_frontier, std_all)`

**Validation**: Before training, we validated correlation between rubric scores and η on all 3000 observations. 7 of 9 rubric items showed positive correlation (r = 0.18 to 0.47). Two items with negative correlation (`debugging_cycles`, `exploration_breadth`) were excluded.

### Results

| Method | Mean AUC ± SEM | ROC-AUC |
|--------|----------------|---------|
| Trajectory + Ridge | 0.5442 ± 0.033 | 0.7157 |
| **Ordered Logit IRT** | **0.4936 ± 0.035** | **0.6924** |
| Baseline IRT | 0.4854 ± 0.031 | 0.7503 |

The method performed **worse than Trajectory + Ridge** despite using the same underlying data.

### Root Cause Analysis

The fundamental problem: **rubric scores from pre-frontier agents cannot distinguish between frontier tasks**.

**Evidence:**
1. On ALL 500 tasks, rubric scores correlate with difficulty (Pearson r = 0.65 with baseline β)
2. On the **34 frontier tasks specifically**, correlation drops to **r ≈ 0** with oracle difficulty

| Rubric Item | r with Oracle β (frontier only) |
|-------------|--------------------------------|
| solution_completeness | -0.119 |
| localization_quality | -0.019 |
| loop_detection | -0.108 |
| **Composite** | **-0.225** |

**Why this happens:**
- Frontier tasks have 0% pre-frontier pass rate by definition
- The 6 rubric agents are all pre-frontier (dates: 2024-07 to 2025-04)
- They **fail on ALL frontier tasks** → rubric scores are all similarly low
- No signal to distinguish "more impossible" from "less impossible" tasks

### Key Insight

**Trajectory features from pre-frontier agents can help predict difficulty across the full task distribution**, but they cannot help distinguish between frontier tasks specifically. To get discriminating signal on frontier tasks, you would need:
- Post-frontier agents (who can solve some frontier tasks)
- Or different features that capture difficulty beyond agent performance

This finding explains why SAD-IRT (which uses trajectory features differently) outperforms simple trajectory-based methods on frontier tasks—it may be capturing different information from the trajectories.

### Files

- `experiment_b/shared/rubric_preprocessing.py` - Data loading and preprocessing
- `experiment_b/shared/ordered_logit_predictor.py` - Model implementation
- `experiment_b/rubric_correlation_eda.py` - Correlation validation plots

## Frontier Feature Analysis

The `frontier_feature_analysis.py` script analyzes how feature-difficulty correlations differ between frontier and non-frontier tasks.

### Running the Analysis

```bash
source .venv/bin/activate
python -m experiment_b.frontier_feature_analysis
```

### Features Analyzed

| Source | Features | Count |
|--------|----------|-------|
| Rubric | Aggregated trajectory rubric scores (mean per task) | 10 |
| LLM Judge | Task semantic features | 10 |
| Trajectory | Per-agent assistant char count and message count | 2 |
| Date | Logistic solve date (x0_days) | 1 |

### Visualization Types

**Continuous features** (rubric, trajectory, date): Scatter plots with trend line showing feature value vs oracle difficulty.

**Discrete features** (LLM judge): Mean difficulty ± standard error at each ordinal level, with trend line fitted through the means. Values with no data are omitted.

### Data Filtering

#### Trajectory Features

Trajectory features have extreme outliers from agents that produce very long outputs. The analysis clips to the 99th percentile of non-frontier task values:

- **Threshold**: ~35,978 characters (99th percentile)
- **Removed**: 5 tasks with values above threshold
- **Impact**: Non-frontier r improves from 0.192 to 0.325

#### Date Features

Logistic solve dates have boundary artifacts from curve-fitting optimization:

- **Valid range**: 0 to 1150 days
- **Removed**: Tasks with x0_days < 0 (would require solving before benchmark) or > 1150 (never predicted solvable)
- **Impact**: Non-frontier r improves from 0.894 to 0.939

#### Per-Agent Trajectory Selection

For trajectory features, the script tests all agents and selects the one with the strongest frontier task correlation (best-case scenario):

| Feature | Best Agent | Non-Frontier r | Frontier r |
|---------|------------|---------------|------------|
| assistant_char_count | 20250716_openhands_kimi_k2 | 0.325 | -0.375 |
| n_assistant_messages | 20250728_zai_glm4-5 | 0.361 | -0.424 |

### Output Files

```
chris_output/experiment_b/frontier_analysis/
├── correlation_comparison.csv   # All features with correlations
├── findings.md                  # Summary table
├── rubric/*.png                 # Rubric feature scatter plots
├── llm_judge/*.png              # LLM judge mean+SE plots
├── trajectory/*.png             # Trajectory scatter plots (best agent)
└── x0_days.png                  # Date scatter plot
```

## Related Experiments

- **Experiment A**: Prior validation - tests how well static task features predict difficulty on held-out tasks
- **Experiment SAD-IRT**: Uses trajectory information for frontier difficulty prediction

## References

- [IRT Models Documentation](../docs/IRT_MODELS.md)
- [Research Proposal](../chris%20proposal.md) - Section 3.2

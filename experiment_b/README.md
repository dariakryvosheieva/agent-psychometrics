# Experiment B: Frontier Task Difficulty Prediction

Predict difficulty of **frontier tasks** (tasks only solvable by newer models) using methods that do NOT have access to post-frontier agents.

## Overview

**Research Question**: Can we predict the difficulty of tasks that are currently beyond the capability of existing models, using only information available before those models were released?

**Setting**:
- **Date-based split**: Pre-frontier vs Post-frontier agents (by release date)
- **No data leakage**: Predictions made using only pre-frontier information; post-frontier agents only used for evaluation

## Evaluation Metrics

The two primary metrics reported are:
1. **ROC-AUC**: Ability to rank (agent, task) pairs by solve probability on frontier tasks
2. **MAE (days)**: Mean absolute error in predicting when tasks become solvable

## Frontier Task Definitions

Two definitions of "frontier task" are supported (both evaluated by default):

1. **Pass-rate based** (`passrate`): Tasks with ≤10% pre-frontier pass rate AND >10% post-frontier pass rate
2. **IRT-based** (`irt`): Tasks where NO pre-frontier agent has ≥50% solve probability under IRT

## Quick Start

```bash
source .venv/bin/activate

# Run on SWE-bench (default) - evaluates both frontier definitions
python -m experiment_b.compare_methods

# Run on TerminalBench
python -m experiment_b.compare_methods --dataset terminalbench

# Run on SWE-bench Pro
python -m experiment_b.compare_methods --dataset swebench_pro

# Run with only one frontier definition
python -m experiment_b.compare_methods --frontier_definitions passrate

# Disable date forecasting (faster)
python -m experiment_b.compare_methods --no_forecast_dates

# Save results to CSV
python -m experiment_b.compare_methods --output_csv results.csv
```

## Results (2026-01-20)

### SWE-bench Verified

**Cutoff**: 2025-05-01 | **Pre-frontier agents**: 76 | **Post-frontier agents**: 55

#### Pass-rate Definition (47 frontier tasks)

| Method | ROC-AUC |
|--------|---------|
| Oracle (upper bound) | 0.8439 |
| SAD-IRT (best) | 0.8036 |
| Feature-IRT (Embedding) | 0.7744 |
| Baseline IRT (pre-frontier only) | 0.7600 ± 0.011 |
| LLM Judge + Ridge | 0.7481 |
| Feature-IRT (LLM Judge) | 0.7480 |
| Embedding + Ridge | 0.7475 |

#### IRT Definition (36 frontier tasks)

| Method | ROC-AUC |
|--------|---------|
| Oracle (upper bound) | 0.7810 |
| SAD-IRT (best) | 0.7315 |
| Feature-IRT (LLM Judge) | 0.7302 |
| LLM Judge + Ridge | 0.7294 |
| Embedding + Ridge | 0.7291 |
| Feature-IRT (Embedding) | 0.7284 |
| Baseline IRT (pre-frontier only) | 0.6966 ± 0.014 |

### SWE-bench Pro

**Cutoff**: 2025-09-01 | **Pre-frontier agents**: 10 | **Post-frontier agents**: 4

#### Pass-rate Definition (60 frontier tasks)

| Method | ROC-AUC |
|--------|---------|
| Oracle (upper bound) | 0.7975 |
| Embedding + Ridge | 0.7412 |
| Baseline IRT (pre-frontier only) | 0.6500 |
| LLM Judge + Ridge | TBD (see Phase 2) |

**Note**: SWE-bench Pro uses public release dates for agents (stored in `data/swebench_pro_agent_dates.json`) since agent names don't follow the YYYYMMDD prefix convention. LLM judge features are pending extraction (see `PHASE2_SWEBENCH_PRO_PROMPT.md`).

### TerminalBench

**Cutoff**: 2025-11-05 | **Pre-frontier agents**: 48 | **Post-frontier agents**: 35

#### Pass-rate Definition (18 frontier tasks)

| Method | ROC-AUC |
|--------|---------|
| Oracle (upper bound) | 0.8224 |
| LLM Judge + Ridge | 0.7457 |
| Feature-IRT (Embedding) | 0.7427 |
| Feature-IRT (LLM Judge) | 0.7414 |
| Baseline IRT (pre-frontier only) | 0.7289 ± 0.012 |
| Embedding + Ridge | 0.7237 |

#### IRT Definition (16 frontier tasks)

| Method | ROC-AUC |
|--------|---------|
| Oracle (upper bound) | 0.8088 |
| Feature-IRT (Embedding) | 0.7426 |
| LLM Judge + Ridge | 0.7376 |
| Feature-IRT (LLM Judge) | 0.7315 |
| Baseline IRT (pre-frontier only) | 0.7287 ± 0.011 |
| Embedding + Ridge | 0.7200 |

**Key observations**:
- **Feature-IRT (Embedding) consistently outperforms Baseline IRT** in ROC-AUC
- **SAD-IRT** (using trajectory information) achieves strong results on SWE-bench
- **Baseline IRT error bars** (± 1 std) from 30 random seed runs show typical variance from IRT training randomness; results use seed=42 for reproducibility
- **Methods without their own IRT** (Embedding + Ridge, LLM Judge + Ridge) cannot produce date forecasts (see Date Forecasting section below)

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

### ROC-AUC with Scale Alignment
1. **Identify anchor tasks**: Tasks with 10-90% pass rate in BOTH pre- and post-frontier groups
2. **Fit affine transformation**: `oracle_β = slope × predicted_β + intercept` on anchors
3. **Compute probabilities**: For each (post-frontier agent, frontier task): `P(success) = sigmoid(θ_oracle - β_shifted)`
4. **Calculate ROC-AUC**: Compare predicted probabilities to actual responses

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
├── compare_methods.py        # Thin orchestration (~275 lines)
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
| `chris_output/experiment_a_swebench_pro/llm_judge_features/` | LLM judge features (pending) |
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
--dataset              Dataset to use: swebench (default), swebench_pro, or terminalbench
--frontier_definitions Space-separated list: 'passrate' 'irt' (default: both)
--no_forecast_dates    Disable date forecasting evaluation (faster, but no MAE metric)
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

## Related Experiments

- **Experiment A**: Prior validation - tests how well static task features predict difficulty on held-out tasks
- **Experiment SAD-IRT**: Uses trajectory information for frontier difficulty prediction

## References

- [IRT Models Documentation](../docs/IRT_MODELS.md)
- [Research Proposal](../chris%20proposal.md) - Section 3.2

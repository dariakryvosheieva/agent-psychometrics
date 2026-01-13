# SWE-bench IRT Analysis

## Overview

This repository applies Item Response Theory (IRT) to SWE-bench Verified benchmark data to model agent abilities and task difficulties. The goal is to go beyond simple pass-rate rankings by fitting IRT models that estimate latent difficulty parameters.

## Repository Structure

```
model_irt/
├── py_irt/                     # IRT library (with ClippedAdam support)
├── experiments/                # SWE-bench experiments repo (has own .git)
│   ├── evaluation/             # Agent results
│   └── analysis/               # Analysis scripts
│
├── swebench_irt/               # IRT model training code
│   ├── train.py                # Train 1D-6D IRT models (supports 1PL and 2PL)
│   ├── train_rep.py            # Multi-seed training
│   ├── compare_dims.py         # AIC/BIC model comparison
│   ├── prep_swebench.py        # Build response matrix
│   └── ...
│
├── experiment_a/               # Experiment A: Prior validation (IRT AUC)
│   ├── config.py               # Configuration parameters
│   ├── data_loader.py          # Load IRT params, responses, task splits
│   ├── difficulty_predictor.py # Predictor protocol + implementations
│   ├── irt_evaluation.py       # AUC computation using 1PL IRT
│   ├── baselines.py            # Agent-only, task-only baselines
│   └── train_evaluate.py       # Main evaluation pipeline
│
├── experiment_b/               # Experiment B: Posterior difficulty prediction
│   ├── config.py               # Configuration parameters
│   ├── data_splits.py          # Agent/task splitting logic
│   ├── prior_model.py          # Linear prior on task features
│   ├── posterior_model.py      # Prior + trajectory correction
│   ├── trajectory_features.py  # Extract features from trajectories
│   └── train_evaluate.py       # Main training/evaluation pipeline
│
├── llm_judge/                  # LLM-as-judge for difficulty prediction
│   ├── llm_judge.py            # Direct LLM feature extraction
│   ├── predict_difficulty.py   # Heuristic feature prediction
│   └── lunette_batch_grading.py # Batch grading via Lunette
│
├── lunette_utils/              # Lunette shared utilities
│   ├── dummy_solver.py         # Dummy solver for cost measurement
│   ├── dummy_swebench_task.py  # SWE-bench task with dummy solver
│   └── lunette_analysis.py     # Lunette-based analysis utilities
│
├── trajectory_upload/          # Trajectory conversion and upload to Lunette
│   ├── trajectory_converter.py # Convert trajectories to unified format
│   └── lunette_reupload_with_metadata.py # Upload with SWE-bench metadata
│
├── clean_data/                 # Trained IRT models (primary location)
│   └── swebench_verified_20251115_full/
│       ├── 1d/                 # 2PL model (a, b parameters)
│       └── 1d_1pl/             # 1PL model (b only)
│
├── chris_output/               # Outputs and analysis results
│   ├── clean_data/             # Response matrices
│   ├── figures/                # Visualizations
│   └── experiment_b/           # Experiment B results
│
├── trajectory_data/            # Downloaded trajectory data
│   └── unified_trajs/          # Unified format trajectories (78 agents)
│
├── tests/                      # Test suite
├── requirements.txt            # Python dependencies
└── CLAUDE.md                   # This file
```

## Current Research Focus

**LLM-as-a-judge to predict 1D task difficulty.**

Model selection (AIC/BIC) indicates that a 1D IRT model best fits the SWE-bench data, suggesting agent performance can be explained by a single latent ability dimension. The next step is to use an LLM to predict the fitted difficulty parameter `b` for each task based on the problem description, enabling:
- Difficulty estimation for new tasks without running agents
- Understanding what textual features correlate with difficulty
- Potential task filtering/stratification for benchmarking

### Model Selection Results

| Model | Log-Lik | # Params | AIC | BIC |
|-------|---------|----------|-----|-----|
| **1D** | -17,481 | 1,123 | **37,209** | **47,346** |
| 2D | -17,175 | 2,246 | 38,842 | 59,116 |
| 3D | -16,867 | 3,369 | 40,471 | 70,882 |

1D is best by both AIC and BIC, indicating additional dimensions don't provide enough improvement to justify the extra parameters.

## Bug Fix: 1D Discrimination Parameter

**Fixed (2025-01-06):** The original code incorrectly applied `np.exp()` to 1D discrimination parameters:

```python
# WRONG (was in original code):
discriminations = [np.exp(i) for i in trainer.best_params["disc"]]

# CORRECT (fixed):
discriminations = list(trainer.best_params["disc"])
```

**Why this was wrong:**
- The `TwoParamLog` model uses `Normal` distribution for discrimination (not `LogNormal`)
- The guide parameter `loc_slope` stores raw values, not log-transformed values
- The MIRT code was correct (had comment: "do NOT exponentiate") but 1D was inconsistent

**Impact:** Before fix, 1D appeared much worse than 2D (AIC 57,799 vs 38,846). After fix, 1D is actually best (AIC 37,209).

## Data Pipeline

```
experiments/evaluation/verified/<agent>/results/results.json
         ↓
    swebench_irt/prep_swebench.py (--complete_matrix --cutoff_date 20251115)
         ↓
chris_output/clean_data/swebench_verified/swebench_verified_20251115_full.jsonl
         ↓
    swebench_irt/train.py (--dims 1 --model 1pl/2pl)
         ↓
clean_data/swebench_verified_20251115_full/{1d,1d_1pl}/
    ├── items.csv   (a, b for 2PL; b only for 1PL)
    └── abilities.csv (theta per agent)
         ↓
    swebench_irt/compare_dims.py (AIC/BIC comparison)
```

## Key Files

| File | Purpose |
|------|---------|
| `swebench_irt/train.py` | Train 1D-6D IRT models via py_irt (supports 1PL and 2PL) |
| `swebench_irt/train_rep.py` | Train with multiple random seeds for stability analysis |
| `swebench_irt/compare_dims.py` | Compare models via AIC/BIC, optional 2D scatter |
| `swebench_irt/prep_swebench.py` | Build JSONL response matrix from experiments repo |
| `experiment_a/train_evaluate.py` | Run Experiment A: prior validation (IRT AUC) |
| `experiment_b/train_evaluate.py` | Run Experiment B: posterior difficulty prediction |
| `llm_judge/llm_judge.py` | Direct LLM feature extraction |
| `llm_judge/predict_difficulty.py` | Heuristic feature prediction |
| `py_irt/` | Local fork of py_irt with Multidim2PL model |
| `tests/test_irt_pipeline.py` | 26 tests covering preprocessing, training, evaluation |

## Model Details

### 1D 2PL (Two-Parameter Logistic)

Probability of agent j solving task i:

```
P(Y=1) = sigmoid( a_i * (theta_j - b_i) )
```

Where:
- `theta_j` — agent ability (single scalar)
- `a_i` — item discrimination (how well the item differentiates ability levels)
- `b_i` — item difficulty (ability level needed for 50% chance of success)

### Training Configuration

**1D (hierarchical priors):**
- Learning rate: 0.1 with decay 0.9999
- Epochs: 5000

**MIRT (2D+):**
- Learning rate: 0.003 (reduced for stability)
- LR decay: 1.0 (disabled)
- Gradient clipping: clip_norm=5 via ClippedAdam
- Initializers: difficulty_from_accuracy + mirt_pca

## Output Structure

```
clean_data/swebench_verified_20251115_full/
├── 1d/               # 2PL model
│   ├── items.csv     # a, b, a_std, b_std (500 tasks)
│   └── abilities.csv # theta, theta_std (130 agents)
└── 1d_1pl/           # 1PL (Rasch) model
    ├── items.csv     # b, b_std only (500 tasks)
    └── abilities.csv # theta, theta_std (130 agents)
```

## Current Dataset

- **130 agents** (cutoff: 2025-11-15)
- **500 tasks** (SWE-bench Verified)
- **65,000 observations** (complete matrix, missing → 0)
- **78 agents** with unified trajectories (for Experiment B)

## Quick Start

```bash
# Activate environment
source .venv/bin/activate

# Train 1D 2PL model
python swebench_irt/train.py \
    --data_path chris_output/clean_data/swebench_verified/swebench_verified_20251115_full.jsonl \
    --dims 1 \
    --model 2pl \
    --output_dir clean_data/swebench_verified_20251115_full \
    --epochs 5000

# Train 1D 1PL (Rasch) model
python swebench_irt/train.py \
    --data_path chris_output/clean_data/swebench_verified/swebench_verified_20251115_full.jsonl \
    --dims 1 \
    --model 1pl \
    --output_dir clean_data/swebench_verified_20251115_full \
    --epochs 5000

# Run Experiment B
python -m experiment_b.train_evaluate

# Run tests
pytest tests/test_irt_pipeline.py -v
```

## Next Steps: LLM Difficulty Prediction

Use the fitted 1D difficulty parameters as labels for supervised learning:

```python
import pandas as pd
items = pd.read_csv("clean_data/swebench_verified_20251115_full/1d/items.csv", index_col=0)

# items.index contains task IDs like "django__django-12345"
# items["b"] contains the fitted difficulty (-2 to +5 range typically)

# Goal: Train LLM to predict items["b"] from task description
```

Potential approaches:
1. **Zero-shot prompting**: Ask LLM to rate difficulty 1-10, correlate with `b`
2. **Few-shot prompting**: Provide examples with known `b` values
3. **Fine-tuning**: Train on (task_description, b) pairs

### LLM Judge Approaches

Two approaches have been tested for predicting IRT difficulty from task features:

#### 1. Direct LLM Feature Extraction (llm_judge/llm_judge.py)

Prompts an LLM to extract discrete features (1-5 scales) from the problem statement and gold patch. Features include: fix_in_description, problem_clarity, error_message_provided, reproduction_steps, fix_locality, domain_knowledge_required, fix_complexity.

**Status:** Code exists but no saved results. Run with:
```bash
python llm_judge/llm_judge.py --num_tasks 50 --output_path chris_output/llm_judge/features.csv
```

#### 2. Lunette-Based Grading (lunette_utils/lunette_analysis.py)

Uses Lunette's GradingPlan API to analyze task + agent trajectory together. The judge has environment access and can examine the actual codebase.

**Results (n=39 tasks, 2025-01-07):**
| Feature | Correlation | p-value |
|---------|-------------|---------|
| fix_in_description | -0.57 | 0.001 |
| problem_clarity | -0.38 | 0.025 |
| fix_complexity | +0.34 | 0.046 |

See `chris_output/lunette/correlation_summary.csv` for full results.

### Heuristic Feature Prediction (llm_judge/predict_difficulty.py)

Using linear models (Ridge/Lasso) with heuristic features:
- **R² = 0.14**, correlation = 0.405
- Top predictors: repo effects (scikit-learn, pylint hardest), human labels, test complexity
- ~86% of variance unexplained by heuristics → semantic understanding needed

### Lunette Integration

[Lunette](https://docs.lunette.dev/) is a platform for evaluating AI agents with **environment access** - judges can examine the actual codebase, run commands, and test hypotheses.

#### Setup

```bash
pip install lunette-sdk
# Configure API key
mkdir -p ~/.lunette
echo '{"api_key": "your-key-here"}' > ~/.lunette/config.json
```

#### Batch Upload System (with SWE-bench Metadata)

**IMPORTANT:** Always use `lunette_reupload_with_metadata.py` for uploading trajectories. This script includes proper SWE-bench metadata (repo, patch, test_patch, version, etc.) in each trajectory.

```bash
# Upload all agents with proper metadata
python trajectory_upload/lunette_reupload_with_metadata.py

# Upload specific agents
python trajectory_upload/lunette_reupload_with_metadata.py --agents 20240620_sweagent_claude3.5sonnet

# Dry run to see what would be uploaded
python trajectory_upload/lunette_reupload_with_metadata.py --dry_run

# Use smaller batch size for agents with large trajectories
python trajectory_upload/lunette_reupload_with_metadata.py --batch_size 10
```

**Key features:**
- Includes full SWE-bench metadata: repo, patch, test_patch, version, created_at, hints_text, base_commit, FAIL_TO_PASS, PASS_TO_PASS
- Automatic retry with smaller batch sizes when encountering 413 errors
- Deletes existing runs before re-uploading (safe to re-run)
- Creates tracking files at `trajectory_data/unified_trajs/<agent>/_lunette_uploads.json`

**Current upload status (as of 2026-01-10):**
- 77/78 agents uploaded with proper metadata
- 36,010 trajectories total
- 1 agent (`20250118_codeshellagent_gemini_2.0_flash_experimental`) partially uploaded due to server issues

**Tracking file structure:**
```json
{
  "agent": "20240620_sweagent_claude3.5sonnet",
  "run_ids": ["d1c72b3d-...", "51528284-...", ...],
  "uploaded_at": "2026-01-10T15:58:57.853428",
  "has_swebench_metadata": true,
  "trajectory_count": 500,
  "trajectories": [
    {
      "task_id": "astropy__astropy-12907",
      "trajectory_id": "09888d9d-...",
      "run_id": "d1c72b3d-...",
      "resolved": false,
      "message_count": 91
    }
  ]
}
```

#### Batch Grading System

Grade uploaded trajectories to evaluate how features predict IRT difficulty and discriminate between agents (see [LUNETTE_BATCH_GRADING.md](trajectory_upload/LUNETTE_BATCH_GRADING.md)):

```bash
# Dry run to see execution plan
python llm_judge/lunette_batch_grading.py --dry_run --n_tasks 50 --n_agents 3

# Grade 50 tasks across 3 agents (~$75)
python llm_judge/lunette_batch_grading.py --n_tasks 50 --n_agents 3

# Grade specific agents
python llm_judge/lunette_batch_grading.py \
  --agents 20240620_sweagent_claude3.5sonnet 20240728_sweagent_gpt4o \
  --n_tasks 50
```

**Features extracted (12 total):**
- **Task-intrinsic (7)**: fix_in_description, problem_clarity, error_message_provided, reproduction_steps, fix_locality, domain_knowledge_required, fix_complexity
- **Trajectory-based (5)**: agent_declared_success_wrongly, agent_looping, agent_expressed_uncertainty, agent_wrong_file_focus, agent_gave_up_early

**Two types of analysis:**
1. **Correlation analysis**: Which features predict task difficulty? (Pearson r with IRT `b`)
2. **Discrimination analysis**: Which features differentiate agents? (between_agent_var / within_agent_var)

**Output files** (saved to `chris_output/lunette_grading/`):
- `grading_results_TIMESTAMP.csv` - Raw feature scores per trajectory
- `correlations_TIMESTAMP.csv` - Correlation with IRT difficulty
- `discrimination_TIMESTAMP.csv` - Feature discrimination ratios

#### Accessing Uploaded Trajectories

```python
import json
from pathlib import Path

# Load tracking file
agent_dir = Path("trajectory_data/unified_trajs/20240620_sweagent_claude3.5sonnet")
with open(agent_dir / "_lunette_uploads.json") as f:
    data = json.load(f)

# Iterate through trajectories
for traj in data["trajectories"]:
    print(f"{traj['task_id']} -> {traj['run_id']}")
```

#### Running SWE-bench Evals

```bash
# Run SWE-bench mini (50 tasks) with Lunette sandbox
lunette eval swebench --model anthropic/claude-sonnet-4-20250514 --limit 1
lunette eval swebench --model openai/gpt-4o-mini --limit 1

# This runs inspect_evals/swe_bench_verified_mini with:
#   --sandbox lunette
#   --sandbox_config_template_file (Lunette's swebench preset)
```

#### Lunette Grading Results (2025-01-07)

Graded 50 SWE-bench tasks using Lunette's GradingPlan API to extract difficulty features. The judge analyzes both the task description and agent trajectory to predict IRT difficulty.

**Correlations with IRT Difficulty (statistically significant, p < 0.05):**

| Feature | Correlation | n | Interpretation |
|---------|-------------|---|----------------|
| fix_in_description | -0.57 | 30 | More hints in description → easier |
| score_aggregate | +0.47 | 25 | Higher Lunette score → harder |
| problem_clarity | -0.38 | 34 | Clearer problems → easier |
| fix_complexity | +0.34 | 36 | Complex fixes → harder |

**Non-significant features:** reproduction_steps (+0.29), domain_knowledge_required (+0.24), fix_locality (+0.11), error_message_provided (+0.05)

**Trajectory signals:** agent_looping, agent_declared_success_wrongly, etc. had zero variance (all successful runs), so no correlation computed.

**Key finding:** `fix_in_description` is the strongest predictor (r=-0.57, p=0.001). When the problem statement contains hints about the fix, tasks are significantly easier for agents.

#### Lunette Grading API

**Retrieving Runs:**
```python
import httpx
import json

with open("~/.lunette/config.json") as f:
    api_key = json.load(f)["api_key"]

with httpx.Client(base_url="https://lunette.dev/api", headers={"X-API-Key": api_key}) as client:
    r = client.get("/runs/")  # TRAILING SLASH REQUIRED
    runs = r.json()
    # Returns: [{"id": "...", "task": "...", "model": "...", "trajectory_count": N}, ...]
```

**Running Grading:**
```python
import asyncio
from lunette import LunetteClient
from lunette.analysis import GradingPlan

async def grade_trajectory(run_id: str):
    async with LunetteClient() as client:
        results = await client.investigate(
            run_id=run_id,
            plan=GradingPlan(
                name="task-difficulty",
                prompt="""Analyze this SWE-bench task to predict difficulty...
                (see lunette_utils/lunette_analysis.py for full prompt)"""
            ),
            limit=1,
        )
        # Returns structured scores or aggregate explanation

asyncio.run(grade_trajectory("your-run-id"))
```

**Deleting Runs:**
```python
import httpx
import json

with open("~/.lunette/config.json") as f:
    api_key = json.load(f)["api_key"]

with httpx.Client(base_url="https://lunette.dev/api", headers={"X-API-Key": api_key}) as client:
    run_id = "your-run-id-here"
    r = client.delete(f"/runs/{run_id}")  # NO trailing slash for DELETE
    # Returns: {"message": "Run ... deleted successfully", "deleted_trajectories": N}
```

**Note:** GET requests require trailing slash (`/runs/`), DELETE requests must NOT have trailing slash (`/runs/{id}`).

#### Downloaded Trajectories

Trajectories from existing SWE-bench experiments are available in the gitignored `experiments/` folder:

```bash
cd experiments
python -m analysis.download_logs evaluation/verified/20240620_sweagent_claude3.5sonnet --only_trajs
# Downloads 500 .traj files to evaluation/verified/.../trajs/
```

#### Related Files

| File | Purpose |
|------|---------|
| `llm_judge/llm_judge.py` | Direct LLM feature extraction |
| `lunette_utils/lunette_analysis.py` | Lunette-based grading utilities |
| `trajectory_upload/lunette_reupload_with_metadata.py` | Upload trajectories with SWE-bench metadata |
| `llm_judge/lunette_batch_grading.py` | Batch grading of uploaded trajectories |
| `chris_output/lunette/correlation_summary.csv` | Correlation results from grading |
| `chris_output/lunette_grading/` | Batch grading results directory |

### Experiment B: Failure-Informed Posterior Difficulty Prediction

Uses failing agent trajectories to improve IRT difficulty prediction. The key insight is that trajectory features from weak models (that fail on a task) provide signal about task difficulty beyond what can be inferred from static task features alone.

#### Model Architecture

```
posterior_difficulty_i = prior(x_i) + psi^T * avg_j[f(tau_ij)]

Loss = MSE(predicted, ground_truth_b) + lambda * ||psi||^2
```

Where:
- `prior(x_i)` = simple linear model on task features (repo, text length, etc.)
- `psi` = single set of learned weights (same for all trajectories)
- `f(tau_ij)` = feature vector from trajectory of agent j on task i
- `avg_j` = average across all weak model trajectories for task i

#### Directory Structure

```
experiment_b/
├── __init__.py
├── config.py                # Configuration parameters (paths, thresholds)
├── data_splits.py           # Agent and task splitting logic
├── prior_model.py           # Simple linear prior on task features
├── trajectory_features.py   # Extract features from trajectories
├── posterior_model.py       # Prior + trajectory correction
└── train_evaluate.py        # Main training/evaluation pipeline
```

#### Module Details

| File | Purpose |
|------|---------|
| `config.py` | `ExperimentConfig` dataclass with paths, split fractions, thresholds |
| `data_splits.py` | Split agents by date (M1/M2/M3), filter tasks by pass rate |
| `prior_model.py` | Ridge regression on task features (problem_len, patch_len, repo). **Note:** Uses StandardScaler, so coefficients in output JSON are on standardized scale, not raw scale |
| `trajectory_features.py` | Extract 5 features: message_count, total_chars, assistant_ratio, message_length, resolved_rate |
| `posterior_model.py` | Learn psi coefficients to correct prior predictions using trajectory features |
| `train_evaluate.py` | Full pipeline: load data → split → train prior → train posterior → evaluate |

#### Data Splitting Strategy

**Agent Splitting (by submission date):**
- Parse YYYYMMDD prefix from agent names
- M1 (oldest 40%): Used for training posterior on D_train
- M2 (middle 40%): Used for evaluating posterior on D_valid
- M3 (newest 20%): Held out for future testing

**Task Splitting (by empirical pass rate):**
- D_train: Tasks with ≤20% pass rate among M1, but >30% among M2 (tasks that got easier)
- D_valid: Tasks with ≤20% pass rate among M2, but >30% among M3 (disjoint from D_train)
- Threshold (20%) is configurable via `--weak_threshold`

#### Usage

```bash
# Activate environment
source .venv/bin/activate

# Run with defaults
python -m experiment_b.train_evaluate

# Adjust pass rate threshold
python -m experiment_b.train_evaluate --weak_threshold 0.1

# Dry run (show config without running)
python -m experiment_b.train_evaluate --dry_run

# Custom output directory
python -m experiment_b.train_evaluate --output_dir chris_output/experiment_b_v2
```

#### Configuration

```python
@dataclass
class ExperimentConfig:
    # Data paths (relative to project root)
    items_path: Path = Path("clean_data/swebench_verified_20251115_full/1d/items.csv")
    responses_path: Path = Path("chris_output/clean_data/swebench_verified/swebench_verified_20251115_full.jsonl")
    trajectories_dir: Path = Path("trajectory_data/unified_trajs")
    output_dir: Path = Path("chris_output/experiment_b")

    # Agent splitting
    m1_fraction: float = 0.4  # Oldest 40%
    m2_fraction: float = 0.4  # Middle 40%
    # M3 = remaining 20%

    # Task selection
    weak_threshold: float = 0.2  # Max pass rate for "hard" tasks
    strong_min_improvement: float = 0.1  # Min improvement for strong group

    # Model parameters
    prior_alpha: float = 1.0  # Ridge alpha for prior
    posterior_alpha: float = 1.0  # Ridge alpha for psi
```

#### Trajectory Features

5 simple features extracted from each trajectory and averaged across agents:

| Feature | Description |
|---------|-------------|
| `avg_message_count` | Average number of messages in trajectory |
| `avg_total_chars` | Average total character count |
| `avg_assistant_ratio` | Ratio of assistant messages to total |
| `avg_message_length` | Average characters per message |
| `resolved_rate` | Fraction of trajectories that resolved the task |

#### Feature Sources

Experiment B supports three feature sources via `--feature_source`:

| Source | Description | Files |
|--------|-------------|-------|
| `simple` (default) | Basic trajectory stats (message count, chars, resolve rate) | `trajectory_features.py` |
| `lunette` | Lunette API grading features (14 features) | `lunette_features.py`, `compute_lunette_features.py` |
| `llm_judge` | Direct LLM API grading (14 features, same as Lunette) | `llm_judge_features.py`, `compute_llm_judge_features.py` |

**LLM Judge Features (14 total):**
- Primary: `llm_judge_difficulty_score` (0-1)
- Competencies (1-4): `backtracking_exploration`, `task_decomposition`, `observation_reading`, `self_verification`
- Failure modes (0-1): `localization_failure`, `strategy_defect`, `implementation_defect`, `incomplete_repair`, `verification_failure`
- Trajectory signals (0-1): `agent_looping`, `agent_gave_up_early`, `agent_wrong_focus`, `context_overflow`

**Usage:**
```bash
# Pre-compute LLM judge features (uses Claude Opus 4.5 by default)
python -m experiment_b.compute_llm_judge_features --dry_run  # See what would be computed
python -m experiment_b.compute_llm_judge_features --limit 50  # Compute 50 features

# Run experiment with LLM judge features
python -m experiment_b.train_evaluate --feature_source llm_judge
```

#### Output

Results saved to `chris_output/experiment_b/experiment_b_results.json`:

```json
{
  "split": {
    "m1_agents": ["agent1", ...],
    "m2_agents": ["agent2", ...],
    "m3_agents": ["agent3", ...],
    "d_train_tasks": ["task1", ...],
    "d_valid_tasks": ["task2", ...]
  },
  "prior_train": {"pearson_r": 0.031, "mse": 1.23, "n": 45},
  "posterior_train": {"pearson_r": 0.114, "mse": 1.18, "n": 45},
  "prior_valid": {"pearson_r": -0.153, "mse": 1.45, "n": 38},
  "posterior_valid": {"pearson_r": -0.087, "mse": 1.41, "n": 38},
  "psi_coefficients": {...},
  "config": {...}
}
```

#### Results (2026-01-12)

**Primary metric: AUC-ROC** - Uses IRT formula P(success) = sigmoid(θ - β) to predict agent-task outcomes. AUC avoids the restricted variance problem with correlation/RMSE on biased subsets.

Results with embedding prior (Qwen3-VL-8B embeddings, alpha=10000) and different trajectory feature sources:

| Feature Source | D_train AUC | D_valid AUC | ΔAUC | Tasks w/ Features | Notes |
|----------------|-------------|-------------|------|-------------------|-------|
| Prior only (embedding) | 0.6830 | 0.7383 | — | 119/119 | Baseline |
| + Simple features | 0.7076 | **0.7444** | **+0.0062** | 119/119 | Best result! |
| + LLM judge features | 0.6929 | 0.7215 | -0.0168 | 100/119 | Overfits on train |
| + Lunette features | 0.6846 | 0.7383 | +0.0000 | 19/119 | Too few features |

**Key finding:** Simple trajectory features (message count, chars, resolve rate) provide a small but positive improvement (+0.62% AUC). LLM judge features overfit and hurt validation. Lunette has insufficient coverage.

**Data used:**
- D_train: 119 tasks (3,689 agent-task pairs) where M1 agents (oldest 40%) mostly failed but M2 passed
- D_valid: 26 tasks (806 agent-task pairs) where M2 agents mostly failed but M3 (newest 20%) passed
- Prior RMSE: 1.12 (D_train), 1.72 (D_valid) — meaningful residuals to predict
- AUC computed using IRT formula with learned agent abilities (θ) and predicted task difficulties (β)

**Prior Residual Analysis (2026-01-12):**

The prior model (heuristic features: problem_len, patch_len, repo) has r=0.031 with ground truth on D_train tasks. Residual analysis reveals:
- High positive residuals (harder than expected): Tasks involving subtle API interactions, multi-model inheritance, pandas transform output. Example: `scikit-learn__scikit-learn-25747` (residual +4.13)
- High negative residuals (easier than expected): Tasks with large patches but simple conceptual fixes. Example: `sympy__sympy-13091` (residual -4.74, 17K char patch but just changing `return False` to `return NotImplemented`)

**Simple trajectory features correlated with residual:**
- `avg_code_blocks`: r=-0.329 (p=0.0003) - More code blocks → lower residual
- `resolve_rate`: r=-0.247 (p=0.007) - Higher resolve rate → lower residual
- `avg_msg_count`: r=-0.242 (p=0.008) - More messages → lower residual

This suggests LLM judge should focus on: (1) whether the problem has a "trick" that makes it harder than patch size suggests, (2) how confused/exploratory vs directed agents are, (3) whether agents give up or succeed with short/focused attempts.

#### ⚠️ Restriction of Range Issue (D_train correlation)

**Key Finding (2026-01-12):** The embedding prior shows r=0.093 correlation on D_train, but this is **misleading** due to the "restriction of range" statistical phenomenon.

**The Problem:**
D_train is a biased subset (tasks that M1 agents failed but M2 passed), which has much lower variance in difficulty:

| Dataset | Mean b | Std b | Range |
|---------|--------|-------|-------|
| All 500 tasks | 0.21 | 2.13 | [-3.78, 5.16] |
| D_train (119) | 0.91 | 0.81 | [-0.43, 3.58] |

When you restrict the range of values, correlation drops even if predictions are accurate!

**Proof that the prior is working:**
```
Embedding prior performance:
  On ALL 500 tasks:      r = 0.731
  On D_train (n=119):    r = 0.093  ← misleadingly low!
  On RANDOM 119 tasks:   r = 0.732  ← similar to full dataset

  D_train RMSE: 1.115  ← actually GOOD (better than overall!)
```

**Implication for Experiment B evaluation:**
- **Use RMSE, not correlation** when evaluating on D_train/D_valid
- Low correlation does NOT mean the prior is failing
- The embedding prior (r=0.63 on held-out test, RMSE≈1.1) is actually performing well

### Experiment A: Prior Validation (IRT AUC)

Evaluates how well a difficulty predictor can predict agent success on held-out tasks using the 1PL IRT model. This validates whether predicted difficulties are useful for forecasting agent performance.

#### Core Formula

```
P(success) = sigmoid(theta_j - beta_i)
```

Where:
- `theta_j` = agent ability (from 1PL IRT model)
- `beta_i` = predicted task difficulty

#### Evaluation Protocol

1. Split **tasks** (not agents) into train/test sets
2. Train difficulty predictor on train tasks
3. Predict difficulty for test tasks
4. For each (agent, task) pair: compute P(success) using IRT formula
5. Calculate AUC comparing predicted probabilities to actual outcomes

#### Usage

```bash
# Dry run to check config
python -m experiment_a.train_evaluate --dry_run

# Run without embeddings (baselines only)
python -m experiment_a.train_evaluate

# Run with pre-computed embeddings
python -m experiment_a.train_evaluate --embeddings_path out/prior_qwen3vl8b/embeddings__*.npz
```

#### Generating Embeddings

Run Daria's pipeline on Engaging cluster:

```bash
sbatch predict_question_difficulty_engaging.sh
```

This produces:
```
out/prior_qwen3vl8b/
├── embeddings__Qwen__Qwen3-VL-8B-Instruct__*.npz  # Use this for experiment_a
├── predictions.csv     # Per-task predictions
└── metrics.json        # Train/test R², Pearson r
```

#### Baseline Results (2026-01-12)

| Method | AUC | Description |
|--------|-----|-------------|
| Oracle (true b) | 0.9447 | Upper bound using ground truth IRT difficulty |
| Constant (mean b) | 0.7176 | Predict mean difficulty for all tasks |
| Agent-only | 0.7178 | Use agent's overall success rate |
| Task-only | 0.5000 | Use mean pass rate (no discrimination) |

#### Directory Structure

```
experiment_a/
├── config.py               # ExperimentAConfig dataclass
├── data_loader.py          # Load abilities, items, responses; task splitting
├── difficulty_predictor.py # DifficultyPredictor protocol + implementations
├── irt_evaluation.py       # AUC computation using 1PL IRT formula
├── baselines.py            # Agent-only, task-only baselines
└── train_evaluate.py       # Main pipeline
```

#### Output

Results saved to `chris_output/experiment_a/experiment_a_results.json`:

```json
{
  "config": {...},
  "data_summary": {"n_agents": 130, "n_tasks_total": 500, "n_train": 400, "n_test": 100},
  "oracle": {"auc": 0.9447},
  "embedding_predictor": {"auc_result": {"auc": 0.XX}, "difficulty_metrics": {...}},
  "constant_baseline": {"auc": 0.7176},
  "agent_only_baseline": {"auc": 0.7178},
  "task_only_baseline": {"auc": 0.5000}
}
```

### IRT Model Variants

#### 1PL vs 2PL Models

The codebase supports both 1-parameter logistic (1PL/Rasch) and 2-parameter logistic (2PL) IRT models:

**1PL (Rasch):**
```
P(Y=1) = sigmoid(theta_j - b_i)
```
- Only difficulty parameter `b`
- Assumes equal discrimination across all tasks

**2PL:**
```
P(Y=1) = sigmoid(a_i * (theta_j - b_i))
```
- Difficulty `b` and discrimination `a` parameters
- Allows tasks to vary in how well they differentiate ability levels

#### Training Both Models

```bash
source .venv/bin/activate

# Train 1PL model (saves to 1d_1pl/)
python swebench_irt/train.py \
    --data_path chris_output/clean_data/swebench_verified/swebench_verified_20251115_full.jsonl \
    --dims 1 \
    --model 1pl \
    --output_dir clean_data/swebench_verified_20251115_full

# Train 2PL model (saves to 1d/)
python swebench_irt/train.py \
    --data_path chris_output/clean_data/swebench_verified/swebench_verified_20251115_full.jsonl \
    --dims 1 \
    --model 2pl \
    --output_dir clean_data/swebench_verified_20251115_full
```

#### Frontier Ability Over Time (Experiment D)

Plots the frontier of model abilities over time to check if progress is linear:

```bash
python chris_output/figures/frontier_ability_over_time_comparison.py
```

**Results (2026-01-12, 130 agents, 500 tasks):**

| Model | Trend (θ/year) | R² |
|-------|----------------|-----|
| 1D 2PL | 2.94 | 0.9833 |
| 1D 1PL | 4.19 | 0.9660 |

Both models show strong linear trends, with 2PL having slightly better fit (higher R²).

**Output files:**
- `chris_output/figures/frontier_ability_over_time_2pl.png`
- `chris_output/figures/frontier_ability_over_time_1pl.png`
- `chris_output/figures/frontier_ability_1pl_vs_2pl.png` (side-by-side comparison)

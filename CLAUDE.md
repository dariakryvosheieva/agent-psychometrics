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
│   ├── train.py                # Train 1D-6D IRT models
│   ├── train_rep.py            # Multi-seed training
│   ├── compare_dims.py         # AIC/BIC model comparison
│   ├── prep_swebench.py        # Build response matrix
│   └── ...
│
├── llm_judge/                  # LLM-as-judge for difficulty prediction
│   ├── llm_judge.py            # Direct LLM feature extraction
│   ├── predict_difficulty.py   # Heuristic feature prediction
│   ├── lunette_batch_grading.py # Batch grading via Lunette
│   └── evolutionary/           # Evolutionary feature discovery (PromptBreeder-inspired)
│       ├── evolution_loop.py   # Main CLI and orchestration
│       ├── feature_generator.py # Initial feature generation
│       ├── feature_evaluator.py # Score extraction & correlation
│       ├── feature_refiner.py  # 5 mutation operators
│       └── analyze_results.py  # Visualization & reports
│
├── lunette_utils/              # Lunette shared utilities
│   ├── dummy_solver.py         # Dummy solver for cost measurement
│   ├── dummy_swebench_task.py  # SWE-bench task with dummy solver
│   ├── grading_plan.yaml       # Grading plan configuration
│   └── lunette_analysis.py     # Lunette-based analysis utilities
│
├── trajectory_upload/          # Trajectory conversion and upload to Lunette
│   ├── trajectory_converter.py # Convert trajectories to unified format
│   ├── trajectory_filter.py    # Filter trajectories by criteria
│   ├── lunette_reupload_with_metadata.py # Upload with SWE-bench metadata
│   ├── lunette_filtered_upload.py # Filtered upload
│   └── lunette_augment_mappings.py # Pre-compute task-to-run mappings
│
├── experiment_c/               # Experiment C: Lunette grading cost comparison
│   ├── experiment_c_single_task.py # Single task test
│   ├── experiment_c_rigorous.py    # Full 10-task experiment
│   └── experiment_c_cost_analysis.py # Cost analysis
│
├── chris_output/               # Outputs and trained models
│   ├── clean_data/             # Trained IRT models
│   ├── figures/                # Visualizations
│   └── difficulty_prediction/  # Prediction outputs
│
├── trajectory_data/            # Downloaded trajectory data
├── predict_question_difficulty.py  # Original difficulty prediction
├── out/                        # Original outputs
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
    swebench_irt/prep_swebench.py (--complete_matrix --cutoff_date 20250930)
         ↓
chris_output/clean_data/swebench_verified/swebench_verified_20250930_full.jsonl
         ↓
    swebench_irt/train.py (--dims 1 2 3)
         ↓
chris_output/clean_data/swebench_verified_20250930_full/{1d,2d,3d}/
    ├── items.csv   (a, b per task)
    └── abilities.csv (theta per agent)
         ↓
    swebench_irt/compare_dims.py (AIC/BIC comparison)
```

## Key Files

| File | Purpose |
|------|---------|
| `swebench_irt/train.py` | Train 1D-6D IRT models via py_irt |
| `swebench_irt/train_rep.py` | Train with multiple random seeds for stability analysis |
| `swebench_irt/compare_dims.py` | Compare models via AIC/BIC, optional 2D scatter |
| `swebench_irt/prep_swebench.py` | Build JSONL response matrix from experiments repo |
| `swebench_irt/check_matrix.py` | Verify agents/tasks/observations in JSONL |
| `llm_judge/llm_judge.py` | Direct LLM feature extraction |
| `llm_judge/predict_difficulty.py` | Heuristic feature prediction |
| `llm_judge/evolutionary/` | Evolutionary feature discovery (PromptBreeder-inspired) |
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
chris_output/clean_data/swebench_verified_20250930_full/
├── 1d/
│   ├── items.csv     # a, b, a_std, b_std (500 tasks)
│   └── abilities.csv # theta, theta_std (123 agents)
├── 2d/
│   └── ...
└── 3d/
    └── ...
```

## Current Dataset

- **123 agents** (cutoff: 2025-09-30)
- **500 tasks** (SWE-bench Verified)
- **61,500 observations** (complete matrix, missing → 0)

## Quick Start

```bash
# Activate environment
source .venv/bin/activate

# Train 1D model
python swebench_irt/train.py \
    --data_path chris_output/clean_data/swebench_verified/swebench_verified_20250930_full.jsonl \
    --dims 1 \
    --output_dir chris_output/clean_data/swebench_verified_20250930_full \
    --epochs 5000

# Compare models (if training multiple dims)
python swebench_irt/compare_dims.py \
    --results_dir chris_output/clean_data/swebench_verified_20250930_full \
    --responses_path chris_output/clean_data/swebench_verified/swebench_verified_20250930_full.jsonl \
    --output_dir chris_output/figures/swebench_verified_20250930_full

# Run tests
pytest tests/test_irt_pipeline.py -v
```

## Next Steps: LLM Difficulty Prediction

Use the fitted 1D difficulty parameters as labels for supervised learning:

```python
import pandas as pd
items = pd.read_csv("chris_output/clean_data/swebench_verified_20250930_full/1d/items.csv", index_col=0)

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

### Evolutionary Feature Discovery (llm_judge/evolutionary/)

Automatically discover and evolve LLM-extracted features that predict IRT difficulty. Inspired by **PromptBreeder** (Fernando et al., 2023).

#### How It Works

1. **Generation 0**: Sample high/low difficulty tasks, prompt LLM to hypothesize distinguishing features
2. **Evaluate**: Extract 1-5 scores for each feature × task, compute Pearson correlation with IRT `b`
3. **Select**: Keep top-K features, remove redundant ones (score correlation > 0.8)
4. **Evolve**: Apply mutation operators to surviving features
5. **Repeat**: Until correlation plateaus or max generations reached

#### Mutation Operators (PromptBreeder-inspired)

| Operator | Weight | Description |
|----------|--------|-------------|
| direct_mutation | 35% | Refine based on failure cases |
| eda_mutation | 25% | Crossover between two features |
| hypermutation | 25% | Self-referentially evolve the mutation prompt |
| zero_order | 15% | Generate novel features with context |

#### Usage

```bash
source .venv/bin/activate

# Dry run (see execution plan, no API calls)
python -m llm_judge.evolutionary.evolution_loop --dry_run

# Small test run with Sonnet (~$127)
python -m llm_judge.evolutionary.evolution_loop

# Full run with Opus 4.5 (~$600-800)
python -m llm_judge.evolutionary.evolution_loop \
    --model claude-opus-4-5-20251101 \
    --initial_features 20 \
    --top_k 10 \
    --tasks_per_eval 100 \
    --max_generations 10

# Resume from checkpoint
python -m llm_judge.evolutionary.evolution_loop --resume

# Analyze results
python -m llm_judge.evolutionary.analyze_results
```

#### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | claude-sonnet-4-20250514 | LLM model (use opus for full runs) |
| `--initial_features` | 10 | Features to generate in gen 0 |
| `--top_k` | 5 | Features to keep each generation |
| `--tasks_per_eval` | 50 | Tasks per evaluation batch |
| `--max_generations` | 5 | Maximum generations |
| `--output_dir` | llm_judge/evolutionary_results | Results directory |
| `--dry_run` | - | Show plan without running |
| `--resume` | - | Resume from checkpoint |

#### Output Structure

```
llm_judge/evolutionary_results/
├── generations/
│   ├── gen_000/
│   │   ├── features.json      # Feature definitions
│   │   ├── evaluations.json   # Per-task scores + correlations
│   │   └── summary.json       # Best correlation, surviving features
│   └── gen_001/...
├── best_features.json         # Top K across all generations
├── evolution_log.json         # Full run history + cost tracking
├── checkpoint.json            # For resume capability
├── report.txt                 # Text summary (after analyze)
├── correlation_progression.png
└── feature_genealogy.png
```

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

#### Mapping Augmentation

Pre-compute task-to-run mappings to avoid repeated API queries (see [LUNETTE_AUGMENT_MAPPINGS.md](trajectory_upload/LUNETTE_AUGMENT_MAPPINGS.md)):

```bash
# Augment all agents (recommended after upload)
python trajectory_upload/lunette_augment_mappings.py

# Augment specific agents
python trajectory_upload/lunette_augment_mappings.py --agents 20240620_sweagent_claude3.5sonnet

# Force re-augmentation
python trajectory_upload/lunette_augment_mappings.py --force
```

**What it does:**
- Queries Lunette API once per agent to determine task distribution across runs
- Stores both forward (`task_id -> run_id`) and reverse (`run_id -> [task_ids]`) mappings
- Augments each trajectory object with its `run_id` field

**Augmented tracking file structure:**
```json
{
  "agent": "20240620_sweagent_claude3.5sonnet",
  "run_ids": ["4b42140c-...", "c5dd5b11-...", ...],
  "task_to_run_map": {
    "django__django-11728": "4b42140c-d28b-4bcd-b6ad-5f09e7c8e785",
    "astropy__astropy-12907": "c5dd5b11-8b03-4082-b20b-ec0c1f012c74"
  },
  "run_to_tasks_map": {
    "4b42140c-d28b-4bcd-b6ad-5f09e7c8e785": ["django__django-11728", ...],
    "c5dd5b11-8b03-4082-b20b-ec0c1f012c74": ["astropy__astropy-12907", ...]
  },
  "task_to_run_map_updated_at": "2026-01-08T14:30:52.123456",
  "trajectories": [
    {
      "task_id": "django__django-11728",
      "trajectory_id": "0f77e97b-0dd2-406d-8b4f-54c123c6e139",
      "resolved": false,
      "message_count": 91,
      "run_id": "4b42140c-d28b-4bcd-b6ad-5f09e7c8e785"
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

**Python API:**
```python
import json
from pathlib import Path

# Load augmented tracking file
agent_dir = Path("trajectory_data/unified_trajs/20240620_sweagent_claude3.5sonnet")
with open(agent_dir / "_lunette_uploads.json") as f:
    data = json.load(f)

# Forward mapping: task_id -> run_id
task_to_run = data["task_to_run_map"]
run_id = task_to_run["django__django-11728"]

# Reverse mapping: run_id -> list of task_ids
run_to_tasks = data["run_to_tasks_map"]
tasks_in_run = run_to_tasks["4b42140c-d28b-4bcd-b6ad-5f09e7c8e785"]

# Iterate through trajectories (each has run_id already)
for traj in data["trajectories"]:
    print(f"{traj['task_id']} -> {traj['run_id']}")
```

**Data loader utility:**
```python
from llm_judge.evolutionary.lunette_data_loader import LunetteDataLoader

loader = LunetteDataLoader(
    items_path=Path("chris_output/clean_data/swebench_verified_20250930_full/1d/items.csv"),
    trajectories_dir=Path("trajectory_data/unified_trajs"),
)

# Get all agents with uploaded trajectories
agents = loader.get_agents()

# Filter by agent
trajs = loader.filter_by_agent("20240620_sweagent_claude3.5sonnet")

# Stratified sample by difficulty
sample = loader.stratified_sample(n_tasks=50, n_agents=3, seed=42)
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
| `lunette_utils/lunette_analysis.py` | Original Lunette-based grading (single-upload) |
| `trajectory_upload/lunette_reupload_with_metadata.py` | Upload trajectories with SWE-bench metadata |
| `trajectory_upload/lunette_augment_mappings.py` | Pre-compute task-to-run mappings |
| `llm_judge/lunette_batch_grading.py` | Batch grading of uploaded trajectories |
| `llm_judge/evolutionary/lunette_data_loader.py` | Data loader for uploaded trajectories |
| `trajectory_upload/LUNETTE_BATCH_GRADING.md` | Batch grading documentation |
| `trajectory_upload/LUNETTE_AUGMENT_MAPPINGS.md` | Mapping augmentation documentation |
| `chris_output/lunette/task_features_50_extracted.csv` | Raw feature values (original grading, n=39) |
| `chris_output/lunette/correlation_summary.csv` | Correlation results (original grading) |
| `chris_output/lunette_grading/` | Batch grading results directory |

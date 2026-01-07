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
│   ├── lunette_analysis.py     # Lunette-based analysis
│   └── predict_difficulty.py   # Heuristic feature prediction
│
├── chris_output/               # Outputs and trained models
│   ├── clean_data/             # Trained IRT models
│   ├── figures/                # Visualizations
│   └── difficulty_prediction/  # Prediction outputs
│
├── trajectory_data/            # Trajectory processing scripts
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

### Current LLM Judge Results (llm_judge/llm_judge.py)

Initial results from 48 tasks using Claude Sonnet to extract discrete features:

| Feature | Correlation with IRT difficulty |
|---------|--------------------------------|
| fix_complexity | +0.41 |
| domain_knowledge_required | +0.39 |
| error_message_provided | -0.33 |
| fix_in_description | -0.30 |

Best regression model (3 features): **r = 0.43** with IRT difficulty (p = 0.002)

```
predicted_b = -2.66 + 0.87×fix_complexity + 0.36×domain_knowledge - 1.40×error_message_provided
```

Limitations: Small sample (n=48), model underpredicts difficulty for hardest tasks.

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

#### Running SWE-bench Evals

```bash
# Run SWE-bench mini (50 tasks) with Lunette sandbox
lunette eval swebench --model anthropic/claude-sonnet-4-20250514 --limit 1
lunette eval swebench --model openai/gpt-4o-mini --limit 1

# This runs inspect_evals/swe_bench_verified_mini with:
#   --sandbox lunette
#   --sandbox_config_template_file (Lunette's swebench preset)
```

#### Lunette Grading API (LLM-as-Judge Replacement)

The Grading API can replace direct LLM feature extraction. It returns structured scores (0-1) with explanations.

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
                name="swebench-difficulty",
                prompt="""Grade this SWE-bench agent trajectory on:
1. Task Understanding: Did the agent correctly identify the problem?
2. Code Localization: How effectively did the agent find relevant code?
3. Fix Quality: Was the proposed fix correct and minimal?
4. Process Efficiency: Did the agent work systematically?

Provide a score (0-1) and brief explanation for each dimension."""
            ),
            limit=1,
        )

        for result in results.results:
            print(f"Dimension: {result.data['name']}")
            print(f"Score: {result.data['score']}")
            print(f"Explanation: {result.data['explanation']}")

asyncio.run(grade_trajectory("your-run-id"))
```

**Example Output:**
```
Dimension: task_understanding
Score: 0.85
Explanation: The agent correctly identified that the AuthenticationForm's username
field doesn't render the maxlength HTML attribute. The agent found the exact location
in the code (line 194 of forms.py) where max_length is set on the form field but not
on the widget's attrs dictionary...
```

**Tested Runs (2025-01-07):**
| Model | Run ID | Task Solved | Grading Score |
|-------|--------|-------------|---------------|
| claude-sonnet-4 | c891a8ba-fd7c-468e-ad80-7b99ce332196 | No | 0.85 |
| gpt-4o-mini | 36df6e4f-0960-4e7d-8a9b-3cdb330b7201 | Yes | 0.94 |

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
| `llm_judge/llm_judge.py` | Direct LLM feature extraction (working) |
| `llm_judge/lunette_analysis.py` | Lunette-based analysis (uses SDK) |
| `chris_output/llm_judge/features_50.csv` | LLM judge results for 49 tasks |

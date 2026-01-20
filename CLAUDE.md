# SWE-bench IRT Analysis

Applies Item Response Theory (IRT) to SWE-bench Verified benchmark data to model agent abilities and task difficulties.

## Research Goal

**Bayesian Inference for Agentic Evaluation of Task Difficulty** — derive calibrated estimates of task difficulty (β) using human-interpretable features.

Two regimes:
1. **Solvable Regime (Prior)**: Predict difficulty from task features alone, cheaper than running agents
2. **Frontier Regime (Posterior)**: Refine difficulty estimates using failure trajectories from weak models

See [chris proposal.md](chris%20proposal.md) for full research motivation.

## Repository Structure

```
model_irt/
├── experiment_a/           # Prior validation (IRT AUC)
├── experiment_b/           # Frontier task difficulty prediction
├── experiment_b_old/       # [ARCHIVED] Old posterior prediction approach
├── experiment_sad_irt/     # SAD-IRT model for frontier difficulty
├── swebench_irt/           # IRT model training
├── llm_judge/              # LLM-as-judge for difficulty prediction
├── lunette_utils/          # Lunette integration utilities
├── trajectory_upload/      # Trajectory conversion and upload
├── py_irt/                 # IRT library (local fork)
├── clean_data/             # Trained IRT models
├── chris_output/           # Outputs and results
├── trajectory_data/        # Downloaded trajectories (76 agents)
├── experiments/            # SWE-bench experiments repo (gitignored)
└── docs/                   # Detailed documentation
```

## Quick Start

**Important:** This project uses a Python virtual environment. Always activate it before running any Python commands:

```bash
source .venv/bin/activate

# Run Experiment A (prior validation)
python -m experiment_a.train_evaluate --dry_run

# Run Experiment B (frontier task difficulty prediction)
python -m experiment_b.compare_methods

# Train IRT model
python swebench_irt/train.py --dims 1 --model 1pl \
    --data_path clean_data/swebench_verified/swebench_verified_20251115_full.jsonl

# Run tests
pytest tests/test_irt_pipeline.py -v
```

## Current Dataset

- **130 agents** (cutoff: 2025-11-15)
- **500 tasks** (SWE-bench Verified)
- **76 agents** with unified trajectories

**Default data files:**
- Response matrix: `clean_data/swebench_verified/swebench_verified_20251120_full.jsonl`
- IRT model outputs: `clean_data/swebench_verified_20251120_full/1d/` (abilities.csv, items.csv)

## Documentation

| Document | Purpose |
|----------|---------|
| [docs/README.md](docs/README.md) | Documentation index |
| [docs/IRT_MODELS.md](docs/IRT_MODELS.md) | IRT theory, model variants |
| [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md) | Data flow |
| [experiment_a/README.md](experiment_a/README.md) | Experiment A details |
| [experiment_b/README.md](experiment_b/README.md) | Experiment B details |
| [lunette_utils/LUNETTE.md](lunette_utils/LUNETTE.md) | Lunette integration |
| [MIT_ENGAGING_SETUP.md](MIT_ENGAGING_SETUP.md) | Cluster setup |

## Key Results

### Model Selection

1D IRT model is best by both AIC and BIC (see [docs/IRT_MODELS.md](docs/IRT_MODELS.md)).

### Experiment A: Prior Validation

| Method | AUC |
|--------|-----|
| Oracle | 0.9447 |
| Embedding | 0.8333 |
| Lunette features | 0.7522 |
| Baselines | ~0.72 |

### Experiment B: Frontier Task Difficulty Prediction

Predicts difficulty of frontier tasks (tasks only solvable by newer models) using various methods. Evaluation uses ROC-AUC after projecting to oracle IRT scale.

| Method | ROC-AUC |
|--------|---------|
| Oracle | Upper bound |
| Baseline IRT | Pre-frontier only |
| Embedding + Ridge | Task embeddings |
| LLM Judge + Ridge | Semantic features |

### Experiment D: Time Horizon

Frontier ability is linear over time (R² = 0.98 for 2PL).

## Key Files

| File | Purpose |
|------|---------|
| `experiment_a/train_evaluate.py` | Run Experiment A |
| `experiment_b/compare_methods.py` | Run Experiment B |
| `swebench_irt/train.py` | Train IRT models |
| `swebench_irt/prep_swebench.py` | Build response matrix |
| `llm_judge/llm_judge.py` | LLM feature extraction |

## Development Guidelines

**Keep git status clean:**
- Always commit changes after completing a group of related modifications
- Add any new experimental output files to `.gitignore` before committing
- Goal: leave the repository in the same clean state you found it

**Fail loudly on data issues:**
- Never write code that silently skips or ignores missing data
- Raise explicit errors when expected data is missing (tasks, agents, features, dates, etc.)
- Use warnings for non-critical issues, but errors for data that should exist
- Example: if a task is expected to have embeddings but doesn't, raise an error rather than silently excluding it
- This makes debugging much easier by catching issues at their source

**MIT Engaging Cluster:**
- HuggingFace cache is stored on scratch to avoid home quota limits
- Always set this in SLURM scripts: `export HF_HOME="$HOME/orcd/scratch/.cache/huggingface"`

**OpenAI API Usage:**
- Use the new Responses API (`client.responses.create()`) instead of the older Chat Completions API (`client.chat.completions.create()`)
- Key differences: use `input=` instead of `messages=`, and access output via `response.output_text`
- See `trajectory_summarization_api/openai_client.py` for async implementation example

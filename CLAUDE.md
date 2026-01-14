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
├── experiment_b/           # Posterior difficulty prediction
├── swebench_irt/           # IRT model training
├── llm_judge/              # LLM-as-judge for difficulty prediction
├── lunette_utils/          # Lunette integration utilities
├── trajectory_upload/      # Trajectory conversion and upload
├── py_irt/                 # IRT library (local fork)
├── clean_data/             # Trained IRT models
├── chris_output/           # Outputs and results
├── trajectory_data/        # Downloaded trajectories (78 agents)
├── experiments/            # SWE-bench experiments repo (gitignored)
└── docs/                   # Detailed documentation
```

## Quick Start

```bash
source .venv/bin/activate

# Run Experiment A (prior validation)
python -m experiment_a.train_evaluate --dry_run

# Run Experiment B (posterior prediction)
python -m experiment_b.train_evaluate --dry_run

# Train IRT model
python swebench_irt/train.py --dims 1 --model 1pl \
    --data_path clean_data/swebench_verified/swebench_verified_20251115_full.jsonl

# Run tests
pytest tests/test_irt_pipeline.py -v
```

## Current Dataset

- **130 agents** (cutoff: 2025-11-15)
- **500 tasks** (SWE-bench Verified)
- **78 agents** with unified trajectories

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

### Experiment B: Posterior Prediction

Simple trajectory features provide +0.6% AUC improvement over prior alone.

### Experiment D: Time Horizon

Frontier ability is linear over time (R² = 0.98 for 2PL).

## Key Files

| File | Purpose |
|------|---------|
| `experiment_a/train_evaluate.py` | Run Experiment A |
| `experiment_b/train_evaluate.py` | Run Experiment B |
| `swebench_irt/train.py` | Train IRT models |
| `swebench_irt/prep_swebench.py` | Build response matrix |
| `llm_judge/llm_judge.py` | LLM feature extraction |

# SWE-bench IRT Analysis

Applies Item Response Theory (IRT) to agentic coding benchmark data to model agent abilities and task difficulties.

## Research Goal

Derive calibrated estimates of task difficulty (β) using human-interpretable features.

Two regimes:
1. **Solvable Regime**: Predict difficulty from task features alone, cheaper than running agents
2. **Frontier Regime** (deprioritized): Refine difficulty estimates using failure trajectories from weak models

## Repository Structure

```
model_irt/
├── experiment_new_tasks/              # Main experiment on the solvable regime (+ shared IRT infrastructure)
├── experiment_appendix_h_hard_tasks/  # Frontier task difficulty prediction
├── llm_judge_feature_extraction/      # LLM-based task feature extraction
├── swebench_irt/                      # IRT model training
├── trajectory_upload/                 # Trajectory conversion and upload
├── trajectory_summarization_api/      # Trajectory summarization
├── py_irt/                            # IRT library (local fork)
├── data/                              # Input data + IRT models (data/{dataset}/irt/)
├── chris_output/                      # Outputs and results
├── trajectory_data/                   # Downloaded trajectories (76 agents)
└── docs/                              # Detailed documentation
```

## Quick Start

**Important:** This project uses a Python virtual environment. Always activate it before running any Python commands:

```bash
source .venv/bin/activate

# Run Experiment New Tasks on all datasets
python -m experiment_new_tasks.run_all_datasets

# Run Appendix H Hard Tasks (frontier task difficulty prediction)
python -m experiment_appendix_h_hard_tasks.compare_methods

# Train IRT model
python swebench_irt/train.py --dims 1 --model 1pl \
    --data_path data/swebench_verified/responses.jsonl
```

## Datasets

All input data lives under `data/{dataset}/`:

| Dataset | Tasks | Agents | Response Matrix | IRT Model |
|---------|-------|--------|----------------|-----------|
| SWE-bench Verified | 500 | 134 | `data/swebench_verified/responses.jsonl` | `data/swebench_verified/irt/1d_1pl/` |
| GSO | 102 | 15 | `data/gso/responses.jsonl` | `data/gso/irt/1d_1pl/` |
| TerminalBench | 89 | 112 | `data/terminalbench/responses.jsonl` | `data/terminalbench/irt/1d_1pl/` |
| SWE-bench Pro | 730 | 14 | `data/swebench_pro/responses.jsonl` | `data/swebench_pro/irt/1d_1pl/` |

## Documentation

| Document | Purpose |
|----------|---------|
| [docs/README.md](docs/README.md) | Documentation index |
| [docs/IRT_MODELS.md](docs/IRT_MODELS.md) | IRT theory, model variants |
| [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md) | Data flow |
| [experiment_new_tasks/README.md](experiment_new_tasks/README.md) | Experiment New Tasks details |
| [experiment_appendix_h_hard_tasks/README.md](experiment_appendix_h_hard_tasks/README.md) | Appendix H Hard Tasks details |
| [MIT_ENGAGING_SETUP.md](MIT_ENGAGING_SETUP.md) | Cluster setup |

## Key Results

1D IRT model is best by both AIC and BIC (see [docs/IRT_MODELS.md](docs/IRT_MODELS.md)).

See [experiment_new_tasks/README.md](experiment_new_tasks/README.md) and [experiment_appendix_h_hard_tasks/README.md](experiment_appendix_h_hard_tasks/README.md) for detailed results tables.

## Key Files

| File | Purpose |
|------|---------|
| `experiment_new_tasks/run_all_datasets.py` | Run Experiment New Tasks |
| `experiment_appendix_h_hard_tasks/compare_methods.py` | Run Appendix H Hard Tasks |
| `swebench_irt/train.py` | Train IRT models |
| `swebench_irt/prep_swebench.py` | Build response matrix |
| `experiment_new_tasks/feature_source.py` | Feature source abstractions (`GroupedFeatureSource`, `RegularizedFeatureSource`) |
| `experiment_new_tasks/feature_predictor.py` | Predictors (`FeatureBasedPredictor`, `GroupedRidgePredictor`) |

## Development Guidelines

**Current context (March 2026):**
- Best available models: Claude Opus 4.6, GPT 5.3-Codex

**Keep git status clean:**
- Always commit changes after completing a group of related modifications
- Add any new experimental output files to `.gitignore` before committing
- Goal: leave the repository in the same clean state you found it

**Fail loudly — no silent fallbacks:**
- Never write code that silently skips or ignores missing data
- Raise explicit errors when expected data is missing (tasks, agents, features, dates, etc.)
- Never use fallbacks that hide problems — if a precondition is not met, throw an error
- Use warnings for non-critical issues, but errors for data that should exist
- Example: if a task is expected to have embeddings but doesn't, raise an error rather than silently excluding it
- This makes debugging much easier by catching issues at their source

**Code reuse — avoid duplication:**
- Always look for existing code that can be reused before writing new code
- Never duplicate code — extract shared logic into functions or classes
- If you find yourself copying code, refactor it into a shared module instead

**MIT Engaging Cluster:**
- **Partitions**: Use `mit_normal` for CPU tasks and `mit_normal_gpu` for GPU tasks
- **GPUs**: Up to 2 H200 GPUs available (`--gres=gpu:h200:2`)
- HuggingFace cache is stored on scratch to avoid home quota limits
- Always set this in SLURM scripts: `export HF_HOME="$HOME/orcd/scratch/.cache/huggingface"`

**Pyro IRT training fragility:**
- Pyro's hierarchical 1PL priors can hit numerical issues (`Expected parameter concentration ... of distribution Dirichlet`) non-deterministically during SVI optimization, especially with smaller datasets (e.g., TerminalBench with 89 tasks)
- This is more likely when training multiple IRT models in parallel (e.g., `run_all_datasets` with `ProcessPoolExecutor`), though the processes don't share state — it's just a resource/timing issue
- Workaround: fold IRT models are cached, so a re-run will skip the failed fold and use the cached result from a successful training
- If a fresh run fails, re-running or using `--sequential` will usually work

**OpenAI API Usage:**
- Use the new Responses API (`client.responses.create()`) instead of the older Chat Completions API (`client.chat.completions.create()`)
- Key differences: use `input=` instead of `messages=`, and access output via `response.output_text`
- See `trajectory_summarization_api/openai_client.py` for async implementation example

# Experiment New Agents

This package mirrors `experiment_new_tasks`, but the held-out unit is an agent
model/scaffold pair instead of a task.

The split is designed for the "new agents" setting:

- The held-out `(LLM, scaffold)` pair is not jointly observed in training.
- The held-out LLM is observed in at least one training pair.
- The held-out scaffold is observed in at least one training pair.

Runs are currently configured for SWE-bench Verified and Terminal-Bench 2.0.

## Quick Start

```bash
source .venv/bin/activate

python -m experiment_new_agents.run_all_datasets --sequential
```

Useful options:

```bash
python -m experiment_new_agents.run_all_datasets \
  --datasets swebench_verified terminalbench \
  --k_folds 5 \
  --n_fold_seeds 20 \
  --irt_device cuda \
  --irt_epochs 2000
```

## Methods

- `Baseline`: each task's empirical solve rate among training agents.
- `Model+Scaffold`: train-agent model+scaffold IRT parameters for the held-out
  agent pair, with train-fold item difficulties.
- `Oracle`: full standard agent IRT trained on all agents and items.

Fold IRT models are cached under each dataset output directory in `irt_splits/`.
The oracle IRT is cached under `irt_oracle/`.

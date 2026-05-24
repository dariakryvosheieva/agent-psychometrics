# Experiment New Responses

This package mirrors `experiment_new_agents`, but the held-out unit is an
individual `(agent, task)` observation instead of an agent model/scaffold pair.

The split is designed for the "new responses" setting:

- Each held-out cell is removed from train-fold IRT fitting.
- The held-out agent remains observed in at least one training cell.
- The held-out task remains observed in at least one training cell.

Runs are currently configured for SWE-bench Verified, SWE-bench Pro,
Terminal-Bench 2.0, and GSO.

## Quick Start

```bash
source .venv/bin/activate

python -m experiment_new_responses.run_all_datasets --sequential
```

Useful options:

```bash
python -m experiment_new_responses.run_all_datasets \
  --datasets swebench_verified terminalbench \
  --k_folds 5 \
  --n_fold_seeds 20 \
  --irt_device cuda \
  --irt_epochs 2000
```

## Methods

- `Standard IRT`: standard agent/item 1PL IRT trained on non-held-out
  observations; this is the baseline for paired comparisons.
- `Model+Scaffold`: model+scaffold 1PL IRT trained on non-held-out
  observations.
- `Oracle`: full standard agent/item IRT trained on all observations.

Fold IRT models are cached under each dataset output directory in `irt_splits/`.
The oracle IRT is cached under `irt_oracle/`.

`run_all_datasets` also runs a four-benchmark `Model+Scaffold` experiment over
Verified, Pro, Terminal-Bench, and GSO together. That run uses 20 split seeds by
default and does not compute a baseline, oracle, or bootstrap intervals.

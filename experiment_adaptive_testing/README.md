# Adaptive Task Selection via Fisher Information

How few benchmark tasks does an agent developer need to run before they can reliably estimate the ability of a new LLM--scaffold combination?

## Overview

We instantiate classical computerized adaptive testing (CAT) in our New Agents setting: the benchmark has been thoroughly evaluated on existing agents but the new agent has no responses yet. After each task we re-estimate the new agent's ability by MAP under a Gaussian prior and use the current estimate to pick the next, most informative task by Fisher information. The paper-specific contribution is using the IRT-Agent prediction (an LLM+scaffold ability decomposition trained on the other agents) as an informative prior, which shifts the win exactly to the small-budget regime that matters for iterative development.

**Three methods compared** (all use the fold's IRT difficulties `b_i` for ability estimation):

| Method | Task selection | Prior on `θ` |
|---|---|---|
| **Random + weak prior** | Uniform random | `N(0, 3²)` |
| **Fisher + weak prior** | Greedy max Fisher info | `N(0, 3²)` |
| **Fisher + IRT-Agent prior** (ours) | Greedy max Fisher info | `N(θ_LLM + θ_scaffold, σ_prior²)` from the held-out fold's IRT-Agent |

`σ_prior` is one global scalar: the empirical RMSE of `(θ_prior, θ_true)` across all held-out agents.

## Quick Start

```bash
source .venv/bin/activate

# Run the new-agent CAT experiment on SWE-bench Verified.
# Fold-specific IRT models are cached under
# output/experiment_new_agents_verified/irt_splits/ on first run.
python -m experiment_adaptive_testing.run_new_agent_experiment \
    --dataset swebench_verified \
    --split_seed 2 \
    --max_tasks 100 \
    --n_random_subsets 20 \
    --output_dir output/experiment_adaptive_testing/new_agent
```

`--split_seed 2` is the canonical seed: it passes the eligible-pairs filter in `stable_k_fold_split_agent_pairs` (LLMs and scaffolds with only one observation are dropped, so the held-out pair's marginals are guaranteed in training). Several other seeds fail this check; if you change the seed, expect a `RuntimeError` on bad ones.

## Output

```
output/experiment_adaptive_testing/new_agent/
├── error_curves.pdf       # Headline figure: MAE vs K, with 95% bootstrap CIs
├── error_curves.csv       # per-step mean/lo/hi for each method
├── summary.json           # n_agents, sigma_prior, prior_rmse, prior_mae
└── per_agent_records.csv  # per (fold, agent) rows: theta_true, theta_prior
```

## Headline numbers (SWE-bench Verified, split_seed=2, 30 held-out pairs)

- Prior alone (`K = 0`): MAE 0.79; matched by Random only after ~10 tasks.
- To reach MAE ≈ 0.5: Fisher + IRT-Agent needs ≈ 5 tasks; Fisher + weak ≈ 15; Random ≈ 20.
- Methods converge by `K ≈ 40`; the prior is washed out by the data.

## How It Works

Per held-out (LLM, scaffold) pair `i`:

1. Load the fold-specific IRT-Agent from `output/experiment_new_agents_verified/irt_splits/seed{S}_fold{F}of5_1d_1pl/`. This gives item difficulties `b_i`, plus `θ_LLM` and `θ_scaffold` for every LLM and scaffold in the training agents.
2. **Ground truth**: MAP estimate `θ*_i` using all 500 of the agent's binary responses against the fold's `b_i`.
3. **Prior**: `θ_prior = θ_LLM + θ_scaffold` (combine="sum"), with global `σ_prior` = RMSE of `(θ_prior, θ*_i)` across folds.
4. For `K = 1 ... max_tasks`, each method picks the next task, observes the response, re-estimates `θ̂_K` by MAP, and records `|θ̂_K - θ*_i|`.

Bands on the headline figure are **bootstrap CIs on the mean** across the 30 held-out agents (10 000 resamples), not per-agent percentiles, since the question is "is the average error of method A lower than method B" not "what's the worst-case error for an individual new agent."

## Old experiment (kept for reference)

A different framing of adaptive testing — assuming we have *no* response data for the target benchmark and must predict difficulties cross-benchmark — lives in [run_experiment.py](run_experiment.py). It corresponds to the New Benchmarks setting and was the original version of this experiment; it has been superseded by the new-agent version for the paper. See git history for the rationale.

```bash
# Old experiment (cross-benchmark difficulty prediction on SWE-bench Pro)
python -m experiment_new_benchmarks.run_all_datasets \
    --heldout_datasets swebench_pro \
    --output_dir output/experiment_adaptive_testing/ood_predictions
python -m experiment_adaptive_testing.run_experiment
```

## Files

```
experiment_adaptive_testing/
├── cat_simulation.py            # Shared: FisherSelector, MAP estimator, bands
├── new_agent_simulation.py      # New-agent CAT simulation (per-fold leave-pair-out)
├── run_new_agent_experiment.py  # CLI for the new-agent experiment (headline)
└── run_experiment.py            # CLI for the old cross-benchmark experiment
```

# Subset Extrapolation Experiment

Given a randomly chosen subset of S < N benchmark tasks and the agent responses on those tasks, predict each agent's overall % correct on the entire benchmark.

This is useful for benchmark designers who want to evaluate their agents on a cheap subset and extrapolate to the full benchmark to estimate difficulty.

## Methods compared (default sweep)

- **Empirical-subset (baseline)**: `predicted_pct[agent] = observed_successes / observed_trials`
- **Combined (Embedding + LLM-Judge)**: train fold IRT on the observed subset, fit the Grouped Ridge predictor over the concatenated DeepSeek-R1 embedding + LLM-judge feature space to predict held-out task difficulties, then `predicted_pct[agent] = (observed_successes + Σ_heldout sigmoid(θ_agent − β̂_task) · trials) / total_trials`.
- **Combined + calibration**: same as Combined, but each agent's held-out predictions are shifted by `shift_a = actual_obs_rate_a − predicted_obs_rate_a` (clipped to [0, 1]) — see the calibration discussion below.
- **Oracle (full IRT)**: same aggregation formula but with `θ` and `β` from the full IRT trained on the entire benchmark — represents the IRT model's best possible extrapolation.

Two single-source variants — `embedding` and `embedding_calibrated` (Ridge on embeddings only, with/without calibration) — are also available via `--methods` but are excluded from the default sweep because Combined dominates them across all tested cells.

### Why calibration

The fold IRT recovers per-agent training-set accuracy almost exactly (MAE ≈ 0.01 in a typical cell). But IRT parameters are only identified up to an additive shift, and Pyro's hierarchical priors settle on different shifts for the fold vs full data, so applying fold-IRT θ to tasks outside its training set introduces a per-agent location bias of a few percent. The per-agent calibration shift uses the observed data as a free anchor: it forces `mean(predicted_p on observed) = mean(actual on observed)`, while preserving the model's ranking of held-out tasks. This is what makes the IRT methods able to beat the empirical baseline at small subset sizes.

## Metric

For each (dataset, subset_size, seed) cell we compute per-agent
`|predicted_pct - actual_pct|`, then average across agents to get one MAE per
cell. We then take mean ± std of MAE across seeds (different random subset
draws) as the headline result; error bars on the plot are seed-level std.

## Sweep

- Datasets: SWE-bench Verified, SWE-bench Pro, GSO, Terminal-Bench 2.0
- Subset sizes: 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50
- Target successful seeds per cell: 20
- Excluded cell: `(terminalbench, 0.10)` — only ~9 tasks, below the comfortable training range for the fold IRT.

## Quick Start

```bash
source .venv/bin/activate

# Full sweep (default: 4 datasets, 7 sizes, 20 seeds; parallel across cells)
python -m experiment_subset_extrapolation.run_all_datasets --plot

# Fast smoke test on one dataset
python -m experiment_subset_extrapolation.run_all_datasets \
    --datasets terminalbench \
    --subset_sizes 0.25 0.50 \
    --n_seeds 3 \
    --plot

# Render just the plot from an existing summary.json
python -m experiment_subset_extrapolation.plot
```

The full sweep is designed for a 96-CPU node. See `slurm/run_subset_sweep.sbatch` for the matching job script.

## Useful flags

| Flag | Purpose |
|------|---------|
| `--datasets ...` | Subset of `{swebench_verified, swebench_pro, gso, terminalbench}` |
| `--subset_sizes A B C` | Override the size sweep |
| `--methods ...` | Subset of `{empirical, embedding, embedding_calibrated, combined, combined_calibrated, oracle}` |
| `--n_seeds N` | Target successful seeds per (dataset, size). Default: 20 |
| `--n_seeds_to_attempt M` | Seeds submitted per cell. Defaults to `ceil(1.5 * n_seeds)` so Pyro IRT failures on small datasets don't reduce the effective sample size. |
| `--seed_start S` | First seed (default 0). Useful for adding more seeds to an existing run. |
| `--max_workers W` | ProcessPoolExecutor worker count. Default 4. Set to 96 on a fat node. |
| `--sequential` | Run cells one at a time (debugging only). |
| `--plot` | Render the MAE-vs-size figure after the sweep. |

## Outputs

- `output/experiment_subset_extrapolation/summary.json` — full results (per-cell + per-(dataset, size) aggregates).
- `output/experiment_subset_extrapolation/mae_vs_subset_size.png` — 2×2 figure, one panel per dataset, lines per method with std bands.
- `output/experiment_subset_extrapolation/{dataset}/irt_splits/size{NNNN}_seed{S}/...` — cached fold IRT models, one directory per (size, seed). These can be safely deleted; cells will retrain on next run.

## Implementation notes

- **Cache safety**: `train_irt_split.get_split_cache_dir` keys its cache by `(split_seed, fold_idx, k_folds, model_type)`, not by the actual `train_tasks` list. This experiment uses many different `train_tasks` subsets for the same `split_seed`, so we route every (size, seed) cell to its own `irt_cache_dir` to avoid silent stale-cache loads. See `config.SubsetExtrapolationConfig.cache_dir_for`.
- **Empirical baseline path**: skips the fold-IRT entirely — only needs the response matrix and the canonical agent set. The empirical baseline is always evaluated even if the fold-IRT for that cell fails.
- **Pyro IRT fragility**: the hierarchical 1PL prior can fail non-deterministically on small datasets. Seeds where IRT fails are recorded but skipped from model-method aggregates; the default `n_seeds_to_attempt = 1.5 × n_seeds` provides headroom.
- **Reuse**: All predictors come from `experiment_new_tasks.pipeline.build_cv_predictors` with the default factory; no Ridge-alpha overrides.

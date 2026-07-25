# Subset Extrapolation Experiment

Given a randomly chosen subset of S < N benchmark tasks and agent responses on those tasks, predict each agent's overall % correct on the entire benchmark.

This is useful for benchmark designers who want to evaluate their agents on a cheap subset and extrapolate to the full benchmark to estimate difficulty.

## Approach

The main method (`combined_calibrated`) grounds its predictions in evaluation data from all the *other* benchmarks. Concretely, for each (target benchmark, subset of S tasks, seed) cell:

1. Train a single **multi-benchmark 1PL IRT with model+scaffold decomposition** on the union of (a) full responses from the three non-target benchmarks and (b) responses on the S observed tasks of the target benchmark. Shared `θ_model` and `θ_scaffold` parameters across benchmarks anchor target-agent abilities even when S is tiny.

2. Train a **cross-benchmark block Ridge** (DeepSeek-R1-Distill embedding block + LLM-judge feature block) on every item in the IRT training set (~1.4k items across 4 benchmarks), with the IRT's `b` values as targets. Use this to predict difficulty `b̂(task)` for each held-out target task.

3. Score each target agent: `θ_agent = θ_model + θ_scaffold`, then `p(agent, task) = sigmoid(θ_agent − b)` using the IRT's `b` for observed tasks and `b̂` for held-out tasks.

4. **Per-agent calibration shift** (constant offset): force each agent's predicted observed-rate to equal their actual observed-rate by adding `shift_a = (obs_successes − Σ obs p(a,t)·trials) / obs_trials` to every held-out prediction (clipped to [0, 1]). The agent's overall % prediction is then `(obs_successes + Σ_heldout p_cal(a,t)·trials) / total_trials`.

## Methods

- **empirical (baseline)**: `predicted_pct[agent] = observed_successes / observed_trials` — no extrapolation.
- **combined_calibrated**: the method described above.
- **oracle**: same scoring formula but with `θ_agent` and `b(task)` from the canonical full single-benchmark IRT — represents the IRT model's best-case extrapolation.

## Metric

For each (dataset, count, seed) cell we compute per-agent `|predicted_pct − actual_pct|`, then average across agents → one MAE per cell. The headline number is mean ± std MAE across seeds.

### Agent universe

The comparison only includes agents whose subject ID decomposes into `(model, scaffold)` via `swebench_irt/split_agents_model_scaffold.py` (with per-benchmark overrides for Pro / GSO / Terminal-Bench). Agents that don't decompose — typically proprietary agentic stacks whose backbone isn't in the name (e.g., `factory_code_droid`, `honeycomb`, `amazon-q-developer-agent` on Verified) — are excluded from **every** method (empirical, combined_calibrated, oracle) so all three are scored on the same population. Per-cell counts:

| Dataset | Total agents in responses.jsonl | Parseable agents (evaluated) |
|---|---|---|
| swebench_verified | 134 | 87 |
| swebench_pro | 14 | 12 |
| terminalbench | 112 | 108 |
| gso | 15 | 15 |

`n_total_agents` and `n_parseable_agents` are reported per cell in the raw summary.

## Sweep

- **Datasets**: SWE-bench Verified, SWE-bench Pro, GSO, Terminal-Bench 2.0.
- **Subset counts**: per-dataset absolute task counts, starting at 2 and stepping by 2 up to ~20% of the benchmark (the regime where the empirical baseline is already strong).
  - swebench_verified (500 tasks): 2, 4, …, 100 (50 counts)
  - swebench_pro (730 tasks): 2, 4, …, 146 (73 counts)
  - terminalbench (89 tasks): 2, 4, …, 18 (9 counts)
  - gso (102 tasks): 2, 4, …, 20 (10 counts)
- **Target successful seeds per cell**: 20.

## Quick Start

```bash
source .venv/bin/activate

# Full sweep with defaults (4 datasets, per-dataset count sweeps, 20 seeds)
python -m experiment_subset_extrapolation.run_all_datasets --plot

# Fast sanity test on one dataset
python -m experiment_subset_extrapolation.run_all_datasets \
    --datasets swebench_verified \
    --subset_counts 2 6 20 \
    --n_seeds 3 \
    --plot

# On the engaging cluster (cap 96 CPUs)
python -m experiment_subset_extrapolation.run_all_datasets --max_workers 64 --plot

# Render just the plot from an existing summary.json
python -m experiment_subset_extrapolation.plot
```

## Useful flags

| Flag | Purpose |
|------|---------|
| `--datasets ...` | Subset of `{swebench_verified, swebench_pro, gso, terminalbench}`. |
| `--subset_counts N N N` | Override: single list of absolute task counts applied to every dataset. Default: per-dataset sweep above. |
| `--methods ...` | Subset of `{empirical, combined_calibrated, oracle}`. |
| `--n_seeds N` | Target successful seeds per (dataset, count). Default: 20. |
| `--n_seeds_to_attempt M` | Seeds submitted per cell. Defaults to `ceil(1.5 × n_seeds)` (extras cover Pyro IRT failures). |
| `--seed_start S` | First seed (default 0). |
| `--max_workers W` | ProcessPoolExecutor worker count. Default 4. Engaging allows up to 96. |
| `--irt_epochs E` | Pyro SVI epochs per multi-benchmark IRT fit. Default 5000. |
| `--irt_device cpu/cuda` | IRT training device. Default `cpu` (parallelism comes from workers). |
| `--sequential` | Run cells one at a time (debugging only). |
| `--plot` | Render the MAE-vs-count figure after the sweep. |

## Outputs

- `output/experiment_subset_extrapolation/summary.json` — full results (per-cell + per-(dataset, count) aggregates).
- `output/experiment_subset_extrapolation/mae_vs_subset_count.png` — 2×2 figure, one panel per dataset, lines per method with std bands.
- `output/experiment_subset_extrapolation/{dataset}/irt_splits/count{NNNN}_seed{S}/` — per-cell IRT cache. Contents: `model_abilities.csv`, `scaffold_abilities.csv`, `items.csv`, `training_items_by_bench.json`. Re-runs reuse the cache; safe to delete to force retraining.

## Implementation notes

- **Multi-bench IRT** uses the implementation in `swebench_irt/train_model_scaffold_shared.py` (`load_multibench_split_irt_data`, `ModelScaffold1PL`).
- **Cross-benchmark Ridge** uses the existing block-Ridge stack in `utils/difficulty_prediction.py` (`_try_load_concat_embeddings_from_single_benchmark_caches`, `_build_judge_index`, `_load_judge_vector`, `_select_block_alphas_inner_cv`, `_fit_block_ridge`, `_predict_block_ridge`). No new Ridge code.
- **Cache key**: `(dataset, count, seed)` — each cell gets its own directory because the IRT training data depends on the random subset draw.
- **CPU parallelism**: each worker trains one Pyro IRT independently. `OMP_NUM_THREADS=1` etc. are set in the worker initializer to avoid oversubscription.
- **Pyro fragility**: the hierarchical 1PL prior can fail non-deterministically on small datasets. The default `n_seeds_to_attempt = 1.5 × n_seeds` provides headroom; cache hits make re-runs cheap.
- **Fail loudly**: unparseable agents are filtered out of the comparison up-front (see "Agent universe" above) — the predictor never gets asked about them and raises if it ever is. Observed tasks missing from the IRT training set raise; held-out tasks missing embeddings or judge features raise.

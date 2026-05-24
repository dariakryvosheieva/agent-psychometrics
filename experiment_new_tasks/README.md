# New Tasks Experiment 

Evaluates how well a difficulty predictor can predict agent success on held-out tasks using the 1PL IRT model.

## Overview

Given a predicted difficulty beta_hat_i and known agent ability theta_j, compute:

```
P(success) = sigmoid(theta_j - beta_hat_i)
```

Then measure AUC by comparing these predicted probabilities to actual binary outcomes across 5-fold cross-validation on tasks.

When the response file uses the per-attempt format (`{"successes": k, "trials": n}` per cell — see [Binomial IRT for Terminal-Bench](#binomial-irt-for-terminal-bench)), each cell is expanded into n labeled observations sharing the same predicted probability before computing AUC.

## Quick Start

```bash
source .venv/bin/activate

# Run all datasets (Table 2)
python -m experiment_new_tasks.run_all_datasets

# Feature source ablation (Table 3)
python -m experiment_new_tasks.run_information_ablation

# Run specific datasets only
python -m experiment_new_tasks.run_all_datasets --datasets gso terminalbench

# Plot feature source ablation bar graph (Table 3)
python -m experiment_new_tasks.plot_information_ablation
```

### Useful `run_all_datasets` flags

| Flag | Purpose |
|------|---------|
| `--datasets ...` | Subset of `{swebench_verified, swebench_pro, gso, terminalbench}` |
| `--responses_path PATH` | Override the response matrix JSONL. Supports `{dataset}` template. Use a per-attempt file to switch to binomial-likelihood IRT (see below). |
| `--per_dataset_output_dir PATH` | Override each dataset's output dir (controls the fold-IRT cache). Supports `{dataset}` template. Pair with `--responses_path` to isolate the cache from the default binary results. |
| `--llm_judge_features_path PATH` | Override LLM-judge features CSV. Supports `{dataset}` template. |
| `--embeddings_path PATH` | Override embeddings .npz. Supports `{dataset}` template. |
| `--output PATH` | Save summary table as CSV. |
| `--sequential` | Run datasets one at a time instead of in parallel. |
| `--n_fold_seeds N` | Number of independent shuffled 5-fold splits to run. Default: 20. |
| `--fold_seed_start S` | First fold split seed. The run uses consecutive seeds `S, ..., S + N - 1`. Default: 0. |
| `--n_bootstrap B` | Number of seed-level bootstrap samples for method-vs-baseline differences. Default: 10,000. |
| `--bootstrap_seed S` | Random seed for bootstrap resampling. Default: 0. |

### Binomial IRT for Terminal-Bench

For Terminal-Bench 2.0 we also publish a per-attempt response file `data/terminalbench/responses_per_attempt.jsonl` (each `(agent, task)` cell is `{"successes": int, "trials": int}` rather than 0/1). Pointing the experiment at that file trains the 1PL model with binomial likelihood (via `dist.Binomial(total_count=trials, ...)` in [py_irt](../py_irt/models/one_param_logistic.py)) and evaluates AUC over per-attempt-expanded observations. Run:

```bash
python -m experiment_new_tasks.run_all_datasets \
    --datasets terminalbench \
    --responses_path data/terminalbench/responses_per_attempt.jsonl \
    --per_dataset_output_dir output/experiment_a_terminalbench_binomial \
    --output_dir output/experiment_a_terminalbench_binomial \
    --output output/experiment_a_terminalbench_binomial/results.csv
```

The per-attempt file is produced by `python swebench_irt/prep_terminalbench.py --fetch_per_attempt_from data/terminalbench/responses.jsonl` (which fetches each agent's detail page via the `detail_url` field already stored in the binary file, so the agent set lines up exactly).

## Results

### Main Results (Table 2)

Run with: `python -m experiment_new_tasks.run_all_datasets`

| Benchmark | Baseline | Embedding | LLM-as-a-Judge | Combined | Oracle |
|-----------|----------|-----------|----------------|----------|--------|
| SWE-bench Verified | 0.7175 | 0.8237 | 0.8409 | **0.8419** | 0.9447 |
| SWE-bench Pro | 0.6569 | 0.7529 | 0.7417 | **0.7591** | 0.9183 |
| GSO | 0.7137 | 0.7610 | 0.7856 | **0.8044** | 0.9139 |
| Terminal-Bench 2.0 | 0.7335 | 0.7744 | 0.8059 | **0.8101** | 0.9317 |

### Feature Source Ablation (Table 3)

Run with: `python -m experiment_new_tasks.run_information_ablation`

LLM-as-a-Judge AUC by information level. Rows 1-4 are the cumulative additive
ablation (Problem → +Auditor → +Test → +Solution). Rows 5-6 are non-cumulative:
row 5 drops the test patch from the +Solution context, and row 6 drops the
auditor features from the +Solution pool.

| Info Level | SWE-bench Verified | SWE-bench Pro | GSO | Terminal-Bench 2.0 |
|---|---|---|---|---|
| Baseline | 0.7175 | 0.6569 | 0.7137 | 0.7335 |
| Problem | 0.7869 | 0.7184 | 0.7277 | 0.7986 |
| + Auditor | 0.7981 | 0.7373 | 0.7284 | 0.8073 |
| + Test | 0.8338 | 0.7459 | 0.7287 | 0.8073 |
| + Solution (Full) | 0.8481 | 0.7507 | 0.7952 | 0.8099 |
| Problem + Auditor + Solution (No Test) | 0.8380 | 0.7343 | 0.7675 | 0.8051 |
| Full Minus Auditor | 0.8425 | 0.7316 | 0.7920 | 0.8084 |
| Oracle | 0.9447 | 0.9183 | 0.9139 | 0.9317 |

**Row 5 (Problem + Auditor + Solution, No Test).** Judge sees the problem
statement and gold patch but not the test patch (new `PROBLEM_SOLUTION` info
level override, Claude Opus 4.6, 17 features = 15 statement + 2 solution
re-extracted at this context). Concatenated with the same 8 auditor features
used in rows 2-4; top-15 selected by Ridge. Tests whether the test patch
contributes information beyond what the solution already provides. AUC drops
by 1-3 points versus + Solution (Full), so test-patch features are not
redundant given the solution but are not load-bearing either.

**Row 6 (Full Minus Auditor).** Reuses the 20-feature SOLUTION-override pool
from row 4 (15 statement + 3 test + 2 solution, no extraction) and drops the
8 auditor features; top-15 selected by Ridge. AUC drops by less than 2 points
versus + Solution (Full), so auditor features contribute marginally on top of
the LLM-judge features extracted with full context.

### Backbone Ablation (Appendix C)

Ablates the model used to extract 12 non-repository-state LLM-as-a-judge features, keeping the same 15 features. The 3 repository state features are kept constant.

| Benchmark | Claude Opus 4.6 | GPT-5.4 | Claude Sonnet 4.6 |
|-----------|----------------|---------|-------------------|
| SWE-bench Verified | **0.8419** | 0.8350 | 0.8383 |
| SWE-bench Pro | 0.7591 | **0.7597** | 0.7579 |
| GSO | **0.8044** | 0.7008 | 0.7464 |
| Terminal-Bench 2.0 | 0.8101 | 0.8284 | **0.8303** |

## Evaluation Protocol

1. **Split tasks** (not agents) into train/test sets using 5-fold cross-validation
2. **Train IRT on train tasks only** to get uncontaminated ground truth difficulties
3. **Train difficulty predictor** on train tasks using train-only IRT difficulties as targets
4. **Predict difficulty** for test tasks
5. **Compute IRT probabilities**: P(success) = sigmoid(theta - beta_hat) for each (agent, task) pair
6. **Calculate AUC**: Compare predicted probabilities to actual outcomes

The IRT model is trained separately on train tasks to avoid data leakage. A full IRT model trained on all tasks is used only for the Oracle upper bound.

### Repeated Fold Seeds and Bootstrap

The main New Tasks run repeats the task-level 5-fold cross-validation over
`N=20` different shuffled fold seeds. The entry point is
`run_single_dataset()` in `experiment_new_tasks/run_all_datasets.py`, which
defaults to `k_folds=5`, `n_fold_seeds=20`, `n_bootstrap=10000`, and
`bootstrap_seed=0`. It constructs the consecutive fold seeds
`fold_seed_start, ..., fold_seed_start + n_fold_seeds - 1` and passes them into
`cross_validate_all_predictors_repeated_seeds()` in
`experiment_new_tasks/pipeline.py`.

The repeated-CV aggregation is implemented in
`cross_validate_all_predictors_repeated_seeds()` (`pipeline.py`, around lines
334-489). For each fold seed, it temporarily replaces `config.split_seed` with
that seed and calls the ordinary `cross_validate_all_predictors()` 5-fold
evaluation. The per-seed result stores the fold AUCs for every method. For each
method, the code then computes:

```
auc_mean_s = mean_folds(AUC_method,s,fold)
```

The reported AUC in the summary table is `mean_s(auc_mean_s)`, and the reported
standard deviation is the sample standard deviation across the `N` seed-level
means. This is intentionally different from the old single-seed fold standard
deviation: the uncertainty summary now reflects sensitivity to the randomized
task split, not just variation among the five folds of one split.

For significance testing, each method is compared to the constant baseline using
paired fold differences from the same fold seed and fold index. In
`pipeline.py` around lines 406-432, the method fold AUCs and constant-baseline
fold AUCs are aligned and averaged as:

```
delta_s = mean_folds(AUC_method,s,fold - AUC_baseline,s,fold)
```

This produces `N` seed-level mean differences, one per fold seed. These are the
inputs to `bootstrap_seed_mean_differences()` in
`experiment_new_tasks/bootstrap.py` (around lines 201-256). The bootstrap treats
fold seed as the resampling unit: each bootstrap replicate draws `N` values
with replacement from the observed `{delta_s}` values and computes the replicate
mean.

Concretely, the bootstrap helper:

- Validates that `B >= 1`, `0 < ci < 1`, and all seed-level differences are
  finite.
- Draws a `(B, N)` integer index matrix with `np.random.default_rng(seed)`.
- Computes `B` bootstrap means by indexing the `N` observed `delta_s` values.
- Reports the percentile CI using the 2.5% and 97.5% quantiles when `ci=0.95`.
- Computes a two-sided bootstrap p-value for `H0: mean(delta_s) = 0` by counting
  how many bootstrap means fall on each side of zero and doubling the smaller
  tail count.

The p-value uses an add-one correction:

```
p = min(1, 2 * (min(count(boot_mean <= 0), count(boot_mean >= 0)) + 1) / (B + 1))
```

This avoids reporting an exact `p=0` from finite Monte Carlo samples. If no
bootstrap sample crosses zero, the code marks `p_value_is_upper_bound=True`, and
the printed summary displays the p-value with a leading `<`.

The final per-method outputs are assembled in `pipeline.py` around lines
440-449:

- The observed mean difference, `mean(delta_s)`.
- The 95% percentile confidence interval from the bootstrap sample means.
- A two-sided bootstrap p-value for the null hypothesis that the mean difference
  is zero, using an add-one correction so finite bootstrap runs do not report
  an exact p-value of 0.

CSV output is expanded by `save_results_csv()` in
`experiment_new_tasks/run_all_datasets.py`. It includes the AUC mean, AUC
standard deviation, method-vs-baseline mean difference, 95% CI bounds, and
p-value for each method.

## Feature Sources

### Embeddings

Pre-computed embeddings from DeepSeek-R1-Distill-Qwen-32B, stored in `embeddings/`:
- SWE-bench Verified: `embeddings/embeddings__deepseek-ai__DeepSeek-R1-Distill-Qwen-32B__49b73e4eabfd__maxlen8192.npz`
- SWE-bench Pro: `embeddings/embeddings__deepseek-ai__DeepSeek-R1-Distill-Qwen-32B__7b0a101f1bc0__maxlen8192.npz`
- Terminal-Bench 2.0: `embeddings/embeddings__deepseek-ai__DeepSeek-R1-Distill-Qwen-32B__0accb67658c0__maxlen8192.npz`
- GSO: `embeddings/embeddings__deepseek-ai__DeepSeek-R1-Distill-Qwen-32B__086238f5ec4d__maxlen8192.npz`

### LLM Judge Features

15 unified features extracted via LLM structured output, identical across all datasets:
- **Problem (10)**: atypicality, codebase_scope, debugging_complexity, domain_knowledge_required, error_specificity, logical_reasoning_required, side_effect_risk, similar_issue_likelihood, solution_hint, verification_difficulty
- **Test (1)**: test_edge_case_coverage
- **Solution (1)**: solution_complexity
- **Auditor (3)**: codebase_scale, fix_localization, implementation_language_complexity

Feature paths: `llm_judge_features/defaults/{dataset}/llm_judge_features.csv`

## Data Paths

All datasets follow the same layout under `data/{dataset}/`:

| File | Purpose |
|------|---------|
| `data/{dataset}/responses.jsonl` | Binary response matrix |
| `data/{dataset}/irt/1d_1pl/abilities.csv` | Oracle IRT abilities |
| `data/{dataset}/irt/1d_1pl/items.csv` | Oracle IRT difficulties |

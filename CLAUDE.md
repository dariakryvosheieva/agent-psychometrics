# Agent Psychometrics

Predicts task-level performance in agentic coding benchmarks using Item Response Theory (IRT) augmented with task features and a decomposition of agent ability into LLM and scaffold components.

## Repository Structure

```
model_irt/
├── data/                              # Input data + IRT models (data/{dataset}/irt/)
├── embeddings/                        # Pre-computed task embeddings (.npz)
├── experiment_new_tasks/              # New Tasks experiment (Table 2)
├── experiment_new_responses/          # New Responses experiment (Table 20, Appendix G.1)
├── experiment_new_agents/             # New Agents experiment (Table 4)
├── experiment_new_benchmarks/         # New Benchmarks experiment (Table 5)
├── experiment_adaptive_testing/       # New-agent CAT via Fisher information (paper Fig. 2)
├── llm_judge_feature_extraction/      # LLM-as-a-judge feature extraction
│   └── auditor_agent/                 #   Repository state feature extraction via Docker
├── llm_judge_features/                # Extracted feature CSV files
│   ├── defaults/                      #   Features used in main experiments
│   ├── information_ablation/          #   Per-source features for ablation (Tables 3 & 10)
│   └── backbone_ablation/             #   GPT-5.4 and Claude 4.6 Sonnet features (Appendix C.2)
├── swebench_irt/                      # IRT model training
├── py_irt/                            # IRT library (local fork)
├── utils/                             # Shared experiment utilities (IRT training, embeddings, difficulty prediction)
├── slurm/                             # Batch scripts for the engaging cluster
└── output/                            # Experiment outputs (gitignored)
```

## Quick Start

```bash
source .venv/bin/activate

# Experiment New Tasks — 5-fold CV on held-out tasks (Table 2)
python -m experiment_new_tasks.run_all_datasets

# Feature source ablation (Tables 3 & 10)
python -m experiment_new_tasks.run_information_ablation

# Experiment New Responses — held-out observations (Table 20, Appendix G.1)
python -m experiment_new_responses.run_all_datasets

# Experiment New Agents — held-out LLM-scaffold pairs (Table 4)
python -m experiment_new_agents.run_all_datasets

# Experiment New Benchmarks — OOD benchmarks, holds out SWE-bench Pro and GSO (Table 5)
python -m experiment_new_benchmarks.run_all_datasets

# Adaptive Task Selection — new-agent CAT with IRT-Agent prior (paper figure)
python -m experiment_adaptive_testing.run_new_agent_experiment \
    --dataset swebench_verified \
    --split_seed 2 \
    --max_tasks 100 \
    --n_random_subsets 20 \
    --output_dir output/experiment_adaptive_testing/new_agent

# Train IRT model
python swebench_irt/train.py --dims 1 --model 1pl \
    --data_path data/swebench_verified/responses.jsonl
```

## Datasets

All input data lives under `data/{dataset}/`:

| Dataset | Tasks | Agents | Response Matrix | IRT Model |
|---------|-------|--------|----------------|-----------|
| SWE-bench Verified | 500 | 134 | `data/swebench_verified/responses.jsonl` | `data/swebench_verified/irt/1d_1pl/` |
| SWE-bench Pro | 730 | 14 | `data/swebench_pro/responses.jsonl` | `data/swebench_pro/irt/1d_1pl/` |
| Terminal-Bench 2.0 | 89 | 112 | `data/terminalbench/responses.jsonl` | `data/terminalbench/irt/1d_1pl/` |
| GSO | 102 | 15 | `data/gso/responses.jsonl` | `data/gso/irt/1d_1pl/` |

Terminal-Bench 2.0 also has a per-attempt variant at `data/terminalbench/responses_per_attempt.jsonl` (139 unique agents, 89 tasks; each cell is `{"successes": int, "trials": int}`). This is the binomial-likelihood input — point Experiment New Tasks at it via `--responses_path` to train the 1PL IRT with `dist.Binomial(total_count=trials, ...)` and evaluate AUC over per-attempt-expanded observations.

Produced by `python swebench_irt/prep_terminalbench.py --per_attempt_only --per_attempt_output data/terminalbench/responses_per_attempt.jsonl`, which scrapes the current leaderboard and writes only the per-attempt JSONL (the canonical binary `responses.jsonl` is untouched). The `--fetch_per_attempt_from <binary_jsonl>` mode is an alternative that revisits the binary file's per-agent detail URLs instead of re-scraping the full leaderboard; useful when you want the per-attempt file to be a strict subset of an existing binary snapshot.

Snapshot dates and drift: `responses.jsonl` was scraped on 2026-03-04 (112 agents); `responses_per_attempt.jsonl` was re-scraped on 2026-05-22 for rebuttals (143 raw records, 139 unique agents after collisions on the `create_subject_id` rule when `model_name` is "Multiple": `warp_multiple` ×3, `lemonharness_multiple` ×2, `little_coder_qwen3_6_35b_a3b` ×2). The agent set is **different** from the binary file's 112: 104 agents overlap, 8 binary agents are gone from the leaderboard (or renamed), 35 new agents have appeared (notably `vix_claude_opus_4_7`, `jjagent_*`, multiple `capy_*`, `clnkr_*`, `codebrain_1.5_*`, etc.). Most cells still have 5 trials, but 264 cells have 10 trials and small tails at 1–4 / 6 / 9, reflecting per-agent re-run histories. Task universe (89) is unchanged.

A separate canonical full IRT for the binomial protocol lives at `data/terminalbench/irt_binomial/1d_1pl/` (139 abilities + 89 difficulties). Used as the Oracle baseline when running Experiment New Tasks in binomial mode via `--abilities_path data/terminalbench/irt_binomial/1d_1pl/abilities.csv --items_path data/terminalbench/irt_binomial/1d_1pl/items.csv`. Trained by `python swebench_irt/train.py --dims 1 --model 1pl --data_path data/terminalbench/responses_per_attempt.jsonl --output_dir data/terminalbench/irt_binomial --seed 0` (which auto-detects the binomial format).

The binary file remains the canonical source for the Table 2 binary results in [experiment_new_tasks/README.md](experiment_new_tasks/README.md); the per-attempt file + `irt_binomial/` IRT are the canonical sources for the binomial-likelihood + per-attempt-AUC variant reported in Appendix C.6 (Table 13) of the paper.

## Documentation

| Document | Purpose |
|----------|---------|
| [experiment_new_tasks/README.md](experiment_new_tasks/README.md) | New Tasks experiment details |
| [experiment_new_responses/README.md](experiment_new_responses/README.md) | New Responses experiment details |
| [experiment_new_agents/README.md](experiment_new_agents/README.md) | New Agents experiment details |
| [experiment_new_benchmarks/README.md](experiment_new_benchmarks/README.md) | New Benchmarks experiment details |
| [experiment_adaptive_testing/README.md](experiment_adaptive_testing/README.md) | Adaptive task selection experiment |
| [llm_judge_feature_extraction/README.md](llm_judge_feature_extraction/README.md) | LLM judge feature extraction |

## Key Files

| File | Purpose |
|------|---------|
| `experiment_new_tasks/run_all_datasets.py` | Run Experiment New Tasks (Table 2) |
| `experiment_new_tasks/run_information_ablation.py` | Feature source ablation (Tables 3 & 10) |
| `experiment_new_tasks/plot_information_ablation.py` | Plot feature source ablation bar graph (Tables 3 & 10) |
| `experiment_new_responses/run_all_datasets.py` | Run Experiment New Responses (Table 20, Appendix G.1) |
| `experiment_new_agents/run_all_datasets.py` | Run Experiment New Agents (Table 4) |
| `experiment_new_benchmarks/run_all_datasets.py` | Run Experiment New Benchmarks (Table 5) |
| `experiment_adaptive_testing/run_new_agent_experiment.py` | New-agent CAT experiment (paper Fig. 2); `run_experiment.py` is the old cross-benchmark variant |
| `swebench_irt/train.py` | Train IRT models |
| `swebench_irt/prep_swebench.py` | Build response matrix |
| `experiment_new_tasks/feature_source.py` | Feature source abstractions (`GroupedFeatureSource`, `RegularizedFeatureSource`) |
| `experiment_new_tasks/feature_predictor.py` | Predictors (`FeatureBasedPredictor`, `GroupedRidgePredictor`) |

## Development Guidelines

**Keep git status clean:**
- Always commit changes after completing a group of related modifications
- Add any new experimental output files to `.gitignore` before committing
- Goal: leave the repository in the same clean state you found it

**Fail loudly — no silent fallbacks:**
- Never write code that silently skips or ignores missing data
- Raise explicit errors when expected data is missing (tasks, agents, features, dates, etc.)
- Never use fallbacks that hide problems — if a precondition is not met, throw an error
- Use warnings for non-critical issues, but errors for data that should exist
- This makes debugging much easier by catching issues at their source

**Code reuse — avoid duplication:**
- Always look for existing code that can be reused before writing new code
- Never duplicate code — extract shared logic into functions or classes
- If you find yourself copying code, refactor it into a shared module instead

**Pyro IRT training fragility:**
- Pyro's hierarchical 1PL priors can hit numerical issues (`Expected parameter concentration ... of distribution Dirichlet`) non-deterministically during SVI optimization, especially with smaller datasets (e.g., TerminalBench with 89 tasks)
- This is more likely when training multiple IRT models in parallel (e.g., `run_all_datasets` with `ProcessPoolExecutor`), though the processes don't share state — it's just a resource/timing issue
- Workaround: fold IRT models are cached, so a re-run will skip the failed fold and use the cached result from a successful training
- If a fresh run fails, re-running or using `--sequential` will usually work

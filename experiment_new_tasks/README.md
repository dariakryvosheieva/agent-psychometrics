# New Tasks Experiment (Section 4.1)

Evaluates how well a difficulty predictor can predict agent success on held-out tasks using the 1PL IRT model.

## Overview

Given a predicted difficulty beta_hat_i and known agent ability theta_j, compute:

```
P(success) = sigmoid(theta_j - beta_hat_i)
```

Then measure AUC by comparing these predicted probabilities to actual binary outcomes across 5-fold cross-validation on tasks.

## Quick Start

```bash
source .venv/bin/activate

# Run all datasets (Table 1)
python -m experiment_new_tasks.run_all_datasets

# Feature source ablation (Table 2)
python -m experiment_new_tasks.run_information_ablation

# Run specific datasets only
python -m experiment_new_tasks.run_all_datasets --datasets gso terminalbench
```

## Results

### Main Results (Table 1)

Run with: `python -m experiment_new_tasks.run_all_datasets`

| Dataset | Oracle | Combined | LLM Judge | Embedding | Baseline |
|---------|--------|----------|-----------|-----------|----------|
| SWE-bench Verified | 0.9447 | **0.8427** | 0.8415 | 0.8244 | 0.7175 |
| GSO | 0.9139 | **0.8048** | 0.7971 | 0.7624 | 0.7140 |
| TerminalBench | 0.9317 | **0.8207** | 0.8059 | 0.8171 | 0.7334 |
| SWE-bench Pro | 0.9183 | **0.7622** | 0.7417 | 0.7548 | 0.6565 |

### Feature Source Ablation (Table 2)

Run with: `python -m experiment_new_tasks.run_information_ablation`

LLM Judge AUC by information level, progressively adding features:

| Info Level | SWE-bench Verified | GSO | TerminalBench | SWE-bench Pro |
|---|---|---|---|---|
| Baseline | 0.7175 | 0.7140 | 0.7334 | 0.6565 |
| Problem | 0.7873 | 0.7257 | 0.7987 | 0.7181 |
| + Auditor | 0.7984 | 0.7270 | 0.8070 | 0.7369 |
| + Test | 0.8343 | 0.7251 | 0.8070 | 0.7489 |
| + Solution (Full) | 0.8483 | 0.7971 | 0.8103 | 0.7501 |
| Oracle | 0.9447 | 0.9139 | 0.9317 | 0.9183 |

### Backbone Ablation (Appendix C)

Ablates the model used to extract 12 non-repository-state LLM-as-a-judge features, keeping the same 15 features. The 3 repository state features are kept constant.

| Dataset | Claude Opus 4.6 | GPT-5.4 | Claude Sonnet 4.6 |
|---------|----------------|---------|-------------------|
| SWE-bench Verified | **0.8427** | 0.8367 | 0.8413 |
| SWE-bench Pro | 0.7622 | **0.7654** | 0.7619 |
| GSO | **0.8048** | 0.7334 | 0.7652 |
| TerminalBench | 0.8207 | **0.8367** | 0.8334 |

## Evaluation Protocol

1. **Split tasks** (not agents) into train/test sets using 5-fold cross-validation
2. **Train IRT on train tasks only** to get uncontaminated ground truth difficulties
3. **Train difficulty predictor** on train tasks using train-only IRT difficulties as targets
4. **Predict difficulty** for test tasks
5. **Compute IRT probabilities**: P(success) = sigmoid(theta - beta_hat) for each (agent, task) pair
6. **Calculate AUC**: Compare predicted probabilities to actual outcomes

The IRT model is trained separately on train tasks to avoid data leakage. A full IRT model trained on all tasks is used only for the Oracle upper bound.

## Feature Sources

### Embeddings

Pre-computed embeddings from DeepSeek-R1-Distill-Qwen-32B, stored in `embeddings/`:
- SWE-bench Verified: `embeddings/embeddings__...__princeton-nlp_SWE-bench_Verified__test__maxlen8192.npz`
- SWE-bench Pro: `embeddings/embeddings__...__ScaleAI_SWE-bench_Pro__test__maxlen8192.npz`
- TerminalBench: `embeddings/embeddings__...__json_terminal_bench_tasks.jsonl__test__maxlen8192.npz`
- GSO: `embeddings/embeddings__...__gso-bench_gso__test__maxlen8192.npz`

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

# Experiment A: Prior Validation (IRT AUC) - TerminalBench

Evaluates how well a difficulty predictor can predict agent success on held-out TerminalBench tasks using the **1PL Binomial IRT model**.

## Overview

**Goal**: Validate that predicted task difficulties are useful for forecasting agent performance on terminal/shell tasks without running agents on new tasks.

**Core Idea**: Given a predicted difficulty β̂_i and known agent ability θ_j, compute:

```
P(success) = sigmoid(θ_j - β̂_i)
```

Then measure AUC by comparing these predicted probabilities to actual outcomes (expanded from binomial trials).

## Key Differences from SWE-bench Experiment A

| Aspect | SWE-bench | TerminalBench |
|--------|-----------|---------------|
| Data format | Bernoulli (0/1) | Binomial (k/n trials) |
| IRT model | 1PL | **1PL Binomial** |
| Tasks | 500 | 89 |
| Agents | 130 | 83 |
| Task descriptions | `problem_statement` + `patch` | `instruction` + `solution.sh` |
| Data source | HuggingFace dataset | Local `terminal-bench/` repo |

## Quick Start

```bash
source .venv/bin/activate

# Run with baselines only
python -m experiment_a_terminalbench.train_evaluate

# Run with pre-computed embeddings
python -m experiment_a_terminalbench.train_evaluate \
    --embeddings_path chris_output/experiment_a_terminalbench/embeddings__Qwen__Qwen3-VL-8B-Instruct__pool-lasttoken__maxlen8192.npz

# Run with LLM judge features
python -m experiment_a_terminalbench.train_evaluate \
    --llm_judge_features_path chris_output/experiment_a_terminalbench/llm_judge_features/llm_judge_features.csv

# Run with both (full evaluation)
python -m experiment_a_terminalbench.train_evaluate \
    --embeddings_path chris_output/experiment_a_terminalbench/embeddings__Qwen__Qwen3-VL-8B-Instruct__pool-lasttoken__maxlen8192.npz \
    --llm_judge_features_path chris_output/experiment_a_terminalbench/llm_judge_features/llm_judge_features.csv

# Dry run to check config
python -m experiment_a_terminalbench.train_evaluate --dry_run
```

## Results (2026-01-17)

| Method | AUC | Δ vs Constant |
|--------|-----|---------------|
| **Oracle** (true β) | **0.9076** | +0.2049 |
| **LLM Judge** (GPT-5.2) | **0.7896** | +0.0869 |
| **Embedding** (Qwen3-VL-8B) | **0.7824** | +0.0797 |
| Constant baseline | 0.7027 | — |
| Agent-only baseline | 0.6989 | -0.0038 |
| Task-only baseline | 0.5000 | -0.2027 |

**Note**: Oracle uses full IRT (reference only). All other methods use IRT^train abilities.

### Key Findings

1. **LLM Judge slightly outperforms Embeddings** (0.7896 vs 0.7824 AUC)
2. Both methods provide ~8-9% improvement over the constant baseline
3. Results follow similar pattern to SWE-bench, validating the approach generalizes

### LLM Judge Feature Coefficients

Features selected by Lasso and their Ridge coefficients:

| Feature | Coefficient | Interpretation |
|---------|-------------|----------------|
| `task_complexity` | +0.773 | More complex → harder |
| `atypicality` | +0.487 | More unusual → harder |
| `task_clarity` | -0.372 | Clearer instructions → easier |
| `solution_size` | -0.222 | Larger solutions → easier (counterintuitive) |
| `solution_in_instruction` | -0.117 | More hints → easier |

### Comparison to SWE-bench

| Metric | SWE-bench | TerminalBench |
|--------|-----------|---------------|
| Oracle AUC | 0.9447 | 0.9076 |
| Best method AUC | 0.8337 | 0.7896 |
| Constant baseline | 0.7176 | 0.7027 |
| Improvement over constant | +11.6% | +8.7% |

## Data Sources

### IRT Model Parameters

Trained using 1PL Binomial IRT model:

```
chris_output/terminal_bench_2.0_binomial_1pl/1d/
├── abilities.csv    # Agent abilities (θ), 83 agents
├── items.csv        # Task difficulties (β), 89 tasks
└── model.pkl        # Trained Pyro model
```

### Response Data

```
data/terminal_bench/terminal_bench_2.0_raw.jsonl
```

Format: `{agent_id: {task_id: {successes: k, trials: n}, ...}, ...}`

### Task Descriptions

Loaded from cloned `terminal-bench/` repository:

```
terminal-bench/tasks/{task_id}/
├── task.yaml       # instruction, difficulty, category, tags
└── solution.sh     # Reference solution script
```

**Note:** Task `headless-terminal` is in the IRT data but missing from the repo (88/89 tasks available).

## AUC Calculation: Per-Trial Expansion

For binomial data (k successes out of n trials):

1. Compute predicted probability: `P = sigmoid(θ - β̂)` (1PL formula)
2. Expand to n binary observations: k successes (y=1) and (n-k) failures (y=0)
3. All n observations use the same predicted P
4. Compute standard ROC-AUC using sklearn's `roc_auc_score`

This properly weights agent-task pairs by trial count.

## Data Leakage Prevention

The IRT model provides ground truth difficulties (β) used as training targets. To avoid data leakage, we train **two separate IRT models**:

1. **IRT^train (Train-only IRT)**: Trained on train tasks (T1) only
   - Provides uncontaminated ground truth for training difficulty predictors
   - **Must be used for all actual methods** (embedding, constant, LLM judge, etc.)
   - Agent abilities θ and task difficulties β are both on the "train scale"

2. **IRT^full (Full IRT)**: Trained on all tasks (T1 ∪ T2)
   - **Used ONLY for oracle baseline** - shows theoretical best performance
   - The oracle is NOT a valid method - it's just a reference point for comparison

**Critical**: When computing AUC for any method, we use abilities from IRT^train. This ensures the abilities (θ) and predicted difficulties (β̂) are on the **same IRT scale**. The difficulty predictor is trained to predict β values on the train scale, so evaluation must also use abilities from that same scale. Using full IRT abilities would mix incompatible scales and leak test task information.

The split IRT models are cached in `chris_output/experiment_a_terminalbench/irt_splits/` and automatically reused when the split parameters match.

The IRT training uses **binomial IRT** to properly model the (k successes / n trials) data format:

```bash
# To manually train split IRT model for TerminalBench
python -m experiment_a.train_irt_split --binomial \
    --responses_path data/terminal_bench/terminal_bench_2.0_raw.jsonl \
    --output_dir chris_output/experiment_a_terminalbench/irt_splits \
    --dry_run  # See what would happen

# Or just run train_evaluate.py which handles this automatically
python -m experiment_a_terminalbench.train_evaluate
```

## Feature Sources

### 1. Embeddings

Generated on MIT Engaging cluster using Qwen3-VL-8B-Instruct:

```bash
# On cluster
sbatch slurm_scripts/terminalbench_embeddings.sh

# Output
chris_output/experiment_a_terminalbench/embeddings/
└── embeddings__Qwen__Qwen3-VL-8B-Instruct__pool-lasttoken__maxlen8192.npz
```

Embedding format: Task instruction + solution with difficulty prediction instruction appended.

### 2. LLM Judge Features

8 semantic features extracted using GPT-5.2:

```bash
# Dry run to see cost estimate
python -m experiment_a_terminalbench.compute_llm_judge_features --dry_run

# Full run (~$0.88 for 88 tasks)
python -m experiment_a_terminalbench.compute_llm_judge_features --provider openai --model gpt-5.2
```

**Features:**
- `solution_in_instruction` (0-3): How much the instruction hints at the solution
- `task_clarity` (1-5): How clear/well-specified the task is
- `solution_size` (1-3): Complexity of reference solution script
- `domain_knowledge_required` (1-5): Specialized knowledge needed
- `task_complexity` (1-5): Overall task complexity
- `logical_reasoning_required` (1-5): Reasoning depth needed
- `atypicality` (1-5): How unusual the task pattern is
- `tooling_complexity` (1-5): Tooling/environment setup complexity

**Output:**
```
chris_output/experiment_a_terminalbench/llm_judge_features/
├── {task_id}.json              # Individual feature files
├── llm_judge_features.csv      # Aggregated features
└── compute_stats_*.json        # Run statistics
```

## Module Structure

```
experiment_a_terminalbench/
├── __init__.py                    # Module exports
├── config.py                      # TerminalBenchConfig dataclass
├── data_loader.py                 # Load IRT params, binomial responses, task data
├── irt_evaluation.py              # Binomial AUC with per-trial expansion
├── baselines.py                   # Agent-only, task-only, constant baselines
├── train_evaluate.py              # Main evaluation pipeline
├── generate_embeddings.py         # Generate VLM embeddings for tasks
├── compute_llm_judge_features.py  # Extract LLM judge semantic features
└── llm_judge_prompt.py            # Prompt template for feature extraction

# Shared with experiment_a (reused via import):
# - experiment_a/train_irt_split.py  # Train IRT on train tasks only (--binomial flag)
# - experiment_a/data_loader.py      # stable_split_tasks()
```

## Output

Results saved to `chris_output/experiment_a_terminalbench/experiment_a_results.json`:

```json
{
  "config": {...},
  "data_summary": {
    "n_agents": 83,
    "n_tasks_total": 89,
    "n_train_tasks": 71,
    "n_test_tasks": 18
  },
  "oracle": {"auc": 0.9076},
  "embedding_predictor": {
    "auc_result": {"auc": 0.7824},
    "best_alpha": 1000.0
  },
  "llm_judge_predictor": {
    "auc_result": {"auc": 0.7896},
    "selected_features": ["task_complexity", "atypicality", ...],
    "feature_coefficients": {...}
  },
  "constant_baseline": {"auc": 0.7027},
  "agent_only_baseline": {"auc": 0.6989},
  "task_only_baseline": {"auc": 0.5000}
}
```

## Command Line Options

```
--test_fraction           Fraction of tasks for test set (default: 0.2)
--split_seed              Random seed for train/test split (default: 0)
--embeddings_path         Path to pre-computed embeddings .npz file
--llm_judge_features_path Path to LLM judge features CSV
--output_dir              Output directory (default: chris_output/experiment_a_terminalbench)
--dry_run                 Show configuration without running
```

## SLURM Scripts

```
slurm_scripts/
├── terminalbench_embeddings.sh   # Generate embeddings on cluster (1x H200, 2hr)
└── terminalbench_evaluate.sh     # Run evaluation on cluster (CPU only, 30min)
```

## Prerequisites

### 1. Train 1PL Binomial IRT Model

```bash
python swebench_irt/train_binomial.py \
    --data_path data/terminal_bench/terminal_bench_2.0_raw.jsonl \
    --output_dir chris_output/terminal_bench_2.0_binomial_1pl \
    --model_type 1pl
```

### 2. Clone terminal-bench Repository

```bash
git clone https://github.com/terminal-bench/terminal-bench.git
```

## References

- IRT formula: `P = sigmoid(θ - β)` matches py_irt's 1PL implementation
- Binomial likelihood: `P(k|n,p) = Binom(n, sigmoid(θ - β))`
- [SWE-bench Experiment A](../experiment_a/README.md) - Original SWE-bench version

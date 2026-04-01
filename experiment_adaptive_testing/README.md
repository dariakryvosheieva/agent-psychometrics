# Adaptive Task Selection via Fisher Information

Can we select a small, informative subset of benchmark tasks to evaluate new agents on — using only cross-benchmark difficulty predictions — and still recover the correct agent ranking?

## Overview

**Setting**: We pretend SWE-bench Pro is a new benchmark with no existing response data. We predict task difficulties from other benchmarks (Verified, TerminalBench, GSO) and use Fisher information to adaptively select which tasks to evaluate agents on.

**Metric**: Empirical reliability (`1 - mean(1/I) / var(θ̂)`) of agent ability estimates (14 agents), as a function of subset size. All methods are evaluated using the true IRT difficulty scores for MLE ability estimation and Fisher information — only task selection differs.

**Three methods compared**:

| Method | Task Selection |
|--------|---------------|
| **Fisher (Predicted)** | Maximize Fisher info using cross-benchmark predicted difficulties |
| **Fisher (Oracle)** | Maximize Fisher info using ground truth IRT difficulties |
| **Random** | Fixed random ordering (same for all agents) |

Fisher (Oracle) is an upper bound — what's achievable with perfect difficulty knowledge.

## Quick Start

```bash
source .venv/bin/activate

# Step 1: Generate predicted difficulties (train on Verified+TerminalBench+GSO, predict Pro)
python -m experiment_agent_features.predict_question_difficulty_multi_benchmark \
    --split_by benchmark \
    --train_benchmarks verified,terminalbench,gso \
    --ood_benchmark pro \
    --out_dir output/experiment_adaptive_testing/ood_predictions \
    --method judge

# Step 2: Run adaptive task selection experiment
python -m experiment_adaptive_testing.run_experiment
```

### Options

```bash
# Custom predictions file
python -m experiment_adaptive_testing.run_experiment --predictions_csv path/to/predictions.csv

# Custom seeds (results are averaged across all seeds)
python -m experiment_adaptive_testing.run_experiment --seeds 42 7 123

# Adjust parameters
python -m experiment_adaptive_testing.run_experiment --max_steps 300 --prior_sigma 5.0
```

## How It Works

### Fisher Information for 1PL

For the Rasch model, `P(success) = sigmoid(θ - b)`, the item information is:

```
I(θ, b) = P(1 - P)
```

This is maximized when `P = 0.5` (i.e., `θ ≈ b`), so Fisher selection picks tasks whose difficulty matches the agent's current estimated ability.

### Simulation Loop

All three methods share the same loop:

1. For each step t = 1, ..., max_steps:
   - Selector picks the next task
   - Observe the agent's real binary response
   - Update the agent's score
2. At each step, compute empirical reliability of agent ability estimates (using oracle IRT difficulties for all methods)

For Fisher methods, the selector is adaptive (depends on the agent's evolving θ̂ after each response). For Random, all agents share the same fixed task ordering. Since Random depends on task order, results are averaged across multiple random seeds (5 by default).

### MLE Ability Estimation

Fisher methods estimate agent ability via MAP with a weak Gaussian prior (σ = 3.0 by default):

```
maximize: Σ [y_j log P_j + (1-y_j) log(1-P_j)] - θ²/(2σ²)
```

Optimized with L-BFGS-B, bounded to [-6, 6].

## Output

```
output/experiment_adaptive_testing/averaged/
├── config.json              # Experiment parameters (including seeds)
├── results.csv              # Averaged empirical reliability at each step
└── reliability_curves.png   # Main figure
```

## Data

| File | Purpose |
|------|---------|
| `data/swebench_pro/responses.jsonl` | Response matrix (14 agents × 730 tasks) |
| `data/swebench_pro/irt/1d_1pl/items.csv` | Oracle IRT difficulties |
| `output/experiment_adaptive_testing/ood_predictions/predictions.csv` | Cross-benchmark predicted difficulties |

## Directory Structure

```
experiment_adaptive_testing/
├── __init__.py
├── __main__.py
├── cat_simulation.py     # TaskSelector ABC, FisherSelector, RandomSelector, MLE, simulation loop
├── run_experiment.py     # CLI entry point, plotting
└── README.md
```

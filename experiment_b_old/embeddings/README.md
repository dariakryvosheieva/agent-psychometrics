# Embedding-Based Posterior Difficulty Prediction

Uses trajectory embeddings from VLMs to predict IRT difficulty residuals.

## Overview

**Goal**: Replace hand-crafted trajectory features with learned embeddings that capture task difficulty signals from agent failure trajectories.

**Key Insight**: Large language models can encode trajectory information into dense vectors that predict how much harder (or easier) a task is than the prior estimate suggests.

## Architecture

```
posterior_difficulty_i = prior(x_i) + psi^T * aggregate_j[embed(tau_ij)]

Where:
- prior(x_i) = embedding-based prior from task description (Daria's model)
- embed(tau_ij) = trajectory embedding for agent j on task i
- aggregate_j = aggregation across agents (mean, mean+std, etc.)
- psi = learned Ridge regression weights
```

## Pipeline

### Phase 1: Compute Trajectory Embeddings (GPU)

Extract embeddings for each (task, agent) trajectory pair.

```bash
# Test with 2 agents first
sbatch scripts/embedding/test_embedding_job.sh

# Full run with 8 GPUs in parallel (~45 min)
sbatch scripts/embedding/compute_embeddings_multi_gpu.sh
```

**Input**: Trajectory JSON files from `trajectory_data/unified_trajs/{agent}/{task}.json`

**Output**: NPZ files in `chris_output/experiment_b/trajectory_embeddings/{content}_{instruction}/{agent}/{task}.npz`

Each NPZ contains:
- `embedding`: (4096,) float32 vector
- `task_id`, `agent_id`: identifiers
- `content_type`, `instruction_type`: ablation config
- `backbone`, `max_length`, `embedding_dim`: model config
- `resolved`: whether trajectory solved the task

### Phase 2: Train & Evaluate Posterior (CPU)

Aggregate embeddings across agents and fit Ridge regression on residuals.

```bash
python -m experiment_b.train_evaluate_embeddings \
    --embeddings_dir chris_output/experiment_b/trajectory_embeddings/full_difficulty \
    --aggregation mean_std \
    --ridge_alpha cv
```

**Process**:
1. Load pre-computed trajectory embeddings
2. Aggregate across M1 agents per task (mean, std, etc.)
3. Compute residuals: `y = ground_truth_β - prior_prediction`
4. Fit Ridge regression: `residual ~ aggregated_embeddings`
5. Evaluate on D_valid using AUC-ROC

## Content Types

How much trajectory information to include in the embedding input:

| Type | Contents |
|------|----------|
| `full` | task_statement + solution + full trajectory + instruction |
| `condensed` | task_statement + solution + trajectory summary + instruction |
| `failure_focused` | task_statement + solution + errors only + instruction |
| `no_solution` | task_statement + trajectory (no gold solution) |

## Instruction Types

What question to append to guide the embedding:

| Type | Instruction |
|------|-------------|
| `difficulty` | "How difficult is this software engineering task?" |
| `residual` | "How much harder or easier is this task than initially expected?" |
| `progress` | "Based on this trajectory, how close was the agent to solving the task?" |
| `closeness` | "Rate the agent's progress: 0=completely stuck, 1=nearly solved" |

## Aggregation Strategies

How to combine embeddings from multiple agent trajectories per task:

| Strategy | Output Dim | Description |
|----------|------------|-------------|
| `mean_only` | 4096 | Mean embedding across agents |
| `mean_std` | 8192 | Mean + standard deviation |
| `weighted` | 4096 | Weighted by agent ability (θ) |
| `all_stats` | 16384 | Mean + std + min + max |

## Model Backbones

Tested models (in priority order):

1. **Qwen/Qwen3-VL-8B-Instruct** - Baseline (matches Daria's prior)
2. **Qwen/Qwen3-VL-32B-Instruct** - Larger VL model
3. **Qwen/Qwen3-30B-A3B** - MoE model (efficient scaling)
4. **Qwen/Qwen3-32B** - Dense 32B
5. **Qwen/Qwen3-Embedding-8B** - Purpose-built embedding model

## Ablation Matrix

| Ablation | Options | Compute |
|----------|---------|---------|
| Backbone | 5 models | GPU (re-embed) |
| Content | full, condensed, failure_focused, no_solution | GPU (re-embed) |
| Instruction | difficulty, residual, progress, closeness | GPU (re-embed) |
| Aggregation | mean_only, mean_std, weighted, all_stats | CPU only |
| Ridge Alpha | RidgeCV with [1e-4, ..., 1e5] | CPU only |

## File Structure

```
experiment_b/
├── compute_trajectory_embeddings.py   # Phase 1: GPU embedding extraction
├── embedding_aggregator.py            # Multi-trajectory aggregation
├── embedding_posterior_model.py       # Ridge regression on embeddings
├── train_evaluate_embeddings.py       # Phase 2: training pipeline
└── EMBEDDINGS.md                      # This file

scripts/embedding/
├── test_embedding_job.sh              # Quick 2-agent test
├── compute_embeddings_multi_gpu.sh    # Full run with 8 GPUs
├── compute_trajectory_embeddings_engaging.sh  # Single GPU baseline
└── launch_embedding_ablations.sh      # Ablation sweep launcher
```

## Output Format

Results saved to `chris_output/experiment_b/embedding_results/`:

- `results__{run_id}.json` - Individual run results
- `all_results.jsonl` - Master log for aggregation

Each result includes:
- Ablation parameters (backbone, content, instruction, aggregation)
- Prior and posterior AUC on D_train and D_valid
- Ridge alpha used (from CV or specified)
- Embedding dimension and model metadata

## Quick Start

```bash
# 1. Run test job on cluster
sbatch scripts/embedding/test_embedding_job.sh

# 2. Verify output
ls chris_output/experiment_b/trajectory_embeddings_test/full_difficulty/

# 3. Run full embedding computation
sbatch scripts/embedding/compute_embeddings_multi_gpu.sh

# 4. Train and evaluate
python -m experiment_b.train_evaluate_embeddings \
    --embeddings_dir chris_output/experiment_b/trajectory_embeddings/full_difficulty
```

## Comparison to Hand-Crafted Features

| Approach | Features | D_valid AUC | Notes |
|----------|----------|-------------|-------|
| Prior only | - | 0.7383 | Baseline |
| Simple features | 5 | 0.7444 | +0.6% |
| LLM judge | 14 | 0.7215 | Overfits |
| **Embeddings** | 4096+ | TBD | This approach |

## References

- [predict_question_difficulty.py](../predict_question_difficulty.py) - Daria's prior embedding code
- [Experiment B README](README.md) - Overall experiment context
- [Research Proposal](../chris%20proposal.md) - Section 3.2

# MLP Ablation Study

This directory contains ablation studies for MLP-based difficulty prediction.

## ⚠️ Important: Two Architectures

**IRT-Style MLP (`MLPPredictor`)** - SANITY CHECK ONLY
- Architecture: `P = sigmoid(θ_agent - β_task)` — identical to IRT formula
- By design, this CANNOT exceed IRT/Ridge performance
- Was used only to verify that end-to-end BCE training works
- **Do not use for experiments trying to beat baselines**

**Full MLP (`FullMLPPredictor`)** - PRIMARY ARCHITECTURE
- Architecture: `[agent_one_hot | features] → Hidden → P(success)`
- Can learn arbitrary agent-task interactions
- **Use this for experiments trying to beat Ridge regression**
- Key finding: needs very strong regularization (`weight_decay=100-1000`)

## Primary Experiment: Full MLP on Embeddings

Run with:
```bash
sbatch experiment_a/mlp_ablation/slurm_full_mlp.sh
```

See `test_full_mlp.py` for the main experiment configuration.

---

# IRT-Style MLP (Sanity Check Documentation)

The following documents the IRT-style MLP experiments, which were used to verify
that the IRT approach works with gradient descent. These results are NOT the
main experiments — they serve only as a sanity check.

## The Problem: Gradient Competition

When training an IRT-style MLP from scratch, **agent embeddings dominate over task features**:

| Method | Test AUC |
|--------|----------|
| Baseline (learned from scratch) | 0.756 |
| Frozen IRT abilities | 0.817 |
| Ridge regression (closed-form) | 0.817 |

The MLP has two learnable components:
- **Agent abilities θ**: `nn.Embedding(n_agents, 1)` — 131 parameters
- **Task difficulty β**: `nn.Linear(feature_dim, 1)` — 9 parameters (for LLM Judge)

With standard training, the agent embeddings can fit the training data without learning useful feature weights, leading to poor generalization.

## The Solution: Two-Stage Training

**Two-stage training** solves gradient competition by ensuring features are learned first:

1. **Stage 1**: Initialize agent abilities from IRT, freeze them, train features only
2. **Stage 2**: Unfreeze agents, fine-tune both with low agent learning rate

```python
from experiment_a.shared.mlp_predictor import MLPPredictor

predictor = MLPPredictor(
    source=feature_source,
    two_stage=True,           # Enable two-stage training
    stage1_epochs=250,        # 50% of total epochs for stage 1
    stage2_agent_lr_scale=0.1,  # Low LR for agents in stage 2
    learning_rate=0.01,
    n_epochs=500,
)
```

## Experimental Results

### Two-Stage Training Ablation (LLM Judge features)

| Method | Test AUC |
|--------|----------|
| Baseline (learned from scratch) | 0.756 |
| Frozen IRT | 0.817 |
| **Two-stage (s1=50%, s2_lr=0.1)** | **0.818** |
| Two-stage (s1=50%, s2_lr=0.01) | 0.817 |
| Two-stage (s1=75%, s2_lr=0.1) | 0.818 |
| Two-stage (s1=75%, s2_lr=0.01) | 0.817 |
| Oracle (true β) | 0.944 |

**Key findings:**
- Two-stage training matches or slightly beats frozen IRT
- Hyperparameters (stage split, stage 2 LR) don't matter much — all configs work
- IRT initialization is the critical factor

### Learning Rate Ablation (without IRT initialization)

We also tested whether slowing agent learning alone (without IRT initialization) could work:

| Method | Test AUC |
|--------|----------|
| agent_lr_scale=1.0 (baseline) | 0.756 |
| agent_lr_scale=0.1 | 0.697 |
| agent_lr_scale=0.01 | 0.664 |
| agent_lr_scale=0.001 | 0.673 |
| Frozen IRT | 0.817 |

**Conclusion**: Slowing agent learning with random initialization makes things **worse**. The problem is initialization, not learning speed.

## Recommended Settings

### For LLM Judge (low-dim features, ~9 features)

```python
predictor = MLPPredictor(
    source=llm_judge_source,
    two_stage=True,
    stage1_epochs=250,
    stage2_agent_lr_scale=0.1,
    learning_rate=0.01,
    n_epochs=500,
)
```

Or simply use frozen IRT (equivalent performance, simpler):

```python
predictor = MLPPredictor(
    source=llm_judge_source,
    freeze_abilities=True,
    learning_rate=0.01,
    n_epochs=500,
)
```

### For Embeddings (high-dim features, ~768 features)

Use strong regularization on feature weights:

```python
predictor = MLPPredictor(
    source=embedding_source,
    freeze_abilities=True,  # or two_stage=True
    feature_weight_decay=1.0,  # Strong regularization for high-dim
    learning_rate=0.01,
    n_epochs=500,
)
```

## Why This Works

The IRT formula `P = sigmoid(θ - β)` has an identifiability issue: you can shift all θ values up and all β values up by the same amount without changing predictions. This makes joint learning difficult.

**IRT initialization** provides a well-calibrated starting point:
- Agent abilities θ are pre-computed from the full response matrix using maximum likelihood
- This anchors the scale and allows the feature layer to learn β relative to known θ values

**Two-stage training** preserves this calibration:
- Stage 1 learns features with frozen, well-calibrated abilities
- Stage 2 allows minor adjustments while maintaining the learned feature weights

## Files in This Directory

| File | Purpose |
|------|---------|
| `run_mlp_ablation.py` | Main ablation: baseline, frozen_irt, strong_reg, both_fixes |
| `test_two_stage.py` | Two-stage training ablation |
| `test_lr_ablation.py` | Learning rate scale ablation |
| `sanity_check_mlp_vs_ridge.py` | Verify MLP matches Ridge when properly configured |
| `plot_mlp_training_loss.py` | Visualize training convergence |
| `slurm_*.sh` | SLURM scripts for cluster execution |

## Running Experiments

```bash
# Two-stage ablation
python -m experiment_a.mlp_ablation.test_two_stage

# Full ablation (all feature sources)
python -m experiment_a.mlp_ablation.run_mlp_ablation --source all

# On cluster
sbatch experiment_a/mlp_ablation/slurm_two_stage.sh
```

# SAD-IRT: State-Aware Deep Item Response Theory

This experiment implements SAD-IRT for SWE-bench task difficulty estimation using agent trajectories. The goal is to improve response prediction (and ultimately difficulty estimation) by conditioning on agent trajectory information.

## Research Question

**Can we improve IRT response prediction by encoding agent trajectories?**

Standard 1PL IRT models success probability as:
```
P(y_ij = 1) = σ(θ_j - β_i)
```

SAD-IRT adds a trajectory-dependent interaction term:
```
P(y_ij = 1 | θ_j, β_i, ψ_ij) = σ(θ_j - (β_i + ψ_ij))
```

Where `ψ_ij` is predicted from the agent's trajectory on that task via a neural encoder.

## Current Results

| Model | Validation AUC | Notes |
|-------|---------------|-------|
| Standard 1PL IRT (baseline) | **0.94** | No trajectory features |
| SAD-IRT v1 (interrupted) | 0.885 | Step ~3100, checkpoint lost due to overwrite |

**Historical note**: The 0.885 AUC at step ~3100 was achieved with Qwen3-0.6B, r=16, 2-layer MLP head, BatchNorm constraint. Training was interrupted by job timeout, and the checkpoint was inadvertently overwritten before versioned checkpoints were implemented.

The baseline is strong because θ and β can memorize agent-task pairs seen in training. SAD-IRT must demonstrate that trajectory features provide additional predictive signal.

## Model Architecture

### Core Equation
```
logit = θ_j - (β_i + ψ_ij)
```

### Components

1. **Agent Ability (θ)**: `nn.Embedding(num_agents, 1)`
   - Initialized with `N(0, 0.1)`
   - Learning rate: 1e-3

2. **Task Difficulty (β)**: `nn.Embedding(num_tasks, 1)`
   - Initialized with `N(0, 0.1)`
   - Learning rate: 1e-3

3. **Trajectory Encoder**: Qwen3-0.6B + LoRA
   - Base model frozen, only LoRA adapters trained
   - LoRA config: r=16, alpha=32, dropout=0.1
   - Target modules: q_proj, k_proj, v_proj, o_proj
   - Gradient checkpointing enabled
   - Learning rate: 1e-4

4. **ψ Head**: Single linear projection
   ```python
   nn.Linear(hidden_size, 1)
   ```
   - Initialized to zeros (so ψ ≈ 0 at start, model begins as standard IRT)

5. **Zero-Mean Constraint**: `nn.BatchNorm1d(1, affine=False)`
   - Ensures identifiability by centering ψ values
   - Uses momentum=0.1 for running statistics

### Input Format

Each sample combines:
```
[PROBLEM]
{problem_statement from SWE-bench}

[SOLUTION]
{gold_patch from SWE-bench}

[TRAJECTORY]
{agent conversation: [SYSTEM]...[USER]...[ASSISTANT]...}
```

- Max length: 8192 tokens
- If too long, trajectory is truncated from the **beginning** (keeps suffix)
- Representation: last non-padding token's hidden state

## Training Configuration

### Current Settings

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Batch size | 16 | Stable BatchNorm stats (H200 has 141GB VRAM) |
| Gradient accumulation | 2 | Effective batch 32 |
| Epochs | 10 | May need more for convergence |
| Encoder LR | 1e-4 | Standard for LoRA fine-tuning |
| Embedding LR | 1e-3 | 10x higher for small params |
| Weight decay | 0.01 | Standard regularization |
| Warmup ratio | 0.1 | 10% of total steps |
| LR schedule | Linear warmup → Cosine decay | Standard practice |
| Max grad norm | 1.0 | Gradient clipping |
| Eval frequency | Every 100 steps | Monitor AUC progress |

### Data

- **Response matrix**: 131 agents × 500 tasks
- **Trajectories available**: 74 agents with both trajectories AND responses (~57% of agents)
- **Total trajectory files**: 35,989 (not all agents have all 500 tasks)
- **Train/test split**: 80/20 by (agent, task) pairs
- **Training samples**: ~28,791 pairs (80% of trajectory files)
- **Test samples**: ~7,198 pairs (20% of trajectory files)

## Evaluation

### Part 2: Full AUC (PRIMARY)
- Train SAD-IRT on all agents, random 80% of (agent, task) pairs
- Test on held-out 20% of pairs (both agent and task seen in training)
- **Metric**: AUC-ROC on binary response prediction
- **Baseline**: Standard 1PL IRT on same split
- **Goal**: SAD-IRT AUC > Baseline AUC

### Part 1: Calibration (SECONDARY)
- Train on M1+M2 agents only
- Compare difficulty estimates (β + mean(ψ)) to oracle (all agents)
- Focus on hard tasks (pass rate ≤ 20%)

---

## Architectural/Optimization Decisions & Ablation Opportunities

Below is a comprehensive list of all design choices, their rationale, and whether they should be ablated.

### A. Model Architecture

| Decision | Current Choice | Rationale | Ablate? | Priority |
|----------|---------------|-----------|---------|----------|
| **Base encoder** | Qwen3-0.6B | Small, fast, good performance | **YES** | HIGH |
| **LoRA rank (r)** | 16 | Balance between capacity and efficiency | **YES** | MEDIUM |
| **LoRA alpha** | 32 | 2x rank is standard | Maybe | LOW |
| **LoRA target modules** | q,k,v,o_proj | Standard attention modules | Maybe | LOW |
| **LoRA dropout** | 0.1 | Regularization | Maybe | LOW |
| **ψ head architecture** | [Linear→ReLU→Dropout→Linear] | Simple 2-layer MLP | **YES** | MEDIUM |
| **ψ head hidden dim** | 256 | Arbitrary choice | Maybe | LOW |
| **ψ head dropout** | 0.1 | Regularization | Maybe | LOW |
| **Zero-init final layer** | Yes | Start as standard IRT | Maybe | LOW |
| **Zero-mean constraint** | BatchNorm (affine=False) | Following SAD-IRT paper | **YES** | HIGH |
| **BatchNorm momentum** | 0.1 | Default choice | Maybe | LOW |
| **Representation extraction** | Last token hidden state | Standard for causal LMs | **YES** | MEDIUM |
| **Gradient checkpointing** | Enabled | Memory efficiency | No | - |

### B. Input Format

| Decision | Current Choice | Rationale | Ablate? | Priority |
|----------|---------------|-----------|---------|----------|
| **Input format** | [PROBLEM]+[SOLUTION]+[TRAJECTORY] | User specification | **YES** | HIGH |
| **Truncation strategy** | Keep suffix | Last actions most informative? | **YES** | MEDIUM |
| **Max length** | 8192 | Memory vs context tradeoff | Maybe | MEDIUM |
| **Include problem text** | Yes | Task context | **YES** | HIGH |
| **Include solution text** | Yes | Ground truth reference | **YES** | HIGH |

### C. Training

| Decision | Current Choice | Rationale | Ablate? | Priority |
|----------|---------------|-----------|---------|----------|
| **Optimizer** | AdamW | Standard choice | No | - |
| **Separate LR for embeddings** | Yes (10x encoder) | Embeddings are small, need faster updates | **YES** | MEDIUM |
| **Encoder LR** | 1e-4 | Standard for LoRA | **YES** | MEDIUM |
| **Embedding LR** | 1e-3 | Higher for small params | **YES** | MEDIUM |
| **Weight decay** | 0.01 | Standard | Maybe | LOW |
| **Warmup ratio** | 0.1 | Standard | Maybe | LOW |
| **LR schedule** | Linear warmup + Cosine | Standard | No | - |
| **Gradient clipping** | max_norm=1.0 | Stability | Maybe | LOW |
| **Effective batch size** | 16-32 | Limited by memory | Maybe | LOW |
| **Loss function** | BCEWithLogitsLoss | Standard for binary classification | No | - |

### D. Data/Evaluation

| Decision | Current Choice | Rationale | Ablate? | Priority |
|----------|---------------|-----------|---------|----------|
| **Train/test split** | Random 80/20 by pairs | Standard evaluation | **YES** | HIGH |
| **Missing trajectory handling** | Drop pair | Clean approach | Maybe | LOW |

---

## Recommended Ablations (Prioritized)

### High Priority (likely to impact results)

1. **Base encoder size**
   - Try: Qwen3-1.8B, Qwen3-4B
   - Hypothesis: Larger model → better trajectory understanding → better ψ
   - Cost: More memory/time

2. **Zero-mean constraint method**
   - Try: No constraint, LayerNorm, explicit mean subtraction
   - Hypothesis: BatchNorm may be unstable with small batches
   - Risk: Without constraint, ψ may absorb β signal

3. **Input format variants**
   - Try: Trajectory only, Problem+Trajectory, different delimiters
   - Hypothesis: Problem/solution may confuse or help the encoder
   - Cost: Low

4. **Train/test split strategy**
   - Try: Hold out entire agents, hold out entire tasks
   - Hypothesis: Current split may be too easy (memorization)
   - Risk: May need different experimental design

### Medium Priority

5. **LoRA rank**
   - Try: r=4, r=8, r=32, r=64
   - Hypothesis: Higher rank → more expressive → better fit

6. **Representation extraction method**
   - Try: Mean pooling, [CLS] token (if available), attention-weighted
   - Hypothesis: Last token may not be optimal

7. **ψ head architecture**
   - Try: 1-layer (linear), 3-layer, different hidden dims
   - Hypothesis: May need more/less capacity

8. **Truncation strategy**
   - Try: Keep prefix, middle, or smart truncation
   - Hypothesis: Beginning may contain important context

9. **Learning rate ratio**
   - Try: Same LR for all, 5x, 20x
   - Hypothesis: Optimal ratio may differ

### Low Priority (unlikely to change results significantly)

10. LoRA alpha, dropout
11. BatchNorm momentum
12. Weight decay
13. Warmup ratio
14. ψ head dropout
15. Max length (unless hitting limits)

---

## Usage

### Local dry run
```bash
source .venv/bin/activate
python -m experiment_sad_irt.train_evaluate --dry_run --max_samples 50
```

### Smoke test (verify code paths)
```bash
python -m experiment_sad_irt.train_evaluate --smoke_test
```

### Overfit test (verify gradients flow)
```bash
python -m experiment_sad_irt.train_evaluate --overfit_test
```

### Full training
```bash
python -m experiment_sad_irt.train_evaluate \
    --mode full_auc \
    --model_name Qwen/Qwen3-0.6B \
    --max_length 8192 \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --epochs 10 \
    --output_dir chris_output/sad_irt
```

### MIT Engaging cluster
```bash
# 2x H200 GPUs (recommended)
sbatch slurm_scripts/run_sad_irt_h200.sh

# Monitor
squeue -u $USER
tail -f logs/sad_irt_*.out
```

### Resume from checkpoint
```bash
# Auto-resumes from most recent checkpoint
sbatch slurm_scripts/run_sad_irt_h200.sh

# Or specify explicitly
python -m experiment_sad_irt.train_evaluate \
    --resume_from chris_output/sad_irt/checkpoint_best_step3100_20260115_*.pt
```

## Files

| File | Purpose |
|------|---------|
| `config.py` | Configuration dataclass with all hyperparameters |
| `model.py` | SADIRT and StandardIRT model definitions |
| `dataset.py` | TrajectoryIRTDataset with input formatting |
| `train.py` | Trainer class with gradient accumulation, scheduling |
| `evaluate.py` | Metrics computation (AUC, Brier, calibration) |
| `train_evaluate.py` | Main entry point with smoke/overfit tests |

## Checkpoints

Checkpoints are saved with versioned filenames:
```
checkpoint_{type}_step{step}_{timestamp}.pt
```

Example:
```
chris_output/sad_irt/
├── checkpoint_best_step1500_20260115_143022.pt
├── checkpoint_best_step2800_20260115_152341.pt
├── checkpoint_epoch_1_step1840_20260115_150000.pt
└── ...
```

Each checkpoint contains:
- `model_state_dict`: Trainable weights (θ, β, LoRA, ψ_head)
- `optimizer_state_dict`: Optimizer state
- `scheduler_state_dict`: LR scheduler state
- `global_step`: Training step
- `epoch`: Current epoch
- `best_auc`: Best validation AUC so far
- `metrics`: Eval metrics at checkpoint time (if "best" checkpoint)
- `timestamp`: When saved

## Next Steps

1. **Complete current training run** to see final AUC
2. **If SAD-IRT < baseline**: Try ablations (larger encoder, different input format)
3. **If SAD-IRT ≈ baseline**: ψ may not be adding signal; try harder splits
4. **If SAD-IRT > baseline**: Great! Analyze what ψ learned

---

## Future Changes (TODO)

Things to try in subsequent runs, based on initial experiments:

### Decided to Keep (for now)
- **Train/test split**: Random 80/20 by (agent, task) pairs
- **Input format**: [PROBLEM] + [SOLUTION] + [TRAJECTORY]
- **Representation**: Last non-padding token's hidden state
- **Truncation**: Keep suffix (truncate from beginning)
- **LR ratio**: 10x higher for embeddings vs encoder

### Recently Implemented Changes

| Change | Status | Notes |
|--------|--------|-------|
| **θ/β init from pre-trained IRT** | ✅ Done | Uses `clean_data/swebench_verified_20251120_full/1d/` |
| **Skip baseline training** | ✅ Done | Loads pre-trained IRT instead of retraining |
| **Versioned checkpoints** | ✅ Done | Format: `checkpoint_{type}_step{step}_{timestamp}.pt` |
| **1-layer ψ head** | ✅ Done | Single linear projection instead of 2-layer MLP |
| **Batch size 16** | ✅ Done | H200 has 141GB VRAM, can fit batch_size=16 |

### To Try in Future Runs

| Change | Current | Proposed | Rationale |
|--------|---------|----------|-----------|
| **Encoder size** | Qwen3-0.6B | Qwen3-4B | Better trajectory understanding |
| **LoRA rank** | r=16 | r=8 or r=32 | Find optimal capacity |

### Ablation Queue (ordered by priority)

1. **Encoder size**: 0.6B → 4B (if 0.6B doesn't beat baseline)
2. **LoRA rank**: Try r=8, r=32
3. **Truncation strategy**: Suffix → prefix (if context seems important)

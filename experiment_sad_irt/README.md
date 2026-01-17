# SAD-IRT: State-Aware Deep Item Response Theory

This experiment implements SAD-IRT for SWE-bench task difficulty estimation using agent trajectories. The goal is to predict task difficulty for **frontier tasks** (tasks that pre-frontier models cannot solve but post-frontier models can) using trajectory information from pre-frontier models.

## Research Question

**Can we use trajectory information from weaker models to predict task difficulty for frontier tasks?**

Standard 1PL IRT models success probability as:
```
P(y_ij = 1) = σ(θ_j - β_i)
```

SAD-IRT adds a trajectory-dependent interaction term:
```
P(y_ij = 1 | θ_j, β_i, ψ_ij) = σ(θ_j - (β_i + ψ_ij))
```

Where `ψ_ij` is predicted from the agent's trajectory on that task via a neural encoder.

The key insight: with a zero-mean constraint on ψ, the learned β values absorb the true task difficulty. We can then evaluate how well these β values predict oracle difficulties for frontier tasks.

## Experimental Setup

### Data Split

- **Cutoff date**: August 7, 2025 (gpt-5-mini release)
- **Pre-frontier agents**: Models before cutoff (107 total, 64 with trajectories)
- **Post-frontier agents**: Models after cutoff (24 total, 14 with trajectories)

### Frontier Tasks

Tasks where:
- Pre-frontier pass rate ≤ 10%
- Post-frontier pass rate > 10%

These are tasks that only became solvable with the newer generation of models.

### Evaluation

1. Train oracle IRT on **all agents** (pre + post frontier) → get ground truth β
2. Train SAD-IRT on **pre-frontier agents only** → learned β values
3. For frontier tasks, compute **Spearman ρ** between SAD-IRT β and oracle β

### Baseline

Standard IRT trained only on pre-frontier responses (no trajectories). This isolates the contribution of trajectory encoding.

## Model Architecture

### Core Equation
```
logit = θ_j - (β_i + ψ_ij)
```

### Components

1. **Agent Ability (θ)**: `nn.Embedding(num_agents, 1)`
   - Initialized from oracle IRT
   - Learning rate: 1e-3

2. **Task Difficulty (β)**: `nn.Embedding(num_tasks, 1)`
   - Initialized from oracle IRT
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
   - Uses default Kaiming initialization

5. **Zero-Mean Constraint**: BatchNorm or centering
   - Ensures identifiability by centering ψ values
   - β absorbs the true difficulty

### Input Format

Trajectory summaries from `chris_output/trajectory_summaries_api/`:
- LLM-generated summaries (~400-600 tokens)
- Max length: 1024 tokens

## Usage

### Dry run
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
    --model_name Qwen/Qwen3-0.6B \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --epochs 10 \
    --output_dir chris_output/sad_irt
```

### Configure frontier cutoff
```bash
python -m experiment_sad_irt.train_evaluate \
    --frontier_cutoff_date 20250807 \
    --pre_frontier_threshold 0.1 \
    --post_frontier_threshold 0.1
```

## Files

| File | Purpose |
|------|---------|
| `config.py` | Configuration dataclass with all hyperparameters |
| `model.py` | SADIRT and StandardIRT model definitions |
| `dataset.py` | TrajectoryIRTDataset with input formatting |
| `data_splits.py` | Agent/task splitting by date cutoff |
| `train.py` | Trainer class with gradient accumulation, scheduling |
| `evaluate.py` | Metrics computation (Spearman ρ, AUC, etc.) |
| `train_evaluate.py` | Main entry point |

## Key Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `frontier_cutoff_date` | 20250807 | Date separating pre/post frontier |
| `pre_frontier_threshold` | 0.1 | Max pass rate for pre-frontier |
| `post_frontier_threshold` | 0.1 | Min pass rate for post-frontier |
| `oracle_irt_dir` | `clean_data/swebench_verified_20251120_full/1d` | Pre-trained IRT |

## Baseline Variance Analysis

To understand whether improvements in Spearman ρ are meaningful, we evaluated the variance of the baseline IRT across 50 random seeds. This establishes the noise floor for the metric.

### Results (n=50 seeds)

| Statistic | Spearman ρ |
|-----------|------------|
| Mean | 0.378 |
| Std Dev | 0.029 |
| Min | 0.328 |
| Max | 0.447 |
| Range | 0.120 |

**Percentiles:**

| Percentile | Value |
|------------|-------|
| 5th | 0.337 |
| 25th | 0.355 |
| 50th (Median) | 0.371 |
| 75th | 0.401 |
| 95th | 0.427 |

**95% Confidence Interval: [0.322, 0.435]**

### Interpretation

The baseline IRT has substantial variance due to Pyro's stochastic variational inference. A single run can produce Spearman ρ anywhere from ~0.33 to ~0.45 purely due to random seed.

**To claim meaningful improvement over baseline:**
- Results should consistently exceed the **95th percentile (~0.43)**
- Or show improvement across multiple seeds

### Running the Variance Analysis

```bash
# On MIT Engaging cluster
sbatch slurm_scripts/run_baseline_variance.sh

# With custom parameters
sbatch slurm_scripts/run_baseline_variance.sh --num_seeds 100 --start_seed 0
```

Results are saved to `chris_output/baseline_variance/summary.json`.

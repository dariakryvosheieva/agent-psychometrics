# IRT Models

This document covers the Item Response Theory models used in the project.

## Model Selection Results

Model selection (AIC/BIC) indicates that a **1D IRT model** best fits the SWE-bench data:

| Model | Log-Lik | # Params | AIC | BIC |
|-------|---------|----------|-----|-----|
| **1D** | -17,481 | 1,123 | **37,209** | **47,346** |
| 2D | -17,175 | 2,246 | 38,842 | 59,116 |
| 3D | -16,867 | 3,369 | 40,471 | 70,882 |

1D is best by both AIC and BIC, indicating additional dimensions don't provide enough improvement to justify the extra parameters.

## 1PL vs 2PL Models

### 1PL (Rasch Model)

```
P(Y=1) = sigmoid(theta_j - b_i)
```

- Only difficulty parameter `b`
- Assumes equal discrimination across all tasks
- Simpler, more interpretable

### 2PL (Two-Parameter Logistic)

```
P(Y=1) = sigmoid(a_i * (theta_j - b_i))
```

Where:
- `theta_j` — agent ability (single scalar)
- `a_i` — item discrimination (how well the item differentiates ability levels)
- `b_i` — item difficulty (ability level needed for 50% chance of success)

## Training Commands

```bash
source .venv/bin/activate

# Train 1PL model (saves to 1d_1pl/)
python swebench_irt/train.py \
    --data_path clean_data/swebench_verified/swebench_verified_20251115_full.jsonl \
    --dims 1 \
    --model 1pl \
    --output_dir clean_data/swebench_verified_20251115_full \
    --epochs 5000

# Train 2PL model (saves to 1d/)
python swebench_irt/train.py \
    --data_path clean_data/swebench_verified/swebench_verified_20251115_full.jsonl \
    --dims 1 \
    --model 2pl \
    --output_dir clean_data/swebench_verified_20251115_full \
    --epochs 5000
```

## Training Configuration

**1D (hierarchical priors):**
- Learning rate: 0.1 with decay 0.9999
- Epochs: 5000

**MIRT (2D+):**
- Learning rate: 0.003 (reduced for stability)
- LR decay: 1.0 (disabled)
- Gradient clipping: clip_norm=5 via ClippedAdam
- Initializers: difficulty_from_accuracy + mirt_pca

## Output Structure

```
clean_data/swebench_verified_20251115_full/
├── 1d/               # 2PL model
│   ├── items.csv     # a, b, a_std, b_std (500 tasks)
│   └── abilities.csv # theta, theta_std (130 agents)
└── 1d_1pl/           # 1PL (Rasch) model
    ├── items.csv     # b, b_std only (500 tasks)
    └── abilities.csv # theta, theta_std (130 agents)
```

## Key Files

| File | Purpose |
|------|---------|
| `swebench_irt/train.py` | Train 1D-6D IRT models via py_irt |
| `swebench_irt/train_rep.py` | Train with multiple random seeds |
| `swebench_irt/compare_dims.py` | Compare models via AIC/BIC |
| `py_irt/` | Local fork with Multidim2PL model |

## Historical Note: Bug Fix

**Fixed (2025-01-06):** The original code incorrectly applied `np.exp()` to 1D discrimination parameters. The fix was to use raw values directly:

```python
# WRONG (was in original code):
discriminations = [np.exp(i) for i in trainer.best_params["disc"]]

# CORRECT (fixed):
discriminations = list(trainer.best_params["disc"])
```

This is because `TwoParamLog` model uses `Normal` distribution for discrimination (not `LogNormal`).

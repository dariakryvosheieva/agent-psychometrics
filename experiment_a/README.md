# Experiment A: Prior Validation (IRT AUC)

Evaluates how well a difficulty predictor can predict agent success on held-out tasks using the 1PL IRT model.

## Overview

**Goal**: Validate that predicted task difficulties are useful for forecasting agent performance without running agents on new tasks.

**Core Idea**: Given a predicted difficulty β̂_i and known agent ability θ_j, compute:

```
P(success) = sigmoid(θ_j - β̂_i)
```

Then measure AUC by comparing these predicted probabilities to actual binary outcomes.

## Quick Start

```bash
# Activate environment
source .venv/bin/activate

# Run with baselines only (no embeddings required)
python -m experiment_a.train_evaluate

# Run with pre-computed embeddings
python -m experiment_a.train_evaluate --embeddings_path /path/to/embeddings.npz

# Dry run to check config
python -m experiment_a.train_evaluate --dry_run
```

## Evaluation Protocol

1. **Split tasks** (not agents) into train/test sets using deterministic hash-based splitting
2. **Train difficulty predictor** on train tasks (e.g., embeddings → Ridge regression → difficulty)
3. **Predict difficulty** for test tasks
4. **Compute IRT probabilities**: For each (agent, task) pair, compute P(success) = sigmoid(θ - β̂)
5. **Calculate AUC**: Compare predicted probabilities to actual 0/1 outcomes

## Inputs

| Input | Description | Path |
|-------|-------------|------|
| Agent abilities (θ) | 1PL IRT ability parameters | `clean_data/.../1d_1pl/abilities.csv` |
| Task difficulties (b) | Ground truth IRT difficulties | `clean_data/.../1d_1pl/items.csv` |
| Response matrix | Binary success/failure outcomes | `chris_output/.../swebench_verified_20251115_full.jsonl` |
| Embeddings (optional) | Pre-computed task embeddings | `.npz` file from Daria's pipeline |

## Generating Embeddings

Run Daria's pipeline on the Engaging cluster:

```bash
sbatch predict_question_difficulty_engaging.sh
```

This produces:
```
out/prior_qwen3vl8b/
├── embeddings__Qwen__Qwen3-VL-8B-Instruct__*.npz  # Input for experiment_a
├── predictions.csv     # Per-task predictions (item_id, diff_true, diff_pred, split)
└── metrics.json        # Train/test R², Pearson r, RMSE
```

The `.npz` file contains:
- `task_ids`: Array of task identifiers (500 tasks)
- `X`: Embedding matrix (500 × embedding_dim)

## Baselines

| Method | AUC | Description |
|--------|-----|-------------|
| **Oracle** | 0.9447 | Upper bound using ground truth IRT difficulty |
| **Constant** | 0.7176 | Predict mean difficulty for all tasks |
| **Agent-only** | 0.7178 | Use agent's overall success rate, ignoring task |
| **Task-only** | 0.5000 | Use mean pass rate, ignoring agent (no discrimination) |

## Module Structure

```
experiment_a/
├── __init__.py               # Module exports
├── config.py                 # ExperimentAConfig dataclass
├── data_loader.py            # Load IRT params, responses; create splits
├── difficulty_predictor.py   # DifficultyPredictor protocol + implementations
├── irt_evaluation.py         # AUC computation using 1PL IRT formula
├── baselines.py              # Agent-only, task-only baselines
└── train_evaluate.py         # Main pipeline
```

### Key Classes

#### `DifficultyPredictor` (Protocol)

```python
class DifficultyPredictor(Protocol):
    def fit(self, task_ids: List[str], ground_truth_b: np.ndarray) -> None:
        """Train on tasks with known IRT difficulties."""
        ...

    def predict(self, task_ids: List[str]) -> Dict[str, float]:
        """Predict difficulty for tasks. Returns {task_id: predicted_b}."""
        ...
```

#### Implementations

| Class | Description |
|-------|-------------|
| `EmbeddingPredictor` | Load pre-computed embeddings + Ridge regression |
| `ConstantPredictor` | Predict mean difficulty for all tasks (baseline) |
| `GroundTruthPredictor` | Use actual IRT difficulties (oracle upper bound) |

### Key Functions

#### `compute_auc()`

```python
def compute_auc(
    predicted_difficulties: Dict[str, float],
    abilities: pd.DataFrame,
    responses: Dict[str, Dict[str, int]],
    task_ids: List[str],
) -> Dict[str, Any]:
    """Compute AUC for predicted difficulties on test tasks.

    For each (agent, task) pair:
    - y_true = actual response (0 or 1)
    - y_score = sigmoid(theta_j - predicted_beta_i)

    Returns dict with 'auc', 'n_pairs', 'n_tasks', 'n_agents'.
    """
```

#### `stable_split_tasks()`

```python
def stable_split_tasks(
    task_ids: List[str],
    test_fraction: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    """Deterministic train/test split using hash-based splitting.

    Same logic as Daria's stable_split_ids() for consistency.
    """
```

## Output

Results saved to `chris_output/experiment_a/experiment_a_results.json`:

```json
{
  "config": {
    "test_fraction": 0.2,
    "split_seed": 0,
    "embeddings_path": "/path/to/embeddings.npz",
    "ridge_alpha": 10000.0
  },
  "data_summary": {
    "n_agents": 130,
    "n_tasks_total": 500,
    "n_train_tasks": 400,
    "n_test_tasks": 100
  },
  "oracle": {"auc": 0.9447, "n_pairs": 13000},
  "embedding_predictor": {
    "auc_result": {"auc": 0.XX, "n_pairs": 13000},
    "difficulty_metrics": {"pearson_r": 0.XX, "rmse": X.XX}
  },
  "constant_baseline": {"auc": 0.7176},
  "agent_only_baseline": {"auc": 0.7178},
  "task_only_baseline": {"auc": 0.5000}
}
```

## Command Line Options

```
--test_fraction     Fraction of tasks for test set (default: 0.2)
--split_seed        Random seed for train/test split (default: 0)
--embeddings_path   Path to pre-computed embeddings .npz file
--ridge_alpha       Ridge regression alpha (default: 10000.0)
--output_dir        Output directory (default: chris_output/experiment_a)
--dry_run           Show configuration without running
```

## Relationship to Other Experiments

| Experiment | Focus | Metric |
|------------|-------|--------|
| **A (this)** | Can we predict success probability? | AUC |
| **B** | Can trajectory features improve difficulty prediction? | Pearson r |
| **D** | Is frontier ability linear over time? | R² of linear fit |

Experiment A validates the **prior** (difficulty prediction from task features alone).
Experiment B extends this with a **posterior** (adding trajectory features for hard tasks).

## References

- [Truong et al. (2025)](https://arxiv.org/pdf/2503.13335) - Amortized model-based evaluation
- IRT formula: `P = sigmoid(theta - beta)` matches py_irt's 1PL implementation

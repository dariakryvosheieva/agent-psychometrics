# Experiment A: Prior Validation (IRT AUC)

Evaluates how well a difficulty predictor can predict agent success on held-out tasks using the 1PL IRT model.

## Overview

**Goal**: Validate that predicted task difficulties are useful for forecasting agent performance without running agents on new tasks.

**Core Idea**: Given a predicted difficulty β̂_i and known agent ability θ_j, compute:

```
P(success) = sigmoid(θ_j - β̂_i)
```

Then measure AUC by comparing these predicted probabilities to actual binary outcomes.

This corresponds to **Section 3.1** in the [research proposal](../chris%20proposal.md).

## Quick Start

```bash
source .venv/bin/activate

# Run with baselines only (no embeddings required)
python -m experiment_a.train_evaluate

# Run with pre-computed embeddings
python -m experiment_a.train_evaluate --embeddings_path /path/to/embeddings.npz

# Run with Lunette features
python -m experiment_a.train_evaluate --lunette_features_path chris_output/experiment_a/sandbox_features/lunette_features.csv

# Dry run to check config
python -m experiment_a.train_evaluate --dry_run
```

## Evaluation Protocol

1. **Split tasks** (not agents) into train/test sets using deterministic hash-based splitting
2. **Train IRT on train tasks only** to get uncontaminated ground truth difficulties (avoids data leakage)
3. **Train difficulty predictor** on train tasks using train-only IRT difficulties as targets
4. **Predict difficulty** for test tasks
5. **Compute IRT probabilities**: For each (agent, task) pair, compute P(success) = sigmoid(θ - β̂)
6. **Calculate AUC**: Compare predicted probabilities to actual 0/1 outcomes

### Data Leakage Prevention

The IRT model provides ground truth difficulties (β) used as training targets. To avoid data leakage, we train **two separate IRT models**:

1. **IRT^train (Train-only IRT)**: Trained on train tasks (T1) only
   - Provides uncontaminated ground truth for training difficulty predictors
   - **Must be used for all actual methods** (embedding, constant, LLM judge, etc.)
   - Agent abilities θ and task difficulties β are both on the "train scale"

2. **IRT^full (Full IRT)**: Trained on all tasks (T1 ∪ T2)
   - **Used ONLY for oracle baseline** - shows theoretical best performance
   - The oracle is NOT a valid method - it's just a reference point for comparison

**Critical**: When computing AUC for any method, we use abilities from IRT^train. This ensures the abilities (θ) and predicted difficulties (β̂) are on the **same IRT scale**. The difficulty predictor is trained to predict β values on the train scale, so evaluation must also use abilities from that same scale. Using full IRT abilities would mix incompatible scales and leak test task information.

The split IRT models are cached in `chris_output/experiment_a/irt_splits/` and automatically reused when the split parameters match.

```bash
# To manually train split IRT model
python -m experiment_a.train_irt_split --dry_run  # See what would happen
python -m experiment_a.train_irt_split            # Train on train tasks only
python -m experiment_a.train_irt_split --force    # Force retrain even if cached
```

## Results (2026-01-13)

| Method | AUC | Description |
|--------|-----|-------------|
| Oracle (true b) | 0.9447 | Upper bound using ground truth IRT difficulty |
| **Embedding (MLE)** | **0.8337** | Direct MLE training (Truong et al. 2025) |
| **Embedding** | **0.8333** | Qwen3-VL-8B embeddings + Ridge (plug-in) |
| **LLM Judge** | **0.8071** | 9 semantic features with Lasso selection |
| **Lunette v2** | **0.7522** | 24 features with Lasso selection |
| Constant baseline | 0.7176 | Predict mean difficulty for all tasks |
| Agent-only | 0.7178 | Use agent's overall success rate |
| Task-only | 0.5000 | Use mean pass rate (no discrimination) |

### MLE vs Plug-in Training

Two approaches for training the embedding→difficulty mapping:

1. **Plug-in (default)**: Fit Ridge regression on ground-truth IRT difficulties as targets
2. **MLE**: Directly maximize the IRT log-likelihood of agent responses

The MLE approach (Truong et al. 2025) achieves slightly better AUC (0.8337 vs 0.8333), but the improvement is minimal on our dataset. This may be due to:
- Smaller dataset (500 tasks vs 78K items in their paper)
- Already well-tuned Ridge regularization (α=10000)

**MC ability marginalization**: The paper uses Monte Carlo sampling from N(0,1) to marginalize over abilities. We tested this approach (`--mle_use_mc_abilities`) but it performs worse (0.8258 AUC) because:
- We have reliable IRT-estimated abilities for all 130 agents
- MC marginalization ignores agent-specific information

To enable MLE training:
```bash
# With fixed abilities (recommended)
python -m experiment_a.train_evaluate --embeddings_path /path/to/embeddings.npz --use_mle_embedding

# With MC ability marginalization (not recommended for our setting)
python -m experiment_a.train_evaluate --embeddings_path /path/to/embeddings.npz --use_mle_embedding --mle_use_mc_abilities
```

## Feature Sources

### 1. Embeddings (Best Performance)

Run Daria's pipeline on the Engaging cluster:

```bash
sbatch predict_question_difficulty_engaging.sh
```

This produces:
```
out/prior_qwen3vl8b/
├── embeddings__Qwen__Qwen3-VL-8B-Instruct__*.npz  # Input for experiment_a
├── predictions.csv     # Per-task predictions
└── metrics.json        # Train/test R², Pearson r
```

### 2. Lunette Features (with Sandbox Access)

Lunette feature extraction uses a **two-step process** to ensure the grading judge has sandbox access:

#### Step 1: Create Sandbox Runs

Run a dummy solver through Inspect with `--sandbox lunette` to provision Docker containers:

```bash
# Dry run to see execution plan
python -m experiment_a.run_dummy_sandbox --dry_run

# Run with batching (25 tasks per Inspect invocation)
python -m experiment_a.run_dummy_sandbox --batch_size 25

# Resume after interruption
python -m experiment_a.run_dummy_sandbox --resume

# Monitor progress
cat chris_output/experiment_a/sandbox_runs/tracking.json | python -c "import json,sys; d=json.load(sys.stdin); print(f\"Progress: {d['stats']['completed']}/{d['stats']['total']}\")"
```

This creates sandboxes with the actual repo checkout at the correct commit for each task.

#### Step 2: Grade Sandbox Runs

Extract features using Lunette's investigate() API:

```bash
# Dry run
python -m experiment_a.grade_sandbox_runs --dry_run

# Grade all completed sandbox runs
python -m experiment_a.grade_sandbox_runs --skip_existing

# Grade specific tasks
python -m experiment_a.grade_sandbox_runs --task_ids "django__django-12345,astropy__astropy-12907"
```

**Important limitations:**
- Each investigation takes ~55 seconds
- Batch grading (limit > 1) causes 504 Gateway Timeout errors
- Tasks must be graded one at a time

**Output:** `chris_output/experiment_a/sandbox_features/lunette_features.csv`

#### Feature Schema (25 features)

**Environment-based (15)**: repo_file_count, repo_line_count, patch_file_count, patch_line_count, test_file_count, related_file_count, import_count, class_count_in_file, function_count_in_file, test_count_fail_to_pass, test_count_pass_to_pass, git_commit_count, directory_depth, has_conftest, has_init

**Semantic (10)**: fix_in_description (0-3), problem_clarity (1-5), error_message_provided (0/1), reproduction_steps (0/1), fix_locality (1-3), domain_knowledge_required (1-5), fix_complexity (1-5), logical_reasoning_required (1-5), atypicality (1-5), reasoning (text)

**Note:** The Lunette API currently returns a simplified `{name, score, explanation}` format instead of the full structured schema. Check with Fulcrum team about `result_schema` support.

### 3. LLM Judge Features (Alternative)

For feature extraction without sandbox access, use the direct LLM judge approach:

```bash
python -m experiment_a.compute_llm_judge_features --dry_run
python -m experiment_a.compute_llm_judge_features --limit 100
```

This uses Claude directly with structured output but cannot run shell commands in the repo.

## Module Structure

```
experiment_a/
├── __init__.py                    # Module exports
├── config.py                      # ExperimentAConfig dataclass
├── data_loader.py                 # Load IRT params, responses; create splits
├── train_irt_split.py             # Train IRT on train tasks only (avoids leakage)
├── difficulty_predictor.py        # DifficultyPredictor protocol + implementations
├── irt_evaluation.py              # AUC computation using 1PL IRT formula
├── baselines.py                   # Agent-only, task-only baselines
├── train_evaluate.py              # Main evaluation pipeline
├── run_dummy_sandbox.py           # Step 1: Create sandbox runs via Inspect
├── grade_sandbox_runs.py          # Step 2: Grade runs with Lunette investigate()
├── lunette_structured_output.py   # Pydantic schemas for structured output
├── compute_llm_judge_features.py  # Alternative: Direct LLM feature extraction
└── llm_judge_prompt.py            # Prompts for LLM judge
```

## Output

Results saved to `chris_output/experiment_a/experiment_a_results.json`:

```json
{
  "config": {...},
  "data_summary": {"n_agents": 130, "n_tasks_total": 500, "n_train": 400, "n_test": 100},
  "oracle": {"auc": 0.9447},
  "embedding_predictor": {"auc_result": {"auc": 0.XX}},
  "constant_baseline": {"auc": 0.7176},
  "agent_only_baseline": {"auc": 0.7178},
  "task_only_baseline": {"auc": 0.5000}
}
```

**Note**: The oracle uses full IRT (reference only). All other methods use IRT^train abilities.

## Command Line Options

```
--test_fraction       Fraction of tasks for test set (default: 0.2)
--split_seed          Random seed for train/test split (default: 0)
--embeddings_path     Path to pre-computed embeddings .npz file
--lunette_features_path  Path to Lunette features CSV
--llm_judge_features_path  Path to LLM judge features CSV
--ridge_alpha         Ridge regression alpha (default: 10000.0)
--output_dir          Output directory (default: chris_output/experiment_a)
--use_mle_embedding   Enable MLE embedding predictor (Truong et al. 2025)
--mle_l2_lambda       L2 regularization for MLE weights (default: 0.15)
--dry_run             Show configuration without running
```

## Known Issues

**Train/Test Split Bias:** The hash-based split has statistically significant bias (p=0.025):
- Train tasks: mean b = +0.61
- Test tasks: mean b = -0.27
- Effect size: Cohen's d = 0.25 (small)

For fully reliable results, run feature extraction on all 500 tasks.

**Lunette Structured Output:** The Lunette API currently ignores custom `result_schema` and returns a hardcoded `{name, score, explanation}` format. The features need to be parsed from the explanation text or use the LLM Judge alternative.

## References

- [Truong et al. (2025)](https://arxiv.org/pdf/2503.13335) - Amortized model-based evaluation
- IRT formula: `P = sigmoid(theta - beta)` matches py_irt's 1PL implementation
- [Lunette documentation](../lunette_utils/LUNETTE.md) - Detailed Lunette integration guide

# Trajectory Features for Experiment B

Extracts behavioral features from agent trajectories to predict task difficulty.

## Features Extracted

| Feature | Scale | Description | Expected Direction |
|---------|-------|-------------|-------------------|
| `loop_detection` | 0-5 | Did the agent repeat similar failed approaches? | + (more loops = harder) |
| `localization_quality` | 0-5 | How well did the agent identify the correct code location? | - (better = easier) |
| `debugging_cycles` | count | Number of debug-fix cycles observed | + (more cycles = harder) |
| `error_recovery` | 0-5 | How well did the agent recover from errors? | - (better recovery = easier) |
| `exploration_breadth` | count | How many distinct files/approaches explored? | + (more exploration = harder) |
| `focus_drift` | 0-5 | Did the agent stay on task or get distracted? | + (more drift = harder) |
| `solution_completeness` | 0-5 | How complete was the attempted solution? | - (more complete = easier) |
| `edge_case_handling` | 0-5 | Did the agent consider and handle edge cases? | - (better handling = easier) |
| `test_verification` | 0-5 | Did the agent verify their solution works? | - (better verification = easier) |

## Output Files

- `chris_output/trajectory_features/raw_features_500tasks_6agents.csv` - Raw per-trajectory features (500 tasks x 6 agents = 3000 rows)
- `chris_output/trajectory_features/aggregated_features.csv` - Task-level aggregated features (500 tasks x 20 columns)

### Aggregated Feature Columns (20 total)

For each raw feature, the aggregation produces:
- `{feat}_mean` - Mean across all agents (9 features)
- `{feat}_std` - Standard deviation across agents (9 features)

Plus trajectory metadata:
- `trajectory_length_mean` - Mean trajectory length across agents
- `trajectory_length_std` - Std of trajectory length

**Note:** The `--behavioral_only` flag excludes potentially leaky features (pass_rate, n_pass, n_fail, ability_weighted) to ensure the model doesn't learn from outcome information.

## Agent Selection

6 pre-frontier agents (date < 20250501) selected for optimal ability spread:

| Agent | IRT Ability (θ) | Avg Tokens |
|-------|----------------|------------|
| 20250415_openhands | +1.65 | 27.4K |
| 20250410_cortexa | +1.03 | 3.2K |
| 20241029_OpenHands-CodeAct-2.1-sonnet-20241022 | +0.50 | 26.2K |
| 20241108_autocoderover-v2.0-claude-3-5-sonnet-20241022 | -0.22 | 17.9K |
| 20240721_amazon-q-developer-agent-20240719-dev | -0.91 | 1.9K |
| 20241002_lingma-agent_lingma-swe-gpt-72b | -1.60 | 10.8K |

Selection criteria:
1. Date before cutoff (20250501) to avoid contamination
2. 100% task coverage (500/500 tasks in unified_trajs)
3. Optimal ability spread (gaps of 0.53-0.72 points)

## Usage

### Extract features for all tasks
```bash
python -m experiment_b.trajectory_features.extract_missing \
    --existing_path chris_output/trajectory_features/raw_features_prefrontier_clean.csv \
    --output_path chris_output/trajectory_features/raw_features_500tasks_6agents.csv \
    --n_tasks 500 \
    --agents_per_task 6 \
    --parallel 100
```

### Aggregate features (behavioral only, no outcome leakage)
```bash
python -m experiment_b.trajectory_features.aggregate_features \
    --input_path chris_output/trajectory_features/raw_features_500tasks_6agents.csv \
    --output_path chris_output/trajectory_features/aggregated_features.csv \
    --behavioral_only
```

### Dry run to see extraction plan
```bash
python -m experiment_b.trajectory_features.extract_missing \
    --existing_path chris_output/trajectory_features/raw_features_prefrontier_clean.csv \
    --output_path chris_output/trajectory_features/raw_features_500tasks_6agents.csv \
    --n_tasks 500 \
    --agents_per_task 6 \
    --dry_run
```

## Cost Estimates

Using Claude Sonnet 4.5 ($3/M input, $15/M output):
- ~$0.044 per trajectory
- 500 tasks x 6 agents = ~$130 total

## Files

- `config.py` - Agent selection and feature definitions
- `prompts.py` - LLM prompt for feature extraction
- `extract_features.py` - Core extraction logic
- `extract_missing.py` - Script to fill in missing trajectories
- `aggregate_features.py` - Aggregate raw features to task-level

## Experiment B Results

With behavioral-only features (no outcome leakage):

| Method | ROC-AUC (pass-rate) | ROC-AUC (IRT) |
|--------|---------------------|---------------|
| Oracle | 0.8439 | 0.7810 |
| SAD-IRT (best) | 0.8036 | 0.7315 |
| Feature-IRT (Embedding) | 0.7747 | 0.7286 |
| Grouped Ridge (Embedding + Trajectory) | 0.7495 | 0.7208 |
| Trajectory + Ridge | 0.7234 | 0.7111 |
| Baseline IRT | 0.7600 | 0.6966 |

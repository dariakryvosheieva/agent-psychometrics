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

- `chris_output/trajectory_features/raw_features_500tasks_6agents.csv` - Raw per-trajectory features (499 tasks × ~9 agents per task)
- `chris_output/trajectory_features/aggregated_features.csv` - Task-level aggregated features (499 tasks × 33 columns)

### Aggregated Feature Columns

For each raw feature, the aggregation produces:
- `{feat}_mean` - Mean across all agents
- `{feat}_std` - Standard deviation across agents
- `{feat}_ability_weighted` - Mean weighted by agent IRT ability

Plus metadata: `n_agents`, `n_pass`, `n_fail`, `pass_rate`, `trajectory_length_mean`, `trajectory_length_std`

## Usage

### Extract features for new tasks
```bash
python -m experiment_b.trajectory_features.extract_missing \
    --existing_path chris_output/trajectory_features/raw_features_all.csv \
    --output_path chris_output/trajectory_features/raw_features_500tasks_6agents.csv \
    --n_tasks 500 \
    --agents_per_task 6 \
    --parallel 100
```

### Dry run to see extraction plan
```bash
python -m experiment_b.trajectory_features.extract_missing \
    --existing_path chris_output/trajectory_features/raw_features_all.csv \
    --output_path chris_output/trajectory_features/raw_features_500tasks_6agents.csv \
    --n_tasks 500 \
    --agents_per_task 6 \
    --dry_run
```

## Agent Selection

20 agents selected to span the IRT ability spectrum (θ from -1.26 to +2.24), all with trajectories < 120K tokens to fit in Claude's context window. See `config.py` for the full list.

## Cost Estimates

Using Claude Sonnet 4.5 ($3/M input, $15/M output):
- ~$0.044 per trajectory
- 500 tasks × 6 agents = ~$130 total

### Aggregate features
```bash
python -m experiment_b.trajectory_features.aggregate_features \
    --input_path chris_output/trajectory_features/raw_features_500tasks_6agents.csv \
    --output_path chris_output/trajectory_features/aggregated_features.csv
```

## Files

- `config.py` - Agent selection and feature definitions
- `prompts.py` - LLM prompt for feature extraction
- `extract_features.py` - Core extraction logic
- `extract_missing.py` - Script to fill in missing trajectories
- `aggregate_features.py` - Aggregate raw features to task-level

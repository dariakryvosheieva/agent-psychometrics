# Tensor Analysis

Extends binary response matrices (agents × tasks → 0/1) with trajectory-level features and runs PCA analysis on response matrices.

## Modules

| Module | Purpose |
|--------|---------|
| `trajectory_features.py` | Extract features from agent trajectories (char counts) |
| `eda.py` | Distribution analysis and visualization |
| `response_matrix.py` | PCA analysis on response matrices |

## Data Coverage

### Response Matrices (PCA)

Complete response matrices for all 4 datasets:

| Dataset | Agents | Tasks | Entries |
|---------|--------|-------|---------|
| SWE-bench Verified | 131 | 500 | 65,500 |
| SWE-bench Pro | 14 | 730 | 10,220 |
| TerminalBench | 83 | 88 | 7,304 |
| GSO | 14 | 102 | 1,428 |

### Trajectory Features

Trajectory data is only available for SWE-bench datasets:

| Dataset | Agents | Tasks | Trajectories |
|---------|--------|-------|--------------|
| SWE-bench Verified | 44 | 500 | 22,000 |
| SWE-bench Pro | 10 | 398 | 3,980 |

**Filtering applied:**
- **Verified**: 44 agents with complete coverage (500 tasks, 0 broken). Excluded agents with incomplete task coverage or broken trajectories (e.g., `20250203_openhands_4x_scaled` has 22% broken).
- **Pro**: 10 agents after excluding 4 with >10% broken trajectories (`gpt_5`, `gpt_5_codex`, `gpt_5_high`, `gpt_oss`). Restricted to 398 tasks common across all 10 remaining agents.

## Key Results

### PCA Analysis

| Dataset | PC1 | PC2 | PC1+PC2 |
|---------|-----|-----|---------|
| SWE-bench Verified | 33.9% | 6.1% | 40.1% |
| SWE-bench Pro | 26.4% | 11.6% | 38.0% |
| TerminalBench | 29.1% | 6.4% | 35.5% |
| GSO | 24.2% | **17.8%** | 42.0% |

GSO has notably higher PC2 variance, suggesting meaningful structure beyond overall ability.

### Trajectory EDA

**Failures have ~55% longer trajectories than successes:**
- Success: mean=28.2K chars, median=11.1K
- Failure: mean=43.4K chars, median=21.4K

## Usage

```bash
# Extract trajectory features
python -m tensor_analysis.trajectory_features

# Run distribution EDA
python -m tensor_analysis.eda

# Run PCA analysis
python -m tensor_analysis.response_matrix
```

## Output Files

All outputs saved to `chris_output/tensor_analysis/`:

- `swebench_verified_char_counts.csv` - Character counts (44 agents × 500 tasks)
- `swebench_pro_char_counts.csv` - Character counts (10 agents × 398 tasks)
- `pca_results.json` - PCA explained variance and agent scores
- `plots/` - EDA and PCA visualizations

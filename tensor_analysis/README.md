# Tensor Analysis

Extends binary response matrices (agents × tasks → 0/1) with trajectory-level features and runs PCA analysis on response matrices.

## Modules

| Module | Purpose |
|--------|---------|
| `trajectory_features.py` | Extract features from agent trajectories (char counts) |
| `eda.py` | Distribution analysis and visualization |
| `response_matrix.py` | PCA analysis on response matrices |
| `tensor_decomposition.py` | CP/Tucker tensor decomposition on (agents × tasks × features) |
| `per_agent_correlation.py` | Per-agent correlation analysis (char count vs IRT difficulty) |

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

### PC1/PC2 Structure: The Guttman Effect

The PC1-PC2 scatter plot for SWE-bench Verified shows a striking inverted parabola pattern. This is the **Guttman effect** (also called horseshoe/arch effect), a well-known artifact in PCA when data is truly unidimensional.

**Mathematical basis:** When data lies along a single latent factor, the kth principal component becomes an orthogonal polynomial of degree k in the first factor ([Guttman 1953](https://www.researchgate.net/publication/228760485_The_Guttman_effect_Its_interpretation_and_a_new_redressing_method), [Benzécri 1973](https://pmc.ncbi.nlm.nih.gov/articles/PMC9196093/)):
- PC1 ≈ linear function of latent factor
- PC2 ≈ quadratic polynomial (degree 2)
- PC3 ≈ cubic polynomial (degree 3)

**Our findings:**
- PC1 vs IRT ability: tanh fit R²=0.998 (much better than linear R²=0.948)
- PC2 vs IRT ability: inverted Gaussian R²=0.923 (captures quadratic structure)
- PC1 ≈ Score (r=0.9946), and Score ≈ TCC(θ) ≈ tanh(θ) by IRT theory

**Interpretation:** The parabolic PC2 is NOT a meaningful second dimension — it's a mathematical consequence of applying PCA to unidimensional data. This provides independent validation that SWE-bench Verified is truly unidimensional, consistent with 1D IRT being best by AIC/BIC.

### Trajectory EDA

**Failures have ~55% longer trajectories than successes:**
- Success: mean=28.2K chars, median=11.1K
- Failure: mean=43.4K chars, median=21.4K

### Tensor Decomposition (CP/Tucker)

Built 3D tensors (agents × tasks × 2 features) where features are [resolved, standardized_char_count] and ran CP and Tucker decomposition.

**SWE-bench Verified (44 × 500 × 2):**

| Method | Rank/Size | Explained Variance |
|--------|-----------|-------------------|
| CP | 1 | 38.0% |
| CP | 2 | 70.2% |
| CP | 3 | 70.8% |
| CP | 5 | 80.5% |
| Tucker | (2,2,2) | 70.2% |
| Tucker | (5,5,2) | 81.4% |

**Feature Loadings (CP rank=2):**
- Component 1: +resolved, −char_count (success = shorter trajectories)
- Component 2: ~0 resolved, +char_count (trajectory length dimension)

**Validation Against IRT:**
- Agent Factor 1 vs IRT ability: **r = 0.924** (p < 1e-18)
- Task Factor 1 vs IRT difficulty: **r = −0.958** (p < 1e-272)

**Guttman Effect Check:**
- Factor 2 vs IRT ability: **quadratic R² = 0.087** ✓ (no Guttman effect)

**What Factor 2 Captures:**
- Factor 2 vs mean agent char count: **r = 0.997** (near-perfect!)
- Factor 2 represents **agent-level verbosity** — some agents consistently write longer trajectories than others, independent of ability

**Interpretation:** The tensor decomposition reveals **two meaningful dimensions**:
1. **Component 1**: Ability/difficulty axis (r = 0.92 with IRT ability)
2. **Component 2**: Agent verbosity axis (r = 0.997 with mean trajectory length)

The second dimension is NOT the Guttman artifact — it captures real variance in how verbose different agents are, regardless of their success rate. Examples:
- High verbosity (Factor 2 > 3): honeycomb, frogmini-14b, agentless-1.5
- Low verbosity (Factor 2 < -0.8): cortexa, bloop, cortexa_o3

Note: PCA on binary responses alone DOES show the Guttman effect. But when trajectory length is included as a second feature, it captures genuine agent-level variance.

### Per-Agent Correlation Analysis

The tensor decomposition found aggregate correlation between trajectory length and task difficulty (r = −0.958 for Task Factor 1). We investigated whether this holds at the **per-agent level** and critically, whether it persists on **frontier tasks**.

**All Tasks (n=500):**

| Metric | Value |
|--------|-------|
| Mean r | **+0.205** |
| Median r | +0.210 |
| Std | 0.132 |
| Range | [−0.38, +0.47] |
| Significant (p<0.05) | 41/44 (93%) |
| Negative & significant | 1/44 (2%) |

Within each agent, longer trajectories correlate with **harder** tasks (positive r).

**Frontier Tasks Only (n=34, zero_pre definition):**

| Metric | Value |
|--------|-------|
| Mean r | **−0.099** |
| Median r | −0.097 |
| Std | 0.135 |
| Range | [−0.36, +0.23] |
| Significant (p<0.05) | 2/43 (5%) |

**Key Finding:** The correlation **disappears on frontier tasks**. This confirms the same phenomenon found in Experiment B's ordered logit IRT analysis: trajectory features from pre-frontier agents cannot distinguish between frontier tasks because all agents fail on them. The trajectories are all similarly long/failed, providing no signal to differentiate "more impossible" from "less impossible" tasks.

**Implication:** Per-agent trajectory length features will NOT help predict difficulty for frontier tasks in Experiment B.

**SWE-bench Pro (10 × 398 × 2):**

| Method | Rank/Size | Explained Variance |
|--------|-----------|-------------------|
| CP | 1 | 37.1% |
| CP | 2 | 51.9% |
| CP | 3 | 72.9% |
| CP | 5 | 85.4% |
| Tucker | (2,2,2) | 57.6% |
| Tucker | (5,5,2) | 85.8% |

Feature loadings match Verified pattern. No IRT data available for Pro validation.

Note: `resolved` status is joined from the response matrix (`swebench_pro.jsonl`) since trajectory files don't include outcome data.

## Usage

```bash
# Extract trajectory features
python -m tensor_analysis.trajectory_features

# Run distribution EDA
python -m tensor_analysis.eda

# Run PCA analysis
python -m tensor_analysis.response_matrix

# Run tensor decomposition
python -m tensor_analysis.tensor_decomposition

# Run per-agent correlation analysis
python -m tensor_analysis.per_agent_correlation
```

## Output Files

All outputs saved to `chris_output/tensor_analysis/`:

- `swebench_verified_char_counts.csv` - Character counts (44 agents × 500 tasks)
- `swebench_pro_char_counts.csv` - Character counts (10 agents × 398 tasks)
- `per_agent_correlations_all_tasks.csv` - Per-agent correlation results (all 500 tasks)
- `per_agent_correlations_frontier.csv` - Per-agent correlation results (34 frontier tasks)
- `per_agent_correlation_histograms.png` - Side-by-side histograms of r values
- `agent_char_count_features.csv` - Z-score normalized feature matrix (500 × 44)
- `pca_results.json` - PCA explained variance and agent scores
- `plots/` - EDA and PCA visualizations
- `decomposition/` - Tensor decomposition results:
  - `decomposition_summary.json` - Explained variance for all methods
  - `*_scree.png` - Scree plots for rank selection
  - `*_feature_loadings.png` - How features load on components
  - `*_agent_factors.png` - Agent factor scatter + IRT validation
  - `*_task_factors.png` - Task factor scatter + IRT validation
  - `*_guttman_check.png` - Quadratic fit to detect Guttman artifact

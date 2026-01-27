# LLM Judge Feature Extraction

Extract and evaluate features for predicting task difficulty using LLM judges and deterministic analysis.

## Quick Start

```bash
# Extract deterministic features (no API cost)
python -m experiment_ab_shared.llm_judge deterministic \
    --dataset swebench \
    --output chris_output/deterministic_features.csv

# Evaluate feature quality
python -m experiment_ab_shared.llm_judge quick_eval \
    --features-csv chris_output/deterministic_features.csv \
    --irt-items clean_data/swebench_verified_20251120_full/1d/items.csv

# Compare hard vs easy tasks qualitatively
python -m experiment_ab_shared.llm_judge compare \
    --dataset swebench --n-pairs 3
```

## CLI Commands

### `deterministic` - Extract Deterministic Features

Extract features computed directly from task data (no LLM calls required).

```bash
python -m experiment_ab_shared.llm_judge deterministic \
    --dataset swebench \
    --output output.csv
```

**Supported datasets:** `swebench`, `swebench_pro`, `gso`, `terminalbench`

**Features extracted (25 for SWE-bench):**
- Patch features: `num_files_modified`, `num_hunks`, `num_lines_changed`, `log_lines_changed`, `patch_adds`, `patch_deletes`, `patch_chars`, `patch_files_gt2`, `patch_files_gt3`, `patch_lines_gt20`, `patch_lines_gt50`
- Test patch features: `test_patch_chars`, `test_patch_hunks`, `test_patch_files`, `test_patch_lines`
- Statement features: `stmt_words`, `stmt_chars`, `stmt_lines`, `stmt_lines_gt80`, `has_http_link`, `has_code_block`, `has_stack_trace`, `feature_request_phrasing`
- Metadata: `is_django`, `is_sympy`

---

### `quick_eval` - Evaluate Feature Quality

Compute correlations with IRT difficulty, run Lasso feature selection, and detect redundant features.

```bash
python -m experiment_ab_shared.llm_judge quick_eval \
    --features-csv features.csv \
    --irt-items clean_data/swebench_verified_20251120_full/1d/items.csv \
    --output results.json
```

**Options:**
- `--correlation-threshold` - p-value threshold for significance (default: 0.05)
- `--redundancy-threshold` - Correlation threshold for redundant pairs (default: 0.9)

**Output includes:**
- Individual Pearson/Spearman correlations with p-values
- Lasso coefficients for feature selection
- Redundant feature pairs (|r| > 0.9)
- Recommendations: which features to keep/drop

---

### `compare` - Qualitative Comparison

Sample and display hard vs easy tasks side-by-side for manual feature validation.

```bash
# Show one pair
python -m experiment_ab_shared.llm_judge compare --dataset swebench

# Show multiple pairs
python -m experiment_ab_shared.llm_judge compare --dataset swebench --n-pairs 5

# Interactive mode (press Enter for more pairs)
python -m experiment_ab_shared.llm_judge compare --dataset swebench -i

# Highlight a specific feature
python -m experiment_ab_shared.llm_judge compare \
    --dataset swebench \
    --feature cross_cutting_fix \
    --features-csv features.csv
```

---

### `extract` - Extract LLM Judge Features

Extract semantic features via LLM API calls (parallel, resumable).

```bash
# Dry run (cost estimate)
python -m experiment_ab_shared.llm_judge extract \
    --dataset swebench_v3 \
    --provider openai \
    --model gpt-4.1 \
    --dry-run

# Full extraction (parallel)
python -m experiment_ab_shared.llm_judge extract \
    --dataset swebench_v3 \
    --provider openai \
    --model gpt-4.1 \
    --parallel \
    --concurrency 10 \
    --output-dir chris_output/llm_judge_v3

# Add deterministic features to output
python -m experiment_ab_shared.llm_judge extract \
    --dataset swebench_v2 \
    --add-deterministic
```

**Options:**
- `--provider` - `openai` or `anthropic`
- `--model` - Model name (e.g., `gpt-4.1`, `claude-sonnet-4-20250514`)
- `--parallel` - Run extraction in parallel
- `--concurrency` - Max concurrent API calls (default: 10)
- `--limit` - Max tasks to process
- `--task-ids` - Comma-separated specific task IDs
- `--add-deterministic` - Add deterministic features to output CSV

---

### `verify` - Verify Extraction Completeness

Check which tasks have been extracted and report any missing/failed.

```bash
python -m experiment_ab_shared.llm_judge verify \
    --dataset swebench \
    --output-dir chris_output/llm_judge_features
```

---

### `aggregate` - Aggregate JSON to CSV

Combine per-task JSON files into a single CSV.

```bash
python -m experiment_ab_shared.llm_judge aggregate \
    --dataset swebench \
    --output-dir chris_output/llm_judge_features
```

---

### `correlations` - Analyze Feature Correlations

Run detailed correlation analysis with IRT difficulty.

```bash
python -m experiment_ab_shared.llm_judge correlations \
    --features-csv features.csv \
    --irt-items clean_data/swebench_verified_20251120_full/1d/items.csv \
    --dataset swebench
```

---

## Available Prompts

| Prompt | Features | Description |
|--------|----------|-------------|
| `swebench` | 9 | Original SWE-bench features |
| `swebench_v2` | 9 | Strong features + V5 additions |
| `swebench_v3` | 6 | Cross-cutting semantic features |
| `swebench_unified` | 6 | Standardized features for comparison |
| `terminalbench` | 5 | TerminalBench-specific features |
| `gso` | 6 | GSO optimization features |

### swebench_v3 Features

New semantic features focused on cross-cutting concerns:
- `cross_cutting_fix` (1-5): Requires coordinated multi-module edits
- `domain_category` (0-9): Primary domain (infrastructure, auth, database, etc.)
- `change_scope` (1-3): Local/module/system-wide changes
- `api_boundary_crossing` (0/1): Crosses public API boundaries
- `implicit_requirements` (1-5): Undocumented requirements
- `coordination_complexity` (1-5): Inter-component coordination needed

---

## Workflow

```
1. DETERMINISTIC FEATURES (no cost)
   └── python -m ... deterministic --dataset swebench --output det.csv

2. QUICK EVAL (validate features)
   └── python -m ... quick_eval --features-csv det.csv --irt-items items.csv

3. QUALITATIVE CHECK (manual sanity)
   └── python -m ... compare --dataset swebench -i

4. LLM EXTRACTION (if needed)
   └── python -m ... extract --dataset swebench_v3 --parallel

5. COMBINE & EVALUATE
   └── Concatenate deterministic + LLM features
   └── Run quick_eval on combined features
```

---

## Key Files

| File | Purpose |
|------|---------|
| `__main__.py` | CLI entry point with all subcommands |
| `deterministic_features.py` | Patch, test patch, statement, metadata features |
| `quick_eval.py` | Correlation analysis, Lasso, redundancy detection |
| `qualitative_compare.py` | Hard vs easy task comparison |
| `extract_pipeline.py` | Extraction verification and augmentation |
| `extractor.py` | LLM feature extraction with caching |
| `prompts/` | Prompt configurations for each dataset |

---

## Integration with Experiment A

The extracted features integrate with the experiment_a pipeline via `CSVFeatureSource`:

```python
from experiment_ab_shared.feature_source import CSVFeatureSource

# Use LLM judge features
llm_source = CSVFeatureSource(
    csv_path="chris_output/llm_judge_features/llm_judge_features.csv",
    task_id_column="_instance_id",
    feature_columns=["fix_complexity", "logical_reasoning_required", ...],
)

# Combine with embeddings using GroupedFeatureSource
from experiment_ab_shared.feature_source import GroupedFeatureSource

grouped = GroupedFeatureSource(
    sources={
        "llm_judge": llm_source,
        "embedding": embedding_source,
    }
)
```

Run the full comparison:

```bash
python -m experiment_a.run_all_datasets --unified_judge
```

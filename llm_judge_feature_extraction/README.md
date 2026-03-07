# LLM Judge Feature Extraction

Extract task difficulty features using LLM judges with batched, info-level-aware extraction and prefix caching.

## Architecture

**28 features** across 4 info levels, applied to all datasets:

| Info Level | # Features | Data Seen | Extraction Method |
|------------|-----------|-----------|-------------------|
| PROBLEM | 15 | Problem statement only | `BatchedFeatureExtractor` |
| ENVIRONMENT | 8 | Problem + shell exploration | Auditor agent (Docker/Inspect) |
| TEST | 3 | Problem + test artifact | `BatchedFeatureExtractor` |
| SOLUTION | 2 | Problem + tests + gold solution | `BatchedFeatureExtractor` |

Features are defined in a single registry (`feature_registry.py`) with per-dataset scale text variants (`code`, `terminal`, `optimization`).

## Quick Start

```bash
# Dry run — see batch plan and cost estimate
python -m llm_judge_feature_extraction extract \
    --all --dataset swebench_verified --dry-run

# Extract all non-environment features (parallel)
python -m llm_judge_feature_extraction extract \
    --all --dataset swebench_verified --parallel --concurrency 10

# Extract specific features
python -m llm_judge_feature_extraction extract \
    --features solution_hint,problem_clarity,test_comprehensiveness \
    --dataset swebench_verified

# Aggregate existing JSONs to CSV
python -m llm_judge_feature_extraction aggregate \
    --dataset swebench_verified \
    --output-dir chris_output/experiment_a/llm_judge_features

# Analyze feature correlations with IRT difficulty
python -m llm_judge_feature_extraction correlations \
    --features-csv path/to/features.csv \
    --irt-items data/swebench_verified/irt/1d_1pl/items.csv
```

## CLI Commands

### `extract` — Extract LLM Judge Features

Features are automatically grouped by info level and batched (≤7 per API call) with prefix caching.

```bash
python -m llm_judge_feature_extraction extract \
    --all --dataset swebench_verified \
    --provider openai --model gpt-5.4-2026-03-05 \
    --parallel --concurrency 10
```

**Options:**
- `--dataset` — `gso`, `swebench_pro`, `swebench_verified`, `terminalbench`
- `--features` — Comma-separated feature names, or `--all` for all non-environment features
- `--provider` — `openai` (default) or `anthropic`
- `--model` — Model name (default: provider's default)
- `--batch-size` — Max features per API call (default: 7)
- `--parallel` / `--concurrency` — Parallel extraction
- `--limit` / `--task-ids` — Filter tasks
- `--dry-run` — Show plan and cost estimate without running

### `aggregate` — Aggregate JSON to CSV

Combine per-task JSON files into a single CSV.

```bash
python -m llm_judge_feature_extraction aggregate \
    --dataset swebench_verified \
    --output-dir chris_output/experiment_a/llm_judge_features
```

### `correlations` — Analyze Feature Correlations

Run correlation analysis between features and IRT difficulty.

```bash
python -m llm_judge_feature_extraction correlations \
    --features-csv features.csv \
    --irt-items data/swebench_verified/irt/1d_1pl/items.csv \
    --dataset swebench_verified
```

## Datasets

| Dataset | Key | Scale Variant |
|---------|-----|---------------|
| SWE-bench Verified | `swebench_verified` | `code` |
| SWE-bench Pro | `swebench_pro` | `code` |
| TerminalBench | `terminalbench` | `terminal` |
| GSO | `gso` | `optimization` |

## Key Files

| File | Purpose |
|------|---------|
| `__main__.py` | CLI entry point (`extract`, `aggregate`, `correlations`) |
| `feature_registry.py` | Single registry of all 28 features with scale text |
| `prompt_config.py` | `FeatureDefinition` and `InfoLevel` data types |
| `task_context.py` | Per-dataset, per-info-level task formatters |
| `batched_extractor.py` | Batched extraction with info level grouping |
| `api_client.py` | LLM API client with prefix caching |
| `task_loaders.py` | Task loading functions for all datasets |
| `response_parser.py` | JSON response parsing and validation |
| `analyze_feature_correlations.py` | Correlation analysis with IRT difficulty |
| `sandbox_utils.py` | Docker sandbox configuration helpers |
| `auditor_agent/` | Environment-level feature extraction via Inspect |

## Environment Features (Auditor Agent)

Environment-level features (8 features) are extracted via an agentic pipeline using Docker containers and the Inspect framework. See `auditor_agent/` for details.

```bash
# Run auditor on SWE-bench Verified
inspect eval llm_judge_feature_extraction/auditor_agent/inspect_tasks.py@auditor_task_v4_swebench_verified \
    --model anthropic/claude-opus-4-6 --limit 1
```

## Integration with Experiment A

```python
from experiment_new_tasks.feature_source import CSVFeatureSource

llm_source = CSVFeatureSource(
    csv_path="chris_output/experiment_a/llm_judge_features/llm_judge_features.csv",
    task_id_column="_instance_id",
    feature_columns=["solution_hint", "problem_clarity", "solution_complexity"],
)
```

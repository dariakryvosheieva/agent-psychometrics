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
| `sandbox_utils.py` | Docker sandbox configuration and cleanup helpers |

## Environment Features (Auditor Agent)

Environment-level features (8 features) are extracted via an **agentic pipeline**: an LLM explores each task's Docker container via bash and Python tools, then rates the environment on 8 difficulty axes. This uses the [Inspect AI](https://inspect.ai) framework.

### How It Works

1. Each task is spun up in a Docker container with the repo at `/testbed`
2. The auditor agent (GPT-5.4) explores the codebase using bash and Python (8–15 tool calls)
3. After exploration, the agent submits a JSON report rating 8 features on a 1–5 scale
4. Results are parsed from Inspect logs into an incremental CSV

### 8 Environment Features

| Feature | Scale | What It Measures |
|---------|-------|-----------------|
| `fix_localization` | 1–5 | How spread out is the fix? (1=many modules, 5=single function) |
| `entry_point_clarity` | 1–5 | How easy to find where the problem manifests? |
| `change_blast_radius` | 1–5 | How many components are affected? |
| `environment_setup_complexity` | 1–5 | How complex is the runtime/tooling? |
| `implementation_language_complexity` | 1–5 | How complex is the tech stack? |
| `testing_infrastructure_quality` | 1–5 | How good is the testing/validation setup? |
| `dependency_complexity` | 1–5 | How complex are dependencies? |
| `codebase_scale` | 1–5 | How large/complex is the codebase? |

### Quick Start

```bash
# Test on a single task
inspect eval llm_judge_feature_extraction/auditor_agent/inspect_tasks.py@auditor_task_v4_swebench_verified \
    --model openai/gpt-5.4-2026-03-05 --limit 1

# Run a full dataset with batching, Docker cleanup, and S3 sync
python -m llm_judge_feature_extraction.auditor_agent.run_auditor \
    --dataset terminalbench \
    --batch_size 50 \
    --max_connections 25 \
    --model openai/gpt-5.4-2026-03-05 \
    --s3_bucket my-bucket

# Just aggregate existing logs to CSV (skip running)
python -m llm_judge_feature_extraction.auditor_agent.run_auditor \
    --dataset swebench_verified --aggregate_only
```

### `run_auditor` Options

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset` | `swebench_verified` | Dataset: `gso`, `swebench_pro`, `swebench_verified`, `terminalbench` |
| `--batch_size` | 10 | Tasks per batch (Docker cleanup runs between batches) |
| `--max_connections` | 10 | Parallel Docker containers per batch |
| `--model` | `openai/gpt-5.4-2026-03-05` | LLM model |
| `--log_dir` | auto | Output directory (default: `chris_output/auditor_features/{dataset}_v4`) |
| `--s3_bucket` | none | Sync incremental CSV to S3 after each batch |
| `--limit` | none | Limit to first N tasks (for testing) |
| `--sample_ids` | none | Comma-separated specific task IDs |
| `--aggregate_only` | false | Only aggregate existing logs, skip running |
| `--skip_cleanup` | false | Skip Docker cleanup between batches |

### Resumability

The auditor is fully resumable. After each batch, results are appended to `auditor_features_incremental.csv`. On restart, completed tasks are skipped automatically. This is critical for long runs on spot instances.

### AWS Deployment

For running all 4 datasets on EC2, see `aws_setup/`:

```bash
# Launch spot instance
bash aws_setup/launch_spot.sh

# On EC2: setup + run all datasets
bash aws_setup/setup_instance.sh
bash aws_setup/run_all_auditor.sh  # runs all 4 datasets, syncs to S3, auto-terminates
```

### Auditor Agent Files

| File | Purpose |
|------|---------|
| `auditor_agent/inspect_tasks.py` | Inspect task definitions for all 4 datasets |
| `auditor_agent/run_auditor.py` | Batched orchestration with pre-pull, cleanup, S3 sync |
| `auditor_agent/parse_outputs.py` | Parse Inspect logs → CSV with 8 features + reasoning |
| `auditor_agent/verify_commands.py` | Validation tool for agent outputs |

## Integration with Experiment A

```python
from experiment_new_tasks.feature_source import CSVFeatureSource

llm_source = CSVFeatureSource(
    csv_path="chris_output/experiment_a/llm_judge_features/llm_judge_features.csv",
    task_id_column="_instance_id",
    feature_columns=["solution_hint", "problem_clarity", "solution_complexity"],
)
```

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

### Info Level of Agent Input

The `input` field in Inspect determines what text the agent sees as its initial message. The Docker container (`/testbed`) contains only the repo source code at the base commit — no benchmark scripts, test patches, or gold patches are present (verified by inspecting the Docker images).

| Dataset | `input=` field | Info Level of Input |
|---------|---------------|---------------------|
| SWE-bench Verified | `problem_statement` | PROBLEM |
| SWE-bench Pro | `problem_statement` | PROBLEM |
| TerminalBench | `instruction.md` text | PROBLEM |
| GSO | `prob_script` (benchmark script) | TEST |

**GSO note**: The current `inspect_tasks.py` passes `input="prob_script"` (line 364), giving the auditor the full performance benchmark script (TEST-level information). This is intentional for the default pipeline (Experiment New Tasks), which overrides all features to solution level anyway, and GSO tasks carry almost no information without the benchmark script (the `api` field is just a function name).

**For the information ablation study**: To run a clean GSO ablation where the ENVIRONMENT level sits below TEST, new auditor features must be extracted with only PROBLEM-level input (repo + API name). The fix is to change `input="prob_script"` to `input="api"` (or post-process samples to construct a minimal prompt) and move `prob_script` to `metadata`. The existing features should be kept for Experiment New Tasks; the new clean features would be used only for the ablation.

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

For running all 4 datasets on EC2, see `llm_judge_feature_extraction/auditor_agent/`:

```bash
# Launch spot instance
bash llm_judge_feature_extraction/auditor_agent/launch_spot.sh

# On EC2: setup + run all datasets
bash llm_judge_feature_extraction/auditor_agent/setup_instance.sh
bash llm_judge_feature_extraction/auditor_agent/run_all_auditor.sh  # runs all 4 datasets, syncs to S3, auto-terminates
```

### Auditor Agent Files

| File | Purpose |
|------|---------|
| `auditor_agent/inspect_tasks.py` | Inspect task definitions for all 4 datasets |
| `auditor_agent/run_auditor.py` | Batched orchestration with pre-pull, cleanup, S3 sync |
| `auditor_agent/parse_outputs.py` | Parse Inspect logs → CSV with 8 features + reasoning |
| `auditor_agent/verify_commands.py` | Validation tool for agent outputs |

## Extracted Feature Data (`chris_output/llm_judge_features/`)

### Active Default: `v7_unified_15/`

15 unified features used across all 4 datasets, selected from 28 (20 judge + 8 auditor).

**Base model**: Claude Opus 4.6 (`claude-opus-4-6`)
**Info Level**: Solution override (all features see full task info including gold patch)
**Features**: 15 (same set for all datasets: 10 Problem + 1 Test + 1 Solution + 3 Auditor)

**Problem (10)**: atypicality, codebase_scope, debugging_complexity, domain_knowledge_required, error_specificity, logical_reasoning_required, side_effect_risk, similar_issue_likelihood, solution_hint, verification_difficulty
**Test (1)**: test_edge_case_coverage
**Solution (1)**: solution_complexity
**Auditor (3)**: codebase_scale, fix_localization, implementation_language_complexity

| Dataset | Path | Tasks |
|---------|------|-------|
| SWE-bench Verified | `v7_unified_15/swebench_verified/llm_judge_features.csv` | 500 |
| GSO | `v7_unified_15/gso/llm_judge_features.csv` | 102 |
| SWE-bench Pro | `v7_unified_15/swebench_pro/llm_judge_features.csv` | 730 |
| TerminalBench | `v7_unified_15/terminalbench/llm_judge_features.csv` | 89 |

### Full Feature Source: `v7_opus_solution/`

All 20 base features before auditor augmentation and top-15 selection.

| Dataset | Path | Tasks |
|---------|------|-------|
| SWE-bench Verified | `v7_opus_solution/swebench_verified/llm_judge_features.csv` | 500 |
| GSO | `v7_opus_solution/gso/llm_judge_features.csv` | 102 |
| SWE-bench Pro | `v7_opus_solution/swebench_pro/llm_judge_features.csv` | 731 |
| TerminalBench | `v7_opus_solution/terminalbench/llm_judge_features.csv` | 89 |

#### Feature List (20 features)

| Feature | Scale | Info Level (natural) | Source |
|---------|-------|---------------------|--------|
| solution_hint | 0-3 | Problem | Problem Statement |
| problem_clarity | 1-5 | Problem | Problem Statement |
| domain_knowledge_required | 1-5 | Problem | Problem Statement |
| logical_reasoning_required | 1-5 | Problem | Problem Statement |
| atypicality | 1-5 | Problem | Problem Statement |
| verification_difficulty | 1-5 | Problem | Problem Statement |
| standard_pattern_available | 0-1 | Problem | Problem Statement |
| error_specificity | 1-5 | Problem | Problem Statement |
| reproduction_clarity | 1-5 | Problem | Problem Statement |
| expected_behavior_clarity | 1-5 | Problem | Problem Statement |
| debugging_complexity | 1-5 | Problem | Problem Statement |
| codebase_scope | 1-5 | Problem | Problem Statement |
| information_completeness | 1-5 | Problem | Problem Statement |
| similar_issue_likelihood | 1-5 | Problem | Problem Statement |
| side_effect_risk | 1-5 | Problem | Problem Statement |
| test_comprehensiveness | 1-5 | Test | Test Patch |
| test_assertion_complexity | 1-5 | Test | Test Patch |
| test_edge_case_coverage | 1-5 | Test | Test Patch |
| solution_complexity | 1-5 | Solution | Solution Patch |
| integration_complexity | 1-5 | Solution | Solution Patch |

Note: "Info Level (natural)" shows what each feature would see without override. With solution override, all features see the full task including gold patch.

### Auditor-Combined Features

20 v7 Opus 4.6 judge features + 8 GPT 5.4 auditor features = 28 features per task.

| Directory | Dataset | Tasks |
|-----------|---------|-------|
| `swebench_verified_v7_plus_auditor/` | SWE-bench Verified | 500 |
| `gso_v7_plus_auditor/` | GSO | 102 |
| `swebench_pro_v7_plus_auditor/` | SWE-bench Pro | 730 |
| `terminalbench_v7_plus_auditor/` | TerminalBench | 89 |

### Auditor Feature Directories (`chris_output/auditor_features/`)

| Directory | Dataset | Model | Tools | Tasks | Notes |
|-----------|---------|-------|-------|-------|-------|
| `swebench_verified_v4_gpt54/` | SWE-bench Verified | GPT 5.4 | bash + python | 500 | **Active** |
| `gso_v4_gpt54/` | GSO | GPT 5.4 | bash + python | 102 | **Active** |
| `swebench_pro_v4_gpt54/` | SWE-bench Pro | GPT 5.4 | bash + python | 730 | **Active** |
| `terminalbench_v4_gpt54/` | TerminalBench | GPT 5.4 | bash + python | 89 | **Active** |
| `gso_v4_opus_obsolete/` | GSO | Opus 4.6 | bash only | 102 (78 valid, 24 parse errors) | Obsolete — superseded by gso_v4_gpt54 |

### Feature Variant History

Provider × info-level comparison (3×2 design):

| Directory | Provider | Model | Info Level | Datasets | Features | Notes |
|-----------|----------|-------|-----------|----------|----------|-------|
| `v7_opus_solution/` | Anthropic | Opus 4.6 | Solution override | All 4 | 20 | **Current default** |
| `v8_opus_natural/` | Anthropic | Opus 4.6 | Natural | All 4 | 20 | |
| `v5_sonnet_solution/` | Anthropic | Sonnet 4.6 | Solution override | All 4 | 20 | |
| `v6_anthropic_natural/` | Anthropic | Sonnet 4.6 | Natural | All 4 | 20 | |
| `v3_solution_level/` | OpenAI | GPT 5.4 | Solution override | All 4 | 20 | |
| `v2_full_20features/` | OpenAI | GPT 5.4 | Natural | All 4 | 20 | |

Note: v5 and v6 were labeled "anthropic" but actually used Sonnet 4.6, not Opus.

### Archived

| Directory | Contents |
|-----------|----------|
| `experiment_a_old_defaults/` | Previous curated defaults (9-15 features per dataset) |
| `ablation_studies/` | SWE-bench Verified information-level ablation experiments |
| `unified_features/` | Standardized cross-dataset feature sets (staged for deletion) |

## Information Level Ablation CSVs (v8)

Top-15 feature CSVs for the information level ablation, generated by
`python -m experiment_new_tasks.run_information_ablation`. Uses v8 features
(natural info levels, no leakage). For each dataset, Ridge regression selects
the top 15 features by coefficient magnitude at each cumulative info level:

| Level | File | Features Available | Count |
|-------|------|--------------------|-------|
| Problem | `ablation/{dataset}/1_problem_15.csv` | 15 PROBLEM | 15 |
| + Auditor | `ablation/{dataset}/2_problem_auditor_15.csv` | PROBLEM + ENVIRONMENT | 23 |
| + Test | `ablation/{dataset}/3_problem_auditor_test_15.csv` | PROBLEM + ENVIRONMENT + TEST | 26 |
| + Solution (Full) | `ablation/{dataset}/4_full_15.csv` | All 28 features | 28 |

Each CSV contains `instance_id` + 15 selected feature columns. Source data:
`chris_output/llm_judge_features/{dataset}_v8_plus_auditor/llm_judge_features.csv`

Regenerate with:
```bash
python -m experiment_new_tasks.run_information_ablation --rebuild_csvs
```

## Integration with Experiment New Tasks

```python
from experiment_new_tasks.feature_source import CSVFeatureSource

llm_source = CSVFeatureSource(
    csv_path="chris_output/experiment_a/llm_judge_features/llm_judge_features.csv",
    task_id_column="_instance_id",
    feature_columns=["solution_hint", "problem_clarity", "solution_complexity"],
)
```

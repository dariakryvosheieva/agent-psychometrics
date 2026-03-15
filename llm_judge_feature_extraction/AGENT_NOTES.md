# LLM Judge Feature Extraction — Internal Notes

Internal reference for agent workflows. Not intended for the public README.

## Feature Variant Naming

Features in `output/` follow a version naming convention:

| Directory | Provider | Model | Info Level | Features |
|-----------|----------|-------|-----------|----------|
| `v7_opus_solution/` | Anthropic | Opus 4.6 | Solution override | 20 | **Current default source** |
| `v8_opus_natural/` | Anthropic | Opus 4.6 | Natural | 20 |
| `v5_sonnet_solution/` | Anthropic | Sonnet 4.6 | Solution override | 20 |
| `v6_anthropic_natural/` | Anthropic | Sonnet 4.6 | Natural | 20 |
| `v3_solution_level/` | OpenAI | GPT 5.4 | Solution override | 20 |
| `v2_full_20features/` | OpenAI | GPT 5.4 | Natural | 20 |

Note: v5 and v6 were labeled "anthropic" but actually used Sonnet 4.6, not Opus.

### Default Features (v7_unified_15)

15 features selected from 28 (20 judge + 8 auditor) via greedy forward selection.
Backbone: Claude Opus 4.6 with solution override (all features see full task info).

**Problem (10)**: atypicality, codebase_scope, debugging_complexity, domain_knowledge_required, error_specificity, logical_reasoning_required, side_effect_risk, similar_issue_likelihood, solution_hint, verification_difficulty
**Test (1)**: test_edge_case_coverage
**Solution (1)**: solution_complexity
**Auditor (3)**: codebase_scale, fix_localization, implementation_language_complexity

Committed as `llm_judge_features/defaults/`.

### Full 20-Feature Set (v7_opus_solution)

All 20 base judge features before auditor augmentation and top-15 selection.

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

"Info Level (natural)" shows what each feature would see without override.

### Auditor-Combined Features

20 judge features + 8 GPT 5.4 auditor features = 28 features per task.
In `output/`: `{dataset}_v7_plus_auditor/`

## Auditor Feature Directories (`output/auditor_features/`)

| Directory | Dataset | Model | Tools | Tasks | Notes |
|-----------|---------|-------|-------|-------|-------|
| `swebench_verified_v4_gpt54/` | SWE-bench Verified | GPT 5.4 | bash + python | 500 | Active |
| `gso_v4_gpt54/` | GSO | GPT 5.4 | bash + python | 102 | Active |
| `swebench_pro_v4_gpt54/` | SWE-bench Pro | GPT 5.4 | bash + python | 730 | Active |
| `terminalbench_v4_gpt54/` | TerminalBench | GPT 5.4 | bash + python | 89 | Active |
| `gso_v4_opus_obsolete/` | GSO | Opus 4.6 | bash only | 102 (78 valid) | Obsolete |

## Info Level of Auditor Agent Input

The `input` field in Inspect determines what text the agent sees as its initial message.
The Docker container (`/testbed`) contains only the repo source code at the base commit —
no benchmark scripts, test patches, or gold patches are present.

| Dataset | `input=` field | Info Level of Input |
|---------|---------------|---------------------|
| SWE-bench Verified | `problem_statement` | PROBLEM |
| SWE-bench Pro | `problem_statement` | PROBLEM |
| TerminalBench | `instruction.md` text | PROBLEM |
| GSO | `prob_script` (benchmark script) | TEST |

**GSO note**: `inspect_tasks.py` passes `input="prob_script"` (line 364), giving the auditor
the full performance benchmark script (TEST-level information). This is intentional for the
default pipeline (Experiment New Tasks), which overrides all features to solution level anyway,
and GSO tasks carry almost no information without the benchmark script (the `api` field is just
a function name).

**For the information ablation study**: To run a clean GSO ablation where the ENVIRONMENT level
sits below TEST, new auditor features must be extracted with only PROBLEM-level input (repo +
API name). The fix is to change `input="prob_script"` to `input="api"` and move `prob_script`
to `metadata`. The existing features should be kept for Experiment New Tasks; the new clean
features would be used only for the ablation.

## AWS Deployment

For running all 4 datasets on EC2:

```bash
# Launch spot instance
bash llm_judge_feature_extraction/auditor_agent/launch_spot.sh

# On EC2: setup + run all datasets
bash llm_judge_feature_extraction/auditor_agent/setup_instance.sh
bash llm_judge_feature_extraction/auditor_agent/run_all_auditor.sh  # runs all 4, syncs to S3, auto-terminates
```

## Information Level Ablation — Detail

Top-15 feature CSVs for the information level ablation. Each ablation level
uses features extracted with **info level override** — the LLM sees the full
context available at that level when extracting all non-ENVIRONMENT features.
Ridge regression selects the top 15 features by coefficient magnitude.

### Per-Level Source CSVs

Each ablation level has its own source CSV with features extracted at the
appropriate info level override:

| Level | Source File | Features | Total Columns |
|-------|------------|----------|---------------|
| Problem | `per_level_source/{dataset}/problem.csv` | 15 PROBLEM + 8 ENV | 24 |
| + Auditor | `per_level_source/{dataset}/environment.csv` | 15 PROBLEM + 8 ENV | 24 |
| + Test | `per_level_source/{dataset}/test.csv` | 18 PROBLEM+TEST + 8 ENV | 27 |
| + Solution (Full) | `per_level_source/{dataset}/solution.csv` | 20 non-ENV + 8 ENV | 29 |

At the Problem and +Auditor levels, PROBLEM features are at their natural level.
At +Test and +Solution, PROBLEM features are **re-extracted** with the LLM seeing
test/solution context. ENVIRONMENT features always come from the natural source
(auditor agent pipeline, not re-extractable via LLM).

### Final Ablation CSVs

| Level | File | Features Selected |
|-------|------|------------------|
| Problem | `ablation/{dataset}/1_problem_15.csv` | Top 15 from 24 |
| + Auditor | `ablation/{dataset}/2_problem_auditor_15.csv` | Top 15 from 24 |
| + Test | `ablation/{dataset}/3_problem_auditor_test_15.csv` | Top 15 from 27 |
| + Solution (Full) | `ablation/{dataset}/4_full_15.csv` | Top 15 from 29 |

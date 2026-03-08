# LLM Judge Features

Semantic task features extracted via LLM structured output for difficulty prediction.

## Active Default: `v7_opus_solution_top15/`

Top 15 features per dataset selected by Ridge coefficient magnitude from v7 Opus 4.6 solution-level features + auditor features.

**Base model**: Claude Opus 4.6 (`claude-opus-4-6`)
**Info Level**: Solution override (all features see full task info including gold patch)
**Features**: 15 per dataset (selected from 20-28 depending on auditor availability)
**Selection**: Ridge regression (alpha=1.0) on standardized features vs IRT difficulty

| Dataset | Path | Features | Auditor Features Included |
|---------|------|----------|--------------------------|
| SWE-bench Verified | `v7_opus_solution_top15/swebench_verified/llm_judge_features.csv` | 15 (from 23) | 2/3 (entry_point_clarity, fix_localization) |
| GSO | `v7_opus_solution_top15/gso/llm_judge_features.csv` | 15 (from 28) | 1/8 (codebase_scale) |
| SWE-bench Pro | `v7_opus_solution_top15/swebench_pro/llm_judge_features.csv` | 15 (from 20) | 0 (none available) |
| TerminalBench | `v7_opus_solution_top15/terminalbench/llm_judge_features.csv` | 15 (from 28) | 4/8 (dependency_complexity, fix_localization, testing_infrastructure_quality, codebase_scale) |

Note: The top 15 are selected per-dataset. A future goal is a unified feature set that performs comparably across all datasets.

## Full Feature Source: `v7_opus_solution/`

All 20 base features before auditor augmentation and top-15 selection.

| Dataset | Path | Tasks |
|---------|------|-------|
| SWE-bench Verified | `v7_opus_solution/swebench_verified/llm_judge_features.csv` | 500 |
| GSO | `v7_opus_solution/gso/llm_judge_features.csv` | 102 |
| SWE-bench Pro | `v7_opus_solution/swebench_pro/llm_judge_features.csv` | 731 |
| TerminalBench | `v7_opus_solution/terminalbench/llm_judge_features.csv` | 89 |

### Feature List (20 features)

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

## Auditor-Combined Features

Merge of v7 judge features + auditor agent environment features.

| Directory | Dataset | Judge Model | Auditor Model | Auditor Tools | Features | Tasks |
|-----------|---------|-------------|---------------|---------------|----------|-------|
| `gso_v7_plus_auditor/` | GSO | Opus 4.6 | Opus 4.6 | bash only | 28 (20+8) | 102 |
| `terminalbench_v7_plus_auditor/` | TerminalBench | Opus 4.6 | GPT 5.4 | bash + python | 28 (20+8) | 89 |

## Auditor Agent Features (`chris_output/auditor_features/`)

Environment-level features extracted by an agent that explores the task's Docker container.

| Directory | Dataset | Model | Tools | Tasks | Notes |
|-----------|---------|-------|-------|-------|-------|
| `gso_v4_gpt54/` | GSO | GPT 5.4 | bash + python | 102 | **Active** |
| `terminalbench_v4_gpt54/` | TerminalBench | GPT 5.4 | bash + python | 89 | **Active** |
| `gso_v4/` | GSO | Opus 4.6 | bash only | 102 (78 valid, 24 parse errors) | Obsolete — 24 parse errors, superseded by gso_v4_gpt54 |

### Auditor Feature List (8 features)

| Feature | Scale | Description |
|---------|-------|-------------|
| fix_localization | 1-5 | How concentrated is the solution? (5=single function) |
| entry_point_clarity | 1-5 | How obvious is where the problem manifests? |
| change_blast_radius | 1-5 | How many components affected? (5=entire codebase) |
| environment_setup_complexity | 1-5 | How complex is the runtime? |
| implementation_language_complexity | 1-5 | How complex is the tech stack? |
| testing_infrastructure_quality | 1-5 | How good is the test setup? |
| dependency_complexity | 1-5 | How complex are dependencies? |
| codebase_scale | 1-5 | How large is the codebase? |

## Feature Variant History

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

## Archived

| Directory | Contents |
|-----------|----------|
| `experiment_a_old_defaults/` | Previous curated defaults (9-15 features per dataset) |
| `ablation_studies/` | SWE-bench Verified information-level ablation experiments |
| `unified_features/` | Standardized cross-dataset feature sets (staged for deletion) |

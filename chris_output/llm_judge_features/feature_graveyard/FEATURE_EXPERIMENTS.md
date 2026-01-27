# Feature Experiments Archive

This file tracks LLM judge features that were tested but didn't provide incremental value beyond unified features.

## Experiment: V3 Cross-Cutting Features (2026-01-27)

**Goal:** Test cross-cutting features focused on multi-module coordination.

**Sample:** 100 SWE-bench tasks

### Features Tested

| Feature | Range | Description | Raw r | p-value | Residual r* | Incremental R** |
|---------|-------|-------------|-------|---------|-------------|-----------------|
| coordination_complexity | 1-5 | Inter-component coordination | 0.455*** | 0.0000 | -0.013 | +0.0033 |
| implicit_requirements | 1-5 | Hidden/undocumented requirements | 0.435*** | 0.0000 | 0.012 | +0.0020 |
| cross_cutting_fix | 1-5 | Coordinated multi-module edits | 0.381*** | 0.0001 | 0.102 | +0.0000 |
| change_scope | 1-3 | Local/module/system-wide | 0.306** | 0.0020 | 0.076 | +0.0019 |
| api_boundary_crossing | 0/1 | Crosses public APIs | -0.058 | 0.5697 | n/a | n/a |
| **domain_category** | 0-9 | Primary domain category | **0.200*** | 0.0464 | **0.146** | **+0.0107** |

*Residual r: Correlation after controlling for existing `integration_complexity`
**Incremental R: Improvement when added to unified features

### Key Finding

Most V3 features are **redundant with existing `integration_complexity`** (unified):
- `coordination_complexity` vs `integration_complexity`: r=0.791
- `implicit_requirements` vs `integration_complexity`: r=0.723
- `cross_cutting_fix` vs `integration_complexity`: r=0.508
- `change_scope` vs `integration_complexity`: r=0.415

Within-V3 correlations were also very high:
- `cross_cutting_fix` vs `change_scope`: r=0.761
- `coordination_complexity` vs `cross_cutting_fix`: r=0.704
- `coordination_complexity` vs `change_scope`: r=0.665

### Decision

- **KEEP:** `domain_category` - provides unique signal (+0.0107 incremental R)
- **DISCARD:** All other V3 features (redundant with integration_complexity)

### Lesson Learned

Features measuring "multi-module coordination" largely duplicate `integration_complexity`. Future features should focus on:
- Task type (bug fix vs feature request)
- Problem specification quality (different from problem_clarity)
- Solution approach (new code vs modification)
- Error patterns or reproduction clarity
- External dependencies or API knowledge requirements

---

## Experiment: V4 Orthogonal Features (2026-01-27)

**Goal:** Test features designed to be orthogonal to V3 and unified features.

**Sample:** Same 100 SWE-bench tasks

### Features Tested

| Feature | Range | Description | Raw r | p-value | Incremental R |
|---------|-------|-------------|-------|---------|---------------|
| **solution_discovery_needed** | 1-5 | Investigation needed to find fix | **0.365***| 0.0002 | **+0.0047** |
| **fix_pattern_type** | 0-4 | Type of fix needed | **0.272**| 0.0063 | **+0.0013** |
| requires_framework_internals | 1-5 | Depth into framework internals | 0.136 | 0.1779 | n/a |
| test_coverage_gap | 0/1 | Missing test coverage | 0.123 | 0.2233 | n/a |
| involves_timing_or_ordering | 0/1 | Timing/ordering issues | 0.088 | 0.3824 | n/a |
| is_edge_case | 0/1 | Edge case vs main path | -0.055 | 0.5873 | n/a |

### Key Finding

`solution_discovery_needed` is highly correlated with existing unified features:
- atypicality: r=0.767
- logical_reasoning_required: r=0.754
- integration_complexity: r=0.752

Despite high correlations, both significant V4 features add small incremental value.

### Decision

- **KEEP:** `solution_discovery_needed` and `fix_pattern_type` - small but additive value
- **DISCARD:** requires_framework_internals, test_coverage_gap, involves_timing_or_ordering, is_edge_case

---

## Combined Results Summary (So Far)

| Feature | Source | Raw r | Incremental R | Keep? |
|---------|--------|-------|---------------|-------|
| domain_category | V3 | 0.200* | +0.0107 | YES |
| solution_discovery_needed | V4 | 0.365*** | +0.0047 | YES |
| fix_pattern_type | V4 | 0.272** | +0.0013 | YES |

**Combined improvement:** +0.0169 R (from 0.7027 to 0.7196)

**Need 2-3 more features** to reach target of 5-6 total.

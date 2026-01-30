# Data Pipeline

This document describes the data flow from raw agent results to trained IRT models.

## Pipeline Overview

```
experiments/evaluation/verified/<agent>/results/results.json
         ↓
    swebench_irt/prep_swebench.py (--complete_matrix --cutoff_date 20251115)
         ↓
clean_data/swebench_verified/swebench_verified_20251115_full.jsonl
         ↓
    swebench_irt/train.py (--dims 1 --model 1pl/2pl)
         ↓
clean_data/swebench_verified_20251115_full/{1d,1d_1pl}/
    ├── items.csv   (a, b for 2PL; b only for 1PL)
    └── abilities.csv (theta per agent)
         ↓
    swebench_irt/compare_dims.py (AIC/BIC comparison)
```

## Current Dataset

- **130 agents** (cutoff: 2025-11-15)
- **500 tasks** (SWE-bench Verified)
- **65,000 observations** (complete matrix, missing → 0)
- **76 agents** with unified trajectories (for Experiment B)

## Step 1: Prepare Response Matrix

```bash
python swebench_irt/prep_swebench.py \
    --complete_matrix \
    --cutoff_date 20251115 \
    --output_path clean_data/swebench_verified/swebench_verified_20251115_full.jsonl
```

**Input:** `experiments/evaluation/verified/<agent>/results/results.json`

**Output format (JSONL):**
```json
{"subject_id": "agent_name", "responses": {"task_1": 1, "task_2": 0, ...}}
```

## Step 2: Train IRT Models

See [IRT_MODELS.md](IRT_MODELS.md) for training commands.

## Step 3: Use Trained Parameters

```python
import pandas as pd

# Load IRT parameters
items = pd.read_csv("clean_data/swebench_verified_20251115_full/1d_1pl/items.csv", index_col=0)
abilities = pd.read_csv("clean_data/swebench_verified_20251115_full/1d_1pl/abilities.csv", index_col=0)

# items.index: task IDs like "django__django-12345"
# items["b"]: difficulty (-2 to +5 range typically)
# abilities.index: agent names
# abilities["theta"]: agent ability
```

## Trajectory Data

Unified trajectories are stored in `trajectory_data/unified_trajs/`:

```
trajectory_data/unified_trajs/
├── 20240620_sweagent_claude3.5sonnet/
│   ├── astropy__astropy-12907.json
│   ├── django__django-12345.json
│   └── ...
├── 20240728_sweagent_gpt4o/
└── ...
```

Each JSON file contains the agent's trajectory for one task in a unified format.

## SWE-bench Bash-Only Dataset

SWE-bench Bash-Only is a variant where agents are restricted to bash-only tool use via the [mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent) scaffold. It uses the **same 500 tasks** as SWE-bench Verified, enabling apples-to-apples comparison.

```
experiments/evaluation/bash-only/<agent>/per_instance_details.json
         ↓
    swebench_irt/prep_swebench_bash_only.py
         ↓
clean_data/swebench_bash_only/swebench_bash_only.jsonl
         ↓
    swebench_irt/train.py (--dims 1 --model 1pl)
         ↓
clean_data/swebench_bash_only/1d_1pl/
    ├── items.csv   (task difficulties)
    └── abilities.csv (agent abilities)
```

**Dataset stats:**
- **26 agents** (after exclusions)
- **500 tasks** (identical to SWE-bench Verified)
- **13,000 observations** (complete matrix)

**Excluded agents:**

The prep script excludes agents with known data quality issues:

| Agent | Reason |
|-------|--------|
| v0.0.0 scaffold agents (5) | Early scaffold version with bugs; results not comparable to v1.x+ |
| gemini-2.0-flash, gemini-2.5-flash, gpt-4.1 | Missing `per_instance_details.json` files |

See `EXCLUDED_AGENTS` in `prep_swebench_bash_only.py` for the full list.

**Prep command:**
```bash
python swebench_irt/prep_swebench_bash_only.py
```

**Input format** (`per_instance_details.json`):
```json
{
  "django__django-12345": {"cost": 0.15, "api_calls": 28, "resolved": true},
  "astropy__astropy-12907": {"cost": 0.35, "api_calls": 41, "resolved": false}
}
```

## Key Files

| File | Purpose |
|------|---------|
| `swebench_irt/prep_swebench.py` | Build JSONL response matrix (Verified) |
| `swebench_irt/prep_swebench_bash_only.py` | Build JSONL response matrix (Bash-Only) |
| `trajectory_upload/trajectory_converter.py` | Convert raw trajectories to unified format |
| `experiments/` | Raw agent results (gitignored, has own .git) |

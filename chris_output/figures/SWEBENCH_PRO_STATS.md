# SWE-bench Pro Statistics

**Analysis Date:** 2026-01-09

## Summary

SWE-bench Pro is a benchmark dataset containing **9,729 agent × problem trajectories** across:
- **14 agents** (LLMs with different configurations)
- **730 problems** (software engineering tasks)
- **27.1% overall resolution rate** (2,640 resolved trajectories)

## Data Source

- **CSV File:** `data/swe-bench-pro.csv` (1.8 MB, 9,729 rows)
- **Website:** https://docent.transluce.org/dashboard/032fb63d-4992-4bfc-911d-3b7dafcb931f
- **Date Range:** Oct 13-22, 2025

## Agent Statistics

| Agent | Tasks | Resolved | Rate |
|-------|------:|------:|-----:|
| Claude 4.5 Sonnet - 10132025 | 730 | 319 | **43.7%** |
| Claude 4 Sonnet - 10132025 | 562 | 240 | 42.7% |
| Claude 4.5 Haiku -- 10222025 | 729 | 288 | 39.5% |
| GPT-5 - 10132025 | 729 | 265 | 36.4% |
| GLM-4.5 -- 10222025 | 729 | 259 | 35.5% |
| Kimi - paper | 729 | 202 | 27.7% |
| GPT-5 Codex -- debug-oct22 | 708 | 191 | 27.0% |
| GPT-5 High - paper | 730 | 189 | 25.9% |
| Claude Opus 4.1 - paper | 663 | 166 | 25.0% |
| Claude Sonnet 4 - paper | 626 | 126 | 20.1% |
| Gemini 2.5 Pro Preview -- debug-oct22 | 728 | 142 | 19.5% |
| GPT OSS - paper | 728 | 118 | 16.2% |
| Gemini 2.5 Pro Preview - paper | 719 | 99 | 13.8% |
| GPT-4o - paper | 619 | 36 | 5.8% |

**Best performer:** Claude 4.5 Sonnet (43.7% resolution rate)

## Problem Statistics

- **Total problems:** 730
- **Problems per agent:** 562-730 (incomplete matrix)
- **Most complete agents:** Claude 4.5 Sonnet, GPT-5 High (730/730 problems)
- **Least complete agent:** Claude 4 Sonnet (562/730 problems, 77.0%)

## Trajectory Statistics

- **Total trajectories:** 9,729
- **Unique agent_run_ids:** 9,729 (no duplicates)
- **Resolved:** 2,640 (27.1%)
- **Failed:** 7,089 (72.9%)

### Turn Statistics (conversation length)

- **Mean:** 51.5 turns
- **Median:** 44 turns
- **Range:** 1-251 turns

## Matrix Completeness

**Status:** ✗ Incomplete matrix

Not all agents evaluated on all 730 problems. Breakdown:

| Completeness | # Agents | Problems |
|--------------|----------|----------|
| 100% (730/730) | 2 | Claude 4.5 Sonnet, GPT-5 High |
| 99.9% (729/730) | 4 | Kimi, GPT-5, GLM-4.5, Claude 4.5 Haiku |
| 99.7% (728/730) | 2 | GPT OSS, Gemini 2.5 debug |
| 98.5% (719/730) | 1 | Gemini 2.5 paper |
| 97.0% (708/730) | 1 | GPT-5 Codex |
| 90.8% (663/730) | 1 | Claude Opus 4.1 |
| 85.8% (626/730) | 1 | Claude Sonnet 4 |
| 84.8% (619/730) | 1 | GPT-4o |
| 77.0% (562/730) | 1 | Claude 4 Sonnet |

## Data Structure

### CSV Columns

| Column | Description | Example |
|--------|-------------|---------|
| `agent_run_id` | Unique trajectory ID | `eaa8e4b1-dda6-46ec-9787-ba7ccebfafa2` |
| `metadata.instance_id` | Problem identifier | `instance_flipt-io__flipt-cd18e54a0371...` |
| `metadata.model_name` | Agent/model name | `Kimi - paper` |
| `metadata.resolved` | Success flag | `true`/`false` |
| `metadata.turns` | Conversation length | `30` |
| `created_at` | Timestamp | `Oct 13, 2025, 10:41 PM UTC` |

### Trajectory URL Format

```
https://docent.transluce.org/dashboard/032fb63d-4992-4bfc-911d-3b7dafcb931f/agent_run/{agent_run_id}
```

Example:
```
https://docent.transluce.org/dashboard/032fb63d-4992-4bfc-911d-3b7dafcb931f/agent_run/eaa8e4b1-dda6-46ec-9787-ba7ccebfafa2
```

## Comparison to SWE-bench Verified

| Metric | SWE-bench Pro | SWE-bench Verified |
|--------|---------------|--------------------|
| Agents | 14 | 123 |
| Problems | 730 | 500 |
| Trajectories | 9,729 | 61,500 |
| Matrix | Incomplete (77-100%) | Complete (100%) |
| Avg Resolution Rate | 27.1% | ~27% (varies) |
| Date Range | Oct 2025 | 2020-2025 |

**Key differences:**
- SWE-bench Pro: Fewer agents, more problems, newer (Oct 2025 only)
- SWE-bench Verified: More agents, fewer problems, complete matrix, longer time span

## Notes

1. **No duplicate trajectories:** Each (agent, problem) pair appears at most once
2. **Incomplete matrix:** Some agents missing certain problems (likely due to timeouts, errors, or ongoing evaluation)
3. **Recent data:** All trajectories from Oct 13-22, 2025 (9-day window)
4. **Turn distribution:** Long tail with some trajectories exceeding 250 turns
5. **Agent naming:** Mix of model versions, dates, and variant tags ("paper", "debug-oct22")

## Next Steps for IRT Analysis

To fit IRT models to SWE-bench Pro:

1. **Handle missing data:**
   - Option A: Treat missing as failures (0)
   - Option B: Use subset of problems all agents attempted
   - Option C: Use IRT models that handle missing data

2. **Preprocessing:**
   ```bash
   # Create JSONL format (similar to prep_swebench.py)
   python swebench_irt/prep_swebench_pro.py \
       --input data/swe-bench-pro.csv \
       --output clean_data/swebench_pro/swebench_pro.jsonl
   ```

3. **Train IRT models:**
   ```bash
   python swebench_irt/train.py \
       --data_path clean_data/swebench_pro/swebench_pro.jsonl \
       --dims 1 2 3 \
       --output_dir clean_data/swebench_pro
   ```

4. **Compare with SWE-bench Verified:**
   - Do difficulty rankings correlate across datasets?
   - Do agent ability rankings correlate?
   - Is 1D still the best fit?

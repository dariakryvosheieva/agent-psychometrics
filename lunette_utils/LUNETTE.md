# Lunette Integration Guide

This document describes how to use [Lunette](https://docs.lunette.dev/) for analyzing SWE-bench agent trajectories.

## Overview

Lunette is a platform for evaluating AI agents with **environment access** - judges can examine the actual codebase, run commands, and test hypotheses. This is more powerful than pure text-based LLM judging.

## Setup

```bash
pip install lunette-sdk

# Configure API key
mkdir -p ~/.lunette
echo '{"api_key": "your-key-here"}' > ~/.lunette/config.json
```

## Key Concepts

### Runs and Trajectories

- **Run**: A collection of trajectories, typically from one agent on multiple tasks
- **Trajectory**: A single agent attempt at solving a task (messages, actions, results)
- Trajectories are uploaded in batches and assigned run_ids

### Analysis Plans

Lunette provides several analysis plan types:

- `GradingPlan`: Outputs `{name, score, explanation}` where score is 0-1
- `IssueDetectionPlan`: Identifies specific issues/bugs
- `BottleneckPlan`: Finds performance bottlenecks

### GradingPlan Output Format

```python
{
    "name": "difficulty-prediction",  # From your plan name
    "score": 0.65,  # 0.0-1.0 numeric score
    "explanation": "..."  # Detailed reasoning text
}
```

The `score` is the main numeric output. For multiple scoring dimensions, include them in the `explanation` using a structured format that you parse.

## Uploading Trajectories with Metadata

**IMPORTANT:** Always use `trajectory_upload/lunette_reupload_with_metadata.py` for uploading trajectories. This includes proper SWE-bench metadata (repo, patch, test_patch, etc.) which the judge uses.

```bash
# Upload all agents with proper metadata
python trajectory_upload/lunette_reupload_with_metadata.py

# Upload specific agents
python trajectory_upload/lunette_reupload_with_metadata.py --agents 20240620_sweagent_claude3.5sonnet

# Dry run
python trajectory_upload/lunette_reupload_with_metadata.py --dry_run
```

Tracking files are saved to `trajectory_data/unified_trajs/<agent>/_lunette_uploads.json`:

```json
{
  "agent": "20240620_sweagent_claude3.5sonnet",
  "run_ids": ["68aac133-...", "896c8926-..."],
  "trajectory_count": 500,
  "has_swebench_metadata": true,
  "trajectories": [
    {
      "task_id": "astropy__astropy-12907",
      "run_id": "68aac133-...",
      "trajectory_id": "0b8cddce-...",
      "resolved": false
    }
  ]
}
```

## Using the GradingPlan API

### Basic Usage

```python
from lunette import LunetteClient
from lunette.analysis import GradingPlan

async def grade_trajectory(run_id: str):
    async with LunetteClient() as client:
        results = await client.investigate(
            run_id=run_id,
            plan=GradingPlan(
                name="difficulty-prediction",
                prompt=YOUR_GRADING_PROMPT,
            ),
            limit=1,  # Grade 1 trajectory from the run
        )

        if results.results:
            data = results.results[0].data
            score = data["score"]  # 0-1 numeric
            explanation = data["explanation"]  # Detailed text
            return score, explanation
```

### Designing Prompts for Multiple Scoring Axes

Since `GradingPlan` outputs a single `score`, use the prompt to:

1. Ask for the primary metric as the score (e.g., difficulty 0-1)
2. Request structured output in the explanation for additional metrics

Example prompt structure:

```
You are analyzing a SWE-bench trajectory to predict task difficulty.

## YOUR TASK: Predict Difficulty (0.0 to 1.0)

Output a difficulty score between 0.0 and 1.0:
- 0.0 = Very easy (agent solved quickly)
- 0.5 = Medium difficulty
- 1.0 = Very hard (agent failed completely)

## In your explanation, include:

COMPETENCIES: backtracking=X/4, decomposition=X/4, observation=X/4, verification=X/4
FAILURES: [list any: localization_failure, strategy_defect, ...]
SIGNALS: [list any: agent_looping, agent_gave_up_early, ...]
REASONING: [1-2 sentences]
```

Then parse the explanation to extract the additional scores.

## API Quirks

### Trailing Slashes

- GET requests require trailing slash: `/runs/`
- DELETE requests must NOT have trailing slash: `/runs/{id}`

### Batch Uploads

Trajectories are uploaded in batches (default 25). Large trajectories may require smaller batch sizes to avoid 413 errors.

### Investigation Results

When calling `client.investigate()`, Lunette runs asynchronously. For blocking behavior:

```python
results = await client.investigate(
    run_id=run_id,
    plan=plan,
    limit=1,
)
# Results are available immediately
```

For non-blocking (poll manually):

```python
r = await http_client.post(
    "/investigations/run",
    json={"plan": plan.model_dump(mode="python"), "run_id": run_id, "blocking": False},
)
inv_id = r.json()["run_id"]

# Poll /investigations/{inv_id}/results until ready
```

## Related Files

| File | Purpose |
|------|---------|
| `trajectory_upload/lunette_reupload_with_metadata.py` | Upload trajectories with SWE-bench metadata |
| `llm_judge/lunette_batch_grading.py` | Batch grading of uploaded trajectories |
| `lunette_utils/lunette_analysis.py` | Lunette grading utilities |
| `experiment_b/compute_lunette_features.py` | Extract features for difficulty prediction |
| `experiment_b/lunette_features.py` | Feature definitions and parsing |

## Viewing Results in Dashboard

After running investigations, view results at:
```
https://lunette.dev/runs/<run_id>
```

The dashboard shows the trajectory, investigation results, and judge reasoning.

## Cost Estimates

- ~$0.05 per trajectory grading
- 500 tasks × 10 agents × $0.05 = ~$250 for full dataset
- Use `--limit` flags for testing before scaling up

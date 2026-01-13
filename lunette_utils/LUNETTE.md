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
- When uploading, trajectories are grouped into runs (batches of ~25)
- Both runs and trajectories get their own UUIDs from Lunette's API

#### Trajectory IDs

When you upload a `Run` with multiple trajectories, Lunette assigns each trajectory its own UUID. The `save_run()` response includes both:

```python
run_meta = await client.save_run(run)
run_id = run_meta["run_id"]           # UUID for the run
traj_ids = run_meta["trajectory_ids"]  # List of UUIDs, one per trajectory
```

You can retrieve individual trajectories directly via their ID:

```python
# GET /trajectories/{trajectory_id}
r = client.get(f"/trajectories/{traj_id}")
trajectory = r.json()
# Keys: id, run_id, task, model, messages, metadata, scores, ...
```

Or retrieve all trajectories in a run:

```python
# GET /runs/{run_id}
r = client.get(f"/runs/{run_id}")
run = r.json()
trajectories = run["trajectories"]  # List of trajectory objects
```

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

### Structured Output with Pydantic Models (Recommended)

Lunette supports **structured output** by passing a Pydantic model schema to the API. This forces the judge to return data matching your exact schema, eliminating the need to parse free-form text.

#### How It Works

1. Define a Pydantic model with your desired output fields
2. Create a custom `AnalysisPlan` subclass that includes the schema
3. Override `model_dump()` to include the JSON schema in the serialized plan
4. The Lunette backend forces the judge to use a tool matching your schema

#### Example: Custom Feature Extraction

```python
from typing import ClassVar, Literal
from pydantic import BaseModel, Field
from lunette.analysis import AnalysisPlanBase

# 1. Define your output schema as a Pydantic model
class TaskDifficultyFeatures(BaseModel):
    """Structured output for task difficulty analysis."""

    fix_complexity: int = Field(ge=1, le=5, description="Complexity of the fix (1-5)")
    domain_knowledge: int = Field(ge=1, le=5, description="Domain knowledge required (1-5)")
    problem_clarity: int = Field(ge=1, le=5, description="Problem statement clarity (1-5)")
    error_provided: Literal[0, 1] = Field(description="Error message provided? (0 or 1)")
    reasoning: str = Field(description="Brief explanation of the difficulty")


# 2. Create a custom plan that includes the schema
class FeatureExtractionPlan(AnalysisPlanBase):
    """Custom plan with structured output schema."""

    kind: Literal["grading"] = "grading"  # Use grading type
    result_schema: ClassVar[type[TaskDifficultyFeatures]] = TaskDifficultyFeatures

    def model_dump(self, **kwargs):
        """Include result_schema in serialized output."""
        d = super().model_dump(**kwargs)
        if self.result_schema is not None:
            d["result_schema"] = self.result_schema.model_json_schema()
        return d


# 3. Use the plan
async def extract_features(run_id: str):
    async with LunetteClient() as client:
        plan = FeatureExtractionPlan(
            name="task-difficulty-features",
            prompt="Analyze this task and extract difficulty features...",
        )

        results = await client.investigate(run_id=run_id, plan=plan, limit=1)

        # Result data directly matches your Pydantic schema!
        features = results.results[0].data
        print(features["fix_complexity"])  # int, 1-5
        print(features["reasoning"])       # str
```

#### Pre-built Plans

This codebase includes pre-built structured output plans:

| Plan | Module | Output Schema |
|------|--------|---------------|
| `FeatureExtractionPlan` | `experiment_a.lunette_structured_output` | `TaskDifficultyFeatures` - 25 features |
| `SemanticFeatureExtractionPlan` | `experiment_a.lunette_structured_output` | `SemanticOnlyFeatures` - 10 features |
| `TrajectoryGradingPlan` | `experiment_b.lunette_structured_output` | `TrajectoryFeatures` - competencies + signals |

#### Usage in Scripts

```bash
# Experiment A: Two-step process with proper sandbox access
# Step 1: Create sandbox runs via Inspect (provisions Docker containers)
python -m experiment_a.run_dummy_sandbox --limit 20 --batch_size 10

# Step 2: Grade the sandbox runs (judge has full shell access)
python -m experiment_a.grade_sandbox_runs --skip_existing

# Run evaluation
python -m experiment_a.train_evaluate \
    --lunette_features_path chris_output/experiment_a/sandbox_features/lunette_features.csv

# Experiment B: Grade trajectories (structured output is default)
python -m experiment_b.compute_lunette_features --limit 10
```

### Legacy: Parsing from Explanation Text

For backward compatibility or when structured output isn't available, you can parse features from the `explanation` field:

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

Then parse the explanation using regex patterns. See `experiment_b/compute_lunette_features.py:parse_features_from_explanation()` for parsing logic.

## Sandbox Creation

Lunette's power comes from **sandbox access** - the ability to run shell commands in an environment containing the repo checkout. However, sandbox creation is **not automatic** and depends on how you create/upload trajectories.

### When Sandboxes Are Created

| Method | Sandbox Created? | Notes |
|--------|------------------|-------|
| Run through Inspect with `--sandbox lunette` | ✅ Yes | Full environment with repo checkout |
| `client.save_run()` + `client.investigate()` | ❌ No | Just stores trajectory data, no environment |
| `client.create_sandbox(service)` | ✅ Yes | Manual sandbox, requires ComposeService config |

**Important:** The `investigate()` API does NOT have an `enable_sandbox` parameter. Sandboxes must be created through Inspect's sandbox system or manually via `create_sandbox()`.

### How It Works: LunetteSandboxEnvironment

When you use `--sandbox lunette` with Inspect, it uses the `LunetteSandboxEnvironment` class (defined in `lunette.inspect_sandbox`). The sandbox creation flow is:

1. **Inspect's task provides a Docker Compose config** - For SWE-bench, this defines a container with the repo checked out at the correct commit
2. **`sample_init()` creates a `ComposeProject`** - This parses the compose config
3. **`client.create_sandbox(service)` is called** - Where `service` is a `ComposeService` from the compose config
4. **The container is built/pulled** - Based on the compose specification
5. **Sandbox ID is stored in metadata** - For later grading access

Key code from `lunette.inspect_sandbox`:
```python
# In LunetteSandboxEnvironment.sample_init():
services = await compose_services(project)
name, service = services.popitem()
client = LunetteClient()
sandbox = await client.create_sandbox(service)  # service is ComposeService
```

This is why `create_sandbox({"image": "python:3.11-slim"})` works for generic containers but doesn't give you a SWE-bench repo checkout - you need the full compose config that SWE-bench provides.

### The Problem: Uploading Without Sandbox

If you upload trajectories directly via `save_run()` and then run `investigate()`, the grading judge will have **no sandbox access**:

```python
# ❌ WRONG - No sandbox created
trajectory = Trajectory(sample=task_id, messages=[...], scores={...})
run = Run(task="swebench-verified", model="my_agent", trajectories=[trajectory])
run_meta = await client.save_run(run)  # Just stores data
results = await client.investigate(run_id=run_meta["run_id"], plan=plan)
# Judge cannot run shell commands - sandbox is empty!
```

When this happens, the grading judge will report errors like:
- "Found 0 Python files"
- "Sandbox appears to be empty"
- Features will be estimated/guessed rather than verified

### Solution 1: Run Through Inspect (Recommended for SWE-bench)

For SWE-bench tasks, run agents through Inspect with the Lunette sandbox:

```bash
# Run with Lunette sandbox - creates proper environment
lunette eval swebench --solver my_solver.py:solver --model gpt-4 --sandbox lunette

# Or with inspect directly
inspect eval inspect_evals/swe_bench_verified_mini \
    --solver my_solver.py:solver \
    --model gpt-4 \
    --sandbox lunette \
    -T sandbox_type=lunette
```

This:
1. Provisions a sandbox with the repo checkout at the correct commit
2. Runs your agent in that sandbox
3. Uploads the trajectory with proper sandbox state
4. Grading judges can then access the same environment

### Solution 2: Create Sandbox Manually

For custom environments (not SWE-bench), create a sandbox directly:

```python
async with LunetteClient() as client:
    # Create sandbox with specific image
    sandbox = await client.create_sandbox({"image": "python:3.11-slim"})

    # Execute commands
    result = await sandbox.aexec("pip install numpy && python -c 'import numpy; print(numpy.__version__)'")
    print(result.stdout)

    # Upload/download files
    await sandbox.aupload(local_path="/path/to/file.py", remote_path="/app/file.py")
    await sandbox.adownload(remote_path="/app/output.txt", local_path="/tmp/output.txt")

    # Cleanup
    await sandbox.destroy()
```

### Sandbox Methods

The `Sandbox` class provides:

| Method | Description |
|--------|-------------|
| `aexec(command)` | Execute shell command, returns `{stdout, stderr, exit_code}` |
| `aupload(local_path, remote_path)` | Upload file to sandbox |
| `adownload(remote_path, local_path)` | Download file from sandbox |
| `destroy()` | Cleanup sandbox resources |

### Verifying Sandbox Access

To check if a grading judge had sandbox access, look at the investigation trajectory:

```python
# Fetch investigation trajectory
r = await client.get(f"/investigations/{inv_id}/trajectory")
traj = r.json()

# Look for shell command results in messages
for msg in traj.get("messages", []):
    if "find . -name" in str(msg.get("content", "")):
        # Judge ran shell commands - sandbox was available
        pass
```

If the judge's trajectory shows commands returning empty results or errors about missing files, the sandbox was likely not properly provisioned.

### API Testing Results (January 2026)

Direct testing of the Lunette SDK revealed:

**`investigate()` signature:**
```python
investigate(
    run_id: str,                    # required
    plan: GradingPlan | ...,        # required
    limit: int | None = None,
    batch_size: int | None = None,
)
# NO enable_sandbox parameter exists!
```

**`LunetteClient` methods:**
- `save_run()` - Upload trajectories (no sandbox)
- `investigate()` - Run analysis plans (no sandbox creation)
- `create_sandbox(service)` - Create sandbox from ComposeService
- `get_run()`, `get_trajectory()` - Fetch data
- `stop_sandboxes()` - Cleanup

**`create_sandbox()` accepts:**
- `ComposeService` object (from Docker Compose)
- `Path` to directory with Dockerfile
- `dict` with `image` or `build` key

**Test results:**
```python
# ✅ Works - creates generic container
sandbox = await client.create_sandbox({"image": "python:3.11-slim"})
result = await sandbox.aexec("pwd")  # Returns "/"

# ❌ Does not work - no enable_sandbox parameter
await client.investigate(run_id=run_id, plan=plan, enable_sandbox=True)
# TypeError: got an unexpected keyword argument 'enable_sandbox'
```

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
| `lunette_utils/dummy_solver.py` | **Dummy solver for feature extraction (use with Inspect + --sandbox lunette)** |
| `lunette_utils/dummy_swebench_task.py` | Pre-built Inspect task with Lunette sandbox config |
| **Experiment A (Task Features)** | |
| `experiment_a/run_dummy_sandbox.py` | **Step 1: Run dummy solver through Inspect with sandbox** |
| `experiment_a/grade_sandbox_runs.py` | **Step 2: Grade sandbox runs with structured output** |
| `experiment_a/lunette_structured_output.py` | Pydantic schemas for structured output |
| **Experiment B (Trajectory Features)** | |
| `experiment_b/lunette_structured_output.py` | Pydantic schemas for trajectory grading |
| `experiment_b/compute_lunette_features.py` | Extract trajectory features with structured output |
| `experiment_b/lunette_features.py` | Feature definitions and loading utilities |

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

## Troubleshooting

### Grading Judge Reports Empty Sandbox

**Symptom:** Investigation trajectory shows:
- "Found 0 Python files"
- "Cannot access repository"
- Features are estimated rather than verified

**Cause:** Trajectories were uploaded via `save_run()` without running through Inspect, so no sandbox was provisioned.

**Solution:** Use one of:
1. Run dummy solver through Inspect with `--sandbox lunette` flag (recommended for SWE-bench)
2. Create sandbox manually via `create_sandbox()` (for generic containers)

See [Sandbox Creation](#sandbox-creation) section for details.

### Partial Feature Coverage (Some Tasks Missing)

**Symptom:** Only 359/500 tasks have Lunette features despite running extraction on all 500.

**Possible causes:**
1. Sandbox creation failures for some tasks
2. API timeouts or rate limiting
3. Tasks with unusual repo structures

**Diagnosis:**
```bash
# Check for error files
ls chris_output/experiment_a/lunette_features/*_error.json | wc -l

# Check raw responses without parsed features
ls chris_output/experiment_a/lunette_features/*_raw.json | wc -l
```

**Solution:**
- Re-run failed tasks: `python -m experiment_a.run_dummy_sandbox --resume`
- Then grade: `python -m experiment_a.grade_sandbox_runs --skip_existing`
- Or use the LLM Judge approach (no sandbox needed) as a fallback - see `experiment_a/compute_llm_judge_features.py`

### Investigation Timeout

**Symptom:** `investigate()` call hangs or returns timeout error.

**Cause:** Complex repos or slow sandbox provisioning.

**Solution:**
```python
# Increase timeout
results = await client.investigate(
    run_id=run_id,
    plan=plan,
    timeout=600,  # 10 minutes
)
```

### Feature Extraction Differs Between Runs

**Symptom:** Same task gets different feature values on different extraction runs.

**Cause:** This is expected since features are extracted by an LLM judge. Some variance is normal.

**Mitigation:**
- Run multiple extractions and average
- Use features with lower variance for predictions
- Consider using the simpler LLM Judge approach (no sandbox) for more consistent results

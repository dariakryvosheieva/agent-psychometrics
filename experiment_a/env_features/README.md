# Environment Features Extraction

Extract deterministic environment features from SWE-bench task environments using Inspect AI's sandbox execution.

## Overview

This module runs bash commands inside SWE-bench Docker containers to extract features about the repository structure, codebase size, test infrastructure, etc. These features are:

1. **Deterministic**: Same results every run (no LLM involved)
2. **Environment-derived**: Information from inside the actual task environment, not just the problem statement
3. **Complementary**: Can augment LLM judge features for difficulty prediction

## Prerequisites

1. **Docker Desktop** must be running
2. **SWE-bench Docker images** will be pulled automatically (first run is slow)
3. **Python environment** with `inspect-ai` and `inspect-evals` installed

## Quick Start

### Step 1: Verify Consistency (2 tasks)

Run extraction twice on 2 tasks to verify determinism:

```bash
# Activate virtual environment
source .venv/bin/activate

# First run
inspect eval experiment_a/env_features/inspect_task.py --limit 2 \
    --log-dir chris_output/env_features/validation_run1/

# Second run (same 2 tasks)
inspect eval experiment_a/env_features/inspect_task.py --limit 2 \
    --log-dir chris_output/env_features/validation_run2/

# Compare results - ALL values must match
python -m experiment_a.env_features.validate_consistency \
    chris_output/env_features/validation_run1/ \
    chris_output/env_features/validation_run2/
```

### Step 2: Full Extraction

After consistency is verified, run full extraction with batching:

```bash
# Run with default settings (batch_size=10, removes images after each batch)
python -m experiment_a.env_features.run_extraction

# Or run manually with inspect CLI
inspect eval experiment_a/env_features/inspect_task.py \
    --limit 500 \
    --max-connections 10 \
    --log-dir chris_output/env_features/swebench_verified/logs/
```

### Step 3: Aggregate to CSV

```bash
# If using run_extraction.py, CSV is created automatically
# Otherwise, aggregate manually:
python -m experiment_a.env_features.run_extraction --aggregate_only
```

Output: `chris_output/env_features/swebench_verified/env_features.csv`

## Features Extracted

### File System Structure
- `env_total_files`: Total files in repo
- `env_total_dirs`: Total directories
- `env_dir_depth_max`: Max directory depth

### Python Files
- `env_python_files`: Python file count
- `env_python_loc`: Total Python lines of code
- `env_init_files`: Package `__init__.py` files

### Test Infrastructure
- `env_test_files`: Test file count
- `env_test_dirs`: Test directories
- `env_conftest_files`: Pytest conftest files
- `env_has_pytest_ini`: Has pytest.ini (0/1)
- `env_has_tox`: Has tox.ini (0/1)

### Build/Package Configuration
- `env_has_setup_py`: Has setup.py (0/1)
- `env_has_setup_cfg`: Has setup.cfg (0/1)
- `env_has_pyproject`: Has pyproject.toml (0/1)
- `env_has_makefile`: Has Makefile (0/1)
- `env_has_dockerfile`: Has Dockerfile (0/1)

### Dependencies
- `env_requirements_count`: Lines in requirements files
- `env_has_requirements`: Number of requirements files

### Git Repository Stats
- `env_git_commits_total`: Total commit count
- `env_git_branches`: Branch count
- `env_git_tags`: Tag count
- `env_git_contributors`: Unique contributors

### Documentation
- `env_doc_files`: Documentation file count (md, rst)
- `env_has_readme`: Has README files
- `env_has_docs_dir`: Has docs/ directory (0/1)
- `env_sphinx_conf`: Sphinx config files

### Code Complexity Proxies
- `env_class_count`: Class definitions
- `env_function_count`: Function definitions
- `env_import_count`: Import statements
- `env_todo_count`: TODO/FIXME comments

### Other File Types
- `env_json_files`: JSON files
- `env_yaml_files`: YAML files
- `env_config_files`: Config files (cfg, ini, conf)
- `env_shell_scripts`: Shell scripts

## Memory Management

Each SWE-bench task requires its own Docker image (~500MB-1GB each). To prevent memory overflow:

1. **Batch processing**: Run 10 tasks at a time
2. **Docker cleanup**: Remove images after each batch
3. **Parallel execution**: 10 containers simultaneously within each batch

```bash
# Default behavior removes images after each batch
python -m experiment_a.env_features.run_extraction

# Keep images (faster if you have enough disk space)
python -m experiment_a.env_features.run_extraction --keep_images
```

## Integration with Experiment A

After extraction, use the CSV with `CSVFeatureSource`:

```python
from pathlib import Path
from experiment_ab_shared.feature_source import CSVFeatureSource, GroupedFeatureSource

env_source = CSVFeatureSource(
    Path("chris_output/env_features/swebench_verified/env_features.csv"),
    name="Environment"
)

# Combine with existing features
grouped = GroupedFeatureSource([
    embedding_source,
    llm_judge_source,
    env_source,  # New environment features
])
```

## File Structure

```
experiment_a/env_features/
├── __init__.py               # Module exports
├── feature_definitions.py    # All feature names and commands
├── extractor_solver.py       # Deterministic Inspect solver
├── inspect_task.py           # Inspect Task with dynamic sandboxing
├── run_extraction.py         # CLI with batching and Docker cleanup
├── validate_consistency.py   # Compare two runs
└── README.md                 # This file
```

## Troubleshooting

### Docker not running
```
failed to connect to the docker API
```
→ Start Docker Desktop

### Image not found
```
Unable to find image 'swebench/sweb.eval.x86_64...
```
→ Images are pulled automatically on first use. This is slow (~500MB per image).

### Memory issues
```
Out of memory
```
→ Reduce `--batch_size` and ensure images are being removed after each batch.

### Rosetta error (Apple Silicon)
```
exec format error
```
→ Enable Rosetta emulation in Docker Desktop settings (images are x86_64 only).

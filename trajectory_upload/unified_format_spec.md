# Unified Trajectory Format Specification

## Purpose
This format normalizes all SWE-bench trajectory formats into a single JSON structure suitable for analysis.

## Schema

```json
{
  "task_id": "django__django-12345",
  "agent": "20240620_sweagent_claude3.5sonnet",
  "resolved": true,
  "messages": [
    {
      "role": "system" | "user" | "assistant",
      "content": "...",
      "timestamp": null
    }
  ],
  "metadata": {
    "source_format": "traj" | "yaml" | "json_chat" | "log_text" | ...,
    "source_file": "path/to/original.traj",
    "environment": "swe_main",
    "total_steps": 34,
    "has_patch": true
  }
}
```

## Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | string | SWE-bench task ID (e.g., "django__django-12345") |
| `agent` | string | Agent/submission name |
| `resolved` | boolean | Whether the agent solved the task (from results.json) |
| `messages` | array | Normalized conversation messages |
| `messages[].role` | string | One of: "system", "user", "assistant" |
| `messages[].content` | string | Message content |
| `messages[].timestamp` | string/null | ISO timestamp if available |
| `metadata.source_format` | string | Original format type |
| `metadata.source_file` | string | Path to source file |
| `metadata.environment` | string | Environment name if available |
| `metadata.total_steps` | integer | Number of steps in original trajectory |
| `metadata.has_patch` | boolean | Whether a patch/submission was generated |

## Message Role Mapping

Different formats map to roles as follows:

| Source Format | Original Field | Maps To |
|--------------|----------------|---------|
| SWE-agent .traj | action | user |
| SWE-agent .traj | observation | assistant |
| YAML history | role=user | user |
| YAML history | role=assistant | assistant |
| YAML history | role=system | system |
| JSON chat list | role field | same |
| Text logs | command/action | user |
| Text logs | output/result | assistant |

## Example Output

```json
{
  "task_id": "astropy__astropy-12907",
  "agent": "20240620_sweagent_claude3.5sonnet",
  "resolved": true,
  "messages": [
    {
      "role": "user",
      "content": "create reproduce_separability_issue.py",
      "timestamp": null
    },
    {
      "role": "assistant",
      "content": "[File: /astropy__astropy/reproduce_separability_issue.py (1 lines total)]\n1:",
      "timestamp": null
    },
    {
      "role": "user",
      "content": "edit 1:1\nfrom astropy.modeling import models as m\n...\nend_of_edit",
      "timestamp": null
    },
    {
      "role": "assistant",
      "content": "[File: /astropy__astropy/reproduce_separability_issue.py (19 lines total)]\n1:from astropy.modeling...",
      "timestamp": null
    }
  ],
  "metadata": {
    "source_format": "traj",
    "source_file": "experiments/evaluation/verified/20240620_sweagent_claude3.5sonnet/trajs/astropy__astropy-12907.traj",
    "environment": "swe_main",
    "total_steps": 34,
    "has_patch": true
  }
}
```

## Filtering Options

When `--filter` is applied, only edit-related messages are kept:
- File operations: edit, create, open, cat
- Code execution: python, pytest, make
- Search results: grep/find_file/search_file with matches


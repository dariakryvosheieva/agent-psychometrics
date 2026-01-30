"""
Prepare SWE-bench Bash-Only results into JSONL for IRT training.

Expected repo layout:
  experiments/evaluation/bash-only/<agent_name>/per_instance_details.json

The per_instance_details.json format:
{
  "<task_id>": {
    "cost": <float>,
    "api_calls": <int>,
    "resolved": <bool>
  },
  ...
}
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]

# Agents excluded by default:
# - v0.0.0 scaffold agents (early version with potential bugs)
# - Agents with missing per_instance_details.json files
EXCLUDED_AGENTS = {
    # v0.0.0 scaffold agents (early version, results not comparable)
    "20250720_mini-v0.0.0-Llama-4-Maverick-17B-Instruct",
    "20250720_mini-v0.0.0-Llama-4-Scout-17B-Instruct",
    "20250720_mini-v0.0.0-claude-3-7-sonnet-20250219",
    "20250720_mini-v0.0.0-gpt-4o-2024-11-20",
    "20250720_mini-v0.0.0_gpt-4.1-mini-2025-04-14",
    # Agents with missing per_instance_details.json
    "20250726_mini-v1.0.0_gemini-2.0-flash",
    "20250726_mini-v1.0.0_gemini-2.5-flash",
    "20250726_mini-v1.0.0_gpt-4.1-2025-04-14",
}


def resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else (ROOT / path)


def load_per_instance_details(details_path: Path) -> Tuple[Dict[str, int], set[str]]:
    """Load per_instance_details.json and extract resolved status for each task."""
    with details_path.open() as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Unexpected per_instance_details.json format at {details_path}")

    responses: Dict[str, int] = {}
    for task_id, details in data.items():
        if not isinstance(details, dict):
            raise ValueError(f"Unexpected task details format for {task_id} at {details_path}")
        if "resolved" not in details:
            raise ValueError(f"Missing 'resolved' field for task {task_id} at {details_path}")
        responses[task_id] = 1 if details["resolved"] else 0

    return responses, set(responses.keys())


def collect_agent(agent_dir: Path) -> Tuple[Dict[str, int], set[str], str]:
    """Collect responses from a single agent directory."""
    details_path = agent_dir / "per_instance_details.json"
    if details_path.exists():
        responses, items = load_per_instance_details(details_path)
        return responses, items, "per_instance_details.json"
    return {}, set(), "missing"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SWE-bench Bash-Only response JSONL")
    parser.add_argument(
        "--experiments_dir",
        type=str,
        default="experiments/evaluation/bash-only",
        help="Path to experiments/evaluation/bash-only",
    )
    parser.add_argument(
        "--cutoff_date",
        type=str,
        default=None,
        help="Include agents with date prefix <= cutoff (YYYYMMDD)",
    )
    parser.add_argument(
        "--max_agents",
        type=int,
        default=None,
        help="Limit number of agents after sorting by name",
    )
    parser.add_argument(
        "--no_complete_matrix",
        action="store_true",
        help="Don't fill missing tasks (sparse matrix). Default is to fill with 0.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="clean_data/swebench_bash_only/swebench_bash_only.jsonl",
        help="Output JSONL path",
    )
    args = parser.parse_args()

    experiments_dir = resolve_path(args.experiments_dir)
    output_path = resolve_path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_items: set[str] = set()
    summary = []
    agents_payload = []

    agent_dirs = [d for d in sorted(experiments_dir.iterdir()) if d.is_dir()]
    excluded_count = 0
    for agent_dir in agent_dirs:
        # Skip agents known to have missing data
        if agent_dir.name in EXCLUDED_AGENTS:
            excluded_count += 1
            continue
        responses, items, source = collect_agent(agent_dir)
        if responses:
            all_items.update(items)
        agents_payload.append((agent_dir, responses, source))

    if excluded_count > 0:
        print(f"Excluded {excluded_count} agents with known missing data")

    # Check for agents with missing data files
    missing_data_agents = [
        agent_dir.name for agent_dir, responses, source in agents_payload
        if source == "missing"
    ]
    if missing_data_agents:
        raise ValueError(
            f"Found {len(missing_data_agents)} agents without per_instance_details.json. "
            f"Cannot proceed with incomplete data. Missing agents:\n"
            + "\n".join(f"  - {name}" for name in missing_data_agents)
        )

    selected = []
    for agent_dir, responses, source in agents_payload:
        if args.cutoff_date is not None:
            date_prefix = agent_dir.name.split("_", 1)[0]
            if date_prefix > args.cutoff_date:
                continue
        selected.append((agent_dir, responses, source))

    if args.max_agents is not None:
        selected = selected[: args.max_agents]

    items_list = sorted(all_items)
    records = []
    for agent_dir, responses, source in selected:
        if not responses and args.no_complete_matrix:
            summary.append((agent_dir.name, 0, 0, source))
            continue
        if not args.no_complete_matrix:
            responses = {iid: int(responses.get(iid, 0)) for iid in items_list}
        resolved_ct = sum(responses.values())
        summary.append((agent_dir.name, len(responses), resolved_ct, source))
        records.append({"subject_id": agent_dir.name, "responses": responses})

    with output_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record))
            f.write("\n")

    print(f"Wrote {len(records)} agents to {output_path}")
    print(f"Unique tasks observed: {len(all_items)}")
    obs_total = sum(len(r["responses"]) for r in records)
    if records:
        counts = [len(r["responses"]) for r in records]
        print(f"Total observations: {obs_total}")
        print(f"Tasks per agent: min={min(counts)} max={max(counts)} mean={obs_total / len(counts):.2f}")
        if not args.no_complete_matrix:
            print("Complete matrix: yes")
        else:
            print("Complete matrix: no (sparse)")
    if summary:
        empty = sum(1 for _name, count, _res, _src in summary if count == 0)
        print(f"Agents with no responses: {empty}")


if __name__ == "__main__":
    main()

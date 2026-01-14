"""
Prepare SWE-bench Verified results into JSONL for IRT training.

Expected repo layout (experiments repo checked out under this repo):
  experiments/evaluation/verified/<agent_name>/results/results.json
  or
  experiments/evaluation/verified/<agent_name>/logs/<instance_id>/report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else (ROOT / path)


def _list_items(value: object) -> Iterable[str]:
    if not isinstance(value, list):
        return []
    return [v for v in value if isinstance(v, str)]


def load_results_json(results_path: Path) -> Tuple[Dict[str, int], set[str]]:
    with results_path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected results.json format at {results_path}")

    resolved = set(_list_items(data.get("resolved", [])))
    all_items = set()
    for value in data.values():
        all_items.update(_list_items(value))

    responses = {iid: (1 if iid in resolved else 0) for iid in all_items}
    return responses, all_items


def load_logs(logs_dir: Path) -> Tuple[Dict[str, int], set[str]]:
    responses: Dict[str, int] = {}
    for report_path in logs_dir.glob("*/report.json"):
        try:
            with report_path.open() as f:
                record = json.load(f)
        except json.JSONDecodeError:
            continue
        resolved = record.get("resolved")
        if resolved is None:
            continue
        instance_id = report_path.parent.name
        responses[instance_id] = 1 if bool(resolved) else 0
    return responses, set(responses.keys())


def collect_agent(agent_dir: Path) -> Tuple[Dict[str, int], set[str], str]:
    results_path = agent_dir / "results" / "results.json"
    logs_dir = agent_dir / "logs"
    if results_path.exists():
        responses, items = load_results_json(results_path)
        return responses, items, "results.json"
    if logs_dir.exists():
        responses, items = load_logs(logs_dir)
        return responses, items, "logs"
    return {}, set(), "missing"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SWE-bench Verified response JSONL")
    parser.add_argument(
        "--experiments_dir",
        type=str,
        default="experiments/evaluation/verified",
        help="Path to experiments/evaluation/verified",
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
        default="clean_data/swebench_verified/swebench_verified.jsonl",
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
    for agent_dir in agent_dirs:
        responses, items, source = collect_agent(agent_dir)
        if responses:
            all_items.update(items)
        agents_payload.append((agent_dir, responses, source))

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

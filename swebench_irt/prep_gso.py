"""
Prepare GSO agent report JSONs into IRT response-matrix JSONL.

Input:
  gso-experiments/results/reports/<model_name>.json

We treat:
  - instance_sets.opt_commit_ids => success (1)
  - remaining task ids (typically instance_sets.completed_ids) => failure (0)

Output JSONL matches the format used by:
  - fulcrum/fellowship/out/chris_irt/swebench_verified_20251115_full.jsonl
  - fulcrum/fellowship/out/chris_irt/swebench_pro.jsonl

Each JSONL line:
  {"subject_id": "<model_name>", "responses": {"<task_id>": 0/1, ...}}
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, Tuple

logger = logging.getLogger(__name__)

# Keep consistent with other prep scripts in this repo.
ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else (ROOT / path)


def _list_str(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [v for v in value if isinstance(v, str)]


def _collect_task_ids(instance_sets: object) -> set[str]:
    """
    Collect all task ids we can find inside `instance_sets`.

    We primarily expect:
      - completed_ids
      - passed_ids
      - *_failed_ids / error_ids / empty_patch_ids

    But to be robust to schema tweaks, we include any list-of-strings values
    whose key ends with "_ids".
    """
    if not isinstance(instance_sets, dict):
        return set()

    items: set[str] = set()

    # Prefer canonical list if present.
    items.update(_list_str(instance_sets.get("completed_ids")))

    # Add passed and failure buckets explicitly (common keys).
    items.update(_list_str(instance_sets.get("passed_ids")))
    items.update(_list_str(instance_sets.get("test_failed_ids")))
    items.update(_list_str(instance_sets.get("base_failed_ids")))
    items.update(_list_str(instance_sets.get("patch_failed_ids")))
    items.update(_list_str(instance_sets.get("error_ids")))
    items.update(_list_str(instance_sets.get("empty_patch_ids")))

    # Also include any other "*_ids" lists (e.g., opt_*_ids).
    for k, v in instance_sets.items():
        if isinstance(k, str) and k.endswith("_ids"):
            items.update(_list_str(v))

    return items


def load_report(report_path: Path, *, success_ids_key: str) -> Tuple[set[str], set[str]]:
    """
    Returns:
      - all_task_ids observed in this report
      - success ids set (as configured by `success_ids_key`)
    """
    with report_path.open() as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Unexpected report JSON format at {report_path}")

    instance_sets = obj.get("instance_sets")
    items = _collect_task_ids(instance_sets)
    successes = set()
    if isinstance(instance_sets, dict):
        successes = set(_list_str(instance_sets.get(success_ids_key)))
    return items, successes


def write_jsonl(
    *,
    per_subject: Dict[str, Tuple[set[str], set[str]]],
    all_items: set[str],
    selected_subjects: list[str],
    output_path: Path,
    no_complete_matrix: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    items_list = sorted(all_items)

    records: list[dict] = []
    summary = []
    for subject_id in selected_subjects:
        items, successes = per_subject[subject_id]
        if no_complete_matrix:
            # Sparse: only include tasks this subject attempted.
            subject_items = sorted(items)
            responses = {iid: (1 if iid in successes else 0) for iid in subject_items}
        else:
            # Dense: include all tasks observed across selected subjects.
            responses = {iid: (1 if iid in successes else 0) for iid in items_list}

        resolved_ct = sum(responses.values())
        summary.append((subject_id, len(responses), resolved_ct, len(successes), len(items)))
        records.append({"subject_id": subject_id, "responses": responses})

    with output_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record))
            f.write("\n")

    print(f"Wrote {len(records)} subjects to {output_path}")
    print(f"Unique tasks observed: {len(all_items)}")
    obs_total = sum(len(r["responses"]) for r in records)
    if records:
        counts = [len(r["responses"]) for r in records]
        print(f"Total observations: {obs_total}")
        print(f"Tasks per subject: min={min(counts)} max={max(counts)} mean={obs_total / len(counts):.2f}")
        print(f"Complete matrix: {'no (sparse)' if no_complete_matrix else 'yes'}")
    if summary:
        # (subject, tasks_in_output, successes_in_output, success_ids, attempted_ids)
        best = max(summary, key=lambda t: t[2])
        worst = min(summary, key=lambda t: t[2])
        print(f"Most successes: {best[0]} ({best[2]}/{best[1]})")
        print(f"Fewest successes: {worst[0]} ({worst[2]}/{worst[1]})")


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare GSO report JSONs into IRT response JSONL")
    p.add_argument(
        "--reports_dir",
        type=str,
        default="gso-experiments/results/reports",
        help="Directory containing per-model report JSONs",
    )
    p.add_argument(
        "--model_regex",
        type=str,
        default=None,
        help="Only include reports whose model_name (filename stem) matches this regex.",
    )
    p.add_argument(
        "--max_subjects",
        type=int,
        default=None,
        help="Limit number of subjects after sorting by subject_id",
    )
    p.add_argument(
        "--no_complete_matrix",
        action="store_true",
        help="Don't fill missing tasks (sparse matrix). Default is to fill with 0.",
    )
    p.add_argument(
        "--success_ids_key",
        type=str,
        default="opt_commit_ids",
        help='Which `instance_sets` key to treat as "success". Default: opt_commit_ids',
    )
    p.add_argument(
        "--output_path",
        type=str,
        default="out/chris_irt/gso.jsonl",
        help="Output JSONL path",
    )
    args = p.parse_args()

    reports_dir = resolve_path(args.reports_dir)
    output_path = resolve_path(args.output_path)

    if not reports_dir.exists():
        raise FileNotFoundError(f"reports_dir not found: {reports_dir}")

    rx = re.compile(args.model_regex) if args.model_regex else None

    # subject_id -> (items, passed)
    per_subject: dict[str, Tuple[set[str], set[str]]] = {}
    all_items: set[str] = set()

    report_paths = sorted(reports_dir.glob("*.json"))
    if not report_paths:
        raise ValueError(f"No report JSONs found under {reports_dir}")

    for report_path in report_paths:
        subject_id = report_path.stem
        if rx is not None and not rx.search(subject_id):
            continue

        items, successes = load_report(report_path, success_ids_key=args.success_ids_key)
        if not items:
            logger.warning(f"Skipping {report_path} (no task ids found)")
            continue

        per_subject[subject_id] = (items, successes)
        all_items.update(items)

    if not per_subject:
        raise ValueError("No subjects selected (after filtering / empty reports).")

    selected_subjects = sorted(per_subject.keys())
    if args.max_subjects is not None:
        selected_subjects = selected_subjects[: args.max_subjects]

    # Output (configurable success key; default is opt_commit_ids).
    write_jsonl(
        per_subject=per_subject,
        all_items=all_items,
        selected_subjects=selected_subjects,
        output_path=output_path,
        no_complete_matrix=args.no_complete_matrix,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()


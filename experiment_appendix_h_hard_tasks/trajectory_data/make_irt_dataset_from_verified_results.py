from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List


def _iter_submission_dirs(root: Path) -> Iterable[Path]:
    for p in sorted(root.iterdir()):
        if p.is_dir():
            yield p


def _load_instance_ids_from_hf(dataset_name: str, split: str) -> List[str]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency `datasets`. Activate the project venv or install datasets."
        ) from e

    ds = load_dataset(dataset_name, split=split)
    return [str(x["instance_id"]) for x in ds]


def _load_instance_ids_from_file(path: Path) -> List[str]:
    """
    Supported formats:
    - .json: either a list of instance ids (strings) or objects containing instance_id
    - .jsonl/.jsonlines: one JSON object per line containing instance_id, or a raw string
    - .txt: one instance id per line
    """
    suf = path.suffix.lower()
    if suf == ".txt":
        return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if suf == ".json":
        obj = json.loads(path.read_text())
        if isinstance(obj, list):
            if not obj:
                return []
            if isinstance(obj[0], str):
                return [str(x) for x in obj]
            if isinstance(obj[0], dict) and "instance_id" in obj[0]:
                return [str(x["instance_id"]) for x in obj]
        raise ValueError(f"Unsupported JSON format in {path}")
    if suf in {".jsonl", ".jsonlines"}:
        ids: List[str] = []
        for ln in path.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            obj = json.loads(ln)
            if isinstance(obj, str):
                ids.append(obj)
            elif isinstance(obj, dict) and "instance_id" in obj:
                ids.append(str(obj["instance_id"]))
            else:
                raise ValueError(f"Unsupported JSONL line in {path}: {ln[:200]}")
        return ids
    raise ValueError(f"Unsupported instances file type: {path} (expected .txt/.json/.jsonl/.jsonlines)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build py_irt jsonlines from SWE-bench Verified results.json files.")
    p.add_argument(
        "--verified-root",
        type=Path,
        default=Path("experiments/evaluation/verified"),
        help="Root directory containing per-submission folders (default: experiments/evaluation/verified).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("experiment_appendix_h_hard_tasks/trajectory_data/irt_verified.jsonlines"),
        help="Output JSON Lines path (default: experiment_appendix_h_hard_tasks/trajectory_data/irt_verified.jsonlines).",
    )
    p.add_argument(
        "--instances",
        type=Path,
        default=None,
        help="Optional path to a file containing the canonical instance_id list (overrides HF dataset).",
    )
    p.add_argument(
        "--hf-dataset",
        type=str,
        default="SWE-bench/SWE-bench_Verified",
        help="HuggingFace dataset name to load instance ids from (default: SWE-bench/SWE-bench_Verified).",
    )
    p.add_argument(
        "--hf-split",
        type=str,
        default="test",
        help="HuggingFace split to load (default: test).",
    )

    args = p.parse_args(argv)

    if args.instances is not None:
        instance_ids = _load_instance_ids_from_file(args.instances)
        source = str(args.instances)
    else:
        instance_ids = _load_instance_ids_from_hf(args.hf_dataset, args.hf_split)
        source = f"{args.hf_dataset}:{args.hf_split}"

    if not instance_ids:
        raise ValueError("No instance_ids loaded; cannot build dataset.")

    used = 0
    skipped = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w") as out:
        for subdir in _iter_submission_dirs(args.verified_root):
            results_path = subdir / "results" / "results.json"
            if not results_path.exists():
                skipped += 1
                continue
            results = json.loads(results_path.read_text())
            resolved = results.get("resolved", [])
            if not isinstance(resolved, list):
                # Some older folders store only a resolved count; skip those.
                skipped += 1
                continue
            resolved_set = set(str(x) for x in resolved)
            responses = {iid: (1 if iid in resolved_set else 0) for iid in instance_ids}
            out.write(
                json.dumps(
                    {"subject_id": subdir.name, "responses": responses},
                    separators=(",", ":"),
                    sort_keys=False,
                )
            )
            out.write("\n")
            used += 1

    print(
        json.dumps(
            {
                "verified_root": str(args.verified_root),
                "output": str(args.output),
                "instance_id_source": source,
                "num_instances": len(instance_ids),
                "subjects_written": used,
                "subjects_skipped_missing_or_unsupported": skipped,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())





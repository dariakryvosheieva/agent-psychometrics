from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple


SAMPLE_CHOICES = ("head", "tail", "random_span", "headtail", "headtail_random")
_INSTANCE_ID_RE = re.compile(r"^([A-Za-z0-9_.-]+__[^_]+-\d+)")


@dataclass(frozen=True)
class TrajRow:
    task_id: str
    agent: str
    success: bool
    trajectory: str


def _infer_task_id(name: str) -> Optional[str]:
    """
    Given a filename (not a path), infer the SWE-bench instance_id.

    instance_id strings do not contain '.' so we can safely strip known suffixes.
    """
    # NOTE: we now infer a *candidate* base id, but validity is checked separately.
    lower = name.lower()
    for suf in (".traj.json", ".jsonlines", ".jsonl", ".json", ".traj", ".log", ".txt", ".md", ".yaml", ".yml"):
        if lower.endswith(suf):
            base = name[: -len(suf)]
            return base if base else None
    if "." in name:
        base = name.split(".", 1)[0]
        return base if base else None
    return name or None


def _maybe_strip_suffixes(task_id: str) -> str:
    """
    Normalize task_id candidates that have known suffixes appended by some submissions.
    """
    # Some submissions prefix filenames with "instance_" (e.g. instance_django__django-11163.log).
    # SWE-bench instance_ids do not include this prefix, so strip it if present.
    if task_id.startswith("instance_"):
        task_id = task_id[len("instance_") :]
    for suf in ("_traj", "_agent", "_trail", "_trial", "_vote", "_voting"):
        if task_id.endswith(suf):
            return task_id[: -len(suf)]
    # Handle numbered variants like *_trail_0, *_trail_1, *_trial_2
    for prefix in ("_trail_", "_trial_"):
        if prefix in task_id:
            head, tail = task_id.rsplit(prefix, 1)
            if tail.isdigit():
                return head
    return task_id


def _extract_instance_id_prefix(task_id_candidate: str) -> Optional[str]:
    """
    Extract the SWE-bench instance_id prefix from a filename-derived candidate.

    Many submissions store multiple files per instance with suffixes like:
    - <instance_id>_traj.json
    - <instance_id>_trail_0.txt
    - <instance_id>_voting.json
    - <instance_id>_agent.json
    """
    s = _maybe_strip_suffixes(task_id_candidate)
    m = _INSTANCE_ID_RE.match(s)
    if m:
        return m.group(1)
    # If the candidate itself looks like a full instance id already.
    return s or None


def _load_verified_instance_ids() -> Set[str]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency `datasets`. Activate the project venv or install datasets."
        ) from e
    ds = load_dataset("SWE-bench/SWE-bench_Verified", split="test")
    return {str(x["instance_id"]) for x in ds}


def _read_sampled_text(
    path: Path,
    *,
    max_chars: int,
    strategy: str,
    rng: random.Random,
) -> str:
    if max_chars <= 0:
        max_chars = 0
    # Read a bit more than max_chars in bytes to mitigate UTF-8 expansion.
    # (Worst case 4 bytes/char; we keep it modest.)
    want = max(1, int(max_chars) * 4)

    st = strategy
    data: bytes
    size = path.stat().st_size

    if st == "head":
        with open(path, "rb") as f:
            data = f.read(want)
        return data.decode("utf-8", errors="replace")[:max_chars]

    if st == "tail":
        start = max(0, size - want)
        with open(path, "rb") as f:
            f.seek(start)
            data = f.read()
        text = data.decode("utf-8", errors="replace")
        return text[-max_chars:] if max_chars > 0 else ""

    if st == "random_span":
        if size <= want:
            with open(path, "rb") as f:
                data = f.read()
            return data.decode("utf-8", errors="replace")[:max_chars]
        start = rng.randint(0, max(0, size - want))
        with open(path, "rb") as f:
            f.seek(start)
            data = f.read(want)
        return data.decode("utf-8", errors="replace")[:max_chars]

    if st == "headtail":
        h_chars = max_chars // 2
        t_chars = max_chars - h_chars
        head = _read_sampled_text(path, max_chars=h_chars, strategy="head", rng=rng)
        tail = _read_sampled_text(path, max_chars=t_chars, strategy="tail", rng=rng)
        return head + "\n\n[...TRUNCATED...]\n\n" + tail

    if st == "headtail_random":
        h_chars = max_chars // 3
        t_chars = max_chars // 3
        m_chars = max_chars - h_chars - t_chars
        head = _read_sampled_text(path, max_chars=h_chars, strategy="head", rng=rng)
        tail = _read_sampled_text(path, max_chars=t_chars, strategy="tail", rng=rng)
        mid = _read_sampled_text(path, max_chars=m_chars, strategy="random_span", rng=rng)
        return head + "\n\n[...MID-SAMPLE...]\n\n" + mid + "\n\n[...TRUNCATED...]\n\n" + tail

    raise ValueError(f"Unknown sampling strategy: {strategy}")


def _truncate_text(text: str, *, max_chars: int, strategy: str, rng: random.Random) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if strategy == "head":
        return text[:max_chars]
    if strategy == "tail":
        return text[-max_chars:]
    if strategy == "random_span":
        start = rng.randint(0, max(0, len(text) - max_chars))
        return text[start : start + max_chars]
    if strategy == "headtail":
        h = max_chars // 2
        t = max_chars - h
        return text[:h] + "\n\n[...TRUNCATED...]\n\n" + text[-t:]
    if strategy == "headtail_random":
        h = max_chars // 3
        t = max_chars // 3
        m = max_chars - h - t
        mid_start = rng.randint(0, max(0, len(text) - m))
        mid = text[mid_start : mid_start + m]
        return text[:h] + "\n\n[...MID-SAMPLE...]\n\n" + mid + "\n\n[...TRUNCATED...]\n\n" + text[-t:]
    raise ValueError(f"Unknown sampling strategy: {strategy}")


def _iter_submission_dirs(verified_root: Path) -> Iterable[Path]:
    for p in sorted(verified_root.iterdir()):
        if p.is_dir():
            yield p


def _iter_traj_files(trajs_dir: Path) -> Iterator[Path]:
    # Some submissions store nested traj structures; handle both flat and nested.
    for p in trajs_dir.rglob("*"):
        if p.is_file():
            yield p


def _infer_task_id_from_path_parts(fpath: Path, *, trajs_dir: Path, valid_ids: Set[str]) -> Optional[str]:
    """
    Fallback inference for submissions that store files under a per-instance directory, e.g.:

      trajs/<instance_id>/debug_agent_write_patch_1.json

    In those cases the filename alone doesn't contain the task id, but one of the path
    components often *is* the instance_id.
    """
    try:
        rel_parts = fpath.relative_to(trajs_dir).parts
    except Exception:
        rel_parts = fpath.parts
    for part in rel_parts:
        # Quick exact match first (common case).
        if part in valid_ids:
            return part
        # Sometimes the directory name may include suffixes; try normalization/regex.
        cand = _infer_task_id(part) or part
        base = _extract_instance_id_prefix(cand)
        if base and base in valid_ids:
            return base
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build trajectories.jsonl from verified/*/trajs files.")
    p.add_argument(
        "--verified-root",
        type=Path,
        default=Path("experiments/evaluation/verified"),
        help="Verified submissions root (default: experiments/evaluation/verified).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("experiment_appendix_h_hard_tasks/trajectory_data/verified_trajectories.jsonl"),
        help="Output trajectories JSONL path.",
    )
    p.add_argument(
        "--max-chars",
        type=int,
        default=12000,
        help="Max characters to store per trajectory (default: 12000).",
    )
    p.add_argument(
        "--text-sampling",
        type=str,
        default="tail",
        choices=SAMPLE_CHOICES,
        help="Sampling strategy (default: tail).",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed for random span sampling.")
    p.add_argument(
        "--max-submissions",
        type=int,
        default=0,
        help="If >0, only process this many submissions (debug).",
    )
    p.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="If >0, stop after writing this many trajectory rows (debug).",
    )
    p.add_argument(
        "--log-invalid-task-examples",
        type=int,
        default=0,
        help=(
            "If >0, print up to N examples of traj files skipped due to invalid task_id "
            "(i.e., instance_id not in SWE-bench Verified test set)."
        ),
    )

    args = p.parse_args(argv)
    rng = random.Random(int(args.seed))
    valid_ids = _load_verified_instance_ids()

    total_submissions = 0
    submissions_with_trajs = 0
    submissions_missing_trajs = 0
    submissions_missing_results = 0

    files_seen = 0
    rows_written = 0
    rows_skipped_unparsable_id = 0
    files_skipped_readme = 0
    files_skipped_alternate = 0
    files_skipped_invalid_task = 0
    files_recovered_task_from_path = 0
    invalid_task_examples: List[Dict[str, str]] = []

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as out:
        for subdir in _iter_submission_dirs(args.verified_root):
            total_submissions += 1
            if args.max_submissions > 0 and total_submissions > int(args.max_submissions):
                break

            results_path = subdir / "results" / "results.json"
            if not results_path.exists():
                submissions_missing_results += 1
                continue
            results = json.loads(results_path.read_text())
            resolved = results.get("resolved", [])
            if not isinstance(resolved, list):
                # Older folders may store only a count; skip.
                submissions_missing_results += 1
                continue
            resolved_set = set(str(x) for x in resolved)

            trajs_dir = subdir / "trajs"
            if not trajs_dir.exists():
                submissions_missing_trajs += 1
                continue

            any_files = False
            # Aggregate by (task_id, agent) then emit one concatenated trajectory per task.
            by_task: Dict[str, List[Path]] = {}

            for fpath in _iter_traj_files(trajs_dir):
                any_files = True
                files_seen += 1

                name = fpath.name
                lower = name.lower()
                if lower.startswith("readme"):
                    files_skipped_readme += 1
                    continue
                if "_alternate" in lower:
                    files_skipped_alternate += 1
                    continue

                cand = _infer_task_id(name)
                if cand is None:
                    rows_skipped_unparsable_id += 1
                    continue
                base = _extract_instance_id_prefix(cand)
                if base is None or base not in valid_ids:
                    # Fallback: try to infer instance_id from the directory path.
                    recovered = _infer_task_id_from_path_parts(fpath, trajs_dir=trajs_dir, valid_ids=valid_ids)
                    if recovered is not None:
                        base = recovered
                        files_recovered_task_from_path += 1
                    else:
                        files_skipped_invalid_task += 1
                        if int(args.log_invalid_task_examples) > 0 and len(invalid_task_examples) < int(
                            args.log_invalid_task_examples
                        ):
                            invalid_task_examples.append(
                                {
                                    "agent": subdir.name,
                                    "path": fpath.relative_to(subdir).as_posix(),
                                    "filename": name,
                                    "candidate": str(cand),
                                    "extracted_base": str(base),
                                }
                            )
                        continue

                by_task.setdefault(base, []).append(fpath)

            if by_task:
                for task_id, paths in sorted(by_task.items()):
                    # Deterministic order so concatenation is stable.
                    paths_sorted = sorted(paths, key=lambda p: p.as_posix())
                    # Allocate per-file budget, then truncate the final combined string.
                    per_file = max(200, int(args.max_chars) // max(1, len(paths_sorted)))
                    parts: List[str] = []
                    for pth in paths_sorted:
                        rel = pth.relative_to(trajs_dir).as_posix()
                        chunk = _read_sampled_text(
                            pth,
                            max_chars=per_file,
                            strategy=str(args.text_sampling),
                            rng=rng,
                        )
                        if not chunk.strip():
                            continue
                        parts.append(f"===== FILE: {rel} =====\n{chunk}")
                    combined = "\n\n".join(parts)
                    combined = _truncate_text(combined, max_chars=int(args.max_chars), strategy=str(args.text_sampling), rng=rng)
                    if not combined.strip():
                        continue
                    row = TrajRow(
                        task_id=task_id,
                        agent=subdir.name,
                        success=(task_id in resolved_set),
                        trajectory=combined,
                    )
                    out.write(json.dumps(row.__dict__, ensure_ascii=False, separators=(",", ":")))
                    out.write("\n")
                    rows_written += 1
                    if args.max_files > 0 and rows_written >= int(args.max_files):
                        break
                if args.max_files > 0 and rows_written >= int(args.max_files):
                    break

            if any_files:
                submissions_with_trajs += 1
            else:
                submissions_missing_trajs += 1

            if args.max_files > 0 and rows_written >= int(args.max_files):
                break

    print(
        json.dumps(
            {
                "verified_root": str(args.verified_root),
                "output": str(args.output),
                "max_chars": int(args.max_chars),
                "text_sampling": str(args.text_sampling),
                "seed": int(args.seed),
                "total_submissions": total_submissions,
                "submissions_with_trajs": submissions_with_trajs,
                "submissions_missing_trajs_or_empty": submissions_missing_trajs,
                "submissions_missing_results": submissions_missing_results,
                "traj_files_seen": files_seen,
                "rows_written": rows_written,
                "rows_skipped_unparsable_task_id": rows_skipped_unparsable_id,
                "files_skipped_readme": files_skipped_readme,
                "files_skipped_alternate": files_skipped_alternate,
                "files_skipped_invalid_task": files_skipped_invalid_task,
                "files_recovered_task_from_path": files_recovered_task_from_path,
            },
            indent=2,
        )
    )
    if invalid_task_examples:
        print("\nExamples of files skipped due to invalid task_id (not in SWE-bench Verified):")
        print(json.dumps(invalid_task_examples, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



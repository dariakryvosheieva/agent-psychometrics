#!/usr/bin/env python3
"""
Extract Terminal-Bench 2.0 task statements ("instruction") from the local
`terminal-bench/tasks/` registry and write a SWE-bench-like JSONL.

Output JSONL schema per line:
  {"task_id": "...", "problem_statement": "...", "patch": "...", "tests": "..."}

Gold patches:
  - We store the *entire* contents of `solution.sh` (when present) in the `patch`
    field. Many tasks don't use a `diff --git` patch; they instead generate files,
    run commands, etc. Keeping the whole `solution.sh` captures the intended
    reference solution behavior.
  - If `solution.sh` does not exist, `patch` is left as an empty string.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional


_TOP_LEVEL_KEY_RE = re.compile(r"^[A-Za-z0-9_]+:\s*")


def normalize_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = "\n".join(line.rstrip() for line in s.split("\n"))
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s


def normalize_newlines_preserve_whitespace(s: str) -> str:
    # Keep content as close as possible to the source file, while normalizing
    # line endings to Unix newlines for JSONL portability.
    return s.replace("\r\n", "\n").replace("\r", "\n")


def extract_instruction_from_task_yaml(task_yaml_text: str) -> Optional[str]:
    """
    Extract the `instruction` block scalar from a Terminal-Bench `task.yaml`.

    We intentionally avoid a full YAML dependency (PyYAML) and parse just the
    `instruction: |-` style block used in the benchmark tasks.
    """
    lines = task_yaml_text.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        if line.startswith("instruction:"):
            start_idx = i
            break
    if start_idx is None:
        return None

    # Some tasks use an inline scalar: `instruction: ...`
    inline = lines[start_idx][len("instruction:") :].strip()
    if inline and not inline.startswith("|") and not inline.startswith(">"):
        return normalize_text(inline)

    # Most tasks use `instruction: |-` followed by 2-space indented content.
    instr_lines: List[str] = []
    for j in range(start_idx + 1, len(lines)):
        line = lines[j]
        # Stop at the next top-level YAML key.
        if _TOP_LEVEL_KEY_RE.match(line) and not line.startswith(" "):
            break
        if line.startswith("  "):
            instr_lines.append(line[2:])
        elif line.strip() == "":
            instr_lines.append("")
        else:
            # Unexpected indentation style; best-effort include.
            if line.startswith(" "):
                instr_lines.append(line.lstrip(" "))
            else:
                break

    instr = "\n".join(instr_lines)
    instr = normalize_text(instr)
    return instr if instr else None


_HEREDOC_START_RE = re.compile(r"<<\s*(['\"]?)([A-Za-z0-9_]+)\1")


def _iter_heredoc_blocks(sh_text: str) -> Iterable[str]:
    """
    Yield heredoc bodies from a shell script, for patterns like:
      cat > file << 'EOF'
      ... body ...
      EOF
    """
    lines = sh_text.splitlines()
    i = 0
    while i < len(lines):
        m = _HEREDOC_START_RE.search(lines[i])
        if not m:
            i += 1
            continue
        tag = m.group(2)
        body: List[str] = []
        j = i + 1
        while j < len(lines):
            if lines[j].strip() == tag:
                break
            body.append(lines[j])
            j += 1
        if body:
            yield "\n".join(body).strip("\n")
        i = j + 1


def extract_patch_from_solution_sh(solution_sh_text: str) -> str:
    """
    Return the entire `solution.sh` contents.

    This intentionally does NOT try to infer a unified diff; many Terminal-Bench
    tasks have reference solutions that are not expressed as diffs.
    """
    s = normalize_newlines_preserve_whitespace(solution_sh_text)
    if not s.endswith("\n"):
        s += "\n"
    return s


def _read_text_truncated(path: Path, *, max_chars: int = 50_000) -> str:
    """
    Read a text file with UTF-8 and truncate extremely large files to keep JSONL
    and downstream prompts reasonable.
    """
    try:
        s = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Best-effort: some tasks may contain non-UTF8 bytes in auxiliary files.
        s = path.read_text(encoding="utf-8", errors="replace")
    s = normalize_newlines_preserve_whitespace(s)
    if len(s) > int(max_chars):
        return s[: int(max_chars)].rstrip() + "\n\n# [truncated]\n"
    return s


def extract_tests_from_task_dir(task_dir: Path) -> str:
    """
    Extract Terminal-Bench tests in a prompt-friendly form.

    We include:
      - `run-tests.sh` when present (how evaluation is invoked)
      - all files under `tests/` (actual test cases)
    """
    chunks: List[str] = []

    run_tests = task_dir / "run-tests.sh"
    if run_tests.exists() and run_tests.is_file():
        chunks.append(f"### run-tests.sh\n{_read_text_truncated(run_tests)}")

    tests_dir = task_dir / "tests"
    if tests_dir.exists() and tests_dir.is_dir():
        # Stable ordering
        for p in sorted(tests_dir.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(task_dir).as_posix()
            chunks.append(f"### {rel}\n{_read_text_truncated(p)}")

    s = "\n\n".join(chunks)
    return normalize_text(s) if s.strip() else ""


def load_task_list(meta_json_path: Path) -> List[str]:
    meta = json.loads(meta_json_path.read_text(encoding="utf-8"))
    task_list = meta.get("task_list")
    if not isinstance(task_list, list) or not all(isinstance(t, str) for t in task_list):
        raise ValueError(f"Invalid task_list in meta json: {meta_json_path}")
    return list(task_list)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Path to write JSONL, e.g. terminal_bench_tasks.jsonl")
    ap.add_argument(
        "--tasks-dir",
        default=str(Path(__file__).resolve().parent / "terminal-bench" / "tasks"),
        help="Path to the Terminal-Bench tasks directory",
    )
    ap.add_argument(
        "--meta",
        default=str(Path(__file__).resolve().parent / "data" / "terminal_bench" / "terminal_bench_2.0.meta.json"),
        help="Path to a meta JSON containing a `task_list` to filter tasks (recommended)",
    )
    ap.add_argument("--limit", type=int, default=0, help="If >0, only process this many tasks (debugging)")
    args = ap.parse_args()

    tasks_dir = Path(args.tasks_dir)
    if not tasks_dir.exists():
        raise FileNotFoundError(f"tasks dir not found: {tasks_dir}")

    meta_path = Path(args.meta)
    task_ids = load_task_list(meta_path) if meta_path.exists() else sorted(p.name for p in tasks_dir.iterdir() if p.is_dir())
    if args.limit and args.limit > 0:
        task_ids = task_ids[: int(args.limit)]

    records = []
    missing_instruction: List[str] = []
    missing_task_dir: List[str] = []
    patches_found = 0
    tests_found = 0

    for tid in task_ids:
        task_path = tasks_dir / tid
        if not task_path.exists():
            missing_task_dir.append(tid)
            continue

        task_yaml = task_path / "task.yaml"
        if not task_yaml.exists():
            missing_task_dir.append(tid)
            continue

        instr = extract_instruction_from_task_yaml(task_yaml.read_text(encoding="utf-8"))
        if not instr:
            missing_instruction.append(tid)
            continue

        patch = ""
        solution_sh = task_path / "solution.sh"
        if solution_sh.exists():
            patch = extract_patch_from_solution_sh(solution_sh.read_text(encoding="utf-8"))
            if patch:
                patches_found += 1

        tests = extract_tests_from_task_dir(task_path)
        if tests:
            tests_found += 1

        records.append({"task_id": tid, "problem_statement": instr, "patch": patch, "tests": tests})

    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} tasks to {args.out}")
    print(f"Found {patches_found} tasks with non-empty patches")
    print(f"Found {tests_found} tasks with non-empty tests")
    if missing_task_dir:
        print(f"WARNING: missing task.yaml for {len(missing_task_dir)} tasks, e.g. {missing_task_dir[:10]}")
    if missing_instruction:
        print(f"WARNING: failed to extract instruction for {len(missing_instruction)} tasks, e.g. {missing_instruction[:10]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

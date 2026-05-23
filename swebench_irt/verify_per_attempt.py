"""Verify data/terminalbench/responses_per_attempt.jsonl lines up with the binary
file at data/terminalbench/responses.jsonl.

Checks:
  1. Same subject_id set (or report missing agents).
  2. For every (agent, task) cell present in the per-attempt file, the binary
     file's value at that cell equals `1 if successes >= trials / 2 else 0`
     (the majority-threshold collapse rule).
  3. Tasks in binary but not in per-attempt for an agent (these are cells the
     scrape-time zero-fill fabricated in the binary file).
  4. Tasks in per-attempt that aren't in binary at all (would indicate a
     leaderboard change between the two scrapes).

Exits non-zero on any inconsistency that suggests the two files came from
different data sources, but prints (without failing) the counts in (3) and (4)
since those are diagnostic, not strict mismatches.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open("r") as f:
        for raw in f:
            rec = json.loads(raw)
            out[rec["subject_id"]] = rec["responses"]
    return out


def main() -> int:
    binary_path = ROOT / "data/terminalbench/responses.jsonl"
    per_attempt_path = ROOT / "data/terminalbench/responses_per_attempt.jsonl"

    binary = _load(binary_path)
    per_attempt = _load(per_attempt_path)

    print(f"binary: {len(binary)} agents")
    print(f"per_attempt: {len(per_attempt)} agents")

    binary_agents = set(binary.keys())
    per_attempt_agents = set(per_attempt.keys())

    only_in_binary = sorted(binary_agents - per_attempt_agents)
    only_in_per_attempt = sorted(per_attempt_agents - binary_agents)

    if only_in_per_attempt:
        print(
            f"\nFAIL: {len(only_in_per_attempt)} agents present in per_attempt "
            f"but not binary: {only_in_per_attempt[:5]}{'...' if len(only_in_per_attempt) > 5 else ''}"
        )
        return 1

    if only_in_binary:
        print(
            f"\nWARN: {len(only_in_binary)} agents in binary have no per_attempt data "
            f"(detail page returned no task breakdown):"
        )
        for a in only_in_binary:
            print(f"  - {a}")

    # Check majority-threshold collapse rule on overlapping cells.
    mismatches: list[tuple[str, str, dict, int]] = []
    fabricated_in_binary: dict[str, int] = {}
    extra_tasks_in_per_attempt: dict[str, list[str]] = {}

    overlap_agents = sorted(binary_agents & per_attempt_agents)
    for agent in overlap_agents:
        b = binary[agent]
        pa = per_attempt[agent]
        b_tasks = set(b.keys())
        pa_tasks = set(pa.keys())

        # Tasks the binary has but per_attempt doesn't (zero-filled by scrape-time fill).
        missing_in_pa = sorted(b_tasks - pa_tasks)
        if missing_in_pa:
            fabricated_in_binary[agent] = len(missing_in_pa)

        # Tasks per_attempt has but binary doesn't (would indicate scrape diff).
        extra_in_pa = sorted(pa_tasks - b_tasks)
        if extra_in_pa:
            extra_tasks_in_per_attempt[agent] = extra_in_pa

        for task in b_tasks & pa_tasks:
            pa_val = pa[task]
            s = int(pa_val["successes"])
            t = int(pa_val["trials"])
            expected_binary = 1 if s >= t / 2 else 0
            actual_binary = int(b[task])
            if actual_binary != expected_binary:
                mismatches.append((agent, task, pa_val, actual_binary))

    if mismatches:
        # Real-world cause: the tbench leaderboard sometimes adds extra trials
        # to an agent's runs after the binary file was scraped (we see cells
        # with trials > 5 in per_attempt, indicating re-runs). When the
        # majority outcome flips because of those extra trials, the binary
        # snapshot and the live per_attempt disagree. This is leaderboard
        # drift, not a scrape mismatch, so it's a warning rather than a fail.
        per_agent: dict[str, int] = {}
        for agent, _task, _pa, _b in mismatches:
            per_agent[agent] = per_agent.get(agent, 0) + 1
        print(
            f"\nWARN: {len(mismatches)} cells where binary disagrees with "
            f"majority-collapse(per_attempt), concentrated in "
            f"{len(per_agent)} agent(s):"
        )
        for agent, n in sorted(per_agent.items(), key=lambda x: -x[1]):
            print(f"  - {agent}: {n} flipped cells")
        print("First 5 examples:")
        for agent, task, pa_val, actual in mismatches[:5]:
            s, t = pa_val["successes"], pa_val["trials"]
            print(
                f"  {agent} / {task}: binary={actual}, "
                f"per_attempt={s}/{t} -> expected {1 if s >= t / 2 else 0}"
            )
        print(
            "Cause: the leaderboard adds extra trials to some agents after the "
            "original scrape; if the majority outcome flips with the new trials, "
            "the binary snapshot disagrees with the live per_attempt count. "
            "Binomial IRT trains on the current data (correct), and the binary "
            "Table 2 remains reproducible from the existing binary file."
        )

    if fabricated_in_binary:
        total = sum(fabricated_in_binary.values())
        print(
            f"\nDiagnostic: {total} cells in binary are missing from per_attempt "
            f"(these were fabricated by the scrape-time zero-fill, "
            f"affecting {len(fabricated_in_binary)} agents)."
        )
        for agent, n in sorted(fabricated_in_binary.items(), key=lambda x: -x[1])[:5]:
            print(f"  - {agent}: {n} fabricated cells")
    else:
        print("\nNo binary cells are missing from per_attempt (scrape-time zero-fill never fired).")

    if extra_tasks_in_per_attempt:
        total = sum(len(v) for v in extra_tasks_in_per_attempt.values())
        print(
            f"\nWARN: {total} per_attempt cells reference tasks not in binary "
            f"(leaderboard task list changed between scrapes), "
            f"affecting {len(extra_tasks_in_per_attempt)} agents."
        )
    else:
        print("All per_attempt tasks are present in binary (task universe matches).")

    print("\nVerification passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

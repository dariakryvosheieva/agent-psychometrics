"""
Verify dataset dimensions for IRT training.
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Check JSONL response matrix stats")
    parser.add_argument(
        "--data_path",
        type=str,
        default="clean_data/swebench_verified/swebench_verified.jsonl",
        help="Path to JSONL responses",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise SystemExit(f"Missing data at {data_path}")

    agents = 0
    obs_total = 0
    items = set()
    per_agent = []
    with data_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            responses = rec.get("responses", {})
            agents += 1
            obs_total += len(responses)
            per_agent.append(len(responses))
            items.update(responses.keys())

    if agents == 0:
        print("No agents found.")
        return
    print(f"Agents: {agents}")
    print(f"Unique tasks: {len(items)}")
    print(f"Total observations: {obs_total}")
    print(
        f"Tasks per agent: min={min(per_agent)} max={max(per_agent)} mean={obs_total / agents:.2f}"
    )


if __name__ == "__main__":
    main()

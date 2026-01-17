#!/usr/bin/env python3
"""
IRT Anomaly Analysis: Find cases where smart models fail easy problems
and dumb models pass hard problems.

Uses 1PL IRT model: P(success) = sigmoid(theta - b)
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.special import expit  # sigmoid


def load_data(data_dir: Path):
    """Load IRT parameters and response matrix."""
    # Load 1PL parameters
    abilities = pd.read_csv(data_dir / "1d_1pl" / "abilities.csv", index_col=0)
    items = pd.read_csv(data_dir / "1d_1pl" / "items.csv", index_col=0)

    # Load response matrix
    response_file = data_dir.parent / "swebench_verified" / "swebench_verified_20251120_full.jsonl"
    responses = {}
    with open(response_file) as f:
        for line in f:
            record = json.loads(line)
            responses[record["subject_id"]] = record["responses"]

    return abilities, items, responses


def find_anomalies(abilities, items, responses,
                   high_ability_threshold=2.0,
                   low_ability_threshold=-1.0,
                   easy_difficulty_threshold=-2.5,
                   hard_difficulty_threshold=4.0,
                   high_prob_threshold=0.95,
                   low_prob_threshold=0.05):
    """
    Find anomalous results.

    Type A: High-ability agents failing easy tasks (expected P > high_prob_threshold)
    Type B: Low-ability agents passing hard tasks (expected P < low_prob_threshold)
    """
    type_a_anomalies = []  # Smart fails easy
    type_b_anomalies = []  # Dumb passes hard

    for agent_id, agent_responses in responses.items():
        if agent_id not in abilities.index:
            continue
        theta = abilities.loc[agent_id, "theta"]

        for task_id, actual_result in agent_responses.items():
            if task_id not in items.index:
                continue
            b = items.loc[task_id, "b"]

            # 1PL probability: P = sigmoid(theta - b)
            predicted_prob = expit(theta - b)

            # Type A: High-ability agent fails easy task
            if (theta > high_ability_threshold and
                b < easy_difficulty_threshold and
                actual_result == 0 and
                predicted_prob > high_prob_threshold):
                type_a_anomalies.append({
                    "agent_id": agent_id,
                    "theta": theta,
                    "task_id": task_id,
                    "b": b,
                    "predicted_prob": predicted_prob,
                    "actual_result": actual_result
                })

            # Type B: Low-ability agent passes hard task
            if (theta < low_ability_threshold and
                b > hard_difficulty_threshold and
                actual_result == 1 and
                predicted_prob < low_prob_threshold):
                type_b_anomalies.append({
                    "agent_id": agent_id,
                    "theta": theta,
                    "task_id": task_id,
                    "b": b,
                    "predicted_prob": predicted_prob,
                    "actual_result": actual_result
                })

    return type_a_anomalies, type_b_anomalies


def group_by_task(anomalies):
    """Group anomalies by task_id."""
    by_task = defaultdict(list)
    for a in anomalies:
        by_task[a["task_id"]].append(a)
    return dict(sorted(by_task.items(), key=lambda x: -len(x[1])))


def print_anomalies(type_a, type_b, items, abilities):
    """Print anomaly analysis results."""

    print("=" * 80)
    print("IRT ANOMALY ANALYSIS")
    print("=" * 80)
    print()

    # Summary stats
    print("SUMMARY")
    print("-" * 40)
    print(f"Total agents: {len(abilities)}")
    print(f"Total tasks: {len(items)}")
    print(f"Ability (theta) range: {abilities['theta'].min():.2f} to {abilities['theta'].max():.2f}")
    print(f"Difficulty (b) range: {items['b'].min():.2f} to {items['b'].max():.2f}")
    print()
    print(f"Type A anomalies (smart fails easy): {len(type_a)}")
    print(f"Type B anomalies (dumb passes hard): {len(type_b)}")
    print()

    # Type A: Smart models failing easy problems
    print("=" * 80)
    print("TYPE A: HIGH-ABILITY AGENTS FAILING EASY TASKS")
    print("These tasks might be broken, flaky, or have environment issues")
    print("=" * 80)
    print()

    type_a_by_task = group_by_task(type_a)
    if type_a_by_task:
        for task_id, anomalies in type_a_by_task.items():
            b = items.loc[task_id, "b"]
            print(f"TASK: {task_id}")
            print(f"  Difficulty (b): {b:.2f} (lower = easier)")
            print(f"  Failed by {len(anomalies)} high-ability agents:")
            for a in sorted(anomalies, key=lambda x: -x["theta"]):
                agent_short = a["agent_id"].split("_", 1)[1] if "_" in a["agent_id"] else a["agent_id"]
                print(f"    - {agent_short[:50]:<50} theta={a['theta']:.2f}  P(pass)={a['predicted_prob']:.1%}")
            print()
    else:
        print("No Type A anomalies found with current thresholds.")
        print()

    # Type B: Dumb models passing hard problems
    print("=" * 80)
    print("TYPE B: LOW-ABILITY AGENTS PASSING HARD TASKS")
    print("These tasks might have test gaps or simple pattern-matching solutions")
    print("=" * 80)
    print()

    type_b_by_task = group_by_task(type_b)
    if type_b_by_task:
        for task_id, anomalies in type_b_by_task.items():
            b = items.loc[task_id, "b"]
            print(f"TASK: {task_id}")
            print(f"  Difficulty (b): {b:.2f} (higher = harder)")
            print(f"  Passed by {len(anomalies)} low-ability agents:")
            for a in sorted(anomalies, key=lambda x: x["theta"]):
                agent_short = a["agent_id"].split("_", 1)[1] if "_" in a["agent_id"] else a["agent_id"]
                print(f"    - {agent_short[:50]:<50} theta={a['theta']:.2f}  P(pass)={a['predicted_prob']:.1%}")
            print()
    else:
        print("No Type B anomalies found with current thresholds.")
        print()


def main():
    parser = argparse.ArgumentParser(description="IRT Anomaly Analysis")
    parser.add_argument("--data-dir", type=Path,
                       default=Path("clean_data/swebench_verified_20251120_full"),
                       help="Directory containing IRT model outputs")
    parser.add_argument("--high-ability", type=float, default=2.0,
                       help="Threshold for high-ability agents (default: 2.0)")
    parser.add_argument("--low-ability", type=float, default=-1.0,
                       help="Threshold for low-ability agents (default: -1.0)")
    parser.add_argument("--easy-difficulty", type=float, default=-2.5,
                       help="Threshold for easy tasks (default: -2.5)")
    parser.add_argument("--hard-difficulty", type=float, default=4.0,
                       help="Threshold for hard tasks (default: 4.0)")
    parser.add_argument("--high-prob", type=float, default=0.95,
                       help="Expected probability threshold for Type A (default: 0.95)")
    parser.add_argument("--low-prob", type=float, default=0.05,
                       help="Threshold for Type B (default: 0.05)")

    args = parser.parse_args()

    print(f"Loading data from {args.data_dir}...")
    abilities, items, responses = load_data(args.data_dir)

    print(f"Finding anomalies...")
    print(f"  High ability threshold: theta > {args.high_ability}")
    print(f"  Low ability threshold: theta < {args.low_ability}")
    print(f"  Easy difficulty threshold: b < {args.easy_difficulty}")
    print(f"  Hard difficulty threshold: b > {args.hard_difficulty}")
    print()

    type_a, type_b = find_anomalies(
        abilities, items, responses,
        high_ability_threshold=args.high_ability,
        low_ability_threshold=args.low_ability,
        easy_difficulty_threshold=args.easy_difficulty,
        hard_difficulty_threshold=args.hard_difficulty,
        high_prob_threshold=args.high_prob,
        low_prob_threshold=args.low_prob
    )

    print_anomalies(type_a, type_b, items, abilities)


if __name__ == "__main__":
    main()

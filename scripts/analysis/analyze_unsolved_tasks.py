#!/usr/bin/env python3
"""
Analyze SWE-bench Verified dataset to find:
1. Tasks completely unsolved by any agent
2. Optimal ability threshold to maintain 100+ unsolved tasks
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

# Load data
responses_path = Path("clean_data/swebench_verified/swebench_verified_20250930_full.jsonl")
abilities_path = Path("clean_data/swebench_verified_20250930_full/1d/abilities.csv")
items_path = Path("clean_data/swebench_verified_20250930_full/1d/items.csv")

# Load abilities and items
abilities = pd.read_csv(abilities_path, index_col=0)
items = pd.read_csv(items_path, index_col=0)

# Load responses
response_matrix = {}
with open(responses_path) as f:
    for line in f:
        data = json.loads(line)
        agent = data['subject_id']
        response_matrix[agent] = data['responses']

# Convert to DataFrame for easier analysis
df = pd.DataFrame(response_matrix).T  # Rows=agents, Cols=tasks
df = df.fillna(0).astype(int)

print("=" * 80)
print("OVERALL STATISTICS")
print("=" * 80)
print(f"Total agents: {len(df)}")
print(f"Total tasks: {len(df.columns)}")
print(f"Total observations: {len(df) * len(df.columns)}")

# 1. Find completely unsolved tasks
unsolved_counts = (df == 0).sum(axis=0)  # Count agents that failed each task
completely_unsolved = unsolved_counts[unsolved_counts == len(df)]

print("\n" + "=" * 80)
print("COMPLETELY UNSOLVED TASKS (by all 123 agents)")
print("=" * 80)
print(f"Number of completely unsolved tasks: {len(completely_unsolved)}")

if len(completely_unsolved) > 0:
    print("\nTask IDs with difficulty scores:")
    for task_id in completely_unsolved.index:
        difficulty = items.loc[task_id, 'b']
        discrimination = items.loc[task_id, 'a']
        print(f"  {task_id}: b={difficulty:.3f}, a={discrimination:.3f}")
else:
    print("All tasks were solved by at least one agent.")

# 2. Find optimal threshold for 100+ unsolved tasks
print("\n" + "=" * 80)
print("THRESHOLD ANALYSIS FOR UNSOLVED TASKS")
print("=" * 80)
print("Finding ability threshold where ≥100 tasks remain unsolved\n")

# Sort agents by ability
abilities_sorted = abilities.sort_values('theta', ascending=True)

# Try different thresholds
results = []
for i in range(len(abilities_sorted)):
    threshold = abilities_sorted.iloc[i]['theta']
    agent_name = abilities_sorted.index[i]

    # Get agents below or at this threshold
    agents_below = abilities[abilities['theta'] <= threshold].index.tolist()

    # Count tasks unsolved by ALL agents below threshold
    if len(agents_below) > 0:
        subset_df = df.loc[agents_below]
        unsolved_by_subset = (subset_df == 0).all(axis=0).sum()
    else:
        unsolved_by_subset = len(df.columns)  # All tasks unsolved if no agents

    results.append({
        'threshold': threshold,
        'agent': agent_name,
        'n_agents': len(agents_below),
        'unsolved_tasks': unsolved_by_subset
    })

results_df = pd.DataFrame(results)

# Find first threshold where we have at least 100 unsolved tasks
target_unsolved = 100
valid_thresholds = results_df[results_df['unsolved_tasks'] >= target_unsolved]

if len(valid_thresholds) > 0:
    # Get the highest threshold that still gives us 100+ unsolved tasks
    optimal = valid_thresholds.loc[valid_thresholds['threshold'].idxmax()]

    print(f"Optimal threshold: θ ≤ {optimal['threshold']:.3f}")
    print(f"Number of agents included: {optimal['n_agents']}")
    print(f"Unsolved tasks at this threshold: {optimal['unsolved_tasks']}")
    print(f"Last agent included: {optimal['agent']}")

    print("\n" + "-" * 80)
    print("AGENTS INCLUDED (sorted by ability):")
    print("-" * 80)
    agents_included = abilities[abilities['theta'] <= optimal['threshold']].sort_values('theta', ascending=False)
    for idx, (agent, row) in enumerate(agents_included.iterrows(), 1):
        print(f"{idx:3d}. {agent:50s} θ={row['theta']:7.3f} (±{row['theta_std']:.3f})")

    # Show some example unsolved tasks
    agents_below = abilities[abilities['theta'] <= optimal['threshold']].index.tolist()
    subset_df = df.loc[agents_below]
    unsolved_mask = (subset_df == 0).all(axis=0)
    unsolved_task_ids = subset_df.columns[unsolved_mask].tolist()

    print("\n" + "-" * 80)
    print(f"EXAMPLE UNSOLVED TASKS (showing first 20 of {len(unsolved_task_ids)}):")
    print("-" * 80)
    for task_id in unsolved_task_ids[:20]:
        difficulty = items.loc[task_id, 'b']
        discrimination = items.loc[task_id, 'a']
        print(f"  {task_id:50s} b={difficulty:6.3f}, a={discrimination:.3f}")

    # Show distribution of difficulty for unsolved tasks
    unsolved_difficulties = items.loc[unsolved_task_ids, 'b']
    print("\n" + "-" * 80)
    print("DIFFICULTY DISTRIBUTION OF UNSOLVED TASKS:")
    print("-" * 80)
    print(f"Mean difficulty: {unsolved_difficulties.mean():.3f}")
    print(f"Std difficulty: {unsolved_difficulties.std():.3f}")
    print(f"Min difficulty: {unsolved_difficulties.min():.3f}")
    print(f"Max difficulty: {unsolved_difficulties.max():.3f}")
    print(f"Median difficulty: {unsolved_difficulties.median():.3f}")

    print("\nDifficulty quartiles:")
    print(unsolved_difficulties.quantile([0.25, 0.5, 0.75]))

else:
    print(f"No threshold found that gives at least {target_unsolved} unsolved tasks")
    print("\nShowing thresholds with most unsolved tasks:")
    print(results_df.nlargest(10, 'unsolved_tasks')[['threshold', 'n_agents', 'unsolved_tasks', 'agent']])

# Show a progression table
print("\n" + "=" * 80)
print("THRESHOLD PROGRESSION TABLE")
print("=" * 80)
print(f"{'Ability Threshold':>18} | {'# Agents':>10} | {'Unsolved Tasks':>15}")
print("-" * 80)

# Show every 10th threshold
for i in range(0, len(results_df), 10):
    row = results_df.iloc[i]
    print(f"{row['threshold']:18.3f} | {row['n_agents']:10d} | {row['unsolved_tasks']:15d}")

# Add the last one
if len(results_df) % 10 != 0:
    row = results_df.iloc[-1]
    print(f"{row['threshold']:18.3f} | {row['n_agents']:10d} | {row['unsolved_tasks']:15d}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

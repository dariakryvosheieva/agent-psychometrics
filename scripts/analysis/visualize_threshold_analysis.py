#!/usr/bin/env python3
"""
Visualize the relationship between ability threshold and unsolved tasks.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load data
responses_path = Path("clean_data/swebench_verified/swebench_verified_20250930_full.jsonl")
abilities_path = Path("clean_data/swebench_verified_20250930_full/1d/abilities.csv")
items_path = Path("clean_data/swebench_verified_20250930_full/1d/items.csv")

abilities = pd.read_csv(abilities_path, index_col=0)
items = pd.read_csv(items_path, index_col=0)

# Load responses
response_matrix = {}
with open(responses_path) as f:
    for line in f:
        data = json.loads(line)
        agent = data['subject_id']
        response_matrix[agent] = data['responses']

df = pd.DataFrame(response_matrix).T
df = df.fillna(0).astype(int)

# Calculate threshold progression
abilities_sorted = abilities.sort_values('theta', ascending=True)
results = []
for i in range(len(abilities_sorted)):
    threshold = abilities_sorted.iloc[i]['theta']
    agent_name = abilities_sorted.index[i]
    agents_below = abilities[abilities['theta'] <= threshold].index.tolist()

    if len(agents_below) > 0:
        subset_df = df.loc[agents_below]
        unsolved_by_subset = (subset_df == 0).all(axis=0).sum()
    else:
        unsolved_by_subset = len(df.columns)

    results.append({
        'threshold': threshold,
        'agent': agent_name,
        'n_agents': len(agents_below),
        'unsolved_tasks': unsolved_by_subset
    })

results_df = pd.DataFrame(results)

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('SWE-bench Verified: Unsolved Tasks Analysis', fontsize=16, fontweight='bold')

# Plot 1: Unsolved tasks vs ability threshold
ax1 = axes[0, 0]
ax1.plot(results_df['threshold'], results_df['unsolved_tasks'], 'b-', linewidth=2)
ax1.axhline(y=100, color='r', linestyle='--', linewidth=2, label='Target: 100 unsolved tasks')

# Find optimal threshold
optimal = results_df[results_df['unsolved_tasks'] >= 100].loc[results_df[results_df['unsolved_tasks'] >= 100]['threshold'].idxmax()]
ax1.axvline(x=optimal['threshold'], color='g', linestyle='--', linewidth=2,
            label=f'Optimal: θ={optimal["threshold"]:.3f}')
ax1.scatter([optimal['threshold']], [optimal['unsolved_tasks']], color='red', s=100, zorder=5)

ax1.set_xlabel('Ability Threshold (θ)', fontsize=12)
ax1.set_ylabel('Number of Unsolved Tasks', fontsize=12)
ax1.set_title('Unsolved Tasks vs Ability Threshold', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: Number of agents vs ability threshold
ax2 = axes[0, 1]
ax2.plot(results_df['threshold'], results_df['n_agents'], 'g-', linewidth=2)
ax2.axvline(x=optimal['threshold'], color='r', linestyle='--', linewidth=2,
            label=f'Optimal: {optimal["n_agents"]} agents')
ax2.scatter([optimal['threshold']], [optimal['n_agents']], color='red', s=100, zorder=5)

ax2.set_xlabel('Ability Threshold (θ)', fontsize=12)
ax2.set_ylabel('Number of Agents Included', fontsize=12)
ax2.set_title('Agents Included vs Ability Threshold', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Plot 3: Unsolved tasks vs number of agents
ax3 = axes[1, 0]
ax3.plot(results_df['n_agents'], results_df['unsolved_tasks'], 'purple', linewidth=2)
ax3.axhline(y=100, color='r', linestyle='--', linewidth=2, alpha=0.5)
ax3.scatter([optimal['n_agents']], [optimal['unsolved_tasks']], color='red', s=100, zorder=5,
            label=f'Optimal: {optimal["n_agents"]} agents, {optimal["unsolved_tasks"]} tasks')

ax3.set_xlabel('Number of Agents', fontsize=12)
ax3.set_ylabel('Number of Unsolved Tasks', fontsize=12)
ax3.set_title('Unsolved Tasks vs Number of Agents', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend()

# Plot 4: Difficulty distribution of unsolved tasks at optimal threshold
ax4 = axes[1, 1]
agents_below = abilities[abilities['theta'] <= optimal['threshold']].index.tolist()
subset_df = df.loc[agents_below]
unsolved_mask = (subset_df == 0).all(axis=0)
unsolved_task_ids = subset_df.columns[unsolved_mask].tolist()
unsolved_difficulties = items.loc[unsolved_task_ids, 'b']

ax4.hist(unsolved_difficulties, bins=30, color='orange', alpha=0.7, edgecolor='black')
ax4.axvline(x=unsolved_difficulties.mean(), color='r', linestyle='--', linewidth=2,
            label=f'Mean: {unsolved_difficulties.mean():.2f}')
ax4.axvline(x=unsolved_difficulties.median(), color='g', linestyle='--', linewidth=2,
            label=f'Median: {unsolved_difficulties.median():.2f}')

ax4.set_xlabel('Task Difficulty (b)', fontsize=12)
ax4.set_ylabel('Frequency', fontsize=12)
ax4.set_title(f'Difficulty Distribution of {len(unsolved_task_ids)} Unsolved Tasks\n(at θ ≤ {optimal["threshold"]:.3f})',
              fontsize=13, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
output_path = Path("chris_output/figures/unsolved_tasks_threshold_analysis.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {output_path}")

# Create a second figure showing the detailed breakdown
fig2, ax = plt.subplots(figsize=(16, 8))

# Get agents at optimal threshold
agents_at_threshold = abilities[abilities['theta'] <= optimal['threshold']].sort_values('theta', ascending=False)

# Color code by whether they're included
colors = ['green' if theta <= optimal['threshold'] else 'lightgray'
          for theta in abilities.sort_values('theta', ascending=False)['theta']]

# Plot all agents
y_pos = np.arange(len(abilities))
abilities_plot = abilities.sort_values('theta', ascending=False)
ax.barh(y_pos, abilities_plot['theta'], color=colors, alpha=0.6, edgecolor='black', linewidth=0.5)

# Add threshold line
ax.axvline(x=optimal['threshold'], color='red', linestyle='--', linewidth=3,
           label=f'Threshold: θ = {optimal["threshold"]:.3f}\n({optimal["n_agents"]} agents, {optimal["unsolved_tasks"]} unsolved tasks)')

ax.set_xlabel('Ability (θ)', fontsize=14, fontweight='bold')
ax.set_ylabel('Agent Index (sorted by ability)', fontsize=14, fontweight='bold')
ax.set_title('Agent Abilities with Optimal Threshold for 100+ Unsolved Tasks', fontsize=16, fontweight='bold')
ax.legend(fontsize=12, loc='lower right')
ax.grid(True, alpha=0.3, axis='x')

# Add annotations
ax.text(optimal['threshold'] - 0.5, 110, f'{optimal["n_agents"]} agents\nincluded',
        fontsize=11, ha='right', va='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
ax.text(optimal['threshold'] + 0.5, 110, f'{123 - optimal["n_agents"]} agents\nexcluded',
        fontsize=11, ha='left', va='center', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

plt.tight_layout()
output_path2 = Path("chris_output/figures/agent_abilities_threshold.png")
plt.savefig(output_path2, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {output_path2}")

plt.show()

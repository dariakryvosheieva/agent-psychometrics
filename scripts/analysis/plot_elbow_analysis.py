#!/usr/bin/env python3
"""
Create a focused elbow plot for unsolved tasks vs number of agents.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import UnivariateSpline

# Load data
responses_path = Path("clean_data/swebench_verified/swebench_verified_20250930_full.jsonl")
abilities_path = Path("clean_data/swebench_verified_20250930_full/1d/abilities.csv")

abilities = pd.read_csv(abilities_path, index_col=0)

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

# Find optimal threshold that maximizes agents * unsolved_tasks
results_df['product'] = results_df['n_agents'] * results_df['unsolved_tasks']
optimal = results_df.loc[results_df['product'].idxmax()]

# Calculate second derivative to find elbow
# Use smoothing spline for better derivative estimation
x = results_df['n_agents'].values
y = results_df['unsolved_tasks'].values

# Fit spline
spline = UnivariateSpline(x, y, s=50)  # s is smoothing factor
x_smooth = np.linspace(x.min(), x.max(), 500)
y_smooth = spline(x_smooth)

# Calculate derivatives
dy_dx = spline.derivative(1)(x_smooth)
d2y_dx2 = spline.derivative(2)(x_smooth)

# Find elbow using maximum curvature
curvature = np.abs(d2y_dx2) / (1 + dy_dx**2)**1.5
elbow_idx = np.argmax(curvature)
elbow_x = x_smooth[elbow_idx]
elbow_y = y_smooth[elbow_idx]

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Main plot: Unsolved Tasks vs Number of Agents
ax1 = axes[0]
ax1.plot(results_df['n_agents'], results_df['unsolved_tasks'], 'b-', linewidth=3)

# Mark the optimal point (max product)
ax1.scatter([optimal['n_agents']], [optimal['unsolved_tasks']], color='red', s=200, zorder=5,
            marker='o', edgecolor='black', linewidth=2)

ax1.set_xlabel('Number of Agents Included', fontsize=14, fontweight='bold')
ax1.set_ylabel('Number of Unsolved Tasks', fontsize=14, fontweight='bold')
ax1.set_title('Unsolved Tasks vs Number of Agents', fontsize=16, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 125)
ax1.set_ylim(0, 510)

# Second plot: Product (agents * unsolved_tasks)
ax2 = axes[1]
ax2.plot(results_df['n_agents'], results_df['product'], 'g-', linewidth=3)
ax2.scatter([optimal['n_agents']], [optimal['product']], color='red', s=200, zorder=5,
            marker='o', edgecolor='black', linewidth=2)

ax2.set_xlabel('Number of Agents Included', fontsize=14, fontweight='bold')
ax2.set_ylabel('Agents × Unsolved Tasks', fontsize=14, fontweight='bold')
ax2.set_title('Product: Agents × Unsolved Tasks', fontsize=16, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 125)

plt.tight_layout()
output_path = Path("chris_output/figures/elbow_analysis.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {output_path}")

# Print numerical analysis
print("\n" + "="*80)
print("OPTIMAL THRESHOLD (Maximizing Agents × Unsolved Tasks)")
print("="*80)
print(f"\nOptimal point:")
print(f"  Number of agents: {optimal['n_agents']}")
print(f"  Unsolved tasks: {optimal['unsolved_tasks']}")
print(f"  Product (agents × tasks): {optimal['product']:.0f}")
print(f"  Ability threshold: θ ≤ {optimal['threshold']:.3f}")
print(f"  Last agent included: {optimal['agent']}")

plt.show()

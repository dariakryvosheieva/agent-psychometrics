"""
Plot the frontier of model abilities over time.

X-axis: Date (agent submission date from YYYYMMDD prefix)
Y-axis: Highest IRT ability score of any agent submitted before that date

Note: Uses agent submission dates as proxy for "when capability was available",
since this captures both model improvements AND scaffolding improvements.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

# Load IRT abilities
ROOT = Path(__file__).resolve().parents[2]
abilities = pd.read_csv(
    ROOT / "data/swebench_verified/irt/1d_1pl/abilities.csv",
    index_col=0
)


def extract_submission_date(agent_name: str) -> datetime | None:
    """Extract the submission date from agent name (YYYYMMDD prefix)."""
    try:
        date_str = agent_name[:8]
        return datetime.strptime(date_str, "%Y%m%d")
    except (ValueError, IndexError):
        return None


def get_short_name(agent_name: str) -> str:
    """Get a shortened display name for an agent."""
    # Remove date prefix
    name = agent_name[9:] if len(agent_name) > 9 else agent_name
    # Truncate if too long
    if len(name) > 35:
        name = name[:32] + "..."
    return name


# Build dataframe with submission dates
agent_data = []
for agent_name in abilities.index:
    theta = abilities.loc[agent_name, "theta"]
    submission_date = extract_submission_date(agent_name)

    if submission_date:
        agent_data.append({
            "agent": agent_name,
            "short_name": get_short_name(agent_name),
            "theta": theta,
            "submission_date": submission_date,
        })
    else:
        print(f"Could not extract date from agent: {agent_name}")

df = pd.DataFrame(agent_data)
print(f"Parsed {len(df)} agents with submission dates")

# Sort by submission date
df = df.sort_values("submission_date")

# For each unique date, find the maximum ability of agents submitted on or before that date
df_grouped = df.groupby("submission_date").agg({
    "theta": "max",
    "short_name": lambda x: list(x),
    "agent": lambda x: list(x),
}).reset_index()

df_grouped = df_grouped.sort_values("submission_date")
df_grouped["frontier_theta"] = df_grouped["theta"].cummax()

# Track which agent achieved the frontier
frontier_agents = []
current_max = float("-inf")
current_agent = None
for _, row in df_grouped.iterrows():
    if row["theta"] > current_max:
        current_max = row["theta"]
        # Find the agent with max theta on this date
        for agent, name in zip(row["agent"], row["short_name"]):
            if abilities.loc[agent, "theta"] == row["theta"]:
                current_agent = name
                break
    frontier_agents.append(current_agent)
df_grouped["frontier_agent"] = frontier_agents

print("\n" + "="*80)
print("Frontier progression:")
print("="*80)
prev_frontier = None
for _, row in df_grouped.iterrows():
    if row["frontier_theta"] != prev_frontier:
        print(f"{row['submission_date'].strftime('%Y-%m-%d')}: θ = {row['frontier_theta']:.3f} ({row['frontier_agent']})")
        prev_frontier = row["frontier_theta"]

# Create the plot
fig, ax = plt.subplots(figsize=(14, 8))

# Plot all individual agents as faded points
ax.scatter(df["submission_date"], df["theta"],
           alpha=0.35, s=40, color="#94a3b8", label="Individual agents", zorder=2)

# Plot the frontier as a step function - make it very prominent
ax.step(df_grouped["submission_date"], df_grouped["frontier_theta"],
        where="post", linewidth=3.5, color="#1d4ed8", label="Frontier ability", zorder=4)

# Mark each frontier improvement with a larger point
frontier_changes = df_grouped[df_grouped["frontier_theta"].diff().fillna(1) > 0].copy()
ax.scatter(frontier_changes["submission_date"], frontier_changes["frontier_theta"],
           s=120, color="#1d4ed8", zorder=5, edgecolors="white", linewidths=2)

# Compute linear regression on frontier data
from scipy import stats

# Convert dates to numeric (days since first date)
frontier_dates = frontier_changes["submission_date"]
first_date = frontier_dates.min()
frontier_x = np.array([(d - first_date).days for d in frontier_dates])
frontier_y = frontier_changes["frontier_theta"].values

# Linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(frontier_x, frontier_y)

print(f"\nLinear regression on frontier:")
print(f"  Slope: {slope:.4f} θ/day = {slope * 365:.2f} θ/year")
print(f"  Intercept: {intercept:.3f}")
print(f"  R-value: {r_value:.4f}")
print(f"  R²: {r_value**2:.4f}")
print(f"  p-value: {p_value:.2e}")

# Plot the trendline
trendline_x = np.array([frontier_x.min(), frontier_x.max()])
trendline_y = slope * trendline_x + intercept
trendline_dates = [first_date + pd.Timedelta(days=int(x)) for x in trendline_x]
ax.plot(trendline_dates, trendline_y,
        linewidth=2.5, color="#dc2626", linestyle="--",
        label=f"Trendline (r={r_value:.3f})", zorder=3)

# Add labels for frontier agents (only label significant jumps to avoid clutter)
min_jump = 0.5  # Only label jumps > 0.5 theta
frontier_changes["jump"] = frontier_changes["frontier_theta"].diff().fillna(10)
significant_changes = frontier_changes[frontier_changes["jump"] > min_jump]

for i, (_, row) in enumerate(significant_changes.iterrows()):
    # Alternate label positions to reduce overlap
    offset_y = 18 if i % 2 == 0 else -28
    ax.annotate(row["frontier_agent"],
                (row["submission_date"], row["frontier_theta"]),
                textcoords="offset points",
                xytext=(10, offset_y),
                fontsize=9,
                fontweight="bold",
                alpha=0.9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.7),
                arrowprops=dict(arrowstyle="-", alpha=0.4, color="gray") if abs(offset_y) > 18 else None)

# Add annotation with regression stats
stats_text = f"Frontier trend: {slope * 365:.2f} θ/year\nr = {r_value:.3f}, R² = {r_value**2:.3f}"
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))

# Formatting
ax.set_xlabel("Agent Submission Date", fontsize=12)
ax.set_ylabel("IRT Ability (θ)", fontsize=12)
ax.set_title("Frontier of SWE-bench Agent Ability Over Time\n(1D 2PL IRT on SWE-bench Verified, 123 agents, 500 tasks)", fontsize=14)

# Format x-axis dates
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45, ha="right")

# Add grid
ax.grid(True, alpha=0.3, linestyle="--")
ax.set_axisbelow(True)

# Set y-axis limits with some padding
y_min, y_max = df["theta"].min(), df["theta"].max()
ax.set_ylim(y_min - 0.5, y_max + 0.5)

# Legend
ax.legend(loc="lower right", fontsize=10)

# Tight layout
plt.tight_layout()

# Save
output_dir = ROOT / "output/figures"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "frontier_ability_over_time.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\nSaved plot to: {output_path}")

# Also save as PDF for paper quality
output_pdf = output_dir / "frontier_ability_over_time.pdf"
plt.savefig(output_pdf, bbox_inches="tight")
print(f"Saved PDF to: {output_pdf}")

plt.show()

"""
Plot task first-solve date vs task difficulty.

For each task, finds the earliest submission date of any agent that solved it,
then plots this against the task's IRT difficulty (β).

This provides a complementary view to the frontier_ability_over_time plot:
- That plot: How capable is the best agent at time T?
- This plot: When was a task with difficulty β first solved?
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path
import numpy as np
from scipy import stats

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
RESPONSE_MATRIX = BASE_DIR / "data/swebench_verified/responses.jsonl"
ITEMS_CSV = BASE_DIR / "data/swebench_verified/irt/1d_1pl/items.csv"
OUTPUT_DIR = BASE_DIR / "output/figures"


def extract_submission_date(agent_name: str) -> datetime | None:
    """Extract the submission date from agent name (YYYYMMDD prefix)."""
    try:
        date_str = agent_name[:8]
        return datetime.strptime(date_str, "%Y%m%d")
    except (ValueError, IndexError):
        return None


def load_response_matrix(path: Path) -> dict[str, dict[str, int]]:
    """Load response matrix from JSONL file.

    Returns: dict mapping agent_name -> {task_id: 0|1}
    """
    responses = {}
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            agent_name = data["subject_id"]
            responses[agent_name] = data["responses"]
    return responses


def compute_first_solve_dates(responses: dict[str, dict[str, int]]) -> dict[str, datetime]:
    """Compute the earliest solve date for each task.

    Returns: dict mapping task_id -> first_solve_date (only for solved tasks)
    """
    # Build agent -> submission_date mapping
    agent_dates = {}
    for agent_name in responses:
        date = extract_submission_date(agent_name)
        if date:
            agent_dates[agent_name] = date
        else:
            print(f"Could not extract date from agent: {agent_name}")

    # For each task, find earliest solving agent
    first_solve = {}
    all_tasks = set()

    for agent_name, task_responses in responses.items():
        if agent_name not in agent_dates:
            continue
        agent_date = agent_dates[agent_name]

        for task_id, solved in task_responses.items():
            all_tasks.add(task_id)
            if solved == 1:
                if task_id not in first_solve or agent_date < first_solve[task_id]:
                    first_solve[task_id] = agent_date

    print(f"Total tasks: {len(all_tasks)}")
    print(f"Tasks solved by at least one agent: {len(first_solve)}")
    print(f"Tasks never solved: {len(all_tasks) - len(first_solve)}")

    return first_solve


def main():
    print("Loading response matrix...")
    responses = load_response_matrix(RESPONSE_MATRIX)
    print(f"Loaded {len(responses)} agents")

    print("\nLoading task difficulties...")
    items = pd.read_csv(ITEMS_CSV, index_col=0)
    print(f"Loaded {len(items)} tasks with IRT parameters")

    print("\nComputing first-solve dates...")
    first_solve = compute_first_solve_dates(responses)

    # Build dataframe for plotting
    task_data = []
    for task_id, solve_date in first_solve.items():
        if task_id in items.index:
            difficulty = items.loc[task_id, "b"]  # β (difficulty parameter)
            task_data.append({
                "task_id": task_id,
                "first_solve_date": solve_date,
                "difficulty": difficulty,
            })

    df = pd.DataFrame(task_data)
    print(f"\nTasks with both solve date and difficulty: {len(df)}")

    # Convert dates to days since first solve
    first_date = df["first_solve_date"].min()
    df["days_since_first"] = (df["first_solve_date"] - first_date).dt.days

    print(f"Date range: {first_date.strftime('%Y-%m-%d')} to {df['first_solve_date'].max().strftime('%Y-%m-%d')}")
    print(f"Difficulty range: {df['difficulty'].min():.2f} to {df['difficulty'].max():.2f}")

    # Linear regression: first_solve_days ~ difficulty
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df["difficulty"], df["days_since_first"]
    )

    print(f"\nLinear regression (first_solve_days ~ difficulty):")
    print(f"  Slope: {slope:.2f} days/θ")
    print(f"  Intercept: {intercept:.2f} days")
    print(f"  R-value: {r_value:.4f}")
    print(f"  R²: {r_value**2:.4f}")
    print(f"  p-value: {p_value:.2e}")

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Scatter plot
    scatter = ax.scatter(
        df["difficulty"],
        df["first_solve_date"],
        alpha=0.5,
        s=30,
        color="#2563eb",
        edgecolors="white",
        linewidths=0.5,
    )

    # Trendline
    x_range = np.array([df["difficulty"].min(), df["difficulty"].max()])
    y_days = slope * x_range + intercept
    y_dates = [first_date + pd.Timedelta(days=int(d)) for d in y_days]
    ax.plot(x_range, y_dates, linewidth=2.5, color="#dc2626", linestyle="--",
            label=f"Trendline (r={r_value:.3f})")

    # Add annotation with regression stats
    stats_text = f"First-solve trend: {slope:.1f} days/θ\nr = {r_value:.3f}, R² = {r_value**2:.3f}\np = {p_value:.2e}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))

    # Formatting
    ax.set_xlabel("Task Difficulty (β)", fontsize=12)
    ax.set_ylabel("First Solve Date", fontsize=12)
    ax.set_title(f"Task First-Solve Date vs IRT Difficulty\n(SWE-bench Verified, {len(df)} tasks)", fontsize=14)

    # Format y-axis dates
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.yaxis.set_major_locator(mdates.MonthLocator(interval=3))

    # Grid
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Legend
    ax.legend(loc="lower right", fontsize=10)

    plt.tight_layout()

    # Save
    output_png = OUTPUT_DIR / "first_solve_date_vs_difficulty.png"
    output_pdf = OUTPUT_DIR / "first_solve_date_vs_difficulty.pdf"

    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to: {output_png}")

    plt.savefig(output_pdf, bbox_inches="tight")
    print(f"Saved PDF to: {output_pdf}")

    plt.show()


if __name__ == "__main__":
    main()

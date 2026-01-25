"""Date forecasting utilities for experiment B.

Predicts when tasks will become solvable (50% probability) based on
the linear relationship between frontier ability and time.

Key insight: From IRT, P(success) = sigmoid(theta - beta) = 0.5 when theta = beta.
So a task is solvable with 50% probability when an agent's ability >= task difficulty.
Combined with Experiment D's finding that frontier ability is linear over time,
we can predict when a task will become solvable.

Approach:
1. Fit frontier ability over time: theta = slope * days + intercept
2. Invert to predict solvability: days = (beta - intercept) / slope
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def parse_date(date_str: str) -> datetime:
    """Parse YYYYMMDD date string to datetime."""
    return datetime.strptime(date_str, "%Y%m%d")


@dataclass
class FirstCapableDatesResult:
    """Result from compute_first_capable_dates().

    Attributes:
        first_capable_dates: Dict mapping task_id -> datetime of earliest capable agent.
            Tasks with NO capable agent are NOT included in this dict.
        tasks_without_capable_agent: List of task_ids where no agent has theta >= beta.
            These tasks are "too hard" for any current agent to solve with 50% prob.
        earliest_agent_date: Earliest agent submission date in the dataset.
        latest_agent_date: Latest agent submission date in the dataset.
    """

    first_capable_dates: Dict[str, datetime]
    tasks_without_capable_agent: List[str]
    earliest_agent_date: datetime
    latest_agent_date: datetime


def compute_first_capable_dates(
    oracle_items: pd.DataFrame,
    oracle_abilities: pd.DataFrame,
    agent_dates: Dict[str, str],
    solve_probability: float = 0.5,
) -> FirstCapableDatesResult:
    """Compute the first date when each task became solvable with given probability.

    A task is solvable with probability p when theta_agent >= beta_task + logit(p),
    where logit(p) = log(p / (1-p)). For p=0.5, this simplifies to theta >= beta.

    NOTE: This uses oracle IRT values and is only used for computing GROUND TRUTH
    for evaluation. It should NOT be used for training predictors.

    Args:
        oracle_items: DataFrame with 'b' column (oracle task difficulties)
        oracle_abilities: DataFrame with 'theta' column (oracle agent abilities)
        agent_dates: Dict mapping agent_id -> date string (YYYYMMDD)
        solve_probability: Probability threshold for considering an agent "capable"
            of solving a task (default 0.5, i.e., 50% solve rate)

    Returns:
        FirstCapableDatesResult with:
            - first_capable_dates: task_id -> datetime (only tasks WITH capable agents)
            - tasks_without_capable_agent: list of task_ids with NO capable agent
            - earliest_agent_date, latest_agent_date: date range of agents

    Raises:
        ValueError: If required data columns are missing or no agents have dates
    """
    if "b" not in oracle_items.columns:
        raise ValueError("oracle_items must have 'b' column")
    if "theta" not in oracle_abilities.columns:
        raise ValueError("oracle_abilities must have 'theta' column")

    # Build agent -> (theta, date) mapping
    agent_info = {}
    for agent_id in oracle_abilities.index:
        if agent_id not in agent_dates:
            continue
        theta = oracle_abilities.loc[agent_id, "theta"]
        date = parse_date(agent_dates[agent_id])
        agent_info[agent_id] = (theta, date)

    if not agent_info:
        raise ValueError("No agents found with both abilities and dates")

    # Get date range
    all_dates = [info[1] for info in agent_info.values()]
    earliest_agent_date = min(all_dates)
    latest_agent_date = max(all_dates)

    # For each task, find earliest agent where theta >= beta + logit(p)
    # logit(p) = log(p / (1-p)), so for p=0.5, logit=0; for p=0.3, logit≈-0.847
    threshold_offset = np.log(solve_probability / (1 - solve_probability))
    first_capable_dates = {}
    tasks_without_capable_agent = []

    for task_id in oracle_items.index:
        beta = oracle_items.loc[task_id, "b"]

        # Find all agents capable of solving with given probability
        # P(solve) = sigmoid(theta - beta) >= p  =>  theta >= beta + logit(p)
        capable_agents = [
            (agent_id, info[1])  # (agent_id, date)
            for agent_id, info in agent_info.items()
            if info[0] >= beta + threshold_offset
        ]

        if capable_agents:
            # Find earliest by date
            earliest = min(capable_agents, key=lambda x: x[1])
            first_capable_dates[task_id] = earliest[1]
        else:
            # No agent can currently solve this task with the given probability
            tasks_without_capable_agent.append(task_id)

    return FirstCapableDatesResult(
        first_capable_dates=first_capable_dates,
        tasks_without_capable_agent=tasks_without_capable_agent,
        earliest_agent_date=earliest_agent_date,
        latest_agent_date=latest_agent_date,
    )


def split_tasks_by_first_capable_date(
    first_capable_dates: Dict[str, datetime],
    cutoff_date: datetime,
) -> Tuple[List[str], List[str]]:
    """Split tasks by whether first capable agent is before/after cutoff.

    Args:
        first_capable_dates: Dict from FirstCapableDatesResult.first_capable_dates
        cutoff_date: Date to split on (e.g., frontier cutoff)

    Returns:
        Tuple of (pre_cutoff_tasks, post_cutoff_tasks):
            - pre_cutoff_tasks: Tasks where first capable agent is before cutoff (for training)
            - post_cutoff_tasks: Tasks where first capable agent is on/after cutoff (for eval)
    """
    pre_cutoff_tasks = []
    post_cutoff_tasks = []

    for task_id, first_date in first_capable_dates.items():
        if first_date < cutoff_date:
            pre_cutoff_tasks.append(task_id)
        else:
            post_cutoff_tasks.append(task_id)

    return pre_cutoff_tasks, post_cutoff_tasks


def compute_ground_truth_days(
    task_ids: List[str],
    first_capable_dates: Dict[str, datetime],
    reference_date: datetime,
) -> Dict[str, float]:
    """Convert dates to days since reference date.

    Args:
        task_ids: List of task IDs to process
        first_capable_dates: Dict from FirstCapableDatesResult.first_capable_dates
        reference_date: Reference date (typically earliest date in dataset)

    Returns:
        Dict mapping task_id -> days since reference.
        Tasks NOT in first_capable_dates are excluded from result.
    """
    result = {}
    for task_id in task_ids:
        if task_id in first_capable_dates:
            delta = first_capable_dates[task_id] - reference_date
            result[task_id] = delta.days
    return result


@dataclass
class AbilityOverTimeResult:
    """Result from fit_ability_over_time().

    Attributes:
        slope: Theta units per day (frontier ability growth rate)
        intercept: Initial frontier ability at reference date
        r_squared: R² of the linear fit
        reference_date: The earliest agent date (day 0)
        n_agents: Number of agents used in fitting
        n_frontier_points: Number of frontier points (days where frontier improved)
    """

    slope: float
    intercept: float
    r_squared: float
    reference_date: datetime
    n_agents: int
    n_frontier_points: int


def fit_ability_over_time(
    abilities: Dict[str, float],
    agent_dates: Dict[str, str],
) -> AbilityOverTimeResult:
    """Fit a linear model of frontier ability over time.

    Follows the approach from Experiment D:
    1. Group agents by date
    2. Compute max ability per date
    3. Compute cumulative max (frontier trajectory)
    4. Fit linear regression on frontier points where ability increased

    Args:
        abilities: Dict mapping agent_id -> theta (ability)
        agent_dates: Dict mapping agent_id -> date string (YYYYMMDD)

    Returns:
        AbilityOverTimeResult with slope, intercept, r_squared, etc.

    Raises:
        ValueError: If insufficient data for fitting
    """
    # Build dataframe of agents with dates
    agent_data = []
    for agent_id, theta in abilities.items():
        if agent_id not in agent_dates:
            continue
        date = parse_date(agent_dates[agent_id])
        agent_data.append({"agent_id": agent_id, "theta": theta, "date": date})

    if len(agent_data) < 3:
        raise ValueError(f"Insufficient agents with dates: {len(agent_data)}")

    df = pd.DataFrame(agent_data)
    df = df.sort_values("date")

    reference_date = df["date"].min()

    # Group by date, take max ability per date
    df_grouped = df.groupby("date").agg({"theta": "max"}).reset_index()
    df_grouped = df_grouped.sort_values("date")

    # Compute cumulative max (frontier trajectory)
    df_grouped["frontier_theta"] = df_grouped["theta"].cummax()

    # Find points where frontier improved
    frontier_changes = df_grouped[df_grouped["frontier_theta"].diff().fillna(1) > 0].copy()

    if len(frontier_changes) < 2:
        raise ValueError(f"Insufficient frontier points: {len(frontier_changes)}")

    # Convert dates to days since reference
    frontier_x = np.array([(d - reference_date).days for d in frontier_changes["date"]])
    frontier_y = frontier_changes["frontier_theta"].values

    # Fit linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(frontier_x, frontier_y)

    return AbilityOverTimeResult(
        slope=float(slope),
        intercept=float(intercept),
        r_squared=float(r_value**2),
        reference_date=reference_date,
        n_agents=len(df),
        n_frontier_points=len(frontier_changes),
    )


class DateForecastModel:
    """Model for predicting solvability dates from difficulties using ability regression.

    Fits: frontier_theta = slope * days + intercept
    Predicts: days = (beta - intercept) / slope
    """

    def __init__(self):
        self._ability_fit: Optional[AbilityOverTimeResult] = None
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def slope(self) -> Optional[float]:
        return self._ability_fit.slope if self._ability_fit else None

    @property
    def intercept(self) -> Optional[float]:
        return self._ability_fit.intercept if self._ability_fit else None

    @property
    def r_squared(self) -> Optional[float]:
        return self._ability_fit.r_squared if self._ability_fit else None

    @property
    def reference_date(self) -> Optional[datetime]:
        return self._ability_fit.reference_date if self._ability_fit else None

    def fit(
        self,
        abilities: Dict[str, float],
        agent_dates: Dict[str, str],
    ) -> Dict[str, float]:
        """Fit the ability-over-time model.

        Args:
            abilities: Dict mapping agent_id -> theta (ability)
            agent_dates: Dict mapping agent_id -> date string (YYYYMMDD)

        Returns:
            Dict with fit statistics (slope, intercept, r_squared, n_agents, n_frontier_points)
        """
        self._ability_fit = fit_ability_over_time(abilities, agent_dates)
        self._is_fitted = True

        return {
            "slope": self._ability_fit.slope,
            "intercept": self._ability_fit.intercept,
            "r_squared": self._ability_fit.r_squared,
            "n_agents": self._ability_fit.n_agents,
            "n_frontier_points": self._ability_fit.n_frontier_points,
        }

    def predict(
        self,
        predicted_beta: Dict[str, float],
        task_ids: List[str],
    ) -> Dict[str, Tuple[float, datetime]]:
        """Predict solvability dates for tasks.

        Inverts the ability regression: days = (beta - intercept) / slope

        Args:
            predicted_beta: Predicted difficulties
            task_ids: Task IDs to predict for

        Returns:
            Dict mapping task_id -> (predicted_days, predicted_date)
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        result = {}
        for task_id in task_ids:
            if task_id not in predicted_beta:
                continue

            beta = predicted_beta[task_id]
            # Invert: theta = slope * days + intercept  =>  days = (theta - intercept) / slope
            # At solvability, theta = beta
            days = (beta - self._ability_fit.intercept) / self._ability_fit.slope
            date = self._ability_fit.reference_date + timedelta(days=int(round(days)))
            result[task_id] = (float(days), date)

        return result


def compute_dates_from_predicted_difficulties(
    predicted_beta: Dict[str, float],
    oracle_abilities: pd.DataFrame,
    agent_dates: Dict[str, str],
    task_ids: List[str],
    solve_probability: float = 0.5,
) -> Dict[str, Tuple[float, datetime]]:
    """For each task, find earliest Oracle agent with ability >= predicted difficulty.

    This bypasses ability-over-time regression entirely - directly looks up
    when an agent existed that could solve the task at the predicted difficulty.

    Args:
        predicted_beta: Predicted difficulties from any method
        oracle_abilities: DataFrame with 'theta' column (Oracle agent abilities)
        agent_dates: Dict mapping agent_id -> date string (YYYYMMDD)
        task_ids: Tasks to compute dates for
        solve_probability: Threshold for capability (default 0.5)

    Returns:
        Dict mapping task_id -> (days_since_earliest, date)
        Tasks where no Oracle agent is capable are excluded from result.
    """
    if "theta" not in oracle_abilities.columns:
        raise ValueError("oracle_abilities must have 'theta' column")

    # Build agent -> (theta, date) mapping
    agent_info = {}
    for agent_id in oracle_abilities.index:
        if agent_id not in agent_dates:
            continue
        theta = oracle_abilities.loc[agent_id, "theta"]
        date = parse_date(agent_dates[agent_id])
        agent_info[agent_id] = (theta, date)

    if not agent_info:
        raise ValueError("No agents found with both abilities and dates")

    # Get earliest date as reference
    all_dates = [info[1] for info in agent_info.values()]
    earliest_agent_date = min(all_dates)

    # Compute threshold offset: logit(p) = log(p / (1-p))
    # For p=0.5, this is 0; for p=0.3, this is ~-0.847
    threshold_offset = np.log(solve_probability / (1 - solve_probability))

    result: Dict[str, Tuple[float, datetime]] = {}

    for task_id in task_ids:
        if task_id not in predicted_beta:
            continue

        beta = predicted_beta[task_id]
        required_theta = beta + threshold_offset

        # Find all capable agents (theta >= required_theta)
        capable_agents = [
            (agent_id, info[1])  # (agent_id, date)
            for agent_id, info in agent_info.items()
            if info[0] >= required_theta
        ]

        if capable_agents:
            # Find earliest by date
            earliest = min(capable_agents, key=lambda x: x[1])
            earliest_date = earliest[1]
            days_since_earliest = (earliest_date - earliest_agent_date).days
            result[task_id] = (float(days_since_earliest), earliest_date)
        # If no capable agent, task is excluded from result

    return result


def compute_frontier_ability_intervals(
    oracle_abilities: pd.DataFrame,
    agent_dates: Dict[str, str],
) -> Dict[str, float]:
    """Compute time intervals between new frontier ability models.

    Uses cumulative max to identify points where the frontier ability improved.

    Args:
        oracle_abilities: DataFrame with 'theta' column (Oracle agent abilities)
        agent_dates: Dict mapping agent_id -> date string (YYYYMMDD)

    Returns:
        Dict with: mean_days, median_days, min_days, max_days, n_frontier_jumps
    """
    # Build dataframe of agents with dates
    agent_data = []
    for agent_id in oracle_abilities.index:
        if agent_id not in agent_dates:
            continue
        theta = oracle_abilities.loc[agent_id, "theta"]
        date = parse_date(agent_dates[agent_id])
        agent_data.append({"agent_id": agent_id, "theta": theta, "date": date})

    if len(agent_data) < 2:
        return {
            "mean_days": float("nan"),
            "median_days": float("nan"),
            "min_days": float("nan"),
            "max_days": float("nan"),
            "n_frontier_jumps": 0,
        }

    df = pd.DataFrame(agent_data)
    df = df.sort_values("date")

    # Group by date, take max ability per date
    df_grouped = df.groupby("date").agg({"theta": "max"}).reset_index()
    df_grouped = df_grouped.sort_values("date")

    # Compute cumulative max (frontier trajectory)
    df_grouped["frontier_theta"] = df_grouped["theta"].cummax()

    # Find points where frontier improved (strictly greater than previous)
    frontier_changes = df_grouped[df_grouped["frontier_theta"].diff().fillna(1) > 0].copy()

    if len(frontier_changes) < 2:
        return {
            "mean_days": float("nan"),
            "median_days": float("nan"),
            "min_days": float("nan"),
            "max_days": float("nan"),
            "n_frontier_jumps": 0,
        }

    # Compute intervals between consecutive frontier jumps
    frontier_dates = frontier_changes["date"].tolist()
    intervals = []
    for i in range(1, len(frontier_dates)):
        delta = (frontier_dates[i] - frontier_dates[i - 1]).days
        intervals.append(delta)

    intervals_arr = np.array(intervals)

    return {
        "mean_days": float(np.mean(intervals_arr)),
        "median_days": float(np.median(intervals_arr)),
        "min_days": float(np.min(intervals_arr)),
        "max_days": float(np.max(intervals_arr)),
        "n_frontier_jumps": len(frontier_changes),
    }


def compute_date_forecast_metrics(
    predicted: Dict[str, Tuple[float, datetime]],
    ground_truth_days: Dict[str, float],
    task_ids: List[str],
) -> Dict[str, float]:
    """Compute evaluation metrics for date forecasting.

    Args:
        predicted: Dict from DateForecastModel.predict() (task_id -> (days, date))
        ground_truth_days: Dict from compute_ground_truth_days()
        task_ids: Task IDs to evaluate on

    Returns:
        Dict with metrics:
            - mae_days: Mean absolute error in days
            - rmse_days: Root mean square error in days
            - pearson_r: Pearson correlation
            - pearson_p: Pearson p-value
            - spearman_rho: Spearman correlation
            - spearman_p: Spearman p-value
            - n_tasks: Number of evaluated tasks
            - early_pct: % of predictions earlier than actual
    """
    pred_days = []
    actual_days = []

    for task_id in task_ids:
        if task_id in predicted and task_id in ground_truth_days:
            pred_days.append(predicted[task_id][0])
            actual_days.append(ground_truth_days[task_id])

    n_tasks = len(pred_days)

    if n_tasks == 0:
        return {
            "mae_days": float("nan"),
            "rmse_days": float("nan"),
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "spearman_rho": float("nan"),
            "spearman_p": float("nan"),
            "n_tasks": 0,
            "early_pct": float("nan"),
        }

    pred_arr = np.array(pred_days)
    actual_arr = np.array(actual_days)
    errors = pred_arr - actual_arr

    # MAE and RMSE can be computed with any number of samples >= 1
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))

    # Early prediction percentage (pred < actual)
    early_pct = float(np.mean(pred_arr < actual_arr) * 100)

    # Correlations require at least 3 points to be meaningful
    if n_tasks >= 3:
        pearson_r, pearson_p = stats.pearsonr(pred_arr, actual_arr)
        spearman_rho, spearman_p = stats.spearmanr(pred_arr, actual_arr)
    else:
        pearson_r = pearson_p = spearman_rho = spearman_p = float("nan")

    return {
        "mae_days": mae,
        "rmse_days": rmse,
        "pearson_r": float(pearson_r) if not np.isnan(pearson_r) else float("nan"),
        "pearson_p": float(pearson_p) if not np.isnan(pearson_p) else float("nan"),
        "spearman_rho": float(spearman_rho) if not np.isnan(spearman_rho) else float("nan"),
        "spearman_p": float(spearman_p) if not np.isnan(spearman_p) else float("nan"),
        "n_tasks": n_tasks,
        "early_pct": early_pct,
    }

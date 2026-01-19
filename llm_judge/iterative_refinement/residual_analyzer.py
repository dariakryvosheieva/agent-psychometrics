"""Residual analysis for identifying difficult-to-predict tasks.

Identifies tasks where the current feature set fails to predict difficulty,
then analyzes these tasks to suggest feature improvements.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


@dataclass
class HighResidualTask:
    """A task with high prediction residual."""

    task_id: str
    actual_difficulty: float
    predicted_difficulty: float
    residual: float  # actual - predicted (positive = harder than predicted)
    features: Dict[str, float]
    task_summary: Optional[str] = None  # Brief summary of task

    @property
    def direction(self) -> str:
        """Whether task was harder or easier than predicted."""
        return "harder" if self.residual > 0 else "easier"


@dataclass
class ResidualAnalysis:
    """Results from residual analysis."""

    # High residual tasks
    harder_than_predicted: List[HighResidualTask]  # Tasks that were harder
    easier_than_predicted: List[HighResidualTask]  # Tasks that were easier

    # Overall metrics
    rmse: float
    mean_residual: float
    std_residual: float

    # Feature coefficients from the model
    feature_coefficients: Dict[str, float]

    def get_top_tasks(self, n: int = 10) -> List[HighResidualTask]:
        """Get top-n tasks by absolute residual."""
        all_tasks = self.harder_than_predicted + self.easier_than_predicted
        return sorted(all_tasks, key=lambda t: abs(t.residual), reverse=True)[:n]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "rmse": self.rmse,
            "mean_residual": self.mean_residual,
            "std_residual": self.std_residual,
            "feature_coefficients": self.feature_coefficients,
            "n_harder_than_predicted": len(self.harder_than_predicted),
            "n_easier_than_predicted": len(self.easier_than_predicted),
            "top_harder": [
                {
                    "task_id": t.task_id,
                    "residual": t.residual,
                    "actual": t.actual_difficulty,
                    "predicted": t.predicted_difficulty,
                }
                for t in self.harder_than_predicted[:5]
            ],
            "top_easier": [
                {
                    "task_id": t.task_id,
                    "residual": t.residual,
                    "actual": t.actual_difficulty,
                    "predicted": t.predicted_difficulty,
                }
                for t in self.easier_than_predicted[:5]
            ],
        }


def analyze_residuals(
    features_df: pd.DataFrame,
    ground_truth: pd.Series,
    feature_names: List[str],
    n_top: int = 10,
    task_data: Optional[Dict[str, Dict[str, Any]]] = None,
) -> ResidualAnalysis:
    """Analyze prediction residuals to find difficult-to-predict tasks.

    Args:
        features_df: DataFrame with index=task_id, columns=feature values
        ground_truth: Series with index=task_id, values=ground truth difficulty (b)
        feature_names: List of feature column names to use
        n_top: Number of top residual tasks to include per direction
        task_data: Optional dict of task_id -> task metadata (for summaries)

    Returns:
        ResidualAnalysis with high-residual tasks and metrics
    """
    # Align data
    common_tasks = features_df.index.intersection(ground_truth.index)

    if len(common_tasks) < 5:
        raise ValueError(f"Need at least 5 tasks, got {len(common_tasks)}")

    X = features_df.loc[common_tasks, feature_names].values
    y = ground_truth.loc[common_tasks].values

    # Handle NaN
    X = np.nan_to_num(X, nan=0.0)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit Ridge regression
    model = Ridge(alpha=1.0)
    model.fit(X_scaled, y)

    # Get predictions and residuals
    predictions = model.predict(X_scaled)
    residuals = y - predictions

    # Feature coefficients (scaled back for interpretability)
    # coef_original = coef_scaled / std
    feature_coefficients = {
        name: float(coef / scaler.scale_[i])
        for i, (name, coef) in enumerate(zip(feature_names, model.coef_))
    }

    # Create HighResidualTask objects
    all_tasks = []
    for i, task_id in enumerate(common_tasks):
        features_dict = {
            name: float(features_df.loc[task_id, name])
            for name in feature_names
            if name in features_df.columns
        }

        summary = None
        if task_data and task_id in task_data:
            # Create brief summary from problem statement
            ps = task_data[task_id].get("problem_statement", "")
            summary = ps[:200] + "..." if len(ps) > 200 else ps

        all_tasks.append(
            HighResidualTask(
                task_id=task_id,
                actual_difficulty=float(y[i]),
                predicted_difficulty=float(predictions[i]),
                residual=float(residuals[i]),
                features=features_dict,
                task_summary=summary,
            )
        )

    # Split by direction and sort by absolute residual
    harder = sorted(
        [t for t in all_tasks if t.residual > 0],
        key=lambda t: t.residual,
        reverse=True,
    )[:n_top]

    easier = sorted(
        [t for t in all_tasks if t.residual < 0],
        key=lambda t: t.residual,  # Most negative first
    )[:n_top]

    return ResidualAnalysis(
        harder_than_predicted=harder,
        easier_than_predicted=easier,
        rmse=float(np.sqrt(np.mean(residuals**2))),
        mean_residual=float(np.mean(residuals)),
        std_residual=float(np.std(residuals)),
        feature_coefficients=feature_coefficients,
    )


def format_residual_analysis_for_llm(
    analysis: ResidualAnalysis,
    feature_definitions: List[Any],  # List of FeatureDefinition
) -> str:
    """Format residual analysis as a prompt for LLM refinement.

    Args:
        analysis: The residual analysis results
        feature_definitions: Current feature definitions

    Returns:
        Formatted string describing failures for LLM prompt
    """
    lines = [
        "## PREDICTION FAILURES",
        "",
        f"Current model achieves RMSE = {analysis.rmse:.3f}",
        "",
        "### Feature Coefficients (importance for predicting difficulty)",
        "",
    ]

    # Sort features by absolute coefficient
    sorted_coeffs = sorted(
        analysis.feature_coefficients.items(), key=lambda x: abs(x[1]), reverse=True
    )
    for name, coef in sorted_coeffs:
        sign = "+" if coef >= 0 else ""
        lines.append(f"- {name}: {sign}{coef:.3f}")

    lines.extend(
        [
            "",
            "### Tasks HARDER than predicted (model underestimates difficulty)",
            "",
            "These tasks have features suggesting they should be easy, but they are actually hard.",
            "What makes these tasks deceptively difficult?",
            "",
        ]
    )

    for task in analysis.harder_than_predicted[:5]:
        lines.append(f"**{task.task_id}**")
        lines.append(f"- Predicted: {task.predicted_difficulty:.2f}, Actual: {task.actual_difficulty:.2f} (Δ = +{task.residual:.2f})")
        lines.append(f"- Features: {_format_features(task.features)}")
        if task.task_summary:
            lines.append(f"- Summary: {task.task_summary}")
        lines.append("")

    lines.extend(
        [
            "### Tasks EASIER than predicted (model overestimates difficulty)",
            "",
            "These tasks have features suggesting they should be hard, but they are actually easy.",
            "What makes these tasks deceptively easy?",
            "",
        ]
    )

    for task in analysis.easier_than_predicted[:5]:
        lines.append(f"**{task.task_id}**")
        lines.append(f"- Predicted: {task.predicted_difficulty:.2f}, Actual: {task.actual_difficulty:.2f} (Δ = {task.residual:.2f})")
        lines.append(f"- Features: {_format_features(task.features)}")
        if task.task_summary:
            lines.append(f"- Summary: {task.task_summary}")
        lines.append("")

    return "\n".join(lines)


def _format_features(features: Dict[str, float]) -> str:
    """Format feature dict as compact string."""
    parts = [f"{k}={v:.1f}" for k, v in sorted(features.items())]
    return ", ".join(parts)
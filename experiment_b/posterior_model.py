"""Posterior model: Prior + linear correction from trajectory features."""

from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np
from sklearn.linear_model import Ridge

from .prior_model import PriorModel
from .trajectory_features import (
    TRAJECTORY_FEATURE_NAMES,
    load_trajectories_for_task,
    aggregate_trajectory_features,
)
from .lunette_features import (
    LUNETTE_FEATURE_NAMES,
    load_lunette_features_for_task,
    aggregate_lunette_features,
)
from .llm_judge_features import (
    LLM_JUDGE_FEATURE_NAMES,
    load_llm_judge_features_for_task,
    aggregate_llm_judge_features,
)
from .llm_judge_features_v4 import (
    LLM_JUDGE_V4_FEATURE_NAMES,
    load_llm_judge_v4_features_for_task,
    aggregate_llm_judge_v4_features,
)
from .llm_judge_features_v5 import (
    LLM_JUDGE_V5_FEATURE_NAMES,
    load_llm_judge_v5_features_for_task,
    aggregate_llm_judge_v5_features,
)
from .llm_judge_features_v5_single import (
    LLM_JUDGE_V5_SINGLE_FEATURE_NAMES,
    load_llm_judge_v5_single_features_for_task,
    aggregate_llm_judge_v5_single_features,
)


class PosteriorModel:
    """
    Posterior difficulty = Prior(x_i) + psi^T * avg_features(trajectories)

    From the proposal:
    posterior_difficulty_i = prior(x_i) + psi^T * (1/|M|) * sum_j f(tau_ij)
    """

    def __init__(
        self,
        prior_model: PriorModel,
        alpha: float = 1.0,
        feature_source: Literal["simple", "lunette", "llm_judge", "llm_judge_v4", "llm_judge_v5", "llm_judge_v5_single"] = "simple",
        lunette_features_dir: Optional[Path] = None,
        llm_judge_features_dir: Optional[Path] = None,
        llm_judge_v4_features_dir: Optional[Path] = None,
        llm_judge_v5_features_dir: Optional[Path] = None,
        llm_judge_v5_single_features_dir: Optional[Path] = None,
    ):
        """Initialize posterior model.

        Args:
            prior_model: Trained prior model
            alpha: Ridge regularization parameter for psi
            feature_source: "simple" for message stats, "lunette" for Lunette API,
                           "llm_judge" for direct LLM API, "llm_judge_v4" for V4 features,
                           "llm_judge_v5" for V5 features, "llm_judge_v5_single" for single feature
            lunette_features_dir: Directory containing pre-computed Lunette features
            llm_judge_features_dir: Directory containing pre-computed LLM judge features
            llm_judge_v4_features_dir: Directory for V4 LLM judge features
            llm_judge_v5_features_dir: Directory for V5 LLM judge features
            llm_judge_v5_single_features_dir: Directory for V5 single feature (location_vs_fix_alignment only)
        """
        self.prior_model = prior_model
        self.alpha = alpha
        self.feature_source = feature_source
        self.lunette_features_dir = lunette_features_dir
        self.llm_judge_features_dir = llm_judge_features_dir
        self.llm_judge_v4_features_dir = llm_judge_v4_features_dir
        self.llm_judge_v5_features_dir = llm_judge_v5_features_dir
        self.llm_judge_v5_single_features_dir = llm_judge_v5_single_features_dir
        self.psi_model: Optional[Ridge] = None
        self.training_stats: Dict = {}

        # Set feature names based on source
        if feature_source == "lunette":
            self.feature_names = LUNETTE_FEATURE_NAMES
        elif feature_source == "llm_judge":
            self.feature_names = LLM_JUDGE_FEATURE_NAMES
        elif feature_source == "llm_judge_v4":
            self.feature_names = LLM_JUDGE_V4_FEATURE_NAMES
        elif feature_source == "llm_judge_v5":
            self.feature_names = LLM_JUDGE_V5_FEATURE_NAMES
        elif feature_source == "llm_judge_v5_single":
            self.feature_names = LLM_JUDGE_V5_SINGLE_FEATURE_NAMES
        else:
            self.feature_names = TRAJECTORY_FEATURE_NAMES

    def _load_features_for_task(
        self,
        task_id: str,
        agents: List[str],
        trajectories_dir: Path,
    ) -> Optional[np.ndarray]:
        """Load and aggregate features for a task based on feature_source."""
        if self.feature_source == "lunette":
            if self.lunette_features_dir is None:
                return None
            features = load_lunette_features_for_task(
                task_id, agents, self.lunette_features_dir
            )
            if not features:
                return None
            return aggregate_lunette_features(features)
        elif self.feature_source == "llm_judge":
            if self.llm_judge_features_dir is None:
                return None
            features = load_llm_judge_features_for_task(
                task_id, agents, self.llm_judge_features_dir
            )
            if not features:
                return None
            return aggregate_llm_judge_features(features)
        elif self.feature_source == "llm_judge_v4":
            if self.llm_judge_v4_features_dir is None:
                return None
            features = load_llm_judge_v4_features_for_task(
                task_id, agents, self.llm_judge_v4_features_dir
            )
            if not features:
                return None
            return aggregate_llm_judge_v4_features(features)
        elif self.feature_source == "llm_judge_v5":
            if self.llm_judge_v5_features_dir is None:
                return None
            features = load_llm_judge_v5_features_for_task(
                task_id, agents, self.llm_judge_v5_features_dir
            )
            if not features:
                return None
            return aggregate_llm_judge_v5_features(features)
        elif self.feature_source == "llm_judge_v5_single":
            if self.llm_judge_v5_single_features_dir is None:
                return None
            features = load_llm_judge_v5_single_features_for_task(
                task_id, agents, self.llm_judge_v5_single_features_dir
            )
            if not features:
                return None
            return aggregate_llm_judge_v5_single_features(features)
        else:
            # Simple trajectory features
            traj_features = load_trajectories_for_task(task_id, agents, trajectories_dir)
            if not traj_features:
                return None
            return aggregate_trajectory_features(traj_features)

    def fit(
        self,
        task_ids: List[str],
        ground_truth_difficulties: np.ndarray,
        weak_agents: List[str],
        trajectories_dir: Path,
    ) -> "PosteriorModel":
        """Fit the correction term psi.

        Args:
            task_ids: Training task IDs (D_train)
            ground_truth_difficulties: IRT b values for tasks (aligned with task_ids)
            weak_agents: M1 agents whose trajectories to use
            trajectories_dir: Base directory for trajectories
        """
        # Get prior predictions
        prior_preds = self.prior_model.get_prior_predictions(task_ids)

        # Compute residuals (what prior doesn't explain)
        X_features = []
        y_residuals = []
        valid_task_ids = []
        tasks_with_features = 0

        for i, task_id in enumerate(task_ids):
            if task_id not in prior_preds:
                continue

            # Load features for this task
            feat_vec = self._load_features_for_task(task_id, weak_agents, trajectories_dir)

            if feat_vec is None:
                continue  # No features available

            tasks_with_features += 1
            X_features.append(feat_vec)

            # Residual = ground_truth - prior
            residual = ground_truth_difficulties[i] - prior_preds[task_id]
            y_residuals.append(residual)
            valid_task_ids.append(task_id)

        self.training_stats = {
            "total_tasks": len(task_ids),
            "tasks_with_features": tasks_with_features,
            "tasks_used_for_training": len(valid_task_ids),
            "agents_used": len(weak_agents),
            "feature_source": self.feature_source,
        }

        if not X_features:
            print("Warning: No valid training data for posterior model")
            self.psi_model = None
            return self

        X = np.array(X_features)
        y = np.array(y_residuals)

        # Fit Ridge regression for psi
        self.psi_model = Ridge(alpha=self.alpha)
        self.psi_model.fit(X, y)

        print(f"Posterior model ({self.feature_source}) trained on {len(valid_task_ids)} tasks")
        print(f"  Tasks with features: {tasks_with_features}")
        print(f"  Psi coefficients: {dict(zip(self.feature_names, self.psi_model.coef_))}")

        return self

    def predict(
        self,
        task_ids: List[str],
        weak_agents: List[str],
        trajectories_dir: Path,
    ) -> Dict[str, float]:
        """Predict posterior difficulty.

        posterior = prior + psi^T * features
        """
        # Get prior predictions
        prior_preds = self.prior_model.get_prior_predictions(task_ids)

        predictions = {}
        for task_id in task_ids:
            if task_id not in prior_preds:
                continue

            prior = prior_preds[task_id]

            # If no psi model, just use prior
            if self.psi_model is None:
                predictions[task_id] = prior
                continue

            # Load and aggregate features
            feat_vec = self._load_features_for_task(task_id, weak_agents, trajectories_dir)

            if feat_vec is None:
                predictions[task_id] = prior
                continue

            correction = self.psi_model.predict([feat_vec])[0]
            predictions[task_id] = prior + correction

        return predictions

    def get_feature_importance(self) -> Dict[str, float]:
        """Get psi coefficients as feature importance."""
        if self.psi_model is None:
            return {}
        return dict(zip(self.feature_names, self.psi_model.coef_))

    def get_training_stats(self) -> Dict:
        """Get training statistics."""
        return self.training_stats

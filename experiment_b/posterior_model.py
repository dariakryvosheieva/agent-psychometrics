"""Posterior model: Prior + linear correction from trajectory features."""

import json
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import numpy as np
from sklearn.linear_model import Ridge

from .config import RegressionMode
from .prior_model import PriorModel, EmbeddingPriorModel
from .trajectory_features import (
    TRAJECTORY_FEATURE_NAMES,
    load_trajectories_for_task,
    aggregate_trajectory_features,
)
from .trajectory_features_v2 import (
    EXECUTION_FEATURE_NAMES,
)
from .lunette.features import (
    LUNETTE_FEATURE_NAMES,
    load_lunette_features_for_task,
    aggregate_lunette_features,
)
from .llm_judge.features_v1 import (
    LLM_JUDGE_FEATURE_NAMES,
    load_llm_judge_features_for_task,
    aggregate_llm_judge_features,
)
from .llm_judge.features_v4 import (
    LLM_JUDGE_V4_FEATURE_NAMES,
    load_llm_judge_v4_features_for_task,
    aggregate_llm_judge_v4_features,
)
from .llm_judge.features_v5 import (
    LLM_JUDGE_V5_FEATURE_NAMES,
    load_llm_judge_v5_features_for_task,
    aggregate_llm_judge_v5_features,
)
from .llm_judge.features_v5_single import (
    LLM_JUDGE_V5_SINGLE_FEATURE_NAMES,
    load_llm_judge_v5_single_features_for_task,
    aggregate_llm_judge_v5_single_features,
)
from .llm_judge.features_v6 import (
    LLM_JUDGE_V6_FEATURE_NAMES,
    load_llm_judge_v6_features_for_task,
    aggregate_llm_judge_v6_features,
)
from .llm_judge.features_v7 import (
    LLM_JUDGE_V7_FEATURE_NAMES,
    load_llm_judge_v7_features_for_task,
    aggregate_llm_judge_v7_features,
)
from .test_progression import (
    TEST_PROGRESSION_FEATURE_NAMES,
    features_from_dict,
    aggregate_test_progression_features,
)


class PosteriorModel:
    """
    Posterior difficulty prediction with multiple regression modes.

    Supports three regression modes:
    - "residual": posterior = prior + psi * traj_features (learn correction to prior)
    - "direct_with_prior": posterior = model(traj_features, prior_pred) (prior as feature)
    - "direct_with_prior_features": posterior = model(traj_features, prior_input_features)

    From the proposal:
    posterior_difficulty_i = prior(x_i) + psi^T * (1/|M|) * sum_j f(tau_ij)
    """

    def __init__(
        self,
        prior_model: Union[PriorModel, EmbeddingPriorModel],
        alpha: float = 1.0,
        feature_source: Literal[
            "simple", "lunette", "llm_judge", "llm_judge_v4", "llm_judge_v5",
            "llm_judge_v5_single", "execution", "discoverability", "combined_v2",
            "llm_judge_v7", "mechanical_v7", "test_progression"
        ] = "simple",
        regression_mode: RegressionMode = "residual",
        lunette_features_dir: Optional[Path] = None,
        llm_judge_features_dir: Optional[Path] = None,
        llm_judge_v4_features_dir: Optional[Path] = None,
        llm_judge_v5_features_dir: Optional[Path] = None,
        llm_judge_v5_single_features_dir: Optional[Path] = None,
        execution_features_dir: Optional[Path] = None,
        llm_judge_v6_features_dir: Optional[Path] = None,
        llm_judge_v7_features_dir: Optional[Path] = None,
        test_progression_features_dir: Optional[Path] = None,
    ):
        """Initialize posterior model.

        Args:
            prior_model: Trained prior model
            alpha: Ridge regularization parameter for psi
            feature_source: Feature source to use:
                - "simple": Basic message stats (count, chars, resolved_rate)
                - "lunette": Lunette API features
                - "llm_judge": Direct LLM API (v1)
                - "llm_judge_v4": V4 LLM features
                - "llm_judge_v5": V5 LLM features
                - "llm_judge_v5_single": Single location_vs_fix_alignment feature
                - "execution": Deterministic execution features (v2)
                - "discoverability": LLM judge v6 solution discoverability
                - "combined_v2": execution + discoverability combined
            regression_mode: How to train the model:
                - "residual": Learn correction to prior (posterior = prior + correction)
                - "direct_with_prior": Predict difficulty directly with prior as feature
                - "direct_with_prior_features": Predict difficulty with prior's input features
            lunette_features_dir: Directory containing pre-computed Lunette features
            llm_judge_features_dir: Directory containing pre-computed LLM judge features
            llm_judge_v4_features_dir: Directory for V4 LLM judge features
            llm_judge_v5_features_dir: Directory for V5 LLM judge features
            llm_judge_v5_single_features_dir: Directory for V5 single feature
            execution_features_dir: Directory for execution features (v2)
            llm_judge_v6_features_dir: Directory for v6 discoverability features
            llm_judge_v7_features_dir: Directory for v7 unified semantic features
            test_progression_features_dir: Directory for test progression features
        """
        self.prior_model = prior_model
        self.alpha = alpha
        self.feature_source = feature_source
        self.regression_mode = regression_mode
        self.lunette_features_dir = lunette_features_dir
        self.llm_judge_features_dir = llm_judge_features_dir
        self.llm_judge_v4_features_dir = llm_judge_v4_features_dir
        self.llm_judge_v5_features_dir = llm_judge_v5_features_dir
        self.llm_judge_v5_single_features_dir = llm_judge_v5_single_features_dir
        self.execution_features_dir = execution_features_dir
        self.llm_judge_v6_features_dir = llm_judge_v6_features_dir
        self.llm_judge_v7_features_dir = llm_judge_v7_features_dir
        self.test_progression_features_dir = test_progression_features_dir
        self.psi_model: Optional[Ridge] = None
        self.training_stats: Dict = {}
        self._prior_feature_dim: Optional[int] = None

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
        elif feature_source == "execution":
            self.feature_names = EXECUTION_FEATURE_NAMES
        elif feature_source == "discoverability":
            self.feature_names = LLM_JUDGE_V6_FEATURE_NAMES
        elif feature_source == "combined_v2":
            self.feature_names = EXECUTION_FEATURE_NAMES + LLM_JUDGE_V6_FEATURE_NAMES
        elif feature_source == "llm_judge_v7":
            self.feature_names = LLM_JUDGE_V7_FEATURE_NAMES
        elif feature_source == "mechanical_v7":
            # Mechanical features + v7 semantic features
            self.feature_names = EXECUTION_FEATURE_NAMES + LLM_JUDGE_V7_FEATURE_NAMES
        elif feature_source == "test_progression":
            self.feature_names = TEST_PROGRESSION_FEATURE_NAMES
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
        elif self.feature_source == "execution":
            return self._load_execution_features(task_id)
        elif self.feature_source == "discoverability":
            if self.llm_judge_v6_features_dir is None:
                return None
            features = load_llm_judge_v6_features_for_task(
                task_id, agents, self.llm_judge_v6_features_dir
            )
            if not features:
                return None
            return aggregate_llm_judge_v6_features(features)
        elif self.feature_source == "combined_v2":
            # Combine execution features with discoverability
            exec_feat = self._load_execution_features(task_id)
            if exec_feat is None:
                return None
            if self.llm_judge_v6_features_dir is None:
                # Just return execution features if no v6 dir
                return exec_feat
            v6_features = load_llm_judge_v6_features_for_task(
                task_id, agents, self.llm_judge_v6_features_dir
            )
            if not v6_features:
                # Just return execution features if no v6 data
                return exec_feat
            v6_agg = aggregate_llm_judge_v6_features(v6_features)
            return np.concatenate([exec_feat, v6_agg])
        elif self.feature_source == "llm_judge_v7":
            if self.llm_judge_v7_features_dir is None:
                return None
            features = load_llm_judge_v7_features_for_task(
                task_id, agents, self.llm_judge_v7_features_dir
            )
            if not features:
                return None
            return aggregate_llm_judge_v7_features(features)
        elif self.feature_source == "mechanical_v7":
            # Combine mechanical features with v7 semantic features
            exec_feat = self._load_execution_features(task_id)
            if exec_feat is None:
                return None
            if self.llm_judge_v7_features_dir is None:
                # Just return mechanical features if no v7 dir
                return exec_feat
            v7_features = load_llm_judge_v7_features_for_task(
                task_id, agents, self.llm_judge_v7_features_dir
            )
            if not v7_features:
                # Just return mechanical features if no v7 data
                return exec_feat
            v7_agg = aggregate_llm_judge_v7_features(v7_features)
            return np.concatenate([exec_feat, v7_agg])
        elif self.feature_source == "test_progression":
            return self._load_test_progression_features(task_id, agents)
        else:
            # Simple trajectory features
            traj_features = load_trajectories_for_task(task_id, agents, trajectories_dir)
            if not traj_features:
                return None
            return aggregate_trajectory_features(traj_features)

    def _load_execution_features(self, task_id: str) -> Optional[np.ndarray]:
        """Load pre-computed execution features for a task.

        Args:
            task_id: Task instance ID

        Returns:
            Aggregated feature vector or None
        """
        if self.execution_features_dir is None:
            return None

        feature_file = self.execution_features_dir / f"{task_id}.json"
        if not feature_file.exists():
            return None

        try:
            with open(feature_file) as f:
                data = json.load(f)
            if "aggregated" in data:
                return np.array(data["aggregated"])
            return None
        except (json.JSONDecodeError, IOError):
            return None

    def _load_test_progression_features(
        self, task_id: str, agents: List[str]
    ) -> Optional[np.ndarray]:
        """Load pre-computed test progression features for a task.

        Args:
            task_id: Task instance ID
            agents: List of agent names to load features for

        Returns:
            Aggregated feature vector or None
        """
        if self.test_progression_features_dir is None:
            return None

        # Load features for each agent
        agent_features = {}
        for agent in agents:
            feature_file = self.test_progression_features_dir / agent / f"{task_id}.json"
            if not feature_file.exists():
                continue

            try:
                with open(feature_file) as f:
                    data = json.load(f)
                # Only use features with test output for better signal
                if data.get("has_test_output", False):
                    agent_features[agent] = features_from_dict(data)
            except (json.JSONDecodeError, IOError):
                continue

        if not agent_features:
            return None

        return aggregate_test_progression_features(agent_features)

    def fit(
        self,
        task_ids: List[str],
        ground_truth_difficulties: np.ndarray,
        weak_agents: List[str],
        trajectories_dir: Path,
    ) -> "PosteriorModel":
        """Fit the model based on regression_mode.

        Args:
            task_ids: Training task IDs (D_train)
            ground_truth_difficulties: IRT b values for tasks (aligned with task_ids)
            weak_agents: M1 agents whose trajectories to use
            trajectories_dir: Base directory for trajectories

        Regression modes:
            - "residual": Learn psi s.t. posterior = prior + psi * features
            - "direct_with_prior": Learn model(features, prior) -> difficulty
            - "direct_with_prior_features": Learn model(features, prior_input_features) -> difficulty
        """
        # Get prior predictions
        prior_preds = self.prior_model.get_prior_predictions(task_ids)

        # Get prior input features if needed
        prior_features = None
        if self.regression_mode == "direct_with_prior_features":
            prior_features = self.prior_model.get_prior_features(task_ids)
            self._prior_feature_dim = self.prior_model.get_prior_feature_dim()

        # Build training data
        X_features = []
        y_targets = []
        valid_task_ids = []
        tasks_with_features = 0

        for i, task_id in enumerate(task_ids):
            if task_id not in prior_preds:
                continue

            # Load trajectory features for this task
            traj_feat = self._load_features_for_task(task_id, weak_agents, trajectories_dir)
            if traj_feat is None:
                continue

            tasks_with_features += 1

            # Build feature vector based on regression mode
            if self.regression_mode == "residual":
                # Features: just trajectory features
                # Target: residual (ground_truth - prior)
                features = traj_feat
                target = ground_truth_difficulties[i] - prior_preds[task_id]

            elif self.regression_mode == "direct_with_prior":
                # Features: trajectory features + prior prediction
                # Target: ground truth difficulty
                features = np.concatenate([traj_feat, [prior_preds[task_id]]])
                target = ground_truth_difficulties[i]

            elif self.regression_mode == "direct_with_prior_features":
                # Features: trajectory features + prior's input features (e.g., task embeddings)
                # Target: ground truth difficulty
                if prior_features is None or task_id not in prior_features:
                    continue
                features = np.concatenate([traj_feat, prior_features[task_id]])
                target = ground_truth_difficulties[i]

            else:
                raise ValueError(f"Unknown regression_mode: {self.regression_mode}")

            X_features.append(features)
            y_targets.append(target)
            valid_task_ids.append(task_id)

        self.training_stats = {
            "total_tasks": len(task_ids),
            "tasks_with_features": tasks_with_features,
            "tasks_used_for_training": len(valid_task_ids),
            "agents_used": len(weak_agents),
            "feature_source": self.feature_source,
            "regression_mode": self.regression_mode,
        }

        if not X_features:
            print("Warning: No valid training data for posterior model")
            self.psi_model = None
            return self

        X = np.array(X_features)
        y = np.array(y_targets)

        # Fit Ridge regression
        self.psi_model = Ridge(alpha=self.alpha)
        self.psi_model.fit(X, y)

        print(f"Posterior model ({self.feature_source}, {self.regression_mode}) trained on {len(valid_task_ids)} tasks")
        print(f"  Tasks with features: {tasks_with_features}")
        print(f"  Feature dim: {X.shape[1]}")

        return self

    def predict(
        self,
        task_ids: List[str],
        weak_agents: List[str],
        trajectories_dir: Path,
    ) -> Dict[str, float]:
        """Predict posterior difficulty based on regression_mode.

        Regression modes:
            - "residual": posterior = prior + model(features)
            - "direct_with_prior": posterior = model(features, prior)
            - "direct_with_prior_features": posterior = model(features, prior_input_features)
        """
        # Get prior predictions
        prior_preds = self.prior_model.get_prior_predictions(task_ids)

        # Get prior input features if needed
        prior_features = None
        if self.regression_mode == "direct_with_prior_features":
            prior_features = self.prior_model.get_prior_features(task_ids)

        predictions = {}
        for task_id in task_ids:
            if task_id not in prior_preds:
                continue

            prior = prior_preds[task_id]

            # If no psi model, just use prior
            if self.psi_model is None:
                predictions[task_id] = prior
                continue

            # Load trajectory features
            traj_feat = self._load_features_for_task(task_id, weak_agents, trajectories_dir)
            if traj_feat is None:
                predictions[task_id] = prior
                continue

            # Build feature vector and predict based on regression mode
            if self.regression_mode == "residual":
                # posterior = prior + correction
                correction = self.psi_model.predict([traj_feat])[0]
                predictions[task_id] = prior + correction

            elif self.regression_mode == "direct_with_prior":
                # posterior = model(features, prior)
                features = np.concatenate([traj_feat, [prior]])
                predictions[task_id] = self.psi_model.predict([features])[0]

            elif self.regression_mode == "direct_with_prior_features":
                # posterior = model(features, prior_input_features)
                if prior_features is None or task_id not in prior_features:
                    predictions[task_id] = prior
                    continue
                features = np.concatenate([traj_feat, prior_features[task_id]])
                predictions[task_id] = self.psi_model.predict([features])[0]

        return predictions

    def get_feature_importance(self) -> Dict[str, float]:
        """Get psi coefficients as feature importance."""
        if self.psi_model is None:
            return {}
        return dict(zip(self.feature_names, self.psi_model.coef_))

    def get_training_stats(self) -> Dict:
        """Get training statistics."""
        return self.training_stats

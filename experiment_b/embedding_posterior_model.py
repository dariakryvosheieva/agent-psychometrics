"""
Posterior model using trajectory embeddings.

Predicts:
    posterior_difficulty = prior_difficulty + psi^T * aggregated_embedding

The psi weights are learned by ridge regression on the residuals:
    residual = ground_truth_difficulty - prior_difficulty
"""

from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .embedding_aggregator import (
    AggregationType,
    EmbeddingAggregator,
    batch_aggregate_embeddings,
    aggregate_task_embeddings,
)
from .prior_model import EmbeddingPriorModel, PriorModel


class EmbeddingPosteriorModel:
    """Posterior difficulty model using trajectory embeddings.

    posterior_difficulty = prior + psi^T * f(trajectories)

    Where f(trajectories) aggregates embeddings across agent trajectories.
    """

    def __init__(
        self,
        prior_model: Union[PriorModel, EmbeddingPriorModel],
        aggregation: AggregationType = "mean_std",
        alpha: Union[float, str] = "cv",
        abilities: Optional[Dict[str, float]] = None,
    ):
        """Initialize posterior model.

        Args:
            prior_model: Trained prior model for base predictions
            aggregation: How to aggregate embeddings across agents:
                - "mean_only": Just mean
                - "mean_std": Mean + std (default, captures spread)
                - "weighted": Weight by agent ability
                - "all_stats": Mean + std + min + max
            alpha: Ridge regression alpha. Use "cv" for cross-validation search.
            abilities: Agent abilities (theta) for weighted aggregation
        """
        self.prior_model = prior_model
        self.aggregation = aggregation
        self.alpha = alpha
        self.abilities = abilities

        self.aggregator = EmbeddingAggregator(
            aggregation=aggregation,
            abilities=abilities,
        )

        self.psi_model: Optional[Pipeline] = None
        self.training_stats: Dict = {}
        self.embedding_dim: Optional[int] = None
        self.best_alpha: Optional[float] = None

    def fit(
        self,
        task_ids: List[str],
        ground_truth_difficulties: np.ndarray,
        weak_agents: List[str],
        embeddings_dir: Path,
        min_agents_per_task: int = 1,
    ) -> "EmbeddingPosteriorModel":
        """Fit the correction term psi by ridge regression on residuals.

        Args:
            task_ids: Training task IDs
            ground_truth_difficulties: IRT b values aligned with task_ids
            weak_agents: Agents whose trajectories to use
            embeddings_dir: Directory with pre-computed embeddings
            min_agents_per_task: Minimum embeddings required per task

        Returns:
            self
        """
        # Get prior predictions
        prior_preds = self.prior_model.get_prior_predictions(task_ids)

        # Build feature matrix
        X_features = []
        y_residuals = []
        valid_task_ids = []

        for i, task_id in enumerate(task_ids):
            if task_id not in prior_preds:
                continue

            # Aggregate embeddings for this task
            feat_vec = aggregate_task_embeddings(
                task_id=task_id,
                agents=weak_agents,
                embeddings_dir=embeddings_dir,
                aggregator=self.aggregator,
            )

            if feat_vec is None:
                continue

            # Compute residual
            residual = ground_truth_difficulties[i] - prior_preds[task_id]

            X_features.append(feat_vec)
            y_residuals.append(residual)
            valid_task_ids.append(task_id)

        self.training_stats = {
            "total_tasks": len(task_ids),
            "tasks_with_embeddings": len(valid_task_ids),
            "agents_used": len(weak_agents),
            "aggregation": self.aggregation,
        }

        if not X_features:
            print("Warning: No valid training data for embedding posterior model")
            self.psi_model = None
            return self

        X = np.array(X_features)
        y = np.array(y_residuals)

        self.embedding_dim = X.shape[1]

        # Fit ridge regression
        if self.alpha == "cv":
            # Use cross-validation to find best alpha
            alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000, 100000]
            ridge_cv = RidgeCV(alphas=alphas, cv=5, scoring="neg_mean_squared_error")

            self.psi_model = Pipeline([
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("ridge", ridge_cv),
            ])
            self.psi_model.fit(X, y)
            self.best_alpha = self.psi_model.named_steps["ridge"].alpha_
            print(f"RidgeCV selected alpha={self.best_alpha}")
        else:
            self.psi_model = Pipeline([
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("ridge", Ridge(alpha=float(self.alpha))),
            ])
            self.psi_model.fit(X, y)
            self.best_alpha = float(self.alpha)

        print(f"Embedding posterior ({self.aggregation}) trained on {len(valid_task_ids)} tasks")
        print(f"  Feature dim: {self.embedding_dim}")

        return self

    def predict(
        self,
        task_ids: List[str],
        weak_agents: List[str],
        embeddings_dir: Path,
    ) -> Dict[str, float]:
        """Predict posterior difficulties.

        posterior = prior + psi^T * features

        Args:
            task_ids: Tasks to predict
            weak_agents: Agents whose embeddings to use
            embeddings_dir: Directory with pre-computed embeddings

        Returns:
            Dict mapping task_id -> posterior difficulty
        """
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

            # Get aggregated features
            feat_vec = aggregate_task_embeddings(
                task_id=task_id,
                agents=weak_agents,
                embeddings_dir=embeddings_dir,
                aggregator=self.aggregator,
            )

            if feat_vec is None:
                predictions[task_id] = prior
                continue

            # Predict correction
            correction = self.psi_model.predict([feat_vec])[0]
            predictions[task_id] = prior + correction

        return predictions

    def get_training_stats(self) -> Dict:
        """Get training statistics."""
        stats = self.training_stats.copy()
        if self.best_alpha is not None:
            stats["best_alpha"] = self.best_alpha
        if self.embedding_dim is not None:
            stats["embedding_dim"] = self.embedding_dim
        return stats

    def get_psi_stats(self) -> Dict:
        """Get statistics about learned psi coefficients."""
        if self.psi_model is None:
            return {}

        ridge = self.psi_model.named_steps["ridge"]
        coef = ridge.coef_

        return {
            "n_features": len(coef),
            "coef_mean": float(np.mean(coef)),
            "coef_std": float(np.std(coef)),
            "coef_abs_mean": float(np.mean(np.abs(coef))),
            "coef_abs_max": float(np.max(np.abs(coef))),
            "intercept": float(ridge.intercept_),
        }


def train_and_evaluate_embedding_posterior(
    prior_model: Union[PriorModel, EmbeddingPriorModel],
    train_task_ids: List[str],
    train_difficulties: np.ndarray,
    valid_task_ids: List[str],
    valid_difficulties: np.ndarray,
    train_agents: List[str],
    valid_agents: List[str],
    embeddings_dir: Path,
    aggregation: AggregationType = "mean_std",
    alpha: Union[float, str] = "cv",
    abilities: Optional[Dict[str, float]] = None,
) -> Tuple[EmbeddingPosteriorModel, Dict]:
    """Train and evaluate embedding posterior model.

    Args:
        prior_model: Pre-trained prior model
        train_task_ids: D_train task IDs
        train_difficulties: Ground truth b values for D_train
        valid_task_ids: D_valid task IDs
        valid_difficulties: Ground truth b values for D_valid
        train_agents: M1 agents for training
        valid_agents: M2 agents for validation
        embeddings_dir: Directory with pre-computed embeddings
        aggregation: Aggregation strategy
        alpha: Ridge alpha or "cv"
        abilities: Agent abilities for weighted aggregation

    Returns:
        Tuple of (model, evaluation_dict)
    """
    from scipy import stats

    # Train model
    model = EmbeddingPosteriorModel(
        prior_model=prior_model,
        aggregation=aggregation,
        alpha=alpha,
        abilities=abilities,
    )

    model.fit(
        task_ids=train_task_ids,
        ground_truth_difficulties=train_difficulties,
        weak_agents=train_agents,
        embeddings_dir=embeddings_dir,
    )

    # Evaluate on training set
    train_preds = model.predict(
        task_ids=train_task_ids,
        weak_agents=train_agents,
        embeddings_dir=embeddings_dir,
    )

    # Evaluate on validation set
    valid_preds = model.predict(
        task_ids=valid_task_ids,
        weak_agents=valid_agents,
        embeddings_dir=embeddings_dir,
    )

    # Compute metrics
    def compute_metrics(predictions: Dict[str, float], task_ids: List[str], gt: np.ndarray) -> Dict:
        common = [t for t in task_ids if t in predictions]
        if not common:
            return {"n": 0, "pearson_r": None, "mse": None}

        pred_arr = np.array([predictions[t] for t in common])
        gt_idx = [task_ids.index(t) for t in common]
        gt_arr = np.array([gt[i] for i in gt_idx])

        mse = float(np.mean((pred_arr - gt_arr) ** 2))

        if len(common) > 2:
            r, p = stats.pearsonr(pred_arr, gt_arr)
            return {"n": len(common), "pearson_r": float(r), "mse": mse, "p_value": float(p)}
        return {"n": len(common), "pearson_r": None, "mse": mse}

    results = {
        "train": compute_metrics(train_preds, list(train_task_ids), train_difficulties),
        "valid": compute_metrics(valid_preds, list(valid_task_ids), valid_difficulties),
        "training_stats": model.get_training_stats(),
        "psi_stats": model.get_psi_stats(),
    }

    return model, results

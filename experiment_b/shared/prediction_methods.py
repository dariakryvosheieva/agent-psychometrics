"""Prediction methods for frontier task difficulty prediction.

This module contains:
- FeatureIRTPredictor: Joint IRT + feature learning predictor
- Helper functions for collecting predictions from various methods
- Feature source builders

For OrderedLogitIRTPredictor, see ordered_logit_predictor.py.

All prediction method code should live here. Adding a new method should only
require changes to this file.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from experiment_ab_shared.feature_source import (
    TaskFeatureSource,
    EmbeddingFeatureSource,
    CSVFeatureSource,
    GroupedFeatureSource,
    build_feature_sources as _build_feature_sources,
)
from experiment_ab_shared.feature_predictor import FeatureBasedPredictor, GroupedRidgePredictor
from experiment_b.shared.config_base import DatasetConfig


# =============================================================================
# Dataclasses
# =============================================================================


@dataclass
class FeatureIRTResults:
    """Results from running Feature-IRT methods."""

    predictions: Dict[str, Dict[str, float]]  # method_name -> beta predictions
    abilities: Dict[str, Dict[str, float]]  # method_name -> theta abilities
    diagnostics: Optional[List[Dict[str, Any]]] = None  # Grid search diagnostics
    best_predictors: Dict[str, Any] = field(default_factory=dict)  # source_name -> predictor
    baseline_init_diagnostics: Dict[str, Dict[str, float]] = field(default_factory=dict)  # source_name -> diagnostics


# =============================================================================
# Feature-IRT Predictor
# =============================================================================


class FeatureIRTPredictor:
    """IRT model with feature-based difficulties learned jointly with abilities.

    Instead of fitting Ridge regression on ground-truth IRT difficulties,
    this directly maximizes the IRT log-likelihood:

        minimize: -Σ_ij log P(y_ij | θ_j, b_i) + regularization

    where b_i = features[i] @ w + bias + residual[i] is a linear function
    of features plus an optional per-task residual.

    This approach trains the feature-to-difficulty mapping end-to-end,
    jointly learning abilities θ_j alongside feature weights w.

    Note: This predictor is designed for Experiment B where agents are
    held out (not tasks). All tasks should be seen during training.

    For GroupedFeatureSource, per-source regularization is achieved via group
    scaling: features from each source are scaled by 1/sqrt(alpha) before
    optimization. This is mathematically equivalent to having different L2
    penalties per source with a uniform l2_weight=1.0. StandardScaler is
    applied per-source BEFORE group scaling.
    """

    def __init__(
        self,
        source: Union[TaskFeatureSource, GroupedFeatureSource],
        use_residuals: bool = False,
        init_from_baseline: bool = False,
        l2_weight: float = 0.01,
        l2_residual: float = 10.0,
        l2_ability: float = 0.01,
        per_source_alphas: Optional[Dict[str, float]] = None,
        lr: float = 0.1,
        max_iter: int = 500,
        tol: float = 1e-5,
        device: str = "cpu",
        verbose: bool = True,
    ):
        """Initialize Feature-IRT predictor.

        Args:
            source: TaskFeatureSource or GroupedFeatureSource providing features.
                For GroupedFeatureSource, use per_source_alphas for differential
                regularization across sources.
            use_residuals: Include per-task residuals. Default False because
                residuals tend to overfit; the main benefit comes from joint
                ability learning, not residuals.
            init_from_baseline: If True, initialize residuals from baseline IRT
                difficulties and abilities from baseline IRT abilities, with
                feature weights starting at zero. Requires passing baseline_abilities
                and baseline_agent_ids to fit(). This allows the model to start
                from the Baseline IRT solution and learn feature-based corrections.
            l2_weight: L2 regularization on feature weights. For single sources,
                this is the main regularization. For grouped sources with
                per_source_alphas, set this to 1.0 (group scaling handles per-source).
            l2_residual: L2 regularization on residuals (high = encourage feature usage).
            l2_ability: L2 regularization on mean(abilities)² for identifiability.
            per_source_alphas: For GroupedFeatureSource only. Dict mapping source
                name to alpha value. Features are scaled by 1/sqrt(alpha), achieving
                differential regularization. Higher alpha = more regularization.
                If None and source is GroupedFeatureSource, uses source.group_alphas.
            lr: Learning rate for L-BFGS optimizer.
            max_iter: Maximum optimization iterations.
            tol: Convergence tolerance.
            device: Device for PyTorch tensors ("cpu" or "cuda"). For typical
                   experiment sizes (~100 agents, ~500 tasks), CPU is fast enough
                   (a few seconds). Use "cuda" for very large datasets.
            verbose: Print training progress.
        """
        self.source = source
        self.use_residuals = use_residuals
        self.init_from_baseline = init_from_baseline
        self.l2_weight = l2_weight
        self.l2_residual = l2_residual
        self.l2_ability = l2_ability
        self.per_source_alphas = per_source_alphas
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.device = device
        self.verbose = verbose

        # Model state (set after fit())
        self._weights: Optional[np.ndarray] = None  # (feature_dim,)
        self._bias: Optional[float] = None
        self._residuals: Optional[Dict[str, float]] = None  # task_id -> residual
        self._abilities: Optional[np.ndarray] = None  # (n_agents,)
        self._agent_ids: Optional[List[str]] = None
        self._train_task_ids: Optional[List[str]] = None
        self._scaler: Optional[StandardScaler] = None  # For single sources
        self._per_source_scalers: Dict[str, StandardScaler] = {}  # For grouped sources
        self._is_fitted: bool = False
        self._training_loss_history: List[float] = []
        self._loss_components: List[Dict[str, float]] = []

        # Baseline values for diagnostics (only set when init_from_baseline=True)
        self._baseline_b: Optional[np.ndarray] = None
        self._baseline_theta: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        """Human-readable predictor name."""
        if self.init_from_baseline:
            return f"Baseline-Init Feature-IRT ({self.source.name})"
        suffix = "w/ residuals" if self.use_residuals else "no residuals"
        return f"Feature-IRT ({self.source.name}, {suffix})"

    def _is_grouped_source(self) -> bool:
        """Check if source is a GroupedFeatureSource."""
        return isinstance(self.source, GroupedFeatureSource)

    def _get_per_source_alphas(self) -> Optional[Dict[str, float]]:
        """Get per-source alphas, using source defaults if not provided."""
        if not self._is_grouped_source():
            return None

        return self.per_source_alphas

    def _preprocess_features(
        self, X: np.ndarray, task_ids: List[str], fit_scaler: bool = True
    ) -> np.ndarray:
        """Apply per-source StandardScaler and group scaling.

        For single sources: StandardScaler on all features.
        For grouped sources: Per-source StandardScaler, then group scaling (1/sqrt(alpha)).

        Args:
            X: Raw features from source.get_features(task_ids)
            task_ids: Task IDs (for error messages)
            fit_scaler: If True, fit new scalers. If False, use existing scalers.

        Returns:
            X_scaled: Preprocessed features ready for optimization
        """
        if self._is_grouped_source():
            X_std = np.empty_like(X, dtype=np.float32)
            alphas = self._get_per_source_alphas()

            for src, slice_obj in zip(self.source.sources, self.source.group_slices):
                src_name = src.name

                if fit_scaler:
                    scaler = StandardScaler()
                    X_std[:, slice_obj] = scaler.fit_transform(X[:, slice_obj])
                    self._per_source_scalers[src_name] = scaler
                else:
                    if src_name not in self._per_source_scalers:
                        raise RuntimeError(
                            f"Scaler for source '{src_name}' not found. Call fit() first."
                        )
                    scaler = self._per_source_scalers[src_name]
                    X_std[:, slice_obj] = scaler.transform(X[:, slice_obj])

                # Group scaling: divide by sqrt(alpha) for per-source regularization
                # Higher alpha = smaller features = more regularization
                if alphas is not None:
                    alpha = alphas[src_name]
                    X_std[:, slice_obj] /= np.sqrt(alpha)

            return X_std
        else:
            # Single source: standard preprocessing
            if fit_scaler:
                self._scaler = StandardScaler()
                return self._scaler.fit_transform(X)
            else:
                if self._scaler is None:
                    raise RuntimeError("Scaler not found. Call fit() first.")
                return self._scaler.transform(X)

    def fit(
        self,
        task_ids: List[str],
        ground_truth_b: np.ndarray,
        responses: Dict[str, Dict[str, Any]],
        baseline_abilities: Optional[np.ndarray] = None,
        baseline_agent_ids: Optional[List[str]] = None,
    ) -> None:
        """Fit by maximizing IRT log-likelihood jointly with abilities.

        IMPORTANT: The responses dict should ONLY contain pre-frontier agents.
        This is enforced at the call site by filtering before passing.

        Args:
            task_ids: Training task IDs (should include ALL tasks for Experiment B).
            ground_truth_b: Baseline IRT difficulties. Used for initialization when
                init_from_baseline=True, otherwise ignored for API compatibility.
            responses: Pre-filtered response matrix {agent_id: {task_id: 0|1}}.
                       Must only contain agents that should be used for training.
            baseline_abilities: Baseline IRT abilities (required if init_from_baseline=True).
            baseline_agent_ids: Agent IDs corresponding to baseline_abilities
                (required if init_from_baseline=True).
        """
        try:
            import torch
            from torch.optim import LBFGS
            from torch.distributions import Bernoulli
        except ImportError:
            raise ImportError("PyTorch is required for FeatureIRTPredictor")

        # Determine device
        device = self.device
        if device == "cuda" and not torch.cuda.is_available():
            if self.verbose:
                print("   Warning: CUDA not available, falling back to CPU")
            device = "cpu"

        # Get agent IDs from responses
        agent_ids = list(responses.keys())
        n_agents = len(agent_ids)
        n_tasks = len(task_ids)

        if n_agents == 0:
            raise ValueError("No agents in responses")
        if n_tasks == 0:
            raise ValueError("No tasks to train on")

        # Store training task IDs
        self._train_task_ids = list(task_ids)
        self._agent_ids = agent_ids

        # Get features from source
        features = self.source.get_features(task_ids)  # (n_tasks, feature_dim)
        feature_dim = features.shape[1]

        # Preprocess features (per-source StandardScaler + group scaling for grouped sources)
        features_scaled = self._preprocess_features(features, task_ids, fit_scaler=True)

        # Build response matrix: (n_agents, n_tasks)
        response_matrix = np.full((n_agents, n_tasks), np.nan)
        for i, agent_id in enumerate(agent_ids):
            for j, task_id in enumerate(task_ids):
                resp = responses.get(agent_id, {}).get(task_id)
                if resp is not None:
                    response_matrix[i, j] = resp

        # ===== INITIALIZATION =====

        eps = 1e-3
        residual_init = None  # Will be set if init_from_baseline=True

        if self.init_from_baseline:
            # Initialize from Baseline IRT solution
            if ground_truth_b is None:
                raise ValueError("ground_truth_b required when init_from_baseline=True")
            if baseline_abilities is None or baseline_agent_ids is None:
                raise ValueError(
                    "baseline_abilities and baseline_agent_ids required when init_from_baseline=True"
                )

            # Store baseline for diagnostics
            self._baseline_b = ground_truth_b.copy()
            self._baseline_theta = baseline_abilities.copy()

            # Initialize residuals from baseline difficulties
            residual_init = ground_truth_b.astype(np.float32)

            # Initialize abilities from baseline (match agent order in responses)
            baseline_ability_dict = dict(zip(baseline_agent_ids, baseline_abilities))

            # Check all training agents are in baseline
            missing_agents = [a for a in agent_ids if a not in baseline_ability_dict]
            if missing_agents:
                raise ValueError(
                    f"init_from_baseline=True requires all training agents to have baseline abilities. "
                    f"Missing {len(missing_agents)} agents: {missing_agents[:5]}"
                )

            theta_init = np.array(
                [baseline_ability_dict[agent_id] for agent_id in agent_ids],
                dtype=np.float32,
            )

            # Initialize feature weights to zero (features contribute nothing initially)
            w_init = np.zeros(feature_dim, dtype=np.float32)
            bias_init = 0.0

            if self.verbose:
                print(f"   Baseline-Init: Initialized from {len(baseline_agent_ids)} baseline agents")

        else:
            # Default: Empirical initialization + Ridge warm-start

            # 1. Compute empirical task difficulty for warm-start
            task_difficulty_init = np.zeros(n_tasks)
            for j, task_id in enumerate(task_ids):
                total_successes = 0
                total_trials = 0
                for agent_resp in responses.values():
                    resp = agent_resp.get(task_id)
                    if resp is not None:
                        s, t = get_successes_trials(resp)
                        total_successes += s
                        total_trials += t
                if total_trials > 0:
                    acc = max(eps, min(1 - eps, total_successes / total_trials))
                    task_difficulty_init[j] = -np.log(acc / (1 - acc))  # -logit(acc)

            # 2. Warm-start feature weights via Ridge regression
            ridge = Ridge(alpha=1.0)
            ridge.fit(features_scaled, task_difficulty_init)
            w_init = ridge.coef_.astype(np.float32)
            bias_init = float(ridge.intercept_)

            # 3. Compute empirical agent ability for initialization
            theta_init = np.zeros(n_agents, dtype=np.float32)
            for i, agent_id in enumerate(agent_ids):
                agent_resp = responses.get(agent_id, {})
                correct = sum(agent_resp.values())
                total = len(agent_resp)
                if total > 0:
                    acc = max(eps, min(1 - eps, correct / total))
                    theta_init[i] = np.log(acc / (1 - acc))  # logit(acc)

        # Convert to PyTorch tensors
        features_tensor = torch.tensor(
            features_scaled, dtype=torch.float32, device=device
        )
        response_tensor = torch.tensor(
            response_matrix, dtype=torch.float32, device=device
        )
        mask = ~torch.isnan(response_tensor)

        # Initialize learnable parameters
        w = torch.tensor(w_init, requires_grad=True, device=device)
        b = torch.tensor([bias_init], requires_grad=True, device=device)
        theta = torch.tensor(theta_init, requires_grad=True, device=device)

        # Residuals: use_residuals or init_from_baseline (which requires residuals)
        use_residuals_effective = self.use_residuals or self.init_from_baseline
        if use_residuals_effective:
            if residual_init is not None:
                # Initialize from baseline difficulties
                residuals_tensor = torch.tensor(
                    residual_init, requires_grad=True, device=device
                )
            else:
                # Initialize to zeros
                residuals_tensor = torch.zeros(n_tasks, requires_grad=True, device=device)
            params = [w, b, theta, residuals_tensor]
        else:
            residuals_tensor = None
            params = [w, b, theta]

        # L-BFGS optimizer
        optim = LBFGS(
            params,
            lr=self.lr,
            max_iter=20,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        self._training_loss_history = []
        self._loss_components = []
        l2_w = self.l2_weight
        l2_r = self.l2_residual
        l2_a = self.l2_ability

        # For init_from_baseline: regularize deviation from Oracle, not deviation from 0
        # This keeps difficulty_latent close to Oracle while features learn a correction
        baseline_b_tensor = None
        if self.init_from_baseline and self._baseline_b is not None:
            baseline_b_tensor = torch.tensor(
                self._baseline_b.astype(np.float32), device=device
            )

        def closure():
            optim.zero_grad()

            # Compute difficulties: b_i = features @ w + bias + difficulty_latent
            # Where difficulty_latent is initialized from Oracle and regularized to stay close
            # The features (w @ f + bias) learn a correction term
            diff = torch.matmul(features_tensor, w) + b
            if residuals_tensor is not None:
                diff = diff + residuals_tensor

            # P(success) = sigmoid(theta - diff)
            # theta: (n_agents,) -> (n_agents, 1)
            # diff: (n_tasks,) -> (1, n_tasks)
            probs = torch.sigmoid(theta[:, None] - diff[None, :])

            # Negative log-likelihood (only for valid responses)
            nll = -Bernoulli(probs=probs[mask]).log_prob(response_tensor[mask]).mean()

            # Regularization components
            weight_reg = l2_w * torch.sum(w**2)
            # Difficulty latent regularization:
            # - If init_from_baseline: penalize deviation from Oracle (||r - oracle||²)
            # - Otherwise: penalize deviation from 0 (||r||²)
            if residuals_tensor is not None:
                if baseline_b_tensor is not None:
                    # Keep difficulty_latent close to Oracle initialization
                    residual_reg = l2_r * torch.sum((residuals_tensor - baseline_b_tensor)**2)
                else:
                    residual_reg = l2_r * torch.sum(residuals_tensor**2)
            else:
                residual_reg = torch.tensor(0.0)
            ability_reg = l2_a * (theta.mean() ** 2)

            loss = nll + weight_reg + residual_reg + ability_reg
            loss.backward()

            # Track loss components for diagnostics
            self._loss_components.append({
                'iter': len(self._loss_components),
                'nll': nll.item(),
                'weight_reg': weight_reg.item(),
                'residual_reg': residual_reg.item() if residuals_tensor is not None else 0.0,
                'ability_reg': ability_reg.item(),
                'total': loss.item(),
            })

            return loss

        # Training loop
        if self.verbose:
            print(f"   Feature-IRT Training: {n_tasks} tasks, {n_agents} agents")
            print(f"   Feature dim: {feature_dim}, Device: {device}")
            print(f"   Valid response pairs: {mask.sum().item()}")
            print(f"   Hyperparams: l2_weight={l2_w}, l2_residual={l2_r}")
            if self._is_grouped_source():
                alphas = self._get_per_source_alphas()
                print(f"   Grouped source: {[s.name for s in self.source.sources]}")
                print(f"   Per-source alphas: {alphas}")

        for iteration in range(self.max_iter):
            if iteration > 0:
                previous_loss = loss.clone().detach()

            loss = optim.step(closure)
            self._training_loss_history.append(loss.item())

            if iteration > 0:
                d_loss = abs((previous_loss - loss).item())

                if self.verbose and (iteration % 50 == 0 or iteration < 5):
                    print(
                        f"     Iter {iteration}: loss={loss.item():.6f}, d_loss={d_loss:.2e}"
                    )

                # Check convergence
                if d_loss < self.tol:
                    if self.verbose:
                        print(f"   Converged at iteration {iteration}")
                    break

        # Store learned parameters (move to CPU for numpy conversion)
        self._weights = w.detach().cpu().numpy()
        self._bias = b.detach().cpu().item()
        self._abilities = theta.detach().cpu().numpy()

        # Store residuals as dict for task_id lookup
        if residuals_tensor is not None:
            residuals_np = residuals_tensor.detach().cpu().numpy()
            self._residuals = {
                task_id: float(residuals_np[i])
                for i, task_id in enumerate(task_ids)
            }
        else:
            self._residuals = None

        self._is_fitted = True

        if self.verbose:
            print(f"   Final loss: {loss.item():.6f}")

    def predict(self, task_ids: List[str]) -> Dict[str, float]:
        """Predict difficulty for tasks.

        Args:
            task_ids: List of task identifiers to predict for.

        Returns:
            Dictionary mapping task_id -> predicted difficulty.

        Raises:
            RuntimeError: If predict() is called before fit().
            ValueError: If any task was not seen during training.
        """
        if not self._is_fitted:
            raise RuntimeError("Predictor must be fit before calling predict()")

        # Validate all tasks were seen during training
        train_task_set = set(self._train_task_ids)
        unseen = [t for t in task_ids if t not in train_task_set]
        if unseen:
            raise ValueError(
                f"FeatureIRTPredictor cannot predict on unseen tasks. "
                f"{len(unseen)} tasks were not in training set. "
                f"First few: {unseen[:3]}. "
                f"Use FeatureBasedPredictor for unseen task prediction."
            )

        # Get features from source
        features = self.source.get_features(task_ids)

        # Preprocess features using fitted scalers (same transform as training)
        features_scaled = self._preprocess_features(features, task_ids, fit_scaler=False)

        # Predict: diff = features @ w + bias + residual
        preds = features_scaled @ self._weights + self._bias

        # Add residuals
        if self._residuals is not None:
            for i, task_id in enumerate(task_ids):
                preds[i] += self._residuals[task_id]

        return {task_id: float(pred) for task_id, pred in zip(task_ids, preds)}

    @property
    def feature_weights(self) -> Optional[Dict[str, float]]:
        """Return feature weights if feature names are available."""
        if not self._is_fitted:
            return None

        feature_names = self.source.feature_names
        if feature_names is None:
            return None

        return {name: float(w) for name, w in zip(feature_names, self._weights)}

    @property
    def learned_abilities(self) -> Optional[Dict[str, float]]:
        """Return learned agent abilities."""
        if not self._is_fitted or self._agent_ids is None:
            return None

        return {
            agent_id: float(theta)
            for agent_id, theta in zip(self._agent_ids, self._abilities)
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the fitted model."""
        if not self._is_fitted:
            return {"is_fitted": False}

        info = {
            "is_fitted": True,
            "feature_source": self.source.name,
            "n_features": len(self._weights),
            "n_agents": len(self._abilities),
            "n_training_tasks": len(self._train_task_ids) if self._train_task_ids else 0,
            "use_residuals": self.use_residuals,
            "l2_weight": self.l2_weight,
            "l2_residual": self.l2_residual,
            "final_loss": (
                self._training_loss_history[-1]
                if self._training_loss_history
                else None
            ),
            "bias": self._bias,
        }

        # Add feature weights if names available
        feature_weights = self.feature_weights
        if feature_weights is not None:
            info["feature_weights"] = feature_weights

        # Add grouped source info if applicable
        if self._is_grouped_source():
            info["is_grouped_source"] = True
            info["source_names"] = [s.name for s in self.source.sources]
            info["per_source_alphas"] = self._get_per_source_alphas()
            info["source_dims"] = {
                s.name: s.feature_dim for s in self.source.sources
            }

        return info

    def print_model_summary(self) -> None:
        """Print a summary of the fitted model."""
        info = self.get_model_info()

        if not info["is_fitted"]:
            print("Model not fitted yet")
            return

        print(f"  Feature source: {info['feature_source']}")
        print(f"  Number of features: {info['n_features']}")
        print(f"  Number of agents: {info['n_agents']}")
        print(f"  Training tasks: {info['n_training_tasks']}")
        print(f"  Use residuals: {info['use_residuals']}")
        print(f"  Final loss: {info['final_loss']:.6f}")
        print(f"  Bias: {info['bias']:.4f}")

        if info.get("is_grouped_source"):
            print(f"  Grouped source: {info['source_names']}")
            print(f"  Per-source alphas: {info['per_source_alphas']}")
            print(f"  Source dimensions: {info['source_dims']}")

        if "feature_weights" in info:
            print("  Feature weights (sorted by |weight|):")
            sorted_weights = sorted(
                info["feature_weights"].items(),
                key=lambda x: abs(x[1]),
                reverse=True,
            )
            for name, weight in sorted_weights:
                print(f"    {name}: {weight:+.4f}")

    def get_training_diagnostics(self) -> Dict[str, Any]:
        """Get training diagnostics including loss history and components."""
        if not self._is_fitted:
            raise RuntimeError("Predictor must be fit before getting diagnostics")

        return {
            "loss_history": self._training_loss_history,
            "loss_components": self._loss_components,
            "n_iterations": len(self._training_loss_history),
            "final_loss": self._training_loss_history[-1] if self._training_loss_history else None,
        }

    def analyze_contributions(self) -> Dict[str, Any]:
        """Analyze variance contributions from features vs residuals.

        Computes what fraction of the predicted difficulty variance comes from
        the feature-based component vs the residual component.

        Returns:
            Dictionary with variance ratios and covariance.

        Raises:
            RuntimeError: If predictor is not fitted or residuals are not enabled.
        """
        if not self._is_fitted:
            raise RuntimeError("Predictor must be fit before analyzing contributions")

        if self._residuals is None:
            raise RuntimeError(
                "analyze_contributions() requires residuals to be enabled. "
                "This predictor was fitted with use_residuals=False."
            )

        # Get scaled features for training tasks
        features = self.source.get_features(self._train_task_ids)
        features_scaled = self._preprocess_features(features, self._train_task_ids, fit_scaler=False)

        # Compute feature-based difficulty component
        feature_component = features_scaled @ self._weights

        # Compute residual component
        residual_component = np.array([
            self._residuals[t] for t in self._train_task_ids
        ])

        # Total difficulty (excluding constant bias which doesn't affect variance)
        total_difficulty = feature_component + residual_component

        # Compute variances
        var_feature = float(np.var(feature_component))
        var_residual = float(np.var(residual_component))
        var_total = float(np.var(total_difficulty))

        # Compute covariance
        if len(feature_component) > 1:
            covariance = float(np.cov(feature_component, residual_component)[0, 1])
        else:
            covariance = 0.0

        return {
            "var_feature": var_feature,
            "var_residual": var_residual,
            "var_total": var_total,
            "feature_ratio": var_feature / var_total if var_total > 0 else 0.0,
            "residual_ratio": var_residual / var_total if var_total > 0 else 0.0,
            "covariance": covariance,
        }

    def get_baseline_init_diagnostics(self) -> Dict[str, float]:
        """Get diagnostics about drift from baseline initialization.

        Only available when init_from_baseline=True. Returns metrics showing:
        - Feature weight learning (are features contributing?)
        - Drift from baseline difficulties and abilities
        - Contribution analysis (feature vs residual variance)

        Returns:
            Dictionary with diagnostic metrics.

        Raises:
            RuntimeError: If predictor is not fitted or init_from_baseline=False.
        """
        if not self._is_fitted:
            raise RuntimeError("Predictor must be fit before getting diagnostics")

        if not self.init_from_baseline or self._baseline_b is None:
            raise RuntimeError(
                "get_baseline_init_diagnostics() only available when init_from_baseline=True"
            )

        if self._residuals is None:
            raise RuntimeError("Expected residuals to be enabled with init_from_baseline=True")

        # Get learned values
        residuals = np.array([self._residuals[t] for t in self._train_task_ids])
        theta = self._abilities

        # Feature component
        features = self.source.get_features(self._train_task_ids)
        features_scaled = self._preprocess_features(features, self._train_task_ids, fit_scaler=False)
        feature_component = features_scaled @ self._weights + self._bias

        # Variance contributions
        var_feature = float(np.var(feature_component))
        var_residual = float(np.var(residuals))

        return {
            # Feature weight diagnostics
            "weight_norm": float(np.linalg.norm(self._weights)),
            "weight_nonzero": int(np.sum(np.abs(self._weights) > 0.01)),
            "bias": float(self._bias),

            # Drift from baseline
            "difficulty_drift_mean": float(np.mean(np.abs(residuals - self._baseline_b))),
            "difficulty_drift_max": float(np.max(np.abs(residuals - self._baseline_b))),
            "ability_drift_mean": float(np.mean(np.abs(theta - self._baseline_theta))),
            "ability_drift_max": float(np.max(np.abs(theta - self._baseline_theta))),

            # Contribution analysis
            "var_feature_component": var_feature,
            "var_residual_component": var_residual,
            "feature_contribution_ratio": float(
                var_feature / (var_feature + var_residual + 1e-10)
            ),
        }


# =============================================================================
# Feature Source Builders
# =============================================================================


def build_feature_sources(
    config: DatasetConfig,
    embeddings_path_override: Optional[Path] = None,
    llm_judge_path_override: Optional[Path] = None,
    trajectory_features_path_override: Optional[Path] = None,
) -> List[Tuple[str, TaskFeatureSource]]:
    """Build list of available feature sources.

    Args:
        config: Dataset configuration
        embeddings_path_override: Optional path to override config embeddings
        llm_judge_path_override: Optional path to override config LLM judge features
        trajectory_features_path_override: Optional path to override config trajectory features

    Returns:
        List of (source_name, feature_source) tuples
    """
    embeddings_path = embeddings_path_override or config.embeddings_path
    llm_judge_path = llm_judge_path_override or config.llm_judge_path

    # Trajectory features are only available for SWE-bench Verified
    trajectory_features_path = trajectory_features_path_override or getattr(
        config, "trajectory_features_path", None
    )
    trajectory_feature_cols = getattr(config, "trajectory_feature_cols", None)

    return _build_feature_sources(
        embeddings_path=embeddings_path,
        llm_judge_path=llm_judge_path,
        llm_judge_feature_cols=config.llm_judge_feature_cols,
        trajectory_features_path=trajectory_features_path,
        trajectory_feature_cols=trajectory_feature_cols,
        verbose=True,
    )


# =============================================================================
# Prediction Collection Functions
# =============================================================================


def collect_ridge_predictions(
    feature_sources: List[Tuple[str, TaskFeatureSource]],
    train_task_ids: List[str],
    ground_truth_b: np.ndarray,
    all_task_ids: List[str],
) -> Dict[str, Dict[str, float]]:
    """Train Ridge regressors on available feature sources.

    Args:
        feature_sources: List of (source_name, feature_source) tuples
        train_task_ids: Task IDs for training
        ground_truth_b: Ground truth difficulties from baseline IRT
        all_task_ids: All task IDs to predict for

    Returns:
        Dict mapping method_name -> predictions dict
    """
    predictions: Dict[str, Dict[str, float]] = {}

    for source_name, source in feature_sources:
        method_name = f"{source_name} + Ridge"
        print(f"\nTraining {method_name}...")
        print(f"  Training on {len(train_task_ids)} tasks")
        try:
            predictor = FeatureBasedPredictor(source)
            predictor.fit(train_task_ids, ground_truth_b)
            predictions[method_name] = predictor.predict(all_task_ids)
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    return predictions


def collect_grouped_ridge_predictions(
    feature_sources: List[Tuple[str, TaskFeatureSource]],
    train_task_ids: List[str],
    ground_truth_b: np.ndarray,
    all_task_ids: List[str],
    alpha_grids: Optional[Dict[str, List[float]]] = None,
) -> Dict[str, Dict[str, float]]:
    """Train grouped ridge on combined feature sources with per-source regularization.

    This function creates grouped predictors for all pairwise combinations of
    feature sources, plus the full combination if there are 3+ sources.

    Args:
        feature_sources: List of (source_name, feature_source) tuples to combine
        train_task_ids: Task IDs for training
        ground_truth_b: Ground truth difficulties from baseline IRT
        all_task_ids: All task IDs to predict for
        alpha_grids: Optional per-source alpha grids {source_name: [alphas]}.
            If not provided, uses default grid [0.1, 1.0, 10.0, 100.0, 1000.0].

    Returns:
        Dict mapping method names to predictions for each combination
    """
    from itertools import combinations

    predictions: Dict[str, Dict[str, float]] = {}

    if len(feature_sources) < 2:
        # Need at least 2 sources to combine
        return predictions

    # Generate all combinations of size 2 to N
    all_combinations = []
    for r in range(2, len(feature_sources) + 1):
        all_combinations.extend(combinations(feature_sources, r))

    for source_combo in all_combinations:
        # Build grouped source from this combination
        combined = GroupedFeatureSource([source for _, source in source_combo])

        method_name = f"Grouped Ridge ({combined.name})"
        print(f"\nTraining {method_name}...")
        print(f"  Training on {len(train_task_ids)} tasks")
        print(f"  Sources: {[s.name for s in combined.sources]}")

        try:
            predictor = GroupedRidgePredictor(combined, alpha_grids=alpha_grids)
            predictor.fit(train_task_ids, ground_truth_b)

            info = predictor.get_model_info()
            for source_info in info["sources"]:
                print(f"    {source_info['name']}: best_alpha={source_info['best_alpha']}")

            predictions[method_name] = predictor.predict(all_task_ids)
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    return predictions


def collect_feature_irt_predictions(
    feature_sources: List[Tuple[str, TaskFeatureSource]],
    train_task_ids: List[str],
    ground_truth_b: np.ndarray,
    train_responses: Dict[str, Dict[str, int]],
    oracle_items: pd.DataFrame,
    oracle_abilities: pd.DataFrame,
    responses: Dict[str, Dict[str, int]],
    frontier_task_ids: List[str],
    anchor_task_ids: List[str],
    post_frontier_agents: List[str],
    baseline_abilities: Optional[pd.DataFrame] = None,
    alignment_method: str = "affine",
    include_baseline_init: bool = True,
    verbose: bool = False,
) -> FeatureIRTResults:
    """Run Feature-IRT with grid search over hyperparameters.

    For each feature source, runs grid search over regularization hyperparameters
    and initialization modes. Selects the configuration with best AUC.

    Args:
        feature_sources: List of (source_name, feature_source) tuples
        train_task_ids: Task IDs for training (all tasks)
        ground_truth_b: Ground truth difficulties from baseline IRT
        train_responses: Pre-filtered responses (pre-frontier agents only)
        oracle_items: Oracle IRT items for evaluation
        oracle_abilities: Oracle abilities for evaluation
        responses: Full responses for evaluation
        frontier_task_ids: Frontier tasks for metric computation
        anchor_task_ids: Anchor tasks for scale alignment
        post_frontier_agents: Post-frontier agents for evaluation
        baseline_abilities: Baseline IRT abilities (required for baseline-init)
        alignment_method: Scale alignment method
        include_baseline_init: If True and baseline_abilities available, include
            baseline-initialized variants in the grid search
        verbose: Print verbose output

    Returns:
        FeatureIRTResults with predictions, abilities, and baseline_init_diagnostics
    """
    from experiment_b.shared.evaluation import compute_method_metrics

    # Wide hyperparameter grid (same range for weights and residuals)
    l2_grid = [0.001, 0.01, 0.1, 1.0, 10.0]

    # Initialization modes: standard and optionally baseline-init
    can_baseline_init = baseline_abilities is not None and include_baseline_init
    init_from_baseline_grid = [False, True] if can_baseline_init else [False]

    # Prepare baseline abilities data if needed
    baseline_agent_ids = None
    baseline_theta = None
    if can_baseline_init:
        baseline_agent_ids = list(baseline_abilities.index)
        baseline_theta = baseline_abilities["theta"].values

    predictions: Dict[str, Dict[str, float]] = {}
    abilities: Dict[str, Dict[str, float]] = {}
    best_predictors: Dict[str, Any] = {}
    baseline_init_diagnostics: Dict[str, Dict[str, float]] = {}

    for source_name, source in feature_sources:
        print(f"\nRunning Feature-IRT grid search for {source_name}...")
        print(f"  Training on {len(train_task_ids)} tasks with {len(train_responses)} pre-frontier agents")
        n_configs = len(init_from_baseline_grid) * len(l2_grid) * len(l2_grid)
        print(f"  Grid: {n_configs} configurations")

        best_auc = -1.0
        best_config = None
        best_predictions: Optional[Dict[str, float]] = None
        best_abilities: Optional[Dict[str, float]] = None
        best_predictor: Optional[Any] = None

        for init_baseline in init_from_baseline_grid:
            for l2_w in l2_grid:
                for l2_r in l2_grid:
                    # When init_from_baseline, always use residuals (they hold baseline_b)
                    use_res = True if init_baseline else False

                    if verbose:
                        config_str = f"init_baseline={init_baseline}, l2_w={l2_w}, l2_r={l2_r}"
                        print(f"  Testing: {config_str}")

                    try:
                        predictor = FeatureIRTPredictor(
                            source,
                            use_residuals=use_res,
                            init_from_baseline=init_baseline,
                            l2_weight=l2_w,
                            l2_residual=l2_r,
                            verbose=False,
                        )

                        predictor.fit(
                            task_ids=train_task_ids,
                            ground_truth_b=ground_truth_b,
                            responses=train_responses,
                            baseline_abilities=baseline_theta,
                            baseline_agent_ids=baseline_agent_ids,
                        )

                        preds = predictor.predict(train_task_ids)

                        # Evaluate AUC for grid search
                        metrics = compute_method_metrics(
                            predicted_beta=preds,
                            oracle_items=oracle_items,
                            oracle_abilities=oracle_abilities,
                            responses=responses,
                            frontier_task_ids=frontier_task_ids,
                            anchor_task_ids=anchor_task_ids,
                            eval_agents=post_frontier_agents,
                            alignment_method=alignment_method,
                        )
                        auc = metrics.get('auc', 0) or 0

                        if verbose:
                            print(f"    AUC: {auc:.4f}")

                        if auc > best_auc:
                            best_auc = auc
                            best_config = (init_baseline, l2_w, l2_r)
                            best_predictions = preds
                            best_abilities = predictor.learned_abilities
                            best_predictor = predictor

                    except Exception as e:
                        if verbose:
                            print(f"    Error: {e}")

        # Store results
        if best_predictions is not None:
            init_baseline, l2_w, l2_r = best_config
            method_name = (
                f"Baseline-Init Feature-IRT ({source_name})"
                if init_baseline
                else f"Feature-IRT ({source_name})"
            )
            print(f"  Best: {method_name}, AUC={best_auc:.4f}, l2_w={l2_w}, l2_r={l2_r}")

            predictions[method_name] = best_predictions
            if best_abilities is not None:
                abilities[method_name] = best_abilities
            best_predictors[source_name] = best_predictor

            # Collect baseline-init diagnostics if applicable
            if init_baseline and best_predictor is not None:
                try:
                    baseline_init_diagnostics[source_name] = best_predictor.get_baseline_init_diagnostics()
                except Exception as e:
                    print(f"  Warning: Could not get baseline-init diagnostics: {e}")

    return FeatureIRTResults(
        predictions=predictions,
        abilities=abilities,
        diagnostics=None,
        best_predictors=best_predictors,
        baseline_init_diagnostics=baseline_init_diagnostics,
    )



"""Feature-IRT predictor: joint feature weight + ability learning.

Implements JointTrainingCVPredictor which jointly learns feature weights and
agent abilities by maximizing the IRT log-likelihood. This is an alternative
to the two-stage Ridge approach (train IRT → predict difficulty from features).

Supports per-source L2 regularization for GroupedFeatureSource.
"""

import itertools
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from experiment_ab_shared.dataset import ExperimentData
from experiment_ab_shared.evaluator import compute_irt_probability
from experiment_ab_shared.feature_source import TaskFeatureSource, GroupedFeatureSource


def feature_irt_predictor_factory(source_name, source, config):
    """Feature-IRT predictor factory for use with build_cv_predictors().

    Returns a JointTrainingCVPredictor that jointly learns feature weights
    and agent abilities, instead of the default Ridge regression approach.
    """
    return JointTrainingCVPredictor(source, verbose=False)


class JointTrainingCVPredictor:
    """Feature-IRT predictor that jointly learns feature weights and abilities.

    This predictor maximizes IRT log-likelihood:
        L = sum_ij log P(y_ij | theta_j, b_i)

    where b_i = w^T f_i + bias (no residual term).

    After training on train tasks, it can predict difficulty for unseen test tasks
    using only the learned feature weights, enabling cross-validation.

    Uses Bernoulli likelihood: y_ij ~ Bernoulli(sigmoid(theta_j - b_i))

    Hyperparameter selection:
    - Uses internal k-fold CV (default k=3) to select best L2 weights
    - For single sources: searches over a single L2 weight grid
    - For GroupedFeatureSource: searches over per-source L2 weight combinations

    Per-source regularization:
    - When using GroupedFeatureSource, applies different L2 penalties per source
    - Loss becomes: weight_reg = sum_g (l2_g * ||w_g||^2)
    - This allows high-dim embeddings and low-dim LLM features to have appropriate regularization
    """

    # Default L2 weight grid (similar range to RidgeCV alphas)
    DEFAULT_L2_WEIGHTS = [0.01, 0.1, 1.0, 10.0]

    # Per-source L2 grids (matching GroupedRidgePredictor.SOURCE_ALPHA_GRIDS)
    # High-dim features need stronger regularization, low-dim need weaker
    SOURCE_L2_GRIDS = {
        "Embedding": [100.0, 1000.0, 10000.0],      # High-dim: strong regularization
        "LLM Judge": [0.01, 0.1, 1.0, 10.0],        # Low-dim: weak regularization
    }

    def __init__(
        self,
        source: TaskFeatureSource,
        l2_weights: Optional[List[float]] = None,
        l2_ability: float = 0.01,
        inner_cv_folds: int = 3,
        lr: float = 0.1,
        max_iter: int = 500,
        tol: float = 1e-5,
        verbose: bool = False,
    ):
        """Initialize Feature-IRT CV predictor.

        Args:
            source: TaskFeatureSource providing features for tasks. Can be a single source
                or GroupedFeatureSource for per-source regularization.
            l2_weights: List of L2 regularization values to try for feature weights.
                Uses internal CV to select the best one. Defaults to [0.01, 0.1, 1.0, 10.0].
                For GroupedFeatureSource, this is ignored in favor of SOURCE_L2_GRIDS.
            l2_ability: L2 regularization on mean(abilities)^2 for identifiability.
            inner_cv_folds: Number of folds for internal CV to select l2_weight.
            lr: Learning rate for L-BFGS optimizer.
            max_iter: Maximum optimization iterations.
            tol: Convergence tolerance.
            verbose: Print training progress.
        """
        self.source = source
        self.l2_weights = l2_weights or self.DEFAULT_L2_WEIGHTS
        self.l2_ability = l2_ability
        self.inner_cv_folds = inner_cv_folds
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

        # Check if source is grouped (for per-source regularization)
        self._is_grouped = isinstance(source, GroupedFeatureSource)

        # Model state (set after fit())
        self._weights: Optional[np.ndarray] = None
        self._bias: Optional[float] = None
        # For grouped sources, use per-source scalers; otherwise single scaler
        self._scaler: Optional[StandardScaler] = None
        self._per_source_scalers: Optional[Dict[str, StandardScaler]] = None
        self._learned_abilities: Optional[Dict[str, float]] = None
        self._is_fitted: bool = False
        self._predicted_difficulties: Dict[str, float] = {}
        # For grouped sources, this becomes a dict; otherwise a single float
        self._best_l2_weight: Optional[Union[float, Dict[str, float]]] = None

    def _get_l2_grid_for_source(self, source_name: str) -> List[float]:
        """Get L2 weight grid for a specific source.

        Args:
            source_name: Name of the feature source (e.g., "Embedding", "LLM Judge")

        Returns:
            List of L2 weights to search over.

        Raises:
            ValueError: If source is not in SOURCE_L2_GRIDS.
        """
        if source_name in self.SOURCE_L2_GRIDS:
            return self.SOURCE_L2_GRIDS[source_name]
        raise ValueError(
            f"No L2 grid defined for source '{source_name}'. "
            f"Add it to SOURCE_L2_GRIDS or use a non-grouped source. "
            f"Known sources: {list(self.SOURCE_L2_GRIDS.keys())}"
        )

    def _scale_features(
        self,
        features: np.ndarray,
        fit: bool = True,
        scalers: Optional[Dict[str, StandardScaler]] = None,
    ) -> Tuple[np.ndarray, Dict[str, StandardScaler]]:
        """Scale features using per-source or single StandardScaler.

        Args:
            features: Raw feature matrix (n_tasks, n_features).
            fit: If True, fit scalers; if False, use provided scalers.
            scalers: Pre-fitted scalers (required if fit=False).

        Returns:
            Tuple of (scaled_features, scalers_dict).
            For grouped sources, scalers_dict maps source_name -> scaler.
            For single sources, scalers_dict has key "_single".
        """
        if self._is_grouped:
            if fit:
                fitted_scalers, X_std = self.source.fit_scalers(features)
                return X_std, fitted_scalers
            else:
                return self.source.apply_scalers(features, scalers), scalers
        else:
            if fit:
                scaler = StandardScaler()
                return scaler.fit_transform(features), {"_single": scaler}
            else:
                return scalers["_single"].transform(features), scalers

    def _prepare_data(
        self,
        data: ExperimentData,
        task_ids: List[str],
        agent_ids: List[str],
    ) -> dict:
        """Prepare response data for training.

        Returns dict with 'responses_binary' and 'agent_ids_with_responses'.
        """
        responses_binary: Dict[str, Dict[str, int]] = {}
        for agent_id in agent_ids:
            if agent_id not in data.responses:
                continue
            agent_responses = {}
            for task_id in task_ids:
                if task_id in data.responses[agent_id]:
                    agent_responses[task_id] = data.responses[agent_id][task_id]
            if agent_responses:
                responses_binary[agent_id] = agent_responses
        return {
            'responses_binary': responses_binary,
            'agent_ids_with_responses': list(responses_binary.keys()),
        }

    def _fit_single(
        self,
        features_scaled: np.ndarray,
        task_ids: List[str],
        prepared_data: dict,
        l2_weights: Union[float, Dict[str, float]],
    ) -> dict:
        """Fit model with given L2 weights.

        Args:
            features_scaled: Scaled feature matrix (n_tasks, n_features).
            task_ids: List of task IDs.
            prepared_data: Prepared response data from _prepare_data.
            l2_weights: Either a single float (for single source) or a dict
                mapping source_name -> L2 weight (for GroupedFeatureSource).

        Returns:
            Dict with 'weights', 'bias', 'abilities', 'final_nll'.
        """
        import torch
        from torch.optim import LBFGS
        from torch.distributions import Bernoulli

        agent_ids_with_responses = prepared_data['agent_ids_with_responses']
        responses_binary = prepared_data['responses_binary']
        n_agents = len(agent_ids_with_responses)
        n_tasks = len(task_ids)
        device = "cpu"

        # Prepare per-group L2 weights and slices for grouped sources
        if isinstance(l2_weights, dict):
            # Grouped source: per-source L2 weights
            group_slices = self.source.group_slices
            group_l2_weights = [l2_weights[s.name] for s in self.source.sources]
        else:
            # Single source: one L2 weight for all features
            group_slices = None
            group_l2_weights = None

        # Build response matrix
        response_matrix = np.full((n_agents, n_tasks), np.nan)
        for i, agent_id in enumerate(agent_ids_with_responses):
            for j, task_id in enumerate(task_ids):
                if task_id in responses_binary.get(agent_id, {}):
                    response_matrix[i, j] = responses_binary[agent_id][task_id]

        response_tensor = torch.tensor(response_matrix, dtype=torch.float32, device=device)
        mask = ~torch.isnan(response_tensor)

        # Compute empirical task pass rate for initialization
        eps = 1e-3
        task_difficulty_init = np.zeros(n_tasks)
        for j, task_id in enumerate(task_ids):
            successes = sum(
                1 for r in responses_binary.values() if r.get(task_id, None) == 1
            )
            total = sum(1 for r in responses_binary.values() if task_id in r)
            if total > 0:
                acc = max(eps, min(1 - eps, successes / total))
                task_difficulty_init[j] = -np.log(acc / (1 - acc))

        # Compute empirical agent ability
        theta_init = np.zeros(n_agents, dtype=np.float32)
        for i, agent_id in enumerate(agent_ids_with_responses):
            agent_resp = responses_binary.get(agent_id, {})
            correct = sum(agent_resp.values())
            total = len(agent_resp)
            if total > 0:
                acc = max(eps, min(1 - eps, correct / total))
                theta_init[i] = np.log(acc / (1 - acc))

        # Warm-start feature weights via Ridge regression
        ridge = Ridge(alpha=1.0)
        ridge.fit(features_scaled, task_difficulty_init)
        w_init = ridge.coef_.astype(np.float32)
        bias_init = float(ridge.intercept_)

        # Convert to PyTorch tensors
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32, device=device)

        # Initialize learnable parameters
        w = torch.tensor(w_init, requires_grad=True, device=device)
        b = torch.tensor([bias_init], requires_grad=True, device=device)
        theta = torch.tensor(theta_init, requires_grad=True, device=device)

        params = [w, b, theta]

        # L-BFGS optimizer
        optim = LBFGS(
            params,
            lr=self.lr,
            max_iter=20,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        l2_a = self.l2_ability
        final_nll = None

        # Define weight regularization function based on source type
        if group_slices is not None:
            # Per-group L2 regularization: sum_g (l2_g * ||w_g||^2)
            def compute_weight_reg():
                reg = torch.tensor(0.0, device=device)
                for slice_i, l2_i in zip(group_slices, group_l2_weights):
                    reg = reg + l2_i * torch.sum(w[slice_i] ** 2)
                return reg
        else:
            # Single L2 regularization
            l2_w = l2_weights  # Single float
            def compute_weight_reg():
                return l2_w * torch.sum(w ** 2)

        def closure():
            nonlocal final_nll
            optim.zero_grad()
            diff = torch.matmul(features_tensor, w) + b
            probs = torch.sigmoid(theta[:, None] - diff[None, :])
            nll = -Bernoulli(probs=probs[mask]).log_prob(response_tensor[mask]).mean()
            weight_reg = compute_weight_reg()
            ability_reg = l2_a * (theta.mean() ** 2)
            loss = nll + weight_reg + ability_reg
            loss.backward()
            final_nll = nll.item()
            return loss

        # Training loop
        for iteration in range(self.max_iter):
            if iteration > 0:
                previous_loss = loss.clone().detach()

            loss = optim.step(closure)

            if iteration > 0:
                d_loss = abs((previous_loss - loss).item())
                if d_loss < self.tol:
                    break

        abilities_dict = {
            agent_id: float(theta[i].detach().cpu().item())
            for i, agent_id in enumerate(agent_ids_with_responses)
        }

        return {
            'weights': w.detach().cpu().numpy(),
            'bias': b.detach().cpu().item(),
            'abilities': abilities_dict,
            'final_nll': final_nll,
        }

    def _compute_held_out_nll(
        self,
        weights: np.ndarray,
        bias: float,
        abilities: Dict[str, float],
        features_scaled: np.ndarray,
        task_ids: List[str],
        prepared_data: dict,
    ) -> float:
        """Compute negative log-likelihood on held-out data."""
        import torch
        from torch.distributions import Bernoulli

        agent_ids_with_responses = prepared_data['agent_ids_with_responses']
        responses_binary = prepared_data['responses_binary']

        # Filter to agents we have abilities for
        agent_ids_with_responses = [a for a in agent_ids_with_responses if a in abilities]
        if not agent_ids_with_responses:
            return float('inf')

        n_agents = len(agent_ids_with_responses)
        n_tasks = len(task_ids)
        device = "cpu"

        # Compute difficulties
        diff = features_scaled @ weights + bias

        response_matrix = np.full((n_agents, n_tasks), np.nan)
        for i, agent_id in enumerate(agent_ids_with_responses):
            for j, task_id in enumerate(task_ids):
                if task_id in responses_binary.get(agent_id, {}):
                    response_matrix[i, j] = responses_binary[agent_id][task_id]

        response_tensor = torch.tensor(response_matrix, dtype=torch.float32, device=device)
        mask = ~torch.isnan(response_tensor)

        if mask.sum() == 0:
            return float('inf')

        theta_arr = np.array([abilities[a] for a in agent_ids_with_responses])
        theta_tensor = torch.tensor(theta_arr, dtype=torch.float32, device=device)
        diff_tensor = torch.tensor(diff, dtype=torch.float32, device=device)

        probs = torch.sigmoid(theta_tensor[:, None] - diff_tensor[None, :])
        nll = -Bernoulli(probs=probs[mask]).log_prob(response_tensor[mask]).mean()
        return nll.item()

    def fit(self, data: ExperimentData, train_task_ids: List[str]) -> None:
        """Fit by maximizing IRT log-likelihood jointly with abilities.

        Uses internal k-fold CV to select the best L2 weights:
        - For single sources: searches over a single L2 weight grid
        - For GroupedFeatureSource: searches over per-source L2 weight combinations

        Args:
            data: ExperimentData containing responses and agent information
            train_task_ids: Task IDs to train on
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for JointTrainingCVPredictor")

        # Clear cached predictions for new fold
        self._predicted_difficulties = {}

        agent_ids = list(data.train_abilities.index)

        # Get features and scale
        features = self.source.get_features(train_task_ids)
        features_scaled, scalers = self._scale_features(features, fit=True)

        # Store scalers for predict_probability
        if self._is_grouped:
            self._per_source_scalers = scalers
        else:
            self._scaler = scalers["_single"]

        # Prepare full data
        full_prepared = self._prepare_data(data, train_task_ids, agent_ids)

        if len(full_prepared['agent_ids_with_responses']) == 0:
            raise ValueError("No agents with responses on training tasks")

        # Determine L2 weight candidates based on source type
        if self._is_grouped:
            # Grouped source: grid search over per-source L2 weight combinations
            source_names = [s.name for s in self.source.sources]
            source_grids = [self._get_l2_grid_for_source(name) for name in source_names]
            l2_combinations = list(itertools.product(*source_grids))
            l2_candidates = [
                {name: l2 for name, l2 in zip(source_names, combo)}
                for combo in l2_combinations
            ]
            if self.verbose:
                print(f"   Grouped source: {len(l2_combinations)} L2 weight combinations to search")
        else:
            l2_candidates = list(self.l2_weights)

        # Skip CV if only one candidate
        if len(l2_candidates) == 1:
            best_l2 = l2_candidates[0]
            if self.verbose:
                print(f"   Using single L2 weight: {best_l2}")
        else:
            # Internal CV to select best L2 weight(s)
            n_tasks = len(train_task_ids)
            k = min(self.inner_cv_folds, n_tasks)

            # Use deterministic fold assignment
            np.random.seed(42)
            fold_indices = np.random.permutation(n_tasks) % k

            # Store CV scores (use tuple for dict key when l2 is a dict)
            def to_key(l2):
                return tuple(sorted(l2.items())) if isinstance(l2, dict) else l2

            cv_scores: Dict = {to_key(l2): [] for l2 in l2_candidates}

            for fold_idx in range(k):
                inner_train_mask = fold_indices != fold_idx
                inner_val_mask = fold_indices == fold_idx

                inner_train_tasks = [train_task_ids[i] for i in range(n_tasks) if inner_train_mask[i]]
                inner_val_tasks = [train_task_ids[i] for i in range(n_tasks) if inner_val_mask[i]]

                if len(inner_train_tasks) == 0 or len(inner_val_tasks) == 0:
                    continue

                # Scale features for inner train/val (fit on inner train only)
                inner_train_features, inner_scalers = self._scale_features(
                    features[inner_train_mask], fit=True
                )
                inner_val_features, _ = self._scale_features(
                    features[inner_val_mask], fit=False, scalers=inner_scalers
                )

                inner_train_prepared = self._prepare_data(data, inner_train_tasks, agent_ids)
                inner_val_prepared = self._prepare_data(data, inner_val_tasks, agent_ids)

                if len(inner_train_prepared['agent_ids_with_responses']) == 0:
                    continue

                for l2 in l2_candidates:
                    result = self._fit_single(
                        inner_train_features,
                        inner_train_tasks,
                        inner_train_prepared,
                        l2,
                    )
                    val_nll = self._compute_held_out_nll(
                        result['weights'],
                        result['bias'],
                        result['abilities'],
                        inner_val_features,
                        inner_val_tasks,
                        inner_val_prepared,
                    )
                    cv_scores[to_key(l2)].append(val_nll)

            # Select best L2 weight(s) (lowest mean validation NLL)
            mean_scores = {
                key: np.mean(scores) if scores else float('inf')
                for key, scores in cv_scores.items()
            }
            best_key = min(mean_scores, key=mean_scores.get)
            best_l2 = dict(best_key) if isinstance(best_key, tuple) else best_key

            if self.verbose:
                print(f"   Best L2 weights: {best_l2}")
                print(f"   Best NLL: {mean_scores[best_key]:.6f}")

        self._best_l2_weight = best_l2

        # Final fit on all train data with best L2 weight(s)
        if self.verbose:
            print(f"   Final training with L2={best_l2}")

        result = self._fit_single(
            features_scaled,
            train_task_ids,
            full_prepared,
            best_l2,
        )

        self._weights = result['weights']
        self._bias = result['bias']
        self._learned_abilities = result['abilities']
        self._is_fitted = True

        if self.verbose:
            print(f"   Final NLL: {result['final_nll']:.6f}")

    def predict_probability(
        self, data: ExperimentData, agent_id: str, task_id: str
    ) -> float:
        """Predict success probability using IRT formula.

        Uses learned feature weights to predict difficulty, and jointly-learned
        abilities for theta.

        Args:
            data: ExperimentData (used for test_tasks list)
            agent_id: Agent identifier
            task_id: Task identifier

        Returns:
            Predicted probability of success
        """
        if not self._is_fitted:
            raise RuntimeError("Predictor must be fit before calling predict_probability()")

        # Lazily compute difficulties for test tasks
        if task_id not in self._predicted_difficulties:
            test_tasks = data.test_tasks
            features = self.source.get_features(test_tasks)

            # Scale using fitted scalers
            scalers = self._per_source_scalers if self._is_grouped else {"_single": self._scaler}
            features_scaled, _ = self._scale_features(features, fit=False, scalers=scalers)

            preds = features_scaled @ self._weights + self._bias
            self._predicted_difficulties.update({
                t: float(p) for t, p in zip(test_tasks, preds)
            })

        if task_id not in self._predicted_difficulties:
            raise ValueError(f"No predicted difficulty for task {task_id}")

        beta = self._predicted_difficulties[task_id]

        # Use jointly-learned abilities
        if agent_id not in self._learned_abilities:
            raise ValueError(
                f"Agent {agent_id} not found in learned abilities. "
                f"This agent may not have responses on training tasks."
            )

        theta = self._learned_abilities[agent_id]
        return compute_irt_probability(theta, beta)

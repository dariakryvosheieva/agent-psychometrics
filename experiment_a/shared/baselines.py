"""Baseline predictors and adapters for Experiment A evaluation.

All predictors implement the CVPredictor protocol from cross_validation.py,
allowing them to be used with the unified cross-validation framework.

This module provides:
1. AgentOnlyPredictor - Predicts based on agent's empirical success rate
2. ConstantPredictor - Predicts using mean training difficulty
3. OraclePredictor - Uses true IRT difficulties (upper bound)
4. DifficultyPredictorAdapter - Wraps any DifficultyPredictorBase to implement CVPredictor
5. FeatureIRTCVPredictor - Joint feature + IRT learning predictor
"""

import itertools
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from experiment_ab_shared.dataset import ExperimentData
from experiment_ab_shared.evaluator import compute_irt_probability
from experiment_ab_shared.feature_source import TaskFeatureSource, GroupedFeatureSource
from experiment_ab_shared.predictor_base import DifficultyPredictorBase


class DifficultyPredictorAdapter:
    """Adapts any DifficultyPredictorBase to the CVPredictor protocol.

    Takes a difficulty-based predictor and wraps it to provide predicted
    probabilities using the IRT formula: P(success) = sigmoid(θ - β).
    """

    def __init__(
        self,
        predictor: DifficultyPredictorBase,
        use_full_abilities: bool = False,
    ):
        """Initialize the adapter.

        Args:
            predictor: Any predictor implementing DifficultyPredictorBase
            use_full_abilities: If True, use full IRT abilities (oracle only)
        """
        self._predictor = predictor
        self._use_full_abilities = use_full_abilities
        self._predicted_difficulties: Dict[str, float] = {}

    def fit(self, data: ExperimentData, train_task_ids: List[str]) -> None:
        """Fit the underlying difficulty predictor."""
        ground_truth_b = data.train_items.loc[train_task_ids, "b"].values
        self._predictor.fit(train_task_ids, ground_truth_b)
        # Clear cached predictions for new fold
        self._predicted_difficulties = {}

    def predict_probability(
        self, data: ExperimentData, agent_id: str, task_id: str
    ) -> float:
        """Predict success probability using IRT formula."""
        # Lazily predict difficulty for this task if not already done
        if task_id not in self._predicted_difficulties:
            # Predict for all test tasks at once for efficiency
            test_tasks = data.test_tasks
            predictions = self._predictor.predict(test_tasks)
            self._predicted_difficulties.update(predictions)

        if task_id not in self._predicted_difficulties:
            raise ValueError(f"No predicted difficulty for task {task_id}")

        beta = self._predicted_difficulties[task_id]
        abilities = data.full_abilities if self._use_full_abilities else data.train_abilities

        if agent_id not in abilities.index:
            raise ValueError(f"Agent {agent_id} not found in abilities")

        theta = abilities.loc[agent_id, "ability"]
        return compute_irt_probability(theta, beta)


class AgentOnlyPredictor:
    """Baseline that predicts based on agent success rates.

    For each agent, computes their success rate on training tasks.
    At prediction time, ignores task difficulty and returns the
    agent's empirical success rate.
    """

    def __init__(self) -> None:
        self._agent_success_rates: Dict[str, float] = {}

    def fit(self, data: ExperimentData, train_task_ids: List[str]) -> None:
        """Compute agent success rates on training tasks."""
        self._agent_success_rates = {}

        for agent_id in data.train_abilities.index:
            if agent_id not in data.responses:
                continue

            # Compute success rate using expand_for_auc to handle binary/binomial
            all_outcomes: List[int] = []
            for task_id in train_task_ids:
                if task_id in data.responses[agent_id]:
                    outcomes, _ = data.expand_for_auc(agent_id, task_id, 0.0)
                    all_outcomes.extend(outcomes)

            if all_outcomes:
                self._agent_success_rates[agent_id] = float(np.mean(all_outcomes))
            else:
                self._agent_success_rates[agent_id] = 0.5

    def predict_probability(
        self, data: ExperimentData, agent_id: str, task_id: str
    ) -> float:
        """Return agent's empirical success rate (ignores task)."""
        if agent_id not in self._agent_success_rates:
            raise ValueError(f"No success rate computed for agent {agent_id}")
        return self._agent_success_rates[agent_id]


class ConstantPredictor:
    """Baseline that predicts the mean training difficulty for all tasks.

    Uses IRT formula: P(success) = sigmoid(θ - β_mean)
    """

    def __init__(self) -> None:
        self._mean_difficulty: float = 0.0

    def fit(self, data: ExperimentData, train_task_ids: List[str]) -> None:
        """Compute mean difficulty from training tasks."""
        difficulties = data.get_train_difficulties()
        self._mean_difficulty = float(np.mean(difficulties))

    def predict_probability(
        self, data: ExperimentData, agent_id: str, task_id: str
    ) -> float:
        """Return P(success) using mean difficulty and agent's ability."""
        from experiment_ab_shared.evaluator import compute_irt_probability

        theta = data.train_abilities.loc[agent_id, "ability"]
        return compute_irt_probability(theta, self._mean_difficulty)


class OraclePredictor:
    """Oracle baseline that uses true IRT difficulties from full model.

    This represents the best possible performance given the IRT framework.
    Uses the full IRT model (trained on all data) for evaluation.
    """

    def __init__(self) -> None:
        pass

    def fit(self, data: ExperimentData, train_task_ids: List[str]) -> None:
        """No training needed - uses oracle difficulties."""
        pass

    def predict_probability(
        self, data: ExperimentData, agent_id: str, task_id: str
    ) -> float:
        """Return P(success) using true difficulty and oracle ability."""
        from experiment_ab_shared.evaluator import compute_irt_probability

        # Use full (oracle) abilities and difficulties
        theta = data.full_abilities.loc[agent_id, "ability"]
        beta = data.full_items.loc[task_id, "b"]
        return compute_irt_probability(theta, beta)


class FeatureIRTCVPredictor:
    """Feature-IRT predictor that jointly learns feature weights and abilities.

    This predictor maximizes IRT log-likelihood:
        L = sum_ij log P(y_ij | theta_j, b_i)

    where b_i = w^T f_i + bias (no residual term).

    After training on train tasks, it can predict difficulty for unseen test tasks
    using only the learned feature weights, enabling cross-validation.

    Supports both:
    - Binary data (Bernoulli likelihood): y_ij ~ Bernoulli(sigmoid(theta_j - b_i))
    - Binomial data (Binomial likelihood): k_ij ~ Binomial(n_ij, sigmoid(theta_j - b_i))

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
        "Trajectory": [0.01, 0.1, 1.0, 10.0],       # Low-dim: weak regularization
        "Environment": [0.01, 0.1, 1.0, 10.0],      # Low-dim: weak regularization
        "Auditor": [0.01, 0.1, 1.0, 10.0],          # Low-dim: weak regularization
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
            out_scalers: Dict[str, StandardScaler] = {}
            features_scaled = np.empty_like(features)
            for source, slice_obj in zip(self.source.sources, self.source.group_slices):
                if fit:
                    scaler = StandardScaler()
                    features_scaled[:, slice_obj] = scaler.fit_transform(features[:, slice_obj])
                    out_scalers[source.name] = scaler
                else:
                    scaler = scalers[source.name]
                    features_scaled[:, slice_obj] = scaler.transform(features[:, slice_obj])
                    out_scalers[source.name] = scaler
            return features_scaled, out_scalers
        else:
            if fit:
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features)
                return features_scaled, {"_single": scaler}
            else:
                scaler = scalers["_single"]
                return scaler.transform(features), scalers

    def _prepare_data(
        self,
        data: ExperimentData,
        task_ids: List[str],
        agent_ids: List[str],
    ) -> dict:
        """Prepare response data for training.

        Returns dict with keys depending on data type:
        - Binary: 'responses_binary', 'agent_ids_with_responses'
        - Binomial: 'responses_counts', 'agent_ids_with_responses'
        """
        from experiment_ab_shared.dataset import BinomialExperimentData
        is_binomial = isinstance(data, BinomialExperimentData)

        if is_binomial:
            responses_counts: Dict[str, Dict[str, tuple]] = {}
            for agent_id in agent_ids:
                if agent_id not in data.responses:
                    continue
                agent_responses = {}
                for task_id in task_ids:
                    if task_id in data.responses[agent_id]:
                        resp = data.responses[agent_id][task_id]
                        agent_responses[task_id] = (resp["successes"], resp["trials"])
                if agent_responses:
                    responses_counts[agent_id] = agent_responses
            return {
                'is_binomial': True,
                'responses_counts': responses_counts,
                'agent_ids_with_responses': list(responses_counts.keys()),
            }
        else:
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
                'is_binomial': False,
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
        from torch.distributions import Bernoulli, Binomial

        is_binomial = prepared_data['is_binomial']
        agent_ids_with_responses = prepared_data['agent_ids_with_responses']
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

        if is_binomial:
            responses_counts = prepared_data['responses_counts']

            # Build counts and trials matrices
            counts_matrix = np.full((n_agents, n_tasks), np.nan)
            trials_matrix = np.full((n_agents, n_tasks), np.nan)
            for i, agent_id in enumerate(agent_ids_with_responses):
                for j, task_id in enumerate(task_ids):
                    if task_id in responses_counts.get(agent_id, {}):
                        k, n = responses_counts[agent_id][task_id]
                        counts_matrix[i, j] = k
                        trials_matrix[i, j] = n

            counts_tensor = torch.tensor(counts_matrix, dtype=torch.float32, device=device)
            trials_tensor = torch.tensor(trials_matrix, dtype=torch.float32, device=device)
            mask = ~torch.isnan(counts_tensor)

            # Compute empirical task pass rate for initialization
            eps = 1e-3
            task_difficulty_init = np.zeros(n_tasks)
            for j, task_id in enumerate(task_ids):
                total_successes = 0
                total_trials = 0
                for agent_id in agent_ids_with_responses:
                    if task_id in responses_counts.get(agent_id, {}):
                        k, n = responses_counts[agent_id][task_id]
                        total_successes += k
                        total_trials += n
                if total_trials > 0:
                    acc = max(eps, min(1 - eps, total_successes / total_trials))
                    task_difficulty_init[j] = -np.log(acc / (1 - acc))

            # Compute empirical agent ability
            theta_init = np.zeros(n_agents, dtype=np.float32)
            for i, agent_id in enumerate(agent_ids_with_responses):
                agent_resp = responses_counts.get(agent_id, {})
                total_successes = sum(k for k, n in agent_resp.values())
                total_trials = sum(n for k, n in agent_resp.values())
                if total_trials > 0:
                    acc = max(eps, min(1 - eps, total_successes / total_trials))
                    theta_init[i] = np.log(acc / (1 - acc))
        else:
            responses_binary = prepared_data['responses_binary']

            # Binary data
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

        if is_binomial:
            def closure():
                nonlocal final_nll
                optim.zero_grad()
                diff = torch.matmul(features_tensor, w) + b
                logits = theta[:, None] - diff[None, :]
                nll = -Binomial(total_count=trials_tensor[mask], logits=logits[mask]).log_prob(
                    counts_tensor[mask]
                ).mean()
                weight_reg = compute_weight_reg()
                ability_reg = l2_a * (theta.mean() ** 2)
                loss = nll + weight_reg + ability_reg
                loss.backward()
                final_nll = nll.item()
                return loss
        else:
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
        from torch.distributions import Bernoulli, Binomial

        is_binomial = prepared_data['is_binomial']
        agent_ids_with_responses = prepared_data['agent_ids_with_responses']

        # Filter to agents we have abilities for
        agent_ids_with_responses = [a for a in agent_ids_with_responses if a in abilities]
        if not agent_ids_with_responses:
            return float('inf')

        n_agents = len(agent_ids_with_responses)
        n_tasks = len(task_ids)
        device = "cpu"

        # Compute difficulties
        diff = features_scaled @ weights + bias

        if is_binomial:
            responses_counts = prepared_data['responses_counts']
            counts_matrix = np.full((n_agents, n_tasks), np.nan)
            trials_matrix = np.full((n_agents, n_tasks), np.nan)
            for i, agent_id in enumerate(agent_ids_with_responses):
                for j, task_id in enumerate(task_ids):
                    if task_id in responses_counts.get(agent_id, {}):
                        k, n = responses_counts[agent_id][task_id]
                        counts_matrix[i, j] = k
                        trials_matrix[i, j] = n

            counts_tensor = torch.tensor(counts_matrix, dtype=torch.float32, device=device)
            trials_tensor = torch.tensor(trials_matrix, dtype=torch.float32, device=device)
            mask = ~torch.isnan(counts_tensor)

            if mask.sum() == 0:
                return float('inf')

            theta_arr = np.array([abilities[a] for a in agent_ids_with_responses])
            theta_tensor = torch.tensor(theta_arr, dtype=torch.float32, device=device)
            diff_tensor = torch.tensor(diff, dtype=torch.float32, device=device)

            logits = theta_tensor[:, None] - diff_tensor[None, :]
            nll = -Binomial(total_count=trials_tensor[mask], logits=logits[mask]).log_prob(
                counts_tensor[mask]
            ).mean()
            return nll.item()
        else:
            responses_binary = prepared_data['responses_binary']
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
            raise ImportError("PyTorch is required for FeatureIRTCVPredictor")

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


class FullFeatureIRTAdapter:
    """Adapter for FeatureIRTPredictor that trains on ALL tasks like Oracle.

    This predictor tests whether task features add information beyond the IRT
    response matrix itself. Unlike FeatureIRTCVPredictor which trains only on
    train tasks, this trains on ALL tasks with per-task residuals.

    The model learns: b_i = w^T f_i + bias + r_i
    where r_i is a per-task residual that can capture task-specific difficulty
    not explained by features.

    Comparison:
    - Oracle: Uses pre-computed full IRT difficulties
    - FullFeatureIRT: Learns difficulties jointly with feature weights + residuals

    If FullFeatureIRT beats Oracle, features provide regularization benefit even
    with full response data.

    Wraps experiment_b's FeatureIRTPredictor to conform to CVPredictor protocol.
    """

    def __init__(
        self,
        source: TaskFeatureSource,
        l2_weight: float = 0.01,
        l2_residual: float = 10.0,
        l2_ability: float = 0.01,
        use_residuals: bool = True,
        verbose: bool = False,
    ):
        """Initialize Full Feature-IRT adapter.

        Args:
            source: TaskFeatureSource providing features for tasks.
            l2_weight: L2 regularization on feature weights.
            l2_residual: L2 regularization on per-task residuals.
            l2_ability: L2 regularization on mean(abilities)^2.
            use_residuals: Include per-task residuals (default True for full training).
            verbose: Print training progress.
        """
        self.source = source
        self.l2_weight = l2_weight
        self.l2_residual = l2_residual
        self.l2_ability = l2_ability
        self.use_residuals = use_residuals
        self.verbose = verbose

        # Model state (set after fit())
        self._predictor = None
        self._difficulties: Dict[str, float] = {}
        self._is_fitted: bool = False

    @property
    def name(self) -> str:
        """Human-readable predictor name."""
        return f"Full Feature-IRT ({self.source.name})"

    def fit(self, data: ExperimentData, train_task_ids: List[str]) -> None:
        """Train on ALL tasks (ignores train_task_ids).

        Unlike other predictors, this uses ALL tasks in data.full_items,
        similar to how OraclePredictor uses the full IRT model.

        Training happens only once - subsequent calls are no-ops for speed.
        WARNING: If you change hyperparameters (l2_weight, l2_residual, etc.),
        you must create a new FullFeatureIRTAdapter instance. Reusing an already-fitted
        instance with different hyperparameters will NOT retrain the model.

        Args:
            data: ExperimentData containing responses and IRT parameters
            train_task_ids: Ignored - we train on all tasks
        """
        # Skip if already fitted (trains once, reused across folds for speed)
        # Note: This means changing hyperparameters requires a new instance
        if self._is_fitted:
            if self.verbose:
                print(f"  Full Feature-IRT ({self.source.name}) already fitted, skipping training")
            return

        # Import here to avoid circular imports
        from experiment_b.shared.prediction_methods import FeatureIRTPredictor

        # Get ALL task IDs (like Oracle)
        all_task_ids = list(data.full_items.index)

        # Get ground truth difficulties from full IRT (for initialization)
        ground_truth_b = data.full_items.loc[all_task_ids, "b"].values

        # Build responses dict for Feature-IRT
        # Use ALL agents (not just train_abilities)
        responses: Dict[str, Dict[str, any]] = {}
        for agent_id in data.responses:
            agent_resp = {}
            for task_id in all_task_ids:
                if task_id in data.responses[agent_id]:
                    agent_resp[task_id] = data.responses[agent_id][task_id]
            if agent_resp:
                responses[agent_id] = agent_resp

        if self.verbose:
            print(f"  Training Full Feature-IRT ({self.source.name}) on {len(all_task_ids)} tasks, {len(responses)} agents")

        # Create and fit the predictor
        # Use init_from_baseline=False for empirical initialization
        # This initializes abilities from empirical agent performance and
        # difficulties from empirical task difficulty, then learns corrections
        self._predictor = FeatureIRTPredictor(
            source=self.source,
            use_residuals=self.use_residuals,
            init_from_baseline=False,
            l2_weight=self.l2_weight,
            l2_residual=self.l2_residual,
            l2_ability=self.l2_ability,
            verbose=self.verbose,
        )

        self._predictor.fit(
            task_ids=all_task_ids,
            ground_truth_b=ground_truth_b,
            responses=responses,
        )

        # Cache predictions for all tasks
        # NOTE: predict() returns w^T f_i + bias + r_i (includes residuals when use_residuals=True)
        # This is the full learned difficulty, not just the feature-based part
        self._difficulties = self._predictor.predict(all_task_ids)
        self._is_fitted = True

    def predict_probability(
        self, data: ExperimentData, agent_id: str, task_id: str
    ) -> float:
        """Predict success probability using IRT formula.

        Uses the full learned difficulty (w^T f_i + bias + r_i) and jointly-learned
        agent abilities.

        Args:
            data: ExperimentData (unused - we use learned abilities)
            agent_id: Agent identifier
            task_id: Task identifier

        Returns:
            Predicted probability of success
        """
        if not self._is_fitted:
            raise RuntimeError("Predictor must be fit before calling predict_probability()")

        if task_id not in self._difficulties:
            raise ValueError(f"No predicted difficulty for task {task_id}")

        # beta = w^T f_i + bias + r_i (full learned difficulty including residual)
        beta = self._difficulties[task_id]

        # Use jointly-learned abilities
        learned_abilities = self._predictor.learned_abilities
        if learned_abilities is None:
            raise RuntimeError("Predictor has no learned abilities")

        if agent_id not in learned_abilities:
            raise ValueError(
                f"Agent {agent_id} not found in learned abilities. "
                f"This agent may not have responses in training data."
            )

        theta = learned_abilities[agent_id]
        return compute_irt_probability(theta, beta)

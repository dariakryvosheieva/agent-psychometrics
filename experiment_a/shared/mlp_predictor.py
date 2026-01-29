"""MLP predictors for P(success) prediction from (agent, task) pairs.

This module contains two MLP architectures:

1. IRTStyleMLP / MLPPredictor (SANITY CHECK ONLY):
   - Mirrors the IRT formula: P = sigmoid(θ_agent - β_task)
   - IMPORTANT: This was only used to verify the IRT approach works with BCE loss.
   - By design, it CANNOT exceed IRT performance since it's structurally identical.
   - DO NOT USE for experiments trying to beat Ridge/IRT baselines.

2. SimpleMLP / FullMLPPredictor (PRIMARY ARCHITECTURE):
   - Takes [agent_one_hot | task_features] through a hidden layer
   - Can learn arbitrary agent-task interactions (not limited to θ - β)
   - Use this for experiments trying to beat Ridge regression.
   - Key: needs strong regularization (weight_decay=100-1000) for embeddings.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from experiment_ab_shared.dataset import BinomialExperimentData, ExperimentData
from experiment_ab_shared.feature_source import TaskFeatureSource


def build_input_vector(
    agent_idx: int,
    n_agents: int,
    task_features: np.ndarray,
) -> np.ndarray:
    """Build input vector for MLP: [agent_one_hot | task_features].

    Args:
        agent_idx: Index of the agent in the agent list.
        n_agents: Total number of agents.
        task_features: Scaled task feature vector.

    Returns:
        Concatenated input vector of shape (n_agents + feature_dim,).
    """
    agent_one_hot = np.zeros(n_agents, dtype=np.float32)
    agent_one_hot[agent_idx] = 1.0
    return np.concatenate([agent_one_hot, task_features])


class SimpleMLP(nn.Module):
    """Simple 2-layer MLP that takes concatenated [agent_one_hot | task_features].

    Architecture:
        Input: [agent_one_hot (n_agents) | task_features (feature_dim)]
        → Linear(input_dim, hidden_size)
        → ReLU
        → Dropout (if dropout > 0)
        → Linear(hidden_size, 1)
        → Sigmoid
        Output: P(success) in [0, 1]

    This is a standard MLP that can learn arbitrary interactions between
    agent identity and task features. Unlike IRTStyleMLP, it doesn't impose
    the IRT structure (θ - β).
    """

    def __init__(self, input_dim: int, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x.squeeze(-1)


class DeepMLP(nn.Module):
    """Deeper MLP with multiple hidden layers.

    Architecture:
        Input → Linear → ReLU → Dropout → Linear → ReLU → Dropout → ... → Linear → Sigmoid

    Multiple narrower layers can regularize better than a single wide layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: List[int],
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = []
        prev_size = input_dim

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class SwiGLUMLP(nn.Module):
    """MLP using SwiGLU activation instead of ReLU.

    SwiGLU: SwiGLU(x, W, V) = Swish(xW) ⊙ (xV)
    where Swish(x) = x * sigmoid(x)

    This is the activation used in modern LLMs like LLaMA and has been shown
    to improve training dynamics.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        # SwiGLU uses two projections: one for gate, one for value
        self.gate_proj = nn.Linear(input_dim, hidden_size, bias=False)
        self.up_proj = nn.Linear(input_dim, hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: swish(gate) * up
        gate = self.gate_proj(x)
        gate = gate * torch.sigmoid(gate)  # Swish activation
        up = self.up_proj(x)
        hidden = gate * up
        hidden = self.dropout(hidden)
        out = self.down_proj(hidden)
        return self.sigmoid(out).squeeze(-1)


class DualPathMLP(nn.Module):
    """Dual-path MLP that processes different feature sources separately before combining.

    Architecture:
        Embedding path: embeddings → Linear → ReLU → hidden_emb
        Judge path: judge_features → Linear → ReLU → hidden_judge
        Combined: [agent_one_hot | hidden_emb | hidden_judge] → Linear → ReLU → Linear → Sigmoid

    This allows the network to learn different representations for each feature
    type before combining them with agent information.
    """

    def __init__(
        self,
        n_agents: int,
        emb_dim: int,
        judge_dim: int,
        emb_hidden: int = 64,
        judge_hidden: int = 16,
        combined_hidden: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        # Embedding path
        self.emb_path = nn.Sequential(
            nn.Linear(emb_dim, emb_hidden),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        # Judge path
        self.judge_path = nn.Sequential(
            nn.Linear(judge_dim, judge_hidden),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        # Combined path: agent_one_hot + both hidden representations
        combined_input = n_agents + emb_hidden + judge_hidden
        self.combined_path = nn.Sequential(
            nn.Linear(combined_input, combined_hidden),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(combined_hidden, 1),
            nn.Sigmoid(),
        )

        self.n_agents = n_agents
        self.emb_dim = emb_dim
        self.judge_dim = judge_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, n_agents + emb_dim + judge_dim)
               where the layout is [agent_one_hot | embeddings | judge_features]

        Returns:
            Probabilities of shape (batch,)
        """
        # Split input into components
        agent_one_hot = x[:, :self.n_agents]
        emb_features = x[:, self.n_agents:self.n_agents + self.emb_dim]
        judge_features = x[:, self.n_agents + self.emb_dim:]

        # Process each path
        emb_hidden = self.emb_path(emb_features)
        judge_hidden = self.judge_path(judge_features)

        # Combine all
        combined = torch.cat([agent_one_hot, emb_hidden, judge_hidden], dim=1)
        return self.combined_path(combined).squeeze(-1)


class IRTStyleMLP(nn.Module):
    """IRT-style architecture that explicitly learns θ_agent - β_task.

    IMPORTANT: This is a SANITY CHECK architecture only. It cannot exceed IRT
    performance by design since it's structurally identical to IRT. Use
    SimpleMLP/FullMLPPredictor instead for experiments trying to beat baselines.

    Mirrors the IRT formula: P = sigmoid(θ_agent - β_task)
    - Agent pathway: agent_one_hot → θ (learned ability per agent)
    - Feature pathway: task_features → β (learned difficulty from features)
    - Output: sigmoid(θ - β)

    This is structurally identical to IRT, just learned end-to-end with BCE loss.
    """

    def __init__(self, n_agents: int, feature_dim: int, dropout: float = 0.0):
        super().__init__()
        # Agent abilities: one θ per agent (like IRT)
        self.agent_abilities = nn.Embedding(n_agents, 1)

        # Difficulty from features: features → scalar β
        self.difficulty_layer = nn.Linear(feature_dim, 1)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.sigmoid = nn.Sigmoid()

    def forward(
        self, agent_indices: torch.Tensor, task_features: torch.Tensor
    ) -> torch.Tensor:
        # Get agent abilities: (batch,) → (batch, 1)
        theta = self.agent_abilities(agent_indices)  # (batch, 1)

        # Get task difficulties: (batch, feature_dim) → (batch, 1)
        beta = self.difficulty_layer(task_features)  # (batch, 1)
        beta = self.dropout(beta)

        # IRT formula: P = sigmoid(θ - β)
        logit = theta - beta  # (batch, 1)
        prob = self.sigmoid(logit)

        return prob.squeeze(-1)  # (batch,)


class MLPPredictor:
    """IRT-style predictor that learns P = sigmoid(θ_agent - β_task) end-to-end.

    This predictor mirrors the IRT formula exactly:
    - θ: learned ability per agent (via embedding)
    - β: learned difficulty from task features (via linear layer)
    - P = sigmoid(θ - β)

    The key difference from standard IRT is that β is predicted from features
    rather than learned as free parameters per task.

    Training uses binary cross-entropy loss with Adam optimizer.
    """

    def __init__(
        self,
        source: TaskFeatureSource,
        hidden_size: int = 64,  # Unused, kept for API compatibility
        dropout: float = 0.0,
        learning_rate: float = 0.001,
        agent_lr_scale: float = 1.0,
        weight_decay: float = 0.01,
        feature_weight_decay: Optional[float] = None,
        n_epochs: int = 200,
        verbose: bool = False,
        freeze_abilities: bool = False,
        two_stage: bool = False,
        stage1_epochs: Optional[int] = None,
        stage2_agent_lr_scale: float = 0.1,
        pca_dim: Optional[int] = None,
        early_stopping: bool = False,
        val_fraction: float = 0.1,
        patience: int = 20,
    ):
        """Initialize IRT-style MLP predictor.

        Args:
            source: TaskFeatureSource providing features for tasks.
            hidden_size: Unused (kept for API compatibility).
            dropout: Dropout probability on difficulty prediction (0.0 = no dropout).
            learning_rate: Learning rate for feature weights (Adam optimizer).
            agent_lr_scale: Scale factor for agent learning rate relative to learning_rate.
                Use < 1.0 (e.g., 0.01-0.1) to slow down agent learning and prevent
                gradient competition where agents dominate over features.
            weight_decay: L2 regularization on agent abilities (Adam weight_decay).
            feature_weight_decay: L2 regularization on feature weights. If None, uses weight_decay.
                Use higher values (e.g., 1.0-10.0) for high-dim features like embeddings.
            n_epochs: Number of training epochs.
            verbose: Print training progress.
            freeze_abilities: If True, initialize agent abilities from IRT and freeze them.
                This isolates the difficulty layer learning and prevents agent memorization.
            two_stage: If True, use two-stage training:
                Stage 1: Initialize agents from IRT, freeze them, train features only
                Stage 2: Unfreeze agents, fine-tune both with low agent LR
                This combines good initialization with joint adaptation.
            stage1_epochs: Epochs for stage 1. If None, uses n_epochs // 2.
            stage2_agent_lr_scale: Learning rate scale for agents in stage 2 (default 0.1).
            pca_dim: If set, apply PCA dimensionality reduction to features before training.
                PCA is fit on training tasks only to avoid data leakage.
            early_stopping: If True, use validation-based early stopping.
            val_fraction: Fraction of training data to hold out for validation (default: 0.1).
            patience: Number of epochs without improvement before stopping (default: 20).
        """
        self.source = source
        self.hidden_size = hidden_size  # Unused but kept for compatibility
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.agent_lr_scale = agent_lr_scale
        self.weight_decay = weight_decay
        self.feature_weight_decay = feature_weight_decay if feature_weight_decay is not None else weight_decay
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.freeze_abilities = freeze_abilities
        self.two_stage = two_stage
        self.stage1_epochs = stage1_epochs if stage1_epochs is not None else n_epochs // 2
        self.stage2_agent_lr_scale = stage2_agent_lr_scale
        self.pca_dim = pca_dim
        self.early_stopping = early_stopping
        self.val_fraction = val_fraction
        self.patience = patience

        # Model state (set after fit())
        self._model: Optional[IRTStyleMLP] = None
        self._scaler: Optional[StandardScaler] = None
        self._pca = None  # sklearn PCA object (if pca_dim is set)
        self._agent_to_idx: Optional[Dict[str, int]] = None
        self._n_agents: int = 0
        self._feature_dim: int = 0
        self._is_fitted: bool = False

        # Training diagnostics
        self._training_losses: List[float] = []
        self._train_auc: Optional[float] = None  # AUC on training data

        # Prediction cache
        self._task_feature_cache: Dict[str, np.ndarray] = {}

    def fit(self, data: ExperimentData, train_task_ids: List[str]) -> None:
        """Fit the IRT-style model on training data.

        Args:
            data: ExperimentData containing responses and agent information.
            train_task_ids: List of task IDs to train on.
        """
        # Clear caches
        self._task_feature_cache = {}
        self._training_losses = []

        # Build agent-to-index mapping
        all_agents = data.get_all_agents()
        self._agent_to_idx = {agent: i for i, agent in enumerate(all_agents)}
        self._n_agents = len(all_agents)

        # Get task features
        task_features = self.source.get_features(train_task_ids)

        # Apply PCA if requested (fit on training data only)
        if self.pca_dim is not None:
            from sklearn.decomposition import PCA
            actual_dim = min(self.pca_dim, task_features.shape[1], task_features.shape[0])
            self._pca = PCA(n_components=actual_dim)
            task_features = self._pca.fit_transform(task_features)
            if self.verbose:
                explained_var = sum(self._pca.explained_variance_ratio_)
                print(f"   PCA: {self.source.feature_dim} -> {actual_dim} dims, explained variance: {explained_var:.3f}")

        # Fit scaler on (possibly PCA-reduced) features
        self._scaler = StandardScaler()
        task_features_scaled = self._scaler.fit_transform(task_features)
        self._feature_dim = task_features_scaled.shape[1]

        # Build task_id -> scaled features mapping for train tasks
        task_to_features = {
            task_id: task_features_scaled[i]
            for i, task_id in enumerate(train_task_ids)
        }

        # Build training data: (agent_idx, task_features, response)
        agent_indices_list: List[int] = []
        features_list: List[np.ndarray] = []
        y_list: List[float] = []

        is_binomial = isinstance(data, BinomialExperimentData)

        for task_id in train_task_ids:
            task_feat = task_to_features[task_id]

            for agent_id in all_agents:
                if agent_id not in data.responses:
                    continue
                if task_id not in data.responses[agent_id]:
                    continue

                agent_idx = self._agent_to_idx[agent_id]
                response = data.responses[agent_id][task_id]

                if is_binomial:
                    # Expand binomial to individual observations
                    k = response["successes"]
                    n = response["trials"]
                    # Add k success observations
                    for _ in range(k):
                        agent_indices_list.append(agent_idx)
                        features_list.append(task_feat)
                        y_list.append(1.0)
                    # Add (n-k) failure observations
                    for _ in range(n - k):
                        agent_indices_list.append(agent_idx)
                        features_list.append(task_feat)
                        y_list.append(0.0)
                else:
                    # Binary response
                    agent_indices_list.append(agent_idx)
                    features_list.append(task_feat)
                    y_list.append(float(response))

        if len(y_list) == 0:
            raise ValueError("No training examples found")

        agent_indices = np.array(agent_indices_list, dtype=np.int64)
        features = np.array(features_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)

        if self.verbose:
            print(f"   Training IRT-style MLP: {len(y)} samples")
            print(f"   Agents: {self._n_agents}, Feature dim: {self._feature_dim}")
            if self.two_stage:
                print(f"   two_stage=True, stage1_epochs={self.stage1_epochs}, stage2_agent_lr_scale={self.stage2_agent_lr_scale}")
            else:
                print(f"   freeze_abilities={self.freeze_abilities}, agent_lr_scale={self.agent_lr_scale}")
            print(f"   weight_decay={self.weight_decay}, feature_weight_decay={self.feature_weight_decay}")

        # Create model
        self._model = IRTStyleMLP(self._n_agents, self._feature_dim, dropout=self.dropout)

        # Check for GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._model.to(device)

        if self.verbose and device.type == "cuda":
            print(f"   Using GPU: {torch.cuda.get_device_name(0)}")

        # Initialize agent abilities from IRT if requested (freeze_abilities or two_stage)
        if self.freeze_abilities or self.two_stage:
            with torch.no_grad():
                for agent_id, idx in self._agent_to_idx.items():
                    if agent_id in data.train_abilities.index:
                        ability = data.train_abilities.loc[agent_id, "ability"]
                        self._model.agent_abilities.weight.data[idx, 0] = float(ability)
                    else:
                        # Default to 0 for agents without IRT abilities
                        self._model.agent_abilities.weight.data[idx, 0] = 0.0
            if self.freeze_abilities:
                # Permanently freeze agent abilities
                self._model.agent_abilities.requires_grad_(False)
                if self.verbose:
                    print(f"   Initialized and froze agent abilities from IRT")
            elif self.two_stage:
                # Temporarily freeze for stage 1
                self._model.agent_abilities.requires_grad_(False)
                if self.verbose:
                    print(f"   Initialized agent abilities from IRT (will unfreeze in stage 2)")

        # Convert to tensors
        agent_tensor = torch.tensor(agent_indices, dtype=torch.long, device=device)
        features_tensor = torch.tensor(features, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=device)

        criterion = nn.BCELoss()

        if self.two_stage:
            # ===== TWO-STAGE TRAINING =====
            # Stage 1: Train features only with frozen IRT abilities
            stage2_epochs = self.n_epochs - self.stage1_epochs

            if self.verbose:
                print(f"   Stage 1: Training features only for {self.stage1_epochs} epochs...")

            optimizer_stage1 = torch.optim.Adam(
                self._model.difficulty_layer.parameters(),
                lr=self.learning_rate,
                weight_decay=self.feature_weight_decay,
            )

            self._model.train()
            for epoch in range(self.stage1_epochs):
                optimizer_stage1.zero_grad()
                y_pred = self._model(agent_tensor, features_tensor)
                loss = criterion(y_pred, y_tensor)
                loss.backward()
                optimizer_stage1.step()

                loss_val = loss.item()
                self._training_losses.append(loss_val)

                if self.verbose and (epoch + 1) % 50 == 0:
                    print(f"      Stage 1 Epoch {epoch + 1}/{self.stage1_epochs}: Loss = {loss_val:.6f}")

            # Stage 2: Unfreeze agents, fine-tune both with low agent LR
            if self.verbose:
                print(f"   Stage 2: Fine-tuning both for {stage2_epochs} epochs (agent_lr_scale={self.stage2_agent_lr_scale})...")

            self._model.agent_abilities.requires_grad_(True)
            agent_lr = self.learning_rate * self.stage2_agent_lr_scale

            optimizer_stage2 = torch.optim.Adam([
                {'params': self._model.agent_abilities.parameters(), 'lr': agent_lr, 'weight_decay': self.weight_decay},
                {'params': self._model.difficulty_layer.parameters(), 'lr': self.learning_rate, 'weight_decay': self.feature_weight_decay},
            ])

            for epoch in range(stage2_epochs):
                optimizer_stage2.zero_grad()
                y_pred = self._model(agent_tensor, features_tensor)
                loss = criterion(y_pred, y_tensor)
                loss.backward()
                optimizer_stage2.step()

                loss_val = loss.item()
                self._training_losses.append(loss_val)

                if self.verbose and (epoch + 1) % 50 == 0:
                    print(f"      Stage 2 Epoch {epoch + 1}/{stage2_epochs}: Loss = {loss_val:.6f}")

        else:
            # ===== SINGLE-STAGE TRAINING =====

            # Split into train/val if early stopping enabled
            if self.early_stopping:
                n_samples = len(y)
                n_val = max(1, int(n_samples * self.val_fraction))
                n_train = n_samples - n_val

                # Random permutation for split
                perm = torch.randperm(n_samples)
                train_idx = perm[:n_train]
                val_idx = perm[n_train:]

                train_agent = agent_tensor[train_idx]
                train_feat = features_tensor[train_idx]
                train_y = y_tensor[train_idx]
                val_agent = agent_tensor[val_idx]
                val_feat = features_tensor[val_idx]
                val_y = y_tensor[val_idx]

                if self.verbose:
                    print(f"   Early stopping: {n_train} train, {n_val} val samples, patience={self.patience}")
            else:
                train_agent = agent_tensor
                train_feat = features_tensor
                train_y = y_tensor

            # Optimizer with per-parameter group regularization and learning rates
            if self.freeze_abilities:
                # Only optimize difficulty layer when abilities are frozen
                optimizer = torch.optim.Adam(
                    self._model.difficulty_layer.parameters(),
                    lr=self.learning_rate,
                    weight_decay=self.feature_weight_decay,
                )
            else:
                # Separate weight_decay and learning rates for agent abilities vs feature weights
                agent_lr = self.learning_rate * self.agent_lr_scale
                optimizer = torch.optim.Adam([
                    {'params': self._model.agent_abilities.parameters(), 'lr': agent_lr, 'weight_decay': self.weight_decay},
                    {'params': self._model.difficulty_layer.parameters(), 'lr': self.learning_rate, 'weight_decay': self.feature_weight_decay},
                ])

            # Early stopping state
            best_val_loss = float('inf')
            best_state_dict = None
            epochs_without_improvement = 0
            stopped_early = False

            # Training loop (full-batch)
            self._model.train()
            for epoch in range(self.n_epochs):
                optimizer.zero_grad()

                # Forward pass on training data
                y_pred = self._model(train_agent, train_feat)
                loss = criterion(y_pred, train_y)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Track loss per iteration
                loss_val = loss.item()
                self._training_losses.append(loss_val)

                # Early stopping check
                if self.early_stopping:
                    self._model.eval()
                    with torch.no_grad():
                        val_pred = self._model(val_agent, val_feat)
                        val_loss = criterion(val_pred, val_y).item()
                    self._model.train()

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_state_dict = {k: v.clone() for k, v in self._model.state_dict().items()}
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1

                    if epochs_without_improvement >= self.patience:
                        stopped_early = True
                        if self.verbose:
                            print(f"      Early stopping at epoch {epoch + 1} (val_loss={val_loss:.6f}, best={best_val_loss:.6f})")
                        break

                    if self.verbose and (epoch + 1) % 50 == 0:
                        print(f"      Epoch {epoch + 1}/{self.n_epochs}: train_loss={loss_val:.6f}, val_loss={val_loss:.6f}")
                else:
                    if self.verbose and (epoch + 1) % 50 == 0:
                        print(f"      Epoch {epoch + 1}/{self.n_epochs}: Loss = {loss_val:.6f}")

            # Restore best model if early stopping was used
            if self.early_stopping and best_state_dict is not None:
                self._model.load_state_dict(best_state_dict)
                if self.verbose and not stopped_early:
                    print(f"   Completed {self.n_epochs} epochs, best val_loss={best_val_loss:.6f}")

        self._is_fitted = True

        # Compute train AUC for diagnostics
        self._model.eval()
        with torch.no_grad():
            y_pred_train = self._model(agent_tensor, features_tensor).cpu().numpy()
        if len(np.unique(y)) > 1:  # Need both classes for AUC
            self._train_auc = roc_auc_score(y, y_pred_train)
        else:
            self._train_auc = None

        if self.verbose:
            train_auc_str = f"{self._train_auc:.4f}" if self._train_auc else "N/A"
            print(f"   Final loss: {self._training_losses[-1]:.6f}, Train AUC: {train_auc_str}")

    def predict_probability(
        self, data: ExperimentData, agent_id: str, task_id: str
    ) -> float:
        """Predict success probability for a specific (agent, task) pair.

        Uses IRT formula: P = sigmoid(θ_agent - β_task)

        Args:
            data: ExperimentData (used for test_tasks list).
            agent_id: Agent identifier.
            task_id: Task identifier.

        Returns:
            Predicted probability of success (0 to 1).
        """
        if not self._is_fitted:
            raise RuntimeError("Predictor must be fit before calling predict_probability()")

        # Lazily cache test task features
        if task_id not in self._task_feature_cache:
            self._cache_test_task_features(data.test_tasks)

        if task_id not in self._task_feature_cache:
            raise ValueError(f"No features for task {task_id}")

        if agent_id not in self._agent_to_idx:
            raise ValueError(f"Unknown agent {agent_id}")

        agent_idx = self._agent_to_idx[agent_id]
        task_feat = self._task_feature_cache[task_id]

        # Get device
        device = next(self._model.parameters()).device

        # Forward pass
        self._model.eval()
        with torch.no_grad():
            agent_tensor = torch.tensor([agent_idx], dtype=torch.long, device=device)
            feat_tensor = torch.tensor(task_feat, dtype=torch.float32, device=device).unsqueeze(0)
            prob = self._model(agent_tensor, feat_tensor).item()

        return prob

    def _cache_test_task_features(self, test_tasks: List[str]) -> None:
        """Cache scaled features for test tasks."""
        features = self.source.get_features(test_tasks)

        # Apply PCA if it was used during training
        if self._pca is not None:
            features = self._pca.transform(features)

        features_scaled = self._scaler.transform(features)

        for i, task_id in enumerate(test_tasks):
            self._task_feature_cache[task_id] = features_scaled[i]

    def get_training_losses(self) -> List[float]:
        """Return list of loss values per iteration for convergence plots."""
        return self._training_losses.copy()

    def get_train_auc(self) -> Optional[float]:
        """Return AUC computed on training data after fit().

        This is useful for diagnosing overfitting: if train AUC >> test AUC,
        the model is overfitting to the training data.
        """
        return self._train_auc

    @property
    def name(self) -> str:
        """Human-readable predictor name."""
        if self.pca_dim is not None:
            return f"MLP (PCA-{self.pca_dim} {self.source.name})"
        return f"MLP ({self.source.name})"


# ============================================================================
# NEW INTERACTION ARCHITECTURES (Part 5 experiments)
# ============================================================================


class TwoTowerModel(nn.Module):
    """Two-tower architecture with dot product scoring.

    Architecture:
        Agent tower: agent_one_hot → Linear → ReLU → agent_repr (emb_dim)
        Task tower: task_features → Linear → ReLU → task_repr (emb_dim)
        Score: dot(agent_repr, task_repr) + agent_bias + task_bias
        Output: sigmoid(score)

    This is similar to recommendation systems - agents and tasks are embedded
    into the same space, and compatibility is measured by dot product.
    """

    def __init__(
        self,
        n_agents: int,
        feature_dim: int,
        emb_dim: int = 32,
        agent_hidden: int = 64,
        task_hidden: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.feature_dim = feature_dim
        self.emb_dim = emb_dim

        # Agent tower: one_hot → hidden → embedding
        self.agent_tower = nn.Sequential(
            nn.Linear(n_agents, agent_hidden),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(agent_hidden, emb_dim),
        )

        # Task tower: features → hidden → embedding
        self.task_tower = nn.Sequential(
            nn.Linear(feature_dim, task_hidden),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(task_hidden, emb_dim),
        )

        # Bias terms (optional, can help calibration)
        self.agent_bias = nn.Linear(n_agents, 1, bias=False)
        self.task_bias = nn.Linear(feature_dim, 1, bias=False)
        self.global_bias = nn.Parameter(torch.zeros(1))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, n_agents + feature_dim)
               Layout: [agent_one_hot | task_features]
        """
        agent_one_hot = x[:, :self.n_agents]
        task_features = x[:, self.n_agents:]

        # Get embeddings
        agent_emb = self.agent_tower(agent_one_hot)  # (batch, emb_dim)
        task_emb = self.task_tower(task_features)    # (batch, emb_dim)

        # Dot product score
        dot_score = (agent_emb * task_emb).sum(dim=1)  # (batch,)

        # Add biases
        agent_b = self.agent_bias(agent_one_hot).squeeze(-1)
        task_b = self.task_bias(task_features).squeeze(-1)
        score = dot_score + agent_b + task_b + self.global_bias

        return self.sigmoid(score)


class BilinearModel(nn.Module):
    """Bilinear interaction model.

    Architecture:
        Agent embedding: agent_one_hot → Linear → agent_emb (agent_dim)
        Task embedding: task_features → Linear → task_emb (task_dim)
        Score: agent_emb^T @ W @ task_emb + biases
        Output: sigmoid(score)

    The bilinear form W allows rich agent-task interactions.
    """

    def __init__(
        self,
        n_agents: int,
        feature_dim: int,
        agent_dim: int = 32,
        task_dim: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.feature_dim = feature_dim

        # Agent encoder
        self.agent_encoder = nn.Sequential(
            nn.Linear(n_agents, agent_dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        # Task encoder
        self.task_encoder = nn.Sequential(
            nn.Linear(feature_dim, task_dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        # Bilinear interaction matrix
        self.bilinear = nn.Bilinear(agent_dim, task_dim, 1, bias=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        agent_one_hot = x[:, :self.n_agents]
        task_features = x[:, self.n_agents:]

        agent_emb = self.agent_encoder(agent_one_hot)  # (batch, agent_dim)
        task_emb = self.task_encoder(task_features)    # (batch, task_dim)

        # Bilinear: agent_emb^T @ W @ task_emb + bias
        score = self.bilinear(agent_emb, task_emb).squeeze(-1)

        return self.sigmoid(score)


class NCFModel(nn.Module):
    """Neural Collaborative Filtering model.

    Combines two pathways:
    1. GMF (Generalized Matrix Factorization): agent_emb ⊙ task_emb → score
    2. MLP: concat(agent_emb, task_emb) → hidden → score

    Final: weighted combination of both scores.

    Reference: He et al., "Neural Collaborative Filtering" (WWW 2017)
    """

    def __init__(
        self,
        n_agents: int,
        feature_dim: int,
        gmf_dim: int = 32,
        mlp_agent_dim: int = 32,
        mlp_task_dim: int = 32,
        mlp_hidden: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.feature_dim = feature_dim

        # GMF pathway embeddings
        self.gmf_agent = nn.Linear(n_agents, gmf_dim)
        self.gmf_task = nn.Linear(feature_dim, gmf_dim)

        # MLP pathway embeddings
        self.mlp_agent = nn.Linear(n_agents, mlp_agent_dim)
        self.mlp_task = nn.Linear(feature_dim, mlp_task_dim)

        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(mlp_agent_dim + mlp_task_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.ReLU(),
        )

        # Final prediction layer (combines GMF and MLP)
        # GMF produces gmf_dim features (element-wise product)
        # MLP produces mlp_hidden // 2 features
        self.output = nn.Linear(gmf_dim + mlp_hidden // 2, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        agent_one_hot = x[:, :self.n_agents]
        task_features = x[:, self.n_agents:]

        # GMF pathway: element-wise product
        gmf_agent_emb = self.gmf_agent(agent_one_hot)
        gmf_task_emb = self.gmf_task(task_features)
        gmf_out = gmf_agent_emb * gmf_task_emb  # (batch, gmf_dim)

        # MLP pathway: concatenate then process
        mlp_agent_emb = self.mlp_agent(agent_one_hot)
        mlp_task_emb = self.mlp_task(task_features)
        mlp_input = torch.cat([mlp_agent_emb, mlp_task_emb], dim=1)
        mlp_out = self.mlp(mlp_input)  # (batch, mlp_hidden // 2)

        # Combine and predict
        combined = torch.cat([gmf_out, mlp_out], dim=1)
        score = self.output(combined).squeeze(-1)

        return self.sigmoid(score)


class MultiplicativeModel(nn.Module):
    """Multiplicative interaction model.

    Instead of concatenating agent and task representations, uses
    element-wise multiplication to create interaction features.

    Architecture:
        agent_repr = Linear(agent_one_hot) → (hidden_dim,)
        task_repr = Linear(task_features) → (hidden_dim,)
        interaction = agent_repr ⊙ task_repr (element-wise product)
        score = Linear(interaction) → sigmoid
    """

    def __init__(
        self,
        n_agents: int,
        feature_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.feature_dim = feature_dim

        # Agent encoder
        self.agent_encoder = nn.Sequential(
            nn.Linear(n_agents, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        # Task encoder
        self.task_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        # Output from multiplicative interaction
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        agent_one_hot = x[:, :self.n_agents]
        task_features = x[:, self.n_agents:]

        agent_repr = self.agent_encoder(agent_one_hot)
        task_repr = self.task_encoder(task_features)

        # Multiplicative interaction
        interaction = agent_repr * task_repr

        return self.output(interaction).squeeze(-1)


class AgentEmbeddingModel(nn.Module):
    """Model using learned low-dimensional agent embeddings instead of one-hot.

    Architecture:
        agent_emb = Embedding(agent_idx) → (emb_dim,)
        combined = concat(agent_emb, task_features)
        score = MLP(combined) → sigmoid

    This reduces parameters from O(n_agents * hidden) to O(n_agents * emb_dim + emb_dim * hidden),
    which can help with generalization when n_agents is large.
    """

    def __init__(
        self,
        n_agents: int,
        feature_dim: int,
        agent_emb_dim: int = 32,
        hidden_sizes: List[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 32]

        self.n_agents = n_agents
        self.feature_dim = feature_dim
        self.agent_emb_dim = agent_emb_dim

        # Learned agent embeddings
        self.agent_embedding = nn.Embedding(n_agents, agent_emb_dim)

        # MLP on concatenated [agent_emb | task_features]
        input_dim = agent_emb_dim + feature_dim
        layers = []
        prev_dim = input_dim

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_size

        layers.append(nn.Linear(prev_dim, 1))
        # No sigmoid - use BCEWithLogitsLoss for numerical stability

        self.mlp = nn.Sequential(*layers)

    def forward(self, agent_indices: torch.Tensor, task_features: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns logits (not probabilities).

        Args:
            agent_indices: Long tensor of shape (batch,) with agent indices
            task_features: Float tensor of shape (batch, feature_dim)
        """
        agent_emb = self.agent_embedding(agent_indices)  # (batch, agent_emb_dim)
        combined = torch.cat([agent_emb, task_features], dim=1)
        return self.mlp(combined).squeeze(-1)  # Returns logits


class FullMLPPredictor:
    """Full MLP predictor that concatenates agent and task features.

    Unlike MLPPredictor (IRTStyleMLP) which separates agent and task processing
    with the IRT formula (θ - β), this predictor concatenates agent one-hot
    encoding with task features and passes through a hidden layer.

    Architecture:
        Input: [agent_one_hot (n_agents) | task_features (feature_dim)]
        -> Linear(input_dim, hidden_size)
        -> ReLU
        -> Dropout (optional)
        -> Linear(hidden_size, 1)
        -> Sigmoid
        Output: P(success)

    This allows learning arbitrary interactions between agent identity and
    task features, but has more parameters than IRTStyleMLP.
    """

    def __init__(
        self,
        source: TaskFeatureSource,
        hidden_size: int = 64,
        dropout: float = 0.0,
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        n_epochs: int = 200,
        verbose: bool = False,
        init_from_irt: bool = False,
        early_stopping: bool = False,
        val_fraction: float = 0.1,
        patience: int = 20,
    ):
        """Initialize full MLP predictor.

        Args:
            source: TaskFeatureSource providing features for tasks.
            hidden_size: Number of hidden units in the MLP.
            dropout: Dropout probability after hidden layer (0.0 = no dropout).
            learning_rate: Learning rate for Adam optimizer.
            weight_decay: L2 regularization (Adam weight_decay).
            n_epochs: Number of training epochs.
            verbose: Print training progress.
            init_from_irt: If True, initialize the agent-portion of fc1 weights
                using pre-trained IRT abilities. This provides a good starting point
                for agent representations.
            early_stopping: If True, use validation-based early stopping.
            val_fraction: Fraction of training data for validation (default: 0.1).
            patience: Epochs without improvement before stopping (default: 20).
        """
        self.source = source
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.init_from_irt = init_from_irt
        self.early_stopping = early_stopping
        self.val_fraction = val_fraction
        self.patience = patience

        # Model state (set after fit())
        self._model: Optional[SimpleMLP] = None
        self._scaler: Optional[StandardScaler] = None
        self._agent_to_idx: Optional[Dict[str, int]] = None
        self._n_agents: int = 0
        self._feature_dim: int = 0
        self._is_fitted: bool = False

        # Training diagnostics
        self._training_losses: List[float] = []
        self._train_auc: Optional[float] = None

        # Prediction cache
        self._task_feature_cache: Dict[str, np.ndarray] = {}

    def fit(self, data: ExperimentData, train_task_ids: List[str]) -> None:
        """Fit the full MLP on training data.

        Args:
            data: ExperimentData containing responses and agent information.
            train_task_ids: List of task IDs to train on.
        """
        # Clear caches
        self._task_feature_cache = {}
        self._training_losses = []

        # Build agent-to-index mapping
        all_agents = data.get_all_agents()
        self._agent_to_idx = {agent: i for i, agent in enumerate(all_agents)}
        self._n_agents = len(all_agents)

        # Get task features and fit scaler
        task_features = self.source.get_features(train_task_ids)
        self._scaler = StandardScaler()
        task_features_scaled = self._scaler.fit_transform(task_features)
        self._feature_dim = task_features_scaled.shape[1]

        # Build task_id -> scaled features mapping
        task_to_features = {
            task_id: task_features_scaled[i]
            for i, task_id in enumerate(train_task_ids)
        }

        # Build training data: concatenated [agent_one_hot | task_features]
        X_list: List[np.ndarray] = []
        y_list: List[float] = []

        is_binomial = isinstance(data, BinomialExperimentData)

        for task_id in train_task_ids:
            task_feat = task_to_features[task_id]

            for agent_id in all_agents:
                if agent_id not in data.responses:
                    continue
                if task_id not in data.responses[agent_id]:
                    continue

                agent_idx = self._agent_to_idx[agent_id]
                x = build_input_vector(agent_idx, self._n_agents, task_feat)
                response = data.responses[agent_id][task_id]

                if is_binomial:
                    k = response["successes"]
                    n = response["trials"]
                    for _ in range(k):
                        X_list.append(x)
                        y_list.append(1.0)
                    for _ in range(n - k):
                        X_list.append(x)
                        y_list.append(0.0)
                else:
                    X_list.append(x)
                    y_list.append(float(response))

        if len(X_list) == 0:
            raise ValueError("No training examples found")

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)

        if self.verbose:
            print(f"   Training Full MLP: {len(X)} samples")
            print(f"   Input dim: {X.shape[1]} (agents={self._n_agents}, features={self._feature_dim})")
            print(f"   hidden_size={self.hidden_size}, weight_decay={self.weight_decay}")

        # Create model
        input_dim = self._n_agents + self._feature_dim
        self._model = SimpleMLP(input_dim, self.hidden_size, dropout=self.dropout)

        # Check for GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._model.to(device)

        if self.verbose and device.type == "cuda":
            print(f"   Using GPU: {torch.cuda.get_device_name(0)}")

        # Initialize agent weights from IRT if requested
        if self.init_from_irt:
            with torch.no_grad():
                # fc1.weight shape: (hidden_size, input_dim)
                # First n_agents columns correspond to agent one-hot inputs
                for agent_id, idx in self._agent_to_idx.items():
                    if agent_id in data.train_abilities.index:
                        ability = float(data.train_abilities.loc[agent_id, "ability"])
                        # Initialize all hidden units to respond to this agent's ability
                        self._model.fc1.weight.data[:, idx] = ability
                    else:
                        self._model.fc1.weight.data[:, idx] = 0.0
            if self.verbose:
                print(f"   Initialized agent weights from IRT abilities")

        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=device)

        # Split for early stopping if enabled
        if self.early_stopping:
            n_samples = len(y)
            n_val = max(1, int(n_samples * self.val_fraction))
            n_train = n_samples - n_val

            perm = torch.randperm(n_samples)
            train_idx = perm[:n_train]
            val_idx = perm[n_train:]

            train_X = X_tensor[train_idx]
            train_y = y_tensor[train_idx]
            val_X = X_tensor[val_idx]
            val_y = y_tensor[val_idx]

            if self.verbose:
                print(f"   Early stopping: {n_train} train, {n_val} val samples, patience={self.patience}")
        else:
            train_X = X_tensor
            train_y = y_tensor

        # Optimizer and loss
        optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.BCELoss()

        # Early stopping state
        best_val_loss = float('inf')
        best_state_dict = None
        epochs_without_improvement = 0
        stopped_early = False

        # Training loop
        self._model.train()
        for epoch in range(self.n_epochs):
            optimizer.zero_grad()
            y_pred = self._model(train_X)
            loss = criterion(y_pred, train_y)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            self._training_losses.append(loss_val)

            # Early stopping check
            if self.early_stopping:
                self._model.eval()
                with torch.no_grad():
                    val_pred = self._model(val_X)
                    val_loss = criterion(val_pred, val_y).item()
                self._model.train()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state_dict = {k: v.clone() for k, v in self._model.state_dict().items()}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= self.patience:
                    stopped_early = True
                    if self.verbose:
                        print(f"      Early stopping at epoch {epoch + 1} (val_loss={val_loss:.6f})")
                    break

                if self.verbose and (epoch + 1) % 50 == 0:
                    print(f"      Epoch {epoch + 1}/{self.n_epochs}: train_loss={loss_val:.6f}, val_loss={val_loss:.6f}")
            else:
                if self.verbose and (epoch + 1) % 50 == 0:
                    print(f"      Epoch {epoch + 1}/{self.n_epochs}: Loss = {loss_val:.6f}")

        # Restore best model if early stopping was used
        if self.early_stopping and best_state_dict is not None:
            self._model.load_state_dict(best_state_dict)

        self._is_fitted = True

        # Compute train AUC for diagnostics
        self._model.eval()
        with torch.no_grad():
            y_pred_train = self._model(X_tensor).cpu().numpy()
        if len(np.unique(y)) > 1:
            self._train_auc = roc_auc_score(y, y_pred_train)
        else:
            self._train_auc = None

        if self.verbose:
            train_auc_str = f"{self._train_auc:.4f}" if self._train_auc else "N/A"
            print(f"   Final loss: {self._training_losses[-1]:.6f}, Train AUC: {train_auc_str}")

    def predict_probability(
        self, data: ExperimentData, agent_id: str, task_id: str
    ) -> float:
        """Predict success probability for a specific (agent, task) pair."""
        if not self._is_fitted:
            raise RuntimeError("Predictor must be fit before calling predict_probability()")

        # Lazily cache test task features
        if task_id not in self._task_feature_cache:
            self._cache_test_task_features(data.test_tasks)

        if task_id not in self._task_feature_cache:
            raise ValueError(f"No features for task {task_id}")

        if agent_id not in self._agent_to_idx:
            raise ValueError(f"Unknown agent {agent_id}")

        agent_idx = self._agent_to_idx[agent_id]
        task_feat = self._task_feature_cache[task_id]

        # Build input vector
        x = build_input_vector(agent_idx, self._n_agents, task_feat)

        # Get device
        device = next(self._model.parameters()).device

        # Forward pass
        self._model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
            prob = self._model(x_tensor).item()

        return prob

    def _cache_test_task_features(self, test_tasks: List[str]) -> None:
        """Cache scaled features for test tasks."""
        features = self.source.get_features(test_tasks)
        features_scaled = self._scaler.transform(features)

        for i, task_id in enumerate(test_tasks):
            self._task_feature_cache[task_id] = features_scaled[i]

    def get_training_losses(self) -> List[float]:
        """Return list of loss values per iteration."""
        return self._training_losses.copy()

    def get_train_auc(self) -> Optional[float]:
        """Return AUC computed on training data after fit()."""
        return self._train_auc

    @property
    def name(self) -> str:
        """Human-readable predictor name."""
        return f"FullMLP ({self.source.name})"


# ============================================================================
# NEW INTERACTION PREDICTORS (Part 5 experiments)
# ============================================================================


class InteractionPredictor:
    """Generic predictor for interaction-based models (TwoTower, Bilinear, NCF, Multiplicative).

    This is a unified predictor class that can use any of the new interaction architectures.
    """

    def __init__(
        self,
        source: TaskFeatureSource,
        model_type: str = "two_tower",  # "two_tower", "bilinear", "ncf", "multiplicative"
        emb_dim: int = 32,
        hidden_dim: int = 64,
        dropout: float = 0.0,
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        n_epochs: int = 500,
        verbose: bool = False,
        init_from_irt: bool = False,
        early_stopping: bool = True,
        val_fraction: float = 0.1,
        patience: int = 30,
    ):
        """Initialize interaction predictor.

        Args:
            source: TaskFeatureSource providing features for tasks.
            model_type: Type of interaction model to use.
            emb_dim: Embedding dimension for agent/task representations.
            hidden_dim: Hidden layer size for towers/encoders.
            dropout: Dropout probability.
            learning_rate: Learning rate for Adam optimizer.
            weight_decay: L2 regularization.
            n_epochs: Number of training epochs.
            verbose: Print training progress.
            init_from_irt: Initialize agent embeddings from IRT abilities.
            early_stopping: Use validation-based early stopping.
            val_fraction: Fraction for validation.
            patience: Early stopping patience.
        """
        self.source = source
        self.model_type = model_type
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.init_from_irt = init_from_irt
        self.early_stopping = early_stopping
        self.val_fraction = val_fraction
        self.patience = patience

        self._model: Optional[nn.Module] = None
        self._scaler: Optional[StandardScaler] = None
        self._agent_to_idx: Optional[Dict[str, int]] = None
        self._n_agents: int = 0
        self._feature_dim: int = 0
        self._is_fitted: bool = False
        self._train_auc: Optional[float] = None
        self._task_feature_cache: Dict[str, np.ndarray] = {}

    def _create_model(self) -> nn.Module:
        """Create the appropriate model based on model_type."""
        if self.model_type == "two_tower":
            return TwoTowerModel(
                n_agents=self._n_agents,
                feature_dim=self._feature_dim,
                emb_dim=self.emb_dim,
                agent_hidden=self.hidden_dim,
                task_hidden=self.hidden_dim,
                dropout=self.dropout,
            )
        elif self.model_type == "bilinear":
            return BilinearModel(
                n_agents=self._n_agents,
                feature_dim=self._feature_dim,
                agent_dim=self.emb_dim,
                task_dim=self.emb_dim,
                dropout=self.dropout,
            )
        elif self.model_type == "ncf":
            return NCFModel(
                n_agents=self._n_agents,
                feature_dim=self._feature_dim,
                gmf_dim=self.emb_dim,
                mlp_agent_dim=self.emb_dim,
                mlp_task_dim=self.emb_dim,
                mlp_hidden=self.hidden_dim,
                dropout=self.dropout,
            )
        elif self.model_type == "multiplicative":
            return MultiplicativeModel(
                n_agents=self._n_agents,
                feature_dim=self._feature_dim,
                hidden_dim=self.hidden_dim,
                dropout=self.dropout,
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def fit(self, data: ExperimentData, train_task_ids: List[str]) -> None:
        """Fit the interaction model on training data."""
        self._task_feature_cache = {}

        # Build agent mapping
        all_agents = data.get_all_agents()
        self._agent_to_idx = {agent: i for i, agent in enumerate(all_agents)}
        self._n_agents = len(all_agents)

        # Get task features
        task_features = self.source.get_features(train_task_ids)
        self._scaler = StandardScaler()
        task_features_scaled = self._scaler.fit_transform(task_features)
        self._feature_dim = task_features_scaled.shape[1]

        # Build task -> features mapping
        task_to_features = {
            task_id: task_features_scaled[i]
            for i, task_id in enumerate(train_task_ids)
        }

        # Build training data
        X_list: List[np.ndarray] = []
        y_list: List[float] = []
        is_binomial = isinstance(data, BinomialExperimentData)

        for task_id in train_task_ids:
            task_feat = task_to_features[task_id]
            for agent_id in all_agents:
                if agent_id not in data.responses:
                    continue
                if task_id not in data.responses[agent_id]:
                    continue

                agent_idx = self._agent_to_idx[agent_id]
                x = build_input_vector(agent_idx, self._n_agents, task_feat)
                response = data.responses[agent_id][task_id]

                if is_binomial:
                    k = response["successes"]
                    n = response["trials"]
                    for _ in range(k):
                        X_list.append(x)
                        y_list.append(1.0)
                    for _ in range(n - k):
                        X_list.append(x)
                        y_list.append(0.0)
                else:
                    X_list.append(x)
                    y_list.append(float(response))

        if len(X_list) == 0:
            raise ValueError("No training examples found")

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)

        if self.verbose:
            print(f"   Training {self.model_type}: {len(X)} samples")
            print(f"   Agents: {self._n_agents}, Features: {self._feature_dim}")
            print(f"   emb_dim={self.emb_dim}, hidden_dim={self.hidden_dim}")

        # Create model
        self._model = self._create_model()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._model.to(device)

        if self.verbose and device.type == "cuda":
            print(f"   Using GPU: {torch.cuda.get_device_name(0)}")

        # Initialize agent embeddings from IRT if requested
        if self.init_from_irt:
            self._init_agent_weights_from_irt(data)

        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=device)

        # Early stopping split
        if self.early_stopping:
            n_samples = len(y)
            n_val = max(1, int(n_samples * self.val_fraction))
            n_train = n_samples - n_val

            perm = torch.randperm(n_samples)
            train_idx = perm[:n_train]
            val_idx = perm[n_train:]

            train_X = X_tensor[train_idx]
            train_y = y_tensor[train_idx]
            val_X = X_tensor[val_idx]
            val_y = y_tensor[val_idx]

            if self.verbose:
                print(f"   Early stopping: {n_train} train, {n_val} val")
        else:
            train_X = X_tensor
            train_y = y_tensor

        optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.BCELoss()

        best_val_loss = float('inf')
        best_state_dict = None
        epochs_without_improvement = 0

        self._model.train()
        for epoch in range(self.n_epochs):
            optimizer.zero_grad()
            y_pred = self._model(train_X)
            loss = criterion(y_pred, train_y)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()

            if self.early_stopping:
                self._model.eval()
                with torch.no_grad():
                    val_pred = self._model(val_X)
                    val_loss = criterion(val_pred, val_y).item()
                self._model.train()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state_dict = {k: v.clone() for k, v in self._model.state_dict().items()}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= self.patience:
                    if self.verbose:
                        print(f"      Early stopping at epoch {epoch + 1}")
                    break

                if self.verbose and (epoch + 1) % 50 == 0:
                    print(f"      Epoch {epoch + 1}: train_loss={loss_val:.4f}, val_loss={val_loss:.4f}")
            else:
                if self.verbose and (epoch + 1) % 50 == 0:
                    print(f"      Epoch {epoch + 1}: loss={loss_val:.4f}")

        if self.early_stopping and best_state_dict is not None:
            self._model.load_state_dict(best_state_dict)

        self._is_fitted = True

        # Compute train AUC
        self._model.eval()
        with torch.no_grad():
            y_pred_train = self._model(X_tensor).cpu().numpy()
        self._train_auc = roc_auc_score(y, y_pred_train) if len(np.unique(y)) > 1 else None

        if self.verbose:
            train_auc_str = f"{self._train_auc:.4f}" if self._train_auc else "N/A"
            print(f"   Final: train_auc={train_auc_str}")

    def _init_agent_weights_from_irt(self, data: ExperimentData) -> None:
        """Initialize agent-related weights from IRT abilities."""
        with torch.no_grad():
            if self.model_type == "two_tower":
                # Initialize agent tower's first layer
                layer = self._model.agent_tower[0]
                for agent_id, idx in self._agent_to_idx.items():
                    if agent_id in data.train_abilities.index:
                        ability = float(data.train_abilities.loc[agent_id, "ability"])
                        layer.weight.data[:, idx] = ability
            elif self.model_type == "bilinear":
                layer = self._model.agent_encoder[0]
                for agent_id, idx in self._agent_to_idx.items():
                    if agent_id in data.train_abilities.index:
                        ability = float(data.train_abilities.loc[agent_id, "ability"])
                        layer.weight.data[:, idx] = ability
            elif self.model_type == "ncf":
                # Initialize both GMF and MLP agent layers
                for layer in [self._model.gmf_agent, self._model.mlp_agent]:
                    for agent_id, idx in self._agent_to_idx.items():
                        if agent_id in data.train_abilities.index:
                            ability = float(data.train_abilities.loc[agent_id, "ability"])
                            layer.weight.data[:, idx] = ability
            elif self.model_type == "multiplicative":
                layer = self._model.agent_encoder[0]
                for agent_id, idx in self._agent_to_idx.items():
                    if agent_id in data.train_abilities.index:
                        ability = float(data.train_abilities.loc[agent_id, "ability"])
                        layer.weight.data[:, idx] = ability

        if self.verbose:
            print(f"   Initialized agent weights from IRT abilities")

    def predict_probability(self, data: ExperimentData, agent_id: str, task_id: str) -> float:
        """Predict success probability."""
        if not self._is_fitted:
            raise RuntimeError("Predictor must be fit first")

        if task_id not in self._task_feature_cache:
            self._cache_test_task_features(data.test_tasks)

        if task_id not in self._task_feature_cache:
            raise ValueError(f"No features for task {task_id}")
        if agent_id not in self._agent_to_idx:
            raise ValueError(f"Unknown agent {agent_id}")

        agent_idx = self._agent_to_idx[agent_id]
        task_feat = self._task_feature_cache[task_id]
        x = build_input_vector(agent_idx, self._n_agents, task_feat)

        device = next(self._model.parameters()).device
        self._model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
            prob = self._model(x_tensor).item()
        return prob

    def _cache_test_task_features(self, test_tasks: List[str]) -> None:
        """Cache scaled features for test tasks."""
        features = self.source.get_features(test_tasks)
        features_scaled = self._scaler.transform(features)
        for i, task_id in enumerate(test_tasks):
            self._task_feature_cache[task_id] = features_scaled[i]

    def get_train_auc(self) -> Optional[float]:
        return self._train_auc

    @property
    def name(self) -> str:
        return f"{self.model_type.replace('_', ' ').title()}"


class TaskBottleneckModel(nn.Module):
    """Model that compresses task features through a bottleneck before combining with agent.

    Architecture:
        task_bottleneck = Linear(task_features) → ReLU → (bottleneck_dim,)
        agent_emb = Embedding(agent_idx) → (agent_emb_dim,)
        combined = concat(agent_emb, task_bottleneck)
        score = MLP(combined) → sigmoid

    This creates balanced representations: both agent and task are low-dimensional
    before being combined, similar to how AgentEmb helped with agents.
    """

    def __init__(
        self,
        n_agents: int,
        feature_dim: int,
        agent_emb_dim: int = 32,
        task_bottleneck_dim: int = 64,
        hidden_sizes: List[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 32]

        self.n_agents = n_agents
        self.feature_dim = feature_dim
        self.agent_emb_dim = agent_emb_dim
        self.task_bottleneck_dim = task_bottleneck_dim

        # Learned agent embeddings
        self.agent_embedding = nn.Embedding(n_agents, agent_emb_dim)

        # Task bottleneck: compress high-dim features to low-dim
        self.task_bottleneck = nn.Sequential(
            nn.Linear(feature_dim, task_bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        # MLP on concatenated [agent_emb | task_bottleneck]
        input_dim = agent_emb_dim + task_bottleneck_dim
        layers = []
        prev_dim = input_dim

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_size

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.mlp = nn.Sequential(*layers)

    def forward(self, agent_indices: torch.Tensor, task_features: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        agent_emb = self.agent_embedding(agent_indices)  # (batch, agent_emb_dim)
        task_emb = self.task_bottleneck(task_features)   # (batch, task_bottleneck_dim)
        combined = torch.cat([agent_emb, task_emb], dim=1)
        return self.mlp(combined).squeeze(-1)


class CrossAttentionModel(nn.Module):
    """Model where agent embedding attends to task feature groups.

    Architecture:
        agent_emb = Embedding(agent_idx) → (agent_emb_dim,)
        task_chunks = reshape(task_features) → (n_chunks, chunk_dim)
        attention_weights = softmax(agent_emb @ task_chunks.T)
        attended_task = attention_weights @ task_chunks
        combined = concat(agent_emb, attended_task)
        score = MLP(combined) → sigmoid

    This allows different agents to "focus" on different aspects of the task.
    """

    def __init__(
        self,
        n_agents: int,
        feature_dim: int,
        agent_emb_dim: int = 32,
        n_chunks: int = 64,  # Split features into this many chunks
        hidden_sizes: List[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 32]

        self.n_agents = n_agents
        self.feature_dim = feature_dim
        self.agent_emb_dim = agent_emb_dim
        self.n_chunks = n_chunks
        self.chunk_dim = feature_dim // n_chunks

        # Pad if needed
        if feature_dim % n_chunks != 0:
            self.padded_dim = (n_chunks - (feature_dim % n_chunks)) + feature_dim
            self.chunk_dim = self.padded_dim // n_chunks
        else:
            self.padded_dim = feature_dim

        # Agent embedding
        self.agent_embedding = nn.Embedding(n_agents, agent_emb_dim)

        # Project agent embedding to attention query space
        self.query_proj = nn.Linear(agent_emb_dim, self.chunk_dim)

        # MLP on concatenated [agent_emb | attended_task]
        input_dim = agent_emb_dim + self.chunk_dim
        layers = []
        prev_dim = input_dim

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_size

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, agent_indices: torch.Tensor, task_features: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch_size = agent_indices.shape[0]

        # Agent embedding and query
        agent_emb = self.agent_embedding(agent_indices)  # (batch, agent_emb_dim)
        query = self.query_proj(agent_emb)  # (batch, chunk_dim)

        # Pad task features if needed
        if self.padded_dim != self.feature_dim:
            padding = torch.zeros(batch_size, self.padded_dim - self.feature_dim,
                                  device=task_features.device)
            task_features = torch.cat([task_features, padding], dim=1)

        # Reshape to chunks: (batch, n_chunks, chunk_dim)
        task_chunks = task_features.view(batch_size, self.n_chunks, self.chunk_dim)

        # Attention: query @ keys.T → (batch, n_chunks)
        attention_scores = torch.bmm(
            query.unsqueeze(1),  # (batch, 1, chunk_dim)
            task_chunks.transpose(1, 2)  # (batch, chunk_dim, n_chunks)
        ).squeeze(1)  # (batch, n_chunks)

        attention_weights = torch.softmax(attention_scores / (self.chunk_dim ** 0.5), dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Attended task representation: (batch, chunk_dim)
        attended_task = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch, 1, n_chunks)
            task_chunks  # (batch, n_chunks, chunk_dim)
        ).squeeze(1)

        # Combine and predict
        combined = torch.cat([agent_emb, attended_task], dim=1)
        return self.mlp(combined).squeeze(-1)


class FeatureGatedModel(nn.Module):
    """Model where agent embedding gates task features.

    Architecture:
        agent_emb = Embedding(agent_idx) → (agent_emb_dim,)
        gate = sigmoid(Linear(agent_emb)) → (feature_dim,)
        gated_features = task_features * gate
        compressed = Linear(gated_features) → (hidden_dim,)
        combined = concat(agent_emb, compressed)
        score = MLP(combined) → sigmoid

    This allows each agent to learn which task features are important for them.
    """

    def __init__(
        self,
        n_agents: int,
        feature_dim: int,
        agent_emb_dim: int = 32,
        compressed_dim: int = 64,
        hidden_sizes: List[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 32]

        self.n_agents = n_agents
        self.feature_dim = feature_dim
        self.agent_emb_dim = agent_emb_dim

        # Agent embedding
        self.agent_embedding = nn.Embedding(n_agents, agent_emb_dim)

        # Gate: agent_emb → feature_dim gates
        self.gate_layer = nn.Linear(agent_emb_dim, feature_dim)

        # Compress gated features
        self.compress = nn.Sequential(
            nn.Linear(feature_dim, compressed_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        # MLP on concatenated [agent_emb | compressed]
        input_dim = agent_emb_dim + compressed_dim
        layers = []
        prev_dim = input_dim

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_size

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.mlp = nn.Sequential(*layers)

    def forward(self, agent_indices: torch.Tensor, task_features: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        agent_emb = self.agent_embedding(agent_indices)  # (batch, agent_emb_dim)

        # Compute agent-specific gates
        gates = torch.sigmoid(self.gate_layer(agent_emb))  # (batch, feature_dim)

        # Apply gates to task features
        gated_features = task_features * gates  # (batch, feature_dim)

        # Compress gated features
        compressed = self.compress(gated_features)  # (batch, compressed_dim)

        # Combine and predict
        combined = torch.cat([agent_emb, compressed], dim=1)
        return self.mlp(combined).squeeze(-1)


class AgentEmbeddingPredictor:
    """Predictor using learned low-dimensional agent embeddings.

    Instead of one-hot encoding (131 dims), learns a dense agent embedding (e.g., 32 dims).
    This reduces parameters and may help generalization.
    """

    def __init__(
        self,
        source: TaskFeatureSource,
        agent_emb_dim: int = 32,
        hidden_sizes: List[int] = None,
        dropout: float = 0.0,
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        n_epochs: int = 500,
        batch_size: int = None,  # None = full-batch, otherwise mini-batch
        verbose: bool = False,
        init_from_irt: bool = True,
        init_noise_scale: float = 0.0,
        early_stopping: bool = True,
        val_fraction: float = 0.1,
        patience: int = 30,
    ):
        if hidden_sizes is None:
            hidden_sizes = [64, 32]

        self.source = source
        self.agent_emb_dim = agent_emb_dim
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.init_from_irt = init_from_irt
        self.init_noise_scale = init_noise_scale
        self.early_stopping = early_stopping
        self.val_fraction = val_fraction
        self.patience = patience

        self._model: Optional[AgentEmbeddingModel] = None
        self._scaler: Optional[StandardScaler] = None
        self._agent_to_idx: Optional[Dict[str, int]] = None
        self._n_agents: int = 0
        self._feature_dim: int = 0
        self._is_fitted: bool = False
        self._train_auc: Optional[float] = None
        self._task_feature_cache: Dict[str, np.ndarray] = {}

    def fit(self, data: ExperimentData, train_task_ids: List[str]) -> None:
        """Fit the agent embedding model."""
        self._task_feature_cache = {}

        all_agents = data.get_all_agents()
        self._agent_to_idx = {agent: i for i, agent in enumerate(all_agents)}
        self._n_agents = len(all_agents)

        task_features = self.source.get_features(train_task_ids)
        self._scaler = StandardScaler()
        task_features_scaled = self._scaler.fit_transform(task_features)
        self._feature_dim = task_features_scaled.shape[1]

        task_to_features = {
            task_id: task_features_scaled[i]
            for i, task_id in enumerate(train_task_ids)
        }

        # Build training data with agent indices (not one-hot)
        agent_indices_list: List[int] = []
        features_list: List[np.ndarray] = []
        y_list: List[float] = []
        is_binomial = isinstance(data, BinomialExperimentData)

        for task_id in train_task_ids:
            task_feat = task_to_features[task_id]
            for agent_id in all_agents:
                if agent_id not in data.responses:
                    continue
                if task_id not in data.responses[agent_id]:
                    continue

                agent_idx = self._agent_to_idx[agent_id]
                response = data.responses[agent_id][task_id]

                if is_binomial:
                    k = response["successes"]
                    n = response["trials"]
                    for _ in range(k):
                        agent_indices_list.append(agent_idx)
                        features_list.append(task_feat)
                        y_list.append(1.0)
                    for _ in range(n - k):
                        agent_indices_list.append(agent_idx)
                        features_list.append(task_feat)
                        y_list.append(0.0)
                else:
                    agent_indices_list.append(agent_idx)
                    features_list.append(task_feat)
                    y_list.append(float(response))

        if len(y_list) == 0:
            raise ValueError("No training examples found")

        agent_indices = np.array(agent_indices_list, dtype=np.int64)
        features = np.array(features_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)

        if self.verbose:
            print(f"   Training AgentEmbedding: {len(y)} samples")
            print(f"   Agents: {self._n_agents}, Features: {self._feature_dim}")
            print(f"   agent_emb_dim={self.agent_emb_dim}, hidden_sizes={self.hidden_sizes}")

        self._model = AgentEmbeddingModel(
            n_agents=self._n_agents,
            feature_dim=self._feature_dim,
            agent_emb_dim=self.agent_emb_dim,
            hidden_sizes=self.hidden_sizes,
            dropout=self.dropout,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._model.to(device)

        # Initialize agent embeddings from IRT
        if self.init_from_irt:
            with torch.no_grad():
                for agent_id, idx in self._agent_to_idx.items():
                    if agent_id in data.train_abilities.index:
                        ability = float(data.train_abilities.loc[agent_id, "ability"])
                        # Initialize embedding to ability value (broadcast)
                        self._model.agent_embedding.weight.data[idx, :] = ability
                # Add noise to break symmetry across dimensions
                if self.init_noise_scale > 0:
                    noise = torch.randn_like(self._model.agent_embedding.weight.data) * self.init_noise_scale
                    self._model.agent_embedding.weight.data += noise
            if self.verbose:
                noise_msg = f" + noise(σ={self.init_noise_scale})" if self.init_noise_scale > 0 else ""
                print(f"   Initialized agent embeddings from IRT abilities{noise_msg}")

        agent_tensor = torch.tensor(agent_indices, dtype=torch.long, device=device)
        features_tensor = torch.tensor(features, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=device)

        if self.early_stopping:
            n_samples = len(y)
            n_val = max(1, int(n_samples * self.val_fraction))
            n_train = n_samples - n_val

            perm = torch.randperm(n_samples)
            train_idx = perm[:n_train]
            val_idx = perm[n_train:]

            train_agent = agent_tensor[train_idx]
            train_feat = features_tensor[train_idx]
            train_y = y_tensor[train_idx]
            val_agent = agent_tensor[val_idx]
            val_feat = features_tensor[val_idx]
            val_y = y_tensor[val_idx]

            if self.verbose:
                print(f"   Early stopping: {n_train} train, {n_val} val")
        else:
            train_agent = agent_tensor
            train_feat = features_tensor
            train_y = y_tensor

        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.BCEWithLogitsLoss()

        best_val_loss = float('inf')
        best_state_dict = None
        epochs_without_improvement = 0

        self._model.train()
        n_train = train_agent.shape[0]

        for epoch in range(self.n_epochs):
            # Mini-batch or full-batch training
            if self.batch_size is not None and self.batch_size < n_train:
                # Mini-batch: shuffle and iterate through batches
                perm = torch.randperm(n_train, device=device)
                epoch_loss = 0.0
                n_batches = 0
                for start in range(0, n_train, self.batch_size):
                    end = min(start + self.batch_size, n_train)
                    batch_idx = perm[start:end]

                    optimizer.zero_grad()
                    logits = self._model(train_agent[batch_idx], train_feat[batch_idx])
                    loss = criterion(logits, train_y[batch_idx])
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1
                loss_val = epoch_loss / n_batches
            else:
                # Full-batch training
                optimizer.zero_grad()
                logits = self._model(train_agent, train_feat)
                loss = criterion(logits, train_y)
                loss.backward()
                optimizer.step()
                loss_val = loss.item()

            if self.early_stopping:
                self._model.eval()
                with torch.no_grad():
                    val_logits = self._model(val_agent, val_feat)
                    val_loss = criterion(val_logits, val_y).item()
                self._model.train()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state_dict = {k: v.clone() for k, v in self._model.state_dict().items()}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= self.patience:
                    if self.verbose:
                        print(f"      Early stopping at epoch {epoch + 1}")
                    break

                if self.verbose and (epoch + 1) % 50 == 0:
                    print(f"      Epoch {epoch + 1}: train_loss={loss_val:.4f}, val_loss={val_loss:.4f}")
            else:
                if self.verbose and (epoch + 1) % 50 == 0:
                    print(f"      Epoch {epoch + 1}: loss={loss_val:.4f}")

        if self.early_stopping and best_state_dict is not None:
            self._model.load_state_dict(best_state_dict)

        self._is_fitted = True

        # Compute train AUC (model outputs logits, apply sigmoid)
        self._model.eval()
        with torch.no_grad():
            logits_train = self._model(agent_tensor, features_tensor)
            y_pred_train = torch.sigmoid(logits_train).cpu().numpy()
        self._train_auc = roc_auc_score(y, y_pred_train) if len(np.unique(y)) > 1 else None

        if self.verbose:
            train_auc_str = f"{self._train_auc:.4f}" if self._train_auc else "N/A"
            print(f"   Final: train_auc={train_auc_str}")

    def predict_probability(self, data: ExperimentData, agent_id: str, task_id: str) -> float:
        """Predict success probability."""
        if not self._is_fitted:
            raise RuntimeError("Predictor must be fit first")

        if task_id not in self._task_feature_cache:
            self._cache_test_task_features(data.test_tasks)

        if task_id not in self._task_feature_cache:
            raise ValueError(f"No features for task {task_id}")
        if agent_id not in self._agent_to_idx:
            raise ValueError(f"Unknown agent {agent_id}")

        agent_idx = self._agent_to_idx[agent_id]
        task_feat = self._task_feature_cache[task_id]

        device = next(self._model.parameters()).device
        self._model.eval()
        with torch.no_grad():
            agent_tensor = torch.tensor([agent_idx], dtype=torch.long, device=device)
            feat_tensor = torch.tensor(task_feat, dtype=torch.float32, device=device).unsqueeze(0)
            logit = self._model(agent_tensor, feat_tensor)
            prob = torch.sigmoid(logit).item()
        return prob

    def _cache_test_task_features(self, test_tasks: List[str]) -> None:
        """Cache scaled features for test tasks."""
        features = self.source.get_features(test_tasks)
        features_scaled = self._scaler.transform(features)
        for i, task_id in enumerate(test_tasks):
            self._task_feature_cache[task_id] = features_scaled[i]

    def get_train_auc(self) -> Optional[float]:
        return self._train_auc

    @property
    def name(self) -> str:
        return f"AgentEmb-{self.agent_emb_dim}"


class TaskBottleneckPredictor:
    """Predictor using task bottleneck to compress high-dim features.

    Compresses task features (e.g., 5120 dims) to a bottleneck (e.g., 64 dims)
    before combining with agent embeddings. This creates balanced representations.
    """

    def __init__(
        self,
        source: TaskFeatureSource,
        agent_emb_dim: int = 32,
        task_bottleneck_dim: int = 64,
        hidden_sizes: List[int] = None,
        dropout: float = 0.0,
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        n_epochs: int = 500,
        verbose: bool = False,
        init_from_irt: bool = True,
        early_stopping: bool = True,
        val_fraction: float = 0.1,
        patience: int = 30,
    ):
        if hidden_sizes is None:
            hidden_sizes = [64, 32]

        self.source = source
        self.agent_emb_dim = agent_emb_dim
        self.task_bottleneck_dim = task_bottleneck_dim
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.init_from_irt = init_from_irt
        self.early_stopping = early_stopping
        self.val_fraction = val_fraction
        self.patience = patience

        self._model: Optional[TaskBottleneckModel] = None
        self._scaler: Optional[StandardScaler] = None
        self._agent_to_idx: Optional[Dict[str, int]] = None
        self._n_agents: int = 0
        self._feature_dim: int = 0
        self._is_fitted: bool = False
        self._train_auc: Optional[float] = None
        self._task_feature_cache: Dict[str, np.ndarray] = {}

    def fit(self, data: ExperimentData, train_task_ids: List[str]) -> None:
        """Fit the task bottleneck model."""
        self._task_feature_cache = {}

        all_agents = data.get_all_agents()
        self._agent_to_idx = {agent: i for i, agent in enumerate(all_agents)}
        self._n_agents = len(all_agents)

        task_features = self.source.get_features(train_task_ids)
        self._scaler = StandardScaler()
        task_features_scaled = self._scaler.fit_transform(task_features)
        self._feature_dim = task_features_scaled.shape[1]

        task_to_features = {
            task_id: task_features_scaled[i]
            for i, task_id in enumerate(train_task_ids)
        }

        # Build training data with agent indices
        agent_indices_list: List[int] = []
        features_list: List[np.ndarray] = []
        y_list: List[float] = []
        is_binomial = isinstance(data, BinomialExperimentData)

        for task_id in train_task_ids:
            task_feat = task_to_features[task_id]
            for agent_id in all_agents:
                if agent_id not in data.responses:
                    continue
                if task_id not in data.responses[agent_id]:
                    continue

                agent_idx = self._agent_to_idx[agent_id]
                response = data.responses[agent_id][task_id]

                if is_binomial:
                    k = response["successes"]
                    n = response["trials"]
                    for _ in range(k):
                        agent_indices_list.append(agent_idx)
                        features_list.append(task_feat)
                        y_list.append(1.0)
                    for _ in range(n - k):
                        agent_indices_list.append(agent_idx)
                        features_list.append(task_feat)
                        y_list.append(0.0)
                else:
                    agent_indices_list.append(agent_idx)
                    features_list.append(task_feat)
                    y_list.append(float(response))

        if len(y_list) == 0:
            raise ValueError("No training examples found")

        agent_indices = np.array(agent_indices_list, dtype=np.int64)
        features = np.array(features_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)

        if self.verbose:
            print(f"   Training TaskBottleneck: {len(y)} samples")
            print(f"   Agents: {self._n_agents}, Features: {self._feature_dim}")
            print(f"   agent_emb={self.agent_emb_dim}, bottleneck={self.task_bottleneck_dim}")

        self._model = TaskBottleneckModel(
            n_agents=self._n_agents,
            feature_dim=self._feature_dim,
            agent_emb_dim=self.agent_emb_dim,
            task_bottleneck_dim=self.task_bottleneck_dim,
            hidden_sizes=self.hidden_sizes,
            dropout=self.dropout,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._model.to(device)

        # Initialize agent embeddings from IRT
        if self.init_from_irt:
            with torch.no_grad():
                for agent_id, idx in self._agent_to_idx.items():
                    if agent_id in data.train_abilities.index:
                        ability = float(data.train_abilities.loc[agent_id, "ability"])
                        self._model.agent_embedding.weight.data[idx, :] = ability
            if self.verbose:
                print(f"   Initialized agent embeddings from IRT abilities")

        agent_tensor = torch.tensor(agent_indices, dtype=torch.long, device=device)
        features_tensor = torch.tensor(features, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=device)

        if self.early_stopping:
            n_samples = len(y)
            n_val = max(1, int(n_samples * self.val_fraction))
            n_train = n_samples - n_val

            perm = torch.randperm(n_samples)
            train_idx = perm[:n_train]
            val_idx = perm[n_train:]

            train_agent = agent_tensor[train_idx]
            train_feat = features_tensor[train_idx]
            train_y = y_tensor[train_idx]
            val_agent = agent_tensor[val_idx]
            val_feat = features_tensor[val_idx]
            val_y = y_tensor[val_idx]

            if self.verbose:
                print(f"   Early stopping: {n_train} train, {n_val} val")
        else:
            train_agent = agent_tensor
            train_feat = features_tensor
            train_y = y_tensor

        optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.BCELoss()

        best_val_loss = float('inf')
        best_state_dict = None
        epochs_without_improvement = 0

        self._model.train()
        for epoch in range(self.n_epochs):
            optimizer.zero_grad()
            y_pred = self._model(train_agent, train_feat)
            loss = criterion(y_pred, train_y)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()

            if self.early_stopping:
                self._model.eval()
                with torch.no_grad():
                    val_pred = self._model(val_agent, val_feat)
                    val_loss = criterion(val_pred, val_y).item()
                self._model.train()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state_dict = {k: v.clone() for k, v in self._model.state_dict().items()}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= self.patience:
                    if self.verbose:
                        print(f"      Early stopping at epoch {epoch + 1}")
                    break

                if self.verbose and (epoch + 1) % 50 == 0:
                    print(f"      Epoch {epoch + 1}: train_loss={loss_val:.4f}, val_loss={val_loss:.4f}")
            else:
                if self.verbose and (epoch + 1) % 50 == 0:
                    print(f"      Epoch {epoch + 1}: loss={loss_val:.4f}")

        if self.early_stopping and best_state_dict is not None:
            self._model.load_state_dict(best_state_dict)

        self._is_fitted = True

        # Compute train AUC
        self._model.eval()
        with torch.no_grad():
            y_pred_train = self._model(agent_tensor, features_tensor).cpu().numpy()
        self._train_auc = roc_auc_score(y, y_pred_train) if len(np.unique(y)) > 1 else None

        if self.verbose:
            train_auc_str = f"{self._train_auc:.4f}" if self._train_auc else "N/A"
            print(f"   Final: train_auc={train_auc_str}")

    def predict_probability(self, data: ExperimentData, agent_id: str, task_id: str) -> float:
        """Predict success probability."""
        if not self._is_fitted:
            raise RuntimeError("Predictor must be fit first")

        if task_id not in self._task_feature_cache:
            self._cache_test_task_features(data.test_tasks)

        if task_id not in self._task_feature_cache:
            raise ValueError(f"No features for task {task_id}")
        if agent_id not in self._agent_to_idx:
            raise ValueError(f"Unknown agent {agent_id}")

        agent_idx = self._agent_to_idx[agent_id]
        task_feat = self._task_feature_cache[task_id]

        device = next(self._model.parameters()).device
        self._model.eval()
        with torch.no_grad():
            agent_tensor = torch.tensor([agent_idx], dtype=torch.long, device=device)
            feat_tensor = torch.tensor(task_feat, dtype=torch.float32, device=device).unsqueeze(0)
            prob = self._model(agent_tensor, feat_tensor).item()
        return prob

    def _cache_test_task_features(self, test_tasks: List[str]) -> None:
        """Cache scaled features for test tasks."""
        features = self.source.get_features(test_tasks)
        features_scaled = self._scaler.transform(features)
        for i, task_id in enumerate(test_tasks):
            self._task_feature_cache[task_id] = features_scaled[i]

    def get_train_auc(self) -> Optional[float]:
        return self._train_auc

    @property
    def name(self) -> str:
        return f"TaskBottleneck-{self.task_bottleneck_dim}"


class CrossAttentionPredictor:
    """Predictor using cross-attention between agent and task features.

    Agent embedding attends to chunks of task features, allowing different
    agents to focus on different aspects of tasks.
    """

    def __init__(
        self,
        source: TaskFeatureSource,
        agent_emb_dim: int = 32,
        n_chunks: int = 64,
        hidden_sizes: List[int] = None,
        dropout: float = 0.0,
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        n_epochs: int = 500,
        verbose: bool = False,
        init_from_irt: bool = True,
        early_stopping: bool = True,
        val_fraction: float = 0.1,
        patience: int = 30,
    ):
        if hidden_sizes is None:
            hidden_sizes = [64, 32]

        self.source = source
        self.agent_emb_dim = agent_emb_dim
        self.n_chunks = n_chunks
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.init_from_irt = init_from_irt
        self.early_stopping = early_stopping
        self.val_fraction = val_fraction
        self.patience = patience

        self._model: Optional[CrossAttentionModel] = None
        self._scaler: Optional[StandardScaler] = None
        self._agent_to_idx: Optional[Dict[str, int]] = None
        self._n_agents: int = 0
        self._feature_dim: int = 0
        self._is_fitted: bool = False
        self._train_auc: Optional[float] = None
        self._task_feature_cache: Dict[str, np.ndarray] = {}

    def fit(self, data: ExperimentData, train_task_ids: List[str]) -> None:
        """Fit the cross-attention model."""
        self._task_feature_cache = {}

        all_agents = data.get_all_agents()
        self._agent_to_idx = {agent: i for i, agent in enumerate(all_agents)}
        self._n_agents = len(all_agents)

        task_features = self.source.get_features(train_task_ids)
        self._scaler = StandardScaler()
        task_features_scaled = self._scaler.fit_transform(task_features)
        self._feature_dim = task_features_scaled.shape[1]

        task_to_features = {
            task_id: task_features_scaled[i]
            for i, task_id in enumerate(train_task_ids)
        }

        # Build training data
        agent_indices_list: List[int] = []
        features_list: List[np.ndarray] = []
        y_list: List[float] = []
        is_binomial = isinstance(data, BinomialExperimentData)

        for task_id in train_task_ids:
            task_feat = task_to_features[task_id]
            for agent_id in all_agents:
                if agent_id not in data.responses:
                    continue
                if task_id not in data.responses[agent_id]:
                    continue

                agent_idx = self._agent_to_idx[agent_id]
                response = data.responses[agent_id][task_id]

                if is_binomial:
                    k = response["successes"]
                    n = response["trials"]
                    for _ in range(k):
                        agent_indices_list.append(agent_idx)
                        features_list.append(task_feat)
                        y_list.append(1.0)
                    for _ in range(n - k):
                        agent_indices_list.append(agent_idx)
                        features_list.append(task_feat)
                        y_list.append(0.0)
                else:
                    agent_indices_list.append(agent_idx)
                    features_list.append(task_feat)
                    y_list.append(float(response))

        if len(y_list) == 0:
            raise ValueError("No training examples found")

        agent_indices = np.array(agent_indices_list, dtype=np.int64)
        features = np.array(features_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)

        if self.verbose:
            print(f"   Training CrossAttention: {len(y)} samples")
            print(f"   Agents: {self._n_agents}, Features: {self._feature_dim}")
            print(f"   agent_emb={self.agent_emb_dim}, n_chunks={self.n_chunks}")

        self._model = CrossAttentionModel(
            n_agents=self._n_agents,
            feature_dim=self._feature_dim,
            agent_emb_dim=self.agent_emb_dim,
            n_chunks=self.n_chunks,
            hidden_sizes=self.hidden_sizes,
            dropout=self.dropout,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._model.to(device)

        # Initialize agent embeddings from IRT
        if self.init_from_irt:
            with torch.no_grad():
                for agent_id, idx in self._agent_to_idx.items():
                    if agent_id in data.train_abilities.index:
                        ability = float(data.train_abilities.loc[agent_id, "ability"])
                        self._model.agent_embedding.weight.data[idx, :] = ability
            if self.verbose:
                print(f"   Initialized agent embeddings from IRT abilities")

        agent_tensor = torch.tensor(agent_indices, dtype=torch.long, device=device)
        features_tensor = torch.tensor(features, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=device)

        if self.early_stopping:
            n_samples = len(y)
            n_val = max(1, int(n_samples * self.val_fraction))
            n_train = n_samples - n_val

            perm = torch.randperm(n_samples)
            train_idx = perm[:n_train]
            val_idx = perm[n_train:]

            train_agent = agent_tensor[train_idx]
            train_feat = features_tensor[train_idx]
            train_y = y_tensor[train_idx]
            val_agent = agent_tensor[val_idx]
            val_feat = features_tensor[val_idx]
            val_y = y_tensor[val_idx]

            if self.verbose:
                print(f"   Early stopping: {n_train} train, {n_val} val")
        else:
            train_agent = agent_tensor
            train_feat = features_tensor
            train_y = y_tensor

        optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.BCELoss()

        best_val_loss = float('inf')
        best_state_dict = None
        epochs_without_improvement = 0

        self._model.train()
        for epoch in range(self.n_epochs):
            optimizer.zero_grad()
            y_pred = self._model(train_agent, train_feat)
            loss = criterion(y_pred, train_y)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()

            if self.early_stopping:
                self._model.eval()
                with torch.no_grad():
                    val_pred = self._model(val_agent, val_feat)
                    val_loss = criterion(val_pred, val_y).item()
                self._model.train()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state_dict = {k: v.clone() for k, v in self._model.state_dict().items()}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= self.patience:
                    if self.verbose:
                        print(f"      Early stopping at epoch {epoch + 1}")
                    break

                if self.verbose and (epoch + 1) % 50 == 0:
                    print(f"      Epoch {epoch + 1}: train_loss={loss_val:.4f}, val_loss={val_loss:.4f}")
            else:
                if self.verbose and (epoch + 1) % 50 == 0:
                    print(f"      Epoch {epoch + 1}: loss={loss_val:.4f}")

        if self.early_stopping and best_state_dict is not None:
            self._model.load_state_dict(best_state_dict)

        self._is_fitted = True

        # Compute train AUC
        self._model.eval()
        with torch.no_grad():
            y_pred_train = self._model(agent_tensor, features_tensor).cpu().numpy()
        self._train_auc = roc_auc_score(y, y_pred_train) if len(np.unique(y)) > 1 else None

        if self.verbose:
            train_auc_str = f"{self._train_auc:.4f}" if self._train_auc else "N/A"
            print(f"   Final: train_auc={train_auc_str}")

    def predict_probability(self, data: ExperimentData, agent_id: str, task_id: str) -> float:
        """Predict success probability."""
        if not self._is_fitted:
            raise RuntimeError("Predictor must be fit first")

        if task_id not in self._task_feature_cache:
            self._cache_test_task_features(data.test_tasks)

        if task_id not in self._task_feature_cache:
            raise ValueError(f"No features for task {task_id}")
        if agent_id not in self._agent_to_idx:
            raise ValueError(f"Unknown agent {agent_id}")

        agent_idx = self._agent_to_idx[agent_id]
        task_feat = self._task_feature_cache[task_id]

        device = next(self._model.parameters()).device
        self._model.eval()
        with torch.no_grad():
            agent_tensor = torch.tensor([agent_idx], dtype=torch.long, device=device)
            feat_tensor = torch.tensor(task_feat, dtype=torch.float32, device=device).unsqueeze(0)
            prob = self._model(agent_tensor, feat_tensor).item()
        return prob

    def _cache_test_task_features(self, test_tasks: List[str]) -> None:
        """Cache scaled features for test tasks."""
        features = self.source.get_features(test_tasks)
        features_scaled = self._scaler.transform(features)
        for i, task_id in enumerate(test_tasks):
            self._task_feature_cache[task_id] = features_scaled[i]

    def get_train_auc(self) -> Optional[float]:
        return self._train_auc

    @property
    def name(self) -> str:
        return f"CrossAttn-{self.n_chunks}"


class FeatureGatedPredictor:
    """Predictor using agent-specific feature gating.

    Agent embedding learns to gate (weight) which task features are important,
    allowing different agents to rely on different task characteristics.
    """

    def __init__(
        self,
        source: TaskFeatureSource,
        agent_emb_dim: int = 32,
        compressed_dim: int = 64,
        hidden_sizes: List[int] = None,
        dropout: float = 0.0,
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        n_epochs: int = 500,
        verbose: bool = False,
        init_from_irt: bool = True,
        early_stopping: bool = True,
        val_fraction: float = 0.1,
        patience: int = 30,
    ):
        if hidden_sizes is None:
            hidden_sizes = [64, 32]

        self.source = source
        self.agent_emb_dim = agent_emb_dim
        self.compressed_dim = compressed_dim
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.init_from_irt = init_from_irt
        self.early_stopping = early_stopping
        self.val_fraction = val_fraction
        self.patience = patience

        self._model: Optional[FeatureGatedModel] = None
        self._scaler: Optional[StandardScaler] = None
        self._agent_to_idx: Optional[Dict[str, int]] = None
        self._n_agents: int = 0
        self._feature_dim: int = 0
        self._is_fitted: bool = False
        self._train_auc: Optional[float] = None
        self._task_feature_cache: Dict[str, np.ndarray] = {}

    def fit(self, data: ExperimentData, train_task_ids: List[str]) -> None:
        """Fit the feature-gated model."""
        self._task_feature_cache = {}

        all_agents = data.get_all_agents()
        self._agent_to_idx = {agent: i for i, agent in enumerate(all_agents)}
        self._n_agents = len(all_agents)

        task_features = self.source.get_features(train_task_ids)
        self._scaler = StandardScaler()
        task_features_scaled = self._scaler.fit_transform(task_features)
        self._feature_dim = task_features_scaled.shape[1]

        task_to_features = {
            task_id: task_features_scaled[i]
            for i, task_id in enumerate(train_task_ids)
        }

        # Build training data
        agent_indices_list: List[int] = []
        features_list: List[np.ndarray] = []
        y_list: List[float] = []
        is_binomial = isinstance(data, BinomialExperimentData)

        for task_id in train_task_ids:
            task_feat = task_to_features[task_id]
            for agent_id in all_agents:
                if agent_id not in data.responses:
                    continue
                if task_id not in data.responses[agent_id]:
                    continue

                agent_idx = self._agent_to_idx[agent_id]
                response = data.responses[agent_id][task_id]

                if is_binomial:
                    k = response["successes"]
                    n = response["trials"]
                    for _ in range(k):
                        agent_indices_list.append(agent_idx)
                        features_list.append(task_feat)
                        y_list.append(1.0)
                    for _ in range(n - k):
                        agent_indices_list.append(agent_idx)
                        features_list.append(task_feat)
                        y_list.append(0.0)
                else:
                    agent_indices_list.append(agent_idx)
                    features_list.append(task_feat)
                    y_list.append(float(response))

        if len(y_list) == 0:
            raise ValueError("No training examples found")

        agent_indices = np.array(agent_indices_list, dtype=np.int64)
        features = np.array(features_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)

        if self.verbose:
            print(f"   Training FeatureGated: {len(y)} samples")
            print(f"   Agents: {self._n_agents}, Features: {self._feature_dim}")
            print(f"   agent_emb={self.agent_emb_dim}, compressed={self.compressed_dim}")

        self._model = FeatureGatedModel(
            n_agents=self._n_agents,
            feature_dim=self._feature_dim,
            agent_emb_dim=self.agent_emb_dim,
            compressed_dim=self.compressed_dim,
            hidden_sizes=self.hidden_sizes,
            dropout=self.dropout,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._model.to(device)

        # Initialize agent embeddings from IRT
        if self.init_from_irt:
            with torch.no_grad():
                for agent_id, idx in self._agent_to_idx.items():
                    if agent_id in data.train_abilities.index:
                        ability = float(data.train_abilities.loc[agent_id, "ability"])
                        self._model.agent_embedding.weight.data[idx, :] = ability
            if self.verbose:
                print(f"   Initialized agent embeddings from IRT abilities")

        agent_tensor = torch.tensor(agent_indices, dtype=torch.long, device=device)
        features_tensor = torch.tensor(features, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=device)

        if self.early_stopping:
            n_samples = len(y)
            n_val = max(1, int(n_samples * self.val_fraction))
            n_train = n_samples - n_val

            perm = torch.randperm(n_samples)
            train_idx = perm[:n_train]
            val_idx = perm[n_train:]

            train_agent = agent_tensor[train_idx]
            train_feat = features_tensor[train_idx]
            train_y = y_tensor[train_idx]
            val_agent = agent_tensor[val_idx]
            val_feat = features_tensor[val_idx]
            val_y = y_tensor[val_idx]

            if self.verbose:
                print(f"   Early stopping: {n_train} train, {n_val} val")
        else:
            train_agent = agent_tensor
            train_feat = features_tensor
            train_y = y_tensor

        optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.BCELoss()

        best_val_loss = float('inf')
        best_state_dict = None
        epochs_without_improvement = 0

        self._model.train()
        for epoch in range(self.n_epochs):
            optimizer.zero_grad()
            y_pred = self._model(train_agent, train_feat)
            loss = criterion(y_pred, train_y)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()

            if self.early_stopping:
                self._model.eval()
                with torch.no_grad():
                    val_pred = self._model(val_agent, val_feat)
                    val_loss = criterion(val_pred, val_y).item()
                self._model.train()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state_dict = {k: v.clone() for k, v in self._model.state_dict().items()}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= self.patience:
                    if self.verbose:
                        print(f"      Early stopping at epoch {epoch + 1}")
                    break

                if self.verbose and (epoch + 1) % 50 == 0:
                    print(f"      Epoch {epoch + 1}: train_loss={loss_val:.4f}, val_loss={val_loss:.4f}")
            else:
                if self.verbose and (epoch + 1) % 50 == 0:
                    print(f"      Epoch {epoch + 1}: loss={loss_val:.4f}")

        if self.early_stopping and best_state_dict is not None:
            self._model.load_state_dict(best_state_dict)

        self._is_fitted = True

        # Compute train AUC
        self._model.eval()
        with torch.no_grad():
            y_pred_train = self._model(agent_tensor, features_tensor).cpu().numpy()
        self._train_auc = roc_auc_score(y, y_pred_train) if len(np.unique(y)) > 1 else None

        if self.verbose:
            train_auc_str = f"{self._train_auc:.4f}" if self._train_auc else "N/A"
            print(f"   Final: train_auc={train_auc_str}")

    def predict_probability(self, data: ExperimentData, agent_id: str, task_id: str) -> float:
        """Predict success probability."""
        if not self._is_fitted:
            raise RuntimeError("Predictor must be fit first")

        if task_id not in self._task_feature_cache:
            self._cache_test_task_features(data.test_tasks)

        if task_id not in self._task_feature_cache:
            raise ValueError(f"No features for task {task_id}")
        if agent_id not in self._agent_to_idx:
            raise ValueError(f"Unknown agent {agent_id}")

        agent_idx = self._agent_to_idx[agent_id]
        task_feat = self._task_feature_cache[task_id]

        device = next(self._model.parameters()).device
        self._model.eval()
        with torch.no_grad():
            agent_tensor = torch.tensor([agent_idx], dtype=torch.long, device=device)
            feat_tensor = torch.tensor(task_feat, dtype=torch.float32, device=device).unsqueeze(0)
            prob = self._model(agent_tensor, feat_tensor).item()
        return prob

    def _cache_test_task_features(self, test_tasks: List[str]) -> None:
        """Cache scaled features for test tasks."""
        features = self.source.get_features(test_tasks)
        features_scaled = self._scaler.transform(features)
        for i, task_id in enumerate(test_tasks):
            self._task_feature_cache[task_id] = features_scaled[i]

    def get_train_auc(self) -> Optional[float]:
        return self._train_auc

    @property
    def name(self) -> str:
        return f"FeatureGated-{self.compressed_dim}"
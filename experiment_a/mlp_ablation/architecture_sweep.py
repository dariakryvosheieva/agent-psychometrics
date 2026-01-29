"""Architecture sweep: Test deeper networks and SwiGLU activation.

Fixed from weight decay sweep:
- weight_decay = 0.2 (best from previous sweep)
- early_stopping = True
- init_from_irt = True

Varying:
1. Network architecture: SimpleMLP, DeepMLP, SwiGLUMLP
2. Hidden layer configurations
3. Early stopping metric: loss vs AUC

Usage:
    python -m experiment_a.mlp_ablation.architecture_sweep
    sbatch experiment_a/mlp_ablation/slurm_architecture_sweep.sh
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from experiment_ab_shared.feature_source import EmbeddingFeatureSource, TaskFeatureSource
from experiment_ab_shared.feature_predictor import FeatureBasedPredictor
from experiment_ab_shared.dataset import BinomialExperimentData, ExperimentData
from experiment_a.shared.cross_validation import k_fold_split_tasks, run_cv, CVPredictor
from experiment_a.shared.pipeline import CVPredictorConfig
from experiment_a.shared.mlp_predictor import (
    SimpleMLP, DeepMLP, SwiGLUMLP, build_input_vector
)
from experiment_a.shared.baselines import (
    OraclePredictor,
    ConstantPredictor,
    DifficultyPredictorAdapter,
)
from experiment_a.swebench.config import ExperimentAConfig
from experiment_ab_shared import load_dataset_for_fold


def extract_train_auc(predictor: CVPredictor, fold_idx: int) -> float | None:
    """Extract train AUC from fitted predictor for diagnostics."""
    if hasattr(predictor, 'get_train_auc'):
        return predictor.get_train_auc()
    return None


class FlexibleMLPPredictor:
    """Flexible MLP predictor supporting different architectures and early stopping metrics."""

    def __init__(
        self,
        source: TaskFeatureSource,
        architecture: str = "simple",  # "simple", "deep", "swiglu"
        hidden_sizes: Union[int, List[int]] = 1024,
        dropout: float = 0.0,
        learning_rate: float = 0.01,
        weight_decay: float = 0.2,
        n_epochs: int = 1000,
        verbose: bool = False,
        init_from_irt: bool = True,
        early_stopping: bool = True,
        early_stopping_metric: str = "loss",  # "loss" or "auc"
        val_fraction: float = 0.1,
        patience: int = 30,
    ):
        self.source = source
        self.architecture = architecture
        self.hidden_sizes = hidden_sizes if isinstance(hidden_sizes, list) else [hidden_sizes]
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.init_from_irt = init_from_irt
        self.early_stopping = early_stopping
        self.early_stopping_metric = early_stopping_metric
        self.val_fraction = val_fraction
        self.patience = patience

        # Model state
        self._model: Optional[nn.Module] = None
        self._scaler: Optional[StandardScaler] = None
        self._agent_to_idx: Optional[Dict[str, int]] = None
        self._n_agents: int = 0
        self._feature_dim: int = 0
        self._is_fitted: bool = False
        self._training_losses: List[float] = []
        self._train_auc: Optional[float] = None
        self._task_feature_cache: Dict[str, np.ndarray] = {}

    def _create_model(self, input_dim: int) -> nn.Module:
        """Create the appropriate model architecture."""
        if self.architecture == "simple":
            return SimpleMLP(input_dim, self.hidden_sizes[0], dropout=self.dropout)
        elif self.architecture == "deep":
            return DeepMLP(input_dim, self.hidden_sizes, dropout=self.dropout)
        elif self.architecture == "swiglu":
            return SwiGLUMLP(input_dim, self.hidden_sizes[0], dropout=self.dropout)
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")

    def _init_agent_weights_from_irt(self, data: ExperimentData) -> None:
        """Initialize the agent-portion of first layer weights from IRT abilities.

        Works for all architectures by finding the first linear layer(s) that take
        the concatenated [agent_one_hot | features] input.
        """
        with torch.no_grad():
            # Get ability values for each agent
            ability_values = {}
            for agent_id, idx in self._agent_to_idx.items():
                if agent_id in data.train_abilities.index:
                    ability_values[idx] = float(data.train_abilities.loc[agent_id, "ability"])
                else:
                    ability_values[idx] = 0.0

            if self.architecture == "simple":
                # SimpleMLP: fc1 is the first layer
                for idx, ability in ability_values.items():
                    self._model.fc1.weight.data[:, idx] = ability

            elif self.architecture == "deep":
                # DeepMLP: first layer in network sequential is nn.Linear
                first_layer = self._model.network[0]
                for idx, ability in ability_values.items():
                    first_layer.weight.data[:, idx] = ability

            elif self.architecture == "swiglu":
                # SwiGLU: both gate_proj and up_proj take the input
                for idx, ability in ability_values.items():
                    self._model.gate_proj.weight.data[:, idx] = ability
                    self._model.up_proj.weight.data[:, idx] = ability

        if self.verbose:
            print(f"   Initialized agent weights from IRT abilities")

    def fit(self, data: ExperimentData, train_task_ids: List[str]) -> None:
        """Fit the MLP on training data."""
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
            print(f"   Training {self.architecture} MLP: {len(X)} samples")
            print(f"   Input dim: {X.shape[1]}, hidden_sizes={self.hidden_sizes}")
            print(f"   weight_decay={self.weight_decay}, dropout={self.dropout}")
            print(f"   Early stopping metric: {self.early_stopping_metric}")

        # Create model
        input_dim = self._n_agents + self._feature_dim
        self._model = self._create_model(input_dim)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._model.to(device)

        if self.verbose and device.type == "cuda":
            print(f"   Using GPU: {torch.cuda.get_device_name(0)}")

        # Initialize agent weights from IRT
        if self.init_from_irt:
            self._init_agent_weights_from_irt(data)

        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=device)

        # Split for early stopping
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
                print(f"   Early stopping ({self.early_stopping_metric}): {n_train} train, {n_val} val")
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
        best_metric = float('inf') if self.early_stopping_metric == "loss" else float('-inf')
        best_state_dict = None
        epochs_without_improvement = 0

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

                    # Compute validation AUC if using AUC metric
                    if self.early_stopping_metric == "auc":
                        val_pred_np = val_pred.cpu().numpy()
                        val_y_np = val_y.cpu().numpy()
                        if len(np.unique(val_y_np)) > 1:
                            val_auc = roc_auc_score(val_y_np, val_pred_np)
                        else:
                            val_auc = 0.5
                        current_metric = val_auc
                        improved = current_metric > best_metric
                    else:
                        current_metric = val_loss
                        improved = current_metric < best_metric

                self._model.train()

                if improved:
                    best_metric = current_metric
                    best_state_dict = {k: v.clone() for k, v in self._model.state_dict().items()}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= self.patience:
                    if self.verbose:
                        metric_str = f"{self.early_stopping_metric}={current_metric:.4f}"
                        print(f"      Early stopping at epoch {epoch + 1} ({metric_str})")
                    break

                if self.verbose and (epoch + 1) % 50 == 0:
                    if self.early_stopping_metric == "auc":
                        print(f"      Epoch {epoch + 1}: loss={loss_val:.4f}, val_auc={current_metric:.4f}")
                    else:
                        print(f"      Epoch {epoch + 1}: loss={loss_val:.4f}, val_loss={current_metric:.4f}")
            else:
                if self.verbose and (epoch + 1) % 50 == 0:
                    print(f"      Epoch {epoch + 1}: loss={loss_val:.4f}")

        # Restore best model
        if self.early_stopping and best_state_dict is not None:
            self._model.load_state_dict(best_state_dict)

        self._is_fitted = True

        # Compute train AUC
        self._model.eval()
        with torch.no_grad():
            y_pred_train = self._model(X_tensor).cpu().numpy()
        if len(np.unique(y)) > 1:
            self._train_auc = roc_auc_score(y, y_pred_train)
        else:
            self._train_auc = None

        if self.verbose:
            train_auc_str = f"{self._train_auc:.4f}" if self._train_auc else "N/A"
            print(f"   Final: loss={self._training_losses[-1]:.4f}, train_auc={train_auc_str}")

    def predict_probability(self, data: ExperimentData, agent_id: str, task_id: str) -> float:
        """Predict success probability for a specific (agent, task) pair."""
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
        return f"FlexMLP-{self.architecture}"


ROOT = Path(__file__).parent.parent.parent


def main():
    parser = argparse.ArgumentParser(description="Architecture sweep for MLP")
    parser.add_argument("--k_folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--part", type=int, choices=[1, 2], default=None,
                        help="Run only part 1 or 2 (for parallel execution)")
    args = parser.parse_args()

    config = ExperimentAConfig()

    # Resolve paths
    embeddings_path = ROOT / config.embeddings_path
    abilities_path = ROOT / config.abilities_path
    items_path = ROOT / config.items_path
    responses_path = ROOT / config.responses_path

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")

    embedding_source = EmbeddingFeatureSource(embeddings_path)
    print(f"Loaded embeddings: {embedding_source.feature_dim} dimensions")

    full_items = pd.read_csv(items_path, index_col=0)
    all_task_ids = list(full_items.index)

    k_folds = args.k_folds
    folds = k_fold_split_tasks(all_task_ids, k=k_folds, seed=config.split_seed)

    def load_fold_data(train_tasks, test_tasks, fold_idx):
        return load_dataset_for_fold(
            abilities_path=abilities_path,
            items_path=items_path,
            responses_path=responses_path,
            train_tasks=train_tasks,
            test_tasks=test_tasks,
            fold_idx=fold_idx,
            k_folds=k_folds,
            split_seed=config.split_seed,
            is_binomial=False,
            irt_cache_dir=Path("chris_output/experiment_a/irt_splits"),
        )

    # Fixed parameters from weight decay sweep
    weight_decay = 0.2
    learning_rate = 0.01
    n_epochs = 1000

    configs: List[CVPredictorConfig] = []

    # === Baselines (only in part 1) ===
    if args.part is None or args.part == 1:
        configs.append(CVPredictorConfig(
            predictor=OraclePredictor(),
            name="oracle",
            display_name="Oracle (true β)",
        ))
        configs.append(CVPredictorConfig(
            predictor=ConstantPredictor(),
            name="constant",
            display_name="Constant (mean β)",
        ))
        configs.append(CVPredictorConfig(
            predictor=DifficultyPredictorAdapter(
                FeatureBasedPredictor(embedding_source, alphas=config.ridge_alphas)
            ),
            name="ridge",
            display_name="Ridge (Embedding)",
        ))

    # === Architecture experiments ===
    # All configs to test
    all_configs = []

    # 1. Baseline: SimpleMLP h=1024 with loss-based early stopping (best from previous)
    all_configs.append({
        "name": "simple_h1024_loss",
        "display": "SimpleMLP (h=1024, stop=loss)",
        "architecture": "simple",
        "hidden_sizes": 1024,
        "early_stopping_metric": "loss",
        "dropout": 0.0,
    })

    # 2. SimpleMLP with AUC-based early stopping
    all_configs.append({
        "name": "simple_h1024_auc",
        "display": "SimpleMLP (h=1024, stop=AUC)",
        "architecture": "simple",
        "hidden_sizes": 1024,
        "early_stopping_metric": "auc",
        "dropout": 0.0,
    })

    # 3. DeepMLP: 2 layers
    for hidden in [[512, 256], [256, 256], [512, 512]]:
        h_str = "-".join(map(str, hidden))
        all_configs.append({
            "name": f"deep_{h_str}_loss",
            "display": f"DeepMLP ({h_str}, stop=loss)",
            "architecture": "deep",
            "hidden_sizes": hidden,
            "early_stopping_metric": "loss",
            "dropout": 0.0,
        })
        all_configs.append({
            "name": f"deep_{h_str}_auc",
            "display": f"DeepMLP ({h_str}, stop=AUC)",
            "architecture": "deep",
            "hidden_sizes": hidden,
            "early_stopping_metric": "auc",
            "dropout": 0.0,
        })

    # 4. DeepMLP: 3 layers
    for hidden in [[512, 256, 128], [256, 256, 256]]:
        h_str = "-".join(map(str, hidden))
        all_configs.append({
            "name": f"deep_{h_str}_auc",
            "display": f"DeepMLP ({h_str}, stop=AUC)",
            "architecture": "deep",
            "hidden_sizes": hidden,
            "early_stopping_metric": "auc",
            "dropout": 0.0,
        })

    # 5. SwiGLU
    for hidden in [512, 1024]:
        all_configs.append({
            "name": f"swiglu_h{hidden}_loss",
            "display": f"SwiGLU (h={hidden}, stop=loss)",
            "architecture": "swiglu",
            "hidden_sizes": hidden,
            "early_stopping_metric": "loss",
            "dropout": 0.0,
        })
        all_configs.append({
            "name": f"swiglu_h{hidden}_auc",
            "display": f"SwiGLU (h={hidden}, stop=AUC)",
            "architecture": "swiglu",
            "hidden_sizes": hidden,
            "early_stopping_metric": "auc",
            "dropout": 0.0,
        })

    # 6. Higher dropout with best architectures
    for dropout in [0.3, 0.5]:
        all_configs.append({
            "name": f"simple_h1024_drop{dropout}_auc",
            "display": f"SimpleMLP (h=1024, drop={dropout}, stop=AUC)",
            "architecture": "simple",
            "hidden_sizes": 1024,
            "early_stopping_metric": "auc",
            "dropout": dropout,
        })

    # Split configs for parallel execution
    if args.part == 1:
        configs_to_run = all_configs[:len(all_configs)//2]
    elif args.part == 2:
        configs_to_run = all_configs[len(all_configs)//2:]
    else:
        configs_to_run = all_configs

    # Build CVPredictorConfig objects
    for cfg in configs_to_run:
        configs.append(CVPredictorConfig(
            predictor=FlexibleMLPPredictor(
                embedding_source,
                architecture=cfg["architecture"],
                hidden_sizes=cfg["hidden_sizes"],
                dropout=cfg["dropout"],
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                n_epochs=n_epochs,
                init_from_irt=True,
                early_stopping=True,
                early_stopping_metric=cfg["early_stopping_metric"],
                val_fraction=0.1,
                patience=30,
                verbose=True,
            ),
            name=cfg["name"],
            display_name=cfg["display"],
        ))

    if args.part is not None:
        print(f"\n*** Running PART {args.part} only ({len(configs)} configs) ***")

    # Run CV
    results = {}

    print("\n" + "=" * 85)
    print("ARCHITECTURE SWEEP")
    print(f"Fixed: weight_decay={weight_decay}, init_from_irt=True, early_stopping=True")
    print(f"Configs to run: {len(configs)}")
    print("=" * 85)

    for i, pc in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] {pc.display_name}")
        print("-" * 60)

        cv_result = run_cv(
            pc.predictor,
            folds,
            load_fold_data,
            verbose=True,
            diagnostics_extractor=extract_train_auc,
        )

        results[pc.name] = {
            "display_name": pc.display_name,
            "mean_auc": cv_result.mean_auc,
            "std_auc": cv_result.std_auc,
            "fold_aucs": cv_result.fold_aucs,
        }

        if cv_result.fold_diagnostics:
            valid_train_aucs = [t for t in cv_result.fold_diagnostics if t is not None]
            if valid_train_aucs:
                results[pc.name]["train_auc"] = float(np.mean(valid_train_aucs))

        print(f"   Mean AUC: {cv_result.mean_auc:.4f} ± {cv_result.std_auc:.4f}")

    # Print summary
    print("\n" + "=" * 85)
    print("ARCHITECTURE SWEEP RESULTS")
    print("=" * 85)
    print(f"\n{'Method':<45} {'Test AUC':>10} {'Train AUC':>12} {'Gap':>10}")
    print("-" * 85)

    sorted_results = sorted(results.items(), key=lambda x: x[1]["mean_auc"], reverse=True)
    for name, r in sorted_results:
        test_auc = r["mean_auc"]
        train_auc = r.get("train_auc")
        if train_auc is not None:
            gap = train_auc - test_auc
            print(f"{r['display_name']:<45} {test_auc:>10.4f} {train_auc:>12.4f} {gap:>+10.4f}")
        else:
            print(f"{r['display_name']:<45} {test_auc:>10.4f} {'N/A':>12} {'N/A':>10}")

    print(f"\n{'-' * 85}")
    print("KEY COMPARISONS:")
    if "ridge" in results:
        ridge_auc = results["ridge"]["mean_auc"]
        print(f"  Ridge baseline: {ridge_auc:.4f}")
        mlp_results = [(n, r) for n, r in results.items() if n not in ["oracle", "constant", "ridge"]]
        if mlp_results:
            best_name, best_r = max(mlp_results, key=lambda x: x[1]["mean_auc"])
            delta = best_r["mean_auc"] - ridge_auc
            print(f"  Best MLP: {best_r['display_name']}: {best_r['mean_auc']:.4f} ({delta:+.4f} vs Ridge)")
    if "oracle" in results:
        print(f"  Oracle upper bound: {results['oracle']['mean_auc']:.4f}")

    # Save results
    output_dir = ROOT / "chris_output/experiment_a/mlp_embedding"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.part is not None:
        output_path = output_dir / f"architecture_sweep_part{args.part}.json"
    else:
        output_path = output_dir / "architecture_sweep.json"

    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    with open(output_path, "w") as f:
        json.dump(convert_numpy(results), f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

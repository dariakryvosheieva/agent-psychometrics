"""Sanity check: MLP with IRT abilities + Ridge weights should match Ridge exactly.

This script verifies that:
1. Using frozen IRT abilities + Ridge-trained weights produces identical predictions
2. The MLP architecture correctly implements P = sigmoid(θ - β)

If this sanity check fails, there's a bug in the MLP implementation.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from scipy.special import expit as sigmoid

from experiment_ab_shared.feature_source import build_feature_sources
from experiment_ab_shared.dataset import _load_binary_responses
from experiment_a.swebench.config import ExperimentAConfig
from experiment_a.shared.cross_validation import k_fold_split_tasks
from experiment_ab_shared import load_dataset_for_fold

ROOT = Path(__file__).parent.parent


def main():
    print("=" * 70)
    print("SANITY CHECK: MLP vs Ridge")
    print("=" * 70)

    config = ExperimentAConfig()

    # Resolve paths
    llm_judge_path = ROOT / config.llm_judge_features_path
    abilities_path = ROOT / config.abilities_path
    items_path = ROOT / config.items_path
    responses_path = ROOT / config.responses_path

    # Build feature sources (just LLM Judge for simplicity)
    feature_source_list = build_feature_sources(
        llm_judge_path=llm_judge_path,
        verbose=True,
    )
    source = feature_source_list[0][1]  # LLM Judge source

    # Load task IDs and create a single fold
    full_items = pd.read_csv(items_path, index_col=0)
    all_task_ids = list(full_items.index)

    folds = k_fold_split_tasks(all_task_ids, k=5, seed=0)
    train_tasks, test_tasks = folds[0]

    print(f"\nTrain tasks: {len(train_tasks)}, Test tasks: {len(test_tasks)}")

    # Load fold data
    data = load_dataset_for_fold(
        abilities_path=abilities_path,
        items_path=items_path,
        responses_path=responses_path,
        train_tasks=train_tasks,
        test_tasks=test_tasks,
        fold_idx=0,
        k_folds=5,
        split_seed=0,
        is_binomial=False,
        irt_cache_dir=Path("chris_output/experiment_a/irt_splits"),
    )

    print(f"Agents: {len(data.train_abilities)}")

    # =========================================================================
    # STEP 1: Train Ridge regression (standard pipeline)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Train Ridge regression")
    print("=" * 70)

    # Get train features and ground truth difficulties
    train_features = source.get_features(train_tasks)
    ground_truth_b = data.train_items.loc[train_tasks, "b"].values

    # Scale features
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)

    # Train Ridge
    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
    ridge.fit(train_features_scaled, ground_truth_b)

    print(f"Ridge selected alpha: {ridge.alpha_}")
    print(f"Ridge weights shape: {ridge.coef_.shape}")
    print(f"Ridge bias: {ridge.intercept_:.4f}")

    # Predict test difficulties
    test_features = source.get_features(test_tasks)
    test_features_scaled = scaler.transform(test_features)
    ridge_predicted_b = ridge.predict(test_features_scaled)

    print(f"Ridge predicted difficulties (first 5): {ridge_predicted_b[:5]}")

    # =========================================================================
    # STEP 2: Compute Ridge AUC using IRT formula
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Compute Ridge AUC using IRT formula")
    print("=" * 70)

    ridge_predictions = []
    ridge_labels = []

    for task_id, pred_b in zip(test_tasks, ridge_predicted_b):
        for agent_id in data.train_abilities.index:
            if agent_id not in data.responses:
                continue
            if task_id not in data.responses[agent_id]:
                continue

            theta = data.train_abilities.loc[agent_id, "ability"]
            prob = sigmoid(theta - pred_b)
            label = data.responses[agent_id][task_id]

            ridge_predictions.append(prob)
            ridge_labels.append(label)

    ridge_auc = roc_auc_score(ridge_labels, ridge_predictions)
    print(f"Ridge AUC (using IRT formula): {ridge_auc:.4f}")

    # =========================================================================
    # STEP 3: Create MLP with Ridge weights and IRT abilities
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Create MLP with Ridge weights + IRT abilities")
    print("=" * 70)

    # Get all agents
    all_agents = list(data.train_abilities.index)
    agent_to_idx = {agent: i for i, agent in enumerate(all_agents)}
    n_agents = len(all_agents)
    feature_dim = train_features_scaled.shape[1]

    print(f"N agents: {n_agents}, Feature dim: {feature_dim}")

    # Create MLP model
    class IRTStyleMLP(nn.Module):
        def __init__(self, n_agents, feature_dim):
            super().__init__()
            self.agent_abilities = nn.Embedding(n_agents, 1)
            self.difficulty_layer = nn.Linear(feature_dim, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, agent_indices, task_features):
            theta = self.agent_abilities(agent_indices)
            beta = self.difficulty_layer(task_features)
            logit = theta - beta
            return self.sigmoid(logit).squeeze(-1)

    model = IRTStyleMLP(n_agents, feature_dim)

    # Initialize with IRT abilities
    with torch.no_grad():
        for agent_id, idx in agent_to_idx.items():
            ability = data.train_abilities.loc[agent_id, "ability"]
            model.agent_abilities.weight.data[idx, 0] = float(ability)

    # Initialize with Ridge weights
    with torch.no_grad():
        model.difficulty_layer.weight.data = torch.tensor(
            ridge.coef_.reshape(1, -1), dtype=torch.float32
        )
        model.difficulty_layer.bias.data = torch.tensor(
            [ridge.intercept_], dtype=torch.float32
        )

    print(f"MLP difficulty weights: {model.difficulty_layer.weight.data.numpy().flatten()[:5]}")
    print(f"MLP difficulty bias: {model.difficulty_layer.bias.data.item():.4f}")
    print(f"Ridge weights: {ridge.coef_[:5]}")
    print(f"Ridge bias: {ridge.intercept_:.4f}")

    # =========================================================================
    # STEP 4: Compare predictions
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Compare MLP predictions to Ridge")
    print("=" * 70)

    model.eval()
    mlp_predictions = []
    mlp_labels = []

    # Build test task features
    test_features_tensor = torch.tensor(test_features_scaled, dtype=torch.float32)
    task_to_features = {task_id: test_features_tensor[i] for i, task_id in enumerate(test_tasks)}

    with torch.no_grad():
        for task_id in test_tasks:
            task_feat = task_to_features[task_id].unsqueeze(0)

            for agent_id in data.train_abilities.index:
                if agent_id not in data.responses:
                    continue
                if task_id not in data.responses[agent_id]:
                    continue

                agent_idx = agent_to_idx[agent_id]
                agent_tensor = torch.tensor([agent_idx], dtype=torch.long)

                prob = model(agent_tensor, task_feat).item()
                label = data.responses[agent_id][task_id]

                mlp_predictions.append(prob)
                mlp_labels.append(label)

    mlp_auc = roc_auc_score(mlp_labels, mlp_predictions)
    print(f"MLP AUC (with Ridge weights + IRT abilities): {mlp_auc:.4f}")

    # Compare predictions directly
    ridge_preds = np.array(ridge_predictions)
    mlp_preds = np.array(mlp_predictions)

    max_diff = np.max(np.abs(ridge_preds - mlp_preds))
    mean_diff = np.mean(np.abs(ridge_preds - mlp_preds))

    print(f"\nPrediction comparison:")
    print(f"  Max difference: {max_diff:.10f}")
    print(f"  Mean difference: {mean_diff:.10f}")
    print(f"  Ridge AUC: {ridge_auc:.6f}")
    print(f"  MLP AUC:   {mlp_auc:.6f}")

    if max_diff < 1e-5:
        print("\n✓ SANITY CHECK PASSED: MLP matches Ridge exactly!")
    else:
        print("\n✗ SANITY CHECK FAILED: MLP does not match Ridge!")
        print("  Debugging: First 10 predictions comparison:")
        for i in range(min(10, len(ridge_preds))):
            print(f"    Ridge: {ridge_preds[i]:.6f}, MLP: {mlp_preds[i]:.6f}, diff: {abs(ridge_preds[i] - mlp_preds[i]):.6f}")

    # =========================================================================
    # STEP 5: Test if training from scratch with frozen abilities reaches same solution
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: Train MLP from scratch with frozen IRT abilities")
    print("=" * 70)

    # Create a fresh model
    model2 = IRTStyleMLP(n_agents, feature_dim)

    # Initialize abilities from IRT and freeze
    with torch.no_grad():
        for agent_id, idx in agent_to_idx.items():
            ability = data.train_abilities.loc[agent_id, "ability"]
            model2.agent_abilities.weight.data[idx, 0] = float(ability)
    model2.agent_abilities.requires_grad_(False)

    # Build training data
    train_features_tensor = torch.tensor(train_features_scaled, dtype=torch.float32)
    task_to_train_features = {task_id: train_features_tensor[i] for i, task_id in enumerate(train_tasks)}

    agent_indices_list = []
    features_list = []
    y_list = []

    for task_id in train_tasks:
        task_feat = task_to_train_features[task_id].numpy()
        for agent_id in all_agents:
            if agent_id not in data.responses:
                continue
            if task_id not in data.responses[agent_id]:
                continue

            agent_idx = agent_to_idx[agent_id]
            label = data.responses[agent_id][task_id]

            agent_indices_list.append(agent_idx)
            features_list.append(task_feat)
            y_list.append(float(label))

    agent_tensor = torch.tensor(agent_indices_list, dtype=torch.long)
    features_tensor = torch.tensor(np.array(features_list), dtype=torch.float32)
    y_tensor = torch.tensor(y_list, dtype=torch.float32)

    print(f"Training samples: {len(y_list)}")

    # Train with BCE loss
    optimizer = torch.optim.Adam(model2.difficulty_layer.parameters(), lr=0.01, weight_decay=0.01)
    criterion = nn.BCELoss()

    model2.train()
    for epoch in range(500):
        optimizer.zero_grad()
        y_pred = model2(agent_tensor, features_tensor)
        loss = criterion(y_pred, y_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}: Loss = {loss.item():.6f}")

    # Evaluate
    model2.eval()
    mlp2_predictions = []
    mlp2_labels = []

    with torch.no_grad():
        for task_id in test_tasks:
            task_feat = task_to_features[task_id].unsqueeze(0)

            for agent_id in data.train_abilities.index:
                if agent_id not in data.responses:
                    continue
                if task_id not in data.responses[agent_id]:
                    continue

                agent_idx = agent_to_idx[agent_id]
                agent_tensor_single = torch.tensor([agent_idx], dtype=torch.long)

                prob = model2(agent_tensor_single, task_feat).item()
                label = data.responses[agent_id][task_id]

                mlp2_predictions.append(prob)
                mlp2_labels.append(label)

    mlp2_auc = roc_auc_score(mlp2_labels, mlp2_predictions)

    print(f"\nResults:")
    print(f"  Ridge AUC:                    {ridge_auc:.4f}")
    print(f"  MLP (Ridge weights):          {mlp_auc:.4f}")
    print(f"  MLP (trained from scratch):   {mlp2_auc:.4f}")

    # Compare learned weights to Ridge weights
    learned_weights = model2.difficulty_layer.weight.data.numpy().flatten()
    learned_bias = model2.difficulty_layer.bias.data.item()

    print(f"\nWeight comparison:")
    print(f"  Ridge weights: {ridge.coef_}")
    print(f"  Ridge bias: {ridge.intercept_:.4f}")
    print(f"  MLP learned weights: {learned_weights}")
    print(f"  MLP learned bias: {learned_bias:.4f}")

    # Correlation between weights
    weight_corr = np.corrcoef(ridge.coef_, learned_weights)[0, 1]
    print(f"  Weight correlation: {weight_corr:.4f}")


if __name__ == "__main__":
    main()

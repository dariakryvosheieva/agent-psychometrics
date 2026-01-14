"""Train and evaluate LLM judge prior vs combined models.

This script compares:
1. LLM Judge Prior - 9 problem features only (no trajectory data)
2. LLM Judge Combined - 9 problem + 4 trajectory features

Both models directly predict difficulty (not residuals), allowing us to test
whether trajectory data adds signal beyond what's in the problem description.

Usage:
    # Run evaluation (requires pre-computed features)
    python -m experiment_b.llm_judge.train_evaluate_llm_prior

    # Dry run to check feature availability
    python -m experiment_b.llm_judge.train_evaluate_llm_prior --dry_run
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit  # sigmoid
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# Add parent to path for imports
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_b.config import ExperimentConfig
from experiment_b.data_splits import create_experiment_split
from experiment_b.llm_judge.features_prior import (
    LLM_JUDGE_PRIOR_FEATURE_NAMES,
    LLMJudgePriorFeatures,
    load_llm_judge_prior_features_batch,
)
from experiment_b.llm_judge.features_combined import (
    LLM_JUDGE_COMBINED_FEATURE_NAMES,
    PROBLEM_FEATURE_NAMES,
    TRAJECTORY_FEATURE_NAMES,
    LLMJudgeCombinedFeatures,
    load_llm_judge_combined_features_batch,
)


# Directories
PRIOR_FEATURES_DIR = ROOT / "chris_output" / "experiment_b" / "llm_judge_prior_features"
COMBINED_FEATURES_DIR = ROOT / "chris_output" / "experiment_b" / "llm_judge_combined_features"


def load_responses(responses_path: Path) -> Dict[str, Dict[str, int]]:
    """Load response matrix from JSONL file."""
    responses = {}
    with open(responses_path) as f:
        for line in f:
            row = json.loads(line)
            agent_id = row["subject_id"]
            if "responses" in row:
                responses[agent_id] = row["responses"]
            else:
                task_id = row["item_id"]
                response = row["response"]
                if agent_id not in responses:
                    responses[agent_id] = {}
                responses[agent_id][task_id] = response
    return responses


def compute_auc(
    predicted_difficulties: Dict[str, float],
    abilities: pd.DataFrame,
    responses: Dict[str, Dict[str, int]],
    task_ids: List[str],
    agent_ids: List[str],
) -> Dict:
    """Compute AUC for predicted difficulties using IRT formula."""
    y_true = []
    y_scores = []

    for task_id in task_ids:
        if task_id not in predicted_difficulties:
            continue
        beta_pred = predicted_difficulties[task_id]

        for agent_id in agent_ids:
            if agent_id not in responses:
                continue
            if task_id not in responses[agent_id]:
                continue
            if agent_id not in abilities.index:
                continue

            theta = float(abilities.loc[agent_id, "theta"])
            actual = responses[agent_id][task_id]
            prob = float(expit(theta - beta_pred))

            y_true.append(int(actual))
            y_scores.append(prob)

    if len(y_true) < 2 or len(set(y_true)) < 2:
        return {"error": "Insufficient data", "n_pairs": len(y_true)}

    auc = roc_auc_score(y_true, y_scores)
    return {
        "auc": float(auc),
        "n_pairs": len(y_true),
        "n_positive": sum(y_true),
        "n_negative": len(y_true) - sum(y_true),
    }


class LLMJudgePriorModel:
    """Ridge regression model using only problem features."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)
        self.scaler = StandardScaler()
        self.feature_names = LLM_JUDGE_PRIOR_FEATURE_NAMES
        self.is_fitted = False

    def fit(
        self,
        task_ids: List[str],
        ground_truth: np.ndarray,
        features_dir: Path = PRIOR_FEATURES_DIR,
    ) -> Dict:
        """Fit the model on training data.

        Args:
            task_ids: Task IDs to train on
            ground_truth: Ground truth difficulty values
            features_dir: Directory containing feature JSON files

        Returns:
            Training stats
        """
        features_dict = load_llm_judge_prior_features_batch(task_ids, features_dir)

        X = []
        y = []
        used_tasks = []

        for i, task_id in enumerate(task_ids):
            if task_id not in features_dict:
                continue
            X.append(features_dict[task_id].to_vector())
            y.append(ground_truth[i])
            used_tasks.append(task_id)

        if len(X) < 5:
            return {"error": "Insufficient data", "n_tasks": len(X)}

        X = np.array(X)
        y = np.array(y)

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True

        return {
            "n_tasks": len(used_tasks),
            "n_features": X.shape[1],
            "r2_train": float(self.model.score(X_scaled, y)),
        }

    def predict(
        self,
        task_ids: List[str],
        features_dir: Path = PRIOR_FEATURES_DIR,
    ) -> Dict[str, float]:
        """Predict difficulties for tasks.

        Args:
            task_ids: Task IDs to predict
            features_dir: Directory containing feature JSON files

        Returns:
            Dict mapping task_id -> predicted difficulty
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        features_dict = load_llm_judge_prior_features_batch(task_ids, features_dir)
        predictions = {}

        for task_id in task_ids:
            if task_id not in features_dict:
                continue
            X = features_dict[task_id].to_vector().reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            predictions[task_id] = float(self.model.predict(X_scaled)[0])

        return predictions

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature coefficients."""
        if not self.is_fitted:
            return {}
        return dict(zip(self.feature_names, self.model.coef_.tolist()))


class LLMJudgeCombinedModel:
    """Ridge regression model using problem + trajectory features."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)
        self.scaler = StandardScaler()
        self.feature_names = LLM_JUDGE_COMBINED_FEATURE_NAMES
        self.is_fitted = False

    def fit(
        self,
        task_ids: List[str],
        ground_truth: np.ndarray,
        features_dir: Path = COMBINED_FEATURES_DIR,
    ) -> Dict:
        """Fit the model on training data."""
        features_dict = load_llm_judge_combined_features_batch(task_ids, features_dir)

        X = []
        y = []
        used_tasks = []

        for i, task_id in enumerate(task_ids):
            if task_id not in features_dict:
                continue
            X.append(features_dict[task_id].to_vector())
            y.append(ground_truth[i])
            used_tasks.append(task_id)

        if len(X) < 5:
            return {"error": "Insufficient data", "n_tasks": len(X)}

        X = np.array(X)
        y = np.array(y)

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True

        return {
            "n_tasks": len(used_tasks),
            "n_features": X.shape[1],
            "r2_train": float(self.model.score(X_scaled, y)),
        }

    def predict(
        self,
        task_ids: List[str],
        features_dir: Path = COMBINED_FEATURES_DIR,
    ) -> Dict[str, float]:
        """Predict difficulties for tasks."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        features_dict = load_llm_judge_combined_features_batch(task_ids, features_dir)
        predictions = {}

        for task_id in task_ids:
            if task_id not in features_dict:
                continue
            X = features_dict[task_id].to_vector().reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            predictions[task_id] = float(self.model.predict(X_scaled)[0])

        return predictions

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature coefficients."""
        if not self.is_fitted:
            return {}
        return dict(zip(self.feature_names, self.model.coef_.tolist()))


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate LLM judge models")
    parser.add_argument("--dry_run", action="store_true",
                        help="Check feature availability without training")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Ridge regression alpha (default: 1.0)")
    args = parser.parse_args()

    print("=" * 70)
    print("LLM JUDGE PRIOR vs COMBINED EVALUATION")
    print("=" * 70)

    # Load config and create splits
    config = ExperimentConfig()
    split = create_experiment_split(
        responses_path=ROOT / config.responses_path,
        trajectories_dir=ROOT / config.trajectories_dir,
        weak_threshold=config.weak_threshold,
        strong_min_improvement=config.strong_min_improvement,
        m1_fraction=config.m1_fraction,
        m2_fraction=config.m2_fraction,
    )

    print(f"\nData splits:")
    print(f"  D_train tasks: {len(split.d_train_tasks)}")
    print(f"  D_valid tasks: {len(split.d_valid_tasks)}")
    print(f"  M1 agents (train): {len(split.m1_agents)}")
    print(f"  M2 agents (valid): {len(split.m2_agents)}")

    # Check feature availability
    all_tasks = list(set(split.d_train_tasks) | set(split.d_valid_tasks))

    print(f"\nChecking feature availability...")

    # Prior features
    prior_features = load_llm_judge_prior_features_batch(all_tasks, PRIOR_FEATURES_DIR)
    prior_train = len([t for t in split.d_train_tasks if t in prior_features])
    prior_valid = len([t for t in split.d_valid_tasks if t in prior_features])
    print(f"  Prior features: {len(prior_features)}/{len(all_tasks)} total")
    print(f"    D_train: {prior_train}/{len(split.d_train_tasks)}")
    print(f"    D_valid: {prior_valid}/{len(split.d_valid_tasks)}")

    # Combined features
    combined_features = load_llm_judge_combined_features_batch(all_tasks, COMBINED_FEATURES_DIR)
    combined_train = len([t for t in split.d_train_tasks if t in combined_features])
    combined_valid = len([t for t in split.d_valid_tasks if t in combined_features])
    print(f"  Combined features: {len(combined_features)}/{len(all_tasks)} total")
    print(f"    D_train: {combined_train}/{len(split.d_train_tasks)}")
    print(f"    D_valid: {combined_valid}/{len(split.d_valid_tasks)}")

    if args.dry_run:
        print("\n=== DRY RUN COMPLETE ===")
        print(f"\nTo compute missing features, run:")
        if prior_train < len(split.d_train_tasks):
            print(f"  python -m experiment_b.llm_judge.compute_features_prior")
        if combined_train < len(split.d_train_tasks):
            print(f"  python -m experiment_b.llm_judge.compute_features_combined")
        return

    # Load ground truth difficulties
    items_path = ROOT / config.items_path
    items_df = pd.read_csv(items_path, index_col=0)
    print(f"\nLoaded ground truth for {len(items_df)} tasks")

    # Load abilities and responses for AUC
    abilities_path = items_path.parent / "abilities.csv"
    abilities_df = pd.read_csv(abilities_path, index_col=0)
    responses = load_responses(ROOT / config.responses_path)

    # Prepare training data
    train_task_ids = split.d_train_tasks
    train_gt = items_df.loc[train_task_ids, "b"].values

    valid_task_ids = split.d_valid_tasks
    valid_gt = items_df.loc[valid_task_ids, "b"]

    results = {}

    # Train and evaluate Prior model
    print("\n" + "=" * 70)
    print("MODEL 1: LLM JUDGE PRIOR (9 problem features)")
    print("=" * 70)

    prior_model = LLMJudgePriorModel(alpha=args.alpha)
    train_stats = prior_model.fit(train_task_ids, train_gt, PRIOR_FEATURES_DIR)
    print(f"Training: {train_stats}")

    if "error" not in train_stats:
        # Predict on train
        prior_train_preds = prior_model.predict(train_task_ids, PRIOR_FEATURES_DIR)
        prior_train_auc = compute_auc(prior_train_preds, abilities_df, responses, train_task_ids, split.m1_agents)
        print(f"D_train AUC: {prior_train_auc.get('auc', 'N/A'):.4f}")

        # Predict on valid
        prior_valid_preds = prior_model.predict(valid_task_ids, PRIOR_FEATURES_DIR)
        prior_valid_auc = compute_auc(prior_valid_preds, abilities_df, responses, valid_task_ids, split.m2_agents)
        print(f"D_valid AUC: {prior_valid_auc.get('auc', 'N/A'):.4f}")

        results["prior"] = {
            "train_auc": prior_train_auc,
            "valid_auc": prior_valid_auc,
            "feature_importance": prior_model.get_feature_importance(),
        }

        print("\nFeature importance (top 5):")
        importance = prior_model.get_feature_importance()
        sorted_imp = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
        for name, coef in sorted_imp[:5]:
            print(f"  {name}: {coef:.4f}")

    # Train and evaluate Combined model
    print("\n" + "=" * 70)
    print("MODEL 2: LLM JUDGE COMBINED (9 problem + 4 trajectory)")
    print("=" * 70)

    combined_model = LLMJudgeCombinedModel(alpha=args.alpha)
    train_stats = combined_model.fit(train_task_ids, train_gt, COMBINED_FEATURES_DIR)
    print(f"Training: {train_stats}")

    if "error" not in train_stats:
        # Predict on train
        combined_train_preds = combined_model.predict(train_task_ids, COMBINED_FEATURES_DIR)
        combined_train_auc = compute_auc(combined_train_preds, abilities_df, responses, train_task_ids, split.m1_agents)
        print(f"D_train AUC: {combined_train_auc.get('auc', 'N/A'):.4f}")

        # Predict on valid
        combined_valid_preds = combined_model.predict(valid_task_ids, COMBINED_FEATURES_DIR)
        combined_valid_auc = compute_auc(combined_valid_preds, abilities_df, responses, valid_task_ids, split.m2_agents)
        print(f"D_valid AUC: {combined_valid_auc.get('auc', 'N/A'):.4f}")

        results["combined"] = {
            "train_auc": combined_train_auc,
            "valid_auc": combined_valid_auc,
            "feature_importance": combined_model.get_feature_importance(),
        }

        print("\nFeature importance (top 5):")
        importance = combined_model.get_feature_importance()
        sorted_imp = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
        for name, coef in sorted_imp[:5]:
            print(f"  {name}: {coef:.4f}")

        # Show trajectory-specific features
        print("\nTrajectory features only:")
        for name in TRAJECTORY_FEATURE_NAMES:
            if name in importance:
                print(f"  {name}: {importance[name]:.4f}")

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)

    print("\n| Model | D_train AUC | D_valid AUC | Delta |")
    print("|-------|-------------|-------------|-------|")

    prior_valid = results.get("prior", {}).get("valid_auc", {}).get("auc", None)
    combined_valid = results.get("combined", {}).get("valid_auc", {}).get("auc", None)

    if prior_valid:
        print(f"| Prior (9 features) | "
              f"{results['prior']['train_auc'].get('auc', 0):.4f} | "
              f"{prior_valid:.4f} | — |")

    if combined_valid:
        delta = combined_valid - prior_valid if prior_valid else 0
        delta_str = f"{delta:+.4f}" if prior_valid else "—"
        print(f"| Combined (13 features) | "
              f"{results['combined']['train_auc'].get('auc', 0):.4f} | "
              f"{combined_valid:.4f} | {delta_str} |")

    # Reference: embedding prior baseline
    print(f"| Embedding Prior (ref) | 0.6830 | 0.7383 | — |")

    if prior_valid and combined_valid:
        if combined_valid > prior_valid:
            print(f"\n=> Combined model improves over Prior by {combined_valid - prior_valid:.4f} AUC")
            print("   Trajectory data DOES contain additional signal!")
        else:
            print(f"\n=> Combined model does NOT improve over Prior")
            print("   Trajectory data adds NO signal beyond problem features.")

    # Save results
    output_path = ROOT / "chris_output" / "experiment_b" / "llm_judge_prior_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

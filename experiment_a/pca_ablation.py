"""PCA ablation study for embedding-based prior difficulty prediction.

Tests the effect of PCA dimensionality reduction combined with varying
ridge alpha values on the embedding prior model.

Usage:
    python -m experiment_a.pca_ablation \
        --embeddings_path out/prior_qwen3vl8b/embeddings__*.npz \
        --output_dir chris_output/experiment_a/pca_ablation
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy import stats as scipy_stats

# Add parent to path for imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_a.data_loader import load_experiment_data
from experiment_a.irt_evaluation import compute_auc, compute_difficulty_prediction_metrics


class PCAEmbeddingPredictor:
    """Embedding predictor with optional PCA dimensionality reduction."""

    def __init__(
        self,
        embeddings_path: Path,
        ridge_alpha: Union[float, str] = "cv",
        pca_components: Optional[int] = None,
    ):
        """Initialize PCA embedding predictor.

        Args:
            embeddings_path: Path to pre-computed embeddings .npz file
            ridge_alpha: Ridge regression alpha. Use "cv" for cross-validation.
            pca_components: Number of PCA components. None = no PCA.
        """
        self.embeddings_path = embeddings_path
        self.ridge_alpha = ridge_alpha
        self.pca_components = pca_components

        self._model: Optional[Pipeline] = None
        self._embeddings: Optional[Dict[str, np.ndarray]] = None
        self._embedding_dim: Optional[int] = None
        self._actual_pca_dim: Optional[int] = None
        self._best_alpha: Optional[float] = None

        # Load embeddings immediately
        self._load_embeddings()

    def _load_embeddings(self) -> None:
        """Load embeddings from .npz file."""
        data = np.load(self.embeddings_path, allow_pickle=True)

        # Extract task IDs and embedding matrix
        task_ids = [str(x) for x in data["task_ids"].tolist()]
        X = data["X"].astype(np.float32)

        self._embedding_dim = int(X.shape[1])
        self._embeddings = {task_id: X[i] for i, task_id in enumerate(task_ids)}

    def fit(self, task_ids: List[str], ground_truth_b: np.ndarray) -> "PCAEmbeddingPredictor":
        """Fit Ridge regression on task embeddings with optional PCA.

        Args:
            task_ids: List of training task identifiers
            ground_truth_b: Array of ground truth difficulty values

        Returns:
            self
        """
        if self._embeddings is None:
            raise RuntimeError("Embeddings not loaded")

        # Get embeddings for training tasks
        available_tasks = [t for t in task_ids if t in self._embeddings]
        if len(available_tasks) < len(task_ids):
            missing = len(task_ids) - len(available_tasks)
            print(f"Warning: {missing} tasks missing from embeddings")

        # Build training matrix
        X = np.stack([self._embeddings[t] for t in available_tasks])
        y = np.array([ground_truth_b[task_ids.index(t)] for t in available_tasks])

        n_samples, n_features = X.shape

        # Build pipeline
        pipeline_steps = [("scaler", StandardScaler(with_mean=True, with_std=True))]

        # Add PCA if requested
        if self.pca_components is not None:
            # Can't have more components than samples - 1 or features
            actual_pca_components = min(self.pca_components, n_samples - 1, n_features)
            self._actual_pca_dim = actual_pca_components
            pipeline_steps.append(("pca", PCA(n_components=actual_pca_components)))

        # Add Ridge regression
        if self.ridge_alpha == "cv":
            alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000, 100000]
            ridge = RidgeCV(alphas=alphas, cv=5, scoring="neg_mean_squared_error")
            pipeline_steps.append(("ridge", ridge))
        else:
            ridge = Ridge(alpha=float(self.ridge_alpha))
            pipeline_steps.append(("ridge", ridge))

        self._model = Pipeline(pipeline_steps)
        self._model.fit(X, y)

        # Store best alpha
        if self.ridge_alpha == "cv":
            self._best_alpha = self._model.named_steps["ridge"].alpha_
        else:
            self._best_alpha = float(self.ridge_alpha)

        return self

    def predict(self, task_ids: List[str]) -> Dict[str, float]:
        """Predict difficulty for tasks.

        Args:
            task_ids: List of task identifiers

        Returns:
            Dict mapping task_id to predicted difficulty
        """
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if self._embeddings is None:
            raise RuntimeError("Embeddings not loaded")

        # Get embeddings for prediction tasks
        available_tasks = [t for t in task_ids if t in self._embeddings]

        if not available_tasks:
            return {}

        X = np.stack([self._embeddings[t] for t in available_tasks])
        preds = self._model.predict(X)

        return dict(zip(available_tasks, preds.tolist()))

    @property
    def embedding_dim(self) -> Optional[int]:
        """Return the original embedding dimensionality."""
        return self._embedding_dim

    @property
    def actual_pca_dim(self) -> Optional[int]:
        """Return the actual PCA dimension used (after clamping)."""
        return self._actual_pca_dim

    @property
    def best_alpha(self) -> Optional[float]:
        """Return the selected ridge alpha."""
        return self._best_alpha

    @property
    def n_embeddings(self) -> int:
        """Return number of loaded embeddings."""
        return len(self._embeddings) if self._embeddings else 0

    def get_pca_explained_variance(self) -> Optional[np.ndarray]:
        """Return cumulative explained variance ratio from PCA."""
        if self._model is None or "pca" not in self._model.named_steps:
            return None
        pca = self._model.named_steps["pca"]
        return np.cumsum(pca.explained_variance_ratio_)


def run_pca_ablation(
    embeddings_path: Path,
    output_dir: Path,
    pca_components_list: List[Optional[int]],
    ridge_alphas: List[Union[float, str]],
    test_fraction: float = 0.2,
    split_seed: int = 0,
) -> Dict[str, Any]:
    """Run PCA ablation study.

    Args:
        embeddings_path: Path to embeddings .npz file
        output_dir: Directory to save results
        pca_components_list: List of PCA component counts to try (None = no PCA)
        ridge_alphas: List of ridge alphas to try ("cv" for cross-validation)
        test_fraction: Fraction of tasks for test set
        split_seed: Random seed for split

    Returns:
        Dict with all results
    """
    print("=" * 60)
    print("PCA ABLATION STUDY FOR EMBEDDING PRIOR")
    print("=" * 60)

    # Load data
    print("\n1. Loading data...")

    abilities_path = ROOT / "clean_data/swebench_verified_20251120_full/1d/abilities.csv"
    items_path = ROOT / "clean_data/swebench_verified_20251120_full/1d/items.csv"
    responses_path = ROOT / "clean_data/swebench_verified/swebench_verified_20251120_full.jsonl"

    data = load_experiment_data(
        abilities_path=abilities_path,
        items_path=items_path,
        responses_path=responses_path,
        test_fraction=test_fraction,
        split_seed=split_seed,
    )

    print(f"   Agents: {data.n_agents}")
    print(f"   Tasks: {data.n_tasks}")
    print(f"   Train tasks: {data.n_train_tasks}")
    print(f"   Test tasks: {data.n_test_tasks}")

    train_b = data.items.loc[data.train_tasks, "b"].values

    # Load embeddings info
    emb_data = np.load(embeddings_path, allow_pickle=True)
    original_dim = emb_data["X"].shape[1]
    print(f"\n2. Embeddings: {original_dim} dimensions, {len(emb_data['task_ids'])} tasks")

    # Run ablation
    print("\n3. Running ablation...")
    results = {
        "config": {
            "embeddings_path": str(embeddings_path),
            "test_fraction": test_fraction,
            "split_seed": split_seed,
            "n_train_tasks": data.n_train_tasks,
            "n_test_tasks": data.n_test_tasks,
            "original_embedding_dim": original_dim,
        },
        "ablation_results": [],
    }

    best_auc = 0.0
    best_config = None

    for pca_components in pca_components_list:
        for ridge_alpha in ridge_alphas:
            pca_label = f"PCA-{pca_components}" if pca_components else "No-PCA"
            alpha_label = f"alpha={ridge_alpha}"
            print(f"\n   {pca_label}, {alpha_label}...")

            try:
                predictor = PCAEmbeddingPredictor(
                    embeddings_path=embeddings_path,
                    ridge_alpha=ridge_alpha,
                    pca_components=pca_components,
                )
                predictor.fit(data.train_tasks, train_b)
                preds = predictor.predict(data.test_tasks)

                # Compute AUC
                auc_result = compute_auc(
                    preds, data.abilities, data.responses, data.test_tasks
                )

                # Compute difficulty prediction metrics
                diff_metrics = compute_difficulty_prediction_metrics(
                    preds, data.items, data.test_tasks
                )

                # Get explained variance if PCA was used
                explained_var = predictor.get_pca_explained_variance()
                explained_var_total = float(explained_var[-1]) if explained_var is not None else None

                result = {
                    "pca_components_requested": pca_components,
                    "pca_components_actual": predictor.actual_pca_dim,
                    "ridge_alpha_requested": ridge_alpha,
                    "ridge_alpha_selected": predictor.best_alpha,
                    "auc": auc_result.get("auc"),
                    "pearson_r": diff_metrics.get("pearson_r"),
                    "mse": diff_metrics.get("mse"),
                    "rmse": diff_metrics.get("rmse"),
                    "explained_variance_ratio": explained_var_total,
                }

                auc = result["auc"]
                if auc is not None:
                    print(f"      AUC: {auc:.4f}, r: {result['pearson_r']:.4f}")

                    if auc > best_auc:
                        best_auc = auc
                        best_config = result

                results["ablation_results"].append(result)

            except Exception as e:
                print(f"      Error: {e}")
                results["ablation_results"].append({
                    "pca_components_requested": pca_components,
                    "ridge_alpha_requested": ridge_alpha,
                    "error": str(e),
                })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\n{'PCA Dims':<12} {'Alpha':<12} {'AUC':>10} {'Pearson r':>12} {'Exp Var':>10}")
    print("-" * 60)

    for r in results["ablation_results"]:
        if "error" in r:
            continue
        pca_str = str(r["pca_components_actual"]) if r["pca_components_actual"] else "None"
        alpha_str = f"{r['ridge_alpha_selected']:.0e}" if r["ridge_alpha_selected"] else "cv"
        auc_str = f"{r['auc']:.4f}" if r.get("auc") else "N/A"
        r_str = f"{r['pearson_r']:.4f}" if r.get("pearson_r") else "N/A"
        var_str = f"{r['explained_variance_ratio']:.2%}" if r.get("explained_variance_ratio") else "N/A"
        print(f"{pca_str:<12} {alpha_str:<12} {auc_str:>10} {r_str:>12} {var_str:>10}")

    if best_config:
        print(f"\nBest config:")
        print(f"  PCA components: {best_config['pca_components_actual']}")
        print(f"  Ridge alpha: {best_config['ridge_alpha_selected']}")
        print(f"  AUC: {best_config['auc']:.4f}")
        print(f"  Pearson r: {best_config['pearson_r']:.4f}")

        results["best_config"] = best_config

    return results


def main():
    parser = argparse.ArgumentParser(
        description="PCA ablation study for embedding prior"
    )
    parser.add_argument(
        "--embeddings_path",
        type=str,
        default="out/prior_qwen3vl8b/embeddings__Qwen__Qwen3-VL-8B-Instruct__pool-lasttoken__qs-sol-instr__qs_sol_instr_b7008f2d__idnorm_instance-v1__princeton-nlp_SWE-bench_Verified__test__n500__maxlen8192__seed0.npz",
        help="Path to embeddings .npz file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="chris_output/experiment_a/pca_ablation",
        help="Output directory",
    )
    parser.add_argument(
        "--test_fraction",
        type=float,
        default=0.2,
        help="Fraction of tasks for test set",
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=0,
        help="Random seed for train/test split",
    )
    args = parser.parse_args()

    embeddings_path = ROOT / args.embeddings_path
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define ablation grid
    # PCA components: None (no PCA), 25, 50, 100, 200, 400
    pca_components_list = [None, 25, 50, 100, 200, 400]

    # Ridge alphas: cross-validation and fixed values
    ridge_alphas = ["cv", 1, 10, 100, 1000, 10000, 100000]

    results = run_pca_ablation(
        embeddings_path=embeddings_path,
        output_dir=output_dir,
        pca_components_list=pca_components_list,
        ridge_alphas=ridge_alphas,
        test_fraction=args.test_fraction,
        split_seed=args.split_seed,
    )

    # Save results
    output_path = output_dir / "pca_ablation_results.json"

    def convert_numpy(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    results = convert_numpy(results)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

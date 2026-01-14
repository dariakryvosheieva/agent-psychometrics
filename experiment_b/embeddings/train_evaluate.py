"""
Training and evaluation pipeline for embedding-based posterior difficulty prediction.

This script evaluates trajectory embeddings for predicting IRT difficulty residuals.
It supports multiple ablations over content types, instructions, and aggregation strategies.

Usage:
    python -m experiment_b.embeddings.train_evaluate \
        --embeddings_dir chris_output/experiment_b/trajectory_embeddings/full_difficulty \
        --aggregation mean_std \
        --alpha cv
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit
from sklearn.metrics import roc_auc_score

# Add parent to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_b.config import ExperimentConfig, RegressionMode
from experiment_b.data_splits import create_experiment_split
from experiment_b.prior_model import PriorModel, EmbeddingPriorModel
from experiment_b.embeddings.aggregator import AggregationType
from experiment_b.embeddings.posterior_model import EmbeddingPosteriorModel


@dataclass
class EmbeddingExperimentConfig:
    """Configuration for embedding-based experiment."""
    # Data paths
    items_path: Path = Path("clean_data/swebench_verified_20251120_full/1d/items.csv")
    responses_path: Path = Path("clean_data/swebench_verified/swebench_verified_20251120_full.jsonl")
    trajectories_dir: Path = Path("trajectory_data/unified_trajs")
    embeddings_dir: Path = Path("chris_output/experiment_b/trajectory_embeddings/full_difficulty")
    output_dir: Path = Path("chris_output/experiment_b/embedding_results")

    # Prior model
    prior_source: str = "embedding"  # "embedding" or "heuristic"
    prior_embeddings_path: Optional[Path] = Path("out/prior_qwen3vl8b/embeddings__Qwen__Qwen3-VL-8B-Instruct__pool-lasttoken__qs-sol-instr__qs_sol_instr_b7008f2d__idnorm_instance-v1__princeton-nlp_SWE-bench_Verified__test__n500__maxlen8192__seed0.npz")
    prior_alpha: float = 10000.0

    # Posterior model
    aggregation: AggregationType = "mean_std"
    posterior_alpha: Union[float, str] = "cv"
    regression_mode: RegressionMode = "residual"

    # Data splits
    m1_fraction: float = 0.4
    m2_fraction: float = 0.4
    weak_threshold: float = 0.2
    strong_min_improvement: float = 0.1

    # Ablation metadata (populated from embeddings or manually set)
    backbone: Optional[str] = None
    content_type: Optional[str] = None
    instruction_type: Optional[str] = None
    max_length: Optional[int] = None
    embedding_dim: Optional[int] = None

    def to_dict(self) -> Dict:
        """Return full config dict for reproducibility and post-hoc analysis."""
        return {
            # Paths
            "items_path": str(self.items_path),
            "responses_path": str(self.responses_path),
            "trajectories_dir": str(self.trajectories_dir),
            "embeddings_dir": str(self.embeddings_dir),
            "output_dir": str(self.output_dir),
            # Prior model
            "prior_source": self.prior_source,
            "prior_embeddings_path": str(self.prior_embeddings_path) if self.prior_embeddings_path else None,
            "prior_alpha": self.prior_alpha,
            # Posterior model
            "aggregation": self.aggregation,
            "posterior_alpha": str(self.posterior_alpha),
            "regression_mode": self.regression_mode,
            # Data splits
            "m1_fraction": self.m1_fraction,
            "m2_fraction": self.m2_fraction,
            "weak_threshold": self.weak_threshold,
            "strong_min_improvement": self.strong_min_improvement,
            # Ablation metadata (key for identifying this run)
            "backbone": self.backbone,
            "content_type": self.content_type,
            "instruction_type": self.instruction_type,
            "max_length": self.max_length,
            "embedding_dim": self.embedding_dim,
        }

    def get_run_id(self) -> str:
        """Generate a unique run ID for this configuration."""
        parts = []
        if self.backbone:
            # Shorten backbone name for filename
            backbone_short = self.backbone.replace("/", "_").replace("-", "_")
            parts.append(backbone_short)
        if self.content_type:
            parts.append(self.content_type)
        if self.instruction_type:
            parts.append(self.instruction_type)
        parts.append(self.aggregation)
        parts.append(self.regression_mode)
        parts.append(f"alpha_{self.posterior_alpha}")
        return "__".join(parts) if parts else "default"


def extract_embedding_metadata(embeddings_dir: Path) -> Dict:
    """Extract metadata from embedding files to identify the ablation configuration.

    Looks at a sample embedding file to get backbone, content_type, instruction_type, etc.

    Returns:
        Dict with metadata fields (backbone, content_type, instruction_type, max_length, embedding_dim)
    """
    metadata = {}

    # Find a sample embedding file
    sample_file = None
    for agent_dir in embeddings_dir.iterdir():
        if agent_dir.is_dir():
            for npz_file in agent_dir.glob("*.npz"):
                sample_file = npz_file
                break
        if sample_file:
            break

    if sample_file is None:
        print(f"Warning: No embedding files found in {embeddings_dir}")
        return metadata

    try:
        data = np.load(sample_file, allow_pickle=True)

        # Extract all available metadata
        for key in ["backbone", "content_type", "instruction_type", "max_length", "embedding_dim", "embedding_layer"]:
            if key in data:
                val = data[key]
                # Handle numpy arrays/scalars
                if hasattr(val, "item"):
                    metadata[key] = val.item()
                elif hasattr(val, "tolist"):
                    val_list = val.tolist()
                    metadata[key] = val_list[0] if isinstance(val_list, list) and len(val_list) == 1 else val_list
                else:
                    metadata[key] = val

        # Also get embedding dimension from the actual embedding
        if "embedding" in data and "embedding_dim" not in metadata:
            metadata["embedding_dim"] = data["embedding"].shape[0]

        print(f"Extracted metadata from {sample_file.name}:")
        for k, v in metadata.items():
            print(f"  {k}: {v}")

    except Exception as e:
        print(f"Warning: Could not extract metadata from {sample_file}: {e}")

    return metadata


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
    """Compute AUC using IRT formula P(success) = sigmoid(theta - beta)."""
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


def evaluate_predictions(predictions: Dict[str, float], ground_truth: pd.Series) -> Dict:
    """Evaluate prediction quality with Pearson r and MSE."""
    common_tasks = set(predictions.keys()) & set(ground_truth.index)
    if len(common_tasks) < 3:
        return {"error": "Too few common tasks", "n": len(common_tasks)}

    pred_arr = np.array([predictions[t] for t in common_tasks])
    gt_arr = np.array([ground_truth[t] for t in common_tasks])

    r, p = stats.pearsonr(pred_arr, gt_arr)
    mse = np.mean((pred_arr - gt_arr) ** 2)

    return {
        "n": len(common_tasks),
        "pearson_r": float(r),
        "p_value": float(p),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
    }


def run_embedding_experiment(config: EmbeddingExperimentConfig) -> Dict:
    """Run embedding-based posterior experiment.

    Args:
        config: Experiment configuration

    Returns:
        Dict with all results
    """
    print("=" * 60)
    print("EXPERIMENT B: EMBEDDING-BASED POSTERIOR")
    print("=" * 60)
    print(f"Embeddings dir: {config.embeddings_dir}")
    print(f"Aggregation: {config.aggregation}")
    print(f"Alpha: {config.posterior_alpha}")

    # Resolve paths
    items_path = ROOT / config.items_path
    responses_path = ROOT / config.responses_path
    trajectories_dir = ROOT / config.trajectories_dir
    embeddings_dir = ROOT / config.embeddings_dir
    output_dir = ROOT / config.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract metadata from embedding files if not already set
    if config.backbone is None or config.content_type is None:
        print("\n0. Extracting embedding metadata...")
        metadata = extract_embedding_metadata(embeddings_dir)
        if config.backbone is None:
            config.backbone = metadata.get("backbone")
        if config.content_type is None:
            config.content_type = metadata.get("content_type")
        if config.instruction_type is None:
            config.instruction_type = metadata.get("instruction_type")
        if config.max_length is None:
            config.max_length = metadata.get("max_length")
        if config.embedding_dim is None:
            config.embedding_dim = metadata.get("embedding_dim")

    # Load IRT parameters
    print("\n1. Loading IRT parameters...")
    items_df = pd.read_csv(items_path, index_col=0)
    print(f"   Loaded {len(items_df)} tasks")

    abilities_path = items_path.parent / "abilities.csv"
    abilities_df = pd.read_csv(abilities_path, index_col=0)
    print(f"   Loaded {len(abilities_df)} agent abilities")

    responses = load_responses(responses_path)
    print(f"   Loaded responses for {len(responses)} agents")

    # Create data splits
    print("\n2. Creating agent/task splits...")
    split = create_experiment_split(
        responses_path=responses_path,
        trajectories_dir=trajectories_dir,
        weak_threshold=config.weak_threshold,
        strong_min_improvement=config.strong_min_improvement,
        m1_fraction=config.m1_fraction,
        m2_fraction=config.m2_fraction,
    )
    print(f"   M1 agents: {len(split.m1_agents)}")
    print(f"   M2 agents: {len(split.m2_agents)}")
    print(f"   D_train tasks: {len(split.d_train_tasks)}")
    print(f"   D_valid tasks: {len(split.d_valid_tasks)}")

    if len(split.d_train_tasks) == 0:
        return {"error": "No training tasks"}

    # Train prior model
    print("\n3. Training prior model...")
    all_task_ids = list(items_df.index)
    all_difficulties = items_df["b"].values

    if config.prior_source == "embedding" and config.prior_embeddings_path:
        prior_embeddings_path = ROOT / config.prior_embeddings_path
        prior_model = EmbeddingPriorModel(prior_embeddings_path, alpha=config.prior_alpha)
    else:
        prior_model = PriorModel(alpha=config.prior_alpha)
    prior_model.fit(all_task_ids, all_difficulties)

    # Get agent abilities for weighted aggregation
    abilities_dict = {
        agent: float(abilities_df.loc[agent, "theta"])
        for agent in abilities_df.index
    }

    # Train embedding posterior model
    print("\n4. Training embedding posterior model...")
    train_difficulties = items_df.loc[split.d_train_tasks, "b"].values

    posterior_model = EmbeddingPosteriorModel(
        prior_model=prior_model,
        aggregation=config.aggregation,
        alpha=config.posterior_alpha,
        abilities=abilities_dict if config.aggregation == "weighted" else None,
        regression_mode=config.regression_mode,
    )

    posterior_model.fit(
        task_ids=split.d_train_tasks,
        ground_truth_difficulties=train_difficulties,
        weak_agents=split.m1_agents,
        embeddings_dir=embeddings_dir,
    )

    # Evaluate prior on D_train
    print("\n5. Evaluating on D_train...")
    prior_train_preds = prior_model.get_prior_predictions(split.d_train_tasks)
    train_gt = items_df.loc[split.d_train_tasks, "b"]
    prior_train_eval = evaluate_predictions(prior_train_preds, train_gt)
    prior_train_auc = compute_auc(prior_train_preds, abilities_df, responses, split.d_train_tasks, split.m1_agents)
    print(f"   Prior AUC: {prior_train_auc.get('auc', 'N/A'):.4f}" if 'auc' in prior_train_auc else f"   Prior AUC: {prior_train_auc}")

    # Evaluate posterior on D_train
    posterior_train_preds = posterior_model.predict(split.d_train_tasks, split.m1_agents, embeddings_dir)
    posterior_train_eval = evaluate_predictions(posterior_train_preds, train_gt)
    posterior_train_auc = compute_auc(posterior_train_preds, abilities_df, responses, split.d_train_tasks, split.m1_agents)
    print(f"   Posterior AUC: {posterior_train_auc.get('auc', 'N/A'):.4f}" if 'auc' in posterior_train_auc else f"   Posterior AUC: {posterior_train_auc}")

    # Evaluate on D_valid
    print("\n6. Evaluating on D_valid...")
    if len(split.d_valid_tasks) > 0:
        valid_gt = items_df.loc[split.d_valid_tasks, "b"]

        prior_valid_preds = prior_model.get_prior_predictions(split.d_valid_tasks)
        prior_valid_eval = evaluate_predictions(prior_valid_preds, valid_gt)
        prior_valid_auc = compute_auc(prior_valid_preds, abilities_df, responses, split.d_valid_tasks, split.m2_agents)
        print(f"   Prior AUC: {prior_valid_auc.get('auc', 'N/A'):.4f}" if 'auc' in prior_valid_auc else f"   Prior AUC: {prior_valid_auc}")

        # Use M1 embeddings for validation (since we computed on M1)
        posterior_valid_preds = posterior_model.predict(split.d_valid_tasks, split.m1_agents, embeddings_dir)
        posterior_valid_eval = evaluate_predictions(posterior_valid_preds, valid_gt)
        posterior_valid_auc = compute_auc(posterior_valid_preds, abilities_df, responses, split.d_valid_tasks, split.m2_agents)
        print(f"   Posterior AUC: {posterior_valid_auc.get('auc', 'N/A'):.4f}" if 'auc' in posterior_valid_auc else f"   Posterior AUC: {posterior_valid_auc}")
    else:
        prior_valid_eval = {"error": "No validation tasks"}
        prior_valid_auc = {"error": "No validation tasks"}
        posterior_valid_eval = {"error": "No validation tasks"}
        posterior_valid_auc = {"error": "No validation tasks"}

    # Get training stats including actual alpha used
    training_stats = posterior_model.get_training_stats()
    psi_stats = posterior_model.get_psi_stats()

    # Add timestamp for tracking
    import datetime
    timestamp = datetime.datetime.now().isoformat()

    # Compile results with comprehensive metadata
    results = {
        "timestamp": timestamp,
        "run_id": config.get_run_id(),
        # Full config for reproducibility
        "config": config.to_dict(),
        # Key ablation parameters (duplicated at top level for easy querying)
        "ablation": {
            "backbone": config.backbone,
            "content_type": config.content_type,
            "instruction_type": config.instruction_type,
            "aggregation": config.aggregation,
            "posterior_alpha_requested": str(config.posterior_alpha),
            "posterior_alpha_used": training_stats.get("best_alpha"),  # Actual alpha (from CV or specified)
            "prior_alpha": config.prior_alpha,
            "embedding_dim": config.embedding_dim,
            "max_length": config.max_length,
        },
        # Data split info
        "split": {
            "n_m1_agents": len(split.m1_agents),
            "n_m2_agents": len(split.m2_agents),
            "n_d_train_tasks": len(split.d_train_tasks),
            "n_d_valid_tasks": len(split.d_valid_tasks),
            "m1_agents": split.m1_agents,
            "m2_agents": split.m2_agents,
            "d_train_tasks": split.d_train_tasks,
            "d_valid_tasks": split.d_valid_tasks,
        },
        # Results
        "prior_train": prior_train_eval,
        "prior_train_auc": prior_train_auc,
        "posterior_train": posterior_train_eval,
        "posterior_train_auc": posterior_train_auc,
        "prior_valid": prior_valid_eval,
        "prior_valid_auc": prior_valid_auc,
        "posterior_valid": posterior_valid_eval,
        "posterior_valid_auc": posterior_valid_auc,
        # Model stats
        "posterior_training_stats": training_stats,
        "psi_stats": psi_stats,
    }

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    prior_auc = prior_valid_auc.get("auc", 0)
    post_auc = posterior_valid_auc.get("auc", 0)
    if isinstance(prior_auc, float) and isinstance(post_auc, float):
        delta_auc = post_auc - prior_auc
        print(f"Prior AUC (D_valid):     {prior_auc:.4f}")
        print(f"Posterior AUC (D_valid): {post_auc:.4f}")
        print(f"ΔAUC:                    {delta_auc:+.4f}")
    else:
        print(f"Prior AUC: {prior_auc}")
        print(f"Posterior AUC: {post_auc}")

    print(f"\nTraining stats: {training_stats}")
    print(f"Psi stats: {psi_stats}")

    # Save results with unique filename based on run_id
    run_id = config.get_run_id()
    output_file = output_dir / f"results__{run_id}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")

    # Also append to a master results log (JSONL format) for easy aggregation
    master_log = output_dir / "all_results.jsonl"
    with open(master_log, "a") as f:
        f.write(json.dumps(results, default=str) + "\n")
    print(f"Appended to master log: {master_log}")

    return results


def run_ablations(
    embeddings_base_dir: Path,
    output_dir: Path,
    content_types: List[str] = ["full", "condensed"],
    instruction_types: List[str] = ["difficulty", "residual"],
    aggregations: List[AggregationType] = ["mean_only", "mean_std", "all_stats"],
) -> Dict[str, Dict]:
    """Run ablations over content, instruction, and aggregation types.

    Args:
        embeddings_base_dir: Base directory containing {content}_{instruction} subdirs
        output_dir: Where to save results
        content_types: Content types to test
        instruction_types: Instruction types to test
        aggregations: Aggregation strategies to test

    Returns:
        Dict mapping config_name -> results
    """
    all_results = {}

    for content in content_types:
        for instruction in instruction_types:
            embeddings_dir = embeddings_base_dir / f"{content}_{instruction}"
            if not embeddings_dir.exists():
                print(f"Skipping {content}_{instruction}: directory not found")
                continue

            for aggregation in aggregations:
                config_name = f"{content}_{instruction}_{aggregation}"
                print(f"\n{'='*60}")
                print(f"Running: {config_name}")
                print(f"{'='*60}")

                config = EmbeddingExperimentConfig(
                    embeddings_dir=embeddings_dir,
                    aggregation=aggregation,
                    output_dir=output_dir,
                    # Pre-set known ablation parameters from directory structure
                    content_type=content,
                    instruction_type=instruction,
                )

                try:
                    results = run_embedding_experiment(config)
                    all_results[config_name] = results
                except Exception as e:
                    print(f"Error: {e}")
                    all_results[config_name] = {"error": str(e)}

    # Summary table
    print("\n" + "=" * 80)
    print("ABLATION SUMMARY")
    print("=" * 80)
    print(f"{'Config':<40} {'Train AUC':>12} {'Valid AUC':>12} {'ΔAUC':>12}")
    print("-" * 80)

    for config_name, results in all_results.items():
        if "error" in results:
            print(f"{config_name:<40} {'ERROR':>12}")
            continue

        train_auc = results.get("posterior_train_auc", {}).get("auc", "N/A")
        valid_auc = results.get("posterior_valid_auc", {}).get("auc", "N/A")
        prior_valid_auc = results.get("prior_valid_auc", {}).get("auc", 0)

        if isinstance(valid_auc, float) and isinstance(prior_valid_auc, float):
            delta = valid_auc - prior_valid_auc
            delta_str = f"{delta:+.4f}"
        else:
            delta_str = "N/A"

        train_str = f"{train_auc:.4f}" if isinstance(train_auc, float) else str(train_auc)
        valid_str = f"{valid_auc:.4f}" if isinstance(valid_auc, float) else str(valid_auc)

        print(f"{config_name:<40} {train_str:>12} {valid_str:>12} {delta_str:>12}")

    # Save all results
    summary_file = output_dir / "ablation_summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nFull results saved to: {summary_file}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate embedding-based posterior model"
    )

    # Single experiment mode
    parser.add_argument(
        "--embeddings_dir",
        type=Path,
        default=None,
        help="Directory containing trajectory embeddings",
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        default="mean_std",
        choices=["mean_only", "mean_std", "weighted", "all_stats"],
        help="Aggregation strategy",
    )
    parser.add_argument(
        "--alpha",
        type=str,
        default="cv",
        help="Ridge alpha (float or 'cv' for cross-validation)",
    )

    # Ablation mode
    parser.add_argument(
        "--ablations",
        action="store_true",
        help="Run full ablation study",
    )
    parser.add_argument(
        "--embeddings_base_dir",
        type=Path,
        default=Path("chris_output/experiment_b/trajectory_embeddings"),
        help="Base directory for ablation embeddings",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("chris_output/experiment_b/embedding_results"),
        help="Output directory for results",
    )

    # Prior config
    parser.add_argument(
        "--prior_source",
        type=str,
        default="embedding",
        choices=["embedding", "heuristic"],
        help="Prior model source",
    )

    # Regression mode
    parser.add_argument(
        "--regression_mode",
        type=str,
        choices=["residual", "direct_with_prior", "direct_with_prior_features"],
        default="residual",
        help="Regression mode: 'residual' (prior + correction), 'direct_with_prior' (prior as feature), 'direct_with_prior_features' (prior input features)",
    )
    parser.add_argument(
        "--compare_modes",
        action="store_true",
        help="Run all regression modes and compare results",
    )

    args = parser.parse_args()

    if args.ablations:
        # Run full ablation study
        run_ablations(
            embeddings_base_dir=args.embeddings_base_dir,
            output_dir=args.output_dir,
        )
    elif args.compare_modes:
        # Compare all regression modes
        if args.embeddings_dir is None:
            args.embeddings_dir = Path("chris_output/experiment_b/trajectory_embeddings/full_difficulty")

        from dataclasses import replace
        modes: List[RegressionMode] = ["residual", "direct_with_prior", "direct_with_prior_features"]
        all_results = {}

        for mode in modes:
            print(f"\n{'='*60}")
            print(f"RUNNING REGRESSION MODE: {mode}")
            print(f"{'='*60}")

            config = EmbeddingExperimentConfig(
                embeddings_dir=args.embeddings_dir,
                aggregation=args.aggregation,
                posterior_alpha=float(args.alpha) if args.alpha != "cv" else "cv",
                output_dir=args.output_dir,
                prior_source=args.prior_source,
                regression_mode=mode,
            )

            try:
                results = run_embedding_experiment(config)
                all_results[mode] = results
            except Exception as e:
                print(f"Error running mode {mode}: {e}")
                all_results[mode] = {"error": str(e)}

        # Print comparison summary
        print("\n" + "=" * 80)
        print("REGRESSION MODE COMPARISON")
        print("=" * 80)
        print(f"{'Mode':<30} {'Train AUC':>12} {'Valid AUC':>12}")
        print("-" * 60)

        for mode in modes:
            if mode not in all_results or "error" in all_results[mode]:
                print(f"{mode:<30} {'ERROR':>12}")
                continue
            train_auc = all_results[mode].get("posterior_train_auc", {}).get("auc")
            valid_auc = all_results[mode].get("posterior_valid_auc", {}).get("auc")
            train_str = f"{train_auc:.4f}" if train_auc else "N/A"
            valid_str = f"{valid_auc:.4f}" if valid_auc else "N/A"
            print(f"{mode:<30} {train_str:>12} {valid_str:>12}")

        # Save comparison results
        output_dir = ROOT / args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        comparison_path = output_dir / "mode_comparison_results.json"
        with open(comparison_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nComparison results saved to: {comparison_path}")
    else:
        # Single experiment
        if args.embeddings_dir is None:
            args.embeddings_dir = Path("chris_output/experiment_b/trajectory_embeddings/full_difficulty")

        config = EmbeddingExperimentConfig(
            embeddings_dir=args.embeddings_dir,
            aggregation=args.aggregation,
            posterior_alpha=float(args.alpha) if args.alpha != "cv" else "cv",
            output_dir=args.output_dir,
            prior_source=args.prior_source,
            regression_mode=args.regression_mode,
        )

        run_embedding_experiment(config)


if __name__ == "__main__":
    main()

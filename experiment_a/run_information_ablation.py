"""Run information source ablation for SWE-bench Verified.

This script runs the full ablation study comparing different information sources:
1. LLM Judge ablation: Problem Only → +Auditor → +Test → +Solution (Full)
2. Embedding ablation: With vs Without solution in embedding prompt

For the LLM Judge ablation, feature count is held constant at 15 via Ridge
coefficient-based selection to isolate the value of each information source.

For consistency in Grouped Ridge:
- No-solution embedding is paired with problem-only LLM features (neither has solution)
- With-solution embedding is paired with full LLM features (both have solution)

Usage:
    python -m experiment_a.run_information_ablation
    python -m experiment_a.run_information_ablation --rebuild_csvs  # Regenerate ablation CSVs
"""

import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from experiment_a.shared.config import ExperimentAConfig, build_spec
from experiment_a.shared.pipeline import run_cross_validation


ROOT = Path(__file__).resolve().parents[1]

# Experiment specification for SWE-bench
SPEC = build_spec("swebench", ROOT)


@dataclass
class AblationConfig:
    """Configuration for the ablation study."""
    # Feature source paths
    problem_orig_path: Path = Path("chris_output/llm_judge_features/swebench_unified_problem_only/llm_judge_features.csv")
    problem_ext_path: Path = Path("chris_output/llm_judge_features/swebench_problem_extended/llm_judge_features.csv")
    auditor_path: Path = Path("chris_output/auditor_pilot/v3_features_top3.csv")
    test_no_sol_path: Path = Path("chris_output/llm_judge_features/swebench_test_quality_no_solution/llm_judge_features.csv")
    test_with_sol_path: Path = Path("chris_output/llm_judge_features/swebench_test_quality_CONTAMINATED_has_solution/llm_judge_features.csv")
    unified_path: Path = Path("chris_output/llm_judge_features/swebench_unified/llm_judge_features.csv")

    # IRT ground truth
    items_path: Path = Path("clean_data/swebench_verified_20251120_full/1d_1pl/items.csv")

    # Embedding paths
    # With solution: uses solcap_nocap (includes solution in embedding prompt)
    emb_with_sol_path: Path = Path(
        "embeddings/embeddings__deepseek-ai__DeepSeek-R1-Distill-Qwen-32B__pool-lasttoken__qs-sol-instr__"
        "__solcap_nocapqs_sol_no_tests_instr_nocap_b7008f2d__idnorm_instance-v1__"
        "princeton-nlp_SWE-bench_Verified__test__maxlen8192.npz"
    )
    # Without solution: uses qs_sol_instr (no solution in embedding prompt)
    emb_no_sol_path: Path = Path(
        "embeddings/embeddings__deepseek-ai__DeepSeek-R1-Distill-Qwen-32B__pool-lasttoken__qs-sol-instr__"
        "qs_sol_instr_b7008f2d__idnorm_instance-v1__"
        "princeton-nlp_SWE-bench_Verified__test__maxlen8192.npz"
    )

    # Output
    output_dir: Path = Path("chris_output/llm_judge_features/swebench_ablation_controlled_v3")

    # CV settings
    k_folds: int = 5


# Feature column definitions
PROBLEM_ORIG_FEATURES = [
    "solution_hint", "problem_clarity", "domain_knowledge_required",
    "logical_reasoning_required", "atypicality", "verification_difficulty",
    "standard_pattern_available"
]

PROBLEM_EXT_FEATURES = [
    "error_specificity", "reproduction_clarity", "expected_behavior_clarity",
    "debugging_complexity", "codebase_scope", "information_completeness",
    "similar_issue_likelihood", "backwards_compatibility_risk"
]

AUDITOR_FEATURES = ["entry_point_clarity", "change_blast_radius", "fix_localization"]

TEST_FEATURES = ["test_comprehensiveness", "test_assertion_complexity", "test_edge_case_coverage"]

SOLUTION_FEATURES = ["solution_complexity", "integration_complexity"]


def select_top_features(df: pd.DataFrame, available_features: List[str], n: int = 15, target_col: str = "b") -> List[str]:
    """Select top N features by Ridge coefficient magnitude."""
    X = df[available_features].values
    y = df[target_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_scaled, y)

    coef_abs = np.abs(ridge.coef_)
    top_indices = np.argsort(coef_abs)[::-1][:n]
    return [available_features[i] for i in top_indices]


def build_ablation_csvs(config: AblationConfig) -> None:
    """Build the ablation CSVs with feature selection."""
    print("Building ablation CSVs...")

    # Load IRT difficulties as target
    items = pd.read_csv(config.items_path, index_col=0)
    items = items.reset_index().rename(columns={"index": "instance_id"})

    # Load feature sources
    problem_orig = pd.read_csv(config.problem_orig_path).rename(columns={"_task_id": "instance_id"})
    problem_ext = pd.read_csv(config.problem_ext_path).rename(columns={"_task_id": "instance_id"})
    auditor = pd.read_csv(config.auditor_path).rename(columns={"task_id": "instance_id"})
    test_no_sol = pd.read_csv(config.test_no_sol_path).rename(columns={"_task_id": "instance_id"})
    test_with_sol = pd.read_csv(config.test_with_sol_path).rename(columns={"_task_id": "instance_id"})
    unified = pd.read_csv(config.unified_path).rename(columns={"_task_id": "instance_id"})

    # Merge all features
    all_features = items[["instance_id", "b"]].copy()
    all_features = all_features.merge(problem_orig[["instance_id"] + PROBLEM_ORIG_FEATURES], on="instance_id")
    all_features = all_features.merge(problem_ext[["instance_id"] + PROBLEM_EXT_FEATURES], on="instance_id")
    all_features = all_features.merge(auditor[["instance_id"] + AUDITOR_FEATURES], on="instance_id")
    all_features = all_features.merge(test_no_sol[["instance_id"] + TEST_FEATURES], on="instance_id")
    all_features = all_features.merge(unified[["instance_id"] + SOLUTION_FEATURES], on="instance_id")

    # For Full level, merge test features WITH solution (different column names)
    test_with_sol_renamed = test_with_sol.rename(columns={
        "test_comprehensiveness": "test_comprehensiveness_withsol",
        "test_assertion_complexity": "test_assertion_complexity_withsol",
        "test_edge_case_coverage": "test_edge_case_coverage_withsol",
    })
    all_features = all_features.merge(
        test_with_sol_renamed[["instance_id", "test_comprehensiveness_withsol",
                               "test_assertion_complexity_withsol", "test_edge_case_coverage_withsol"]],
        on="instance_id"
    )

    print(f"  Merged {len(all_features)} tasks with all features")

    # Define feature sets for each level
    # Levels 1-3 use test features WITHOUT solution
    test_features_nosol = TEST_FEATURES
    # Level 4 (Full) uses test features WITH solution
    test_features_withsol = ["test_comprehensiveness_withsol", "test_assertion_complexity_withsol",
                             "test_edge_case_coverage_withsol"]

    level_configs = [
        ("1_problem_15.csv", PROBLEM_ORIG_FEATURES + PROBLEM_EXT_FEATURES),
        ("2_prob_aud_15.csv", PROBLEM_ORIG_FEATURES + PROBLEM_EXT_FEATURES + AUDITOR_FEATURES),
        ("3_prob_aud_test_15.csv", PROBLEM_ORIG_FEATURES + PROBLEM_EXT_FEATURES + AUDITOR_FEATURES + test_features_nosol),
        ("4_full_15.csv", PROBLEM_ORIG_FEATURES + PROBLEM_EXT_FEATURES + AUDITOR_FEATURES + test_features_withsol + SOLUTION_FEATURES),
    ]

    config.output_dir.mkdir(parents=True, exist_ok=True)

    for filename, available_features in level_configs:
        top_features = select_top_features(all_features, available_features, n=15)
        df = all_features[["instance_id"] + top_features].copy()

        # Rename _withsol columns back to standard names for level 4
        df = df.rename(columns={
            "test_comprehensiveness_withsol": "test_comprehensiveness",
            "test_assertion_complexity_withsol": "test_assertion_complexity",
            "test_edge_case_coverage_withsol": "test_edge_case_coverage",
        })

        output_path = config.output_dir / filename
        df.to_csv(output_path, index=False)
        print(f"  Saved {output_path}")


def get_embedding_dim(embeddings_path: Path) -> int:
    """Read embedding dimension from npz file."""
    data = np.load(embeddings_path, allow_pickle=True)
    return int(data["X"].shape[1])


def get_num_features(csv_path: Path) -> int:
    """Read number of numeric feature columns from CSV file."""
    df = pd.read_csv(csv_path)
    # Count numeric columns excluding metadata (instance_id, _task_id, etc.)
    feature_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and not c.startswith("_")
        and c not in ("instance_id", "task_id")
    ]
    return len(feature_cols)


def run_experiment(embeddings_path: Path, llm_judge_path: Path, k_folds: int) -> Dict[str, Any]:
    """Run a single experiment by calling the pipeline directly.

    Args:
        embeddings_path: Path to embeddings .npz file
        llm_judge_path: Path to LLM judge features CSV
        k_folds: Number of cross-validation folds

    Returns:
        Dict with AUC results for each predictor type
    """
    # Create config with SWE-bench defaults, overriding feature paths
    config = ExperimentAConfig.for_dataset(
        "swebench",
        embeddings_path=embeddings_path,
        llm_judge_features_path=llm_judge_path,
    )

    # Run cross-validation directly
    cv_output = run_cross_validation(config, SPEC, ROOT, k=k_folds)
    cv_results = cv_output["cv_results"]

    # Extract results from the cv_results dict
    results = {}

    # LLM Judge
    if "llm_judge" in cv_results:
        r = cv_results["llm_judge"]
        if r.get("mean_auc") is not None:
            results["llm_auc"] = r["mean_auc"]
            results["llm_std"] = r["std_auc"]

    # Embedding
    if "embedding" in cv_results:
        r = cv_results["embedding"]
        if r.get("mean_auc") is not None:
            results["emb_auc"] = r["mean_auc"]
            results["emb_std"] = r["std_auc"]

    # Grouped (Emb + LLM)
    if "grouped" in cv_results:
        r = cv_results["grouped"]
        if r.get("mean_auc") is not None:
            results["grouped_auc"] = r["mean_auc"]
            results["grouped_std"] = r["std_auc"]

    # Oracle
    if "oracle" in cv_results:
        r = cv_results["oracle"]
        if r.get("mean_auc") is not None:
            results["oracle_auc"] = r["mean_auc"]
            results["oracle_std"] = r["std_auc"]

    return results


def main():
    parser = argparse.ArgumentParser(description="Run information source ablation study")
    parser.add_argument("--rebuild_csvs", action="store_true", help="Regenerate ablation CSVs")
    parser.add_argument("--k_folds", type=int, default=5, help="Number of CV folds")
    args = parser.parse_args()

    config = AblationConfig(k_folds=args.k_folds)

    # Build CSVs if requested or if they don't exist
    if args.rebuild_csvs or not (config.output_dir / "1_problem_15.csv").exists():
        build_ablation_csvs(config)

    print("\n" + "=" * 70)
    print("INFORMATION SOURCE ABLATION STUDY")
    print("=" * 70)

    # LLM Judge ablation
    # For consistency: levels without solution access use no-solution embedding,
    # Full (with solution access) uses with-solution embedding
    llm_ablation_configs = [
        ("Problem Only", config.output_dir / "1_problem_15.csv", config.emb_no_sol_path),
        ("+ Auditor", config.output_dir / "2_prob_aud_15.csv", config.emb_no_sol_path),
        ("+ Test", config.output_dir / "3_prob_aud_test_15.csv", config.emb_no_sol_path),
        ("Full", config.output_dir / "4_full_15.csv", config.emb_with_sol_path),
    ]

    llm_results = []
    oracle_auc = None
    for name, llm_path, emb_path in llm_ablation_configs:
        print(f"\nRunning: {name}")
        results = run_experiment(emb_path, llm_path, config.k_folds)
        results["name"] = name
        results["llm_path"] = llm_path
        results["n_features"] = get_num_features(llm_path)
        llm_results.append(results)
        # Capture Oracle AUC from first run
        if oracle_auc is None and results.get("oracle_auc"):
            oracle_auc = results["oracle_auc"]
        llm_auc = results.get('llm_auc')
        llm_std = results.get('llm_std')
        grouped_auc = results.get('grouped_auc')
        grouped_std = results.get('grouped_std')
        print(f"  LLM Judge: {llm_auc:.4f} ± {llm_std:.4f}" if llm_auc else "  LLM Judge: N/A")
        print(f"  Grouped Ridge: {grouped_auc:.4f} ± {grouped_std:.4f}" if grouped_auc else "  Grouped Ridge: N/A")

    # Embedding ablation
    # No-solution embedding paired with problem-only LLM (for consistency - neither has solution)
    print(f"\nRunning: Embedding (no solution) + Problem-only LLM")
    emb_no_sol_results = run_experiment(
        config.emb_no_sol_path,
        config.output_dir / "1_problem_15.csv",  # Problem-only LLM
        config.k_folds
    )
    emb_auc = emb_no_sol_results.get('emb_auc')
    grouped_auc = emb_no_sol_results.get('grouped_auc')
    print(f"  Embedding alone: {emb_auc:.4f}" if emb_auc else "  Embedding alone: N/A")
    print(f"  Grouped Ridge: {grouped_auc:.4f}" if grouped_auc else "  Grouped Ridge: N/A")

    # With-solution embedding paired with full LLM (for consistency - both have solution)
    print(f"\nRunning: Embedding (with solution) + Full LLM")
    emb_with_sol_results = run_experiment(
        config.emb_with_sol_path,
        config.output_dir / "4_full_15.csv",  # Full LLM
        config.k_folds
    )
    emb_auc = emb_with_sol_results.get('emb_auc')
    grouped_auc = emb_with_sol_results.get('grouped_auc')
    print(f"  Embedding alone: {emb_auc:.4f}" if emb_auc else "  Embedding alone: N/A")
    print(f"  Grouped Ridge: {grouped_auc:.4f}" if grouped_auc else "  Grouped Ridge: N/A")

    # Final summary table
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print("\nLLM Judge Ablation:")
    print(f"{'Method':<25} {'# Feat':<8} {'LLM Judge':<18} {'Grouped Ridge':<18}")
    print("-" * 70)
    for r in llm_results:
        llm_str = f"{r.get('llm_auc', 0):.4f} ± {r.get('llm_std', 0):.4f}"
        grouped_str = f"{r.get('grouped_auc', 0):.4f} ± {r.get('grouped_std', 0):.4f}"
        n_feat = r.get('n_features', 15)
        print(f"{r['name']:<25} {n_feat:<8} {llm_str:<18} {grouped_str:<18}")

    print("\nEmbedding Ablation:")
    print(f"{'Method':<25} {'# Feat':<8} {'Source Alone':<18} {'Grouped Ridge':<18}")
    print("-" * 70)
    emb_dim = get_embedding_dim(config.emb_no_sol_path)
    print(f"{'Without Solution':<25} {emb_dim:<8} {emb_no_sol_results.get('emb_auc', 0):.4f}{'':14} {emb_no_sol_results.get('grouped_auc', 0):.4f}")
    print(f"{'With Solution':<25} {emb_dim:<8} {emb_with_sol_results.get('emb_auc', 0):.4f}{'':14} {emb_with_sol_results.get('grouped_auc', 0):.4f}")

    # LaTeX table
    print("\n" + "=" * 70)
    print("LATEX TABLE")
    print("=" * 70)
    print(r"""
\begin{tabular}{lccc}
\toprule
Information Source & \# Feat. & Source Alone & Grouped Ridge \\
\midrule
\multicolumn{4}{l}{\textit{LLM Judge Ablation}} \\""")

    for r in llm_results:
        prefix = r"\quad " if r["name"] != "Problem Only" else r"\quad "
        if r["name"] == "+ Test":
            prefix = r"\quad \quad "
        elif r["name"] == "Full":
            prefix = r"\quad \quad \quad "

        name_display = r["name"].replace("+ ", "+ ")
        if r["name"] == "Full":
            name_display = "+ Solution Patch (Full)"

        n_feat = r.get('n_features', 15)
        llm_val = f"{r.get('llm_auc', 0):.3f}"
        grouped_val = f"{r.get('grouped_auc', 0):.3f}"

        if r["name"] == "Full":
            print(f"{prefix}{name_display} & {n_feat} & \\textbf{{{llm_val}}} & \\textbf{{{grouped_val}}} \\\\")
        else:
            print(f"{prefix}{name_display} & {n_feat} & {llm_val} & {grouped_val} \\\\")

    print(r"""\midrule
\multicolumn{4}{l}{\textit{Embedding Ablation}} \\""")
    print(f"\\quad Without Solution & {emb_dim} & {emb_no_sol_results.get('emb_auc', 0):.3f} & {emb_no_sol_results.get('grouped_auc', 0):.3f} \\\\")
    print(f"\\quad With Solution & {emb_dim} & {emb_with_sol_results.get('emb_auc', 0):.3f} & {emb_with_sol_results.get('grouped_auc', 0):.3f} \\\\")
    oracle_str = f"{oracle_auc:.3f}" if oracle_auc else "N/A"
    print(rf"""\midrule
Oracle & --- & --- & {oracle_str} \\
\bottomrule
\end{{tabular}}""")


if __name__ == "__main__":
    main()

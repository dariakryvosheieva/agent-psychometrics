"""
Analyze what the second PCA dimension (PC2) of learned model embeddings represents.

PC2 is orthogonal to IRT ability (theta) - we want to understand what it captures.

Hypotheses tested:
1. Verbosity (mean trajectory length)
2. Reasoning style (explicit CoT models vs standard)
3. Consistency vs specialization (success rate variance)
4. Conversation depth (number of messages)
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Add swebench_irt to path for split_agent_name
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "swebench_irt"))
from split_agents_model_scaffold import split_agent_name

# Paths
PC2_CSV = PROJECT_ROOT / "embed_pca_vs_oracle_theta__model.csv"
CHAR_COUNTS_CSV = PROJECT_ROOT / "chris_output" / "tensor_analysis" / "swebench_verified_char_counts.csv"
OUTPUT_DIR = PROJECT_ROOT / "chris_output" / "pc2_analysis"

# ============================================================================
# Reasoning Model Classification
# Based on web research:
# - https://www.anthropic.com/news/claude-3-7-sonnet (Claude 3.7+ has extended thinking)
# - https://huggingface.co/moonshotai/Kimi-K2-Thinking (Kimi K2 Thinking)
# - https://huggingface.co/deepseek-ai/DeepSeek-R1 (DeepSeek R1)
# ============================================================================

# Models that ALWAYS use chain-of-thought reasoning
ALWAYS_REASONING = {
    "o1_crosscheck5",  # OpenAI o1 series
    "o3_mini",  # OpenAI o3 series
    "o4-mini",  # OpenAI o4 series
    "GPT-5",  # GPT-5 has always-reasoning mode
}

# Models with extended thinking (assuming it was enabled per user)
EXTENDED_THINKING = {
    "Claude 3.7 Sonnet",  # First Claude with extended thinking
    "Claude 4 Sonnet",  # Claude 4 series has extended thinking
    "claude-4-opus",  # Claude 4 Opus
    "claude-sonnet-4-5",  # Claude Sonnet 4.5
    "Kimi K2",  # Kimi K2 Thinking variant
    "kimi_k2_instruct",  # Kimi K2 with thinking
    "gemini_2.0_flash_experimental",  # Gemini Flash Thinking
}

# NOT reasoning models (important to exclude):
# - DeepSeek-V3 is the base model, NOT R1
# - Claude 2, 3 Opus, 3.5 Sonnet are pre-extended-thinking
# - GPT-3.5, GPT-4, GPT-4o don't have explicit reasoning mode


def is_reasoning_model(model_name: str) -> bool:
    """Classify whether a model uses explicit reasoning/chain-of-thought."""
    if model_name in ALWAYS_REASONING:
        return True
    if model_name in EXTENDED_THINKING:
        return True
    # Pattern matching for variants
    name_lower = model_name.lower()
    if any(p in name_lower for p in ["o1", "o3", "o4-mini"]):
        return True
    if "gpt-5" in name_lower or "gpt5" in name_lower:
        return True
    if "claude" in name_lower and any(v in name_lower for v in ["3.7", "4", "4.5"]):
        return True
    if "kimi" in name_lower and "k2" in name_lower:
        return True
    return False


def get_model_for_agent(agent: str) -> str | None:
    """Map agent name to canonical model name."""
    result = split_agent_name(agent)
    if result is None:
        return None
    model, scaffold, _, _ = result
    return model


def load_pc2_data() -> pd.DataFrame:
    """Load PC2 data from CSV."""
    df = pd.read_csv(PC2_CSV)
    print(f"Loaded PC2 data: {len(df)} models")
    print(f"  PC1-theta correlation: {df['pc1'].corr(df['theta']):.3f}")
    print(f"  PC2-theta correlation: {df['pc2'].corr(df['theta']):.3f}")
    return df


def load_char_counts() -> pd.DataFrame:
    """Load character counts data."""
    df = pd.read_csv(CHAR_COUNTS_CSV)
    print(f"Loaded char_counts: {len(df)} rows, {df['agent'].nunique()} agents")
    return df


def map_agents_to_models(char_counts: pd.DataFrame) -> pd.DataFrame:
    """Add model column to char_counts by mapping agent names."""
    # Map agents to models
    char_counts = char_counts.copy()
    char_counts["model"] = char_counts["agent"].apply(get_model_for_agent)

    # Report coverage
    n_mapped = char_counts["model"].notna().sum()
    n_total = len(char_counts)
    n_models = char_counts["model"].nunique()
    print(f"Agent → Model mapping: {n_mapped}/{n_total} rows mapped ({n_models} unique models)")

    # Show unmapped agents
    unmapped = char_counts[char_counts["model"].isna()]["agent"].unique()
    if len(unmapped) > 0:
        print(f"  Unmapped agents: {list(unmapped)}")

    return char_counts


def compute_model_features(char_counts: pd.DataFrame) -> pd.DataFrame:
    """Compute model-level features from agent-level char_counts."""
    # Filter to rows with valid model mapping
    df = char_counts[char_counts["model"].notna()].copy()

    # Compute per-model statistics
    model_stats = df.groupby("model").agg(
        mean_char_count=("assistant_char_count", "mean"),
        std_char_count=("assistant_char_count", "std"),
        mean_n_messages=("n_assistant_messages", "mean"),
        std_n_messages=("n_assistant_messages", "std"),
        n_agents=("agent", "nunique"),
        n_observations=("agent", "count"),
        mean_resolved=("resolved", "mean"),  # Success rate
    ).reset_index()

    # Compute derived features
    model_stats["chars_per_message"] = model_stats["mean_char_count"] / model_stats["mean_n_messages"]

    # Compute per-task success rate variance (for consistency hypothesis)
    # First get per-model, per-task success rates
    task_success = df.groupby(["model", "task_id"])["resolved"].mean().reset_index()
    # Then compute variance across tasks for each model
    success_variance = task_success.groupby("model")["resolved"].std().reset_index()
    success_variance.columns = ["model", "success_rate_std"]
    model_stats = model_stats.merge(success_variance, on="model", how="left")

    # Add reasoning model classification
    model_stats["is_reasoning"] = model_stats["model"].apply(is_reasoning_model)

    print(f"\nModel features computed for {len(model_stats)} models")
    print(f"  Reasoning models: {model_stats['is_reasoning'].sum()}")
    print(f"  Standard models: {(~model_stats['is_reasoning']).sum()}")

    return model_stats


def merge_with_pc2(model_features: pd.DataFrame, pc2_df: pd.DataFrame) -> pd.DataFrame:
    """Merge model features with PC2 data."""
    # The PC2 data has model names in 'id' column
    merged = pc2_df.merge(model_features, left_on="id", right_on="model", how="left")

    # Report coverage
    n_matched = merged["model"].notna().sum()
    n_total = len(merged)
    print(f"\nPC2 ↔ model features merge: {n_matched}/{n_total} models matched")

    # Show unmatched models
    unmatched = merged[merged["model"].isna()]["id"].tolist()
    if len(unmatched) > 0:
        print(f"  Unmatched PC2 models (no char_counts data): {unmatched}")

    # For unmatched models, still classify as reasoning/standard using PC2 id
    merged.loc[merged["is_reasoning"].isna(), "is_reasoning"] = merged.loc[
        merged["is_reasoning"].isna(), "id"
    ].apply(is_reasoning_model)

    return merged


def compute_correlations(merged: pd.DataFrame) -> pd.DataFrame:
    """Compute correlations between PC2 and model features."""
    features = [
        "mean_char_count",
        "std_char_count",
        "mean_n_messages",
        "chars_per_message",
        "success_rate_std",
        "mean_resolved",
    ]

    results = []
    for feature in features:
        # Filter to rows with valid values
        valid = merged[["pc2", feature]].dropna()
        if len(valid) < 3:
            continue

        # Pearson correlation
        r_pearson, p_pearson = stats.pearsonr(valid["pc2"], valid[feature])

        # Spearman correlation (more robust)
        r_spearman, p_spearman = stats.spearmanr(valid["pc2"], valid[feature])

        results.append({
            "feature": feature,
            "n": len(valid),
            "pearson_r": r_pearson,
            "pearson_p": p_pearson,
            "spearman_r": r_spearman,
            "spearman_p": p_spearman,
        })

    return pd.DataFrame(results)


def test_reasoning_hypothesis(merged: pd.DataFrame) -> dict:
    """Test if reasoning models differ from standard models on PC2."""
    # Filter to rows with valid is_reasoning
    valid = merged[merged["is_reasoning"].notna()].copy()

    reasoning = valid[valid["is_reasoning"] == True]["pc2"]  # noqa: E712
    standard = valid[valid["is_reasoning"] == False]["pc2"]  # noqa: E712

    if len(reasoning) < 2 or len(standard) < 2:
        return {"error": "Not enough data for comparison"}

    # t-test
    t_stat, t_pvalue = stats.ttest_ind(reasoning, standard)

    # Mann-Whitney U test (non-parametric)
    u_stat, u_pvalue = stats.mannwhitneyu(reasoning, standard, alternative="two-sided")

    return {
        "n_reasoning": len(reasoning),
        "n_standard": len(standard),
        "reasoning_pc2_mean": reasoning.mean(),
        "reasoning_pc2_std": reasoning.std(),
        "standard_pc2_mean": standard.mean(),
        "standard_pc2_std": standard.std(),
        "t_statistic": t_stat,
        "t_pvalue": t_pvalue,
        "u_statistic": u_stat,
        "u_pvalue": u_pvalue,
    }


def plot_correlations(merged: pd.DataFrame, correlations: pd.DataFrame, output_dir: Path):
    """Generate visualization plots."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Get top correlating features
    top_features = correlations.sort_values("spearman_r", key=abs, ascending=False)["feature"].head(4).tolist()

    # Plot 1-4: Scatter plots for top features
    for i, feature in enumerate(top_features):
        ax = axes[i // 3, i % 3]
        valid = merged[["pc2", feature, "id"]].dropna()

        ax.scatter(valid[feature], valid["pc2"], alpha=0.7, s=50)

        # Add regression line
        if len(valid) > 2:
            z = np.polyfit(valid[feature], valid["pc2"], 1)
            p = np.poly1d(z)
            x_line = np.linspace(valid[feature].min(), valid[feature].max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8)

        # Get correlation
        r = correlations[correlations["feature"] == feature]["spearman_r"].values[0]
        ax.set_xlabel(feature)
        ax.set_ylabel("PC2")
        ax.set_title(f"PC2 vs {feature}\n(Spearman r = {r:.3f})")
        ax.grid(True, alpha=0.3)

    # Plot 5: Reasoning vs Standard box plot
    ax = axes[1, 1]
    valid = merged[merged["is_reasoning"].notna()].copy()
    reasoning_data = [
        valid[valid["is_reasoning"] == True]["pc2"].dropna(),  # noqa: E712
        valid[valid["is_reasoning"] == False]["pc2"].dropna(),  # noqa: E712
    ]
    ax.boxplot(reasoning_data, labels=["Reasoning", "Standard"])
    ax.set_ylabel("PC2")
    ax.set_title("PC2 by Model Type")
    ax.grid(True, alpha=0.3)

    # Plot 6: PC1 vs PC2 colored by theta (sanity check)
    ax = axes[1, 2]
    valid = merged[["pc1", "pc2", "theta"]].dropna()
    scatter = ax.scatter(valid["pc1"], valid["pc2"], c=valid["theta"], cmap="viridis", alpha=0.7, s=50)
    plt.colorbar(scatter, ax=ax, label="theta (IRT ability)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PC1 vs PC2 (colored by theta)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "pc2_scatter.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {output_path}")


def write_summary(
    merged: pd.DataFrame,
    correlations: pd.DataFrame,
    reasoning_test: dict,
    output_dir: Path,
):
    """Write analysis summary to markdown."""
    summary_path = output_dir / "analysis_summary.md"

    with open(summary_path, "w") as f:
        f.write("# PC2 Analysis Summary\n\n")

        f.write("## Key Findings\n\n")

        # Find strongest correlation
        best = correlations.sort_values("spearman_r", key=abs, ascending=False).iloc[0]
        f.write(f"**Strongest correlation with PC2**: {best['feature']}\n")
        f.write(f"- Spearman r = {best['spearman_r']:.3f} (p = {best['spearman_p']:.2e})\n")
        f.write(f"- Pearson r = {best['pearson_r']:.3f} (p = {best['pearson_p']:.2e})\n\n")

        # Reasoning model comparison
        f.write("## Reasoning vs Standard Models\n\n")
        if "error" in reasoning_test:
            f.write(f"Error: {reasoning_test['error']}\n\n")
        else:
            f.write(f"- Reasoning models (n={reasoning_test['n_reasoning']}): ")
            f.write(f"PC2 mean = {reasoning_test['reasoning_pc2_mean']:.3f} ")
            f.write(f"(std = {reasoning_test['reasoning_pc2_std']:.3f})\n")
            f.write(f"- Standard models (n={reasoning_test['n_standard']}): ")
            f.write(f"PC2 mean = {reasoning_test['standard_pc2_mean']:.3f} ")
            f.write(f"(std = {reasoning_test['standard_pc2_std']:.3f})\n")
            f.write(f"- Mann-Whitney U test: p = {reasoning_test['u_pvalue']:.3e}\n\n")

        # Full correlation table
        f.write("## All Correlations\n\n")
        f.write("| Feature | n | Spearman r | Spearman p | Pearson r | Pearson p |\n")
        f.write("|---------|---|------------|------------|-----------|----------|\n")
        for _, row in correlations.iterrows():
            f.write(f"| {row['feature']} | {row['n']} | {row['spearman_r']:.3f} | ")
            f.write(f"{row['spearman_p']:.2e} | {row['pearson_r']:.3f} | {row['pearson_p']:.2e} |\n")

        f.write("\n## Data Coverage\n\n")
        f.write(f"- Total models in PC2 data: {len(merged)}\n")
        f.write(f"- Models with char_counts: {merged['model'].notna().sum()}\n")
        f.write(f"- Reasoning models: {merged['is_reasoning'].sum()}\n")

    print(f"Saved: {summary_path}")


def main():
    """Run the full PC2 analysis."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("=" * 60)
    print("Loading data")
    print("=" * 60)
    pc2_df = load_pc2_data()
    char_counts = load_char_counts()

    # Map agents to models
    print("\n" + "=" * 60)
    print("Mapping agents to models")
    print("=" * 60)
    char_counts = map_agents_to_models(char_counts)

    # Compute model features
    print("\n" + "=" * 60)
    print("Computing model-level features")
    print("=" * 60)
    model_features = compute_model_features(char_counts)

    # Merge with PC2
    print("\n" + "=" * 60)
    print("Merging with PC2 data")
    print("=" * 60)
    merged = merge_with_pc2(model_features, pc2_df)

    # Compute correlations
    print("\n" + "=" * 60)
    print("Computing correlations")
    print("=" * 60)
    correlations = compute_correlations(merged)
    print("\nCorrelation results:")
    print(correlations.sort_values("spearman_r", key=abs, ascending=False).to_string(index=False))

    # Test reasoning hypothesis
    print("\n" + "=" * 60)
    print("Testing reasoning hypothesis")
    print("=" * 60)
    reasoning_test = test_reasoning_hypothesis(merged)
    for k, v in reasoning_test.items():
        print(f"  {k}: {v}")

    # Generate plots
    print("\n" + "=" * 60)
    print("Generating visualizations")
    print("=" * 60)
    plot_correlations(merged, correlations, OUTPUT_DIR)

    # Save outputs
    merged.to_csv(OUTPUT_DIR / "model_features.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'model_features.csv'}")

    correlations.to_csv(OUTPUT_DIR / "correlation_results.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'correlation_results.csv'}")

    # Write summary
    write_summary(merged, correlations, reasoning_test, OUTPUT_DIR)

    # Print key finding
    print("\n" + "=" * 60)
    print("KEY FINDING")
    print("=" * 60)
    best = correlations.sort_values("spearman_r", key=abs, ascending=False).iloc[0]
    print(f"Strongest PC2 correlation: {best['feature']} (r = {best['spearman_r']:.3f})")


if __name__ == "__main__":
    main()

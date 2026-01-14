"""Script for producing some basic summary statistics on missingness and IRT visualizations."""

from pathlib import Path
import json
import pandas as pd
from datasets import load_dataset
import matplotlib.pyplot as plt
from pypalettes import load_cmap
import scipy.stats as stats

OUTPUT_DIR = Path("chris_output/tables/")
OUTPUT_VAL = Path("chris_output/values/")
OUTPUT_FIG = Path("chris_output/figures/")
DATA_PATH = Path("clean_data/swebench_verified/swebench_verified.jsonl")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_VAL.mkdir(parents=True, exist_ok=True)
OUTPUT_FIG.mkdir(parents=True, exist_ok=True)


def load_irt_data(filepath):
    """Load and reshape JSONL data for IRT analysis."""
    data_list = []
    with open(filepath, 'r') as f:
        for line in f:
            record = json.loads(line)
            row = {'subject_id': record['subject_id']}
            row.update(record['responses'])
            data_list.append(row)

    df = pd.DataFrame(data_list)
    item_columns = [col for col in df.columns if col != 'subject_id']
    return df, item_columns


def get_category_colors():
    """Create color mapping for MMLU-Pro categories."""
    category_order = ['math', 'physics', 'computer science', 'engineering', 'chemistry',
                     'economics', 'biology', 'health', 'psychology', 'business',
                     'law', 'history', 'philosophy', 'other']

    # Create viridis colormap for all categories except 'other'
    viridis = plt.get_cmap('viridis')
    viridis_colors = [viridis(i / (len(category_order) - 2)) for i in range(len(category_order) - 1)]

    # Add grey for 'other'
    all_colors = viridis_colors + ['grey']

    return dict(zip(category_order, all_colors))


def create_irt_analysis_df(item_columns, irt_results_path):
    """
    Merge IRT results with MMLU-Pro dataset metadata.

    Note: train.py saves 'a' as discrimination and 'b' as difficulty.
    """
    results = pd.read_csv(irt_results_path)

    # Create analysis dataframe - note correct column names from train.py
    analysis_df = pd.DataFrame({
        "question_id": item_columns,
        "irt_disc": results.loc[:, "a"],  # a = discrimination
        "irt_diff": results.loc[:, "b"]   # b = difficulty
    })

    # Load MMLU-Pro metadata
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    question_df = dataset.to_pandas()

    analysis_df['question_id'] = analysis_df['question_id'].astype(int)
    merged_df = analysis_df.merge(question_df, on='question_id', how='left')

    # Clean question text and add length
    merged_df["question"] = merged_df["question"].fillna("")
    merged_df["question_length"] = [len(x) for x in merged_df["question"]]

    return merged_df


def plot_stat_by_category(merged_df, stat, colors):
    """
    Create bar plot and histograms for a given IRT statistic by category.

    Args:
        merged_df: DataFrame with IRT results and category metadata
        stat: Either 'disc' (discrimination) or 'diff' (difficulty)
        colors: Dictionary mapping category names to colors
    """
    # Group by category and calculate mean/std
    cat_grouped = merged_df.groupby("category").agg(
        mean=(f"irt_{stat}", "mean"),
        sd=(f"irt_{stat}", "std")
    )
    cat_ordered = cat_grouped.sort_values(by="mean").reset_index()

    bar_colors = [colors[x] for x in cat_ordered["category"]]

    # Create bar plot
    fig, ax = plt.subplots()
    ax.barh(cat_ordered["category"], cat_ordered["mean"], color=bar_colors)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG / f"{stat}_bar.png")
    plt.close()

    # Create histograms
    cat_list = cat_ordered["category"]
    fig, axes = plt.subplots(ncols=4, nrows=4, figsize=(16, 16))
    axes = axes.flatten()

    for i, axs in enumerate(axes):
        if i < len(cat_list):
            filtered_df = merged_df[merged_df["category"] == cat_list[i]]
            axs.hist(filtered_df[f"irt_{stat}"], edgecolor="0", color=colors[cat_list[i]])
            axs.set_title(cat_list[i].capitalize())
        else:
            axs.axis('off')  # Hide unused subplots

    # Center the last 2 plots in the bottom row
    for i in range(12, 14):  # positions 12 and 13 (last row)
        axes[i].remove()

    # Add centered subplots for the last 2 categories
    for i, cat_idx in enumerate([12, 13]):
        ax = fig.add_subplot(4, 4, cat_idx + 2)  # positions 14 and 15 (centered)
        filtered_df = merged_df[merged_df["category"] == cat_list[cat_idx]]
        ax.hist(filtered_df[f"irt_{stat}"], edgecolor="0", color=colors[cat_list[cat_idx]])
        ax.set_title(cat_list[cat_idx].capitalize())

    plt.tight_layout()
    plt.savefig(OUTPUT_FIG / f"{stat}_hists.png")
    plt.close()


def plot_disc_excluding_low(merged_df, colors, excluded_val=0.01):
    """
    Create histograms for discrimination excluding very low values.

    Args:
        merged_df: DataFrame with IRT results and category metadata
        colors: Dictionary mapping category names to colors
        excluded_val: Threshold below which to exclude discrimination values
    """
    stat = "disc"
    filt = merged_df[merged_df[f"irt_{stat}"] > excluded_val]

    cat_grouped = filt.groupby("category").agg(mean=(f"irt_{stat}", "mean"))
    cat_ordered = cat_grouped.sort_values(by="mean").reset_index()
    cat_list = cat_ordered["category"]

    _fig, axes = plt.subplots(ncols=2, nrows=7, figsize=(10, 20))
    axes = axes.flatten()

    for i, axs in enumerate(axes):
        filtered_df = filt[filt["category"] == cat_list[i]]
        axs.hist(filtered_df[f"irt_{stat}"], edgecolor="0", color=colors[cat_list[i]])
        axs.set_title(cat_list[i].capitalize())

    plt.tight_layout()
    plt.savefig(OUTPUT_FIG / f"{stat}_hists_excl.png")
    plt.close()


def run_bartlett_test(merged_df):
    """
    Run Bartlett's test for homogeneity of variance across categories.

    Args:
        merged_df: DataFrame with IRT results and category metadata

    Returns:
        p_value: P-value from Bartlett's test
    """
    all_cats = merged_df["category"].unique()

    cat_qns = {}
    for cat in all_cats:
        cat_qns[cat] = merged_df[merged_df["category"] == cat]["irt_diff"]
        if len(cat_qns[cat]) == 0:
            del cat_qns[cat]

    scores = list(cat_qns.values())
    _stat, p_value = stats.bartlett(*scores)

    return p_value


def create_irt_visualizations(item_columns, irt_results_path):
    """
    Create all IRT visualizations and run statistical tests.

    Args:
        item_columns: List of item (question) column names
        irt_results_path: Path to the items.csv file from train.py 1D IRT output
    """
    # Create merged analysis dataframe
    merged_df = create_irt_analysis_df(item_columns, irt_results_path)

    # Get category colors
    colors = get_category_colors()

    # Create plots for both discrimination and difficulty
    for stat in ["disc", "diff"]:
        plot_stat_by_category(merged_df, stat, colors)

    # Create supplementary discrimination analysis
    plot_disc_excluding_low(merged_df, colors)

    # Run Bartlett's test for variance homogeneity
    p_value = run_bartlett_test(merged_df)
    print(f"Bartlett's test p-value: {p_value}")

    print(f"IRT visualizations saved to {OUTPUT_FIG}")


def write_summary_stats(data, item_columns):

    # Intermediary dataset to merge in to model_check
    na_items_temp = (data.loc[:,item_columns] 
        .isna()
        .sum(axis = 1)
        .reset_index()
        .rename(columns = {0: "na_items"})
    )

    # Output dataset for how many qns missed per model
    model_check = (data.loc[:,["subject_id"]] 
        .reset_index()
        .merge(na_items_temp, on = "index")
        .loc[:, ["subject_id", "na_items"]]
    )

    model_check.to_csv(OUTPUT_DIR / "mi_per_model.csv", index = False)

    # Output dataset for how many models didn't answer per qn
    mi_check = (data.loc[:,item_columns] 
        .isna()
        .sum(axis = 0)
        .reset_index(name = "na_ct")
        .rename(columns = {"index": "question_id"})
        .loc[lambda d: d["na_ct"] > 0]
    )

    mi_check.to_csv(OUTPUT_DIR / "mi_per_item.csv", index = False)

    # Output dataset for the number of qns missing x number of models
    # - We find that most questions either only have one model missing it
    #   or nearly all models missing it
    ct_type = (mi_check.loc[:,["na_ct"]] 
                .groupby(["na_ct"])
                .agg(counts = ("na_ct", "count"))
                .reset_index(names = "na_ct")
    )

    ct_type.to_csv(OUTPUT_DIR / "mi_extent", index = False)

    # Summary statistics
    model_count = len(data.index)
    print(f"Models Evaluated: {model_count}")

    qn_count = len(item_columns)
    print(f"Questions Asked: {qn_count}")

    qn_mi_count = len(mi_check.index)
    print(f"  - All models answered: {qn_count - qn_mi_count}")
    print(f"  - Not all models answered: {qn_mi_count}") # It appears that some models simply weren't asked all the questions

    with open(OUTPUT_VAL / "stats.tex", "w") as f:
        f.write(f"\\newcommand{{\\modelct}}{{{model_count}}}\n")
        f.write(f"\\newcommand{{\\qnct}}{{{qn_count}}}\n")
        f.write(f"\\newcommand{{\\qnmict}}{{{qn_mi_count}}}\n")
        f.write(f"\\newcommand{{\\qnallct}}{{{qn_count - qn_mi_count}}}\n")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate summary statistics and IRT visualizations'
    )
    parser.add_argument(
        '--stat',
        type=str,
        choices=['summary', 'visuals'],
        help='Run only summary stats or visuals (default: run both)'
    )
    parser.add_argument(
        '--irt_results',
        type=str,
        default='clean_data/training_results_rep/1d/items.csv',
        help='Path to IRT results items.csv file'
    )
    args = parser.parse_args()

    # Load response data
    data, item_columns = load_irt_data("clean_data/swebench_verified/swebench_verified.jsonl")

    # Run summary stats if requested or by default
    if args.stat is None or args.stat == 'summary':
        write_summary_stats(data, item_columns)

    # Run visualizations if requested or by default
    if args.stat is None or args.stat == 'visuals':
        create_irt_visualizations(item_columns, args.irt_results)

if __name__ == "__main__":
    main()
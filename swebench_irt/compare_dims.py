#!/usr/bin/env python3
# %%
"""
Compare 1D vs 2D vs 3D IRT fits using log-likelihood, AIC, and BIC.

Reads parameters from a results directory (e.g., clean_data/.../{1d,2d,3d}/) and
responses from a JSONL file, then prints a compact table and saves it to
chris_output/figures/model_selection.csv by default.
"""

from pathlib import Path
import json
from typing import Dict, Iterable, List, Tuple
from scipy.special import expit as sigmoid_stable
import argparse

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

def resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else (ROOT / path)

DEFAULT_RESULTS_DIR = ROOT / "clean_data" / "training_results"
DEFAULT_RESPONSES_PATH = ROOT / "clean_data" / "mmlu_data" / "model_response_correctness.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "figures"

def overall_skill(abilities_df: pd.DataFrame, items_df: pd.DataFrame, dims: int, D: float = 1.0) -> pd.Series:
    '''
    This function converts the IRT ability and item parameters to return the average 
    probability of a model getting the each question right.
    '''
    if dims == 1:
        theta = abilities_df[["theta"]].values  # [S,1]
        A = items_df[["a"]].values             # [I,1]
        B = items_df[["b"]].values             # [I,1]
    else:
        theta = abilities_df[[f"theta{d}" for d in range(1, dims + 1)]].values  # [S,D]
        A = items_df[[f"a{d}" for d in range(1, dims + 1)]].values              # [I,D]
        B = items_df[[f"b{d}" for d in range(1, dims + 1)]].values              # [I,D]
    # logits per (agent, item): sum_d A * (theta - B)
    z = (A[None, :, :] * (theta[:, None, :] - B[None, :, :])).sum(axis=2)  # [S,I] Obtain z-scores
    P = sigmoid_stable(D * z) # Get the probability of each model getting each question correct
    mean_p = P.mean(axis=1)  # [S] Get the average probability of a model getting the questions right
    return pd.Series(mean_p, index=abilities_df.index)

def load_model(results_dir: Path, dims: int):
    '''
    This functions simply reads in the abilities and items 
    saved when originally fitting the IRT models
    '''
    dim_dir = results_dir / f"{dims}d"
    abilities = pd.read_csv(dim_dir / "abilities.csv", index_col=0)
    items = pd.read_csv(dim_dir / "items.csv", index_col=0)
    return abilities, items

def iter_obs(jsonl_path: Path, subjects: Iterable[str], items: Iterable[str]):
    '''
    This function reads in the particular # TODO: understand this
    '''
    subjects = {str(s) for s in subjects}
    items = {str(i) for i in items}
    with open(jsonl_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            sid = rec.get("subject_id")
            if sid is None:
                continue
            sid = str(sid)
            if sid not in subjects:
                continue
            for iid_str, y in rec.get("responses", {}).items():
                iid = str(iid_str)
                if iid in items and y in (0, 1):
                    yield sid, iid, int(y)

# %%

# abil1, it1 = load_model(RESULTS_DIR, 1)
# jsonl_path = RESPONSES_PATH

# subjects = abil1.index
# print(subjects)
# items = it1.index
# # %%

# with open(jsonl_path, "r") as f:
#     for line in f:
#         if not line.strip():
#             print("not linestrip")
#             continue
#         rec = json.loads(line)
#         sid = rec.get("subject_id")
#         if sid not in subjects:
#             print("not subj")
#             continue
#         for iid, y in rec.get("responses", {}).items():
#             print(iid)
#             if iid in items and y in (0, 1):
#                 print(sid)
#                 print(iid)
#                 print(int(y))


# %%

def log_bernoulli_logits(y: np.ndarray, z: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    return -(y * np.logaddexp(0.0, -z) + (1.0 - y) * np.logaddexp(0.0, z))


def compute_ll(abilities: pd.DataFrame, items: pd.DataFrame, dims: int, jsonl: Path):
    if dims == 1:
        abil = {str(i): float(v) for i, v in abilities['theta'].items()}
        a = {str(i): float(v) for i, v in items['a'].items()}
        b = {str(i): float(v) for i, v in items['b'].items()}
    else:
        abil_cols = [f'theta{d}' for d in range(1, dims+1)]
        a_cols = [f'a{d}' for d in range(1, dims+1)]
        b_cols = [f'b{d}' for d in range(1, dims+1)]
        abil = {str(i): row[abil_cols].values.astype(float) for i, row in abilities[abil_cols].iterrows()}
        a = {str(i): row[a_cols].values.astype(float) for i, row in items[a_cols].iterrows()}
        b = {str(i): row[b_cols].values.astype(float) for i, row in items[b_cols].iterrows()}

    ys: List[int] = []
    zs: List[float] = []
    n = 0
    for sid, iid, y in iter_obs(jsonl, abilities.index, items.index):
        if sid not in abil or iid not in a:
            continue
        if dims == 1:
            z = float(a[iid]) * (float(abil[sid]) - float(b[iid]))
        else:
            z = float(np.dot(a[iid], (np.array(abil[sid]) - np.array(b[iid]))))
        ys.append(y); zs.append(z); n += 1
    if n == 0:
        return float('-inf'), 0
    ll = float(log_bernoulli_logits(np.array(ys), np.array(zs)).sum())
    return ll, n


def n_params(n_agents: int, n_items: int, dims: int) -> int:
    return dims * n_agents + 2 * dims * n_items


def aic_bic(ll: float, k: int, n: int):
    aic = -2.0 * ll + 2.0 * k
    bic = -2.0 * ll + np.log(max(n, 1)) * k
    return aic, bic


# ll1, n1 = compute_ll(abil1, it1, 1, RESPONSES_PATH)
# # %%
# ll2, n2 = compute_ll(abil2, it2, 2, RESPONSES_PATH)
# ll3, n3 = compute_ll(abil3, it3, 3, RESPONSES_PATH)
# n_obs = min(n1, n2, n3)

# k1 = n_params(len(abil1), len(it1), 1)
# k2 = n_params(len(abil1), len(it1), 2)
# k3 = n_params(len(abil1), len(it1), 3)

# a1, b1 = aic_bic(ll1, k1, n_obs)
# a2, b2 = aic_bic(ll2, k2, n_obs)
# a3, b3 = aic_bic(ll3, k3, n_obs)

# rows = [
#     ("1D", ll1, k1, n_obs, a1, b1),
#     ("2D", ll2, k2, n_obs, a2, b2),
#     ("3D", ll3, k3, n_obs, a3, b3),
# ]
# df = pd.DataFrame(rows, columns=["model", "loglik", "n_params", "n_obs", "AIC", "BIC"])
# print(df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

# best_aic = df.sort_values("AIC").iloc[0]
# best_bic = df.sort_values("BIC").iloc[0]
# print(f"\nBest by AIC: {best_aic['model']} (AIC={best_aic['AIC']:.2f})")
# print(f"Best by BIC: {best_bic['model']} (BIC={best_bic['BIC']:.2f})")

# (OUTPUT_DIR / "model_selection.csv").write_text(df.to_csv(index=False))
# print(f"Saved table to {OUTPUT_DIR / 'model_selection.csv'}")

# # Optional: 2D MIRT scatter plot if theta1/theta2 available
# def plot_mirt_2d(abilities_df: pd.DataFrame, out_path: Path, top_n: int = 10):
#     if not {"theta1", "theta2"}.issubset(abilities_df.columns):
#         return
#     df2 = abilities_df.copy()
#     if "magnitude" not in df2.columns:
#         df2["magnitude"] = np.sqrt(df2["theta1"] ** 2 + df2["theta2"] ** 2)

#     fig, ax = plt.subplots(figsize=(10, 8))
#     sc = ax.scatter(df2["theta1"], df2["theta2"], c=df2["magnitude"], cmap="viridis", alpha=0.6, s=35, edgecolors="black", linewidths=0.3)
#     cbar = plt.colorbar(sc, ax=ax)
#     cbar.set_label("Ability Magnitude", rotation=270, labelpad=14)

#     top_agents = df2.nlargest(top_n, "magnitude")
#     for idx, (agent, row) in enumerate(top_agents.iterrows(), 1):
#         ax.annotate(str(idx), (row["theta1"], row["theta2"]), xytext=(4, 4), textcoords="offset points", fontsize=8, bbox=dict(boxstyle="circle,pad=0.2", facecolor="yellow", alpha=0.7, edgecolor="black"))

#     ax.grid(True, alpha=0.3, linestyle="--")
#     ax.axhline(0, color="gray", linestyle="-", linewidth=0.5, alpha=0.3)
#     ax.axvline(0, color="gray", linestyle="-", linewidth=0.5, alpha=0.3)
#     ax.set_xlabel("Dimension 1 (θ1)")
#     ax.set_ylabel("Dimension 2 (θ2)")
#     ax.set_title("Agent Abilities in 2D MIRT Space")
#     plt.tight_layout()
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     plt.savefig(out_path, dpi=300, bbox_inches="tight")
#     plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
#     plt.close()
#     print(f"Saved 2D MIRT scatter to: {out_path}")

# plot_mirt_2d(abil2, OUTPUT_DIR / "mirt_2d_scatter.png", top_n=10)


def main(
    dims_to_test: List[int],
    results_dir: Path,
    responses_path: Path,
    output_dir: Path,
    plot_2d: bool,
):
    """
    Compare IRT models across specified dimensions.
    
    Args:
        dims_to_test: List of dimensions to test (e.g., [1, 2, 3, 4])
    """
    if not responses_path.exists():
        raise SystemExit(f"Missing responses at {responses_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models for all specified dimensions
    models = {}
    for dim in dims_to_test:
        try:
            abil, items = load_model(results_dir, dim)
            models[dim] = (abil, items)
        except FileNotFoundError:
            print(f"Warning: {dim}D model not found, skipping...")
            continue
    
    if not models:
        raise SystemExit("No models found to compare")

    # Compute log-likelihood for each model
    results = {}
    for dim, (abil, items) in models.items():
        ll, n = compute_ll(abil, items, dim, responses_path)
        k = n_params(len(abil), len(items), dim)
        results[dim] = {'ll': ll, 'n': n, 'k': k, 'abil': abil, 'items': items}
    
    # Use minimum n_obs across all models
    n_obs = min(r['n'] for r in results.values())
    
    # Calculate AIC and BIC
    rows = []
    for dim in sorted(results.keys()):
        ll = results[dim]['ll']
        k = results[dim]['k']
        aic, bic = aic_bic(ll, k, n_obs)
        rows.append((f"{dim}D", ll, k, n_obs, aic, bic))
    
    df = pd.DataFrame(rows, columns=["model", "loglik", "n_params", "n_obs", "AIC", "BIC"])
    print(df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    best_aic = df.sort_values("AIC").iloc[0]
    best_bic = df.sort_values("BIC").iloc[0]
    print(f"\nBest by AIC: {best_aic['model']} (AIC={best_aic['AIC']:.2f})")
    print(f"Best by BIC: {best_bic['model']} (BIC={best_bic['BIC']:.2f})")

    (output_dir / "model_selection.csv").write_text(df.to_csv(index=False))
    print(f"Saved table to {output_dir / 'model_selection.csv'}")

    # Optional: 2D MIRT scatter plot if theta1/theta2 available
    def plot_mirt_2d(abilities_df: pd.DataFrame, out_path: Path, top_n: int = 10):
        import matplotlib.pyplot as plt

        if not {"theta1", "theta2"}.issubset(abilities_df.columns):
            return
        df2 = abilities_df.copy()
        if "magnitude" not in df2.columns:
            df2["magnitude"] = np.sqrt(df2["theta1"] ** 2 + df2["theta2"] ** 2)

        fig, ax = plt.subplots(figsize=(10, 8))
        sc = ax.scatter(df2["theta1"], df2["theta2"], c=df2["magnitude"], cmap="viridis", alpha=0.6, s=35, edgecolors="black", linewidths=0.3)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Ability Magnitude", rotation=270, labelpad=14)

        top_agents = df2.nlargest(top_n, "magnitude")
        for idx, (agent, row) in enumerate(top_agents.iterrows(), 1):
            ax.annotate(str(idx), (row["theta1"], row["theta2"]), xytext=(4, 4), textcoords="offset points", fontsize=8, bbox=dict(boxstyle="circle,pad=0.2", facecolor="yellow", alpha=0.7, edgecolor="black"))

        ax.grid(True, alpha=0.3, linestyle="--")
        ax.axhline(0, color="gray", linestyle="-", linewidth=0.5, alpha=0.3)
        ax.axvline(0, color="gray", linestyle="-", linewidth=0.5, alpha=0.3)
        ax.set_xlabel("Dimension 1 (θ1)")
        ax.set_ylabel("Dimension 2 (θ2)")
        ax.set_title("Agent Abilities in 2D MIRT Space")
        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close()
        print(f"Saved 2D MIRT scatter to: {out_path}")

    # Plot 2D if available
    if plot_2d and 2 in results:
        plot_mirt_2d(results[2]['abil'], output_dir / "mirt_2d_scatter.png", top_n=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare IRT models across dimensions")
    parser.add_argument("--dims", type=int, nargs="*", default=[1, 2, 3],
        help="Dims to compare (default: 1 2 3)")
    parser.add_argument("--results_dir", type=str, default=str(DEFAULT_RESULTS_DIR),
        help="Directory containing <dims>d/ results")
    parser.add_argument("--responses_path", type=str, default=str(DEFAULT_RESPONSES_PATH),
        help="Path to JSONL responses")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save AIC/BIC table and plots")
    parser.add_argument("--plot_2d", action="store_true",
        help="Generate 2D scatter plot (requires matplotlib)")
    args = parser.parse_args()

    main(
        dims_to_test=args.dims,
        results_dir=resolve_path(args.results_dir),
        responses_path=resolve_path(args.responses_path),
        output_dir=resolve_path(args.output_dir),
        plot_2d=args.plot_2d,
    )

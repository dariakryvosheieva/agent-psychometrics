"""Analyze hyperparameter ablation results.

Usage:
    python -m experiment_a.mlp_ablation.analyze_ablation_results
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent


def main():
    output_dir = ROOT / "chris_output/experiment_a/mlp_ablation"

    # Load results from both GPUs
    all_results = {}
    for gpu in [0, 1]:
        path = output_dir / f"hyperparameter_ablation_gpu{gpu}.json"
        if path.exists():
            with open(path) as f:
                results = json.load(f)
                all_results.update(results)
            print(f"Loaded {len(results)} results from GPU {gpu}")
        else:
            print(f"Warning: {path} not found")

    if not all_results:
        print("No results found!")
        return

    print(f"\nTotal configurations: {len(all_results)}")

    # Sort by test AUC
    sorted_results = sorted(
        all_results.items(),
        key=lambda x: x[1]["mean_auc"],
        reverse=True
    )

    # Print all results sorted by AUC
    print("\n" + "=" * 100)
    print("ALL RESULTS (sorted by Test AUC)")
    print("=" * 100)
    print(f"\n{'Rank':<5} {'Method':<55} {'Test AUC':>10} {'Train AUC':>12} {'Gap':>10}")
    print("-" * 100)

    for rank, (name, r) in enumerate(sorted_results, 1):
        test_auc = r["mean_auc"]
        train_auc = r.get("train_auc")
        std_auc = r.get("std_auc", 0)
        display_name = r["display_name"][:53]

        if train_auc is not None:
            gap = train_auc - test_auc
            print(f"{rank:<5} {display_name:<55} {test_auc:>10.4f} {train_auc:>12.4f} {gap:>+10.4f}")
        else:
            print(f"{rank:<5} {display_name:<55} {test_auc:>10.4f} {'N/A':>12} {'N/A':>10}")

    # Separate random init vs IRT init results
    random_results = [(n, r) for n, r in sorted_results if "random" in n.lower() or (not r["display_name"].startswith("IRT") and "irt" not in n.lower())]
    irt_results = [(n, r) for n, r in sorted_results if "irt" in n.lower() and "random" not in n.lower()]

    # Find matching pairs (same config, different init)
    print("\n" + "=" * 100)
    print("RANDOM vs IRT INIT COMPARISON")
    print("=" * 100)

    # Group by base config name (strip _random/_irt suffix)
    pairs = {}
    for name, r in all_results.items():
        if name in ["oracle", "ridge"]:
            continue

        # Determine base name and init type
        if name.endswith("_random"):
            base = name[:-7]
            init_type = "random"
        elif name.endswith("_irt"):
            base = name[:-4]
            init_type = "irt"
        elif "random" in name:
            base = name.replace("_random", "").replace("random_", "")
            init_type = "random"
        elif "irt" in name.lower():
            base = name.replace("_irt", "").replace("irt_", "")
            init_type = "irt"
        else:
            continue

        if base not in pairs:
            pairs[base] = {}
        pairs[base][init_type] = r

    # Show pairs with both random and IRT
    print(f"\n{'Config':<45} {'Random AUC':>12} {'IRT AUC':>12} {'Diff':>10}")
    print("-" * 85)

    diffs = []
    for base, inits in sorted(pairs.items()):
        if "random" in inits and "irt" in inits:
            rand_auc = inits["random"]["mean_auc"]
            irt_auc = inits["irt"]["mean_auc"]
            diff = rand_auc - irt_auc
            diffs.append(diff)
            print(f"{base[:43]:<45} {rand_auc:>12.4f} {irt_auc:>12.4f} {diff:>+10.4f}")

    if diffs:
        print("-" * 85)
        print(f"{'Average difference (random - IRT)':<45} {'':<12} {'':<12} {sum(diffs)/len(diffs):>+10.4f}")

    # Top 10 random init configs
    print("\n" + "=" * 100)
    print("TOP 10 RANDOM INIT CONFIGS")
    print("=" * 100)

    random_only = [(n, r) for n, r in sorted_results
                   if "random" in n.lower() or
                   (n.endswith("_random") or "_random_" in n)]

    print(f"\n{'Rank':<5} {'Method':<55} {'Test AUC':>10} {'Std':>8}")
    print("-" * 85)
    for rank, (name, r) in enumerate(random_only[:10], 1):
        print(f"{rank:<5} {r['display_name'][:53]:<55} {r['mean_auc']:>10.4f} {r.get('std_auc', 0):>8.4f}")

    # Top 10 IRT init configs
    print("\n" + "=" * 100)
    print("TOP 10 IRT INIT CONFIGS")
    print("=" * 100)

    irt_only = [(n, r) for n, r in sorted_results
                if ("irt" in n.lower() and "random" not in n.lower()) or n.endswith("_irt")]

    print(f"\n{'Rank':<5} {'Method':<55} {'Test AUC':>10} {'Std':>8}")
    print("-" * 85)
    for rank, (name, r) in enumerate(irt_only[:10], 1):
        print(f"{rank:<5} {r['display_name'][:53]:<55} {r['mean_auc']:>10.4f} {r.get('std_auc', 0):>8.4f}")

    # Baselines
    print("\n" + "=" * 100)
    print("BASELINES")
    print("=" * 100)
    if "oracle" in all_results:
        print(f"Oracle:  {all_results['oracle']['mean_auc']:.4f}")
    if "ridge" in all_results:
        print(f"Ridge:   {all_results['ridge']['mean_auc']:.4f}")

    # Key findings
    print("\n" + "=" * 100)
    print("KEY FINDINGS")
    print("=" * 100)

    if random_only:
        best_random = random_only[0]
        print(f"\nBest random init: {best_random[1]['mean_auc']:.4f} ({best_random[1]['display_name']})")

    if irt_only:
        best_irt = irt_only[0]
        print(f"Best IRT init:    {best_irt[1]['mean_auc']:.4f} ({best_irt[1]['display_name']})")

    if random_only and irt_only:
        gap = best_irt[1]['mean_auc'] - best_random[0][1]['mean_auc']
        print(f"Gap (IRT - random): {gap:+.4f}")

    if "ridge" in all_results and random_only:
        ridge_auc = all_results['ridge']['mean_auc']
        best_rand_auc = random_only[0][1]['mean_auc']
        print(f"\nRidge baseline:   {ridge_auc:.4f}")
        print(f"Best random vs Ridge: {best_rand_auc - ridge_auc:+.4f}")


if __name__ == "__main__":
    main()

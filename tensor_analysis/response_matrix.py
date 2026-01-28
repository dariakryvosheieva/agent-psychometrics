"""PCA analysis on response matrices for all datasets.

Loads response matrices (agents x tasks -> 0/1) and runs PCA to analyze
the structure of agent-task interactions.

Datasets analyzed:
- SWE-bench Verified: 131 agents x 500 tasks
- SWE-bench Pro: 14 agents x 730 tasks
- TerminalBench: 83 agents x 88 tasks
- GSO: 14 agents x 102 tasks
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def load_binary_responses(responses_path: Path) -> Dict[str, Dict[str, int]]:
    """Load binary response matrix from JSONL.

    Format: {"subject_id": "agent_name", "responses": {"task_id": 0/1, ...}}
    """
    responses = {}
    with open(responses_path, "r") as f:
        for line in f:
            record = json.loads(line)
            agent_id = record["subject_id"]
            responses[agent_id] = record["responses"]
    return responses


def responses_to_matrix(
    responses: Dict[str, Dict[str, int]],
    task_ids: List[str] = None,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Convert response dict to numpy matrix.

    Args:
        responses: {agent_id: {task_id: 0/1, ...}}
        task_ids: Optional list of task_ids to include (for filtering)

    Returns:
        Tuple of (matrix, agent_ids, task_ids) where matrix is (n_agents x n_tasks)

    Raises:
        ValueError: If any agent is missing a response for any task
    """
    agent_ids = sorted(responses.keys())

    if task_ids is None:
        # Get all tasks from all agents
        all_tasks = set()
        for agent_responses in responses.values():
            all_tasks.update(agent_responses.keys())
        task_ids = sorted(all_tasks)

    matrix = np.zeros((len(agent_ids), len(task_ids)))

    for i, agent_id in enumerate(agent_ids):
        agent_resp = responses[agent_id]
        for j, task_id in enumerate(task_ids):
            if task_id not in agent_resp:
                raise ValueError(
                    f"Missing response for agent '{agent_id}' on task '{task_id}'. "
                    f"Response matrices must be complete."
                )
            matrix[i, j] = agent_resp[task_id]

    return matrix, agent_ids, task_ids


def run_pca_analysis(
    matrix: np.ndarray,
    agent_ids: List[str],
    n_components: int = 10,
    center: bool = True,
) -> Dict:
    """Run PCA on response matrix.

    Args:
        matrix: (n_agents x n_tasks) response matrix
        agent_ids: List of agent identifiers
        n_components: Number of PCA components
        center: Whether to center the data (subtract column means)

    Returns:
        Dict with PCA results
    """
    # Center the data (subtract column means)
    if center:
        matrix_centered = matrix - np.mean(matrix, axis=0)
    else:
        matrix_centered = matrix

    # Fit PCA
    n_components = min(n_components, min(matrix_centered.shape) - 1)
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(matrix_centered)

    # Build results
    results = {
        "n_agents": matrix.shape[0],
        "n_tasks": matrix.shape[1],
        "n_components": n_components,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
        "agent_scores": {
            agent_id: transformed[i, :].tolist()
            for i, agent_id in enumerate(agent_ids)
        },
    }

    return results


def plot_pca_summary(
    results: Dict,
    dataset_name: str,
    output_path: Path,
) -> None:
    """Plot PCA summary: explained variance and PC1 vs PC2."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Explained variance
    ax = axes[0]
    var_ratio = results["explained_variance_ratio"]
    cum_var = results["cumulative_variance"]
    x = range(1, len(var_ratio) + 1)

    ax.bar(x, var_ratio, alpha=0.7, label="Individual")
    ax.plot(x, cum_var, "r-o", label="Cumulative")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title(f"{dataset_name}: Explained Variance")
    ax.legend()
    ax.set_xticks(x)

    # PC1 vs PC2 scatter
    ax = axes[1]
    agent_scores = results["agent_scores"]

    pc1 = [scores[0] for scores in agent_scores.values()]
    pc2 = [scores[1] for scores in agent_scores.values()]

    ax.scatter(pc1, pc2, alpha=0.7)

    # Label a few points
    agents = list(agent_scores.keys())
    for i, agent in enumerate(agents):
        # Only label outliers (top/bottom 3 on PC1 or PC2)
        if i < 3 or i >= len(agents) - 3:
            ax.annotate(agent[:20], (pc1[i], pc2[i]), fontsize=6, alpha=0.7)

    ax.set_xlabel(f"PC1 ({var_ratio[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({var_ratio[1]:.1%} var)")
    ax.set_title(f"{dataset_name}: Agent Scores")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def analyze_dataset(
    name: str,
    responses_path: Path,
    output_dir: Path,
    n_components: int = 10,
) -> Dict:
    """Load dataset, run PCA, and generate plots."""
    print(f"\n{'='*60}")
    print(f"Analyzing {name}")
    print(f"{'='*60}")

    # Load responses
    responses = load_binary_responses(responses_path)
    print(f"Loaded {len(responses)} agents")

    # Convert to matrix
    matrix, agent_ids, task_ids = responses_to_matrix(responses)
    print(f"Matrix shape: {matrix.shape} (agents x tasks)")

    # Run PCA
    results = run_pca_analysis(matrix, agent_ids, n_components=n_components)
    results["name"] = name

    # Print summary
    var_ratio = results["explained_variance_ratio"]
    print(f"\nPCA Results:")
    print(f"  PC1: {var_ratio[0]:.1%} variance")
    print(f"  PC2: {var_ratio[1]:.1%} variance")
    print(f"  Top 5: {sum(var_ratio[:5]):.1%} cumulative variance")

    # Generate plot
    plot_path = output_dir / "plots" / f"{name.lower().replace(' ', '_').replace('-', '_')}_pca.png"
    plot_pca_summary(results, name, plot_path)

    return results


def main():
    """Run PCA analysis on all 4 datasets."""
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "chris_output" / "tensor_analysis"
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Dataset paths
    datasets = {
        "SWE-bench Verified": base_dir / "clean_data" / "swebench_verified" / "swebench_verified_20251120_full.jsonl",
        "SWE-bench Pro": base_dir / "out" / "chris_irt" / "swebench_pro.jsonl",
        "TerminalBench": base_dir / "data" / "terminal_bench" / "terminal_bench_2.0.jsonl",
        "GSO": base_dir / "out" / "chris_irt" / "gso.jsonl",
    }

    all_results = {}

    for name, path in datasets.items():
        if not path.exists():
            print(f"\nSkipping {name} - file not found: {path}")
            continue

        results = analyze_dataset(name, path, output_dir)
        all_results[name] = results

    # Save all results
    results_path = output_dir / "pca_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved all results to {results_path}")

    # Print comparison summary
    print("\n" + "=" * 60)
    print("Cross-Dataset Comparison")
    print("=" * 60)
    print(f"{'Dataset':<25} {'Agents':>8} {'Tasks':>8} {'PC1':>8} {'PC2':>8} {'PC1+PC2':>10}")
    print("-" * 70)
    for name, results in all_results.items():
        var = results["explained_variance_ratio"]
        print(f"{name:<25} {results['n_agents']:>8} {results['n_tasks']:>8} {var[0]:>7.1%} {var[1]:>7.1%} {var[0]+var[1]:>9.1%}")


if __name__ == "__main__":
    main()

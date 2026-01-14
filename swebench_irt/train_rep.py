"""
This script trains the IRT model multiple times and averages the results to reduce noise.
It works by running train.py in separate processes to ensure complete isolation between runs.

Call the function using the following syntax:
    python swebench_irt/train_rep.py --dims 1 2 3 --reps 5 --epochs 5000
    python swebench_irt/train_rep.py --dims 1 2 3 --reps 5 --epochs 5000 --data_path clean_data/swebench_verified/swebench_verified.jsonl
"""

from pathlib import Path
import numpy as np
import pandas as pd
import argparse
import subprocess
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]

def resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else (ROOT / path)

def average_1d_results(run_dirs, output_dir):
    """
    Average 1D IRT results across multiple runs.

    Args:
        run_dirs: List of directories containing individual run results
        output_dir: Directory to save averaged results
    """
    items_dfs = []
    abilities_dfs = []

    for run_dir in run_dirs:
        items_path = run_dir / "1d" / "items.csv"
        abilities_path = run_dir / "1d" / "abilities.csv"

        if items_path.exists() and abilities_path.exists():
            items_dfs.append(pd.read_csv(items_path, index_col=0))
            abilities_dfs.append(pd.read_csv(abilities_path, index_col=0))

    # Average items parameters
    items_combined = pd.concat(items_dfs)
    items_avg = items_combined.groupby(items_combined.index).mean()

    # Average abilities parameters
    abilities_combined = pd.concat(abilities_dfs)
    abilities_avg = abilities_combined.groupby(abilities_combined.index).mean()
    abilities_avg = abilities_avg.sort_values("theta", ascending=False)

    # Save averaged results
    out_dir = output_dir / "1d"
    out_dir.mkdir(parents=True, exist_ok=True)

    items_avg.to_csv(out_dir / "items.csv")
    abilities_avg.to_csv(out_dir / "abilities.csv")

    print(f"  Averaged 1D results saved to {out_dir}")


def average_md_results(run_dirs, dims, output_dir):
    """
    Average multidimensional IRT results across multiple runs.

    Args:
        run_dirs: List of directories containing individual run results
        dims: Number of dimensions
        output_dir: Directory to save averaged results
    """
    items_dfs = []
    abilities_dfs = []

    for run_dir in run_dirs:
        items_path = run_dir / f"{dims}d" / "items.csv"
        abilities_path = run_dir / f"{dims}d" / "abilities.csv"

        if items_path.exists() and abilities_path.exists():
            items_dfs.append(pd.read_csv(items_path, index_col=0))
            abilities_dfs.append(pd.read_csv(abilities_path, index_col=0))

    # Average items parameters
    items_combined = pd.concat(items_dfs)
    items_avg = items_combined.groupby(items_combined.index).mean()

    # Average abilities parameters
    abilities_combined = pd.concat(abilities_dfs)
    abilities_avg = abilities_combined.groupby(abilities_combined.index).mean()
    abilities_avg["theta_avg"] = abilities_avg[[f"theta{d+1}" for d in range(dims)]].mean(axis=1)
    abilities_avg = abilities_avg.sort_values("theta_avg", ascending=False)

    # Save averaged results
    out_dir = output_dir / f"{dims}d"
    out_dir.mkdir(parents=True, exist_ok=True)

    items_avg.to_csv(out_dir / "items.csv")
    abilities_avg.to_csv(out_dir / "abilities.csv")

    print(f"  Averaged {dims}D results saved to {out_dir}")


def run_training_repetitions(dims_list, reps, epochs, output_dir, data_path):
    """
    Run train.py multiple times for each dimensionality and average results.

    Args:
        dims_list: List of dimensions to train
        reps: Number of repetitions per dimension
        epochs: Number of epochs per run
        output_dir: Final output directory for averaged results
    """
    base_dir = ROOT / "clean_data"
    output_path = base_dir / output_dir
    train_script = Path(__file__).resolve().parent / "train.py"

    for dim in dims_list:
        print(f"\nTraining {dim}D model with {reps} repetitions...")
        run_dirs = []

        # Run train.py multiple times as separate processes
        for rep in range(reps):
            print(f"  Run {rep + 1}/{reps}")
            temp_output = f"{output_dir}_temp_run{rep}"

            # Call train.py as a subprocess - this ensures complete isolation
            cmd = [
                sys.executable, str(train_script),
                "--dims", str(dim),
                "--epochs", str(epochs),
                "--output_dir", temp_output
            ]
            if data_path is not None:
                cmd.extend(["--data_path", str(data_path)])

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"    Error in run {rep + 1}:")
                print(result.stderr)
                continue

            run_dirs.append(base_dir / temp_output)

        # Average the results
        if run_dirs:
            if dim == 1:
                average_1d_results(run_dirs, output_path)
            else:
                average_md_results(run_dirs, dim, output_path)

        # Clean up temporary directories
        for run_dir in run_dirs:
            if run_dir.exists():
                shutil.rmtree(run_dir)


def main():
    parser = argparse.ArgumentParser(description='Train IRT models with multiple repetitions')
    parser.add_argument('--dims', type=int, nargs='*', default=[1, 2, 3, 4, 5, 6],
        help='Dims to fit (default: 1–6)')
    parser.add_argument('--reps', type=int, default=5,
        help='Number of repetitions for each model (default: 5)')
    parser.add_argument('--output_dir', type=str, default="clean_data/training_results_rep",
        help="Directory to save results to")
    parser.add_argument('--data_path', type=str, default="clean_data/swebench_verified/swebench_verified.jsonl",
        help="Path to JSONL responses")
    parser.add_argument('--epochs', type=int, default=5000,
        help='Number of training epochs per run (default: 5000)')
    args = parser.parse_args()

    print(f"Training models with {args.reps} repetitions each")
    print(f"Epochs per run: {args.epochs}")

    data_path = resolve_path(args.data_path) if args.data_path else None
    run_training_repetitions(args.dims, args.reps, args.epochs, args.output_dir, data_path)

    print(f"\nAll models trained and averaged. Results saved to {ROOT / 'clean_data' / args.output_dir}")


if __name__ == "__main__":
    main()

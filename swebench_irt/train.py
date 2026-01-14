"""
This script trains the IRT model and allows specification the number of dimensions.
For the main analysis, we look at dimensions 1 - 6.
Call the function using the following syntax:
    python swebench_irt/train.py --dims 1 2 3 4 5 6 --output_dir training_results --epochs 5000
    python swebench_irt/train.py --data_path clean_data/swebench_verified/swebench_verified.jsonl --dims 1 2 3
"""

import sys
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pyro
import numpy as np
import json
import pandas as pd
import torch
from py_irt.dataset import Dataset
from py_irt.models import Multidim2PL
from py_irt.models import TwoParamLog
from py_irt.models import OneParamLog
from py_irt.config import IrtConfig
from py_irt.training import IrtModelTrainer
import argparse

def resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else (ROOT / path)

def resolve_output_dir(path_str: str) -> Path:
    """
    Resolve output directory in a user-friendly way.

    - Absolute paths are used as-is
    - Relative paths containing a separator are resolved relative to repo ROOT
    - Bare names (e.g. "swebench_verified_20251115_full") go under
      clean_data/<name>
    """
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p
    if "/" in path_str or "\\" in path_str:
        return ROOT / p
    return ROOT / "chris_output" / "clean_data" / p

def _suggest_jsonl_paths(missing_path: Path) -> list[Path]:
    # Look for likely candidates near the requested location, then fallback to repo-wide.
    candidates: list[Path] = []
    parent = missing_path.parent
    if parent.exists():
        candidates.extend(sorted(parent.glob("*.jsonl")))
        candidates.extend(sorted(parent.glob("**/*.jsonl")))
    if not candidates:
        candidates.extend(sorted((ROOT / "chris_output").glob("**/*.jsonl")))
    # De-duplicate while preserving order
    seen = set()
    out: list[Path] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out[:10]

def validate_data_path(data_path: Path) -> None:
    if data_path.exists():
        return
    suggestions = _suggest_jsonl_paths(data_path)
    msg = [
        f"Missing --data_path: {data_path}",
        "",
        "This file is typically produced by swebench_irt/prep_swebench.py (it builds the agent×task response matrix).",
        "Example:",
        f"  python {ROOT/'swebench_irt'/'prep_swebench.py'} \\",
        "    --experiments_dir experiments/evaluation/verified \\",
        f"    --output_path {data_path}",
        "",
    ]
    if suggestions:
        msg.append("Nearby candidate JSONL files:")
        msg.extend([f"  - {p}" for p in suggestions])
        msg.append("")
    raise SystemExit("\n".join(msg))

def set_seed(seed: int) -> None:
    """
    Best-effort reproducibility across reruns.

    Notes:
    - For full determinism of Python hashing, set PYTHONHASHSEED *before* starting Python,
      e.g. `PYTHONHASHSEED=0 python ...` (setting it inside the process is too late for some uses).
    - Deterministic algorithms can be slower; we enable them when possible.
    """
    random.seed(seed)
    np.random.seed(seed)
    pyro.set_rng_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Make torch behavior more deterministic where possible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        # Older torch versions may not support this; ignore.
        pass

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
    return Dataset.from_pandas(df, subject_column="subject_id", item_columns=item_columns), item_columns

def fit_1d_irt(data: Dataset, epochs: int, output_dir: Path) -> IrtModelTrainer:
    config = IrtConfig(
        model_type=TwoParamLog,
        priors="hierarchical",
        initializers=[
            {"name": "difficulty_from_accuracy", "eps": 1e-3},
        ],
    )
    trainer = IrtModelTrainer(config=config, data_path=None, dataset=data)
    trainer.train(epochs=epochs)

    # Extract parameters and uncertainties
    discriminations = list(trainer.best_params["disc"])
    difficulties = list(trainer.best_params["diff"])
    # Preserve alignment with parameter arrays by iterating indices in order
    item_id_map = trainer.best_params["item_ids"]  # {index:int -> item_id:str}
    subject_id_map = trainer.best_params["subject_ids"]  # {index:int -> subject_id:str}
    item_ids = [item_id_map[i] for i in range(len(difficulties))]
    abilities = list(trainer.best_params["ability"])
    subject_ids = [subject_id_map[i] for i in range(len(abilities))]

    ability_std = list(pyro.param("scale_ability").detach().cpu().numpy())
    diff_std = list(pyro.param("scale_diff").detach().cpu().numpy())
    disc_log_std = list(pyro.param("scale_slope").detach().cpu().numpy())

    out_dir = output_dir / "1d"
    out_dir.mkdir(parents=True, exist_ok=True)

    items_df = pd.DataFrame({
        "a": discriminations,
        "b": difficulties,
        "a_std": disc_log_std,
        "b_std": diff_std,
    }, index=item_ids)
    items_df.to_csv(out_dir / "items.csv")

    abilities_df = pd.DataFrame({
        "theta": abilities,
        "theta_std": ability_std,
    }, index=subject_ids).sort_values("theta", ascending=False)
    abilities_df.to_csv(out_dir / "abilities.csv")

    return trainer


def fit_1d_1pl_irt(data: Dataset, epochs: int, output_dir: Path) -> IrtModelTrainer:
    """Fit a 1D 1PL (Rasch) model - no discrimination parameter."""
    config = IrtConfig(
        model_type=OneParamLog,
        priors="hierarchical",
        initializers=[
            {"name": "difficulty_from_accuracy", "eps": 1e-3},
        ],
    )
    trainer = IrtModelTrainer(config=config, data_path=None, dataset=data)
    trainer.train(epochs=epochs)

    # Extract parameters and uncertainties
    difficulties = list(trainer.best_params["diff"])
    item_id_map = trainer.best_params["item_ids"]
    subject_id_map = trainer.best_params["subject_ids"]
    item_ids = [item_id_map[i] for i in range(len(difficulties))]
    abilities = list(trainer.best_params["ability"])
    subject_ids = [subject_id_map[i] for i in range(len(abilities))]

    ability_std = list(pyro.param("scale_ability").detach().cpu().numpy())
    diff_std = list(pyro.param("scale_diff").detach().cpu().numpy())

    out_dir = output_dir / "1d_1pl"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1PL has no discrimination parameter - save only b
    items_df = pd.DataFrame({
        "b": difficulties,
        "b_std": diff_std,
    }, index=item_ids)
    items_df.to_csv(out_dir / "items.csv")

    abilities_df = pd.DataFrame({
        "theta": abilities,
        "theta_std": ability_std,
    }, index=subject_ids).sort_values("theta", ascending=False)
    abilities_df.to_csv(out_dir / "abilities.csv")

    return trainer


def fit_md_irt(data: Dataset, dims:int, epochs:int, output_dir: Path) -> IrtModelTrainer:

    config = IrtConfig(
        model_type=Multidim2PL,
        priors="hierarchical",
        lr=0.003,
        lr_decay=1.0,
        clip_norm=5.0,
        dims=dims,
        initializers=[
            {
                "name": "difficulty_from_accuracy",
                "eps": 1e-3,
                "dims": dims,
                "jitter_std": 0.1,
                "init_disc_std": 0.0,      # PCA will set disc/ability
                "init_ability_std": 0.0,
            },
            {
                "name": "mirt_pca",
                "dims": dims,
                "disc_scale": 0.5,
                "ability_scale": 0.5,
                "center": "item",
            },
        ],
    )
    trainer = IrtModelTrainer(config=config, data_path=None, dataset=data)
    trainer.train(epochs=epochs)

    # Convert from pytorch tensor to numpy array
    abilities = pyro.param("loc_ability").detach().cpu().numpy()  # [S, D]
    difficulties = pyro.param("loc_diff").detach().cpu().numpy()  # [I, D]
    # For MIRT: discrimination uses Normal guide; do NOT exponentiate
    discriminations = pyro.param("loc_disc").detach().cpu().numpy()  # [I, D]

    ability_std = pyro.param("scale_ability").detach().cpu().numpy()
    diff_std = pyro.param("scale_diff").detach().cpu().numpy()
    disc_std = pyro.param("scale_disc").detach().cpu().numpy()

    out_dir = output_dir / f"{dims}d"
    out_dir.mkdir(parents=True, exist_ok=True)

    item_id_map = trainer.best_params["item_ids"]
    subject_id_map = trainer.best_params["subject_ids"]
    item_ids = [item_id_map[i] for i in range(difficulties.shape[0])]
    subject_ids = [subject_id_map[i] for i in range(abilities.shape[0])]

    item_rows = {}
    for i_idx, iid in enumerate(item_ids):
        row = {}
        for d in range(dims):
            row[f"a{d+1}"] = discriminations[i_idx, d]
            row[f"b{d+1}"] = difficulties[i_idx, d]
            row[f"a{d+1}_std"] = disc_std[i_idx, d]
            row[f"b{d+1}_std"] = diff_std[i_idx, d]
        item_rows[iid] = row
    pd.DataFrame.from_dict(item_rows, orient="index").to_csv(out_dir / "items.csv")

    # Abilities
    abil_rows = {}
    for s_idx, sid in enumerate(subject_ids):
        row = {}
        for d in range(dims):
            row[f"theta{d+1}"] = abilities[s_idx, d]
            row[f"theta{d+1}_std"] = ability_std[s_idx, d]
        abil_rows[sid] = row
    abilities_df = pd.DataFrame.from_dict(abil_rows, orient="index")
    abilities_df["theta_avg"] = abilities_df[[f"theta{d+1}" for d in range(dims)]].mean(axis=1)
    abilities_df.sort_values("theta_avg", ascending=False).to_csv(out_dir / "abilities.csv")

    return trainer

def main():
    parser = argparse.ArgumentParser(description='Train IRT models')
    parser.add_argument('--dims', type=int, nargs='*', default=[1, 2, 3, 4, 5, 6],
        help='Dims to fit (default: 1–6)')
    parser.add_argument('--output_dir', type=str, default="clean_data/training_results",
        help="Directory to save results to")
    parser.add_argument('--data_path', type=str, default="clean_data/swebench_verified/swebench_verified.jsonl",
        help="Path to JSONL responses")
    parser.add_argument('--epochs', type=int, default=5000,
        help='Number of training epochs (default: 5000)')
    parser.add_argument('--model', type=str, default="2pl", choices=["1pl", "2pl"],
        help='IRT model type for 1D (1pl=Rasch, 2pl=discrimination+difficulty)')
    parser.add_argument('--seed', type=int, default=None,
        help="Random seed for reproducible training (default: unset)")
    args = parser.parse_args()

    if args.seed is not None:
        # Setting PYTHONHASHSEED here is best-effort; prefer exporting it before launch.
        os.environ.setdefault("PYTHONHASHSEED", str(args.seed))
        set_seed(args.seed)

    data_path = resolve_path(args.data_path)
    validate_data_path(data_path)
    data, item_columns = load_irt_data(data_path)
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for dim in args.dims:
        pyro.clear_param_store()  # Clear parameters between different dimensional models
        if dim == 1:
            if args.model == "1pl":
                print(f"Training 1D 1PL (Rasch) model...")
                fit_1d_1pl_irt(data=data, epochs=args.epochs, output_dir=output_dir)
            else:
                print(f"Training 1D 2PL model...")
                fit_1d_irt(data=data, epochs=args.epochs, output_dir=output_dir)
        else:
            print(f"Training {dim}D 2PL model...")
            fit_md_irt(data=data, dims=dim, epochs=args.epochs, output_dir=output_dir)

if __name__ == "__main__":
    main()

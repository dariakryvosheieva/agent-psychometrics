#!/usr/bin/env python3
"""Run held-out benchmark generalization experiments."""

import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from experiment_new_benchmarks.config import (
    ALL_DATASETS,
    DATASET_DEFAULTS,
    DEFAULT_HELDOUT_DATASETS,
    ExperimentNewBenchmarksConfig,
    display_name_for_dataset,
    expand_dataset_path_template,
)
from experiment_new_benchmarks.pipeline import METHOD_DISPLAY_NAMES
from experiment_new_tasks.results import (
    extract_metrics as extract_metrics_with_mapping,
    format_results_table,
    save_results_csv,
    save_summary_json,
)
from swebench_irt.model_scaffold_combine import THETA_COMBINE_CHOICES


ROOT = Path(__file__).resolve().parents[1]
METHOD_NAME_MAPPINGS = {
    "baseline": "Baseline",
    "embedding": "Embedding",
    "judge": "LLM-as-a-Judge",
    "combined": "Combined",
    "oracle": "Oracle",
}
RESULT_METHODS = ["Baseline", "Embedding", "LLM-as-a-Judge", "Combined", "Oracle"]


def run_single_heldout(
    heldout_dataset: str,
    *,
    output_dir: Path,
    n_bootstrap: int = 10000,
    bootstrap_seed: int = 0,
    responses_path: Optional[str] = None,
    embeddings_path: Optional[str] = None,
    llm_judge_features_path: Optional[str] = None,
    irt_epochs: int = 2000,
    irt_device: str = "cuda",
    irt_lr: float = 0.01,
    irt_model: str = "1d_1pl",
    theta_combine: str = "sum",
) -> Tuple[str, Dict[str, Any]]:
    from experiment_new_benchmarks.pipeline import evaluate_heldout_benchmark

    display_name = f"Held-out {display_name_for_dataset(heldout_dataset)}"
    try:
        config = ExperimentNewBenchmarksConfig.with_overrides(
            output_dir=output_dir,
            responses_paths=expand_dataset_path_template(
                responses_path,
                defaults={
                    dataset: DATASET_DEFAULTS[dataset]["responses_path"]
                    for dataset in ALL_DATASETS
                },
            ),
            embeddings_paths=expand_dataset_path_template(
                embeddings_path,
                defaults={
                    dataset: DATASET_DEFAULTS[dataset]["embeddings_path"]
                    for dataset in ALL_DATASETS
                },
            ),
            llm_judge_features_paths=expand_dataset_path_template(
                llm_judge_features_path,
                defaults={
                    dataset: DATASET_DEFAULTS[dataset]["llm_judge_features_path"]
                    for dataset in ALL_DATASETS
                },
            ),
            irt_epochs=irt_epochs,
            irt_device=irt_device,
            irt_lr=irt_lr,
            irt_model=irt_model,
            theta_combine=theta_combine,
        )
    except Exception as exc:
        return display_name, {"error": f"Config error: {exc}"}

    try:
        results = evaluate_heldout_benchmark(
            config,
            ROOT,
            heldout_dataset=heldout_dataset,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=bootstrap_seed,
        )
        return display_name, results
    except Exception as exc:
        import traceback

        return display_name, {"error": f"Execution error: {exc}\n{traceback.format_exc()}"}


def extract_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    return extract_metrics_with_mapping(results, METHOD_NAME_MAPPINGS)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run train-on-three, held-out benchmark experiments"
    )
    parser.add_argument("--output", type=Path, help="Optional CSV output path")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output/experiment_new_benchmarks"),
    )
    parser.add_argument(
        "--heldout_datasets",
        nargs="+",
        choices=ALL_DATASETS,
        default=DEFAULT_HELDOUT_DATASETS,
        help="Benchmarks to hold out. Default: SWE-bench Pro and GSO.",
    )
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--n_bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=0)
    parser.add_argument("--max_workers", type=int, default=2)
    parser.add_argument(
        "--responses_path",
        type=str,
        default=None,
        help="Override response JSONL path for all datasets; supports {dataset}.",
    )
    parser.add_argument(
        "--embeddings_path",
        type=str,
        default=None,
        help="Override embeddings .npz path for all datasets; supports {dataset}.",
    )
    parser.add_argument(
        "--llm_judge_features_path",
        type=str,
        default=None,
        help="Override LLM judge CSV path for all datasets; supports {dataset}.",
    )
    parser.add_argument("--irt_epochs", type=int, default=2000)
    parser.add_argument("--irt_device", type=str, default="cuda")
    parser.add_argument("--irt_lr", type=float, default=0.01)
    parser.add_argument("--irt_model", type=str, default="1d_1pl", choices=["1d_1pl", "2d_1pl"])
    parser.add_argument("--theta_combine", type=str, default="sum", choices=THETA_COMBINE_CHOICES)
    args = parser.parse_args()

    if args.n_bootstrap < 1:
        parser.error("--n_bootstrap must be >= 1")

    print("Running Experiment New Benchmarks...")
    print(f"Held-out benchmarks: {', '.join(args.heldout_datasets)}")
    print(f"Methods: {', '.join(METHOD_DISPLAY_NAMES[m] for m in METHOD_NAME_MAPPINGS)}")
    print(f"Bootstrap samples: {args.n_bootstrap}")
    print()

    all_results: Dict[str, Dict[str, Any]] = {}
    if args.sequential:
        for heldout_dataset in args.heldout_datasets:
            name, results = run_single_heldout(
                heldout_dataset,
                output_dir=args.output_dir,
                n_bootstrap=args.n_bootstrap,
                bootstrap_seed=args.bootstrap_seed,
                responses_path=args.responses_path,
                embeddings_path=args.embeddings_path,
                llm_judge_features_path=args.llm_judge_features_path,
                irt_epochs=args.irt_epochs,
                irt_device=args.irt_device,
                irt_lr=args.irt_lr,
                irt_model=args.irt_model,
                theta_combine=args.theta_combine,
            )
            all_results[name] = extract_metrics(results)
    else:
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(
                    run_single_heldout,
                    heldout_dataset,
                    output_dir=args.output_dir,
                    n_bootstrap=args.n_bootstrap,
                    bootstrap_seed=args.bootstrap_seed,
                    responses_path=args.responses_path,
                    embeddings_path=args.embeddings_path,
                    llm_judge_features_path=args.llm_judge_features_path,
                    irt_epochs=args.irt_epochs,
                    irt_device=args.irt_device,
                    irt_lr=args.irt_lr,
                    irt_model=args.irt_model,
                    theta_combine=args.theta_combine,
                ): heldout_dataset
                for heldout_dataset in args.heldout_datasets
            }
            for future in as_completed(futures):
                heldout_dataset = futures[future]
                name, results = future.result()
                all_results[name] = extract_metrics(results)
                metrics = all_results[name]
                if "error" in metrics:
                    print(f"{name}: ERROR - {str(metrics['error'])[:100]}...")
                else:
                    combined = metrics.get("Combined")
                    baseline = metrics.get("Baseline")
                    combined_str = f"{combined:.4f}" if combined is not None else "N/A"
                    baseline_str = f"{baseline:.4f}" if baseline is not None else "N/A"
                    print(f"{name}: Baseline={baseline_str}, Combined={combined_str}")

    ordered = {
        f"Held-out {display_name_for_dataset(dataset)}": all_results[
            f"Held-out {display_name_for_dataset(dataset)}"
        ]
        for dataset in args.heldout_datasets
        if f"Held-out {display_name_for_dataset(dataset)}" in all_results
    }

    print("\n" + "=" * 80)
    print("EXPERIMENT NEW BENCHMARKS RESULTS SUMMARY")
    print("=" * 80 + "\n")
    print(format_results_table(ordered, RESULT_METHODS))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_results_csv(ordered, args.output, RESULT_METHODS)
        print(f"\nResults saved to: {args.output}")

    summary_path = args.output_dir / "summary.json"
    save_summary_json(ordered, summary_path)
    print(f"Full results saved to: {summary_path}")

    if len(args.heldout_datasets) == 1:
        heldout_dataset = args.heldout_datasets[0]
        per_heldout_predictions = args.output_dir / heldout_dataset / "predictions.csv"
        top_level_predictions = args.output_dir / "predictions.csv"
        if per_heldout_predictions.exists():
            shutil.copyfile(per_heldout_predictions, top_level_predictions)
            print(f"Judge predictions saved to: {top_level_predictions}")


if __name__ == "__main__":
    main()


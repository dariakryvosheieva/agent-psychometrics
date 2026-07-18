#!/usr/bin/env python3
"""Run the new-responses experiment."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from experiment_new_responses.config import DATASET_DEFAULTS
from experiment_new_tasks.results import (
    apply_holm_correction,
    extract_metrics as extract_metrics_with_mapping,
    format_results_table,
    print_holm_summary,
    save_results_csv,
    save_summary_json,
)
from swebench_irt.model_scaffold_combine import THETA_COMBINE_CHOICES


ROOT = Path(__file__).resolve().parents[1]
ALL_DATASETS = ["swebench_verified", "swebench_pro", "terminalbench", "gso"]
METHOD_NAME_MAPPINGS = {
    "oracle": "Oracle",
    "model_scaffold": "Model+Scaffold",
    "standard_irt_baseline": "Standard IRT",
}
RESULT_METHODS = ["Standard IRT", "Model+Scaffold", "Oracle"]
# Family for the Holm-Bonferroni correction: the per-benchmark
# Model+Scaffold vs Standard IRT comparisons. The four-benchmark row is a
# separate analysis and Oracle is a skyline, so both are excluded.
HOLM_FAMILY_METHODS = ["Model+Scaffold"]
FOUR_BENCHMARK_DISPLAY_NAME = "Verified + Pro + Terminal-Bench + GSO"


def run_single_dataset(
    dataset: str,
    *,
    k_folds: int = 5,
    n_fold_seeds: int = 20,
    fold_seed_start: int = 0,
    n_bootstrap: int = 10000,
    bootstrap_seed: int = 0,
    responses_path: Optional[str] = None,
    output_dir: Optional[Path] = None,
    irt_epochs: int = 2000,
    irt_device: str = "cuda",
    irt_lr: float = 0.01,
    irt_model: str = "1d_1pl",
    theta_combine: str = "sum",
) -> Tuple[str, Dict[str, Any]]:
    from experiment_new_responses.config import ExperimentNewResponsesConfig
    from experiment_new_responses.pipeline import cross_validate_all_predictors_repeated_seeds

    try:
        overrides: Dict[str, Any] = {
            "irt_epochs": int(irt_epochs),
            "irt_device": str(irt_device),
            "irt_lr": float(irt_lr),
            "irt_model": str(irt_model),
            "theta_combine": str(theta_combine),
        }
        if responses_path is not None:
            overrides["responses_path"] = Path(responses_path.replace("{dataset}", dataset))
        if output_dir is not None:
            overrides["output_dir"] = Path(str(output_dir).replace("{dataset}", dataset))
        config = ExperimentNewResponsesConfig.for_dataset(dataset, **overrides)
    except Exception as exc:
        display_name = DATASET_DEFAULTS[dataset]["display_name"]
        return display_name, {"error": f"Config error: {exc}"}

    try:
        fold_seeds = list(range(fold_seed_start, fold_seed_start + n_fold_seeds * 10))
        results = cross_validate_all_predictors_repeated_seeds(
            config,
            ROOT,
            dataset=dataset,
            k=k_folds,
            fold_seeds=fold_seeds,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=bootstrap_seed,
            target_n_fold_seeds=n_fold_seeds,
            baseline_method="standard_irt_baseline",
        )
        if int(results["n_fold_seeds"]) < int(n_fold_seeds):
            raise RuntimeError(
                f"Only collected {results['n_fold_seeds']}/{n_fold_seeds} valid fold seeds"
            )
        return config.display_name, results
    except Exception as exc:
        import traceback
        return config.display_name, {"error": f"Execution error: {exc}\n{traceback.format_exc()}"}


def run_four_benchmark_model_scaffold(
    *,
    k_folds: int = 5,
    n_fold_seeds: int = 20,
    fold_seed_start: int = 0,
    output_dir: Path,
    irt_epochs: int = 2000,
    irt_device: str = "cuda",
    irt_lr: float = 0.01,
    irt_model: str = "1d_1pl",
    theta_combine: str = "sum",
) -> Tuple[str, Dict[str, Any]]:
    from experiment_new_responses.pipeline import (
        cross_validate_four_benchmark_model_scaffold_repeated_seeds,
    )

    try:
        dataset_to_responses_path = {
            dataset: ROOT / DATASET_DEFAULTS[dataset]["responses_path"]
            for dataset in ALL_DATASETS
        }
        fold_seeds = list(range(fold_seed_start, fold_seed_start + n_fold_seeds * 10))
        results = cross_validate_four_benchmark_model_scaffold_repeated_seeds(
            dataset_to_responses_path=dataset_to_responses_path,
            output_dir=output_dir / "four_benchmark_model_scaffold",
            k=k_folds,
            fold_seeds=fold_seeds,
            target_n_fold_seeds=n_fold_seeds,
            irt_epochs=irt_epochs,
            irt_device=irt_device,
            irt_lr=irt_lr,
            irt_model=irt_model,
            theta_combine=theta_combine,
        )
        if int(results["n_fold_seeds"]) < int(n_fold_seeds):
            raise RuntimeError(
                f"Only collected {results['n_fold_seeds']}/{n_fold_seeds} valid fold seeds"
            )
        return FOUR_BENCHMARK_DISPLAY_NAME, results
    except Exception as exc:
        import traceback
        return FOUR_BENCHMARK_DISPLAY_NAME, {
            "error": f"Execution error: {exc}\n{traceback.format_exc()}"
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run new-response observation-pair holdout experiments"
    )
    parser.add_argument("--output", type=Path, help="Optional CSV output path")
    parser.add_argument("--output_dir", type=Path, default=Path("output/experiment_new_responses"))
    parser.add_argument("--datasets", nargs="+", choices=ALL_DATASETS)
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--k_folds", type=int, default=5)
    parser.add_argument("--n_fold_seeds", type=int, default=20)
    parser.add_argument("--fold_seed_start", type=int, default=0)
    parser.add_argument("--n_bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=0)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--responses_path", type=str, default=None)
    parser.add_argument("--per_dataset_output_dir", type=str, default=None)
    parser.add_argument("--irt_epochs", type=int, default=2000)
    parser.add_argument("--irt_device", type=str, default="cuda")
    parser.add_argument("--irt_lr", type=float, default=0.01)
    parser.add_argument("--irt_model", type=str, default="1d_1pl", choices=["1d_1pl", "2d_1pl"])
    parser.add_argument("--theta_combine", type=str, default="sum", choices=THETA_COMBINE_CHOICES)
    parser.add_argument("--skip_four_benchmark", action="store_true")
    args = parser.parse_args()

    if args.n_fold_seeds < 1:
        parser.error("--n_fold_seeds must be >= 1")
    if args.n_bootstrap < 1:
        parser.error("--n_bootstrap must be >= 1")

    datasets_to_run = args.datasets if args.datasets else ALL_DATASETS

    def _per_dataset_output_dir(dataset: str) -> Path:
        if args.per_dataset_output_dir is not None:
            return Path(args.per_dataset_output_dir.replace("{dataset}", dataset))
        return args.output_dir / dataset

    all_results: Dict[str, Dict[str, Any]] = {}
    if args.sequential:
        for dataset in datasets_to_run:
            name, results = run_single_dataset(
                dataset,
                k_folds=args.k_folds,
                n_fold_seeds=args.n_fold_seeds,
                fold_seed_start=args.fold_seed_start,
                n_bootstrap=args.n_bootstrap,
                bootstrap_seed=args.bootstrap_seed,
                responses_path=args.responses_path,
                output_dir=_per_dataset_output_dir(dataset),
                irt_epochs=args.irt_epochs,
                irt_device=args.irt_device,
                irt_lr=args.irt_lr,
                irt_model=args.irt_model,
                theta_combine=args.theta_combine,
            )
            all_results[name] = extract_metrics_with_mapping(results, METHOD_NAME_MAPPINGS)
        if not args.skip_four_benchmark:
            name, results = run_four_benchmark_model_scaffold(
                k_folds=args.k_folds,
                n_fold_seeds=args.n_fold_seeds,
                fold_seed_start=args.fold_seed_start,
                output_dir=args.output_dir,
                irt_epochs=args.irt_epochs,
                irt_device=args.irt_device,
                irt_lr=args.irt_lr,
                irt_model=args.irt_model,
                theta_combine=args.theta_combine,
            )
            all_results[name] = extract_metrics_with_mapping(results, METHOD_NAME_MAPPINGS)
    else:
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(
                    run_single_dataset,
                    dataset,
                    k_folds=args.k_folds,
                    n_fold_seeds=args.n_fold_seeds,
                    fold_seed_start=args.fold_seed_start,
                    n_bootstrap=args.n_bootstrap,
                    bootstrap_seed=args.bootstrap_seed,
                    responses_path=args.responses_path,
                    output_dir=_per_dataset_output_dir(dataset),
                    irt_epochs=args.irt_epochs,
                    irt_device=args.irt_device,
                    irt_lr=args.irt_lr,
                    irt_model=args.irt_model,
                    theta_combine=args.theta_combine,
                ): DATASET_DEFAULTS[dataset]["display_name"]
                for dataset in datasets_to_run
            }
            if not args.skip_four_benchmark:
                futures[
                    executor.submit(
                        run_four_benchmark_model_scaffold,
                        k_folds=args.k_folds,
                        n_fold_seeds=args.n_fold_seeds,
                        fold_seed_start=args.fold_seed_start,
                        output_dir=args.output_dir,
                        irt_epochs=args.irt_epochs,
                        irt_device=args.irt_device,
                        irt_lr=args.irt_lr,
                        irt_model=args.irt_model,
                        theta_combine=args.theta_combine,
                    )
                ] = FOUR_BENCHMARK_DISPLAY_NAME
            for future in as_completed(futures):
                name, results = future.result()
                all_results[name] = extract_metrics_with_mapping(
                    results, METHOD_NAME_MAPPINGS
                )

    ordered = {
        DATASET_DEFAULTS[dataset]["display_name"]: all_results[DATASET_DEFAULTS[dataset]["display_name"]]
        for dataset in datasets_to_run
        if DATASET_DEFAULTS[dataset]["display_name"] in all_results
    }
    if FOUR_BENCHMARK_DISPLAY_NAME in all_results:
        ordered[FOUR_BENCHMARK_DISPLAY_NAME] = all_results[FOUR_BENCHMARK_DISPLAY_NAME]

    print("\n" + "=" * 80)
    print("EXPERIMENT NEW RESPONSES RESULTS SUMMARY")
    print("=" * 80 + "\n")
    print(format_results_table(ordered, RESULT_METHODS))

    per_benchmark_results = {
        name: metrics
        for name, metrics in ordered.items()
        if name != FOUR_BENCHMARK_DISPLAY_NAME
    }
    apply_holm_correction(per_benchmark_results, HOLM_FAMILY_METHODS)
    print_holm_summary(per_benchmark_results, HOLM_FAMILY_METHODS)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_results_csv(ordered, args.output, RESULT_METHODS)
        print(f"\nResults saved to: {args.output}")

    summary_path = args.output_dir / "summary.json"
    save_summary_json(ordered, summary_path)
    print(f"Full results saved to: {summary_path}")


if __name__ == "__main__":
    main()

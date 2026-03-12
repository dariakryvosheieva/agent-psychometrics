"""Extract LLM judge features at a given info level override for information ablation.

For the information ablation experiment, each feature should be extracted with
the LLM seeing all the context available at that ablation level — not just the
context for the feature's natural level. For example, at the TEST ablation level,
PROBLEM features should be re-extracted with the LLM seeing the test patch.

ENVIRONMENT features (from the auditor agent pipeline) cannot be re-extracted via
this script. They are merged in from the natural source CSV during assembly.

Usage:
    # Extract all non-ENV features at a given info level (from scratch)
    python -m llm_judge_feature_extraction.extract_ablation_overrides --info-level problem
    python -m llm_judge_feature_extraction.extract_ablation_overrides --info-level test
    python -m llm_judge_feature_extraction.extract_ablation_overrides --info-level solution

    # Dry run to see cost estimate
    python -m llm_judge_feature_extraction.extract_ablation_overrides --info-level test --dry-run

    # Specific datasets
    python -m llm_judge_feature_extraction.extract_ablation_overrides \
        --info-level test --datasets swebench_verified gso

    # Parallel extraction with custom concurrency
    python -m llm_judge_feature_extraction.extract_ablation_overrides \
        --info-level test --parallel --concurrency 30
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from llm_judge_feature_extraction.batched_extractor import BatchedFeatureExtractor
from llm_judge_feature_extraction.feature_registry import get_features_by_level
from llm_judge_feature_extraction.prompt_config import InfoLevel
from llm_judge_feature_extraction.task_context import get_task_context
from llm_judge_feature_extraction.task_loaders import load_tasks

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]

ALL_DATASETS = ["swebench_verified", "gso", "terminalbench", "swebench_pro"]

# Where raw extraction JSONs go (gitignored)
EXTRACTION_OUTPUT_BASE = ROOT / "output" / "ablation_overrides"

# Where assembled per-level source CSVs go (tracked in git)
PER_LEVEL_SOURCE_BASE = ROOT / "llm_judge_features" / "information_ablation" / "per_level_source"

# Natural source CSVs with all 28 features at natural info levels
NATURAL_SOURCE_BASE = ROOT / "llm_judge_features" / "information_ablation" / "source"

# Info levels that the BatchedFeatureExtractor can handle (not ENVIRONMENT)
EXTRACTABLE_LEVELS = {
    "problem": (InfoLevel.PROBLEM, [InfoLevel.PROBLEM]),
    "test": (InfoLevel.TEST, [InfoLevel.PROBLEM, InfoLevel.TEST]),
    "solution": (InfoLevel.SOLUTION, [InfoLevel.PROBLEM, InfoLevel.TEST, InfoLevel.SOLUTION]),
}


def get_feature_names_for_extraction(level: str) -> List[str]:
    """Get all non-ENVIRONMENT feature names at or below the given level."""
    _, feature_levels = EXTRACTABLE_LEVELS[level]
    names = []
    for fl in feature_levels:
        names.extend(f.name for f in get_features_by_level(fl))
    return names


def get_environment_feature_names() -> List[str]:
    """Get all ENVIRONMENT feature names (from auditor pipeline)."""
    return [f.name for f in get_features_by_level(InfoLevel.ENVIRONMENT)]


def assemble_per_level_source(
    dataset: str,
    level: str,
    extraction_csv: Path,
    natural_source_csv: Path,
    output_path: Path,
) -> Path:
    """Merge extraction CSV with ENVIRONMENT features from natural source.

    The extraction CSV contains non-ENVIRONMENT features extracted at the
    override level. The natural source CSV provides ENVIRONMENT features
    (from the auditor pipeline, which can't be re-extracted).

    Args:
        dataset: Dataset name (for logging).
        level: Info level name (problem, test, solution).
        extraction_csv: CSV with features extracted at the override level.
        natural_source_csv: Natural source CSV with all 28 features.
        output_path: Where to write the assembled CSV.

    Returns:
        Path to the assembled CSV.
    """
    extraction_df = pd.read_csv(extraction_csv)
    natural_df = pd.read_csv(natural_source_csv)

    # Normalize index column
    for df in (extraction_df, natural_df):
        if "instance_id" not in df.columns and "_task_id" in df.columns:
            df.rename(columns={"_task_id": "instance_id"}, inplace=True)

    env_features = get_environment_feature_names()
    extracted_features = get_feature_names_for_extraction(level)

    # Get ENVIRONMENT features from natural source
    env_cols = ["instance_id"] + [f for f in env_features if f in natural_df.columns]
    env_df = natural_df[env_cols].copy()

    # Get extracted features from extraction CSV
    extract_cols = ["instance_id"] + [f for f in extracted_features if f in extraction_df.columns]
    extract_df = extraction_df[extract_cols].copy()

    # Merge
    merged = extract_df.merge(env_df, on="instance_id", how="inner")

    # Warn if extraction has tasks not in natural source (extra tasks, dropped by inner join)
    extra_in_extraction = set(extract_df["instance_id"]) - set(env_df["instance_id"])
    if extra_in_extraction:
        logger.warning(
            f"{dataset}/{level}: {len(extra_in_extraction)} task(s) in extraction CSV "
            f"not in natural source (dropped): {sorted(extra_in_extraction)[:5]}"
        )

    # Error if natural source has tasks missing from extraction (data loss)
    missing_from_extraction = set(env_df["instance_id"]) - set(extract_df["instance_id"])
    if missing_from_extraction:
        raise ValueError(
            f"{dataset}/{level}: {len(missing_from_extraction)} task(s) in natural source "
            f"missing from extraction CSV: {sorted(missing_from_extraction)[:5]}"
        )

    # Order columns: instance_id, then features in registry order, then metadata
    all_feature_names = extracted_features + env_features
    ordered_features = [f for f in all_feature_names if f in merged.columns]
    meta_cols = sorted(c for c in merged.columns if c.startswith("_"))
    ordered = ["instance_id"] + ordered_features + meta_cols
    merged = merged[[c for c in ordered if c in merged.columns]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"  Assembled {output_path.relative_to(ROOT)} ({len(merged)} tasks, {len(ordered_features)} features)")
    return output_path


def extract_and_assemble(
    dataset: str,
    level: str,
    provider: str = "anthropic",
    model: Optional[str] = None,
    parallel: bool = True,
    concurrency: int = 30,
    dry_run: bool = False,
) -> Optional[Path]:
    """Extract features at a given info level and assemble the per-level source CSV.

    Args:
        dataset: Dataset name.
        level: One of "problem", "test", "solution".
        provider: LLM provider.
        model: LLM model override.
        parallel: Whether to use async parallel extraction.
        concurrency: Max concurrent tasks.
        dry_run: Show cost estimate without running.

    Returns:
        Path to the assembled per-level source CSV, or None if dry_run.
    """
    override_level, _ = EXTRACTABLE_LEVELS[level]
    feature_names = get_feature_names_for_extraction(level)

    tasks = load_tasks(dataset)
    task_context = get_task_context(dataset)

    output_dir = EXTRACTION_OUTPUT_BASE / dataset / f"{level}_override"

    extractor = BatchedFeatureExtractor(
        feature_names=feature_names,
        task_context=task_context,
        provider=provider,
        model=model,
        info_level_override=override_level,
    )

    if dry_run:
        extractor.dry_run(tasks, output_dir)
        return None

    print(f"\n{'='*60}")
    print(f"Extracting {len(feature_names)} features at {level.upper()} level for {dataset}")
    print(f"{'='*60}")

    if parallel:
        extraction_csv = extractor.run_parallel(
            tasks, output_dir, concurrency=concurrency
        )
    else:
        extraction_csv = extractor.run(tasks, output_dir)

    if extraction_csv is None:
        raise RuntimeError(f"Extraction failed for {dataset}/{level}: no CSV produced")

    # Assemble with ENVIRONMENT features
    natural_source = NATURAL_SOURCE_BASE / f"{dataset}.csv"
    if not natural_source.exists():
        raise FileNotFoundError(
            f"Natural source CSV not found: {natural_source}. "
            f"Expected at llm_judge_features/information_ablation/source/{dataset}.csv"
        )

    output_path = PER_LEVEL_SOURCE_BASE / dataset / f"{level}.csv"
    return assemble_per_level_source(
        dataset, level, extraction_csv, natural_source, output_path
    )


def main():
    parser = argparse.ArgumentParser(
        description="Extract LLM judge features at an info level override for information ablation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--info-level", type=str, required=True,
        choices=list(EXTRACTABLE_LEVELS.keys()),
        help="Info level to extract at (all non-ENV features see this level's context)",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=ALL_DATASETS,
        choices=ALL_DATASETS, help="Datasets to process (default: all)",
    )
    parser.add_argument(
        "--provider", type=str, default="anthropic",
        choices=["anthropic", "openai"],
        help="LLM provider (default: anthropic)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Specific model to use (default: provider's default)",
    )
    parser.add_argument(
        "--parallel", action="store_true",
        help="Run extraction in parallel (async API calls)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=30,
        help="Max concurrent tasks when --parallel (default: 30)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="Show cost estimate without running",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    for dataset in args.datasets:
        extract_and_assemble(
            dataset=dataset,
            level=args.info_level,
            provider=args.provider,
            model=args.model,
            parallel=args.parallel,
            concurrency=args.concurrency,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()

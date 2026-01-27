"""Feature extraction pipeline with verification and augmentation.

Usage:
    python -m experiment_ab_shared.llm_judge.extract_pipeline deterministic \
        --dataset swebench --output chris_output/deterministic/swebench.csv

    python -m experiment_ab_shared.llm_judge.extract_pipeline verify \
        --output-dir chris_output/llm_judge_features/swebench --dataset swebench

    python -m experiment_ab_shared.llm_judge.extract_pipeline augment \
        --features-csv path/to/features.csv --dataset swebench
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class ExtractionStatus:
    """Status of feature extraction."""
    total_tasks: int
    extracted: int
    missing: List[str]
    failed: List[str]
    has_deterministic: bool

    @property
    def complete(self) -> bool:
        return len(self.missing) == 0 and len(self.failed) == 0

    def print_report(self) -> None:
        print("\n" + "=" * 60)
        print("EXTRACTION STATUS")
        print("=" * 60)
        print(f"Total expected: {self.total_tasks}")
        print(f"Extracted: {self.extracted}")
        print(f"Missing: {len(self.missing)}")
        print(f"Failed: {len(self.failed)}")
        print(f"Has deterministic: {self.has_deterministic}")
        if self.missing:
            print(f"\nMissing (first 10): {self.missing[:10]}")
        if self.complete:
            print("\n[OK] Complete!")
        else:
            print("\n[INCOMPLETE] Run pipeline to continue")


def verify_extraction_complete(output_dir: Path, expected_task_ids: List[str]) -> ExtractionStatus:
    """Verify which tasks have been extracted."""
    output_dir = Path(output_dir)
    extracted, missing, failed = [], [], []

    for task_id in expected_task_ids:
        safe_id = task_id.replace("/", "__")
        json_path = output_dir / f"{safe_id}.json"
        if not json_path.exists():
            missing.append(task_id)
            continue
        try:
            with open(json_path) as f:
                data = json.load(f)
            if "_task_id" not in data:
                failed.append(task_id)
            else:
                extracted.append(task_id)
        except (json.JSONDecodeError, IOError):
            failed.append(task_id)

    csv_path = output_dir / "llm_judge_features.csv"
    has_deterministic = False
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path, nrows=1)
            has_deterministic = any(c in df.columns for c in ["num_files_modified", "patch_adds", "stmt_words"])
        except Exception:
            pass

    return ExtractionStatus(len(expected_task_ids), len(extracted), missing, failed, has_deterministic)


def extract_deterministic_features_only(dataset: str, output_path: Path) -> Path:
    """Extract only deterministic features (no LLM calls)."""
    from experiment_ab_shared.llm_judge.deterministic_features import (
        compute_all_swebench_deterministic_features,
        compute_solution_features,
    )

    tasks = _load_tasks_for_dataset(dataset)
    print(f"Extracting deterministic features for {len(tasks)} tasks...")

    rows, errors = [], []
    for i, task in enumerate(tasks):
        task_id = _get_task_id(task, dataset)
        if (i + 1) % 100 == 0:
            print(f"  [{i + 1}/{len(tasks)}] {task_id}...")
        try:
            if dataset in ["swebench", "swebench_pro", "gso"]:
                features = compute_all_swebench_deterministic_features(task)
            elif dataset == "terminalbench":
                features = compute_solution_features(task.get("solution", ""))
            else:
                raise ValueError(f"Unsupported dataset: {dataset}")
            features["_instance_id"] = task_id
            rows.append(features)
        except Exception as e:
            errors.append((task_id, str(e)))

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for tid, err in errors[:5]:
            print(f"  - {tid}: {err}")

    df = pd.DataFrame(rows)
    if "_instance_id" in df.columns:
        cols = ["_instance_id"] + [c for c in df.columns if c != "_instance_id"]
        df = df[cols]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} tasks to: {output_path}")
    return output_path


def augment_with_deterministic_features(features_csv: Path, dataset: str, output_path: Optional[Path] = None) -> Path:
    """Add deterministic features to an existing features CSV."""
    from experiment_ab_shared.llm_judge.deterministic_features import (
        compute_all_swebench_deterministic_features,
        compute_solution_features,
    )

    df = pd.read_csv(features_csv)
    task_id_col = next((c for c in ["_instance_id", "instance_id", "_task_id", "task_id"] if c in df.columns), df.columns[0])

    tasks = _load_tasks_for_dataset(dataset)
    task_dict = {_get_task_id(t, dataset): t for t in tasks}

    print(f"Augmenting {len(df)} rows...")
    det_features = []
    for _, row in df.iterrows():
        task_id = str(row[task_id_col]).replace("instance_", "")
        task = task_dict.get(task_id) or task_dict.get(f"instance_{task_id}")
        if task is None:
            det_features.append({})
            continue
        try:
            if dataset in ["swebench", "swebench_pro", "gso"]:
                det_features.append(compute_all_swebench_deterministic_features(task))
            elif dataset == "terminalbench":
                det_features.append(compute_solution_features(task.get("solution", "")))
            else:
                det_features.append({})
        except Exception:
            det_features.append({})

    det_df = pd.DataFrame(det_features)
    for col in det_df.columns:
        if col not in df.columns:
            df[col] = det_df[col]

    out = Path(output_path) if output_path else features_csv
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved to: {out}")
    return out


def _load_tasks_for_dataset(dataset: str) -> List[Dict[str, Any]]:
    """Load tasks for a given dataset."""
    if dataset == "swebench":
        from datasets import load_dataset
        return list(load_dataset("princeton-nlp/SWE-bench_Verified", split="test"))
    elif dataset == "swebench_pro":
        from datasets import load_dataset
        return list(load_dataset("nebius/SWE-bench-pro", split="test"))
    elif dataset == "gso":
        from datasets import load_dataset
        return list(load_dataset("nebius/GSO_test", split="test"))
    elif dataset == "terminalbench":
        data_path = Path("clean_data/terminalbench/terminalbench_tasks.json")
        if not data_path.exists():
            raise FileNotFoundError(f"TerminalBench not found at {data_path}")
        with open(data_path) as f:
            return json.load(f)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def _get_task_id(task: Dict[str, Any], dataset: str) -> str:
    """Get task ID from task dictionary."""
    if dataset == "terminalbench":
        return task.get("task_id", task.get("id", ""))
    return task.get("instance_id", task.get("task_id", ""))


def main():
    parser = argparse.ArgumentParser(description="Feature extraction pipeline")
    sub = parser.add_subparsers(dest="command")

    v = sub.add_parser("verify", help="Verify extraction completeness")
    v.add_argument("--output-dir", type=Path, required=True)
    v.add_argument("--dataset", required=True)

    d = sub.add_parser("deterministic", help="Extract deterministic features only")
    d.add_argument("--dataset", required=True)
    d.add_argument("--output", type=Path, required=True)

    a = sub.add_parser("augment", help="Add deterministic features to CSV")
    a.add_argument("--features-csv", type=Path, required=True)
    a.add_argument("--dataset", required=True)
    a.add_argument("--output", type=Path)

    args = parser.parse_args()

    if args.command == "verify":
        tasks = _load_tasks_for_dataset(args.dataset)
        task_ids = [_get_task_id(t, args.dataset) for t in tasks]
        verify_extraction_complete(args.output_dir, task_ids).print_report()
    elif args.command == "deterministic":
        extract_deterministic_features_only(args.dataset, args.output)
    elif args.command == "augment":
        augment_with_deterministic_features(args.features_csv, args.dataset, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

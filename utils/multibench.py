"""Shared multi-benchmark embedding helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from utils import difficulty_prediction as base
from utils.difficulty_prediction import (
    iter_subject_responses_jsonl_generic,
    iter_subject_responses_jsonl_terminal,
    load_all_responses_generic,
    load_all_responses_terminal,
)
from utils.irt_training import (
    _import_shared_irt_module,
    _import_swebench_irt_module,
    build_multibench_obs_from_tagged_responses,
    train_irt_model_scaffold_1pl,
)


def _default_benchmark_embedding_dirs() -> Dict[str, str]:
    repo_root = str(Path(__file__).resolve().parents[1])
    return {
        "verified": os.path.join(repo_root, "data", "swebench_verified"),
        "pro": os.path.join(repo_root, "data", "swebench_pro"),
        "terminal_bench": os.path.join(repo_root, "data", "terminalbench"),
        "gso": os.path.join(repo_root, "data", "gso"),
    }


def _try_load_concat_embeddings_from_single_benchmark_caches(
    *,
    train_benchmarks: Sequence[str],
    required_ids_by_bench: Dict[str, List[str]],
    out_dir: str,
    backbone: str,
    max_length: int,
    embedding_layer: int,
    instruction_sig: str,
) -> Optional[Tuple[List[str], "base.np.ndarray", Dict[str, str]]]:
    roots: List[str] = []
    roots.extend(base._candidate_embedding_roots(out_dir=str(out_dir)))
    bench_dirs = _default_benchmark_embedding_dirs()
    for bench in train_benchmarks:
        bench_dir = str(bench_dirs.get(str(bench), "") or "").strip()
        if bench_dir:
            roots.append(bench_dir)
            roots.append(os.path.join(bench_dir, "embeddings"))
    roots = [str(r) for r in roots if str(r).strip()]

    used_files: Dict[str, str] = {}
    rows: List["base.np.ndarray"] = []
    task_ids: List[str] = []
    seen_ids: Set[str] = set()

    for bench in train_benchmarks:
        bench_key = str(bench)
        required_ids = [str(tid) for tid in list(required_ids_by_bench.get(bench_key, []))]
        if not required_ids:
            raise RuntimeError(f"{bench_key} training benchmark: 0 item_ids remain after response-driven filtering.")

        found = base.find_compatible_embeddings_cache(
            preferred_paths=[],
            search_roots=roots,
            backbone=str(backbone),
            max_length=int(max_length),
            instruction_sig=str(instruction_sig),
            expected_n_items=int(len(required_ids)),
            require_single_dataset_source=True,
        )
        if found is None:
            return None
        cache_path, cache_task_ids, cache_X, _ = found
        used_files[bench_key] = str(cache_path)
        idx_by_id = {str(tid): int(i) for i, tid in enumerate(cache_task_ids)}
        for tid in required_ids:
            if tid not in idx_by_id:
                return None
            if tid in seen_ids:
                continue
            seen_ids.add(tid)
            task_ids.append(tid)
            rows.append(cache_X[int(idx_by_id[tid])].astype(base.np.float32, copy=False))

    if not task_ids or not rows:
        return None
    X = base.np.stack(rows, axis=0).astype(base.np.float32)
    return task_ids, X, used_files

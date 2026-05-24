"""Cross-benchmark Ridge predictor for held-out task difficulties.

For each (target_dataset, size, seed) cell, the multi-benchmark IRT gives us
`diff_by_item` for items it trained on (full 3 non-target benchmarks + target
subset). To estimate per-agent success on the *held-out* target tasks we need
their difficulty too, but they have no responses in training, so the IRT
can't recover their `b`.

This module trains a block Ridge regression (DeepSeek embedding block +
LLM-judge feature block) on the (b, features) pairs of every IRT-training
item across all 4 benchmarks, then predicts `b̂` for each held-out target
task.

This module does NOT define any new Ridge code. It wires together the
existing block-Ridge stack from `predict_question_difficulty.py`:
  - `_try_load_concat_embeddings_from_single_benchmark_caches`
  - `_build_judge_index`, `_load_judge_vector`
  - `_parse_alpha_list`, `_select_block_alphas_inner_cv`,
    `_fit_block_ridge`, `_predict_block_ridge`
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from experiment_subset_extrapolation.multibench_trainer import (
    ALL_BENCHES,
    DATASET_TO_BENCH,
    MultiBenchIRT,
    ROOT,
    get_full_item_ids,
)


# Defaults match experiment_agent_features.predict_question_difficulty_multi_benchmark.
DEFAULT_ALPHA_GRID = "1e-4,1e-3,1e-2,1e-1,1,10,100,1000,10000"
DEFAULT_INNER_SPLITS = 5

JUDGE_FEATURE_DIRS: Dict[str, Path] = {
    "verified": ROOT / "llm_judge_features/defaults/swebench_verified/llm_judge_features.csv",
    "pro": ROOT / "llm_judge_features/defaults/swebench_pro/llm_judge_features.csv",
    "terminal_bench": ROOT / "llm_judge_features/defaults/terminalbench/llm_judge_features.csv",
    "gso": ROOT / "llm_judge_features/defaults/gso/llm_judge_features.csv",
}
# Per-benchmark item-id normalization (matches existing main()):
# verified, pro, gso normalize; terminal_bench does not.
JUDGE_NORMALIZE: Dict[str, bool] = {
    "verified": True,
    "pro": True,
    "terminal_bench": False,
    "gso": True,
}

EMBEDDING_BACKBONE = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
EMBEDDING_MAX_LENGTH = 8192
EMBEDDING_LAYER = -1


@dataclass
class _BenchFeatureCtx:
    bench: str
    judge_features_dir: str
    judge_index: Dict[str, str]
    normalize_item_ids: bool


class HeldoutDifficultyPredictor:
    """Wraps the existing cross-benchmark block-Ridge for one fold.

    Lifecycle:
        p = HeldoutDifficultyPredictor(target_dataset, seed)
        p.fit(irt)                          # trains the block Ridge
        b_hat = p.predict(heldout_task_ids) # per-task predicted difficulty
    """

    def __init__(self, target_dataset: str, seed: int):
        if target_dataset not in DATASET_TO_BENCH:
            raise ValueError(f"Unknown target_dataset {target_dataset!r}")
        self.target_dataset = target_dataset
        self.target_bench = DATASET_TO_BENCH[target_dataset]
        self.seed = int(seed)

        self._bench_ctx: Dict[str, _BenchFeatureCtx] = {}
        self._task_to_bench: Dict[str, str] = {}
        self._emb_X: Optional[np.ndarray] = None
        self._emb_id_to_row: Dict[str, int] = {}
        self._ridge_state: Optional[dict] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, irt: MultiBenchIRT) -> None:
        """Build training matrices and fit block Ridge on all IRT items."""
        # Lazy imports — only the worker process needs the ML deps.
        from experiment_agent_features import predict_question_difficulty as base
        from experiment_agent_features.predict_question_difficulty_multi_benchmark import (
            _try_load_concat_embeddings_from_single_benchmark_caches,
        )

        # 1. Set up per-benchmark judge feature contexts and task->bench map.
        # Each training item is routed to its owning benchmark's feature source.
        self._bench_ctx.clear()
        self._task_to_bench.clear()
        for bench in ALL_BENCHES:
            jdir = str(JUDGE_FEATURE_DIRS[bench])
            self._bench_ctx[bench] = _BenchFeatureCtx(
                bench=bench,
                judge_features_dir=jdir,
                judge_index=base._build_judge_index(
                    jdir, normalize_item_ids=JUDGE_NORMALIZE[bench]
                ),
                normalize_item_ids=JUDGE_NORMALIZE[bench],
            )
            for tid in irt.training_item_ids_by_bench.get(bench, []):
                self._task_to_bench[str(tid)] = bench

        # 2. Load embeddings concatenated across benchmarks.
        # `_try_load_concat_embeddings_from_single_benchmark_caches` matches
        # each per-benchmark cache by *exact* item count (the canonical
        # single-benchmark caches contain the full task universe), so we must
        # pass the FULL per-benchmark item list — not just the IRT-training
        # subset. We later index by task_id, so extra items don't hurt; held-
        # out target tasks need their embeddings looked up too.
        required_ids_by_bench: Dict[str, List[str]] = {
            b: sorted(get_full_item_ids(b)) for b in ALL_BENCHES
        }

        instr_sig = base.prompt_signature(base.DIFFICULTY_INSTRUCTION)
        loaded = _try_load_concat_embeddings_from_single_benchmark_caches(
            train_benchmarks=list(required_ids_by_bench.keys()),
            required_ids_by_bench=required_ids_by_bench,
            out_dir=str(ROOT),
            backbone=EMBEDDING_BACKBONE,
            max_length=EMBEDDING_MAX_LENGTH,
            embedding_layer=EMBEDDING_LAYER,
            instruction_sig=instr_sig,
        )
        if loaded is None:
            raise RuntimeError(
                "Failed to load concatenated benchmark embeddings. Check that all 4 "
                f"per-benchmark embedding caches exist under {ROOT}/embeddings/{{benchmark}}/ "
                "and were generated with the DeepSeek-R1-Distill-Qwen-32B backbone."
            )
        emb_task_ids, X_emb_all, _ = loaded
        self._emb_X = X_emb_all
        self._emb_id_to_row = {str(tid): int(i) for i, tid in enumerate(emb_task_ids)}

        # 3. Assemble training (X_emb, X_judge, y) over items that have a b
        # value AND both feature blocks.
        train_emb_rows: List[np.ndarray] = []
        train_judge_rows: List[np.ndarray] = []
        train_y: List[float] = []
        skipped_no_emb = 0
        skipped_no_judge = 0

        for tid, b in irt.diff_by_item.items():
            tid_s = str(tid)
            row = self._emb_id_to_row.get(tid_s)
            if row is None:
                skipped_no_emb += 1
                continue
            jv = self._judge_vector(tid_s)
            if jv is None:
                skipped_no_judge += 1
                continue
            train_emb_rows.append(self._emb_X[row])
            train_judge_rows.append(jv)
            train_y.append(float(b))

        if not train_y:
            raise RuntimeError(
                "Held-out Ridge had zero training items after feature filtering "
                f"(skipped_no_emb={skipped_no_emb}, skipped_no_judge={skipped_no_judge})."
            )

        X_emb_train = np.stack(train_emb_rows, axis=0).astype(np.float64)
        X_judge_train = np.stack(train_judge_rows, axis=0).astype(np.float64)
        y_train = np.asarray(train_y, dtype=np.float64)

        # 4. Inner CV alpha selection + final fit.
        alphas = base._parse_alpha_list(DEFAULT_ALPHA_GRID)
        alpha_emb, alpha_judge, _ = base._select_block_alphas_inner_cv(
            X_emb=X_emb_train,
            X_judge=X_judge_train,
            y=y_train,
            alphas_emb=alphas,
            alphas_judge=alphas,
            inner_splits=DEFAULT_INNER_SPLITS,
            seed=self.seed,
        )
        self._ridge_state = base._fit_block_ridge(
            X_emb=X_emb_train,
            X_judge=X_judge_train,
            y=y_train,
            alpha_emb=float(alpha_emb),
            alpha_judge=float(alpha_judge),
        )

    def predict(self, heldout_task_ids: List[str]) -> Dict[str, float]:
        """Return {task_id: predicted_b} for each held-out target task."""
        if self._ridge_state is None or self._emb_X is None:
            raise RuntimeError("HeldoutDifficultyPredictor.fit() must be called before predict().")

        from experiment_agent_features import predict_question_difficulty as base

        # Held-out tasks belong to the target benchmark; route their features
        # through that benchmark's judge source.
        ctx = self._bench_ctx[self.target_bench]
        emb_rows: List[np.ndarray] = []
        judge_rows: List[np.ndarray] = []
        kept_ids: List[str] = []
        missing_emb: List[str] = []
        missing_judge: List[str] = []

        for tid in heldout_task_ids:
            tid_s = str(tid)
            row = self._emb_id_to_row.get(tid_s)
            if row is None:
                missing_emb.append(tid_s)
                continue
            jv = base._load_judge_vector(
                tid_s,
                features_dir=ctx.judge_features_dir,
                feature_names=base.JUDGE_FEATURE_NAMES,
                index=ctx.judge_index,
                normalize_item_ids=ctx.normalize_item_ids,
            )
            if jv is None:
                missing_judge.append(tid_s)
                continue
            emb_rows.append(self._emb_X[row])
            judge_rows.append(jv)
            kept_ids.append(tid_s)

        if missing_emb or missing_judge:
            raise RuntimeError(
                f"Held-out features missing on target benchmark {self.target_bench!r}: "
                f"{len(missing_emb)} tasks without embeddings (first: {missing_emb[:3]}), "
                f"{len(missing_judge)} tasks without judge features (first: {missing_judge[:3]})."
            )

        X_emb = np.stack(emb_rows, axis=0).astype(np.float64)
        X_judge = np.stack(judge_rows, axis=0).astype(np.float64)
        preds = base._predict_block_ridge(self._ridge_state, X_emb=X_emb, X_judge=X_judge)

        return {tid: float(p) for tid, p in zip(kept_ids, preds)}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _judge_vector(self, task_id: str) -> Optional[np.ndarray]:
        bench = self._task_to_bench.get(task_id)
        if bench is None:
            return None
        ctx = self._bench_ctx[bench]
        from experiment_agent_features import predict_question_difficulty as base

        return base._load_judge_vector(
            task_id,
            features_dir=ctx.judge_features_dir,
            feature_names=base.JUDGE_FEATURE_NAMES,
            index=ctx.judge_index,
            normalize_item_ids=ctx.normalize_item_ids,
        )

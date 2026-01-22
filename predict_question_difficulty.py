#!/usr/bin/env python3
"""
Embed SWE-bench-style tasks using inputs of the form:

  question statement + solution + instruction

and fit a regression model to predict per-question difficulty.

This script trains an IRT model **per CV fold** (no leakage):

For each of K folds over items/tasks:
  - Train an IRT model (1PL) using ONLY responses for items in the K-1 training folds.
  - Use the IRT-trained item difficulties (b) on the training items as supervision to fit a
    regression model from embeddings -> difficulty.
  - Predict difficulty for the held-out fold items (out-of-fold predictions).
  - Evaluate held-out ROC-AUC on the held-out fold using:
        p(success) = sigmoid(theta_subject - z_item_pred)
    where theta_subject comes from the fold's IRT training (fit on train items only).

We intentionally do NOT compute R^2 on held-out items since they do not have IRT-derived
difficulty parameters from that fold's IRT training (no leakage).

Example:
  python /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/predict_question_difficulty.py \
    --dataset_name princeton-nlp/SWE-bench_Verified --split test \
    --agent_results /path/to/subject_responses.jsonl \
    --cv_folds 5 \
    --irt_epochs 5000 --irt_device cuda \
    --backbone Qwen/Qwen2.5-Coder-14B --max_length 1024 --batch_size 1 --device_map auto \
    --out_dir /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/out/qwen25coder14b_irt_cv
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple


def _require(pkg: str) -> None:
    try:
        __import__(pkg)
    except Exception as e:
        raise RuntimeError(
            f"Missing dependency '{pkg}'. Please install requirements (see "
            f"`fulcrum/fellowship/trajectory_embedding_requirements.txt`). Original error: {e}"
        ) from e


_require("numpy")
_require("torch")
_require("transformers")
_require("tqdm")
_require("sklearn")
_require("datasets")
_require("huggingface_hub")

import numpy as np
import torch
from datasets import load_dataset  # type: ignore
from huggingface_hub import hf_hub_download
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, FineGrainedFP8Config, PreTrainedTokenizerFast
import inspect
from typing import Any
import math


def seed_everything(seed: int, *, deterministic: bool) -> None:
    """
    Best-effort reproducibility across python/numpy/torch/transformers.

    Notes:
    - Some GPU kernels (e.g. flash attention) can be nondeterministic.
    - PYTHONHASHSEED is set best-effort (it is most reliable when set before process start).
    """
    s = int(seed)
    try:
        os.environ.setdefault("PYTHONHASHSEED", str(s))
    except Exception:
        pass

    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(s)
        except Exception:
            pass

    # Transformers helper (if available).
    try:
        from transformers import set_seed as _hf_set_seed  # type: ignore

        _hf_set_seed(s)
    except Exception:
        pass

    if deterministic:
        set_torch_determinism(True)
        # Reduce numeric drift from TF32 on Ampere+ GPUs.
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
        except Exception:
            pass
        try:
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass


def set_torch_determinism(enabled: bool) -> None:
    """
    Toggle PyTorch deterministic algorithm behavior (best-effort).
    """
    on = bool(enabled)
    try:
        torch.use_deterministic_algorithms(on, warn_only=True)
    except TypeError:
        try:
            torch.use_deterministic_algorithms(on)
        except Exception:
            pass
    except Exception:
        pass
    try:
        torch.backends.cudnn.deterministic = on
        torch.backends.cudnn.benchmark = (not on)
    except Exception:
        pass


def _load_tokenizer(backbone: str, *, trust_remote_code: bool) -> Any:
    """
    Load a tokenizer for `backbone`.

    Some model repos publish a `tokenizer_config.json` with a legacy/removed
    tokenizer class name (e.g. `"tokenizer_class": "TokenizersBackend"`), which
    causes `AutoTokenizer.from_pretrained()` to fail even though a usable
    `tokenizer.json` is present. In that case, fall back to loading the fast
    tokenizer directly from `tokenizer.json`.
    """
    try:
        return AutoTokenizer.from_pretrained(backbone, trust_remote_code=trust_remote_code)
    except ValueError as e:
        msg = str(e)
        if "TokenizersBackend" not in msg:
            raise

        # Fallback: instantiate a fast tokenizer directly from tokenizer.json.
        tok_json = hf_hub_download(repo_id=backbone, filename="tokenizer.json")
        tok_cfg_path = None
        try:
            tok_cfg_path = hf_hub_download(repo_id=backbone, filename="tokenizer_config.json")
        except Exception:
            tok_cfg_path = None

        tok_kwargs: Dict[str, Any] = {"tokenizer_file": tok_json}
        extra_special_tokens: Optional[List[str]] = None
        if tok_cfg_path is not None and os.path.exists(tok_cfg_path):
            try:
                with open(tok_cfg_path, "r") as f:
                    cfg = json.load(f)
                for k in ("bos_token", "eos_token", "unk_token", "pad_token"):
                    if isinstance(cfg.get(k), str) and cfg.get(k):
                        tok_kwargs[k] = cfg[k]
                if isinstance(cfg.get("model_max_length"), int):
                    tok_kwargs["model_max_length"] = cfg["model_max_length"]
                if isinstance(cfg.get("extra_special_tokens"), list):
                    extra_special_tokens = [str(x) for x in cfg["extra_special_tokens"]]
            except Exception:
                # If parsing fails, proceed with defaults from tokenizer.json.
                pass

        tok = PreTrainedTokenizerFast(**tok_kwargs)
        if extra_special_tokens:
            # Prefer marking existing tokens as special; if any are missing, HF will add them.
            tok.additional_special_tokens = extra_special_tokens
        return tok


DIFFICULTY_INSTRUCTION = (
    "How difficult is the above task for a coding agent? Please output one floating-point number from 0 (very easy) to 1 (very hard). Your difficulty score:\n"
)

# Match ID normalization used when building the IRT dataset from agent runs:
# - strip leading "instance_"
# - strip trailing "-v..." (including "-vc<hash>" and "-vnan")
_V_SUFFIX_RE = re.compile(r"-v.*$")


def normalize_swebench_item_id(raw_item_id: str) -> str:
    s = str(raw_item_id or "").strip()
    if s.startswith("instance_"):
        s = s[len("instance_") :]
    s = _V_SUFFIX_RE.sub("", s)
    return s.strip()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj: dict) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Backwards compatible with older scikit-learn versions (no `squared=` kwarg).
    mse = mean_squared_error(y_true, y_pred)
    return float(np.sqrt(mse))


def _pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = (np.sqrt((x * x).sum()) * np.sqrt((y * y).sum()))
    if denom <= 0:
        return float("nan")
    return float((x * y).sum() / denom)


def stable_split_ids(ids: Sequence[str], test_fraction: float, seed: int) -> Tuple[List[int], List[int]]:
    """
    Deterministic split by hashing ids. Returns train indices, test indices.
    """
    if not (0.0 < test_fraction < 1.0):
        raise ValueError("test_fraction must be between 0 and 1")

    n_test_target = int(round(len(ids) * float(test_fraction)))

    xs: List[Tuple[float, int]] = []
    for i, s in enumerate(ids):
        h = hashlib.md5((str(s) + f"::{seed}").encode("utf-8")).hexdigest()
        x = int(h[:8], 16) / float(16**8)
        xs.append((x, i))
    xs.sort()

    test_set = set([i for _, i in xs[:n_test_target]])
    test = sorted(test_set)
    train = [i for i in range(len(ids)) if i not in test_set]
    return train, test


def iter_subject_responses_jsonl(path: str) -> Iterator[Tuple[str, Dict[str, int]]]:
    """
    Yield (subject_id, responses) from a JSONL file with schema:
      {"subject_id": "...", "responses": {"task_id": 0/1, ...}}

    Normalizes item ids.
    """
    p = str(path or "").strip()
    if not p:
        return
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            s = (line or "").strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            sid = str(obj.get("subject_id", "") or "").strip()
            resp = obj.get("responses", {}) or {}
            if not sid or not isinstance(resp, dict):
                continue
            out: Dict[str, int] = {}
            for raw_id, v in resp.items():
                tid = normalize_swebench_item_id(str(raw_id))
                if not tid:
                    continue
                try:
                    out[tid] = int(v)
                except Exception:
                    out[tid] = 1 if v else 0
            if out:
                yield sid, out


def load_all_responses(path: str) -> List[Tuple[str, Dict[str, int]]]:
    """
    Materialize all responses from a JSONL, normalizing item ids.
    """
    out: List[Tuple[str, Dict[str, int]]] = []
    for sid, resp in iter_subject_responses_jsonl(path):
        if resp:
            out.append((sid, resp))
    return out


def compute_zero_success_items(all_responses: List[Tuple[str, Dict[str, int]]]) -> List[str]:
    """
    Items with 0 successes across all provided subjects.
    """
    counts: Dict[str, int] = {}
    seen: Set[str] = set()
    for _, resp in all_responses:
        for tid, v in resp.items():
            seen.add(tid)
            counts[tid] = counts.get(tid, 0) + int(v)
    return sorted([tid for tid in seen if counts.get(tid, 0) == 0])


def write_filtered_responses_jsonl(
    *,
    all_responses: List[Tuple[str, Dict[str, int]]],
    item_ids: Sequence[str],
    out_path: str,
) -> Tuple[int, int]:
    """
    Write a py_irt-compatible JSONL with responses restricted to `item_ids`.

    Policy:
    - Include only subjects with at least one observed response among `item_ids`.
    - Write a complete response dict over `item_ids`, filling missing with 0.

    Returns: (n_subjects_written, n_items)
    """
    ensure_dir(os.path.dirname(out_path) or ".")
    items = [normalize_swebench_item_id(x) for x in list(item_ids)]
    items = [x for x in items if x]
    item_set = set(items)

    n_written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for sid, resp in all_responses:
            present = {k: int(v) for k, v in resp.items() if k in item_set}
            if not present:
                continue
            complete = {tid: int(present.get(tid, 0)) for tid in items}
            f.write(json.dumps({"subject_id": sid, "responses": complete}) + "\n")
            n_written += 1
    return n_written, len(items)


def _compute_binary_auroc(scores: List[float], labels: List[int]) -> float:
    """
    ROC-AUC over binary labels. Returns NaN if undefined (e.g. only one class present).
    Mirrors `auroc.py`'s behavior.
    """
    if len(scores) == 0:
        return float("nan")
    uniq = set(int(x) for x in labels)
    if len(uniq) < 2:
        return float("nan")
    _require("torchmetrics")
    from torchmetrics import AUROC  # type: ignore

    auroc = AUROC(task="binary")
    s = torch.tensor(scores, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    return float(auroc(s, y).item())


def last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # last_hidden_state: [B, T, H], attention_mask: [B, T]
    lengths = attention_mask.sum(dim=1).clamp(min=1)  # [B]
    idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, last_hidden_state.size(-1))
    return last_hidden_state.gather(dim=1, index=idx).squeeze(1)  # [B, H]

def train_irt_1pl(
    *,
    responses_jsonl: str,
    epochs: int,
    device: str,
    seed: int,
    out_dir: str,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Train a 1PL IRT model via `py_irt` on the provided response JSONL.

    Returns:
      - theta_by_subject
      - diff_by_item (b)
    """
    _require("pyro")
    import pyro  # type: ignore

    from py_irt.config import IrtConfig  # type: ignore
    from py_irt.training import IrtModelTrainer  # type: ignore

    ensure_dir(str(out_dir))
    pyro.clear_param_store()

    cfg = IrtConfig(
        model_type="1pl",
        epochs=int(epochs),
        priors="hierarchical",
        dims=1,
        seed=int(seed),
    )
    trainer = IrtModelTrainer(data_path=str(responses_jsonl), config=cfg, verbose=False)
    trainer.train(device=str(device))

    trainer.save(os.path.join(out_dir, "parameters.json"))
    best = trainer.best_params or {}
    with open(os.path.join(out_dir, "best_parameters.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, sort_keys=True)

    ability = best.get("ability", [])
    diff = best.get("diff", [])
    subj_map = best.get("subject_ids", {})
    item_map = best.get("item_ids", {})

    theta_by_subject: Dict[str, float] = {}
    for i in range(len(ability)):
        sid = str(subj_map.get(i, "")).strip()
        if sid:
            theta_by_subject[sid] = float(ability[i])

    diff_by_item: Dict[str, float] = {}
    for i in range(len(diff)):
        tid = normalize_swebench_item_id(str(item_map.get(i, "")).strip())
        if tid:
            diff_by_item[tid] = float(diff[i])

    # Also write abilities.csv / items.csv for inspection.
    with open(os.path.join(out_dir, "abilities.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["subject_id", "theta"])
        w.writeheader()
        for sid, theta in sorted(theta_by_subject.items()):
            w.writerow({"subject_id": sid, "theta": float(theta)})

    with open(os.path.join(out_dir, "items.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["item_id", "b"])
        w.writeheader()
        for tid, b in sorted(diff_by_item.items()):
            w.writerow({"item_id": tid, "b": float(b)})

    return theta_by_subject, diff_by_item


def prompt_signature(instruction: str) -> str:
    """
    Short signature for the *actual* prompt formatting used for embeddings.
    """
    h = hashlib.sha1(str(instruction).encode("utf-8")).hexdigest()[:8]
    return f"qs_sol_instr_{h}"


def _sanitize_text(s: str) -> str:
    # Replace any literal ASCII control characters (0x00-0x1F) with spaces.
    # This helps avoid rare tokenizer/model issues and keeps cache reproducible.
    return "".join((" " if (ord(ch) < 32 and ch not in ("\n", "\t")) else ch) for ch in (s or ""))


def format_qs_solution_instruction(*, question_statement: str, solution: str, instruction: str) -> str:
    """
    Format model input with the requested ordering:
      question statement + solution + instruction
    """
    qs = _sanitize_text(str(question_statement or "")).strip()
    sol = _sanitize_text(str(solution or "")).strip()
    instr = _sanitize_text(str(instruction or "")).strip()
    return f"Task statement:\n{qs}\n\nSolution:\n{sol}\n\n{instr}".strip()


# GSO (and other OOD-style) benchmarks can store the task statement as a *test script*
# (`prob_script`) which is meant to be wrapped into a fixed "spec test" prompt template
# before embedding. This mirrors the multi-benchmark script's behavior so embeddings are
# identical by construction.
_GSO_PROMPT_TEMPLATE = """I’ve uploaded a python code repository in the directory workspace_dir_name. Consider the
following test script showing an example usage of the repository:
<test_script>
{SPEC_TEST}
</test_script>
Can you help me implement the necessary changes to the repository so that the runtime of
the <test_script> is optimized? Basic guidelines:
1. Your task is to make changes to non-test files in the /workspace directory to improve the
performance of the <test_script>.
2. Make changes while ensuring the repository is functionally equivalent to the original.
3. Do not overoptimize for just the specific inputs in <test_script>. Make general perfor-
mance improvements for the usage scenario shown.
4. You may need to rebuild the repo for your changes to take effect before testing. Some
rebuilds may take time to run, so be patient with running them.
Follow these steps to improve performance:
1. As a first step, explore the repository structure.
2. Create a script in the /workspace directory (e.g., /workspace/test_opt.py) to reproduce and
time the example, then execute it with python /workspace/<filename.py>.
3. Edit the source code of the repository to improve performance.
4. Rebuild and rerun your script to confirm that performance has improved.
"""


def _wrap_gso_problem_statement(prob_script: str) -> str:
    return _GSO_PROMPT_TEMPLATE.format(SPEC_TEST=str(prob_script or "").strip())


def _is_gso_dataset(*, dataset_name: str, dataset_path: str) -> bool:
    """
    Heuristic: detect GSO-style datasets (e.g. HF `gso-bench/gso` or local `...gso...jsonl`).
    """
    s = " ".join([str(dataset_name or ""), str(dataset_path or "")]).lower()
    return ("gso-bench" in s) or bool(re.search(r"(^|[^a-z0-9])gso([^a-z0-9]|$)", s))


@dataclass(frozen=True)
class ItemRecord:
    item_id: str
    question_statement: str
    solution: str


def iter_swebench_verified_items(*, dataset_name: str, split: str) -> Iterator[ItemRecord]:
    ds = load_dataset(str(dataset_name), split=str(split))
    n_total = len(ds)
    if n_total == 0:
        raise RuntimeError(f"Loaded empty dataset: {dataset_name} split={split}")

    # Try to extract the solution patch robustly across variants.
    solution_keys = ["patch", "gold_patch", "resolved_patch", "solution", "diff", "fix_patch"]
    for i in range(int(n_total)):
        row = ds[int(i)]
        item_id = str(row.get("instance_id", "")).strip()
        qs = str(row.get("problem_statement", "") or "")
        sol = ""
        for k in solution_keys:
            v = row.get(k, None)
            if v is None:
                continue
            s = str(v)
            if s.strip():
                sol = s
                break
        if not item_id:
            # If missing instance_id, fall back to a stable synthetic id.
            item_id = f"row_{int(i)}"
        yield ItemRecord(item_id=item_id, question_statement=qs, solution=sol)


def iter_swebench_items(
    *,
    dataset_name: str,
    split: str,
    dataset_path: str,
) -> Iterator[ItemRecord]:
    """
    Load SWE-bench-style tasks and yield ItemRecords.

    Supports exactly one source:
    - HuggingFace dataset hub: pass --dataset_name <org/name>
    - Local JSON/JSONL via datasets: pass --dataset_path /path/to/file.jsonl

    Field extraction is best-effort:
    - item_id: instance_id | task_id | id
    - question_statement: problem_statement | statement | description
        - GSO-style tasks: `prob_script` (wrapped into a fixed prompt template)
    - solution/patch: patch | gold_patch | resolved_patch | solution | diff | fix_patch
        - GSO-style tasks: `gt_diff`
    """
    dataset_name = str(dataset_name or "").strip()
    dataset_path = str(dataset_path or "").strip()
    if bool(dataset_name) and bool(dataset_path):
        raise ValueError("Provide only one of --dataset_name or --dataset_path (single-benchmark mode).")
    if not dataset_name and not dataset_path:
        raise ValueError("No dataset provided (set --dataset_name or --dataset_path).")

    is_gso = _is_gso_dataset(dataset_name=dataset_name, dataset_path=dataset_path)
    if dataset_path:
        src_name = f"json:{dataset_path}"
        d = load_dataset("json", data_files=str(dataset_path), split="train")
        src_split = "train"
    else:
        src_name = str(dataset_name)
        d = load_dataset(str(dataset_name), split=str(split))
        src_split = str(split)

    n_total = int(len(d))
    if n_total == 0:
        raise RuntimeError(f"Loaded empty dataset: {src_name} split={src_split}")

    solution_keys = ["patch", "gold_patch", "resolved_patch", "solution", "diff", "fix_patch"]
    if is_gso:
        solution_keys = ["gt_diff"] + solution_keys
    id_keys = ["instance_id", "task_id", "id"]
    qs_keys = ["problem_statement", "statement", "description"]
    if is_gso:
        qs_keys = ["prob_script"] + qs_keys

    for i in range(n_total):
        row = d[int(i)]
        item_id = ""
        for k in id_keys:
            v = row.get(k, None)
            if v is None:
                continue
            s = str(v).strip()
            if s:
                item_id = normalize_swebench_item_id(s)
                break
        qs = ""
        qs_key_used = ""
        for k in qs_keys:
            v = row.get(k, None)
            if v is None:
                continue
            s = str(v)
            if str(s).strip():
                qs = s
                qs_key_used = str(k)
                break
        if is_gso and qs_key_used == "prob_script":
            qs = _wrap_gso_problem_statement(qs)
        sol = ""
        for k in solution_keys:
            v = row.get(k, None)
            if v is None:
                continue
            s = str(v)
            if s.strip():
                sol = s
                break
        if not item_id:
            # If missing id, fall back to a stable synthetic id within the dataset.
            item_id = f"row_{int(i)}"
        yield ItemRecord(item_id=item_id, question_statement=qs, solution=sol)


def load_items_by_ids(
    *,
    dataset_name: str,
    split: str,
    dataset_path: str,
    item_ids: Sequence[str],
) -> Tuple[List[ItemRecord], List[str]]:
    """
    Load SWE-bench-style tasks and return ItemRecords for a specific set of item_ids.

    - Preserves the order of `item_ids`.
    - Returns (items_found_in_order, missing_ids).
    """
    want = [normalize_swebench_item_id(x) for x in list(item_ids)]
    want = [x for x in want if str(x).strip()]
    want_set = set(want)
    if not want:
        return [], []

    dataset_name = str(dataset_name or "").strip()
    dataset_path = str(dataset_path or "").strip()
    if bool(dataset_name) and bool(dataset_path):
        raise ValueError("Provide only one of dataset_name or dataset_path (single-benchmark mode).")
    if not dataset_name and not dataset_path:
        raise ValueError("No dataset provided (set dataset_name or dataset_path).")

    if dataset_path:
        name = f"json:{dataset_path}"
        ds = load_dataset("json", data_files=str(dataset_path), split="train")
        ds_split = "train"
    else:
        name = str(dataset_name)
        ds = load_dataset(str(dataset_name), split=str(split))
        ds_split = str(split)

    n_total = int(len(ds))
    if n_total == 0:
        raise RuntimeError(f"Loaded empty dataset: {name} split={ds_split}")

    is_gso = _is_gso_dataset(dataset_name=dataset_name, dataset_path=dataset_path)
    solution_keys = ["patch", "gold_patch", "resolved_patch", "solution", "diff", "fix_patch"]
    if is_gso:
        solution_keys = ["gt_diff"] + solution_keys
    id_keys = ["instance_id", "task_id", "id"]
    qs_keys = ["problem_statement", "statement", "description"]
    if is_gso:
        qs_keys = ["prob_script"] + qs_keys

    found: Dict[str, ItemRecord] = {}
    # Scan dataset once; stop early when all requested ids have been found.
    for i in range(n_total):
        row = ds[int(i)]
        item_id = ""
        for k in id_keys:
            v = row.get(k, None)
            if v is None:
                continue
            s = str(v).strip()
            if s:
                item_id = normalize_swebench_item_id(s)
                break
        if not item_id or item_id not in want_set or item_id in found:
            continue

        qs = ""
        qs_key_used = ""
        for k in qs_keys:
            v = row.get(k, None)
            if v is None:
                continue
            s = str(v)
            if str(s).strip():
                qs = s
                qs_key_used = str(k)
                break
        if is_gso and qs_key_used == "prob_script":
            qs = _wrap_gso_problem_statement(qs)
        sol = ""
        for k in solution_keys:
            v = row.get(k, None)
            if v is None:
                continue
            s = str(v)
            if s.strip():
                sol = s
                break

        found[item_id] = ItemRecord(item_id=item_id, question_statement=qs, solution=sol)
        if len(found) >= len(want_set):
            break

    items = [found[tid] for tid in want if tid in found]
    missing = [tid for tid in want if tid not in found]
    return items, missing


def _try_load_model_class(backbone: str, *, trust_remote_code: bool, model_kwargs: dict):
    """
    Load a HF model in a way that supports both LMs and VLMs.

    Why: some VLM checkpoints (e.g. Qwen3-VL) are stored under a *generation* class, so
    loading via plain AutoModel can produce partially-uninitialized weights (bad embeddings).
    """
    # Some remote-code checkpoints expect symbols that may have been renamed across
    # transformers versions. Provide minimal shims so imports succeed.
    #
    # Example seen in practice: moonshotai/Kimi-VL remote code importing
    # `from transformers.activations import PytorchGELUTanh` while newer transformers
    # exposes `GELUTanh`.
    try:
        import transformers.activations as _act  # type: ignore

        if not hasattr(_act, "PytorchGELUTanh") and hasattr(_act, "GELUTanh"):
            _act.PytorchGELUTanh = _act.GELUTanh  # type: ignore[attr-defined]
    except Exception:
        # Best-effort only; failure here should not prevent loading other model types.
        pass

    # Prefer the modern image-text-to-text wrapper, then the deprecated vision2seq,
    # then CausalLM, then bare AutoModel. We import lazily so older transformers
    # versions still work.
    errors = []
    try:
        from transformers import AutoModelForImageTextToText  # type: ignore

        return AutoModelForImageTextToText.from_pretrained(
            backbone, trust_remote_code=trust_remote_code, **model_kwargs
        )
    except Exception as e:
        errors.append(("AutoModelForImageTextToText", e))
    try:
        from transformers import AutoModelForVision2Seq  # type: ignore

        return AutoModelForVision2Seq.from_pretrained(backbone, trust_remote_code=trust_remote_code, **model_kwargs)
    except Exception as e:
        errors.append(("AutoModelForVision2Seq", e))
    try:
        from transformers import AutoModelForCausalLM  # type: ignore

        return AutoModelForCausalLM.from_pretrained(backbone, trust_remote_code=trust_remote_code, **model_kwargs)
    except Exception as e:
        errors.append(("AutoModelForCausalLM", e))
    try:
        return AutoModel.from_pretrained(backbone, trust_remote_code=trust_remote_code, **model_kwargs)
    except Exception as e:
        errors.append(("AutoModel", e))

    msg = "Failed to load model with any supported auto class:\n" + "\n".join(
        [f"- {name}: {type(err).__name__}: {err}" for name, err in errors]
    )
    raise RuntimeError(msg)


def _select_text_submodel(model: torch.nn.Module) -> torch.nn.Module:
    """
    For VLM wrappers, prefer the language/text tower for text-only embedding.
    """
    for attr in ("language_model", "text_model"):
        m = getattr(model, attr, None)
        if isinstance(m, torch.nn.Module):
            return m
    # Many HF *ForCausalLM wrappers store the base model on `.model`.
    m = getattr(model, "model", None)
    if isinstance(m, torch.nn.Module) and hasattr(m, "get_input_embeddings"):
        return m
    return model


def _get_hidden_states_tuple(outputs):
    """
    Best-effort extraction of a tuple/list of per-layer hidden states from HF outputs.
    Returns None if not available.
    """
    for attr in ("hidden_states", "encoder_hidden_states", "decoder_hidden_states"):
        if hasattr(outputs, attr):
            hs = getattr(outputs, attr)
            if hs is not None:
                return hs
    return None


def _extract_hidden_state(outputs, *, embedding_layer: int) -> torch.Tensor:
    """
    Normalize different HF output types to a [B, T, H] hidden state tensor.

    If embedding_layer == -1, uses the last layer by default.
    """
    layer = int(embedding_layer)

    # Fast path: if caller wants "last" and last_hidden_state is provided, use it.
    if layer == -1 and hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state
    if layer == -1 and hasattr(outputs, "encoder_last_hidden_state") and outputs.encoder_last_hidden_state is not None:
        return outputs.encoder_last_hidden_state

    hs = _get_hidden_states_tuple(outputs)
    if hs is not None:
        try:
            return hs[layer]
        except Exception as e:
            raise RuntimeError(
                f"Requested embedding_layer={layer}, but model returned {len(hs)} hidden_states entries. "
                f"Try a value in [-{len(hs)}, {len(hs)-1}] or use --embedding_layer -1 for last."
            ) from e

    # If hidden_states weren't present, we can still sometimes use last_hidden_state for the last layer.
    if layer == -1 and hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state
    if layer == -1 and hasattr(outputs, "encoder_last_hidden_state") and outputs.encoder_last_hidden_state is not None:
        return outputs.encoder_last_hidden_state

    raise RuntimeError(
        "Model outputs did not expose hidden_states needed for selecting a layer. "
        "Try using --embedding_layer -1, and ensure the model supports output_hidden_states=True "
        "(and consider --trust_remote_code / a different backbone)."
    )


def embed_items(
    *,
    items: List[ItemRecord],
    backbone: str,
    trust_remote_code: bool,
    max_length: int,
    batch_size: int,
    device_map: str,
    torch_dtype: str,
    attn_implementation: str,
    instruction: str,
    embedding_layer: int,
) -> Tuple[List[str], Dict[str, np.ndarray], Dict[str, int], int]:
    """
    Returns:
      - ids_sorted
      - embeddings_by_id: {item_id -> embedding}
      - counts_by_id: {item_id -> text_len_chars}
      - embedding_dim
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = _load_tokenizer(backbone, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if torch_dtype == "auto":
        dtype_arg = "auto"
    elif torch_dtype in ("float16", "fp16"):
        dtype_arg = torch.float16
    elif torch_dtype in ("bfloat16", "bf16"):
        dtype_arg = torch.bfloat16
    elif torch_dtype in ("float32", "fp32"):
        dtype_arg = torch.float32
    else:
        raise ValueError(f"Unknown torch_dtype: {torch_dtype}")

    # transformers v5 prefers `dtype`; older versions used `torch_dtype`.
    # Use the signature to stay compatible across versions.
    fp_params = inspect.signature(AutoModel.from_pretrained).parameters
    if "dtype" in fp_params:
        model_kwargs = {"dtype": dtype_arg}
    else:
        model_kwargs = {"torch_dtype": dtype_arg}
    if device_map and device_map != "none":
        model_kwargs["device_map"] = device_map
    if attn_implementation and attn_implementation != "auto":
        model_kwargs["attn_implementation"] = attn_implementation

    # Some backbones (e.g. Devstral/Mistral3) ship pre-quantized FP8 weights.
    # On many HPC clusters, Triton can't JIT-compile its small CUDA/Python helper
    # because system Python headers aren't installed (missing `Python.h`), causing
    # a crash at first FP8 matmul. If this model advertises FP8 quantization, we
    # dequantize to bf16 during loading to avoid Triton entirely.
    #
    # Note: `quantization_config` is accepted via **kwargs in transformers, so it
    # typically won't appear in the function signature.
    try:
        cfg = AutoConfig.from_pretrained(backbone, trust_remote_code=trust_remote_code)
        qc = getattr(cfg, "quantization_config", None)
        if isinstance(qc, dict) and str(qc.get("quant_method", "")).lower() == "fp8":
            model_kwargs["quantization_config"] = FineGrainedFP8Config(dequantize=True)
    except Exception:
        pass

    model = _try_load_model_class(backbone, trust_remote_code=trust_remote_code, model_kwargs=model_kwargs)
    model.eval()
    if device_map in ("", "none", None):
        model.to(device)

    # For VLMs / generation wrappers, embed using the text tower when possible.
    text_model = _select_text_submodel(model)
    # Disable KV caching for embedding forward passes. Some remote-code models
    # (e.g. Kimi-VL) can be incompatible with newer transformers cache objects.
    for m in (model, text_model):
        cfg = getattr(m, "config", None)
        if cfg is not None and hasattr(cfg, "use_cache"):
            try:
                cfg.use_cache = False
            except Exception:
                pass

    try:
        embed_device = text_model.get_input_embeddings().weight.device
    except Exception:
        embed_device = device

    per_id: Dict[str, np.ndarray] = {}
    counts: Dict[str, int] = {}

    batch_ids: List[str] = []
    batch_texts: List[str] = []
    embedding_dim = 0

    def flush() -> None:
        nonlocal batch_ids, batch_texts, per_id, counts, embedding_dim
        if not batch_texts:
            return
        pairs = [(rid, txt) for rid, txt in zip(batch_ids, batch_texts) if str(txt).strip()]
        if not pairs:
            batch_ids = []
            batch_texts = []
            return
        batch_ids = [rid for rid, _ in pairs]
        batch_texts = [txt for _, txt in pairs]

        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        if int(input_ids.shape[1]) == 0:
            batch_ids = []
            batch_texts = []
            return

        input_ids = input_ids.to(embed_device)
        attention_mask = attention_mask.to(embed_device)

        with torch.no_grad():
            fwd_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,
            )
            # Avoid transformers DynamicCache / past_key_values codepaths unless needed.
            try:
                sig = inspect.signature(text_model.forward)  # type: ignore[attr-defined]
                if "use_cache" in sig.parameters:
                    fwd_kwargs["use_cache"] = False
            except Exception:
                # If we can't introspect, best-effort: many models accept use_cache anyway.
                fwd_kwargs["use_cache"] = False

            out = text_model(**fwd_kwargs)
            h = _extract_hidden_state(out, embedding_layer=int(embedding_layer))
            pooled = last_token_pool(h, attention_mask)
            pooled = pooled.detach().float().cpu().numpy()  # [B, H]

        embedding_dim = int(pooled.shape[1])
        for rid, vec, txt in zip(batch_ids, pooled, batch_texts):
            per_id[str(rid)] = vec.astype(np.float32, copy=False)
            counts[str(rid)] = int(len(str(txt)))

        batch_ids = []
        batch_texts = []

    for rec in tqdm(items, desc="embed_items"):
        txt = format_qs_solution_instruction(
            question_statement=rec.question_statement,
            solution=rec.solution,
            instruction=instruction,
        )
        if not txt.strip():
            continue
        batch_ids.append(rec.item_id)
        batch_texts.append(txt)
        if len(batch_texts) >= int(batch_size):
            flush()
    flush()

    ids_sorted = sorted(per_id.keys())
    return ids_sorted, per_id, counts, int(embedding_dim)


def _npz_scalar(value, default=None):
    """
    Robustly convert an NPZ entry to a Python scalar.

    Handles common patterns like:
    - np.array([x])  -> x
    - np.array(x)    -> x
    - ["x"] / [x]    -> x
    """
    if value is None:
        return default
    try:
        import numpy as _np  # local import; numpy is required already

        if isinstance(value, _np.ndarray):
            if value.shape == ():
                return value.item()
            if value.size == 1:
                return value.reshape(-1)[0].item()
            return value.tolist()
    except Exception:
        pass
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return default
        if len(value) == 1:
            return value[0]
        return list(value)
    return value


def _as_1d_float32(x: object) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    return a.astype(np.float32, copy=False)


def _as_float(x: object) -> float:
    try:
        v = np.asarray(x).reshape(-1)
        if v.size == 0:
            return float("nan")
        return float(v[0])
    except Exception:
        return float(x)  # type: ignore[arg-type]


def save_regression_weights(
    *,
    out_dir: str,
    model: Any,
    regressor_name: str,
    feature_dim: int,
    metadata: dict,
) -> Tuple[str, str]:
    """
    Save a minimal, sklearn-version-agnostic representation of the trained regressor.

    Writes:
      - regression_weights.json (metadata)
      - regression_weights.npz  (arrays: coef, intercept, scaler_mean, scaler_scale)
    """
    ensure_dir(out_dir)

    uses_scaler = False
    coef = np.zeros((0,), dtype=np.float32)
    intercept = np.zeros((1,), dtype=np.float32)
    scaler_mean = np.zeros((0,), dtype=np.float32)
    scaler_scale = np.ones((0,), dtype=np.float32)

    if isinstance(model, Pipeline):
        scaler = model.named_steps.get("scaler", None)
        reg = model.named_steps.get("ridge", None)
        if scaler is not None and hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
            uses_scaler = True
            scaler_mean = _as_1d_float32(getattr(scaler, "mean_"))
            scaler_scale = _as_1d_float32(getattr(scaler, "scale_"))
        if reg is not None and hasattr(reg, "coef_") and hasattr(reg, "intercept_"):
            coef = _as_1d_float32(getattr(reg, "coef_"))
            intercept = np.array([_as_float(getattr(reg, "intercept_"))], dtype=np.float32)
    else:
        # LinearRegression (no scaler in this script).
        if hasattr(model, "coef_") and hasattr(model, "intercept_"):
            coef = _as_1d_float32(getattr(model, "coef_"))
            intercept = np.array([_as_float(getattr(model, "intercept_"))], dtype=np.float32)

    if int(coef.size) != int(feature_dim):
        raise RuntimeError(
            f"Saved coef has dim={int(coef.size)} but expected feature_dim={int(feature_dim)}. "
            f"regressor={regressor_name}"
        )
    if uses_scaler and int(scaler_mean.size) != int(feature_dim):
        raise RuntimeError(
            f"Saved scaler_mean has dim={int(scaler_mean.size)} but expected feature_dim={int(feature_dim)}. "
            f"regressor={regressor_name}"
        )
    if uses_scaler and int(scaler_scale.size) != int(feature_dim):
        raise RuntimeError(
            f"Saved scaler_scale has dim={int(scaler_scale.size)} but expected feature_dim={int(feature_dim)}. "
            f"regressor={regressor_name}"
        )

    weights_npz = os.path.join(out_dir, "regression_weights.npz")
    np.savez_compressed(
        weights_npz,
        coef=coef.astype(np.float32, copy=False),
        intercept=intercept.astype(np.float32, copy=False),
        scaler_mean=scaler_mean.astype(np.float32, copy=False),
        scaler_scale=scaler_scale.astype(np.float32, copy=False),
    )

    weights_json = os.path.join(out_dir, "regression_weights.json")
    meta = dict(metadata or {})
    meta.update(
        {
            "regressor": str(regressor_name),
            "feature_dim": int(feature_dim),
            "uses_scaler": bool(uses_scaler),
            "weights_npz": str(weights_npz),
        }
    )
    save_json(weights_json, meta)
    return weights_json, weights_npz


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--dataset_name",
        type=str,
        default="princeton-nlp/SWE-bench_Verified",
        help="HF dataset repo to load (single source). Ignored if --dataset_path is set.",
    )
    p.add_argument("--split", type=str, default="test")
    p.add_argument(
        "--dataset_path",
        type=str,
        default="",
        help=(
            "Optional local JSON/JSONL dataset path (single source). If set, loads via "
            "datasets('json', data_files=..., split='train'). Overrides --dataset_name."
        ),
    )
    p.add_argument("--seed", type=int, default=0)
    # Fixed policy: we always seed non-IRT steps deterministically.

    p.add_argument("--backbone", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--max_length", type=int, default=8192)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--device_map", type=str, default="auto", help="HF device_map (e.g. auto). Use 'none' to force single-device .to(device).")
    p.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--attn_implementation", type=str, default="auto", help="e.g. auto, flash_attention_2")
    p.add_argument(
        "--embedding_layer",
        type=int,
        default=-1,
        help="Which hidden layer to pool embeddings from (0-based over returned hidden_states; negatives allowed). -1 means last.",
    )

    p.add_argument("--instruction", type=str, default=DIFFICULTY_INSTRUCTION, help="Instruction text appended last in the embedding input.")

    p.add_argument("--out_dir", type=str, default="/orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/out/swebench_verified")
    p.add_argument("--embeddings_cache", type=str, default="", help="Optional path to existing embeddings cache (.npz).")
    p.add_argument("--overwrite", action="store_true")

    p.add_argument(
        "--agent_results",
        type=str,
        default="/orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/out/chris_irt/swebench_verified_20251115_full.jsonl",
        help=(
            "Path to a JSONL file with per-subject responses of the form "
            "{'subject_id': ..., 'responses': {'task_id': 0/1, ...}}."
        ),
    )
    p.add_argument(
        "--include_zero_success",
        action="store_true",
        help="Include items with 0 successes in CV/IRT (not recommended; can destabilize IRT).",
    )
    p.add_argument("--irt_epochs", type=int, default=5000)
    p.add_argument("--irt_device", type=str, default="cuda", help="Device for IRT training (cuda or cpu).")
    # Fixed IRT policy: we seed IRT, but keep torch determinism disabled during IRT for stability.
    p.add_argument(
        "--regressor",
        type=str,
        default="ridge_cv",
        choices=["linear", "ridge", "ridge_cv"],
        help="Regression model (same options as trajectory-based script).",
    )
    p.add_argument("--ridge_alpha", type=float, default=10000.0)
    p.add_argument("--ridge_alphas", type=str, default="1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100,1000,10000")
    p.add_argument("--cv_folds", type=int, default=5)
    p.add_argument(
        "--inner_splits",
        type=int,
        default=5,
        help="Inner CV splits for RidgeCV (used when --regressor=ridge_cv). Will be capped by train size; must be >=2.",
    )

    args = p.parse_args(argv)
    ensure_dir(args.out_dir)
    seed_everything(int(args.seed), deterministic=True)

    dataset_name = str(args.dataset_name).strip()
    dataset_path = str(args.dataset_path).strip()
    if dataset_path:
        dataset_sources_str = f"json:{os.path.basename(dataset_path) or 'dataset.jsonl'}"
    else:
        dataset_sources_str = dataset_name or "princeton-nlp/SWE-bench_Verified"

    # Cache path derived from key settings.
    safe_backbone = str(args.backbone).replace("/", "__")
    ds_flag = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(dataset_sources_str))[:64]
    split_flag = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(args.split))[:32]
    instr_sig = prompt_signature(str(args.instruction))
    layer_flag = "" if int(args.embedding_layer) == -1 else f"__layer{int(args.embedding_layer)}"
    idnorm_flag = "__idnorm_instance-v1"
    emb_cache = str(args.embeddings_cache or "").strip()
    if not emb_cache:
        emb_cache = os.path.join(
            args.out_dir,
            f"embeddings__{safe_backbone}__pool-lasttoken{layer_flag}__qs-sol-instr__{instr_sig}{idnorm_flag}__{ds_flag}__{split_flag}__maxlen{int(args.max_length)}.npz",
        )

    # Load or compute embeddings.
    if os.path.exists(emb_cache) and not args.overwrite and not str(args.embeddings_cache or "").strip():
        data = np.load(emb_cache, allow_pickle=True)
        task_ids = [str(x) for x in list(data["task_ids"].tolist())]
        X = data["X"].astype(np.float32)
        counts_kind = str(_npz_scalar(data.get("counts_kind", None), "")) if "counts_kind" in data else ""
        cached_layer = int(_npz_scalar(data.get("embedding_layer", None), -1)) if "embedding_layer" in data else -1
        if int(args.embedding_layer) != int(cached_layer):
            raise RuntimeError(
                f"Embeddings cache was created with embedding_layer={cached_layer}, but you requested "
                f"--embedding_layer={int(args.embedding_layer)}. Use --overwrite, or pick a different cache file."
            )
        print(
            f"Loaded embeddings cache: {emb_cache} (n={len(task_ids)}, dim={X.shape[1]}, counts_kind={counts_kind or 'unknown'}, embedding_layer={cached_layer})"
        )
    elif os.path.exists(emb_cache) and not args.overwrite and str(args.embeddings_cache or "").strip():
        data = np.load(emb_cache, allow_pickle=True)
        task_ids = [str(x) for x in list(data["task_ids"].tolist())]
        X = data["X"].astype(np.float32)
        counts_kind = str(_npz_scalar(data.get("counts_kind", None), "")) if "counts_kind" in data else ""
        cached_layer = int(_npz_scalar(data.get("embedding_layer", None), -1)) if "embedding_layer" in data else -1
        if int(args.embedding_layer) != int(cached_layer):
            raise RuntimeError(
                f"Embeddings cache (explicit) was created with embedding_layer={cached_layer}, but you requested "
                f"--embedding_layer={int(args.embedding_layer)}. Use --overwrite, or point --embeddings_cache to a matching file."
            )
        print(
            f"Loaded embeddings cache (explicit): {emb_cache} (n={len(task_ids)}, dim={X.shape[1]}, counts_kind={counts_kind or 'unknown'}, embedding_layer={cached_layer})"
        )
    else:
        # Collect items to embed.
        items = list(
            iter_swebench_items(
                dataset_name=str(dataset_name),
                split=str(args.split),
                dataset_path=str(dataset_path),
            )
        )
        src = dataset_sources_str
        print(f"Loaded dataset items: {len(items)} (sources={src}, hf_split={args.split}, json_split=train)")

        ids_sorted, emb_by_id, counts_by_id, emb_dim = embed_items(
            items=items,
            backbone=str(args.backbone),
            trust_remote_code=bool(args.trust_remote_code),
            max_length=int(args.max_length),
            batch_size=int(args.batch_size),
            device_map=str(args.device_map),
            torch_dtype=str(args.torch_dtype),
            attn_implementation=str(args.attn_implementation),
            instruction=str(args.instruction),
            embedding_layer=int(args.embedding_layer),
        )
        if not ids_sorted:
            raise RuntimeError("No embeddings were produced (empty ids set).")

        X = np.stack([emb_by_id[r] for r in ids_sorted], axis=0).astype(np.float32)
        counts_arr = np.array([int(counts_by_id.get(r, 0)) for r in ids_sorted], dtype=np.int64)

        np.savez_compressed(
            emb_cache,
            task_ids=np.array(ids_sorted, dtype=object),
            X=X,
            counts_kind=np.array(["text_len_chars"], dtype=object),
            counts=counts_arr,
            dataset_name=np.array([str(dataset_sources_str)], dtype=object),
            split=np.array([str(args.split)], dtype=object),
            dataset_path=np.array([str(dataset_path)], dtype=object),
            n_items=np.array([int(len(ids_sorted))], dtype=np.int64),
            instruction=np.array([str(args.instruction)], dtype=object),
            instruction_signature=np.array([str(instr_sig)], dtype=object),
            backbone=np.array([str(args.backbone)], dtype=object),
            max_length=np.array([int(args.max_length)], dtype=np.int64),
            embedding_dim=np.array([int(emb_dim)], dtype=np.int64),
            embedding_layer=np.array([int(args.embedding_layer)], dtype=np.int64),
        )
        print(f"Wrote embeddings cache: {emb_cache} (n={len(ids_sorted)}, dim={X.shape[1]}, embedding_layer={int(args.embedding_layer)})")
        task_ids = ids_sorted

    # Align embeddings with response JSONL items.
    id_to_row = {tid: i for i, tid in enumerate(task_ids)}

    all_responses = load_all_responses(str(args.agent_results))
    if not all_responses:
        raise RuntimeError(f"Loaded 0 subject responses from --agent_results={args.agent_results!r}")

    response_items: Set[str] = set()
    for _, resp in all_responses:
        response_items.update(resp.keys())

    overlap_ids = [tid for tid in task_ids if tid in response_items]
    if not overlap_ids:
        raise RuntimeError("No overlap between embedded task_ids and item_ids found in --agent_results responses.")

    zero_success_ids = compute_zero_success_items(all_responses)
    zero_success_set = set(zero_success_ids)
    exclude_zero_success = not bool(args.include_zero_success)
    if exclude_zero_success:
        eligible = [tid for tid in overlap_ids if tid not in zero_success_set]
        print(
            f"Excluding zero-success items from CV/IRT: {len(overlap_ids) - len(eligible)}/{len(overlap_ids)} overlapped items "
            f"(agent_results={args.agent_results})"
        )
    else:
        eligible = list(overlap_ids)

    if not eligible:
        raise RuntimeError("After filtering, no items remain for CV/IRT.")

    Xy = np.stack([X[id_to_row[tid]] for tid in eligible], axis=0).astype(np.float32)

    regressor_name = str(args.regressor)
    alphas: np.ndarray = np.array([], dtype=np.float64)

    def _make_model(*, n_train: int, fold_seed: int):
        nonlocal alphas
        if regressor_name == "linear":
            return LinearRegression()
        if regressor_name == "ridge":
            alpha = float(args.ridge_alpha)
            if not (alpha > 0):
                raise ValueError("--ridge_alpha must be > 0")
            return Pipeline(
                steps=[("scaler", StandardScaler(with_mean=True, with_std=True)), ("ridge", Ridge(alpha=alpha))]
            )
        if regressor_name == "ridge_cv":
            try:
                alphas = np.array(
                    [float(x.strip()) for x in str(args.ridge_alphas).split(",") if x.strip()],
                    dtype=np.float64,
                )
            except Exception as e:
                raise ValueError(f"Failed to parse --ridge_alphas={args.ridge_alphas!r}: {e}") from e
            if alphas.size == 0:
                raise ValueError("Expected at least one alpha in --ridge_alphas")
            req_inner = int(args.inner_splits)
            if req_inner < 2:
                raise ValueError("--inner_splits must be >= 2")
            inner_splits = int(min(req_inner, max(2, int(n_train))))
            inner_cv = KFold(n_splits=int(inner_splits), shuffle=True, random_state=int(fold_seed))
            return Pipeline(
                steps=[
                    ("scaler", StandardScaler(with_mean=True, with_std=True)),
                    ("ridge", RidgeCV(alphas=alphas, cv=inner_cv)),
                ]
            )
        raise AssertionError(f"Unhandled regressor: {regressor_name}")

    # K-fold CV over items. Each fold trains IRT on the K-1 training folds only.
    outer_cv = KFold(n_splits=int(args.cv_folds), shuffle=True, random_state=int(args.seed))
    cv_test_auc_folds: List[float] = []
    cv_test_n_obs_folds: List[int] = []
    yhat_oof = np.full((int(len(eligible)),), np.nan, dtype=np.float64)
    fold_of_item = np.full((int(len(eligible)),), -1, dtype=np.int32)

    best_fold_auc = -float("inf")
    best_fold = -1
    best_model = None

    for fold, (tr, te) in enumerate(outer_cv.split(Xy), start=1):
        train_items = [eligible[int(i)] for i in tr.tolist()]
        test_items = [eligible[int(i)] for i in te.tolist()]

        fold_root = os.path.join(str(args.out_dir), "irt_folds", f"fold_{int(fold):02d}")
        ensure_dir(fold_root)

        train_jsonl = os.path.join(fold_root, "train_responses.jsonl")
        n_subj_written, n_items_written = write_filtered_responses_jsonl(
            all_responses=all_responses, item_ids=train_items, out_path=train_jsonl
        )
        if n_subj_written == 0 or n_items_written == 0:
            raise RuntimeError(f"Fold {fold}: wrote 0 subjects/items to {train_jsonl} (check filtering).")

        irt_device = str(args.irt_device or "cpu").strip() or "cpu"
        if irt_device.startswith("cuda") and not torch.cuda.is_available():
            print("WARNING: --irt_device=cuda requested but CUDA is unavailable; falling back to cpu for IRT.")
            irt_device = "cpu"

        # Fixed IRT policy: seeded RNG, but torch determinism OFF for IRT only.
        set_torch_determinism(False)
        seed_everything(int(args.seed), deterministic=False)

        theta_by_subject, diff_by_item = train_irt_1pl(
            responses_jsonl=train_jsonl,
            epochs=int(args.irt_epochs),
            device=str(irt_device),
            seed=int(args.seed),
            out_dir=os.path.join(fold_root, "irt_1pl"),
        )
        # Restore global determinism setting after IRT.
        set_torch_determinism(True)
        if not theta_by_subject:
            raise RuntimeError(f"Fold {fold}: IRT produced 0 subject thetas (unexpected).")
        if not diff_by_item:
            raise RuntimeError(f"Fold {fold}: IRT produced 0 item difficulties (unexpected).")

        train_labeled = [tid for tid in train_items if tid in diff_by_item]
        if len(train_labeled) < 2:
            raise RuntimeError(
                f"Fold {fold}: only {len(train_labeled)} train items had IRT difficulties; cannot fit regressor."
            )

        # Seed post-IRT for any downstream randomness (sklearn CV shuffles, etc).
        seed_everything(int(args.seed) + int(fold), deterministic=True)

        X_train = np.stack([X[id_to_row[tid]] for tid in train_labeled], axis=0).astype(np.float32)
        y_train = np.array([float(diff_by_item[tid]) for tid in train_labeled], dtype=np.float32)

        m = _make_model(n_train=int(len(train_labeled)), fold_seed=int(args.seed) + int(fold))
        m.fit(X_train, y_train)

        X_test = np.stack([X[id_to_row[tid]] for tid in test_items], axis=0).astype(np.float32)
        pred = m.predict(X_test).astype(np.float64)
        yhat_oof[te] = pred
        fold_of_item[te] = int(fold)

        # Held-out AUROC using held-out items only, with theta from fold's IRT.
        z_by_item = {tid: float(z) for tid, z in zip(test_items, pred.tolist())}
        scores: List[float] = []
        labels: List[int] = []
        test_set = set(test_items)
        for sid, resp in all_responses:
            theta = theta_by_subject.get(sid, None)
            if theta is None:
                continue
            th = float(theta)
            for item_id, y_obs in resp.items():
                if item_id not in test_set:
                    continue
                z = z_by_item.get(item_id, None)
                if z is None:
                    continue
                scores.append(1.0 / (1.0 + math.exp(-(th - float(z)))))
                labels.append(int(y_obs))

        fold_auc = float(_compute_binary_auroc(scores, labels))
        cv_test_auc_folds.append(float(fold_auc))
        cv_test_n_obs_folds.append(int(len(labels)))
        if fold_auc == fold_auc and fold_auc > best_fold_auc:
            best_fold_auc = float(fold_auc)
            best_fold = int(fold)
            best_model = m

    if np.isnan(yhat_oof).any() or (fold_of_item < 0).any():
        raise RuntimeError("KFold CV produced incomplete out-of-fold predictions (unexpected).")
    if best_model is None or best_fold < 1:
        raise RuntimeError("Failed to select a best CV fold model by ROC-AUC (all folds NaN?).")

    auc_arr = np.asarray(cv_test_auc_folds, dtype=np.float64)
    auc_mean = float(np.nanmean(auc_arr)) if auc_arr.size else float("nan")
    auc_std = float(np.nanstd(auc_arr, ddof=0)) if auc_arr.size else float("nan")
    print(f"{int(args.cv_folds)}-fold CV test ROC-AUC: mean={auc_mean} std={auc_std}")
    print("Per-fold ROC-AUC: " + ", ".join([str(x) for x in cv_test_auc_folds]))

    # Use the best-fold model for saving weights + predicting excluded items.
    model = best_model

    ridge_alpha = None
    if regressor_name in ("ridge", "ridge_cv"):
        try:
            ridge_alpha = float(model.named_steps["ridge"].alpha_)
        except Exception:
            ridge_alpha = None

    metrics = {
        "n_items_total": int(len(task_ids)),
        "n_items_with_responses": int(len(overlap_ids)),
        "n_items_eligible_cv_irt": int(len(eligible)),
        "exclude_zero_success": bool(exclude_zero_success),
        "n_items_zero_success_in_responses": int(len(zero_success_ids)),
        "embedding_dim": int(Xy.shape[1]),
        "seed": int(args.seed),
        "deterministic": True,
        "irt_seeded": True,
        "irt_deterministic": False,
        "cv_n_splits": int(args.cv_folds),
        "cv_best_auc_fold": int(best_fold),
        "cv_best_auc": float(best_fold_auc),
        "cv_test_auc_folds": [float(x) for x in cv_test_auc_folds],
        "cv_test_auc_mean": float(auc_mean),
        "cv_test_auc_std": float(auc_std),
        "cv_test_n_obs_folds": [int(x) for x in cv_test_n_obs_folds],
        "irt_epochs": int(args.irt_epochs),
        "irt_device": str(args.irt_device),
        "regressor": regressor_name,
        "ridge_alpha": ridge_alpha,
        "ridge_alphas_searched": [float(x) for x in np.asarray(alphas).tolist()],
        "inner_splits": int(args.inner_splits),
        "backbone": str(args.backbone),
        "pooling": "last_token_of_hidden_state",
        "embedding_layer": int(args.embedding_layer),
        "max_length": int(args.max_length),
        "dataset_sources": str(dataset_sources_str),
        "dataset_name": (dataset_name or None),
        "dataset_path": (dataset_path or None),
        "split": str(args.split),
        "agent_results": str(args.agent_results),
        "instruction": str(args.instruction),
        "instruction_signature": instr_sig,
        "batch_size": int(args.batch_size),
        "device_map": str(args.device_map),
        "torch_dtype": str(args.torch_dtype),
        "attn_implementation": str(args.attn_implementation),
        "embeddings_cache": emb_cache,
    }

    # Save regression weights (coef/intercept + optional StandardScaler stats) for reuse.
    # This is intentionally a minimal representation so it can be applied without sklearn.
    weights_meta = {
        "script": os.path.abspath(__file__),
        "backbone": str(args.backbone),
        "trust_remote_code": bool(args.trust_remote_code),
        "pooling": "last_token_of_hidden_state",
        "embedding_layer": int(args.embedding_layer),
        "max_length": int(args.max_length),
        "instruction": str(args.instruction),
        "instruction_signature": str(instr_sig),
        "device_map": str(args.device_map),
        "torch_dtype": str(args.torch_dtype),
        "attn_implementation": str(args.attn_implementation),
        "dataset_sources": str(dataset_sources_str),
        "dataset_name": (dataset_name or None),
        "dataset_path": (dataset_path or None),
        "split": str(args.split),
        "id_normalization": "strip instance_ prefix; strip -v.* suffix",
        "seed": int(args.seed),
        "deterministic": True,
        "irt_seeded": True,
        "irt_deterministic": False,
        "cv_n_splits": int(args.cv_folds),
        "cv_best_auc_fold": int(best_fold),
        "cv_best_auc": float(best_fold_auc),
        "ridge_alpha": ridge_alpha,
        "ridge_alphas_searched": [float(x) for x in np.asarray(alphas).tolist()],
        "inner_splits": int(args.inner_splits),
    }
    weights_json, weights_npz = save_regression_weights(
        out_dir=str(args.out_dir),
        model=model,
        regressor_name=str(regressor_name),
        feature_dim=int(Xy.shape[1]),
        metadata=weights_meta,
    )
    metrics.update({"regression_weights_json": weights_json, "regression_weights_npz": weights_npz})

    # Predict on zero-success items (excluded from CV/IRT, if requested).
    zero_embedded: List[str] = []
    yhat_zero: Optional[np.ndarray] = None
    if bool(exclude_zero_success) and zero_success_set:
        zero_embedded = [tid for tid in task_ids if tid in zero_success_set]
        if zero_embedded:
            X_zero = np.stack([X[id_to_row[tid]] for tid in zero_embedded], axis=0).astype(np.float32)
            yhat_zero = model.predict(X_zero).astype(np.float64)
        else:
            print("NOTE: zero-success ids provided, but none were present in embedded task_ids; nothing to predict.")

    metrics.update(
        {
            "n_items_zero_success_embedded": int(len(zero_embedded)),
            "n_items_zero_success_predicted": int(0 if yhat_zero is None else int(np.asarray(yhat_zero).size)),
        }
    )
    save_json(os.path.join(args.out_dir, "metrics.json"), metrics)

    # Write per-item predictions (OOF CV + optional zero_success rows).
    pred_path = os.path.join(args.out_dir, "predictions.csv")
    with open(pred_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item_id", "diff_pred", "split", "fold"])
        w.writeheader()

        # Eligible rows: one per item with out-of-fold prediction.
        for i, tid in enumerate(eligible):
            w.writerow(
                {
                    "item_id": tid,
                    "diff_pred": float(yhat_oof[i]),
                    "split": "cv_val",
                    "fold": int(fold_of_item[i]),
                }
            )

        # Zero-success rows (separate split label).
        if yhat_zero is not None and zero_embedded:
            for tid, score in zip(zero_embedded, yhat_zero.tolist()):
                w.writerow(
                    {
                        "item_id": tid,
                        "diff_pred": float(score),
                        "split": "zero_success",
                        "fold": "",
                    }
                )

    # Print a sorted list to stdout for convenience.
    if yhat_zero is not None and zero_embedded:
        pairs = list(zip(zero_embedded, yhat_zero.tolist()))
        pairs.sort(key=lambda kv: float(kv[1]), reverse=True)
        print(
            f"\n=== ZERO_SUCCESS_PREDICTIONS_SORTED (task_id, diff_pred) "
            f"[model=best_by_auc fold={best_fold} auc={best_fold_auc}] ==="
        )
        for tid, score in pairs:
            print(f"{tid}\t{float(score):.6f}")

    print(f"Wrote metrics: {os.path.join(args.out_dir, 'metrics.json')}")
    print(f"Wrote predictions: {pred_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



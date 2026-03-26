
from __future__ import annotations

import argparse
import csv
import sys
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from torchmetrics import AUROC
from transformers import AutoConfig, AutoModel, AutoTokenizer, FineGrainedFP8Config, PreTrainedTokenizerFast

import pyro
from py_irt.config import IrtConfig
from py_irt.training import IrtModelTrainer

def seed_everything(seed: int, *, deterministic: bool) -> None:
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

    try:
        from transformers import set_seed as _hf_set_seed

        _hf_set_seed(s)
    except Exception:
        pass

    if deterministic:
        set_torch_determinism(True)

        try:
            torch.backends.cuda.matmul.allow_tf32 = False
        except Exception:
            pass
        try:
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass

def set_torch_determinism(enabled: bool) -> None:
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
    try:
        return AutoTokenizer.from_pretrained(backbone, trust_remote_code=trust_remote_code)
    except ValueError as e:
        msg = str(e)
        if "TokenizersBackend" not in msg:
            raise

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
                pass

        tok = PreTrainedTokenizerFast(**tok_kwargs)
        if extra_special_tokens:

            tok.additional_special_tokens = extra_special_tokens
        return tok

DIFFICULTY_INSTRUCTION = (
    "How difficult is the above task for a coding agent? Please output one floating-point number from 0 (very easy) to 1 (very hard). Your difficulty score:\n"
)

JUDGE_FEATURE_NAMES: List[str] = [
    "atypicality",
    "codebase_scale",
    "codebase_scope",
    "debugging_complexity",
    "domain_knowledge_required",
    "error_specificity",
    "fix_localization",
    "implementation_language_complexity",
    "logical_reasoning_required",
    "side_effect_risk",
    "similar_issue_likelihood",
    "solution_complexity",
    "solution_hint",
    "test_edge_case_coverage",
    "verification_difficulty",
]

_V_SUFFIX_RE = re.compile(r"-v(?:\d+|[0-9a-f]{6,}|nan)$", re.IGNORECASE)
EMBEDDING_TEXT_FORMAT = "qs_solution_instruction_v1"

def _canon_benchmark_name(name: str) -> str:
    s = str(name or "").strip().lower().replace("-", "_")
    if s == "terminalbench":
        s = "terminal_bench"
    if s not in {"verified", "pro", "terminal_bench", "gso"}:
        raise ValueError(f"Unknown benchmark name: {name!r}. Allowed: verified, pro, terminal-bench, gso.")
    return s

def _get_benchmark_defaults(benchmark: str) -> Dict[str, str]:
    b = _canon_benchmark_name(benchmark)
    repo_root = str(Path(__file__).resolve().parents[1])
    defaults: Dict[str, str] = {
        "verified": {
            "dataset_name": "princeton-nlp/SWE-bench_Verified",
            "dataset_path": "",
            "split": "test",
            "agent_results": os.path.join(repo_root, "data/swebench_verified/responses.jsonl"),
            "judge_features_dir": os.path.join(repo_root, "llm_judge_features/defaults/swebench_verified/llm_judge_features.csv"),
            "out_dir": os.path.join(repo_root, "data/swebench_verified"),
        },
        "pro": {
            "dataset_name": "ScaleAI/SWE-bench_Pro",
            "dataset_path": "",
            "split": "test",
            "agent_results": os.path.join(repo_root, "data/swebench_pro/responses.jsonl"),
            "judge_features_dir": os.path.join(repo_root, "llm_judge_features/defaults/swebench_pro/llm_judge_features.csv"),
            "out_dir": os.path.join(repo_root, "data/swebench_pro"),
        },
        "terminal_bench": {
            "dataset_name": "",
            "dataset_path": os.path.join(repo_root, "data/terminalbench/tasks.jsonl"),
            "split": "train",
            "agent_results": os.path.join(repo_root, "data/terminalbench/responses.jsonl"),
            "judge_features_dir": os.path.join(repo_root, "llm_judge_features/defaults/terminalbench/llm_judge_features.csv"),
            "out_dir": os.path.join(repo_root, "data/terminalbench"),
        },
        "gso": {
            "dataset_name": "gso-bench/gso",
            "dataset_path": "",
            "split": "test",
            "agent_results": os.path.join(repo_root, "data/gso/responses.jsonl"),
            "judge_features_dir": os.path.join(repo_root, "llm_judge_features/defaults/gso/llm_judge_features.csv"),
            "out_dir": os.path.join(repo_root, "data/gso"),
        },
    }
    return defaults[b]

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

def stable_split_ids(ids: Sequence[str], test_fraction: float, seed: int) -> Tuple[List[int], List[int]]:
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
    out: List[Tuple[str, Dict[str, int]]] = []
    for sid, resp in iter_subject_responses_jsonl(path):
        if resp:
            out.append((sid, resp))
    return out

def compute_zero_success_items(all_responses: List[Tuple[str, Dict[str, int]]]) -> List[str]:
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
    if len(scores) == 0:
        return float("nan")
    uniq = set(int(x) for x in labels)
    if len(uniq) < 2:
        return float("nan")
    auroc = AUROC(task="binary")
    s = torch.tensor(scores, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    return float(auroc(s, y).item())

def _sigmoid(x: float) -> float:
    v = float(x)
    if v >= 0.0:
        z = math.exp(-v)
        return 1.0 / (1.0 + z)
    z = math.exp(v)
    return z / (1.0 + z)

def last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:

    lengths = attention_mask.sum(dim=1).clamp(min=1)
    idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, last_hidden_state.size(-1))
    return last_hidden_state.gather(dim=1, index=idx).squeeze(1)

def train_irt_1pl(
    *,
    responses_jsonl: str,
    epochs: int,
    device: str,
    seed: int,
    out_dir: str,
) -> Tuple[Dict[str, float], Dict[str, float]]:
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

def train_oracle_irt_1pl_and_save(
    *,
    args: argparse.Namespace,
    all_responses: List[Tuple[str, Dict[str, int]]],
    item_ids: Sequence[str],
) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, float]]:
    oracle_dir = os.path.join(str(args.out_dir), "irt_oracle")
    if os.path.exists(oracle_dir):
        shutil.rmtree(oracle_dir, ignore_errors=True)
    ensure_dir(oracle_dir)

    items = [normalize_swebench_item_id(str(x)) for x in list(item_ids)]
    items = [x for x in items if x]
    if not items:
        raise RuntimeError("Oracle IRT: item_ids was empty after normalization.")

    train_jsonl = os.path.join(oracle_dir, "train_responses.jsonl")
    n_subj_written, n_items_written = write_filtered_responses_jsonl(
        all_responses=all_responses, item_ids=items, out_path=train_jsonl
    )
    if n_subj_written == 0 or n_items_written == 0:
        raise RuntimeError(f"Oracle IRT: wrote 0 subjects/items to {train_jsonl} (check filtering).")

    irt_device = str(getattr(args, "irt_device", "cpu") or "cpu").strip() or "cpu"
    if irt_device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: --irt_device=cuda requested but CUDA is unavailable; falling back to cpu for oracle IRT.")
        irt_device = "cpu"

    set_torch_determinism(False)
    seed_everything(int(args.seed), deterministic=False)
    theta_by_subject, diff_by_item = train_irt_1pl(
        responses_jsonl=train_jsonl,
        epochs=int(getattr(args, "irt_epochs", 5000)),
        device=str(irt_device),
        seed=int(args.seed),
        out_dir=str(oracle_dir),
    )
    set_torch_determinism(True)
    seed_everything(int(args.seed), deterministic=True)

    meta: Dict[str, Any] = {
        "oracle_irt_dir": str(oracle_dir),
        "oracle_irt_train_responses_jsonl": str(train_jsonl),
        "oracle_irt_best_parameters_json": str(os.path.join(oracle_dir, "best_parameters.json")),
        "oracle_irt_abilities_csv": str(os.path.join(oracle_dir, "abilities.csv")),
        "oracle_irt_items_csv": str(os.path.join(oracle_dir, "items.csv")),
        "oracle_irt_n_subjects": int(n_subj_written),
        "oracle_irt_n_items": int(n_items_written),
        "oracle_irt_epochs": int(getattr(args, "irt_epochs", 5000)),
        "oracle_irt_device": str(irt_device),
    }
    return meta, theta_by_subject, diff_by_item

def prompt_signature(instruction: str) -> str:
    h = hashlib.sha1(str(instruction).encode("utf-8")).hexdigest()[:8]
    return f"qs_sol_instr_{h}"

def _sanitize_text(s: str) -> str:

    return "".join((" " if (ord(ch) < 32 and ch not in ("\n", "\t")) else ch) for ch in (s or ""))

def format_qs_solution_instruction(*, question_statement: str, solution: str, instruction: str) -> str:
    qs = _sanitize_text(str(question_statement or "")).strip()
    sol = _sanitize_text(str(solution or "")).strip()
    instr = _sanitize_text(str(instruction or "")).strip()
    return f"Task statement:\n{qs}\n\nSolution:\n{sol}\n\n{instr}".strip()

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

            item_id = f"row_{int(i)}"
        yield ItemRecord(item_id=item_id, question_statement=qs, solution=sol)

def iter_swebench_items(
    *,
    dataset_name: str,
    split: str,
    dataset_path: str,
) -> Iterator[ItemRecord]:
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

            item_id = f"row_{int(i)}"
        yield ItemRecord(item_id=item_id, question_statement=qs, solution=sol)

def load_items_by_ids(
    *,
    dataset_name: str,
    split: str,
    dataset_path: str,
    item_ids: Sequence[str],
) -> Tuple[List[ItemRecord], List[str]]:
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

    try:
        import transformers.activations as _act

        if not hasattr(_act, "PytorchGELUTanh") and hasattr(_act, "GELUTanh"):
            _act.PytorchGELUTanh = _act.GELUTanh
    except Exception:

        pass

    errors = []
    try:
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText.from_pretrained(
            backbone, trust_remote_code=trust_remote_code, **model_kwargs
        )
    except Exception as e:
        errors.append(("AutoModelForImageTextToText", e))
    try:
        from transformers import AutoModelForVision2Seq

        return AutoModelForVision2Seq.from_pretrained(backbone, trust_remote_code=trust_remote_code, **model_kwargs)
    except Exception as e:
        errors.append(("AutoModelForVision2Seq", e))
    try:
        from transformers import AutoModelForCausalLM

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
    for attr in ("language_model", "text_model"):
        m = getattr(model, attr, None)
        if isinstance(m, torch.nn.Module):
            return m

    m = getattr(model, "model", None)
    if isinstance(m, torch.nn.Module) and hasattr(m, "get_input_embeddings"):
        return m
    return model

def _get_hidden_states_tuple(outputs):
    for attr in ("hidden_states", "encoder_hidden_states", "decoder_hidden_states"):
        if hasattr(outputs, attr):
            hs = getattr(outputs, attr)
            if hs is not None:
                return hs
    return None

def _extract_hidden_state(outputs, *, embedding_layer: int) -> torch.Tensor:
    layer = int(embedding_layer)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = _load_tokenizer(backbone, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"

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

    fp_params = inspect.signature(AutoModel.from_pretrained).parameters
    if "dtype" in fp_params:
        model_kwargs = {"dtype": dtype_arg}
    else:
        model_kwargs = {"torch_dtype": dtype_arg}
    if device_map and device_map != "none":
        model_kwargs["device_map"] = device_map
    if attn_implementation and attn_implementation != "auto":
        model_kwargs["attn_implementation"] = attn_implementation

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

    text_model = _select_text_submodel(model)

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

        want_hidden_states = int(embedding_layer) != -1

        with torch.inference_mode():
            fwd_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=bool(want_hidden_states),
            )

            try:
                sig = inspect.signature(text_model.forward)
                if "use_cache" in sig.parameters:
                    fwd_kwargs["use_cache"] = False
            except Exception:

                fwd_kwargs["use_cache"] = False

            out = text_model(**fwd_kwargs)
            try:
                h = _extract_hidden_state(out, embedding_layer=int(embedding_layer))
            except RuntimeError:

                if not want_hidden_states:
                    fwd_kwargs["output_hidden_states"] = True
                    out = text_model(**fwd_kwargs)
                    h = _extract_hidden_state(out, embedding_layer=int(embedding_layer))
                else:
                    raise
            pooled = last_token_pool(h, attention_mask)
            pooled = pooled.detach().float().cpu().numpy()

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
    if value is None:
        return default
    try:
        if isinstance(value, np.ndarray):
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

def _meta_str(value: object, default: str = "") -> str:
    v = _npz_scalar(value, default)
    if isinstance(v, np.ndarray):
        if v.size == 1:
            v = v.reshape(-1)[0]
    if isinstance(v, (list, tuple)) and len(v) == 1:
        v = v[0]
    s = str(v if v is not None else default).strip()
    if (s.startswith("['") and s.endswith("']")) or (s.startswith('["') and s.endswith('"]')):
        s = s[2:-2].strip()
    return s

def _to_boolish(value: object, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(int(value))
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)

def _candidate_embedding_roots(*, out_dir: str) -> List[str]:
    roots: List[str] = []
    out = str(out_dir or "").strip()
    if out:
        roots.extend([out, os.path.join(out, "embeddings")])
    repo_root = str(Path(__file__).resolve().parents[1])
    roots.extend([os.path.join(repo_root, "embeddings"), os.path.join(repo_root, "data")])
    seen: Set[str] = set()
    out_roots: List[str] = []
    for p in roots:
        ap = os.path.abspath(str(p))
        if ap in seen or not os.path.isdir(ap):
            continue
        seen.add(ap)
        out_roots.append(ap)
    return out_roots

def _shared_embeddings_dir() -> str:
    repo_root = str(Path(__file__).resolve().parents[1])
    return os.path.join(repo_root, "embeddings")

def _iter_embedding_npz_candidates(search_roots: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for root in search_roots:
        rp = os.path.abspath(str(root))
        if not os.path.isdir(rp):
            continue
        patterns = [
            os.path.join(rp, "*.npz"),
            os.path.join(rp, "embeddings", "*.npz"),
            os.path.join(rp, "*", "*.npz"),
        ]
        for pat in patterns:
            try:
                for p in Path(rp).glob(str(Path(pat).relative_to(rp))):
                    ap = str(p.resolve())
                    if ap in seen:
                        continue
                    seen.add(ap)
                    out.append(ap)
            except Exception:
                continue
    out.sort(key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0.0, reverse=True)
    return out

def load_compatible_embeddings_cache(
    path: str,
    *,
    backbone: str,
    max_length: int,
    instruction_sig: str,
    required_task_ids: Optional[Sequence[str]] = None,
    expected_n_items: Optional[int] = None,
    require_single_dataset_source: bool = False,
) -> Optional[Tuple[List[str], np.ndarray, Dict[str, object]]]:
    p = str(path or "").strip()
    if (not p) or (not os.path.exists(p)):
        return None
    try:
        with np.load(p, allow_pickle=True) as data:
            if ("task_ids" not in data) or ("X" not in data):
                return None
            task_ids = [str(x) for x in list(data["task_ids"].tolist())]
            X = data["X"].astype(np.float32)
            if X.ndim != 2 or X.shape[0] != len(task_ids) or X.shape[1] <= 0:
                return None
            if len(set(task_ids)) != len(task_ids):
                return None

            cached_instr_sig = _meta_str(data.get("instruction_signature", None), "")
            req_instr_sig = str(instruction_sig or "").strip()
            same_prompt_template_family = False
            if cached_instr_sig and req_instr_sig:
                same_prompt_template_family = (
                    str(cached_instr_sig) == str(req_instr_sig)
                    or (str(cached_instr_sig).startswith("qs_sol_") and str(req_instr_sig).startswith("qs_sol_"))
                )

            cached_backbone = _meta_str(data.get("backbone", None), "")
            if cached_backbone and cached_backbone != str(backbone):
                return None

            cached_dataset_source = _meta_str(data.get("dataset_name", None), "")
            if bool(require_single_dataset_source) and ("|" in cached_dataset_source):
                return None

            includes_solution = _to_boolish(_npz_scalar(data.get("includes_solution", None), None), default=False) if "includes_solution" in data else False
            text_format = _meta_str(data.get("text_format", None), "")
            cache_prompt_template_ok = (
                includes_solution
                or text_format == EMBEDDING_TEXT_FORMAT
                or cached_instr_sig.startswith("qs_sol_")
            )
            if not (cache_prompt_template_ok and (same_prompt_template_family or (not req_instr_sig))):
                return None

            if required_task_ids:
                id_set = set(task_ids)
                for tid in required_task_ids:
                    if str(tid) not in id_set:
                        return None
            if expected_n_items is not None:
                n_cached = int(len(task_ids))
                n_expected = int(expected_n_items)
                pro_off_by_one_ok = (n_expected == 730 and n_cached == 731)
                if (n_cached != n_expected) and (not pro_off_by_one_ok):
                    return None

            cached_layer = int(_npz_scalar(data.get("embedding_layer", None), -1)) if "embedding_layer" in data else -1
            cached_maxlen = int(_npz_scalar(data.get("max_length", None), int(max_length))) if "max_length" in data else int(max_length)

            meta = {
                "path": str(p),
                "n_items": int(len(task_ids)),
                "dim": int(X.shape[1]),
                "embedding_layer": int(cached_layer),
                "instruction_signature": str(cached_instr_sig),
                "max_length": int(cached_maxlen),
                "backbone": str(cached_backbone),
                "dataset_name": str(cached_dataset_source),
                "text_format": str(text_format),
            }
            return task_ids, X, meta
    except Exception:
        return None

def find_compatible_embeddings_cache(
    *,
    preferred_paths: Sequence[str],
    search_roots: Sequence[str],
    backbone: str,
    max_length: int,
    instruction_sig: str,
    required_task_ids: Optional[Sequence[str]] = None,
    expected_n_items: Optional[int] = None,
    require_single_dataset_source: bool = False,
) -> Optional[Tuple[str, List[str], np.ndarray, Dict[str, object]]]:
    candidates: List[str] = []
    seen: Set[str] = set()
    for p in preferred_paths:
        ap = os.path.abspath(str(p))
        if ap in seen:
            continue
        seen.add(ap)
        candidates.append(ap)
    for p in _iter_embedding_npz_candidates(search_roots):
        ap = os.path.abspath(str(p))
        if ap in seen:
            continue
        seen.add(ap)
        candidates.append(ap)

    for p in candidates:
        loaded = load_compatible_embeddings_cache(
            p,
            backbone=str(backbone),
            max_length=int(max_length),
            instruction_sig=str(instruction_sig),
            required_task_ids=required_task_ids,
            expected_n_items=expected_n_items,
            require_single_dataset_source=bool(require_single_dataset_source),
        )
        if loaded is None:
            continue
        task_ids, X, meta = loaded
        return str(p), task_ids, X, meta
    return None

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
        return float(x)

def save_regression_weights(
    *,
    out_dir: str,
    model: Any,
    regressor_name: str,
    feature_dim: int,
    metadata: dict,
) -> Tuple[str, str]:
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

_JUDGE_INDEX_CACHE: Dict[Tuple[str, bool], Dict[str, str]] = {}
_JUDGE_CSV_HEADER_CACHE: Dict[str, List[str]] = {}
_JUDGE_CSV_CACHE: Dict[Tuple[str, bool, Tuple[str, ...]], Dict[str, np.ndarray]] = {}

def _looks_like_csv_path(p: str) -> bool:
    s = str(p or "").strip().lower()
    return bool(s.endswith(".csv"))

def _load_judge_csv_feature_names(features_csv: str) -> List[str]:
    root = os.path.abspath(str(features_csv))
    if root in _JUDGE_CSV_HEADER_CACHE:
        return list(_JUDGE_CSV_HEADER_CACHE[root])

    if not os.path.exists(root):
        raise FileNotFoundError(f"Judge features CSV not found: {features_csv!r}")
    if not os.path.isfile(root):
        raise ValueError(f"Expected a judge features CSV file path, got: {features_csv!r}")

    with open(root, "r", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        header = next(r, [])
    header = [str(x).strip() for x in header if str(x).strip()]
    if not header or "instance_id" not in header:
        raise ValueError(f"Judge features CSV missing required header column 'instance_id': {features_csv!r}")
    feats = [h for h in header if h != "instance_id"]
    _JUDGE_CSV_HEADER_CACHE[root] = feats
    return list(feats)

def _load_judge_csv_vectors(
    features_csv: str,
    *,
    feature_names: Sequence[str],
    normalize_item_ids: bool = True,
) -> Dict[str, np.ndarray]:
    root = os.path.abspath(str(features_csv))
    key = (root, bool(normalize_item_ids), tuple([str(x) for x in feature_names]))
    if key in _JUDGE_CSV_CACHE:
        return _JUDGE_CSV_CACHE[key]

    if not os.path.exists(root):
        raise FileNotFoundError(f"Judge features CSV not found: {features_csv!r}")
    if not os.path.isfile(root):
        raise ValueError(f"Expected a judge features CSV file path, got: {features_csv!r}")

    out: Dict[str, np.ndarray] = {}
    with open(root, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        fieldnames = list(r.fieldnames or [])
        if "instance_id" not in fieldnames:
            raise ValueError(f"Judge features CSV missing required column 'instance_id': {features_csv!r}")
        missing_cols = [k for k in feature_names if k not in fieldnames]
        if missing_cols:
            raise ValueError(
                f"Judge features CSV missing columns {missing_cols!r} (have {fieldnames!r}): {features_csv!r}"
            )
        for row in r:
            iid_raw = str((row.get("instance_id") or "")).strip()
            if not iid_raw:
                continue
            iid = (normalize_swebench_item_id(iid_raw) or iid_raw) if normalize_item_ids else iid_raw
            xs: List[float] = []
            ok = True
            for k in feature_names:
                v = row.get(str(k), "")
                if v is None:
                    ok = False
                    break
                s = str(v).strip()
                if s == "":
                    ok = False
                    break
                try:
                    xs.append(float(s))
                except Exception:
                    ok = False
                    break
            if not ok:
                continue
            out[iid] = np.asarray(xs, dtype=np.float32)

    _JUDGE_CSV_CACHE[key] = out
    return out

def _build_judge_index(features_dir: str, *, normalize_item_ids: bool = True) -> Dict[str, str]:
    root = os.path.abspath(str(features_dir))
    key = (root, bool(normalize_item_ids))
    if key in _JUDGE_INDEX_CACHE:
        return _JUDGE_INDEX_CACHE[key]

    if _looks_like_csv_path(features_dir):
        if not os.path.exists(root):
            raise FileNotFoundError(f"Judge features CSV not found: {features_dir!r}")
        if not os.path.isfile(root):
            raise ValueError(f"Expected a judge features CSV file path, got: {features_dir!r}")
        _JUDGE_INDEX_CACHE[key] = {}
        return {}

    idx: Dict[str, str] = {}
    try:
        names = [x for x in os.listdir(root) if x.endswith(".json")]
    except Exception:
        names = []
    for fn in names:
        stem = fn[:-5]
        norm = (normalize_swebench_item_id(stem) or stem) if normalize_item_ids else str(stem).strip()
        if not norm:
            continue
        idx.setdefault(norm, os.path.join(root, fn))

    _JUDGE_INDEX_CACHE[key] = idx
    return idx

def _load_judge_vector(
    task_id: str,
    *,
    features_dir: str,
    feature_names: Sequence[str],
    index: Dict[str, str],
    normalize_item_ids: bool = True,
) -> Optional[np.ndarray]:
    tid = str(task_id or "").strip()
    if not tid:
        return None

    if _looks_like_csv_path(features_dir):
        m = _load_judge_csv_vectors(features_dir, feature_names=feature_names, normalize_item_ids=normalize_item_ids)
        norm = (normalize_swebench_item_id(tid) or tid) if normalize_item_ids else tid
        return m.get(norm, None)

    p = os.path.join(str(features_dir), f"{tid}.json")
    if not os.path.exists(p):
        norm = (normalize_swebench_item_id(tid) or tid) if normalize_item_ids else tid
        p = index.get(norm, "")
        if not p or not os.path.exists(p):
            return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return None

    xs: List[float] = []
    for k in feature_names:
        v = obj.get(k, None)
        if v is None:
            return None
        try:
            xs.append(float(v))
        except Exception:
            return None
    return np.asarray(xs, dtype=np.float32)

def _parse_alpha_list(s: str) -> np.ndarray:
    try:
        xs = [float(x.strip()) for x in str(s or "").split(",") if x.strip()]
    except Exception as e:
        raise ValueError(f"Failed to parse alpha list {s!r}: {e}") from e
    if not xs:
        raise ValueError("Expected at least one alpha.")
    arr = np.asarray(xs, dtype=np.float64)
    if not np.all(arr > 0):
        raise ValueError(f"All alphas must be > 0; got {arr.tolist()}")
    return arr

def _fit_block_ridge(
    *,
    X_emb: np.ndarray,
    X_judge: np.ndarray,
    y: np.ndarray,
    alpha_emb: float,
    alpha_judge: float,
) -> dict:
    ae = float(alpha_emb)
    aj = float(alpha_judge)
    if not (ae > 0 and aj > 0):
        raise ValueError(f"alpha_emb and alpha_judge must be > 0; got {ae}, {aj}")

    X_emb = np.asarray(X_emb, dtype=np.float64)
    X_judge = np.asarray(X_judge, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if X_emb.shape[0] != X_judge.shape[0] or X_emb.shape[0] != y.shape[0]:
        raise ValueError(f"Row mismatch: X_emb={X_emb.shape} X_judge={X_judge.shape} y={y.shape}")

    emb_scaler = StandardScaler(with_mean=True, with_std=True)
    judge_scaler = StandardScaler(with_mean=True, with_std=True)
    X_emb_s = emb_scaler.fit_transform(X_emb)
    X_judge_s = judge_scaler.fit_transform(X_judge)

    X_t = np.concatenate([X_emb_s / math.sqrt(ae), X_judge_s / math.sqrt(aj)], axis=1)
    model = Ridge(alpha=1.0, fit_intercept=True, random_state=None)
    model.fit(X_t, y)

    return {
        "ridge": model,
        "emb_scaler": emb_scaler,
        "judge_scaler": judge_scaler,
        "alpha_emb": ae,
        "alpha_judge": aj,
        "n_emb": int(X_emb.shape[1]),
        "n_judge": int(X_judge.shape[1]),
    }

def _predict_block_ridge(state: dict, *, X_emb: np.ndarray, X_judge: np.ndarray) -> np.ndarray:
    X_emb = np.asarray(X_emb, dtype=np.float64)
    X_judge = np.asarray(X_judge, dtype=np.float64)
    X_emb_s = state["emb_scaler"].transform(X_emb)
    X_judge_s = state["judge_scaler"].transform(X_judge)
    X_t = np.concatenate(
        [
            X_emb_s / math.sqrt(float(state["alpha_emb"])),
            X_judge_s / math.sqrt(float(state["alpha_judge"])),
        ],
        axis=1,
    )
    pred = state["ridge"].predict(X_t)
    return np.asarray(pred, dtype=np.float64).reshape(-1)

def _decompose_block_ridge_single(state: dict, *, x_emb_raw: np.ndarray, x_judge_raw: np.ndarray) -> Dict[str, float]:
    ridge = state["ridge"]
    coef = np.asarray(getattr(ridge, "coef_", []), dtype=np.float64).reshape(-1)
    if coef.size == 0:
        raise RuntimeError("Model has no coef_; cannot decompose.")

    n_emb = int(state["n_emb"])
    w_emb_t = coef[:n_emb].reshape(-1)
    w_judge_t = coef[n_emb:].reshape(-1)

    alpha_emb = float(state["alpha_emb"])
    alpha_judge = float(state["alpha_judge"])
    emb_scaler = state["emb_scaler"]
    judge_scaler = state["judge_scaler"]

    x_emb_raw = np.asarray(x_emb_raw, dtype=np.float64).reshape(1, -1)
    x_judge_raw = np.asarray(x_judge_raw, dtype=np.float64).reshape(1, -1)

    x_emb_s = emb_scaler.transform(x_emb_raw)[0]
    x_judge_s = judge_scaler.transform(x_judge_raw)[0]
    w_emb_std = w_emb_t / math.sqrt(alpha_emb)
    w_judge_std = w_judge_t / math.sqrt(alpha_judge)
    emb_contrib_std = float(np.dot(x_emb_s, w_emb_std))
    judge_contrib_std = float(np.dot(x_judge_s, w_judge_std))
    intercept_model = float(getattr(ridge, "intercept_", 0.0))

    emb_scale = np.asarray(getattr(emb_scaler, "scale_", None), dtype=np.float64).reshape(-1)
    judge_scale = np.asarray(getattr(judge_scaler, "scale_", None), dtype=np.float64).reshape(-1)
    emb_mean = np.asarray(getattr(emb_scaler, "mean_", None), dtype=np.float64).reshape(-1)
    judge_mean = np.asarray(getattr(judge_scaler, "mean_", None), dtype=np.float64).reshape(-1)

    emb_scale = np.where(emb_scale == 0.0, 1.0, emb_scale)
    judge_scale = np.where(judge_scale == 0.0, 1.0, judge_scale)

    w_emb_raw = w_emb_std / emb_scale
    w_judge_raw = w_judge_std / judge_scale
    emb_dot_raw = float(np.dot(x_emb_raw.reshape(-1), w_emb_raw))
    judge_dot_raw = float(np.dot(x_judge_raw.reshape(-1), w_judge_raw))
    intercept_raw = intercept_model - float(np.dot(emb_mean, w_emb_raw)) - float(np.dot(judge_mean, w_judge_raw))

    pred_check = intercept_model + emb_contrib_std + judge_contrib_std
    pred_raw = intercept_raw + emb_dot_raw + judge_dot_raw

    return {
        "pred": float(pred_check),
        "pred_raw": float(pred_raw),
        "intercept_model": float(intercept_model),
        "intercept_raw": float(intercept_raw),
        "emb_contrib_std": float(emb_contrib_std),
        "judge_contrib_std": float(judge_contrib_std),
        "emb_dot_raw": float(emb_dot_raw),
        "judge_dot_raw": float(judge_dot_raw),
    }

def _select_block_alphas_inner_cv(
    *,
    X_emb: np.ndarray,
    X_judge: np.ndarray,
    y: np.ndarray,
    alphas_emb: np.ndarray,
    alphas_judge: np.ndarray,
    inner_splits: int,
    seed: int,
    verbose: bool = False,
) -> Tuple[float, float, float]:
    X_emb = np.asarray(X_emb, dtype=np.float64)
    X_judge = np.asarray(X_judge, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = int(y.shape[0])
    k = int(min(int(inner_splits), max(2, n)))
    inner_cv = KFold(n_splits=k, shuffle=True, random_state=int(seed))

    best_ae: Optional[float] = None
    best_aj: Optional[float] = None
    best_mse: float = float("inf")
    total = int(len(alphas_emb)) * int(len(alphas_judge))
    seen = 0
    for ae in alphas_emb:
        for aj in alphas_judge:
            seen += 1
            mse_sum = 0.0
            n_folds = 0
            for tr, va in inner_cv.split(y):
                st = _fit_block_ridge(
                    X_emb=X_emb[tr],
                    X_judge=X_judge[tr],
                    y=y[tr],
                    alpha_emb=float(ae),
                    alpha_judge=float(aj),
                )
                p = _predict_block_ridge(st, X_emb=X_emb[va], X_judge=X_judge[va])
                err = y[va] - p
                mse_sum += float(np.mean(err * err))
                n_folds += 1
            mse = mse_sum / max(1, n_folds)
            if mse < best_mse:
                best_mse = float(mse)
                best_ae = float(ae)
                best_aj = float(aj)
            if verbose and (seen == 1 or seen % 10 == 0 or seen == total):
                print(
                    f"Block-ridge inner CV: tried {seen}/{total} (alpha_emb={float(ae):g}, alpha_judge={float(aj):g}) "
                    f"mse={float(mse):.6g} best_mse={float(best_mse):.6g}"
                )
    if best_ae is None or best_aj is None:
        raise RuntimeError("Inner CV failed to select block alphas.")
    return float(best_ae), float(best_aj), float(best_mse)

def _extract_block_ridge_raw_weights(state: dict) -> Tuple[np.ndarray, np.ndarray, float]:
    ridge = state["ridge"]
    coef = np.asarray(getattr(ridge, "coef_", []), dtype=np.float64).reshape(-1)
    if coef.size == 0:
        raise RuntimeError("Model has no coef_.")

    n_emb = int(state["n_emb"])
    w_emb_t = coef[:n_emb].reshape(-1)
    w_judge_t = coef[n_emb:].reshape(-1)

    alpha_emb = float(state["alpha_emb"])
    alpha_judge = float(state["alpha_judge"])
    emb_scaler = state["emb_scaler"]
    judge_scaler = state["judge_scaler"]

    emb_scale = np.asarray(getattr(emb_scaler, "scale_", None), dtype=np.float64).reshape(-1)
    judge_scale = np.asarray(getattr(judge_scaler, "scale_", None), dtype=np.float64).reshape(-1)
    emb_mean = np.asarray(getattr(emb_scaler, "mean_", None), dtype=np.float64).reshape(-1)
    judge_mean = np.asarray(getattr(judge_scaler, "mean_", None), dtype=np.float64).reshape(-1)
    emb_scale = np.where(emb_scale == 0.0, 1.0, emb_scale)
    judge_scale = np.where(judge_scale == 0.0, 1.0, judge_scale)

    w_emb_std = w_emb_t / math.sqrt(alpha_emb)
    w_judge_std = w_judge_t / math.sqrt(alpha_judge)
    w_emb_raw = w_emb_std / emb_scale
    w_judge_raw = w_judge_std / judge_scale

    intercept_model = float(getattr(ridge, "intercept_", 0.0))
    intercept_raw = intercept_model - float(np.dot(emb_mean, w_emb_raw)) - float(np.dot(judge_mean, w_judge_raw))
    return w_emb_raw.astype(np.float32, copy=False), w_judge_raw.astype(np.float32, copy=False), float(intercept_raw)

def save_regression_weights_block_ridge(
    *,
    out_dir: str,
    state: dict,
    judge_feature_names: Sequence[str],
    metadata: dict,
) -> Tuple[str, str]:
    ensure_dir(out_dir)
    w_emb_raw, w_judge_raw, intercept_raw = _extract_block_ridge_raw_weights(state)

    emb_scaler = state["emb_scaler"]
    judge_scaler = state["judge_scaler"]
    emb_mean = np.asarray(getattr(emb_scaler, "mean_", []), dtype=np.float32).reshape(-1)
    emb_scale = np.asarray(getattr(emb_scaler, "scale_", []), dtype=np.float32).reshape(-1)
    judge_mean = np.asarray(getattr(judge_scaler, "mean_", []), dtype=np.float32).reshape(-1)
    judge_scale = np.asarray(getattr(judge_scaler, "scale_", []), dtype=np.float32).reshape(-1)

    weights_npz = os.path.join(str(out_dir), "regression_weights.npz")
    np.savez_compressed(
        weights_npz,
        coef_emb_raw=np.asarray(w_emb_raw, dtype=np.float32).reshape(-1),
        coef_judge_raw=np.asarray(w_judge_raw, dtype=np.float32).reshape(-1),
        intercept_raw=np.asarray([float(intercept_raw)], dtype=np.float32),
        alpha_emb=np.asarray([float(state["alpha_emb"])], dtype=np.float32),
        alpha_judge=np.asarray([float(state["alpha_judge"])], dtype=np.float32),
        emb_scaler_mean=emb_mean,
        emb_scaler_scale=emb_scale,
        judge_scaler_mean=judge_mean,
        judge_scaler_scale=judge_scale,
        judge_feature_names=np.asarray(list(judge_feature_names), dtype=object),
        n_emb=np.asarray([int(state["n_emb"])], dtype=np.int64),
        n_judge=np.asarray([int(state["n_judge"])], dtype=np.int64),
    )

    weights_json = os.path.join(str(out_dir), "regression_weights.json")
    meta = dict(metadata or {})
    meta.update(
        {
            "regressor": "block_ridge",
            "n_emb": int(state["n_emb"]),
            "n_judge": int(state["n_judge"]),
            "alpha_emb": float(state["alpha_emb"]),
            "alpha_judge": float(state["alpha_judge"]),
            "judge_feature_names": list(judge_feature_names),
            "weights_npz": str(weights_npz),
        }
    )
    save_json(weights_json, meta)
    return weights_json, weights_npz

def _run_with_judge_features(
    *,
    args: argparse.Namespace,
    emb_cache: str,
    task_ids: List[str],
    X: np.ndarray,
    id_to_row: Dict[str, int],
    all_responses: List[Tuple[str, Dict[str, int]]],
    overlap_ids: List[str],
) -> int:
    zero_success_ids = compute_zero_success_items(all_responses)
    zero_success_set = set(zero_success_ids)
    exclude_zero_success = bool(getattr(args, "exclude_zero_success", False))
    zero_success_mode = "exclude" if exclude_zero_success else "include"

    if exclude_zero_success:
        eligible = [tid for tid in overlap_ids if tid not in zero_success_set]
        print(
            f"Excluding zero-success items from CV/IRT: {len(overlap_ids) - len(eligible)}/{len(overlap_ids)} items "
            f"(agent_results={args.agent_results})"
        )
    else:
        eligible = list(overlap_ids)
        if zero_success_set:
            print(
                f"Including zero-success items in CV/IRT: {len(zero_success_set)}/{len(overlap_ids)} items "
                f"(agent_results={args.agent_results})"
            )
    if not eligible:
        raise RuntimeError("After filtering, no items remain for CV.")

    oracle_meta, oracle_theta_by_subject, oracle_diff_by_item = train_oracle_irt_1pl_and_save(
        args=args,
        all_responses=all_responses,
        item_ids=list(eligible),
    )
    try:
        print(f"Wrote oracle IRT: {oracle_meta.get('oracle_irt_dir', '')}")
    except Exception:
        pass

    Xy = np.stack([X[id_to_row[tid]] for tid in eligible], axis=0).astype(np.float32)

    feat_dir = str(getattr(args, "judge_features_dir", "")).strip()
    if not feat_dir:
        raise ValueError("--judge_features_dir must be set when --method=combined is used.")
    schema = "csv" if _looks_like_csv_path(feat_dir) else "dir_json"
    idx = _build_judge_index(feat_dir)
    if _looks_like_csv_path(feat_dir):
        judge_feature_names = _load_judge_csv_feature_names(feat_dir)
    else:
        judge_feature_names = JUDGE_FEATURE_NAMES

    outer_cv = KFold(n_splits=int(args.cv_folds), shuffle=True, random_state=int(args.seed))
    fold_aucs: List[float] = []
    fold_aucs_oracle_irt: List[float] = []
    fold_n_obs: List[int] = []
    fold_n_items_scored: List[int] = []
    fold_aucs_embedding_only: List[float] = []
    fold_alpha_emb: List[float] = []
    fold_alpha_judge: List[float] = []

    eligible_index = {tid: i for i, tid in enumerate(eligible)}
    yhat_oof = np.full((int(len(eligible)),), np.nan, dtype=np.float64)
    fold_of_item = np.full((int(len(eligible)),), -1, dtype=np.int32)

    best_fold_auc = -float("inf")
    best_fold = -1
    best_joint_state: Optional[dict] = None

    for fold, (tr, te) in enumerate(outer_cv.split(Xy), start=1):
        train_items = [eligible[int(i)] for i in tr.tolist()]
        test_items = [eligible[int(i)] for i in te.tolist()]

        fold_root = os.path.join(str(args.out_dir), "irt_folds", f"fold_{int(fold):02d}")
        ensure_dir(fold_root)

        irt_dir = os.path.join(str(fold_root), "1d_1pl")
        if os.path.exists(irt_dir):
            shutil.rmtree(irt_dir, ignore_errors=True)

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

        set_torch_determinism(False)
        seed_everything(int(args.seed), deterministic=False)
        theta_by_subject, diff_by_item = train_irt_1pl(
            responses_jsonl=train_jsonl,
            epochs=int(args.irt_epochs),
            device=str(irt_device),
            seed=int(args.seed),
            out_dir=str(irt_dir),
        )
        set_torch_determinism(True)
        seed_everything(int(args.seed), deterministic=True)
        if not theta_by_subject or not diff_by_item:
            raise RuntimeError(f"Fold {fold}: IRT produced empty outputs under {irt_dir}")

        train_labeled = [tid for tid in train_items if tid in diff_by_item]
        if len(train_labeled) < 2:
            raise RuntimeError(f"Fold {fold}: only {len(train_labeled)} train items had IRT difficulties.")

        seed_everything(int(args.seed) + int(fold), deterministic=True)
        X_train = np.stack([X[id_to_row[tid]] for tid in train_labeled], axis=0).astype(np.float32)
        y_train = np.array([float(diff_by_item[tid]) for tid in train_labeled], dtype=np.float32)

        reg = str(args.regressor or "ridge_cv").strip()
        if reg == "linear":
            raise ValueError("--regressor=linear is not supported when --method=combined is used.")
        emb_model = None

        emb_model = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("ridge", Ridge(alpha=float(args.ridge_alpha))),
            ]
        )
        if reg == "ridge_cv":

            alphas = np.array(
                [float(x.strip()) for x in str(args.ridge_alphas).split(",") if x.strip()],
                dtype=np.float64,
            )
            req_inner = int(args.inner_splits)
            inner_splits = int(min(req_inner, max(2, int(len(train_labeled)))))
            inner_cv = KFold(n_splits=int(inner_splits), shuffle=True, random_state=int(args.seed) + int(fold))
            emb_model = Pipeline(
                steps=[
                    ("scaler", StandardScaler(with_mean=True, with_std=True)),
                    ("ridge", RidgeCV(alphas=alphas, cv=inner_cv)),
                ]
            )
        emb_model.fit(X_train, y_train)

        X_test = np.stack([X[id_to_row[tid]] for tid in test_items], axis=0).astype(np.float32)
        emb_pred_test = emb_model.predict(X_test).astype(np.float32)
        emb_pred_by_item_test = {tid: float(z) for tid, z in zip(test_items, emb_pred_test.tolist())}

        joint_emb_train_rows: List[np.ndarray] = []
        joint_judge_train_rows: List[np.ndarray] = []
        joint_y_train_rows: List[float] = []
        joint_train_items_used: List[str] = []
        for tid in train_labeled:
            j = _load_judge_vector(tid, features_dir=feat_dir, feature_names=judge_feature_names, index=idx)
            if j is None:
                continue
            joint_emb_train_rows.append(X[id_to_row[tid]].astype(np.float32))
            joint_judge_train_rows.append(j.astype(np.float32))
            joint_y_train_rows.append(float(diff_by_item[tid]))
            joint_train_items_used.append(tid)
        if len(joint_train_items_used) < 2:
            raise RuntimeError(
                f"Fold {fold}: only {len(joint_train_items_used)} train items had judge features; cannot fit joint block-ridge."
            )

        X_emb_joint_train = np.stack(joint_emb_train_rows, axis=0).astype(np.float32)
        X_judge_joint_train = np.stack(joint_judge_train_rows, axis=0).astype(np.float32)
        y_joint_train = np.asarray(joint_y_train_rows, dtype=np.float32)

        if reg == "ridge":
            alpha_emb = float(getattr(args, "ridge_alpha_emb", float("nan")))
            alpha_judge = float(getattr(args, "ridge_alpha_judge", float("nan")))
            if not math.isfinite(alpha_emb):
                alpha_emb = float(args.ridge_alpha)
            if not math.isfinite(alpha_judge):
                alpha_judge = float(args.ridge_alpha)
            joint_state = _fit_block_ridge(
                X_emb=X_emb_joint_train,
                X_judge=X_judge_joint_train,
                y=y_joint_train,
                alpha_emb=float(alpha_emb),
                alpha_judge=float(alpha_judge),
            )
        else:
            ae_grid_s = str(getattr(args, "ridge_alphas_emb", "") or "").strip() or str(args.ridge_alphas)
            aj_grid_s = str(getattr(args, "ridge_alphas_judge", "") or "").strip() or str(args.ridge_alphas)
            ae_grid = _parse_alpha_list(ae_grid_s)
            aj_grid = _parse_alpha_list(aj_grid_s)
            alpha_emb, alpha_judge, _inner_best_mse = _select_block_alphas_inner_cv(
                X_emb=X_emb_joint_train,
                X_judge=X_judge_joint_train,
                y=y_joint_train,
                alphas_emb=ae_grid,
                alphas_judge=aj_grid,
                inner_splits=int(args.inner_splits),
                seed=int(args.seed) + 2000 + int(fold),
            )
            joint_state = _fit_block_ridge(
                X_emb=X_emb_joint_train,
                X_judge=X_judge_joint_train,
                y=y_joint_train,
                alpha_emb=float(alpha_emb),
                alpha_judge=float(alpha_judge),
            )

        fold_alpha_emb.append(float(joint_state["alpha_emb"]))
        fold_alpha_judge.append(float(joint_state["alpha_judge"]))

        final_pred_by_item: Dict[str, float] = {}
        contrib_by_item: Dict[str, Dict[str, float]] = {}
        n_missing_judge = 0
        for tid in test_items:
            j = _load_judge_vector(tid, features_dir=feat_dir, feature_names=judge_feature_names, index=idx)
            if j is None:
                n_missing_judge += 1
                continue
            x_emb = X[id_to_row[tid]].reshape(1, -1).astype(np.float32)
            x_j = j.reshape(1, -1).astype(np.float32)
            z_final = float(_predict_block_ridge(joint_state, X_emb=x_emb, X_judge=x_j)[0])
            final_pred_by_item[tid] = z_final
            try:
                contrib = _decompose_block_ridge_single(joint_state, x_emb_raw=x_emb, x_judge_raw=x_j)
                contrib_by_item[tid] = {
                    "pred": float(contrib["pred"]),
                    "pred_raw": float(contrib["pred_raw"]),
                    "intercept_model": float(contrib["intercept_model"]),
                    "intercept_raw": float(contrib["intercept_raw"]),
                    "emb_dot_raw": float(contrib["emb_dot_raw"]),
                    "judge_dot_raw": float(contrib["judge_dot_raw"]),
                    "emb_contrib_std": float(contrib["emb_contrib_std"]),
                    "judge_contrib_std": float(contrib["judge_contrib_std"]),
                }
            except Exception:

                pass

        try:
            ensure_dir(fold_root)
            save_json(os.path.join(str(fold_root), "block_contributions_test_items.json"), contrib_by_item)
        except Exception:
            pass

        for tid in test_items:
            i = eligible_index.get(tid, None)
            if i is None:
                continue
            fold_of_item[int(i)] = int(fold)
            if tid in final_pred_by_item:
                yhat_oof[int(i)] = float(final_pred_by_item[tid])

        scored_items = set(final_pred_by_item.keys())
        scores_final: List[float] = []
        scores_emb: List[float] = []
        labels: List[int] = []
        scores_oracle: List[float] = []
        labels_oracle: List[int] = []
        test_set = set(test_items)
        for sid, resp in all_responses:
            th = theta_by_subject.get(sid, None)
            if th is None:
                continue
            theta = float(th)
            th_o = oracle_theta_by_subject.get(sid, None)
            for item_id, y_obs in resp.items():
                if item_id not in test_set:
                    continue
                if item_id not in scored_items:
                    continue
                z = final_pred_by_item.get(item_id, None)
                if z is None:
                    continue
                z_emb = emb_pred_by_item_test.get(item_id, None)
                if z_emb is None:
                    continue
                scores_final.append(_sigmoid(theta - float(z)))
                scores_emb.append(_sigmoid(theta - float(z_emb)))
                labels.append(int(y_obs))
                if th_o is not None:
                    b_o = oracle_diff_by_item.get(item_id, None)
                    if b_o is not None:
                        scores_oracle.append(_sigmoid(float(th_o) - float(b_o)))
                        labels_oracle.append(int(y_obs))

        if len(labels) < 2 or len(set(int(x) for x in labels)) < 2:
            fold_auc = float("nan")
            fold_auc_emb = float("nan")
        else:
            fold_auc = float(roc_auc_score(labels, scores_final))
            fold_auc_emb = float(roc_auc_score(labels, scores_emb))
        fold_aucs.append(float(fold_auc))
        if len(labels_oracle) < 2 or len(set(int(x) for x in labels_oracle)) < 2:
            fold_auc_oracle = float("nan")
        else:
            fold_auc_oracle = float(roc_auc_score(labels_oracle, scores_oracle))
        fold_aucs_oracle_irt.append(float(fold_auc_oracle))
        fold_aucs_embedding_only.append(float(fold_auc_emb))
        fold_n_obs.append(int(len(labels)))
        fold_n_items_scored.append(int(len(final_pred_by_item)))

        if fold_auc == fold_auc and float(fold_auc) > float(best_fold_auc):
            best_fold_auc = float(fold_auc)
            best_fold = int(fold)
            best_joint_state = joint_state

        if bool(getattr(args, "debug", False)):
            try:
                print(
                    f"Fold {fold:02d} debug: emb_auc={fold_auc_emb:.4f} final_auc={fold_auc:.4f} "
                    f"alpha_emb={float(joint_state['alpha_emb']):.3g} alpha_judge={float(joint_state['alpha_judge']):.3g}"
                )
            except Exception:
                pass

        print(f"Fold {fold:02d}: auc={fold_auc} oracle_auc={fold_auc_oracle}")

    auc_arr = np.asarray(fold_aucs, dtype=np.float64)
    auc_mean = float(np.nanmean(auc_arr)) if auc_arr.size else float("nan")
    auc_std = float(np.nanstd(auc_arr, ddof=0)) if auc_arr.size else float("nan")
    print(f"{int(args.cv_folds)}-fold CV test ROC-AUC: mean={auc_mean} std={auc_std}")

    oracle_auc_arr = np.asarray(fold_aucs_oracle_irt, dtype=np.float64)
    oracle_auc_mean = float(np.nanmean(oracle_auc_arr)) if oracle_auc_arr.size else float("nan")
    oracle_auc_std = float(np.nanstd(oracle_auc_arr, ddof=0)) if oracle_auc_arr.size else float("nan")
    print(f"{int(args.cv_folds)}-fold CV oracle ROC-AUC: mean={oracle_auc_mean} std={oracle_auc_std}")

    if best_joint_state is None or best_fold < 1:
        raise RuntimeError("Failed to select a best CV fold model by ROC-AUC (all folds NaN?).")

    zero_embedded = [tid for tid in task_ids if tid in zero_success_set] if exclude_zero_success else []
    yhat_zero: Dict[str, float] = {}
    if zero_embedded:
        for tid in zero_embedded:
            j = _load_judge_vector(tid, features_dir=feat_dir, feature_names=judge_feature_names, index=idx)
            if j is None:
                continue
            x_emb = X[id_to_row[tid]].reshape(1, -1).astype(np.float32)
            x_j = j.reshape(1, -1).astype(np.float32)
            yhat_zero[tid] = float(_predict_block_ridge(best_joint_state, X_emb=x_emb, X_judge=x_j)[0])

    pred_path = os.path.join(str(args.out_dir), "predictions.csv")
    with open(pred_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["item_id", "diff_pred", "split", "fold"])
        w.writeheader()

        for i, tid in enumerate(eligible):
            v = float(yhat_oof[int(i)])
            fold_id = int(fold_of_item[int(i)]) if int(fold_of_item[int(i)]) > 0 else ""
            split = "cv_val" if (v == v) else "missing_judge"
            w.writerow({"item_id": tid, "diff_pred": (v if v == v else ""), "split": split, "fold": fold_id})

        if yhat_zero:
            for tid, v in sorted(yhat_zero.items()):
                w.writerow({"item_id": tid, "diff_pred": float(v), "split": "zero_success", "fold": ""})

    metrics_out = os.path.join(str(args.out_dir), "metrics.json")
    save_json(
        metrics_out,
        {
            "n_items_total": int(len(task_ids)),
            "n_items_with_responses": int(len(overlap_ids)),
            "n_items_eligible_cv": int(len(eligible)),
            "exclude_zero_success": bool(exclude_zero_success),
            "seed": int(args.seed),
            "cv_best_auc_fold": int(best_fold),
            "cv_best_auc": float(best_fold_auc),
            "cv_test_auc_folds": [float(x) for x in fold_aucs],
            "cv_test_auc_mean": float(auc_mean),
            "cv_test_auc_std": float(auc_std),
            "cv_test_auc_folds_oracle_irt": [float(x) for x in fold_aucs_oracle_irt],
            "cv_test_auc_mean_oracle_irt": float(oracle_auc_mean),
            "cv_test_auc_std_oracle_irt": float(oracle_auc_std),
            "cv_test_auc_folds_embedding_only": [float(x) for x in fold_aucs_embedding_only],
            "oracle_irt_dir": str(oracle_meta.get("oracle_irt_dir", "")),
        },
    )

    print(f"Wrote metrics: {metrics_out}")
    print(f"Wrote predictions: {pred_path}")
    return 0

def _run_judge_only(
    *,
    args: argparse.Namespace,
    task_ids: List[str],
    all_responses: List[Tuple[str, Dict[str, int]]],
    overlap_ids: List[str],
    dataset_sources_str: str,
    dataset_name: Optional[str],
    dataset_path: Optional[str],
    split: str,
    instruction_signature: str,
) -> int:
    method = "judge"
    regressor_name = str(args.regressor or "ridge_cv").strip()
    if regressor_name == "linear":
        raise ValueError("--regressor=linear is not supported when --method=judge is used.")

    feat_dir = str(getattr(args, "judge_features_dir", "")).strip()
    if not feat_dir:
        raise ValueError("--judge_features_dir must be set when --method=judge is used.")

    schema = "csv" if _looks_like_csv_path(feat_dir) else "dir_json"
    idx = _build_judge_index(feat_dir)
    if _looks_like_csv_path(feat_dir):
        judge_feature_names = _load_judge_csv_feature_names(feat_dir)
    else:
        judge_feature_names = JUDGE_FEATURE_NAMES

    zero_success_ids = compute_zero_success_items(all_responses)
    zero_success_set = set(zero_success_ids)
    exclude_zero_success = bool(getattr(args, "exclude_zero_success", False))
    zero_success_mode = "exclude" if exclude_zero_success else "include"

    if exclude_zero_success:
        eligible = [tid for tid in overlap_ids if tid not in zero_success_set]
        print(
            f"Excluding zero-success items from CV/IRT: {len(overlap_ids) - len(eligible)}/{len(overlap_ids)} items "
            f"(agent_results={args.agent_results})"
        )
    else:
        eligible = list(overlap_ids)
        if zero_success_set:
            print(
                f"Including zero-success items in CV/IRT: {len(zero_success_set)}/{len(overlap_ids)} items "
                f"(agent_results={args.agent_results})"
            )
    if not eligible:
        raise RuntimeError("After filtering, no items remain for CV/IRT.")

    eligible_used: List[str] = []
    judge_rows: List[np.ndarray] = []
    for tid in eligible:
        j = _load_judge_vector(tid, features_dir=feat_dir, feature_names=judge_feature_names, index=idx)
        if j is None:
            continue
        eligible_used.append(tid)
        judge_rows.append(j.astype(np.float32, copy=False))
    n_dropped_missing = int(len(eligible) - len(eligible_used))
    if n_dropped_missing:
        print(
            f"WARNING: --method=judge: dropped {n_dropped_missing}/{len(eligible)} eligible items with missing judge features "
            f"(judge_features_dir={feat_dir})."
        )
    eligible = eligible_used
    if not eligible:
        raise RuntimeError("After filtering to items with judge features, no items remain for CV/IRT.")

    oracle_meta, oracle_theta_by_subject, oracle_diff_by_item = train_oracle_irt_1pl_and_save(
        args=args,
        all_responses=all_responses,
        item_ids=list(eligible),
    )
    try:
        print(f"Wrote oracle IRT: {oracle_meta.get('oracle_irt_dir', '')}")
    except Exception:
        pass
    Xy = np.stack(judge_rows, axis=0).astype(np.float32)

    alphas: np.ndarray = np.array([], dtype=np.float64)

    def _make_model(*, n_train: int, fold_seed: int):
        nonlocal alphas
        if regressor_name == "ridge":
            alpha = float(args.ridge_alpha)
            if not (alpha > 0):
                raise ValueError("--ridge_alpha must be > 0")
            return Pipeline(steps=[("scaler", StandardScaler(with_mean=True, with_std=True)), ("ridge", Ridge(alpha=alpha))])
        if regressor_name == "ridge_cv":
            try:
                alphas = np.array([float(x.strip()) for x in str(args.ridge_alphas).split(",") if x.strip()], dtype=np.float64)
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

    outer_cv = KFold(n_splits=int(args.cv_folds), shuffle=True, random_state=int(args.seed))
    cv_test_auc_folds: List[float] = []
    cv_test_auc_folds_oracle_irt: List[float] = []
    cv_test_n_obs_folds: List[int] = []
    yhat_oof = np.full((int(len(eligible)),), np.nan, dtype=np.float64)
    fold_of_item = np.full((int(len(eligible)),), -1, dtype=np.int32)

    best_fold_auc = -float("inf")
    best_fold = -1
    best_model = None

    eligible_index = {tid: i for i, tid in enumerate(eligible)}

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

        set_torch_determinism(False)
        seed_everything(int(args.seed), deterministic=False)
        theta_by_subject, diff_by_item = train_irt_1pl(
            responses_jsonl=train_jsonl,
            epochs=int(args.irt_epochs),
            device=str(irt_device),
            seed=int(args.seed),
            out_dir=os.path.join(fold_root, "1d_1pl"),
        )
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

        seed_everything(int(args.seed) + int(fold), deterministic=True)
        train_idx = [int(eligible_index[tid]) for tid in train_labeled]
        X_train = Xy[np.asarray(train_idx, dtype=np.int64)]
        y_train = np.array([float(diff_by_item[tid]) for tid in train_labeled], dtype=np.float32)

        m = _make_model(n_train=int(len(train_labeled)), fold_seed=int(args.seed) + int(fold))
        m.fit(X_train, y_train)

        X_test = Xy[np.asarray(te, dtype=np.int64)]
        pred = m.predict(X_test).astype(np.float64)
        yhat_oof[te] = pred
        fold_of_item[te] = int(fold)

        z_by_item = {tid: float(z) for tid, z in zip(test_items, pred.tolist())}
        scores: List[float] = []
        labels: List[int] = []
        scores_oracle: List[float] = []
        labels_oracle: List[int] = []
        test_set = set(test_items)
        for sid, resp in all_responses:
            theta = theta_by_subject.get(sid, None)
            if theta is None:
                continue
            th = float(theta)
            th_o = oracle_theta_by_subject.get(sid, None)
            for item_id, y_obs in resp.items():
                if item_id not in test_set:
                    continue
                z = z_by_item.get(item_id, None)
                if z is None:
                    continue
                scores.append(_sigmoid(th - float(z)))
                labels.append(int(y_obs))
                if th_o is not None:
                    b_o = oracle_diff_by_item.get(item_id, None)
                    if b_o is not None:
                        scores_oracle.append(_sigmoid(float(th_o) - float(b_o)))
                        labels_oracle.append(int(y_obs))

        fold_auc = float(_compute_binary_auroc(scores, labels))
        fold_auc_oracle = float(_compute_binary_auroc(scores_oracle, labels_oracle))
        cv_test_auc_folds.append(float(fold_auc))
        cv_test_auc_folds_oracle_irt.append(float(fold_auc_oracle))
        cv_test_n_obs_folds.append(int(len(labels)))
        print(f"Fold {fold:02d}: auc={fold_auc} oracle_auc={fold_auc_oracle}")
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

    oracle_auc_arr = np.asarray(cv_test_auc_folds_oracle_irt, dtype=np.float64)
    oracle_auc_mean = float(np.nanmean(oracle_auc_arr)) if oracle_auc_arr.size else float("nan")
    oracle_auc_std = float(np.nanstd(oracle_auc_arr, ddof=0)) if oracle_auc_arr.size else float("nan")
    print(f"{int(args.cv_folds)}-fold CV oracle ROC-AUC: mean={oracle_auc_mean} std={oracle_auc_std}")

    model = best_model

    zero_items: List[str] = []
    yhat_zero: Optional[np.ndarray] = None
    if exclude_zero_success and zero_success_set:
        zero_items = [tid for tid in task_ids if tid in zero_success_set]
        zero_rows: List[np.ndarray] = []
        zero_used: List[str] = []
        for tid in zero_items:
            j = _load_judge_vector(tid, features_dir=feat_dir, feature_names=judge_feature_names, index=idx)
            if j is None:
                continue
            zero_used.append(tid)
            zero_rows.append(j.astype(np.float32, copy=False))
        zero_items = zero_used
        if zero_rows:
            X_zero = np.stack(zero_rows, axis=0).astype(np.float32)
            yhat_zero = model.predict(X_zero).astype(np.float64)
        else:
            print("NOTE: zero-success ids provided, but none had judge features; nothing to predict.")

    metrics = {
        "method": method,
        "n_items_total": int(len(task_ids)),
        "n_items_with_responses": int(len(overlap_ids)),
        "n_items_eligible_cv_irt": int(len(eligible)),
        "exclude_zero_success": bool(exclude_zero_success),
        "seed": int(args.seed),
        "cv_best_auc_fold": int(best_fold),
        "cv_best_auc": float(best_fold_auc),
        "cv_test_auc_folds": [float(x) for x in cv_test_auc_folds],
        "cv_test_auc_mean": float(auc_mean),
        "cv_test_auc_std": float(auc_std),
        "cv_test_auc_folds_oracle_irt": [float(x) for x in cv_test_auc_folds_oracle_irt],
        "cv_test_auc_mean_oracle_irt": float(oracle_auc_mean),
        "cv_test_auc_std_oracle_irt": float(oracle_auc_std),
        "oracle_irt_dir": str(oracle_meta.get("oracle_irt_dir", "")),
    }
    save_json(os.path.join(str(args.out_dir), "metrics.json"), metrics)

    pred_path = os.path.join(str(args.out_dir), "predictions.csv")
    with open(pred_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item_id", "diff_pred", "split", "fold"])
        w.writeheader()
        for i, tid in enumerate(eligible):
            w.writerow({"item_id": tid, "diff_pred": float(yhat_oof[i]), "split": "cv_val", "fold": int(fold_of_item[i])})
        if yhat_zero is not None and zero_items:
            for tid, score in zip(zero_items, yhat_zero.tolist()):
                w.writerow({"item_id": tid, "diff_pred": float(score), "split": "zero_success", "fold": ""})

    print(f"Wrote metrics: {os.path.join(str(args.out_dir), 'metrics.json')}")
    print(f"Wrote predictions: {pred_path}")
    return 0

def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--benchmark",
        type=str,
        default="",
        help=(
            "Benchmark name to infer dataset, agent_results, judge_features_dir, out_dir. "
            "One of: verified, pro, terminal-bench, gso. If set, overrides --dataset_name, --dataset_path, "
            "--agent_results, --judge_features_dir, --out_dir with standard paths for that benchmark."
        ),
    )
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

    p.add_argument("--out_dir", type=str, default="output/swebench_verified")
    p.add_argument("--embeddings_cache", type=str, default="", help="Optional path to existing embeddings cache (.npz).")
    p.add_argument("--overwrite", action="store_true")

    p.add_argument(
        "--agent_results",
        type=str,
        default="data/swebench_verified/responses.jsonl",
        help=(
            "Path to a JSONL file with per-subject responses of the form "
            "{'subject_id': ..., 'responses': {'task_id': 0/1, ...}}."
        ),
    )
    p.add_argument(
        "--exclude_zero_success",
        action="store_true",
        help=(
            "If set, exclude items with 0 successes across all subjects in --agent_results from the CV/IRT pool. "
            "By default, these items are included in CV/IRT."
        ),
    )
    p.add_argument("--irt_epochs", type=int, default=5000)
    p.add_argument("--irt_device", type=str, default="cuda", help="Device for IRT training (cuda or cpu).")

    p.add_argument(
        "--method",
        type=str,
        default="embedding",
        choices=["embedding", "judge", "combined"],
        help=(
            "Which features to use for difficulty prediction. "
            "'embedding' (default) trains ridge/linear on the embedding vector only (historical default). "
            "'combined' uses embedding + LLM-judge features with a joint (block) ridge (separate penalties per block). "
            "'judge' trains ridge on judge features only (no embeddings)."
        ),
    )
    p.add_argument(
        "--regressor",
        type=str,
        default="ridge_cv",
        choices=["linear", "ridge", "ridge_cv"],
        help="Regression model (same options as trajectory-based script).",
    )
    p.add_argument("--ridge_alpha", type=float, default=10000.0)
    p.add_argument("--ridge_alphas", type=str, default="1e-4,1e-3,1e-2,1e-1,1,10,100,1000,10000")
    p.add_argument("--cv_folds", type=int, default=5)
    p.add_argument(
        "--inner_splits",
        type=int,
        default=5,
        help="Inner CV splits for RidgeCV (used when --regressor=ridge_cv). Will be capped by train size; must be >=2.",
    )

    p.add_argument(
        "--judge_features_dir",
        type=str,
        default="llm_judge_features/defaults/swebench_verified/llm_judge_features.csv",
        help=(
            "Judge features (CSV like llm_judge_features/defaults/swebench_verified/llm_judge_features.csv, "
            "or directory of per-task JSONs)."
        ),
    )
    p.add_argument(
        "--ridge_alpha_emb",
        type=float,
        default=float("nan"),
        help="(combined mode, --regressor=ridge) Embedding block alpha. Defaults to --ridge_alpha when unset.",
    )
    p.add_argument(
        "--ridge_alpha_judge",
        type=float,
        default=float("nan"),
        help="(combined mode, --regressor=ridge) Judge block alpha. Defaults to --ridge_alpha when unset.",
    )
    p.add_argument(
        "--ridge_alphas_emb",
        type=str,
        default="",
        help="(combined mode, --regressor=ridge_cv) Embedding alpha grid. Defaults to --ridge_alphas when unset.",
    )
    p.add_argument(
        "--ridge_alphas_judge",
        type=str,
        default="",
        help="(combined mode, --regressor=ridge_cv) Judge alpha grid. Defaults to --ridge_alphas when unset.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="(combined mode) Print per-fold debug diagnostics.",
    )

    args = p.parse_args(argv)
    raw_argv = list(argv) if argv is not None else []
    if not raw_argv:
        raw_argv = list(sys.argv[1:])
    argv_str = " " + " ".join(str(x) for x in raw_argv) + " " if raw_argv else ""

    def _was_passed(flag: str) -> bool:
        return f" {flag}" in argv_str or f" {flag}=" in argv_str

    benchmark_spec = str(getattr(args, "benchmark", "") or "").strip()
    if benchmark_spec:
        defaults = _get_benchmark_defaults(benchmark_spec)
        if not _was_passed("--dataset_name"):
            args.dataset_name = defaults["dataset_name"]
        if not _was_passed("--dataset_path"):
            args.dataset_path = defaults["dataset_path"]
        if not _was_passed("--split"):
            args.split = defaults["split"]
        if not _was_passed("--agent_results"):
            args.agent_results = defaults["agent_results"]
        if not _was_passed("--judge_features_dir"):
            args.judge_features_dir = defaults["judge_features_dir"]
        if not _was_passed("--out_dir"):
            args.out_dir = defaults["out_dir"]
        print(f"Using benchmark={benchmark_spec}: dataset={'path=' + args.dataset_path if args.dataset_path else 'name=' + args.dataset_name}, "
              f"agent_results={args.agent_results}, out_dir={args.out_dir}")

    ensure_dir(args.out_dir)
    seed_everything(int(args.seed), deterministic=True)

    method = str(getattr(args, "method", "embedding") or "embedding").strip().lower()
    if method not in {"embedding", "judge", "combined"}:
        raise ValueError(f"Unknown --method: {getattr(args, 'method', None)!r}")

    dataset_name = str(args.dataset_name).strip()
    dataset_path = str(args.dataset_path).strip()
    if dataset_path:
        dataset_sources_str = f"json:{os.path.basename(dataset_path) or 'dataset.jsonl'}"
    else:
        dataset_sources_str = dataset_name or "princeton-nlp/SWE-bench_Verified"

    instr_sig = prompt_signature(str(args.instruction))
    emb_cache = ""

    if method in {"judge", "combined"}:
        feat_dir = str(getattr(args, "judge_features_dir", "") or "").strip()

    if method == "judge":
        if str(args.embeddings_cache or "").strip():
            print("NOTE: --method=judge ignores --embeddings_cache (no embedding is performed).")

        items = list(
            iter_swebench_items(
                dataset_name=str(dataset_name),
                split=str(args.split),
                dataset_path=str(dataset_path),
            )
        )
        ids_sorted = sorted(
            normalize_swebench_item_id(str(it.item_id).strip())
            for it in items
            if str(it.item_id).strip()
        )
        task_ids = ids_sorted
        X = None
        id_to_row = {tid: i for i, tid in enumerate(task_ids)}
    else:

        safe_backbone = str(args.backbone).replace("/", "__")
        layer_flag = "" if int(args.embedding_layer) == -1 else f"__layer{int(args.embedding_layer)}"
        idnorm_flag = "__idnorm_instance-v1"
        emb_cache = str(args.embeddings_cache or "").strip()
        if not emb_cache:
            cache_meta = {
                "backbone": str(args.backbone),
                "max_length": int(args.max_length),
                "batch_size": int(args.batch_size),
                "device_map": str(args.device_map),
                "torch_dtype": str(args.torch_dtype),
                "attn_implementation": str(args.attn_implementation),
                "instruction": str(args.instruction),
                "instruction_sig": str(instr_sig),
                "embedding_layer": int(args.embedding_layer),
                "normalize_item_ids": True,
                "idnorm_flag": idnorm_flag,
                "dataset_sources": str(dataset_sources_str),
                "split": str(args.split),
            }
            cache_key = hashlib.sha1(json.dumps(cache_meta, sort_keys=True).encode("utf-8")).hexdigest()[:12]
            model_short = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(safe_backbone))[:48].strip("_") or "model"
            short_basename = f"embeddings__{model_short}__{cache_key}__maxlen{int(args.max_length)}.npz"
            emb_cache = os.path.join(_shared_embeddings_dir(), short_basename)

        if not bool(args.overwrite):
            explicit_cache = str(args.embeddings_cache or "").strip()
            if explicit_cache:
                loaded = load_compatible_embeddings_cache(
                    emb_cache,
                    backbone=str(args.backbone),
                    max_length=int(args.max_length),
                    instruction_sig=str(instr_sig),
                )
                if loaded is not None:
                    task_ids, X, meta = loaded
                    print(
                        f"Loaded embeddings cache (explicit): {emb_cache} "
                        f"(n={len(task_ids)}, dim={X.shape[1]}, embedding_layer={meta.get('embedding_layer', -1)})"
                    )
                elif os.path.exists(emb_cache):
                    raise RuntimeError(
                        f"Embeddings cache (explicit) was incompatible with this run: {emb_cache}. "
                        "Use --overwrite, or point --embeddings_cache to a compatible file."
                    )
                else:
                    print(
                        f"WARNING: --embeddings_cache was provided but file does not exist: {emb_cache} "
                        f"(cwd={os.getcwd()}). Will recompute embeddings."
                    )
            else:
                search_roots = _candidate_embedding_roots(out_dir=str(args.out_dir))
                found = find_compatible_embeddings_cache(
                    preferred_paths=[str(emb_cache)],
                    search_roots=search_roots,
                    backbone=str(args.backbone),
                    max_length=int(args.max_length),
                    instruction_sig=str(instr_sig),
                )
                if found is not None:
                    found_path, task_ids, X, meta = found
                    emb_cache = str(found_path)
                    print(
                        f"Loaded embeddings cache (auto): {emb_cache} "
                        f"(n={len(task_ids)}, dim={X.shape[1]}, embedding_layer={meta.get('embedding_layer', -1)})"
                    )

        if X is None:

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
                text_format=np.array([str(EMBEDDING_TEXT_FORMAT)], dtype=object),
                includes_solution=np.array([True], dtype=np.bool_),
                backbone=np.array([str(args.backbone)], dtype=object),
                max_length=np.array([int(args.max_length)], dtype=np.int64),
                embedding_dim=np.array([int(emb_dim)], dtype=np.int64),
                embedding_layer=np.array([int(args.embedding_layer)], dtype=np.int64),
            )
            print(
                f"Wrote embeddings cache: {emb_cache} (n={len(ids_sorted)}, dim={X.shape[1]}, embedding_layer={int(args.embedding_layer)})"
            )
            task_ids = ids_sorted

    id_to_row = {tid: i for i, tid in enumerate(task_ids)}

    all_responses = load_all_responses(str(args.agent_results))
    if not all_responses:
        raise RuntimeError(f"Loaded 0 subject responses from --agent_results={args.agent_results!r}")

    response_items: Set[str] = set()
    for _, resp in all_responses:
        response_items.update(resp.keys())

    overlap_ids = [tid for tid in task_ids if tid in response_items]
    if not overlap_ids:
        raise RuntimeError("No overlap between dataset task_ids and item_ids found in --agent_results responses.")

    if method == "judge":
        return _run_judge_only(
            args=args,
            task_ids=list(task_ids),
            all_responses=all_responses,
            overlap_ids=overlap_ids,
            dataset_sources_str=str(dataset_sources_str),
            dataset_name=(dataset_name or None),
            dataset_path=(dataset_path or None),
            split=str(args.split),
            instruction_signature=str(instr_sig),
        )
    if method == "combined":
        return _run_with_judge_features(
            args=args,
            emb_cache=str(emb_cache),
            task_ids=list(task_ids),
            X=X,
            id_to_row=dict(id_to_row),
            all_responses=all_responses,
            overlap_ids=overlap_ids,
        )

    zero_success_ids = compute_zero_success_items(all_responses)
    zero_success_set = set(zero_success_ids)

    exclude_zero_success = bool(getattr(args, "exclude_zero_success", False))
    zero_success_mode = "exclude" if exclude_zero_success else "include"

    if exclude_zero_success:
        eligible = [tid for tid in overlap_ids if tid not in zero_success_set]
        print(
            f"Excluding zero-success items from CV/IRT: {len(overlap_ids) - len(eligible)}/{len(overlap_ids)} items "
            f"(agent_results={args.agent_results})"
        )
    else:
        eligible = list(overlap_ids)
        if zero_success_set:
            print(
                f"Including zero-success items in CV/IRT: {len(zero_success_set)}/{len(overlap_ids)} items "
                f"(agent_results={args.agent_results})"
            )

    if not eligible:
        raise RuntimeError("After filtering, no items remain for CV/IRT.")

    oracle_meta, oracle_theta_by_subject, oracle_diff_by_item = train_oracle_irt_1pl_and_save(
        args=args,
        all_responses=all_responses,
        item_ids=list(eligible),
    )
    try:
        print(f"Wrote oracle IRT: {oracle_meta.get('oracle_irt_dir', '')}")
    except Exception:
        pass

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

    outer_cv = KFold(n_splits=int(args.cv_folds), shuffle=True, random_state=int(args.seed))
    cv_test_auc_folds: List[float] = []
    cv_test_auc_folds_oracle_irt: List[float] = []
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

        set_torch_determinism(False)
        seed_everything(int(args.seed), deterministic=False)

        theta_by_subject, diff_by_item = train_irt_1pl(
            responses_jsonl=train_jsonl,
            epochs=int(args.irt_epochs),
            device=str(irt_device),
            seed=int(args.seed),
            out_dir=os.path.join(fold_root, "1d_1pl"),
        )

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

        seed_everything(int(args.seed) + int(fold), deterministic=True)

        X_train = np.stack([X[id_to_row[tid]] for tid in train_labeled], axis=0).astype(np.float32)
        y_train = np.array([float(diff_by_item[tid]) for tid in train_labeled], dtype=np.float32)

        m = _make_model(n_train=int(len(train_labeled)), fold_seed=int(args.seed) + int(fold))
        m.fit(X_train, y_train)

        X_test = np.stack([X[id_to_row[tid]] for tid in test_items], axis=0).astype(np.float32)
        pred = m.predict(X_test).astype(np.float64)
        yhat_oof[te] = pred
        fold_of_item[te] = int(fold)

        z_by_item = {tid: float(z) for tid, z in zip(test_items, pred.tolist())}
        scores: List[float] = []
        labels: List[int] = []
        scores_oracle: List[float] = []
        labels_oracle: List[int] = []
        test_set = set(test_items)
        for sid, resp in all_responses:
            theta = theta_by_subject.get(sid, None)
            if theta is None:
                continue
            th = float(theta)
            th_o = oracle_theta_by_subject.get(sid, None)
            for item_id, y_obs in resp.items():
                if item_id not in test_set:
                    continue
                z = z_by_item.get(item_id, None)
                if z is None:
                    continue
                scores.append(1.0 / (1.0 + math.exp(-(th - float(z)))))
                labels.append(int(y_obs))
                if th_o is not None:
                    b_o = oracle_diff_by_item.get(item_id, None)
                    if b_o is not None:
                        scores_oracle.append(_sigmoid(float(th_o) - float(b_o)))
                        labels_oracle.append(int(y_obs))

        fold_auc = float(_compute_binary_auroc(scores, labels))
        fold_auc_oracle = float(_compute_binary_auroc(scores_oracle, labels_oracle))
        cv_test_auc_folds.append(float(fold_auc))
        cv_test_auc_folds_oracle_irt.append(float(fold_auc_oracle))
        cv_test_n_obs_folds.append(int(len(labels)))
        print(f"Fold {fold:02d}: auc={fold_auc} oracle_auc={fold_auc_oracle}")
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

    oracle_auc_arr = np.asarray(cv_test_auc_folds_oracle_irt, dtype=np.float64)
    oracle_auc_mean = float(np.nanmean(oracle_auc_arr)) if oracle_auc_arr.size else float("nan")
    oracle_auc_std = float(np.nanstd(oracle_auc_arr, ddof=0)) if oracle_auc_arr.size else float("nan")
    print(f"{int(args.cv_folds)}-fold CV oracle ROC-AUC: mean={oracle_auc_mean} std={oracle_auc_std}")

    model = best_model

    metrics = {
        "n_items_total": int(len(task_ids)),
        "n_items_with_responses": int(len(overlap_ids)),
        "n_items_eligible_cv_irt": int(len(eligible)),
        "exclude_zero_success": bool(str(zero_success_mode) == "exclude"),
        "seed": int(args.seed),
        "cv_best_auc_fold": int(best_fold),
        "cv_best_auc": float(best_fold_auc),
        "cv_test_auc_folds": [float(x) for x in cv_test_auc_folds],
        "cv_test_auc_mean": float(auc_mean),
        "cv_test_auc_std": float(auc_std),
        "cv_test_auc_folds_oracle_irt": [float(x) for x in cv_test_auc_folds_oracle_irt],
        "cv_test_auc_mean_oracle_irt": float(oracle_auc_mean),
        "cv_test_auc_std_oracle_irt": float(oracle_auc_std),
        "oracle_irt_dir": str(oracle_meta.get("oracle_irt_dir", "")),
    }

    zero_embedded: List[str] = []
    yhat_zero: Optional[np.ndarray] = None
    if str(zero_success_mode) == "exclude" and zero_success_set:
        zero_embedded = [tid for tid in task_ids if tid in zero_success_set]
        if zero_embedded:
            X_zero = np.stack([X[id_to_row[tid]] for tid in zero_embedded], axis=0).astype(np.float32)
            yhat_zero = model.predict(X_zero).astype(np.float64)
        else:
            print("NOTE: zero-success ids provided, but none were present in embedded task_ids; nothing to predict.")

    save_json(os.path.join(args.out_dir, "metrics.json"), metrics)

    pred_path = os.path.join(args.out_dir, "predictions.csv")
    with open(pred_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item_id", "diff_pred", "split", "fold"])
        w.writeheader()

        for i, tid in enumerate(eligible):
            w.writerow(
                {
                    "item_id": tid,
                    "diff_pred": float(yhat_oof[i]),
                    "split": "cv_val",
                    "fold": int(fold_of_item[i]),
                }
            )

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
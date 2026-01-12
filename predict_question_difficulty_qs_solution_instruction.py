#!/usr/bin/env python3
"""
Embed SWE-bench Verified tasks using inputs of the form:

  question statement + solution + instruction

and fit a regression model to predict per-question difficulty.

This is a sibling of `predict_question_difficulty.py`, but it:
- Loads tasks directly from the HF `datasets` hub (default: SWE-bench Verified test split).
- Uses the dataset's solution patch (no trajectories).
- Embeds ~500 inputs (SWE-bench Verified test size), rather than ~56k trajectories.

Example:
  /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/.venv/bin/python \
    /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/predict_question_difficulty_qs_solution_instruction.py \
    --difficulties /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/out/question_difficulties.csv \
    --backbone Qwen/Qwen2.5-Coder-14B \
    --max_length 1024 \
    --batch_size 1 \
    --device_map auto \
    --out_dir /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/out/qwen25coder14b_qs_sol_instr_lr
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple


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

import numpy as np
import torch
from datasets import load_dataset  # type: ignore
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import inspect


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
    xs: List[Tuple[float, int]] = []
    for i, s in enumerate(ids):
        h = hashlib.md5((str(s) + f"::{seed}").encode("utf-8")).hexdigest()
        x = int(h[:8], 16) / float(16**8)
        xs.append((x, i))
    xs.sort()
    n_test = int(round(len(ids) * float(test_fraction)))
    test = [i for _, i in xs[:n_test]]
    train = [i for _, i in xs[n_test:]]
    return train, test


def last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # last_hidden_state: [B, T, H], attention_mask: [B, T]
    lengths = attention_mask.sum(dim=1).clamp(min=1)  # [B]
    idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, last_hidden_state.size(-1))
    return last_hidden_state.gather(dim=1, index=idx).squeeze(1)  # [B, H]


def load_difficulties_csv(path: str) -> Dict[str, float]:
    diffs: Dict[str, float] = {}
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        if "item_id" not in (r.fieldnames or []) or "diff" not in (r.fieldnames or []):
            raise ValueError(f"Expected columns item_id,diff in {path}; got {r.fieldnames}")
        for row in r:
            diffs[str(row["item_id"])] = float(row["diff"])
    return diffs


def load_difficulties_ordered(path: str) -> List[Tuple[int, str]]:
    """
    Load an (item_ix, item_id) list in file order.
    Used to export predictions in the same schema as IRT difficulty CSVs.
    """
    out: List[Tuple[int, str]] = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        if "item_ix" not in (r.fieldnames or []) or "item_id" not in (r.fieldnames or []):
            raise ValueError(f"Expected columns item_ix,item_id in {path}; got {r.fieldnames}")
        for row in r:
            out.append((int(row["item_ix"]), str(row["item_id"])))
    out.sort(key=lambda x: x[0])
    return out


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


@dataclass(frozen=True)
class ItemRecord:
    item_id: str
    question_statement: str
    solution: str


def iter_swebench_verified_items(
    *,
    dataset_name: str,
    split: str,
    n_inputs: int,
    seed: int,
    shuffle: bool,
) -> Iterator[ItemRecord]:
    ds = load_dataset(str(dataset_name), split=str(split))
    n_total = len(ds)
    if n_total == 0:
        raise RuntimeError(f"Loaded empty dataset: {dataset_name} split={split}")

    idxs = list(range(n_total))
    if shuffle:
        rng = random.Random(int(seed))
        rng.shuffle(idxs)
    if n_inputs > 0:
        idxs = idxs[: int(n_inputs)]

    # Try to extract the solution patch robustly across variants.
    solution_keys = ["patch", "gold_patch", "resolved_patch", "solution", "diff", "fix_patch"]
    for i in idxs:
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
    n_inputs: int,
    seed: int,
    shuffle: bool,
) -> Iterator[ItemRecord]:
    """
    Load SWE-bench-style tasks and yield ItemRecords.

    Supports:
    - HuggingFace dataset hub: pass --dataset_name <org/name> and --dataset_path "".
    - Local JSON/JSONL via datasets: pass --dataset_path /path/to/file.jsonl (and dataset_name is ignored).

    Field extraction is best-effort:
    - item_id: instance_id | task_id | id
    - question_statement: problem_statement | statement | description
    - solution/patch: patch | gold_patch | resolved_patch | solution | diff | fix_patch
    """
    ds = None
    ds_path = str(dataset_path or "").strip()
    if ds_path:
        # Local dataset file(s). JSONL is treated as a "train" split.
        ds = load_dataset("json", data_files=str(ds_path), split="train")
        ds_name = f"json:{ds_path}"
        ds_split = "train"
    else:
        ds = load_dataset(str(dataset_name), split=str(split))
        ds_name = str(dataset_name)
        ds_split = str(split)

    n_total = len(ds)
    if n_total == 0:
        raise RuntimeError(f"Loaded empty dataset: {ds_name} split={ds_split}")

    idxs = list(range(n_total))
    if shuffle:
        rng = random.Random(int(seed))
        rng.shuffle(idxs)
    if n_inputs > 0:
        idxs = idxs[: int(n_inputs)]

    solution_keys = ["patch", "gold_patch", "resolved_patch", "solution", "diff", "fix_patch"]
    id_keys = ["instance_id", "task_id", "id"]
    qs_keys = ["problem_statement", "statement", "description"]

    for i in idxs:
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
        qs = ""
        for k in qs_keys:
            v = row.get(k, None)
            if v is None:
                continue
            s = str(v)
            if str(s).strip():
                qs = s
                break
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
    tokenizer = AutoTokenizer.from_pretrained(backbone, trust_remote_code=trust_remote_code)
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

    # NOTE: transformers uses `torch_dtype`, not `dtype`.
    model_kwargs = {"torch_dtype": dtype_arg}
    if device_map and device_map != "none":
        model_kwargs["device_map"] = device_map
    if attn_implementation and attn_implementation != "auto":
        model_kwargs["attn_implementation"] = attn_implementation

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


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--difficulties", type=str, default="/orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/out/question_difficulties.csv")

    p.add_argument("--dataset_name", type=str, default="princeton-nlp/SWE-bench_Verified")
    p.add_argument("--split", type=str, default="test")
    p.add_argument(
        "--dataset_path",
        type=str,
        default="",
        help="Optional local JSON/JSONL path. If set, loads via datasets('json', data_files=...) and ignores --dataset_name/--split.",
    )
    p.add_argument("--n_inputs", type=int, default=500, help="Number of dataset items to embed (0 means all).")
    p.add_argument("--shuffle", action="store_true", help="Shuffle dataset rows before taking the first --n_inputs.")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--backbone", type=str, default="Qwen/Qwen2.5-Coder-14B")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--device_map", type=str, default="auto", help="HF device_map (e.g. auto). Use 'none' to force single-device .to(device).")
    p.add_argument("--torch_dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--attn_implementation", type=str, default="auto", help="e.g. auto, flash_attention_2")
    p.add_argument(
        "--embedding_layer",
        type=int,
        default=-1,
        help="Which hidden layer to pool embeddings from (0-based over returned hidden_states; negatives allowed). -1 means last.",
    )

    p.add_argument("--instruction", type=str, default=DIFFICULTY_INSTRUCTION, help="Instruction text appended last in the embedding input.")

    p.add_argument("--out_dir", type=str, default="/orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/out/qwen25coder14b_qs_sol_instr_lr")
    p.add_argument("--embeddings_cache", type=str, default="", help="Optional path to existing embeddings cache (.npz).")
    p.add_argument("--overwrite", action="store_true")

    p.add_argument("--test_fraction", type=float, default=0.2)
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

    args = p.parse_args(argv)
    ensure_dir(args.out_dir)

    # Cache path derived from key settings.
    safe_backbone = str(args.backbone).replace("/", "__")
    ds_flag = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(args.dataset_name))[:64]
    split_flag = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(args.split))[:32]
    instr_sig = prompt_signature(str(args.instruction))
    layer_flag = "" if int(args.embedding_layer) == -1 else f"__layer{int(args.embedding_layer)}"
    idnorm_flag = "__idnorm_instance-v1"
    emb_cache = str(args.embeddings_cache or "").strip()
    if not emb_cache:
        emb_cache = os.path.join(
            args.out_dir,
            f"embeddings__{safe_backbone}__pool-lasttoken{layer_flag}__qs-sol-instr__{instr_sig}{idnorm_flag}__{ds_flag}__{split_flag}__n{int(args.n_inputs)}__maxlen{int(args.max_length)}__seed{int(args.seed)}.npz",
        )

    # Load or compute embeddings.
    if os.path.exists(emb_cache) and not args.overwrite and not str(args.embeddings_cache or "").strip():
        data = np.load(emb_cache, allow_pickle=True)
        task_ids = [str(x) for x in list(data["task_ids"].tolist())]
        X = data["X"].astype(np.float32)
        counts_kind = str(data["counts_kind"].tolist()) if "counts_kind" in data else ""
        cached_layer = int(data["embedding_layer"].tolist()) if "embedding_layer" in data else -1
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
        counts_kind = str(data["counts_kind"].tolist()) if "counts_kind" in data else ""
        cached_layer = int(data["embedding_layer"].tolist()) if "embedding_layer" in data else -1
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
                dataset_name=str(args.dataset_name),
                split=str(args.split),
                dataset_path=str(args.dataset_path),
                n_inputs=int(args.n_inputs),
                seed=int(args.seed),
                shuffle=bool(args.shuffle),
            )
        )
        src = (f"json:{args.dataset_path}" if str(args.dataset_path).strip() else f"{args.dataset_name} split={args.split}")
        print(f"Loaded dataset items: {len(items)} ({src})")

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
            dataset_name=np.array([str(args.dataset_name)], dtype=object),
            split=np.array([str(args.split)], dtype=object),
            n_inputs=np.array([int(len(ids_sorted))], dtype=np.int64),
            instruction=np.array([str(args.instruction)], dtype=object),
            instruction_signature=np.array([str(instr_sig)], dtype=object),
            backbone=np.array([str(args.backbone)], dtype=object),
            max_length=np.array([int(args.max_length)], dtype=np.int64),
            embedding_dim=np.array([int(emb_dim)], dtype=np.int64),
            embedding_layer=np.array([int(args.embedding_layer)], dtype=np.int64),
        )
        print(f"Wrote embeddings cache: {emb_cache} (n={len(ids_sorted)}, dim={X.shape[1]}, embedding_layer={int(args.embedding_layer)})")
        task_ids = ids_sorted

    diffs = load_difficulties_csv(str(args.difficulties))
    difficulty_order = load_difficulties_ordered(str(args.difficulties))

    # Align X with y by item_id / instance_id
    id_to_row = {tid: i for i, tid in enumerate(task_ids)}
    common = [tid for tid in task_ids if tid in diffs]
    missing_diff = [tid for tid in task_ids if tid not in diffs]
    if missing_diff:
        print(f"WARNING: {len(missing_diff)} item_ids missing difficulty; ignoring (e.g. {missing_diff[:3]})")
    if not common:
        raise RuntimeError("No overlap between embedded ids and difficulty CSV item_id values.")

    Xy = np.stack([X[id_to_row[tid]] for tid in common], axis=0).astype(np.float32)
    y = np.array([diffs[tid] for tid in common], dtype=np.float32)

    # Deterministic split
    train_idx, test_idx = stable_split_ids(common, test_fraction=float(args.test_fraction), seed=int(args.seed))
    X_train, y_train = Xy[train_idx], y[train_idx]
    X_test, y_test = Xy[test_idx], y[test_idx]

    regressor_name = str(args.regressor)
    alphas: np.ndarray = np.array([], dtype=np.float64)
    if regressor_name == "linear":
        model = LinearRegression()
        model.fit(X_train, y_train)
    elif regressor_name == "ridge":
        alpha = float(args.ridge_alpha)
        if not (alpha > 0):
            raise ValueError("--ridge_alpha must be > 0")
        model = Pipeline(steps=[("scaler", StandardScaler(with_mean=True, with_std=True)), ("ridge", Ridge(alpha=alpha))])
        model.fit(X_train, y_train)
    elif regressor_name == "ridge_cv":
        try:
            alphas = np.array([float(x.strip()) for x in str(args.ridge_alphas).split(",") if x.strip()], dtype=np.float64)
        except Exception as e:
            raise ValueError(f"Failed to parse --ridge_alphas={args.ridge_alphas!r}: {e}") from e
        if alphas.size == 0:
            raise ValueError("Expected at least one alpha in --ridge_alphas")
        cv = KFold(n_splits=int(args.cv_folds), shuffle=True, random_state=int(args.seed))
        model = Pipeline(
            steps=[("scaler", StandardScaler(with_mean=True, with_std=True)), ("ridge", RidgeCV(alphas=alphas, cv=cv))]
        )
        model.fit(X_train, y_train)
    else:
        raise AssertionError(f"Unhandled regressor: {regressor_name}")

    yhat_train = model.predict(X_train).astype(np.float64)
    yhat_test = model.predict(X_test).astype(np.float64)
    yhat_all = model.predict(Xy).astype(np.float64)

    ridge_alpha = None
    if regressor_name in ("ridge", "ridge_cv"):
        try:
            ridge_alpha = float(model.named_steps["ridge"].alpha_)
        except Exception:
            ridge_alpha = None

    metrics = {
        "n_items_total": int(len(task_ids)),
        "n_items_with_difficulty": int(len(common)),
        "embedding_dim": int(Xy.shape[1]),
        "train_fraction": float(1.0 - args.test_fraction),
        "test_fraction": float(args.test_fraction),
        "seed": int(args.seed),
        "train_r2": float(r2_score(y_train, yhat_train)),
        "test_r2": float(r2_score(y_test, yhat_test)),
        "train_rmse": float(_rmse(y_train, yhat_train)),
        "test_rmse": float(_rmse(y_test, yhat_test)),
        "train_pearson": float(_pearsonr(y_train, yhat_train)),
        "test_pearson": float(_pearsonr(y_test, yhat_test)),
        "regressor": regressor_name,
        "ridge_alpha": ridge_alpha,
        "ridge_alphas_searched": [float(x) for x in np.asarray(alphas).tolist()],
        "cv_folds": int(args.cv_folds) if regressor_name == "ridge_cv" else None,
        "backbone": str(args.backbone),
        "pooling": "last_token_of_hidden_state",
        "embedding_layer": int(args.embedding_layer),
        "max_length": int(args.max_length),
        "dataset_name": str(args.dataset_name),
        "split": str(args.split),
        "n_inputs_requested": int(args.n_inputs),
        "shuffle": bool(args.shuffle),
        "instruction": str(args.instruction),
        "instruction_signature": instr_sig,
        "batch_size": int(args.batch_size),
        "device_map": str(args.device_map),
        "torch_dtype": str(args.torch_dtype),
        "attn_implementation": str(args.attn_implementation),
        "embeddings_cache": emb_cache,
        "difficulties_csv": str(args.difficulties),
    }
    save_json(os.path.join(args.out_dir, "metrics.json"), metrics)

    # Write per-item predictions
    split_set = set(test_idx)
    pred_path = os.path.join(args.out_dir, "predictions.csv")
    with open(pred_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item_id", "diff_true", "diff_pred", "split"])
        w.writeheader()
        for i, tid in enumerate(common):
            w.writerow(
                {
                    "item_id": tid,
                    "diff_true": float(y[i]),
                    "diff_pred": float(yhat_all[i]),
                    "split": "test" if i in split_set else "train",
                }
            )

    # Also write a convenience file matching the IRT schema (item_ix,item_id,diff),
    # but with diff replaced by the regression-predicted difficulty.
    pred_by_id = {tid: float(yhat_all[i]) for i, tid in enumerate(common)}
    pred_irt_path = os.path.join(args.out_dir, "predicted_question_difficulties.csv")
    with open(pred_irt_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item_ix", "item_id", "diff"])
        w.writeheader()
        missing = 0
        for item_ix, item_id in difficulty_order:
            if item_id not in pred_by_id:
                missing += 1
                continue
            w.writerow({"item_ix": int(item_ix), "item_id": item_id, "diff": float(pred_by_id[item_id])})
    if missing:
        print(f"WARNING: {missing} item_ids in difficulty CSV were missing predictions; wrote remaining rows.")

    print(f"Wrote metrics: {os.path.join(args.out_dir, 'metrics.json')}")
    print(f"Wrote predictions: {pred_path}")
    print(f"Wrote predicted difficulties: {pred_irt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



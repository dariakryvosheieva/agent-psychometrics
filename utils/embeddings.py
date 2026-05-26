"""Generate task embedding caches."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - embedding generation will fail loudly if needed.
    torch = None  # type: ignore[assignment]
try:
    from datasets import load_dataset
except Exception:  # pragma: no cover
    load_dataset = None  # type: ignore[assignment]
try:
    from huggingface_hub import hf_hub_download
except Exception:  # pragma: no cover
    hf_hub_download = None  # type: ignore[assignment]
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]
try:
    from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedTokenizerFast
except Exception:  # pragma: no cover
    AutoConfig = AutoModel = AutoTokenizer = PreTrainedTokenizerFast = None  # type: ignore[assignment]


DIFFICULTY_INSTRUCTION = (
    "How difficult is the above task for a coding agent? Please output one "
    "floating-point number from 0 (very easy) to 1 (very hard). Your difficulty score:\n"
)

EMBEDDING_TEXT_FORMAT = "qs_solution_instruction_v1"

_V_SUFFIX_RE = re.compile(r"-v(?:\d+|[0-9a-f]{6,}|nan)$", re.IGNORECASE)


def _require_embedding_dependency(name: str, obj: Any) -> None:
    if obj is None:
        raise RuntimeError(
            f"Missing dependency '{name}'. Install the embedding-generation dependencies "
            "before generating embedding caches."
        )


def seed_everything(seed: int, *, deterministic: bool) -> None:
    _require_embedding_dependency("torch", torch)
    s = int(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(s))
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

    try:
        from transformers import set_seed as hf_set_seed

        hf_set_seed(s)
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
    _require_embedding_dependency("torch", torch)
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
        torch.backends.cudnn.benchmark = not on
    except Exception:
        pass


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_swebench_item_id(raw_item_id: str) -> str:
    s = str(raw_item_id or "").strip()
    if s.startswith("instance_"):
        s = s[len("instance_") :]
    s = _V_SUFFIX_RE.sub("", s)
    return s.strip()


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


_GSO_PROMPT_TEMPLATE = """I've uploaded a python code repository in the directory workspace_dir_name. Consider the
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


def iter_swebench_items(
    *,
    dataset_name: str,
    split: str,
    dataset_path: str,
) -> Iterator[ItemRecord]:
    dataset_name = str(dataset_name or "").strip()
    dataset_path = str(dataset_path or "").strip()
    if bool(dataset_name) and bool(dataset_path):
        raise ValueError("Provide only one of dataset_name or dataset_path.")
    if not dataset_name and not dataset_path:
        raise ValueError("No dataset provided (set dataset_name or dataset_path).")
    _require_embedding_dependency("datasets", load_dataset)

    is_gso = _is_gso_dataset(dataset_name=dataset_name, dataset_path=dataset_path)
    if dataset_path:
        source_name = f"json:{dataset_path}"
        dataset = load_dataset("json", data_files=str(dataset_path), split="train")
        source_split = "train"
    else:
        source_name = str(dataset_name)
        dataset = load_dataset(str(dataset_name), split=str(split))
        source_split = str(split)

    n_total = int(len(dataset))
    if n_total == 0:
        raise RuntimeError(f"Loaded empty dataset: {source_name} split={source_split}")

    solution_keys = ["patch", "gold_patch", "resolved_patch", "solution", "diff", "fix_patch"]
    if is_gso:
        solution_keys = ["gt_diff"] + solution_keys
    id_keys = ["instance_id", "task_id", "id"]
    qs_keys = ["problem_statement", "statement", "description"]
    if is_gso:
        qs_keys = ["prob_script"] + qs_keys

    for i in range(n_total):
        row = dataset[int(i)]
        item_id = ""
        for key in id_keys:
            value = row.get(key, None)
            if value is None:
                continue
            candidate = str(value).strip()
            if candidate:
                item_id = normalize_swebench_item_id(candidate)
                break

        question_statement = ""
        question_key_used = ""
        for key in qs_keys:
            value = row.get(key, None)
            if value is None:
                continue
            candidate = str(value)
            if candidate.strip():
                question_statement = candidate
                question_key_used = str(key)
                break
        if is_gso and question_key_used == "prob_script":
            question_statement = _wrap_gso_problem_statement(question_statement)

        solution = ""
        for key in solution_keys:
            value = row.get(key, None)
            if value is None:
                continue
            candidate = str(value)
            if candidate.strip():
                solution = candidate
                break

        if not item_id:
            item_id = f"row_{int(i)}"
        yield ItemRecord(item_id=item_id, question_statement=question_statement, solution=solution)


def _load_tokenizer(backbone: str, *, trust_remote_code: bool) -> Any:
    _require_embedding_dependency("transformers", AutoTokenizer)
    try:
        return AutoTokenizer.from_pretrained(backbone, trust_remote_code=trust_remote_code)
    except ValueError as exc:
        if "TokenizersBackend" not in str(exc):
            raise

        _require_embedding_dependency("huggingface_hub", hf_hub_download)
        tok_json = hf_hub_download(repo_id=backbone, filename="tokenizer.json")
        tok_cfg_path = None
        try:
            tok_cfg_path = hf_hub_download(repo_id=backbone, filename="tokenizer_config.json")
        except Exception:
            pass

        tok_kwargs: Dict[str, Any] = {"tokenizer_file": tok_json}
        extra_special_tokens: Optional[List[str]] = None
        if tok_cfg_path is not None and os.path.exists(tok_cfg_path):
            with open(tok_cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            for key in ("bos_token", "eos_token", "unk_token", "pad_token"):
                if isinstance(cfg.get(key), str) and cfg.get(key):
                    tok_kwargs[key] = cfg[key]
            if isinstance(cfg.get("model_max_length"), int):
                tok_kwargs["model_max_length"] = cfg["model_max_length"]
            if isinstance(cfg.get("extra_special_tokens"), list):
                extra_special_tokens = [str(x) for x in cfg["extra_special_tokens"]]

        tokenizer = PreTrainedTokenizerFast(**tok_kwargs)
        if extra_special_tokens:
            tokenizer.additional_special_tokens = extra_special_tokens
        return tokenizer


def _try_load_model_class(backbone: str, *, trust_remote_code: bool, model_kwargs: dict) -> torch.nn.Module:
    _require_embedding_dependency("torch", torch)
    _require_embedding_dependency("transformers", AutoModel)
    try:
        import transformers.activations as activations

        if not hasattr(activations, "PytorchGELUTanh") and hasattr(activations, "GELUTanh"):
            activations.PytorchGELUTanh = activations.GELUTanh
    except Exception:
        pass

    errors = []
    for class_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
        try:
            module = __import__("transformers", fromlist=[class_name])
            cls = getattr(module, class_name)
            return cls.from_pretrained(backbone, trust_remote_code=trust_remote_code, **model_kwargs)
        except Exception as exc:
            errors.append((class_name, exc))

    try:
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(backbone, trust_remote_code=trust_remote_code, **model_kwargs)
    except Exception as exc:
        errors.append(("AutoModelForCausalLM", exc))

    try:
        return AutoModel.from_pretrained(backbone, trust_remote_code=trust_remote_code, **model_kwargs)
    except Exception as exc:
        errors.append(("AutoModel", exc))

    message = "Failed to load model with any supported auto class:\n" + "\n".join(
        f"- {name}: {type(err).__name__}: {err}" for name, err in errors
    )
    raise RuntimeError(message)


def _select_text_submodel(model: torch.nn.Module) -> torch.nn.Module:
    for attr in ("language_model", "text_model"):
        submodel = getattr(model, attr, None)
        if isinstance(submodel, torch.nn.Module):
            return submodel

    submodel = getattr(model, "model", None)
    if isinstance(submodel, torch.nn.Module) and hasattr(submodel, "get_input_embeddings"):
        return submodel
    return model


def _get_hidden_states_tuple(outputs: Any) -> Any:
    for attr in ("hidden_states", "encoder_hidden_states", "decoder_hidden_states"):
        if hasattr(outputs, attr):
            hidden_states = getattr(outputs, attr)
            if hidden_states is not None:
                return hidden_states
    return None


def _extract_hidden_state(outputs: Any, *, embedding_layer: int) -> torch.Tensor:
    layer = int(embedding_layer)

    if layer == -1 and hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state
    if layer == -1 and hasattr(outputs, "encoder_last_hidden_state") and outputs.encoder_last_hidden_state is not None:
        return outputs.encoder_last_hidden_state

    hidden_states = _get_hidden_states_tuple(outputs)
    if hidden_states is not None:
        try:
            return hidden_states[layer]
        except Exception as exc:
            raise RuntimeError(
                f"Requested embedding_layer={layer}, but model returned {len(hidden_states)} hidden_states entries. "
                f"Try a value in [-{len(hidden_states)}, {len(hidden_states) - 1}] or use --embedding_layer -1."
            ) from exc

    raise RuntimeError(
        "Model outputs did not expose hidden states. Try --embedding_layer -1 and ensure the model supports "
        "output_hidden_states=True."
    )


def last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    lengths = attention_mask.sum(dim=1).clamp(min=1)
    idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, last_hidden_state.size(-1))
    return last_hidden_state.gather(dim=1, index=idx).squeeze(1)


def _torch_dtype_arg(torch_dtype: str) -> Any:
    _require_embedding_dependency("torch", torch)
    if torch_dtype == "auto":
        return "auto"
    if torch_dtype in ("float16", "fp16"):
        return torch.float16
    if torch_dtype in ("bfloat16", "bf16"):
        return torch.bfloat16
    if torch_dtype in ("float32", "fp32"):
        return torch.float32
    raise ValueError(f"Unknown torch_dtype: {torch_dtype}")


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
    _require_embedding_dependency("torch", torch)
    _require_embedding_dependency("transformers", AutoModel)
    _require_embedding_dependency("tqdm", tqdm)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = _load_tokenizer(backbone, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"

    dtype_arg = _torch_dtype_arg(torch_dtype)
    fp_params = inspect.signature(AutoModel.from_pretrained).parameters
    model_kwargs: Dict[str, Any] = {"dtype" if "dtype" in fp_params else "torch_dtype": dtype_arg}
    if device_map and device_map != "none":
        model_kwargs["device_map"] = device_map
    if attn_implementation and attn_implementation != "auto":
        model_kwargs["attn_implementation"] = attn_implementation

    try:
        cfg = AutoConfig.from_pretrained(backbone, trust_remote_code=trust_remote_code)
        quantization_config = getattr(cfg, "quantization_config", None)
        if isinstance(quantization_config, dict) and str(quantization_config.get("quant_method", "")).lower() == "fp8":
            try:
                from transformers import FineGrainedFP8Config

                model_kwargs["quantization_config"] = FineGrainedFP8Config(dequantize=True)
            except Exception:
                pass
    except Exception:
        pass

    model = _try_load_model_class(backbone, trust_remote_code=trust_remote_code, model_kwargs=model_kwargs)
    model.eval()
    if device_map in ("", "none", None):
        model.to(device)

    text_model = _select_text_submodel(model)
    for module in (model, text_model):
        cfg = getattr(module, "config", None)
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
        nonlocal batch_ids, batch_texts, embedding_dim
        if not batch_texts:
            return

        pairs = [(rid, txt) for rid, txt in zip(batch_ids, batch_texts) if str(txt).strip()]
        batch_ids = []
        batch_texts = []
        if not pairs:
            return

        ids = [rid for rid, _ in pairs]
        texts = [txt for _, txt in pairs]
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        if int(input_ids.shape[1]) == 0:
            return

        input_ids = input_ids.to(embed_device)
        attention_mask = attention_mask.to(embed_device)
        want_hidden_states = int(embedding_layer) != -1

        with torch.inference_mode():
            fwd_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "return_dict": True,
                "output_hidden_states": bool(want_hidden_states),
            }
            try:
                sig = inspect.signature(text_model.forward)
                if "use_cache" in sig.parameters:
                    fwd_kwargs["use_cache"] = False
            except Exception:
                fwd_kwargs["use_cache"] = False

            outputs = text_model(**fwd_kwargs)
            try:
                hidden = _extract_hidden_state(outputs, embedding_layer=int(embedding_layer))
            except RuntimeError:
                if want_hidden_states:
                    raise
                fwd_kwargs["output_hidden_states"] = True
                outputs = text_model(**fwd_kwargs)
                hidden = _extract_hidden_state(outputs, embedding_layer=int(embedding_layer))

            pooled = last_token_pool(hidden, attention_mask).detach().float().cpu().numpy()

        embedding_dim = int(pooled.shape[1])
        for item_id, vec, text in zip(ids, pooled, texts):
            per_id[str(item_id)] = vec.astype(np.float32, copy=False)
            counts[str(item_id)] = int(len(str(text)))

    for rec in tqdm(items, desc="embed_items"):
        text = format_qs_solution_instruction(
            question_statement=rec.question_statement,
            solution=rec.solution,
            instruction=instruction,
        )
        if not text.strip():
            continue
        batch_ids.append(rec.item_id)
        batch_texts.append(text)
        if len(batch_texts) >= int(batch_size):
            flush()
    flush()

    ids_sorted = sorted(per_id.keys())
    return ids_sorted, per_id, counts, int(embedding_dim)


def _canon_benchmark_name(name: str) -> str:
    s = str(name or "").strip().lower().replace("-", "_")
    if s == "terminal_bench":
        s = "terminalbench"
    if s not in {"verified", "swebench_verified", "pro", "swebench_pro", "terminalbench", "gso"}:
        raise ValueError(
            f"Unknown benchmark name: {name!r}. Allowed: verified, swebench_verified, pro, "
            "swebench_pro, terminalbench, gso."
        )
    if s == "verified":
        return "swebench_verified"
    if s == "pro":
        return "swebench_pro"
    return s


def get_benchmark_dataset_defaults(benchmark: str) -> Dict[str, str]:
    b = _canon_benchmark_name(benchmark)
    repo_root = str(Path(__file__).resolve().parents[1])
    defaults: Dict[str, Dict[str, str]] = {
        "swebench_verified": {
            "dataset_name": "princeton-nlp/SWE-bench_Verified",
            "dataset_path": "",
            "split": "test",
        },
        "swebench_pro": {
            "dataset_name": "ScaleAI/SWE-bench_Pro",
            "dataset_path": "",
            "split": "test",
        },
        "terminalbench": {
            "dataset_name": "",
            "dataset_path": os.path.join(repo_root, "data/terminalbench/tasks.jsonl"),
            "split": "train",
        },
        "gso": {
            "dataset_name": "gso-bench/gso",
            "dataset_path": "",
            "split": "test",
        },
    }
    return defaults[b]


def _shared_embeddings_dir() -> str:
    repo_root = str(Path(__file__).resolve().parents[1])
    return os.path.join(repo_root, "embeddings")


def default_embeddings_cache_path(
    *,
    backbone: str,
    max_length: int,
    batch_size: int,
    device_map: str,
    torch_dtype: str,
    attn_implementation: str,
    instruction: str,
    embedding_layer: int,
    dataset_sources: str,
    split: str,
) -> str:
    safe_backbone = str(backbone).replace("/", "__")
    idnorm_flag = "__idnorm_instance-v1"
    instr_sig = prompt_signature(str(instruction))
    cache_meta = {
        "backbone": str(backbone),
        "max_length": int(max_length),
        "batch_size": int(batch_size),
        "device_map": str(device_map),
        "torch_dtype": str(torch_dtype),
        "attn_implementation": str(attn_implementation),
        "instruction": str(instruction),
        "instruction_sig": str(instr_sig),
        "embedding_layer": int(embedding_layer),
        "normalize_item_ids": True,
        "idnorm_flag": idnorm_flag,
        "dataset_sources": str(dataset_sources),
        "split": str(split),
    }
    cache_key = hashlib.sha1(json.dumps(cache_meta, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    model_short = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(safe_backbone))[:48].strip("_") or "model"
    return os.path.join(_shared_embeddings_dir(), f"embeddings__{model_short}__{cache_key}__maxlen{int(max_length)}.npz")


def save_embeddings_cache(
    *,
    path: str,
    task_ids: Sequence[str],
    embeddings_by_id: Dict[str, np.ndarray],
    counts_by_id: Dict[str, int],
    embedding_dim: int,
    dataset_sources: str,
    split: str,
    dataset_path: str,
    instruction: str,
    backbone: str,
    max_length: int,
    embedding_layer: int,
) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    ids_sorted = list(task_ids)
    X = np.stack([embeddings_by_id[item_id] for item_id in ids_sorted], axis=0).astype(np.float32)
    counts_arr = np.array([int(counts_by_id.get(item_id, 0)) for item_id in ids_sorted], dtype=np.int64)
    np.savez_compressed(
        path,
        task_ids=np.array(ids_sorted, dtype=object),
        X=X,
        counts_kind=np.array(["text_len_chars"], dtype=object),
        counts=counts_arr,
        dataset_name=np.array([str(dataset_sources)], dtype=object),
        split=np.array([str(split)], dtype=object),
        dataset_path=np.array([str(dataset_path)], dtype=object),
        n_items=np.array([int(len(ids_sorted))], dtype=np.int64),
        instruction=np.array([str(instruction)], dtype=object),
        instruction_signature=np.array([str(prompt_signature(instruction))], dtype=object),
        text_format=np.array([str(EMBEDDING_TEXT_FORMAT)], dtype=object),
        includes_solution=np.array([True], dtype=np.bool_),
        backbone=np.array([str(backbone)], dtype=object),
        max_length=np.array([int(max_length)], dtype=np.int64),
        embedding_dim=np.array([int(embedding_dim)], dtype=np.int64),
        embedding_layer=np.array([int(embedding_layer)], dtype=np.int64),
    )


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
    if isinstance(v, np.ndarray) and v.size == 1:
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
    repo_root = str(Path(__file__).resolve().parents[1])
    roots = [
        str(out_dir),
        os.path.join(str(out_dir), "embeddings"),
        os.path.join(repo_root, "embeddings"),
        os.path.join(repo_root, "data"),
    ]
    seen: Set[str] = set()
    out: List[str] = []
    for root in roots:
        ap = os.path.abspath(str(root))
        if ap in seen or not os.path.isdir(ap):
            continue
        seen.add(ap)
        out.append(ap)
    return out


def _iter_embedding_npz_candidates(search_roots: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for root in search_roots:
        rp = os.path.abspath(str(root))
        if not os.path.isdir(rp):
            continue
        for rel_pattern in ("*.npz", "embeddings/*.npz", "*/*.npz"):
            for path in Path(rp).glob(rel_pattern):
                ap = str(path.resolve())
                if ap in seen:
                    continue
                seen.add(ap)
                out.append(ap)
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
    if not p or not os.path.exists(p):
        return None
    try:
        with np.load(p, allow_pickle=True) as data:
            if "task_ids" not in data or "X" not in data:
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
                    cached_instr_sig == req_instr_sig
                    or (cached_instr_sig.startswith("qs_sol_") and req_instr_sig.startswith("qs_sol_"))
                )

            cached_backbone = _meta_str(data.get("backbone", None), "")
            if cached_backbone and cached_backbone != str(backbone):
                return None

            cached_dataset_source = _meta_str(data.get("dataset_name", None), "")
            if bool(require_single_dataset_source) and "|" in cached_dataset_source:
                return None

            includes_solution = (
                _to_boolish(_npz_scalar(data.get("includes_solution", None), None), default=False)
                if "includes_solution" in data
                else False
            )
            text_format = _meta_str(data.get("text_format", None), "")
            cache_prompt_template_ok = (
                includes_solution or text_format == EMBEDDING_TEXT_FORMAT or cached_instr_sig.startswith("qs_sol_")
            )
            if not (cache_prompt_template_ok and (same_prompt_template_family or not req_instr_sig)):
                return None

            if required_task_ids:
                id_set = set(task_ids)
                for tid in required_task_ids:
                    if str(tid) not in id_set:
                        return None
            if expected_n_items is not None:
                n_cached = int(len(task_ids))
                n_expected = int(expected_n_items)
                pro_off_by_one_ok = n_expected == 730 and n_cached == 731
                if n_cached != n_expected and not pro_off_by_one_ok:
                    return None

            cached_layer = int(_npz_scalar(data.get("embedding_layer", None), -1)) if "embedding_layer" in data else -1
            cached_maxlen = (
                int(_npz_scalar(data.get("max_length", None), int(max_length)))
                if "max_length" in data
                else int(max_length)
            )
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
    for path in preferred_paths:
        ap = os.path.abspath(str(path))
        if ap not in seen:
            seen.add(ap)
            candidates.append(ap)
    for path in _iter_embedding_npz_candidates(search_roots):
        ap = os.path.abspath(str(path))
        if ap not in seen:
            seen.add(ap)
            candidates.append(ap)

    for path in candidates:
        loaded = load_compatible_embeddings_cache(
            path,
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
        return str(path), task_ids, X, meta
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Experiment New Tasks embedding cache (.npz).")
    parser.add_argument("--benchmark", type=str, default="", help="One of: verified, pro, terminalbench, gso.")
    parser.add_argument("--dataset_name", type=str, default="princeton-nlp/SWE-bench_Verified")
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backbone", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--attn_implementation", type=str, default="auto")
    parser.add_argument("--embedding_layer", type=int, default=-1)
    parser.add_argument("--instruction", type=str, default=DIFFICULTY_INSTRUCTION)
    parser.add_argument("--out", type=str, default="", help="Output .npz path. Defaults to embeddings/<cache-key>.npz.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if args.benchmark:
        defaults = get_benchmark_dataset_defaults(args.benchmark)
        args.dataset_name = defaults["dataset_name"]
        args.dataset_path = defaults["dataset_path"]
        args.split = defaults["split"]

    dataset_name = str(args.dataset_name).strip()
    dataset_path = str(args.dataset_path).strip()
    if dataset_path:
        dataset_sources = f"json:{os.path.basename(dataset_path) or 'dataset.jsonl'}"
    else:
        dataset_sources = dataset_name or "princeton-nlp/SWE-bench_Verified"

    out_path = str(args.out or "").strip()
    if not out_path:
        out_path = default_embeddings_cache_path(
            backbone=str(args.backbone),
            max_length=int(args.max_length),
            batch_size=int(args.batch_size),
            device_map=str(args.device_map),
            torch_dtype=str(args.torch_dtype),
            attn_implementation=str(args.attn_implementation),
            instruction=str(args.instruction),
            embedding_layer=int(args.embedding_layer),
            dataset_sources=str(dataset_sources),
            split=str(args.split),
        )

    if os.path.exists(out_path) and not bool(args.overwrite):
        raise FileExistsError(f"Embeddings cache already exists: {out_path}. Pass --overwrite to replace it.")

    seed_everything(int(args.seed), deterministic=True)
    items = list(iter_swebench_items(dataset_name=dataset_name, split=str(args.split), dataset_path=dataset_path))
    print(f"Loaded dataset items: {len(items)} (sources={dataset_sources}, split={args.split})")
    ids_sorted, embeddings_by_id, counts_by_id, embedding_dim = embed_items(
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
        raise RuntimeError("No embeddings were produced.")

    save_embeddings_cache(
        path=out_path,
        task_ids=ids_sorted,
        embeddings_by_id=embeddings_by_id,
        counts_by_id=counts_by_id,
        embedding_dim=int(embedding_dim),
        dataset_sources=str(dataset_sources),
        split=str(args.split),
        dataset_path=str(dataset_path),
        instruction=str(args.instruction),
        backbone=str(args.backbone),
        max_length=int(args.max_length),
        embedding_layer=int(args.embedding_layer),
    )
    print(f"Wrote embeddings cache: {out_path} (n={len(ids_sorted)}, dim={embedding_dim})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
GSO_ASSUMED_SCAFFOLD: str = "OpenHands"
SWEBENCH_PRO_ASSUMED_SCAFFOLD: str = "SWE-agent 1.0"


def assumed_scaffold_for_benchmark(benchmark: str) -> Optional[str]:
    b = str(benchmark or "").strip().lower().replace("-", "_")
    if b == "gso":
        return GSO_ASSUMED_SCAFFOLD
    if b == "pro":
        return SWEBENCH_PRO_ASSUMED_SCAFFOLD
    return None


_AT_SUFFIX_RE = re.compile(r"_at_.*$", flags=re.IGNORECASE)
_TRAILING_DATE_RE = re.compile(r"[-_]\d{8}$")


def _strip_at_suffix(agent: str) -> str:
    raw = (agent or "").strip()
    if "_at_" not in raw.lower():
        return raw
    base = re.sub(_AT_SUFFIX_RE, "", raw).strip()
    base = re.sub(_TRAILING_DATE_RE, "", base).strip()
    return base


def _model_leaf(token: str) -> str:
    t = (token or "").strip()
    if not t:
        return t
    if "/" in t:
        t = t.split("/")[-1]
    return t


def resolve_path(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    return p if p.is_absolute() else (ROOT / p)


def _is_model_token(token: str) -> bool:
    t = token.strip()
    if not t:
        return False
    tl = _model_leaf(t).lower()
    known_exact = {
        "gpt5",
        "gpt4o",
        "gpt4",
        "gpt35",
        "claude2",
        "claude3opus",
        "claude3.5sonnet",
        "kimi_k2",
        "devstral_small",
    }
    if tl in known_exact:
        return True
    prefixes = (
        "gpt",
        "claude-",
        "claude_",
        "claude",
        "gemini",
        "kimi",
        "qwen",
        "deepseek",
        "llama",
        "swellama",
        "lingma",
        "glm",
        "devstral",
        "grok",
        "minimax",
        "o1",
        "o3",
        "o4",
    )
    if tl.startswith(prefixes):
        return True
    has_digit = any(ch.isdigit() for ch in tl)
    if re.fullmatch(r"\d+(?:\.\d+)?b", tl):
        return False
    if re.fullmatch(r"lm[_-]\d+(?:\.\d+)?b", tl):
        return False
    if has_digit and (tl.endswith("b") or "b-" in tl or "-b" in tl):
        return True
    return False


_SCAFFOLD_ALIASES: dict[str, str] = {
    "lingxi": "Lingxi v1.5",
    "lingxi v1.5": "Lingxi v1.5",
    "lingxi-v1.5": "Lingxi v1.5",
    "lingxi_v1.5": "Lingxi v1.5",
    "jules": "google_jules",
    "google-jules": "google_jules",
    "google_jules": "google_jules",
    "openhands": "OpenHands",
    "tools": "Tools",
    "sweagent": "SWE-agent",
    "harness": "Harness",
    "rag": "RAG",
    "none": "NoScaffold",
    "null": "NoScaffold",
    "nan": "NoScaffold",
    "na": "NoScaffold",
    "n/a": "NoScaffold",
}


def _canonical_scaffold(s: str) -> str:
    sl = s.strip()
    if not sl:
        return "NoScaffold"
    low = sl.lower()
    return _SCAFFOLD_ALIASES.get(low, sl)


_PRO_TRAILING_SUFFIX_RE = re.compile(r"\s+(?:--|-)\s+.+$")


def _normalize_model_key(name: str) -> str:
    if not name:
        return ""
    return re.sub(r"[\s_]+", "-", name.lower().strip())


_MODEL_EXACT: dict[str, str] = {
    "gpt5": "GPT-5",
    "gpt-5": "GPT-5",
    "gpt-5-codex": "GPT-5-Codex",
    "gpt-5-codex-mini": "GPT-5-Codex Mini",
    "gpt-5-codex-max": "GPT-5-Codex Max",
    "gpt5-codex": "GPT-5-Codex",
    "gpt4o": "GPT-4o",
    "gpt-4o": "GPT-4o",
    "gpt4o-mini": "GPT-4o-mini",
    "gpt-4o-mini": "GPT-4o-mini",
    "gpt4": "GPT-4",
    "gpt-4": "GPT-4",
    "gpt35": "GPT-3.5",
    "gpt-3.5": "GPT-3.5",
    "gpt-oss": "GPT OSS 120B",
    "gptoss": "GPT OSS 120B",
    "kimi": "Kimi K2 Instruct",
    "claude-sonnet-4": "Claude Sonnet 4",
    "claude-4.5-sonnet": "Claude Sonnet 4.5",
    "claude-4.5-connet": "Claude Sonnet 4.5",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gemini-2.5-pro-preview": "Gemini 2.5 Pro",
    "claude2": "Claude 2",
    "claude-2": "Claude 2",
    "claude3opus": "Claude 3 Opus",
    "claude-3-opus": "Claude 3 Opus",
    "claude-4.5-opus": "Claude Opus 4.5",
    "kimi-k2": "Kimi K2 Instruct",
    "kimi-k2-instruct": "Kimi K2 Instruct",
    "qwen-3-coder-480b": "Qwen3-Coder-480B-A35B-Instruct",
    "qwen3-coder-480b": "Qwen3-Coder-480B-A35B-Instruct",
    "glm-4.5": "GLM 4.5",
    "glm-4-5": "GLM 4.5",
    "glm4-5": "GLM 4.5",
    "glm4.5": "GLM 4.5",
    "glm-4.6": "GLM 4.6",
    "glm4-6": "GLM 4.6",
    "glm4.6": "GLM 4.6",
    "glm-4-6": "GLM 4.6",
    "gemini-3-pro": "Gemini 3 Pro",
    "gemini-3-pro-preview": "Gemini 3 Pro",
    "gemini-3-flash": "Gemini 3 Flash",
    "gemini-3-flash-preview": "Gemini 3 Flash",
    "gemini-2-5-pro": "Gemini 2.5 Pro",
    "qwen3-coder-480b-a35b-instruct-fp8": "Qwen3-Coder-480B-A35B-Instruct-FP8",
    "qwen3-coder-480b-a35b-instruct": "Qwen3-Coder-480B-A35B-Instruct",
    "qwen3-coder-30b-a3b-instruct": "Qwen3-Coder-30B-A3B-Instruct",
    "qwen3-coder-30b-a3b-instruct-fp8": "Qwen3-Coder-30B-A3B-Instruct",
    "qwen3-coder": "Qwen3-Coder-480B-A35B-Instruct-FP8",
    "qwencoder30ba3b": "Qwen3-Coder-30B-A3B-Instruct",
    "claude-sonnet-4-5": "Claude Sonnet 4.5",
    "claude-4-5-sonnet": "Claude Sonnet 4.5",
    "claude-haiku-4-5": "Claude Haiku 4.5",
    "claude-4-opus": "Claude Opus 4",
    "claude-4-0-opus": "Claude Opus 4",
    "claude-opus-4": "Claude Opus 4",
    "claude-opus-4.5": "Claude Opus 4.5",
    "claude-4-5-opus": "Claude Opus 4.5",
    "claude-opus-4-1": "Claude Opus 4.1",
    "claude-opus-4.1": "Claude Opus 4.1",
    "claude-sonnet-4.5": "Claude Sonnet 4.5",
    "claude4sonnet": "Claude Sonnet 4",
    "claude3.5sonnet": "Claude 3.5 Sonnet",
    "claude35haiku": "Claude 3.5 Haiku",
    "claude3.5haiku": "Claude 3.5 Haiku",
    "claude37sonnet": "Claude 3.7 Sonnet",
    "gpt-oss-20b": "GPT OSS 20B",
    "gptoss-20b": "GPT OSS 20B",
    "gpt-oss-120b": "GPT OSS 120B",
    "gptoss-120b": "GPT OSS 120B",
    "glm-4p6": "GLM 4.6",
    "amazon.nova-premier-v1.0": "Amazon Nova Premier 1.0",
    "amazon-nova-premier-v1.0": "Amazon Nova Premier 1.0",
    "frogboss-32b": "FrogBoss-32B",
    "frogmini-14b": "FrogMini-14B",
}

_MODEL_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^gpt-oss$"), "GPT OSS"),
    (re.compile(r"^gptoss$"), "GPT OSS"),
    (re.compile(r"^claude-opus-4-5(?:[-_]?\d{8})?$"), "Claude Opus 4.5"),
    (re.compile(r"^claude-3-5-sonnet(?:-\d{8})?(?:-updated)?$"), "Claude 3.5 Sonnet"),
    (re.compile(r"^claude-3\.5-sonnet(?:-\d{8})?(?:-updated)?$"), "Claude 3.5 Sonnet"),
    (re.compile(r"^claude-3-5-haiku(?:-\d{8})?(?:-updated)?$"), "Claude 3.5 Haiku"),
    (re.compile(r"^claude-3\.5-haiku(?:-\d{8})?(?:-updated)?$"), "Claude 3.5 Haiku"),
    (re.compile(r"^claude-3-7-sonnet(?:-\d{8})?(?:-updated)?$"), "Claude 3.7 Sonnet"),
    (re.compile(r"^claude-3\.7-sonnet(?:-\d{8})?(?:-updated)?$"), "Claude 3.7 Sonnet"),
    (re.compile(r"^claude-4-sonnet(?:-\d{8})?(?:-updated)?$"), "Claude Sonnet 4"),
    (re.compile(r"^gemini-2(?:-|\.|\s)?5-pro$"), "Gemini 2.5 Pro"),
    (re.compile(r"^qwen3-coder-30b-a3b-instruct(?:-fp8)?$"), "Qwen3-Coder-30B-A3B-Instruct"),
    (re.compile(r"^amazon[.-]nova-premier-v1\.0$"), "Amazon Nova Premier 1.0"),
]


def _canonicalize_model_unified(
    name: str,
    *,
    strip_pro_suffix: bool = False,
    preserve_pretty: bool = False,
) -> str:
    raw = (name or "").strip()
    if not raw:
        return raw
    if strip_pro_suffix:
        raw = re.sub(_PRO_TRAILING_SUFFIX_RE, "", raw).strip()
    leaf = _model_leaf(raw)
    pretty_low = leaf.strip().lower()
    m_pretty = re.fullmatch(r"claude\s+(\d+(?:\.\d+)?)\s+(sonnet|opus|haiku)\s*", pretty_low)
    if m_pretty is not None:
        ver, size = m_pretty.group(1), m_pretty.group(2)
        if int(ver.split(".", 1)[0]) >= 4:
            return f"Claude {size.title()} {ver}"
    if preserve_pretty and any(ch.isupper() for ch in leaf) and ((" " in leaf) or bool(re.match(r"GPT-\d", leaf))):
        return leaf
    normalized = _normalize_model_key(leaf)
    if normalized in {"gpt-5-mini", "gpt-5-nano"}:
        return normalized.replace("gpt-5", "GPT-5")
    gpt5_base = None
    if re.match(r"gpt-?5(?:[-_.]?1)(?:$|[-_])", normalized):
        gpt5_base = "GPT-5.1"
    elif re.match(r"gpt-?5(?:[-_.]?2)(?:$|[-_])", normalized):
        gpt5_base = "GPT-5.2"
    elif re.match(r"gpt-?5(?:$|[-_\\.])", normalized) or normalized == "gpt5":
        gpt5_base = "GPT-5"
    if gpt5_base is not None:
        if "codex" in normalized:
            if "codex-mini" in normalized:
                return f"{gpt5_base}-Codex Mini"
            if "codex-max" in normalized:
                return f"{gpt5_base}-Codex Max"
            return f"{gpt5_base}-Codex"
        return gpt5_base
    if normalized in _MODEL_EXACT:
        return _MODEL_EXACT[normalized]
    for pattern, canonical in _MODEL_PATTERNS:
        if pattern.fullmatch(normalized):
            return canonical
    return leaf


def _canonical_model(m: str) -> str:
    return _canonicalize_model_unified(m, preserve_pretty=True)


def _version_scaffold_for_agent(agent: str, scaffold: str) -> str:
    a = (agent or "").strip()
    s = (scaffold or "").strip()
    if s != "SWE-agent":
        return s
    prefix = a.split("_", 1)[0]
    if len(prefix) == 8 and prefix.isdigit() and prefix >= "20250225":
        return "SWE-agent 1.0"
    return s


_AGENT_EXCLUSIONS: frozenset[str] = frozenset({
    "20241106_navie-2-gpt4o-sonnet",
})

_AGENT_OVERRIDES: dict[str, tuple[str, str]] = {
    "20250804_codesweep_sweagent_kimi_k2_instruct": ("kimi_k2_instruct", "sweagent"),
    "20250110_learn_by_interact_claude3.5": ("claude-3-5-sonnet", "learn_by_interact"),
    "20250405_swe-rizzo_claude37": ("claude-3-7-sonnet", "swe-rizzo"),
    "20241108_devlo": ("claude-3-5-sonnet", "20241108_devlo"),
    "20241016_composio_swekit": ("claude-3-5-sonnet", "20241016_composio_swekit"),
    "20241029_openhands-codeact-2.1-sonnet-20241022": ("claude-3-5-sonnet-20241022", "OpenHands-CodeAct-2.1"),
    "20241125_enginelabs": ("claude-3-5-sonnet", "enginelabs"),
    "20250120_bracket": ("gpt-4o-mini", "Bracket"),
    "20250112_ugaiforge": ("claude-3-5-sonnet", "ugaiforge"),
    "20250206_agentscope": ("claude-3-5-sonnet", "agentscope"),
    "20250514_aime_coder": ("claude-3-7-sonnet", "aime-coder"),
    "20250511_sweagent_lm_32b": ("SWE-agent-LM-32B", "SWE-agent"),
    "20250710_bloop": ("claude-4-sonnet", "bloop"),
    "20250731_harness_ai": ("claude-4-sonnet", "Harness"),
    "20250924_artemis_agent_v2": ("claude-4-sonnet", "Artemis Agent v2"),
    "20241120_artemis_agent": ("claude-3-5-sonnet", "Artemis Agent v1"),
    "20250527_amazon.nova-premier-v1.0": ("amazon.nova-premier-v1.0", "None"),
    "20250616_skywork-swe-32b": ("Skywork-SWE-32B", "OpenHands"),
    "20250629_deepswerl_r2eagent": ("DeepSWE-Preview", "R2E-Gym"),
    "20251110_frogboss-32b": ("FrogBoss-32B", "R2E-Gym"),
    "20251110_frogmini-14b": ("FrogMini-14B", "R2E-Gym"),
}


def split_agent_name(agent: str) -> Optional[tuple[str, str, str, str]]:
    raw = agent.strip()
    if not raw:
        return None
    if "_at_" in raw.lower():
        if "," in raw:
            return None
        raw = _strip_at_suffix(raw)
        if not raw:
            return None
    raw_low = raw.lower()
    if "refact_agent" in raw_low or "refact-agent" in raw_low or "cortexa" in raw_low:
        return None
    if raw_low in _AGENT_EXCLUSIONS:
        return None
    if raw_low.endswith("_zai_glm4-5"):
        model_raw, scaffold_raw = "glm4-5", "OpenHands"
        return (_canonical_model(model_raw), _canonical_scaffold(scaffold_raw), model_raw, scaffold_raw)
    if raw_low in _AGENT_OVERRIDES:
        model_raw, scaffold_raw = _AGENT_OVERRIDES[raw_low]
        return (_canonical_model(model_raw), _canonical_scaffold(scaffold_raw), model_raw, scaffold_raw)
    parts = raw.split("_")
    if len(parts) < 2:
        return None
    if parts[0].isdigit() and len(parts[0]) >= 6:
        parts = parts[1:]
    if parts and parts[-1].isdigit() and len(parts[-1]) == 8:
        parts = parts[:-1]
    if not parts:
        return None
    if len(parts) == 1 and re.fullmatch(r"autocoderover-v\d{8}", parts[0].lower()):
        scaffold_raw = parts[0]
        model_raw = "gpt4"
        return (_canonical_model(model_raw), _canonical_scaffold(scaffold_raw), model_raw, scaffold_raw)
    for start_idx in range(len(parts) - 1, 0, -1):
        model_raw = "_".join(parts[start_idx:])
        if _is_model_token(model_raw):
            scaffold_raw = "_".join(parts[:start_idx])
            return (_canonical_model(model_raw), _canonical_scaffold(scaffold_raw), model_raw, scaffold_raw)
    if len(parts) == 1:
        remainder = parts[0]
        if remainder.lower() == "skywork-swe-32b":
            return None
        dash_parts = remainder.split("-")
        if len(dash_parts) >= 2:
            for k in range(min(6, len(dash_parts)), 0, -1):
                model_raw = "-".join(dash_parts[-k:])
                scaffold_raw = "-".join(dash_parts[:-k])
                if scaffold_raw and _is_model_token(model_raw):
                    return (
                        _canonical_model(model_raw),
                        _canonical_scaffold(scaffold_raw),
                        model_raw,
                        scaffold_raw,
                    )
        low = remainder.lower()
        for prefix, canonical in (("openhands-", "OpenHands"), ("prometheus-", "Prometheus")):
            if low.startswith(prefix):
                model_raw = remainder[len(prefix) :]
                if not model_raw or not _is_model_token(model_raw):
                    return None
                return (_canonical_model(model_raw), canonical, model_raw, canonical)
        return None
    model_raw = parts[-1]
    scaffold_raw = "_".join(parts[:-1])
    if not _is_model_token(model_raw):
        return None
    return (_canonical_model(model_raw), _canonical_scaffold(scaffold_raw), model_raw, scaffold_raw)


def _read_agents_md(path: Path) -> list[str]:
    agents: list[str] = []
    with path.open("r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            agents.append(s)
    return agents


def _read_agents_results_jsonl(path: Path) -> list[str]:
    agents: list[str] = []
    with path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            agents.append(obj["subject_id"])
    return agents


def canonicalize_pro_model(model_name: str) -> str:
    return _canonicalize_model_unified(model_name, strip_pro_suffix=True)


def _scaffold_for_subject(subject_id: str, *, treat_as_pro: bool) -> Optional[str]:
    subj = str(subject_id or "").strip()
    if not subj:
        return None
    if treat_as_pro:
        return SWEBENCH_PRO_ASSUMED_SCAFFOLD
    split = split_agent_name(subj)
    if split is None:
        return None
    _, scaffold, _, _ = split
    scaffold = _version_scaffold_for_agent(subj, scaffold)
    return scaffold


def _model_for_subject(subject_id: str, *, treat_as_pro: bool) -> Optional[str]:
    subj = str(subject_id or "").strip()
    if not subj:
        return None
    if treat_as_pro:
        return canonicalize_pro_model(subj)
    split = split_agent_name(subj)
    if split is None:
        return None
    model, _, _, _ = split
    return model


def _read_pro_agents_results_jsonl(path: Path) -> list[str]:
    records: list[dict] = []
    with path.open("r") as f:
        for line in f:
            s = (line or "").strip()
            if not s:
                continue
            records.append(json.loads(s))
    prefer_paper_models = {"Claude Sonnet 4", "Gemini 2.5 Pro"}
    pro_is_paper = {str(r["subject_id"]): ("paper" in str(r["subject_id"]).lower()) for r in records}
    pro_canon = {str(r["subject_id"]): canonicalize_pro_model(str(r["subject_id"])) for r in records}
    have_paper_for_model = {
        m for subj, m in pro_canon.items() if (m in prefer_paper_models and pro_is_paper.get(subj, False))
    }
    out: list[str] = []
    for r in records:
        subj = str(r["subject_id"])
        m = pro_canon[subj]
        if m in have_paper_for_model and m in prefer_paper_models and not pro_is_paper.get(subj, False):
            continue
        out.append(subj)
    return out


def _read_terminal_bench_subjects_and_ms_jsonl(path: Path) -> tuple[list[str], dict[str, tuple[str, str]]]:
    agents: list[str] = []
    ms_by_agent: dict[str, tuple[str, str]] = {}
    with path.open("r") as f:
        for line in f:
            s = (line or "").strip()
            if not s:
                continue
            obj = json.loads(s)
            sid = str(obj.get("subject_id", "") or "").strip()
            if not sid:
                continue
            agents.append(sid)
            m1 = str(obj.get("model", "") or "").strip()
            a1 = str(obj.get("agent", "") or "").strip()
            if m1 and a1 and not a1.isdigit():
                model_raw, scaffold_raw = m1, a1
            else:
                m2 = str(obj.get("date", "") or "").strip()
                sc2 = m1
                if not (m2 and sc2):
                    continue
                model_raw, scaffold_raw = m2, sc2
            low = str(model_raw).strip().lower()
            if "," in model_raw or low == "multiple":
                continue
            model = _canonical_model(model_raw)
            scaffold = _canonical_scaffold(scaffold_raw)
            if model and scaffold:
                ms_by_agent[sid] = (str(model), str(scaffold))
    return agents, ms_by_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Split agent names into (model, scaffold)")
    parser.add_argument("--results_jsonl", type=str, required=True, help="Path to results-matrix JSONL")
    parser.add_argument(
        "--pro_results_jsonl",
        type=str,
        default=None,
        help=(
            "Optional path to a Pro results-matrix JSONL. If provided, Pro subjects are mapped with "
            f"model=canonicalize_pro_model(subject_id) and scaffold='{SWEBENCH_PRO_ASSUMED_SCAFFOLD}' "
            "(matching shared training)."
        ),
    )
    parser.add_argument("--agents_md", type=str, default=None, help="Optional path to agent list (one per line)")
    parser.add_argument(
        "--terminal_bench_results_jsonl",
        type=str,
        default=None,
        help=(
            "Optional path to Terminal-Bench 2.0 results-matrix JSONL. If provided, subjects are treated like "
            "Verified-style agent ids (subject_id encodes scaffold+model, often with an '_at_' provider suffix)."
        ),
    )
    parser.add_argument("--output_csv", type=str, required=True, help="Where to write CSV mapping")
    parser.add_argument(
        "--unsplittable_txt",
        type=str,
        default=None,
        help="Optional path to write unsplittable agent names",
    )
    args = parser.parse_args()
    results_jsonl = resolve_path(args.results_jsonl)
    pro_results_jsonl = resolve_path(args.pro_results_jsonl) if args.pro_results_jsonl else None
    agents_md = resolve_path(args.agents_md) if args.agents_md else None
    terminal_bench_results_jsonl = (
        resolve_path(args.terminal_bench_results_jsonl) if args.terminal_bench_results_jsonl else None
    )
    output_csv = resolve_path(args.output_csv)
    unsplittable_txt = resolve_path(args.unsplittable_txt) if args.unsplittable_txt else None
    results_agents = _read_agents_results_jsonl(results_jsonl)
    md_agents = _read_agents_md(agents_md) if agents_md else []
    pro_agents = _read_pro_agents_results_jsonl(pro_results_jsonl) if pro_results_jsonl else []
    terminal_bench_agents = (
        _read_agents_results_jsonl(terminal_bench_results_jsonl) if terminal_bench_results_jsonl else []
    )
    terminal_bench_ms_by_agent: dict[str, tuple[str, str]] = {}
    if terminal_bench_results_jsonl is not None:
        try:
            terminal_bench_agents, terminal_bench_ms_by_agent = _read_terminal_bench_subjects_and_ms_jsonl(
                terminal_bench_results_jsonl
            )
        except Exception:
            terminal_bench_ms_by_agent = {}
    sources_by_agent: dict[str, set[str]] = {}
    for a in results_agents:
        sources_by_agent.setdefault(a, set()).add("results_jsonl")
    for a in md_agents:
        sources_by_agent.setdefault(a, set()).add("agents_md")
    for a in pro_agents:
        sources_by_agent.setdefault(a, set()).add("pro_results_jsonl")
    for a in terminal_bench_agents:
        sources_by_agent.setdefault(a, set()).add("terminal_bench_results_jsonl")
    splittable_rows: list[dict[str, str]] = []
    unsplittable: list[str] = []
    for agent in sorted(sources_by_agent.keys()):
        sources = sources_by_agent.get(agent, set())
        if "pro_results_jsonl" in sources:
            model = canonicalize_pro_model(agent)
            scaffold = SWEBENCH_PRO_ASSUMED_SCAFFOLD
        elif "terminal_bench_results_jsonl" in sources and agent in terminal_bench_ms_by_agent:
            model, scaffold = terminal_bench_ms_by_agent[agent]
        else:
            split = split_agent_name(agent)
            if split is None:
                unsplittable.append(agent)
                continue
            model, scaffold, model_raw, scaffold_raw = split
            scaffold = _version_scaffold_for_agent(agent, scaffold)
        splittable_rows.append(
            {
                "agent": agent,
                "model": model,
                "scaffold": scaffold,
            }
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["agent", "model", "scaffold"],
        )
        w.writeheader()
        w.writerows(splittable_rows)
    if unsplittable_txt is not None:
        unsplittable_txt.parent.mkdir(parents=True, exist_ok=True)
        with unsplittable_txt.open("w") as f:
            for a in unsplittable:
                f.write(a)
                f.write("\n")
    print(f"Total unique agents (union): {len(sources_by_agent)}")
    print(f"Splittable: {len(splittable_rows)}")
    print(f"Unsplittable: {len(unsplittable)}")
    print(f"Wrote mapping CSV: {output_csv}")
    if unsplittable_txt is not None:
        print(f"Wrote unsplittable list: {unsplittable_txt}")


if __name__ == "__main__":
    main()

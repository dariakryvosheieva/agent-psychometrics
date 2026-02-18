"""
Split SWE-bench Verified agent names into (model, scaffold).

This script is intentionally lightweight (no ML training) and is safe to run
on a login node.

Inputs
------
- A results matrix JSONL in the format used by swebench_irt/train.py:
    {"subject_id": "<agent_name>", "responses": {...}}
- Optionally, a Pro results-matrix JSONL where subject_id is a model-name string.
- Optionally, a plain-text list of agent names (one per line), e.g. verified_agents.md

Output
------
CSV with columns:
  agent, model, scaffold

Only agents that can be confidently split are included. Others are ignored.

Usage
-----
  .venv/bin/python swebench_irt/split_agents_model_scaffold.py \
    --results_jsonl fulcrum/fellowship/out/chris_irt/swebench_verified_20251115_full.jsonl \
    --pro_results_jsonl fulcrum/fellowship/out/chris_irt/swebench_pro.jsonl \
    --agents_md fulcrum/fellowship/out/chris_irt/verified_agents.md \
    --output_csv fulcrum/fellowship/out/chris_irt/agent_model_scaffold.csv \
    --unsplittable_txt fulcrum/fellowship/out/chris_irt/agent_unsplittable.txt
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional
import re


ROOT = Path(__file__).resolve().parents[1]

# -----------------------------
# Assumed scaffolds (model-only subject_id exports)
# -----------------------------
#
# Policy (analysis convention):
# - GSO: subject_id is model-only; assume scaffold is OpenHands.
# - SWE-bench Pro: subject_id is model-only; assume scaffold is SWE-agent 1.0.

GSO_ASSUMED_SCAFFOLD: str = "OpenHands"
SWEBENCH_PRO_ASSUMED_SCAFFOLD: str = "SWE-agent 1.0"


def assumed_scaffold_for_benchmark(benchmark: str) -> Optional[str]:
    """
    Return the assumed scaffold for benchmarks whose subjects are model-only strings.

    `benchmark` should be a canonical key such as: verified, pro, terminal_bench, gso.
    """
    b = str(benchmark or "").strip().lower().replace("-", "_")
    if b == "gso":
        return GSO_ASSUMED_SCAFFOLD
    if b == "pro":
        return SWEBENCH_PRO_ASSUMED_SCAFFOLD
    return None

_AT_SUFFIX_RE = re.compile(r"_at_.*$", flags=re.IGNORECASE)
_TRAILING_DATE_RE = re.compile(r"[-_]\d{8}$")  # e.g. "-20251101" or "_20251101"


def _strip_at_suffix(agent: str) -> str:
    """
    Terminal-Bench convention:
      <scaffold_and_model>_at_<provider>
    Sometimes a date suffix appears just before "_at_" (e.g. "...-20251101_at_*").

    This returns the prefix before "_at_" and also strips a trailing YYYYMMDD token
    if present immediately before the "_at_" segment.
    """
    raw = (agent or "").strip()
    if "_at_" not in raw.lower():
        return raw
    base = re.sub(_AT_SUFFIX_RE, "", raw).strip()
    base = re.sub(_TRAILING_DATE_RE, "", base).strip()
    return base


def _model_leaf(token: str) -> str:
    """
    Normalize model tokens that may include provider prefixes (e.g. "openai/gpt-oss-120b")
    into their leaf name used for detection/canonicalization.
    """
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
    """
    Conservative heuristic: treat the final token as a "model" token only if it
    very likely refers to a base model name.
    """
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

    # Open-weight models often include a parameter count like "7b", "72b", etc.
    has_digit = any(ch.isdigit() for ch in tl)
    # But a bare count token like "32b" is not a model identifier by itself.
    if re.fullmatch(r"\d+(?:\.\d+)?b", tl):
        return False
    # And "lm_32b" is an agent-family token in this dataset, not a base model name.
    if re.fullmatch(r"lm[_-]\d+(?:\.\d+)?b", tl):
        return False
    if has_digit and (tl.endswith("b") or "b-" in tl or "-b" in tl):
        return True

    return False


def _canonical_scaffold(s: str) -> str:
    sl = s.strip()
    if not sl:
        # Avoid empty scaffolds: treat as explicit "no scaffold" category.
        return "NoScaffold"
    low = sl.lower()
    # Verified convention: "Lingxi" refers to the Lingxi v1.5 scaffold.
    if low in {"lingxi", "lingxi v1.5", "lingxi-v1.5", "lingxi_v1.5"}:
        return "Lingxi v1.5"
    # Verified convention: merge "jules" scaffold into the canonical "google_jules".
    if low in {"jules", "google-jules", "google_jules"}:
        return "google_jules"
    if low == "openhands":
        return "OpenHands"
    if low == "tools":
        return "Tools"
    if low == "sweagent":
        return "SWE-agent"
    if low == "harness":
        return "Harness"
    if low == "rag":
        return "RAG"
    # IMPORTANT: avoid tokens that pandas may auto-parse as NA/NaN.
    # Use a stable explicit category instead of the literal string "None".
    if low in {"none", "null", "nan", "na", "n/a"}:
        return "NoScaffold"
    return sl


def _canonical_model(m: str) -> str:
    ml = m.strip()
    if not ml:
        return ml
    leaf = _model_leaf(ml)
    # Merge known "pretty-name" aliases before the pretty-name preservation rule below.
    # This prevents accidental fragmentation when two human-readable variants refer to the same base model.
    pretty_low = leaf.strip().lower()
    # Canonicalize Claude v4+ ordering: "Claude <ver> <size>" -> "Claude <size> <ver>".
    # Apply before the "pretty name preservation" rule, otherwise we'd lock in the non-canonical order.
    m_pretty = re.fullmatch(r"claude\s+(\d+(?:\.\d+)?)\s+(sonnet|opus|haiku)\s*", pretty_low)
    if m_pretty is not None:
        ver = m_pretty.group(1)
        size = m_pretty.group(2)
        major = int(ver.split(".", 1)[0])
        if major >= 4:
            return f"Claude {size.title()} {ver}"
    if pretty_low in {"claude 4.5 sonnet", "claude 4.5 connet"}:
        return "Claude Sonnet 4.5"
    if pretty_low in {"claude 4.5 opus"}:
        return "Claude Opus 4.5"
    # Merge Kimi K2 into Kimi K2 Instruct (analysis convention).
    if pretty_low in {"kimi k2", "kimi k2 instruct"}:
        return "Kimi K2 Instruct"
    # If the source is already using a “pretty” name (e.g. Terminal-Bench columns like
    # "Claude Sonnet 4.5", "Claude Opus 4.6", "GPT-5.3-Codex"), preserve it verbatim.
    # This avoids lossy re-canon (e.g. dropping "5.3" or changing separator style).
    if any(ch.isupper() for ch in leaf) and ((" " in leaf) or bool(re.match(r"GPT-\d", leaf))):
        return leaf
    # Normalize separators: many exports use underscores; some use spaces in "pretty" names.
    # Treat spaces like hyphens so canonicalization is stable across sources.
    low = re.sub(r"\s+", "-", leaf.lower().replace("_", "-"))

    # -----------------------------
    # Terminal-Bench-oriented canon
    # -----------------------------
    # GPT-5 family
    # Keep mini/nano variants distinct.
    if low in {"gpt-5-mini", "gpt-5-nano"}:
        return low.replace("gpt-5", "GPT-5")

    # Keep GPT-5, GPT-5.1, GPT-5.2 distinct.
    # Accept ".", "-" or "_" as the version separator (e.g. "gpt-5.1", "gpt-5_1", "gpt-5-1").
    if re.match(r"gpt-?5(?:[-_.]?1)(?:$|[-_])", low):
        gpt5_base = "GPT-5.1"
    elif re.match(r"gpt-?5(?:[-_.]?2)(?:$|[-_])", low):
        gpt5_base = "GPT-5.2"
    # Treat any other GPT-5-* prefix as base GPT-5 (e.g. "gpt-5-codex").
    elif re.match(r"gpt-?5(?:$|[-_\\.])", low) or low == "gpt5":
        gpt5_base = "GPT-5"
    else:
        gpt5_base = None

    if gpt5_base is not None:
        # Preserve Codex variants
        if "codex" in low:
            if "codex-mini" in low:
                return f"{gpt5_base} Codex Mini"
            if "codex-max" in low:
                return f"{gpt5_base} Codex Max"
            return f"{gpt5_base} Codex"
        return gpt5_base

    # OpenAI open-weight family (keep sizes distinct; Pro "GPT OSS" treated as 120B)
    if re.fullmatch(r"gpt-oss-20b", low) or re.fullmatch(r"gptoss-20b", low):
        return "GPT OSS 20B"
    if re.fullmatch(r"gpt-oss-120b", low) or re.fullmatch(r"gptoss-120b", low):
        return "GPT OSS 120B"
    if low.startswith("gpt-oss") or low.startswith("gptoss"):
        return "GPT OSS"

    # Claude 4.5 family (Terminal-Bench uses hyphenated tokens)
    if re.fullmatch(r"claude-sonnet-4-5", low):
        return "Claude Sonnet 4.5"
    if re.fullmatch(r"claude-4[-_]?5-sonnet", low) or re.fullmatch(r"claude-4-5-sonnet", low):
        return "Claude Sonnet 4.5"
    if re.fullmatch(r"claude-haiku-4-5", low):
        return "Claude Haiku 4.5"
    # Opus variants seen in Terminal-Bench / OOD exports.
    # IMPORTANT: keep 4 vs 4.5 distinct.
    # Some datasets use "claude-4-opus" while others use "claude-opus-4" for the same model.
    if re.fullmatch(r"claude-4-opus", low) or re.fullmatch(r"claude-4[-_]?0-opus", low):
        return "Claude Opus 4"
    if re.fullmatch(r"claude-opus-4", low):
        return "Claude Opus 4"
    if (
        re.fullmatch(r"claude-opus-4-5(?:[-_]?\\d{8})?", low)
        or re.fullmatch(r"claude-opus-4\.5", low)
        or re.fullmatch(r"claude-4[-_]?5-opus", low)
        or re.fullmatch(r"claude-4-5-opus", low)
    ):
        return "Claude Opus 4.5"
    if re.fullmatch(r"claude-opus-4-1", low) or re.fullmatch(r"claude-opus-4\.1", low):
        return "Claude Opus 4.1"

    # Claude 4 Sonnet naming variants seen in OOD exports (e.g. GSO)
    if re.fullmatch(r"claude-sonnet-4", low):
        return "Claude Sonnet 4"

    # Claude Sonnet 4.5 dot variant (e.g. "claude-sonnet-4.5")
    if re.fullmatch(r"claude-sonnet-4\.5", low):
        return "Claude Sonnet 4.5"

    # Gemini 3 naming variants in some OOD exports
    if low == "gemini-3-flash":
        return "gemini-3-flash-preview"
    if low == "gemini-3-pro":
        return "gemini-3-pro-preview"

    # Gemini 2.5 Pro naming in Terminal-Bench exports
    if re.fullmatch(r"gemini-2(?:-|\.|\s)?5-pro", low) or re.fullmatch(r"gemini-2-5-pro", low):
        return "Gemini 2.5 Pro Preview"

    # Qwen3 coder fp8 suffix variants (keep FP8 distinct for 480B)
    if re.fullmatch(r"qwen3-coder-480b-a35b-instruct-fp8", low):
        return "Qwen3-Coder-480B-A35B-Instruct-FP8"
    if re.fullmatch(r"qwen3-coder-480b-a35b-instruct", low):
        return "Qwen3-Coder-480B-A35B-Instruct"
    if re.fullmatch(r"qwen3-coder-30b-a3b-instruct(?:-fp8)?", low):
        return "Qwen3-Coder-30B-A3B-Instruct"
    # Some OOD exports use a coarse family token ("qwen3-coder") without size/suffix.
    # For our purposes, merge it into the strongest/most-common Terminal-Bench variant.
    if low == "qwen3-coder":
        return "Qwen3-Coder-480B-A35B-Instruct-FP8"

    # GLM fireworks naming (e.g. "glm-4p6" -> glm4-6)
    # Merge GLM-4.5 variants (human label vs compact token)
    if low in {"glm4-5", "glm-4.5", "glm-4-5"}:
        return "GLM-4.5"
    if re.fullmatch(r"glm-4p6", low):
        return "glm4-6"
    # Normalize legacy Qwen coder tokens used in some agent ids.
    # Example agent id: 20250901_entroPO_R2E_QwenCoder30BA3B
    if low == "qwencoder30ba3b":
        return "Qwen3-Coder-30B-A3B-Instruct"
    # Normalize Claude 3.5 Sonnet variants (dated / updated)
    # Examples:
    # - claude3.5sonnet
    # - claude-3-5-sonnet
    # - claude-3-5-sonnet-20241022
    # - claude-3-5-sonnet-updated
    if (
        re.fullmatch(r"claude-3-5-sonnet(?:-\d{8})?(?:-updated)?", low)
        or re.fullmatch(r"claude-3\.5-sonnet(?:-\d{8})?(?:-updated)?", low)
        or low == "claude3.5sonnet"
    ):
        return "Claude 3.5 Sonnet"
    # Normalize Claude 3.7 Sonnet variants (dated / underscore)
    # Examples:
    # - claude-3-7-sonnet
    # - claude-3-7-sonnet-20250219
    # - claude_3_7_sonnet (becomes claude-3-7-sonnet after '_' -> '-')
    if (
        re.fullmatch(r"claude-3-7-sonnet(?:-\d{8})?(?:-updated)?", low)
        or re.fullmatch(r"claude-3\.7-sonnet(?:-\d{8})?(?:-updated)?", low)
        or low == "claude37sonnet"
    ):
        return "Claude 3.7 Sonnet"
    # Normalize Claude 4 Sonnet variants (dated / underscore)
    # Examples:
    # - claude-4-sonnet
    # - claude-4-sonnet-20250514
    # - claude_4_sonnet  (becomes claude-4-sonnet after '_' -> '-')
    if re.fullmatch(r"claude-4-sonnet(?:-\d{8})?(?:-updated)?", low) or low == "claude4sonnet":
        return "Claude Sonnet 4"
    # Normalize Amazon Nova Premier v1.0 variants
    # Examples:
    # - amazon.nova-premier-v1.0
    # - amazon-nova-premier-v1.0
    if re.fullmatch(r"amazon[.-]nova-premier-v1\.0", low):
        return "Amazon Nova Premier 1.0"
    # Normalize FrogBoss 32B variants
    if re.fullmatch(r"frogboss-32b", low):
        return "FrogBoss-32B"
    # Normalize FrogMini 14B variants
    if re.fullmatch(r"frogmini-14b", low):
        return "FrogMini-14B"
    mapping = {
        "gpt5": "GPT-5",
        "gpt-5": "GPT-5",
        "gpt4o": "GPT-4o",
        "gpt-4o": "GPT-4o",
        "gpt4o-mini": "GPT-4o-mini",
        "gpt-4o-mini": "GPT-4o-mini",
        "gpt4": "GPT-4",
        "gpt-4": "GPT-4",
        "gpt35": "GPT-3.5",
        "gpt-3.5": "GPT-3.5",
        "claude2": "Claude 2",
        "claude-2": "Claude 2",
        "claude3opus": "Claude 3 Opus",
        "claude-3-opus": "Claude 3 Opus",
        "kimi-k2": "Kimi K2 Instruct",
    }
    return mapping.get(low, leaf)


def _version_scaffold_for_agent(agent: str, scaffold: str) -> str:
    """
    Post-process scaffold labels using dataset conventions.

    Policy:
    - For agents dated on/after 2025-02-25, rename "SWE-agent" -> "SWE-agent 1.0".
    """
    a = (agent or "").strip()
    s = (scaffold or "").strip()
    if s != "SWE-agent":
        return s
    # Agent ids are typically "YYYYMMDD_<rest>".
    prefix = a.split("_", 1)[0]
    if len(prefix) == 8 and prefix.isdigit() and prefix >= "20250225":
        return "SWE-agent 1.0"
    return s


def split_agent_name(agent: str) -> Optional[tuple[str, str, str, str]]:
    """
    Attempt to split an agent name into (model, scaffold).

    Returns:
      (model, scaffold, model_raw, scaffold_raw) or None if unsplittable.
    """
    raw = agent.strip()
    if not raw:
        return None

    # Terminal-Bench convention: strip provider suffix.
    # If an agent is a multi-model bundle (comma-separated), treat as unsplittable.
    if "_at_" in raw.lower():
        if "," in raw:
            return None
        raw = _strip_at_suffix(raw)
        if not raw:
            return None

    # Dataset-specific exclusion: Refact.ai Agent uses multiple base models
    # (see Verified metadata tags.model), so it should not be forced into a single (model, scaffold).
    if "refact_agent" in raw.lower() or "refact-agent" in raw.lower():
        return None

    # Dataset-specific exclusion: Nemotron-CORTEXA uses multiple base models
    # (see Verified metadata tags.model), so it should not be forced into a single (model, scaffold).
    if "cortexa" in raw.lower():
        return None

    # Dataset-specific special-case: this agent id contains a composite suffix
    # ("gpt4o-sonnet") that is not a base model token in our decomposition scheme.
    # Treat it as unsplittable so it is written to --unsplittable_txt.
    if raw.lower() == "20241106_navie-2-gpt4o-sonnet":
        return None

    # Dataset-specific special-case: CodeSweep SWE-agent run.
    # The agent id encodes the SWE-agent family in the middle; map it to SWE-agent
    # so scaffold versioning applies (>= 2025-02-25 -> SWE-agent 1.0).
    if raw.lower() == "20250804_codesweep_sweagent_kimi_k2_instruct":
        scaffold_raw = "sweagent"
        model_raw = "kimi_k2_instruct"
        return (_canonical_model(model_raw), _canonical_scaffold(scaffold_raw), model_raw, scaffold_raw)

    # Dataset-specific special-case: GLM4-5 runs are OpenHands scaffolded (not "zai").
    # Example: "20250728_zai_glm4-5"
    if raw.lower().endswith("_zai_glm4-5"):
        scaffold_raw = "OpenHands"
        model_raw = "glm4-5"
        return (_canonical_model(model_raw), _canonical_scaffold(scaffold_raw), model_raw, scaffold_raw)

    # Dataset-specific special-case: this run uses "claude3.5" in the agent id, but
    # we want it grouped with Claude 3.5 Sonnet for model+scaffold decomposition.
    if raw.lower() == "20250110_learn_by_interact_claude3.5":
        scaffold_raw = "learn_by_interact"
        model_raw = "claude-3-5-sonnet"
        return (_canonical_model(model_raw), _canonical_scaffold(scaffold_raw), model_raw, scaffold_raw)

    # Dataset-specific special-case: this run uses "claude37" in the agent id, but
    # we want it grouped with Claude 3.7 Sonnet for model+scaffold decomposition.
    if raw.lower() == "20250405_swe-rizzo_claude37":
        scaffold_raw = "swe-rizzo"
        model_raw = "claude-3-7-sonnet"
        return (_canonical_model(model_raw), _canonical_scaffold(scaffold_raw), model_raw, scaffold_raw)

    # Dataset-specific special-case: "devlo" agents omit an explicit model token.
    # For this specific run, keep the full agent name as the scaffold.
    if raw.lower() == "20241108_devlo":
        scaffold_raw = "20241108_devlo"
        model_raw = "claude-3-5-sonnet"
        return (_canonical_model(model_raw), _canonical_scaffold(scaffold_raw), model_raw, scaffold_raw)

    # Dataset-specific special-case: some agents omit an explicit model token, but
    # we still want the full agent string as the scaffold (to avoid date-stripping).
    if raw.lower() == "20241016_composio_swekit":
        scaffold_raw = "20241016_composio_swekit"
        model_raw = "claude-3-5-sonnet"
        return (_canonical_model(model_raw), _canonical_scaffold(scaffold_raw), model_raw, scaffold_raw)

    # Dataset-specific special-case: OpenHands CodeAct run encodes model as "sonnet-YYYYMMDD"
    # rather than "claude-3-5-sonnet-YYYYMMDD". Keep scaffold at the framework version.
    if raw.lower() == "20241029_openhands-codeact-2.1-sonnet-20241022":
        scaffold_raw = "OpenHands-CodeAct-2.1"
        model_raw = "claude-3-5-sonnet-20241022"
        return (_canonical_model(model_raw), _canonical_scaffold(scaffold_raw), model_raw, scaffold_raw)

    # Dataset-specific special-case: some agents omit an explicit model token.
    if raw.lower() == "20241125_enginelabs":
        scaffold_raw = "enginelabs"
        model_raw = "claude-3-5-sonnet"
        return (_canonical_model(model_raw), _canonical_scaffold(scaffold_raw), model_raw, scaffold_raw)

    # Dataset-specific special-case: Bracket run omits an explicit model token.
    if raw.lower() == "20250120_bracket":
        scaffold_raw = "Bracket"
        model_raw = "gpt-4o-mini"
        return (_canonical_model(model_raw), _canonical_scaffold(scaffold_raw), model_raw, scaffold_raw)

    # Dataset-specific special-case: ugaiforge run omits an explicit model token.
    if raw.lower() == "20250112_ugaiforge":
        scaffold_raw = "ugaiforge"
        model_raw = "claude-3-5-sonnet"
        return (_canonical_model(model_raw), _canonical_scaffold(scaffold_raw), model_raw, scaffold_raw)

    # Dataset-specific special-case: agentscope run omits an explicit model token.
    if raw.lower() == "20250206_agentscope":
        scaffold_raw = "agentscope"
        model_raw = "claude-3-5-sonnet"
        return (_canonical_model(model_raw), _canonical_scaffold(scaffold_raw), model_raw, scaffold_raw)

    # Dataset-specific special-case: AIME coder run omits an explicit model token.
    if raw.lower() == "20250514_aime_coder":
        scaffold_raw = "aime-coder"
        model_raw = "claude-3-7-sonnet"
        return (_canonical_model(model_raw), _canonical_scaffold(scaffold_raw), model_raw, scaffold_raw)

    # Dataset-specific special-case: treat SWE-agent LM-32B as a distinct "model"
    # family with SWE-agent as the scaffold.
    if raw.lower() == "20250511_sweagent_lm_32b":
        scaffold_raw = "SWE-agent"
        model_raw = "SWE-agent-LM-32B"
        return (_canonical_model(model_raw), _canonical_scaffold(scaffold_raw), model_raw, scaffold_raw)

    # Dataset-specific special-case: bloop run omits an explicit model token.
    if raw.lower() == "20250710_bloop":
        scaffold_raw = "bloop"
        model_raw = "claude-4-sonnet"
        return (_canonical_model(model_raw), _canonical_scaffold(scaffold_raw), model_raw, scaffold_raw)

    # Dataset-specific special-case: harness_ai run omits an explicit model token.
    if raw.lower() == "20250731_harness_ai":
        scaffold_raw = "Harness"
        model_raw = "claude-4-sonnet"
        return (_canonical_model(model_raw), _canonical_scaffold(scaffold_raw), model_raw, scaffold_raw)

    # Dataset-specific special-case: Artemis Agent v2 run omits an explicit model token.
    if raw.lower() == "20250924_artemis_agent_v2":
        scaffold_raw = "Artemis Agent v2"
        model_raw = "claude-4-sonnet"
        return (_canonical_model(model_raw), _canonical_scaffold(scaffold_raw), model_raw, scaffold_raw)

    # Dataset-specific special-case: Artemis Agent v1 run omits an explicit model token.
    if raw.lower() == "20241120_artemis_agent":
        scaffold_raw = "Artemis Agent v1"
        model_raw = "claude-3-5-sonnet"
        return (_canonical_model(model_raw), _canonical_scaffold(scaffold_raw), model_raw, scaffold_raw)

    # Dataset-specific special-case: Amazon Nova Premier runs should be treated as "model-only"
    # (no meaningful scaffold to estimate separately).
    if raw.lower() == "20250527_amazon.nova-premier-v1.0":
        scaffold_raw = "None"
        model_raw = "amazon.nova-premier-v1.0"
        return (_canonical_model(model_raw), _canonical_scaffold(scaffold_raw), model_raw, scaffold_raw)

    # Dataset-specific special-case: this agent should be treated as OpenHands + Skywork-SWE-32B.
    # Note: without this override, "skywork-swe-32b" is excluded in the hyphen-split logic.
    if raw.lower() == "20250616_skywork-swe-32b":
        scaffold_raw = "OpenHands"
        model_raw = "Skywork-SWE-32B"
        return (_canonical_model(model_raw), _canonical_scaffold(scaffold_raw), model_raw, scaffold_raw)

    # Dataset-specific special-case: DeepSWE preview run under R2E-Gym scaffold.
    if raw.lower() == "20250629_deepswerl_r2eagent":
        scaffold_raw = "R2E-Gym"
        model_raw = "DeepSWE-Preview"
        return (_canonical_model(model_raw), _canonical_scaffold(scaffold_raw), model_raw, scaffold_raw)

    # Dataset-specific special-case: FrogBoss 32B run under R2E-Gym scaffold.
    if raw.lower() == "20251110_frogboss-32b":
        scaffold_raw = "R2E-Gym"
        model_raw = "FrogBoss-32B"
        return (_canonical_model(model_raw), _canonical_scaffold(scaffold_raw), model_raw, scaffold_raw)

    # Dataset-specific special-case: FrogMini 14B run under R2E-Gym scaffold.
    if raw.lower() == "20251110_frogmini-14b":
        scaffold_raw = "R2E-Gym"
        model_raw = "FrogMini-14B"
        return (_canonical_model(model_raw), _canonical_scaffold(scaffold_raw), model_raw, scaffold_raw)

    # Drop leading numeric prefix (usually YYYYMMDD, sometimes shorter IDs)
    parts = raw.split("_")
    if len(parts) < 2:
        return None
    if parts[0].isdigit() and len(parts[0]) >= 6:
        parts = parts[1:]
    # Drop trailing date tag (common in some agent naming schemes)
    if parts and parts[-1].isdigit() and len(parts[-1]) == 8:
        parts = parts[:-1]
    if not parts:
        return None

    # Dataset-specific special-case: early AutoCodeRover runs encode the version as a date
    # (e.g. "autocoderover-v20240620") and omit the base model token entirely.
    # For model+scaffold IRT splitting, treat these as scaffold="autocoderover-vYYYYMMDD"
    # and model="GPT-4" (per analysis convention).
    if len(parts) == 1 and re.fullmatch(r"autocoderover-v\d{8}", parts[0].lower()):
        scaffold_raw = parts[0]
        model_raw = "gpt4"
        return (_canonical_model(model_raw), _canonical_scaffold(scaffold_raw), model_raw, scaffold_raw)

    # Prefer underscore-based split, allowing multi-token model names like
    # "gemini_2.0_flash_experimental" or "kimi_k2".
    #
    # We scan from the right and pick the *latest* start position that yields a
    # plausible model token, to avoid swallowing scaffold tokens (e.g.
    # "lingma-agent" contains "lingma" but is scaffold-like).
    for start_idx in range(len(parts) - 1, 0, -1):
        model_raw = "_".join(parts[start_idx:])
        if _is_model_token(model_raw):
            scaffold_raw = "_".join(parts[:start_idx])
            return (_canonical_model(model_raw), _canonical_scaffold(scaffold_raw), model_raw, scaffold_raw)

    if len(parts) == 1:
        remainder = parts[0]
        # Dataset-specific exclusion: this is an agent name, not a (scaffold, model) pair.
        if remainder.lower() == "skywork-swe-32b":
            return None
        # Try scaffold-model split via "<scaffold>-<model...>" (hyphen-separated)
        dash_parts = remainder.split("-")
        if len(dash_parts) >= 2:
            # Take the largest suffix that looks like a model token.
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

        # Special-case "<scaffold>-<model>" where scaffold is a known family
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


# -----------------------------
# Pro model canonicalization (match train_model_scaffold_shared.py convention)
# -----------------------------

_PRO_TRAILING_SUFFIX_RE = re.compile(r"\s+(?:--|-)\s+.+$")  # strip " - paper", " - 10132025", " -- debug-oct22", ...


def canonicalize_pro_model(model_name: str) -> str:
    """
    Canonicalize SWE-bench Pro `subject_id` model strings so they align with Verified
    naming conventions where requested.
    """
    raw = (model_name or "").strip()
    if not raw:
        return raw

    # Remove paper/date/debug-ish suffixes (analysis convention).
    base = re.sub(_PRO_TRAILING_SUFFIX_RE, "", raw).strip()
    low = base.lower()

    # Canonicalize Claude v4+ ordering: "Claude <ver> <size>" -> "Claude <size> <ver>".
    m_pretty = re.fullmatch(r"claude\s+(\d+(?:\.\d+)?)\s+(sonnet|opus|haiku)\s*", low)
    if m_pretty is not None:
        ver = m_pretty.group(1)
        size = m_pretty.group(2)
        major = int(ver.split(".", 1)[0])
        if major >= 4:
            return f"Claude {size.title()} {ver}"

    # Merge "Claude Sonnet 4" and "Claude 4 Sonnet" -> canonical "Claude Sonnet 4".
    if low in {"claude sonnet 4"}:
        return "Claude Sonnet 4"

    # Merge Gemini 2.5 Pro Preview variants (paper/debug) -> a single label.
    if low == "gemini 2.5 pro preview":
        return "Gemini 2.5 Pro Preview"

    # Merge GPT-5 (base) with Verified.
    if low == "gpt-5" or low == "gpt 5":
        return "GPT-5"

    # Merge GPT-5 Codex naming variants.
    if low in {"gpt 5 codex", "gpt-5-codex", "gpt-5 codex", "gpt5-codex", "gpt5 codex"}:
        return "GPT-5 Codex"

    # Treat Pro "GPT OSS" as the 120B model.
    if low == "gpt oss" or low == "gpt-oss" or low == "gptoss":
        return "GPT OSS 120B"

    # Merge Claude 4.5 Sonnet with Verified's token (kept as-is in Verified results).
    if low in {"claude 4.5 sonnet", "claude 4.5 connet"}:
        return "Claude Sonnet 4.5"

    # Merge Kimi (paper) with Verified's "kimi_k2_instruct".
    if low == "kimi":
        return "kimi_k2_instruct"

    return base


def _read_pro_agents_results_jsonl(path: Path) -> list[str]:
    """
    Return Pro subject_id strings after applying the shared-training convention:
    - prefer "paper" variants for specific canonical models when present
    - keep subject_id strings as-is (so they match the Pro responses JSONL)
    """
    records: list[dict] = []
    with path.open("r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            records.append(json.loads(s))

    prefer_paper_models = {"Claude Sonnet 4", "Gemini 2.5 Pro Preview"}
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
    """
    Read Terminal-Bench 2.0 leaderboard scrape JSONL.

    Returns:
      - list of subject_id strings (agent ids used in response matrices)
      - mapping {subject_id: (model, scaffold)} using canonical “pretty” names

    Preferred schema:
      - obj["model"] -> model (base LLM)           [pretty]
      - obj["agent"] -> scaffold/agent name       [pretty]

    Back-compat (current scrape output in this repo):
      - obj["date"]  -> model (base LLM)           [pretty]
      - obj["model"] -> scaffold/agent name        [pretty]
      - obj["agent"] is a numeric rank string
    """
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

    # Union of names (we record source(s) per agent).
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

        # Pro subjects are already "model strings" (not dated agent ids). Match shared-training convention:
        # - fixed scaffold
        # - canonicalize the model label
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


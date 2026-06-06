"""Utilities for combining model and scaffold ability estimates."""

import json
from pathlib import Path

THETA_COMBINE_CHOICES = (
    "sum",
    "weighted_sum",
    "model_scaffold_scale",
    "mean",
    "l2",
    "harmonic",
    "product",
    "min",
    "max",
)


def normalize_theta_combine(combine):
    """Normalize and validate a theta-combination functional form."""
    combine_norm = str(combine or "sum").strip().lower().replace("-", "_").replace(" ", "_")
    if combine_norm not in THETA_COMBINE_CHOICES:
        raise ValueError(f"Unknown theta_combine={combine!r}")
    return combine_norm


def theta_combine_cache_suffix(combine):
    """Return the cache-dir suffix for non-default theta-combination forms."""
    combine_norm = normalize_theta_combine(combine)
    if combine_norm == "sum":
        return ""
    return f"_theta-{combine_norm}"


def theta_combine_weights_from_attrs(model_abilities, scaffold_abilities, *, combine: str):
    """Fetch learned theta-combination parameters from DataFrame attrs when needed."""
    combine_norm = normalize_theta_combine(combine)
    if combine_norm not in {"weighted_sum", "model_scaffold_scale"}:
        return {}

    model_attrs = getattr(model_abilities, "attrs", {})
    scaffold_attrs = getattr(scaffold_abilities, "attrs", {})
    if combine_norm == "model_scaffold_scale":
        scales = model_attrs.get(
            "model_scaffold_scales",
            scaffold_attrs.get("model_scaffold_scales"),
        )
        if scales is None and hasattr(model_abilities, "columns") and "scaffold_scale" in model_abilities.columns:
            scales = {
                str(model_id): float(scale)
                for model_id, scale in model_abilities["scaffold_scale"].items()
            }
        if scales is None:
            raise ValueError(
                "model_scaffold_scale theta_combine requires learned per-model scaffold scales"
            )
        return {"model_scaffold_scales": {str(k): float(v) for k, v in dict(scales).items()}}

    w_m = model_attrs.get("w_m", scaffold_attrs.get("w_m"))
    w_s = model_attrs.get("w_s", scaffold_attrs.get("w_s"))
    if w_m is None or w_s is None:
        raise ValueError("weighted_sum theta_combine requires learned w_m and w_s parameters")
    return {"w_m": float(w_m), "w_s": float(w_s)}


def attach_theta_combine_weights_from_path(
    model_abilities,
    scaffold_abilities,
    weights_dir,
    *,
    combine: str,
) -> None:
    """Attach persisted theta-combination parameters to ability DataFrame attrs."""
    combine_norm = normalize_theta_combine(combine)
    if combine_norm not in {"weighted_sum", "model_scaffold_scale"}:
        return
    if combine_norm == "model_scaffold_scale":
        scales_path = Path(weights_dir) / "theta_combine_model_scales.json"
        if scales_path.exists():
            with open(scales_path, "r") as f:
                scales = json.load(f)
        elif hasattr(model_abilities, "columns") and "scaffold_scale" in model_abilities.columns:
            scales = {
                str(model_id): float(scale)
                for model_id, scale in model_abilities["scaffold_scale"].items()
            }
        else:
            raise RuntimeError(f"Missing model_scaffold_scale parameters at {scales_path}")
        if not isinstance(scales, dict) or not scales:
            raise RuntimeError(f"Malformed model_scaffold_scale parameters at {scales_path}")
        scale_map = {str(k): float(v) for k, v in scales.items()}
        if hasattr(model_abilities, "columns") and "scaffold_scale" not in model_abilities.columns:
            model_abilities["scaffold_scale"] = [
                scale_map[str(model_id)] for model_id in model_abilities.index
            ]
        for df in (model_abilities, scaffold_abilities):
            df.attrs["model_scaffold_scales"] = scale_map
        return

    weights_path = Path(weights_dir) / "theta_combine_weights.json"
    if not weights_path.exists():
        raise RuntimeError(f"Missing weighted_sum parameters at {weights_path}")
    with open(weights_path, "r") as f:
        weights = json.load(f)
    if "w_m" not in weights or "w_s" not in weights:
        raise RuntimeError(f"Malformed weighted_sum parameters at {weights_path}")
    for df in (model_abilities, scaffold_abilities):
        df.attrs["w_m"] = float(weights["w_m"])
        df.attrs["w_s"] = float(weights["w_s"])


def combine_theta(
    theta_model: float,
    theta_scaffold: float,
    *,
    combine: str,
    w_m=None,
    w_s=None,
    model_id=None,
    model_scaffold_scales=None,
) -> float:
    """Combine model and scaffold theta values into an agent-level ability."""
    model = float(theta_model)
    scaffold = float(theta_scaffold)
    combine_norm = normalize_theta_combine(combine)
    sign = 1.0 if (model + scaffold) >= 0.0 else -1.0

    if combine_norm == "sum":
        return model + scaffold
    if combine_norm == "weighted_sum":
        if w_m is None or w_s is None:
            raise ValueError("weighted_sum theta_combine requires w_m and w_s")
        model_weight = float(w_m)
        scaffold_weight = float(w_s)
        denominator = model_weight + scaffold_weight
        if denominator == 0.0:
            raise ValueError("Cannot compute weighted_sum theta_combine when w_m + w_s == 0")
        return (model_weight / denominator) * model + (scaffold_weight / denominator) * scaffold
    if combine_norm == "model_scaffold_scale":
        if model_id is None:
            raise ValueError("model_scaffold_scale theta_combine requires model_id")
        if model_scaffold_scales is None:
            raise ValueError("model_scaffold_scale theta_combine requires model_scaffold_scales")
        key = str(model_id)
        if key not in model_scaffold_scales:
            raise ValueError(f"Missing scaffold scale for model {key!r}")
        return model + float(model_scaffold_scales[key]) * scaffold
    if combine_norm == "mean":
        return (model + scaffold) / 2.0
    if combine_norm == "l2":
        return sign * ((model**2 + scaffold**2) ** 0.5)
    if combine_norm == "harmonic":
        denominator = model + scaffold
        if denominator == 0.0:
            raise ValueError(
                "Cannot compute harmonic theta_combine when "
                f"theta_model + theta_scaffold == 0 ({model=}, {scaffold=})"
            )
        return sign * abs(2.0 * model * scaffold / denominator)
    if combine_norm == "product":
        return sign * abs(model * scaffold)
    if combine_norm == "min":
        return min(model, scaffold)
    if combine_norm == "max":
        return max(model, scaffold)
    raise ValueError(f"Unknown theta_combine={combine!r}")


def combine_theta_tensor(
    theta_model,
    theta_scaffold,
    *,
    combine: str,
    w_m=None,
    w_s=None,
    model_idx=None,
    model_scaffold_scale=None,
):
    """Torch equivalent of `combine_theta` for IRT likelihoods."""
    combine_norm = normalize_theta_combine(combine)
    sign = ((theta_model + theta_scaffold) >= 0).to(theta_model.dtype) * 2 - 1
    if combine_norm == "sum":
        return theta_model + theta_scaffold
    if combine_norm == "weighted_sum":
        if w_m is None or w_s is None:
            raise ValueError("weighted_sum theta_combine requires w_m and w_s")
        return (w_m / (w_m + w_s)) * theta_model + (w_s / (w_m + w_s)) * theta_scaffold
    if combine_norm == "model_scaffold_scale":
        if model_idx is None or model_scaffold_scale is None:
            raise ValueError("model_scaffold_scale theta_combine requires model_idx and model_scaffold_scale")
        scale = model_scaffold_scale[model_idx]
        if scale.ndim < theta_scaffold.ndim:
            scale = scale.unsqueeze(-1)
        return theta_model + scale * theta_scaffold
    if combine_norm == "mean":
        return (theta_model + theta_scaffold) / 2.0
    if combine_norm == "l2":
        return sign * (theta_model**2 + theta_scaffold**2).sqrt()
    if combine_norm == "harmonic":
        denominator = theta_model + theta_scaffold
        return sign * (2.0 * theta_model * theta_scaffold / denominator).abs()
    if combine_norm == "product":
        return sign * (theta_model * theta_scaffold).abs()
    if combine_norm == "min":
        return theta_model.minimum(theta_scaffold)
    if combine_norm == "max":
        return theta_model.maximum(theta_scaffold)
    raise ValueError(f"Unknown theta_combine={combine!r}")

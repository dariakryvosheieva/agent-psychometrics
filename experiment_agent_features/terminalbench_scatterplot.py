import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_ABILITIES = Path(
    "data/terminalbench/1d_1pl/model_abilities.csv"
)
AGENT_ABILITIES = Path("data/terminalbench/irt_oracle/abilities.csv")
AGENT_SPLITS = Path("data/terminalbench/1d_1pl/agent_splits.csv")

OUT = Path("data/terminalbench_scatterplot.png")


def read_theta(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    with path.open(newline="") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if not row or all(not c.strip() for c in row):
                continue
            out[str(row[0])] = float(row[1])
    return out


def read_terminus_subject_to_model(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open(newline="") as f:
        r = csv.DictReader(f)
        required = {"agent", "model", "scaffold"}
        if r.fieldnames is None or not required.issubset(set(r.fieldnames)):
            raise ValueError(
                f"{path} must include columns {sorted(required)}; got {r.fieldnames}"
            )
        for row in r:
            if row["scaffold"] != "Terminus 2":
                continue
            subject_id = row["agent"].strip()
            model_id = row["model"].strip()
            if not subject_id or not model_id:
                raise ValueError(
                    f"Found empty Terminus 2 mapping row in {path}: {row}"
                )
            previous = out.get(subject_id)
            if previous is not None and previous != model_id:
                raise ValueError(
                    f"Conflicting model mapping for {subject_id}: {previous} vs {model_id}"
                )
            out[subject_id] = model_id
    if not out:
        raise ValueError(f"No Terminus 2 rows found in {path}")
    return out


def pearsonr(xs, ys) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def ols_fit(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        return float("nan"), float("nan")
    b = sxy / sxx
    a = my - b * mx
    return a, b


model_theta = read_theta(MODEL_ABILITIES)
agent_theta = read_theta(AGENT_ABILITIES)
subject_to_model = read_terminus_subject_to_model(AGENT_SPLITS)

points: list[tuple[float, float, str, str]] = []
missing_subject_ids: list[str] = []
missing_model_ids: list[tuple[str, str]] = []
for sid, y in agent_theta.items():
    mid = subject_to_model.get(sid)
    if mid is None:
        missing_subject_ids.append(sid)
        continue
    x = model_theta.get(mid)
    if x is None:
        missing_model_ids.append((sid, mid))
        continue
    points.append((x, y, mid, sid))

if missing_subject_ids:
    missing_text = ", ".join(sorted(missing_subject_ids))
    raise ValueError(
        "Missing Terminus 2 subject->model mappings in agent_splits.csv for: "
        f"{missing_text}"
    )
if missing_model_ids:
    missing_text = ", ".join(f"{sid}->{mid}" for sid, mid in sorted(missing_model_ids))
    raise ValueError(
        "Missing model abilities in model_abilities.csv for mapped Terminus 2 subjects: "
        f"{missing_text}"
    )
if not points:
    raise ValueError("No points to plot after joining Terminus 2 and model abilities")

points.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
xs = [p[0] for p in points]
ys = [p[1] for p in points]

r = pearsonr(xs, ys)
a, b = ols_fit(xs, ys) if xs else (float("nan"), float("nan"))

plt.figure(figsize=(9.4, 7.8), dpi=210)
plt.scatter(xs, ys, s=26, alpha=0.82)

all_vals = xs + ys
if all_vals:
    lo = min(all_vals)
    hi = max(all_vals)
    pad = (hi - lo) * 0.06 if hi > lo else 1.0
    lo -= pad
    hi += pad
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)

    if not (math.isnan(a) or math.isnan(b)):
        fx = [lo, hi]
        fy = [a + b * lo, a + b * hi]
        plt.plot(
            fx,
            fy,
            linestyle="-",
            linewidth=1.2,
            alpha=0.8,
            label=f"fit: y={a:.2f}+{b:.2f}x",
        )

for i, (x, y, mid, sid) in enumerate(points):
    dx = 5 if (i % 2 == 0) else -5
    dy = 5 if (i % 3 == 0) else (-5 if (i % 3 == 1) else 0)
    plt.annotate(
        mid,
        (x, y),
        textcoords="offset points",
        xytext=(dx, dy),
        ha="left" if dx >= 0 else "right",
        va="bottom" if dy >= 0 else "top",
        fontsize=6.0,
        alpha=0.9,
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.62),
    )

plt.xlim(left=-6)
plt.xlabel("LLM ability")
plt.ylabel("Terminus 2 agent ability")
plt.title(
    f"Terminal-Bench: Terminus 2 (standard IRT) vs full data (IRT-Agent)"
    f" – n={len(points)}, r={r:.3f}"
)
plt.grid(True, alpha=0.25)
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()

OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT)
print(str(OUT))


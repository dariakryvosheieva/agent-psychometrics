import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ABILITIES = Path('out/swe_bench_bash_only/abilities.csv')
ORACLE = Path('out/irt_agent_all_2/irt_model_scaffold_1pl/model_abilities.csv')
OUT = Path('out/swe_bench_bash_only/verified_bash_only_scatterplot.png')


def read_theta(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    with path.open(newline='') as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if not row or all(not c.strip() for c in row):
                continue
            out[str(row[0])] = float(row[1])
    return out


def pearsonr(xs, ys) -> float:
    n = len(xs)
    if n < 2:
        return float('nan')
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx <= 0 or syy <= 0:
        return float('nan')
    return sxy / math.sqrt(sxx * syy)


def ols_fit(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        return float('nan'), float('nan')
    b = sxy / sxx
    a = my - b * mx
    return a, b


swe = read_theta(ABILITIES)
oracle = read_theta(ORACLE)
models = sorted(set(swe) & set(oracle))
xs = [oracle[m] for m in models]
ys = [swe[m] for m in models]

r = pearsonr(xs, ys)
a, b = ols_fit(xs, ys)

plt.figure(figsize=(9.0, 7.6), dpi=210)
plt.scatter(xs, ys, s=28, alpha=0.85)

all_vals = xs + ys
if all_vals:
    lo = min(all_vals)
    hi = max(all_vals)
    pad = (hi - lo) * 0.06 if hi > lo else 1.0
    lo -= pad
    hi += pad
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)

    # y=x reference
    plt.plot([lo, hi], [lo, hi], linestyle='--', linewidth=1.0, alpha=0.55, label='y=x')

    # best-fit line
    if not (math.isnan(a) or math.isnan(b)):
        fx = [lo, hi]
        fy = [a + b * lo, a + b * hi]
        plt.plot(fx, fy, linestyle='-', linewidth=1.2, alpha=0.8, label=f'fit: y={a:.2f}+{b:.2f}x')

# labels
for i, m in enumerate(models):
    x = oracle[m]
    y = swe[m]
    dx = 5 if (i % 2 == 0) else -5
    dy = 5 if (i % 3 == 0) else (-5 if (i % 3 == 1) else 0)
    plt.annotate(
        m,
        (x, y),
        textcoords='offset points',
        xytext=(dx, dy),
        ha='left' if dx >= 0 else 'right',
        va='bottom' if dy >= 0 else 'top',
        fontsize=6.5,
        alpha=0.9,
        bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.65),
    )

plt.xlabel('Oracle IRT ability (theta)')
plt.ylabel('SWE-bench bash-only IRT ability (theta)')
plt.title(f'IRT abilities: bash-only vs oracle (n={len(models)}, r={r:.3f})')
plt.grid(True, alpha=0.25)
plt.legend(loc='lower right', fontsize=8)
plt.tight_layout()

OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT)
print(str(OUT))
#!/usr/bin/env python
"""Does the pre-injection scan PREDICT the gain, or only correlate with it?

The library sweep shows gain falling as the prior rises. Correlation is not
the claim worth making: the operational proposition is that a scan costing
minutes tells you what a ten-GPU-hour injection will return, and that is a
prediction, which has an error bar and can be wrong.

With this few targets a train/test split would be a coin flip, so every model
is scored by leave-one-out cross-validation: fit on n-1 targets, predict the
held-out one, repeat. A predictor earns its keep only by beating the null
model that ignores the scan entirely and predicts the mean gain.

  python scripts/threshold_predictor.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# (target, held fraction before injection %, gain in held-fraction points, n probes)
# All Qwen3-4B-Base 4-bit, one recipe, baseline and served measured on the
# same card. Source: out/exp99_{base,served}_*.json on the 4090.
DATA = [
    ("httpx",       1.3,  11.3, 110),
    ("json",        5.2,  10.5,  21),
    ("typer",       9.5,  13.6, 173),
    ("dataclasses", 10.1, 13.8,  16),
    ("rich",        11.7, 12.7, 293),
    ("click",       13.1, 13.1, 156),
    ("pydantic",    13.1, 10.9, 178),
    ("argparse",    14.7, 11.6,  29),
    ("os",          16.2, 17.8,  46),
    ("requests",    18.6,  3.2,  74),
    ("shutil",      23.5, -5.1,  45),
    ("typing",      25.7,  8.3,  14),
    ("itertools",   43.8, -7.0,  17),
]


def fit_linear(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return (lambda _x: my)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    return (lambda x: a + b * x)


def fit_step(xs, ys):
    """A two-level step: below a knee, one constant; above it, another.

    The knee is chosen on the training fold by minimizing squared error over
    every midpoint between consecutive observed priors, so the shape is fitted
    rather than assumed.
    """
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    sx = [xs[i] for i in order]
    sy = [ys[i] for i in order]
    best = None
    for k in range(1, len(sx)):
        knee = (sx[k - 1] + sx[k]) / 2
        lo = [y for x, y in zip(sx, sy) if x < knee]
        hi = [y for x, y in zip(sx, sy) if x >= knee]
        if not lo or not hi:
            continue
        mlo, mhi = sum(lo) / len(lo), sum(hi) / len(hi)
        sse = sum((y - mlo) ** 2 for y in lo) + sum((y - mhi) ** 2 for y in hi)
        if best is None or sse < best[0]:
            best = (sse, knee, mlo, mhi)
    if best is None:
        m = sum(sy) / len(sy)
        return (lambda _x: m), None
    _, knee, mlo, mhi = best
    return (lambda x: mlo if x < knee else mhi), knee


def loocv(rows, fitter):
    """Mean absolute error of one-target-held-out prediction."""
    errs, knees = [], []
    for i in range(len(rows)):
        tr = [r for j, r in enumerate(rows) if j != i]
        out = fitter([r[1] for r in tr], [r[2] for r in tr])
        f, knee = out if isinstance(out, tuple) else (out, None)
        if knee is not None:
            knees.append(knee)
        errs.append(abs(f(rows[i][1]) - rows[i][2]))
    return sum(errs) / len(errs), errs, knees


def main():
    rows = DATA
    ys = [r[2] for r in rows]
    mean_y = sum(ys) / len(ys)

    print(f"{len(rows)} targets, gain from {min(ys):+.1f} to {max(ys):+.1f}, "
          f"mean {mean_y:+.1f}\n")

    # the null model: ignore the scan, predict the mean of the training fold
    def fit_mean(xs, ys_):
        m = sum(ys_) / len(ys_)
        return (lambda _x: m)

    models = [("null (ignore the scan)", fit_mean),
              ("linear in the prior", fit_linear),
              ("two-level step, knee fitted", fit_step)]

    results = {}
    for name, fitter in models:
        mae, errs, knees = loocv(rows, fitter)
        results[name] = (mae, errs, knees)
        extra = ""
        if knees:
            extra = (f"   knee {min(knees):.1f}-{max(knees):.1f} "
                     f"across folds")
        print(f"{name:30s} LOOCV MAE {mae:5.2f} points{extra}")

    null_mae = results["null (ignore the scan)"][0]
    print()
    for name in ("linear in the prior", "two-level step, knee fitted"):
        mae = results[name][0]
        rel = (null_mae - mae) / null_mae * 100
        verdict = "beats" if mae < null_mae else "LOSES TO"
        print(f"{name:30s} {verdict} the null by {rel:+.0f}%")

    print("\nper-target held-out error, step model:")
    _, errs, _ = loocv(rows, fit_step)
    for r, e in sorted(zip(rows, errs), key=lambda t: -t[1]):
        print(f"  {r[0]:12s} prior {r[1]:5.1f}  actual {r[2]:+6.1f}  "
              f"|error| {e:5.1f}   (n={r[3]})")


if __name__ == "__main__":
    main()

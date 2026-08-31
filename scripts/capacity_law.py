#!/usr/bin/env python
"""Which knob sets the stored information, and can the next point be
predicted before it is measured?

Reads results/bits_<run>.json (eval_bits.py) and fits, on the points that
exist, three one-knob laws at 10k facts on Qwen3-1.7B:

  rank      B = a * P^gamma, P = trainable parameters (r = 16, 32, 64, 128)
            -> predicts r = 256
  width     B = a * w^delta,  w = fill_frac (1/4, 1/2, 1)
  facts     B = a * n^alpha,  n = 3k, 10k at r = 64 -> predicts 30k

and prints the prediction for every held-out point, with the measured
value beside it when that file has arrived. The fits are deliberately
simple power laws: the question is which exponent is near zero (the knob
does not matter) and which is near one (the knob is the budget).

  python scripts/capacity_law.py [--fig paper/figs/capacity_bits.pdf]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

R = Path("results")


def load(stem):
    p = R / f"bits_{stem}.json"
    return json.loads(p.read_text()) if p.exists() else None


def bits(stem):
    d = load(stem)
    return None if d is None else d["stored_bits_total"]


def fit_power(xs, ys):
    """least squares on log-log; returns (a, exponent)"""
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    n = len(xs)
    mx, my = sum(lx) / n, sum(ly) / n
    sxx = sum((x - mx) ** 2 for x in lx)
    if sxx == 0:
        return math.exp(my), 0.0
    g = sum((x - mx) * (y - my) for x, y in zip(lx, ly)) / sxx
    return math.exp(my - g * mx), g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", default=None)
    args = ap.parse_args()
    report = {}

    # rank at 10k: params scale with r
    rank_pts = [(r, f"cap10000_r{r}" if r != 64 else "cap10000_r64_s1")
                for r in (16, 32, 64, 128)]
    pts = [(load(s)["n_trainable"], bits(s), r) for r, s in rank_pts
           if load(s)]
    if len(pts) >= 2:
        a, g = fit_power([p for p, _, _ in pts], [b for _, b, _ in pts])
        held = load("cap10000_r256")
        pred = a * (held["n_trainable"] if held else 4 * pts[-1][0]) ** g
        report["rank"] = dict(points={r: b for _, b, r in pts}, exponent=g,
                              predicted_r256=pred,
                              measured_r256=bits("cap10000_r256"))
        print(f"[rank]  B ~ P^{g:.2f} on r={[r for _, _, r in pts]}; "
              f"r=256 predicted {pred:.3e}, measured "
              f"{bits('cap10000_r256')}")

    # width at 10k, r = 64
    wpts = [(w, bits(s)) for w, s in ((0.25, "cap10000_w25"),
                                      (0.5, "cap10000_w5"),
                                      (1.0, "cap10000_r64_s1")) if bits(s)]
    if len(wpts) >= 2:
        a, d = fit_power([w for w, _ in wpts], [b for _, b in wpts])
        report["width"] = dict(points=dict(wpts), exponent=d)
        print(f"[width] B ~ w^{d:.2f} on w={[w for w, _ in wpts]}")

    # facts at r = 64
    npts = [(n, bits(s)) for n, s in ((3000, "cap3000_r64_s1"),
                                      (10000, "cap10000_r64_s1")) if bits(s)]
    if len(npts) >= 2:
        a, al = fit_power([n for n, _ in npts], [b for _, b in npts])
        pred = a * 30000 ** al
        report["facts"] = dict(points=dict(npts), exponent=al,
                               predicted_30k=pred,
                               measured_30k=bits("cap30000_r64"))
        print(f"[facts] B ~ n^{al:.2f}; 30k predicted {pred:.3e}, measured "
              f"{bits('cap30000_r64')}")

    # exposures and size, reported as ratios to the r64 seed-1 point
    base = bits("cap10000_r64_s1")
    for name, stem in (("48 epochs", "cap10000_ep48"),
                       ("Qwen3-4B", "cap10000_4b"),
                       ("dense full-rank", "cap10000_dense")):
        b = bits(stem)
        if b and base:
            report[name] = dict(bits=b, ratio_to_r64=b / base)
            print(f"[{name}] {b:.3e} bits = {b / base:.2f}x the r64 point")

    if args.fig and pts:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 3, figsize=(9, 2.8))
        axs[0].loglog([p for p, _, _ in pts], [b for _, b, _ in pts], "o-")
        axs[0].set_xlabel("trainable parameters"); axs[0].set_ylabel("bits stored")
        axs[0].set_title("rank, 10k facts")
        if len(wpts) >= 2:
            axs[1].loglog([w for w, _ in wpts], [b for _, b in wpts], "s-")
            axs[1].set_xlabel("fill width (share of half-width)")
            axs[1].set_title("cell width, 10k facts")
        if len(npts) >= 2:
            axs[2].loglog([n for n, _ in npts], [b for _, b in npts], "^-")
            axs[2].set_xlabel("facts presented"); axs[2].set_title("r = 64")
        fig.tight_layout()
        Path(args.fig).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.fig)
        print(f"wrote {args.fig}")
    Path("results/capacity_law.json").write_text(json.dumps(report, indent=2))
    if not report:
        print("no bits files yet")


if __name__ == "__main__":
    main()

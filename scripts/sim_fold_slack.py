#!/usr/bin/env python
"""Is the slack in the folding law a property of the NF4 cells, or of folding?

Proposition 6 gives r' >= r(1 - |t|) per fold, and the measured ratios
(0.82-0.83) sit well above the naive 1 - E|t| (0.48-0.56). An earlier draft
attributed the gap to the asymmetry of NF4 cells. This simulation folds the
same number of weights, driven by the E|tanh| values logged in the archived
sequence (results/exp29_seq_etanh.json), on two grids: the real NF4 cells,
with codes drawn from a Gaussian weight distribution, and equal-width cells
centred on their anchors, which carry no geometric information at all. If
the two agree from the second fold on, the slack belongs to the folding
operator -- min(a, b) applied to random-sign displacements from an
off-centre position -- and not to the grid.

Three shapes of |t| with the same mean are run, because only the mean is
archived and the ratio depends on the distribution.

  python scripts/sim_fold_slack.py --out results/sim_fold_slack.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cellfill.bins import NF4_LEVELS, normalized_cell_edges  # noqa: E402


def draw_t(shape: str, e: float, n: int, g: torch.Generator) -> torch.Tensor:
    """|t| with mean e, then a random sign; every draw satisfies |t| < 1."""
    if shape == "uniform":
        t = torch.rand(n, generator=g) * 2 * e
    elif shape == "halfnorm":
        t = torch.randn(n, generator=g).abs() * e * math.sqrt(math.pi / 2)
    else:  # tanh-shaped: tanh of a Gaussian, rescaled to the target mean
        t = torch.tanh(torch.randn(n, generator=g) * 0.5).abs()
        t = t * e / t.mean()
    t = t.clamp(max=0.999)
    return t * torch.sign(torch.randn(n, generator=g))


def fold(lo, hi, a, shape, means, g):
    rooms = [torch.minimum(a - lo, hi - a).mean().item()]
    for e in means:
        r = torch.minimum(a - lo, hi - a)
        a = a + r * draw_t(shape, e, a.numel(), g)
        rooms.append(torch.minimum(a - lo, hi - a).mean().item())
    return rooms, [rooms[i + 1] / rooms[i] for i in range(len(rooms) - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2_000_000)
    ap.add_argument("--margin", type=float, default=0.01)
    ap.add_argument("--seq", default="results/exp29_seq_etanh.json",
                    help="archived sequence whose mean_abs_tanh drives the folds")
    ap.add_argument("--out", default="results/sim_fold_slack.json")
    args = ap.parse_args()

    hist = json.loads(Path(args.seq).read_text())["history"]
    means = [h["mean_abs_tanh"] for h in hist]
    g = torch.Generator().manual_seed(0)
    lo_e, hi_e = normalized_cell_edges(True)
    w = hi_e - lo_e
    lo_e, hi_e = lo_e + args.margin * w, hi_e - args.margin * w
    L = NF4_LEVELS
    codes = torch.argmin((torch.randn(args.n, generator=g)[:, None]
                          - L[None, :]).abs(), dim=1)
    nf4 = (lo_e[codes], hi_e[codes], L[codes])
    sym = (torch.full((args.n,), -0.5), torch.full((args.n,), 0.5),
           torch.zeros(args.n))

    out = dict(n=args.n, margin=args.margin, mean_abs_tanh=means,
               naive_one_minus_E=[1 - e for e in means], shapes={})
    for shape in ("uniform", "halfnorm", "tanhlike"):
        _, r_nf4 = fold(*nf4, shape, means, torch.Generator().manual_seed(1))
        _, r_sym = fold(*sym, shape, means, torch.Generator().manual_seed(1))
        out["shapes"][shape] = dict(nf4=r_nf4, symmetric=r_sym)
        print(f"{shape:9s} NF4 {[f'{x:.4f}' for x in r_nf4]}  "
              f"symmetric {[f'{x:.4f}' for x in r_sym]}")
    print("naive 1-E|t|:", [f"{x:.4f}" for x in out["naive_one_minus_E"]])
    later = [x for s in out["shapes"].values() for k in ("nf4", "symmetric")
             for x in s[k][1:]]
    out["band_from_second_fold"] = [min(later), max(later)]
    first = {s: (v["nf4"][0], v["symmetric"][0])
             for s, v in out["shapes"].items()}
    out["first_fold_nf4_vs_symmetric"] = first
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"band from the second fold on: {min(later):.3f}-{max(later):.3f}; "
          f"wrote {args.out}")


if __name__ == "__main__":
    main()

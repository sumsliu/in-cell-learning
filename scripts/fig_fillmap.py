#!/usr/bin/env python
"""Where in the network did each injection land?

One panel per corpus: layer (x) by module type (y), colour = share of that
run's total fill energy. The fill energy of a module is sum((M * t)^2) over
its weights, recorded by exp5_qil --fill-stats; normalising per run makes the
panels comparable across corpora of different sizes.

  python scripts/fig_fillmap.py --out paper/figs/fillmap.pdf
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

R = Path("results")
TYPES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
         "down_proj"]
PANELS = [("all real (8 domains)", "fillstats_1p7b_all"),
          ("medicine", "fillstats_1p7b_dom_medicine"),
          ("lottery", "fillstats_1p7b_dom_lottery"),
          ("awards", "fillstats_1p7b_dom_awards"),
          ("launches", "fillstats_1p7b_dom_technology"),
          ("science", "fillstats_1p7b_dom_science")]


def grid(stats):
    layers = sorted({int(m.group(1)) for n in stats
                     if (m := re.search(r"layers\.(\d+)\.", n))})
    tot = sum(v["fill_energy"] for v in stats.values())
    g = [[0.0] * len(layers) for _ in TYPES]
    for name, v in stats.items():
        m = re.search(r"layers\.(\d+)\.", name)
        if not m:
            continue
        li = layers.index(int(m.group(1)))
        for ti, t in enumerate(TYPES):
            if name.endswith(t):
                g[ti][li] = v["fill_energy"] / tot if tot else 0
    return layers, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/figs/fillmap.pdf")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [(lab, json.loads((R / f"{stem}.json").read_text()))
              for lab, stem in PANELS if (R / f"{stem}.json").exists()]
    if not panels:
        raise SystemExit("no fillstats_*.json in results/")
    fig, axes = plt.subplots(len(panels), 1, figsize=(7.2, 1.15 * len(panels) + 0.8),
                             sharex=True)
    if len(panels) == 1:
        axes = [axes]
    vmax = 0
    grids = []
    for lab, stats in panels:
        layers, g = grid(stats)
        grids.append((lab, layers, g))
        vmax = max(vmax, max(max(r) for r in g))
    for ax, (lab, layers, g) in zip(axes, grids):
        im = ax.imshow(g, aspect="auto", cmap="magma", vmin=0, vmax=vmax,
                       interpolation="nearest")
        ax.set_yticks(range(len(TYPES)))
        ax.set_yticklabels([t.replace("_proj", "") for t in TYPES], fontsize=6)
        ax.set_ylabel(lab, fontsize=7)
        ax.tick_params(axis="x", labelsize=6)
    axes[-1].set_xticks(range(0, len(layers), 4))
    axes[-1].set_xticklabels([str(layers[i]) for i in range(0, len(layers), 4)])
    axes[-1].set_xlabel("layer", fontsize=8)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("share of fill energy", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out} ({len(panels)} panels)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""The recall-damage curve: what a deployment actually has to choose between.

The paper reports one operating point per configuration -- 24 epochs, all the
recall, all the damage. A reader deciding whether to ship this wants the
curve, because the endpoint is a choice and not a property: the trajectory
recorded by exp5_qil --eval-every gives recall against relative cross-domain
damage at every few epochs, and the knee is where the two stop trading
evenly.

Damage is quoted as Delta = log(ppl_served / ppl_anchor), the relative
measure the scale section shows is not confounded by the base model's own
perplexity.

  python scripts/fig_tradeoff.py --out paper/figs/tradeoff.pdf
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

R = Path("results")

# stem -> (label, colour index). Any that are missing are skipped.
SERIES = [
    ("exp75_traj_wikitext", "rehearsal: WikiText (default)"),
    ("exp75_traj_mixed", "rehearsal: mixed"),
    ("exp75_traj_pile", "rehearsal: Pile"),
]


def series(stem):
    p = R / f"{stem}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    tr = d.get("trajectory")
    if not tr:
        return None
    anchor_x = (d.get("ppl_lambada") or {}).get("anchor_fp32")
    anchor_w = (d.get("ppl") or {}).get("anchor_fp32")
    pts = []
    for row in tr:
        if not row.get("ppl_lambada") or not anchor_x:
            continue
        pts.append(dict(epoch=row["epoch"], recall=row["recall"],
                        dx=math.log(row["ppl_lambada"] / anchor_x),
                        dw=(math.log(row["ppl"] / anchor_w)
                            if anchor_w else None),
                        sat=row.get("saturation")))
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/figs/tradeoff.pdf")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    got = [(lab, series(stem)) for stem, lab in SERIES]
    got = [(lab, p) for lab, p in got if p]
    if not got:
        raise SystemExit("no trajectory in results/ (run with --eval-every)")

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for lab, pts in got:
        x = [p["dx"] for p in pts]
        y = [100 * p["recall"] for p in pts]
        axes[0].plot(x, y, "o-", ms=3.5, lw=1.2, label=lab)
        for p in pts:
            if p["epoch"] % 8 == 0:
                axes[0].annotate(f"e{p['epoch']}", (p["dx"], 100 * p["recall"]),
                                 fontsize=6, xytext=(3, -6),
                                 textcoords="offset points")
        axes[1].plot([p["epoch"] for p in pts],
                     [100 * (p["sat"] or 0) for p in pts], "o-", ms=3.5,
                     lw=1.2, label=lab)
    axes[0].set_xlabel(r"cross-domain damage $\Delta=\log(\mathrm{ppl}/\mathrm{ppl}_0)$",
                       fontsize=8)
    axes[0].set_ylabel("recall (%)", fontsize=8)
    axes[1].set_xlabel("epoch", fontsize=8)
    axes[1].set_ylabel(r"saturated coordinates ($|t|>0.99$, %)", fontsize=8)
    for a in axes:
        a.tick_params(labelsize=7)
        a.grid(alpha=0.25, lw=0.5)
    axes[0].legend(fontsize=6.5, frameon=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out} ({len(got)} series)")


if __name__ == "__main__":
    main()

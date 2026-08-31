#!/usr/bin/env python
"""Figure for the federation (paper D, f0.pdf; also used by paper B).

a  pooled recall after each round, all areas and per area
b  each node's own recall before the first merge vs the pooled recall of
   its area after the last round
c  bytes that crossed the network (fills) vs bytes that stayed home (data)

  python scripts/fig_fed.py --run clinic8b --out paper/npj/figs/f0.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
AREAS = [("clinic_oncology", "oncology"), ("clinic_rare_genetic", "rare/genetic"),
         ("clinic_other", "other"), ("clinic_cardiometabolic", "cardiometabolic"),
         ("clinic_immuno_dermatology", "immuno/derm"), ("clinic_infectious", "infectious"),
         ("clinic_neuro_psychiatry", "neuro/psych")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="clinic8b")
    ap.add_argument("--out", default="paper/npj/figs/f0.pdf")
    args = ap.parse_args()
    d = json.loads((ROOT / "results" / f"fed_{args.run}.json").read_text())
    hist = d["history"]
    finals = {}
    for k in range(len(d["hosts"])):
        f = ROOT / "fed" / args.run / f"final_{k}.json"
        if f.exists():
            fk = json.loads(f.read_text())
            finals[fk["shard"]] = fk
    own0 = {}
    o = ROOT / "results" / f"fed_{args.run}_own0.json"
    if o.exists():
        own0 = json.loads(o.read_text())
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3), gridspec_kw=dict(width_ratios=[1.3, 1, 0.8]))
    rounds = [h["round"] + 1 for h in hist]
    for dom, label in AREAS:
        ax[0].plot(rounds, [100 * h["by_shard"].get(dom, 0) for h in hist], "-", lw=0.8, alpha=0.6, label=label)
    ax[0].plot(rounds, [100 * h["pooled"] for h in hist], "k-o", lw=1.6, ms=3, label="all areas")
    ax[0].set_xlabel("round"); ax[0].set_ylabel("pooled recall (%)"); ax[0].set_ylim(0, 100)
    ax[0].set_xticks(rounds); ax[0].legend(fontsize=5.5, ncol=2, frameon=False, loc="upper left")
    ax[0].set_title("a  one release, seven writers, six rounds", loc="left", fontsize=8)
    own = [100 * (finals.get(dom, {}).get("history", [{}])[0].get("own_in_place")
                  or own0.get(dom, 0)) for dom, _ in AREAS]
    pooled = [100 * hist[-1]["by_shard"].get(dom, 0) for dom, _ in AREAS]
    x = range(len(AREAS))
    ax[1].bar([i - 0.2 for i in x], own, 0.4, color="#bbbbbb", label="own area, own fill (round 1)")
    ax[1].bar([i + 0.2 for i in x], pooled, 0.4, color="#1f77b4", label="pooled model (round 6)")
    ax[1].set_xticks(list(x)); ax[1].set_xticklabels([l for _, l in AREAS], rotation=45, ha="right", fontsize=6)
    ax[1].set_ylabel("recall (%)"); ax[1].set_ylim(0, 100); ax[1].legend(fontsize=5.5, frameon=False)
    ax[1].set_title("b  before and after pooling", loc="left", fontsize=8)
    fills = sorted((ROOT / "fed" / args.run).glob("round_*/fill_*.pt"))
    fill_mb = sum(f.stat().st_size for f in fills) / 1e6
    data_kb = sum(len(r["text"].encode()) for r in json.loads((ROOT / "data" / "clinic_train.json").read_text())) / 1e3
    ax[2].bar([0, 1], [fill_mb, data_kb / 1e3], color=["#1f77b4", "#bbbbbb"])
    ax[2].set_xticks([0, 1]); ax[2].set_xticklabels([f"fills moved\n({len(fills)} files)", "sentences\n(stayed home)"], fontsize=6)
    ax[2].set_ylabel("MB"); ax[2].set_yscale("log")
    ax[2].set_title("c  what crossed the network", loc="left", fontsize=8)
    for i, v in enumerate([fill_mb, data_kb / 1e3]):
        ax[2].text(i, v * 1.3, f"{v:.2f} MB" if v < 1 else f"{v:.0f} MB", ha="center", fontsize=6)
    fig.tight_layout()
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"[fig] {out}")


if __name__ == "__main__":
    main()

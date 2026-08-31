#!/usr/bin/env python
"""Figures of the multi-writer paper, computed from the archived results.

  python scripts/fig_fusion.py   -> paper/natcomms/figs/f{1,2,3}.pdf
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.make_tables import load  # noqa: E402

OUT = Path("paper/natcomms/figs"); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7, "legend.fontsize": 6,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
BLUE, ORANGE, GREY, GREEN, RED = "#1f5fa8", "#e07b00", "#7f7f7f", "#2a9d57", "#c0392b"


def label(ax, s):
    ax.text(-0.2, 1.08, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def fused(stem, how):
    d = load(stem)
    return 100 * d["arms"][how]["recall"]["fused"] if d and how in d["arms"] else None


def own(stem):
    d = load(stem)
    return 100 * st.mean(p["recall_own"] for p in d["parties"]) if d else None


def rounds_final(stem):
    d = load(stem)
    return 100 * d["history"][-1]["recall"] if d else None


def fig1():
    """Where the knowledge goes: cross-talk, rules, and allocations."""
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4))
    # (a) cross-talk matrices: synthetic K=4 (full), synthetic inert K=2, real K=2 inert
    ax = axs[0]
    d = load("fuse_k4_full_rules")
    xt = d["crosstalk"] if d else None
    if xt:
        keys = sorted(xt)
        M = np.array([[xt[k][t] for t in keys] for k in keys])
        im = ax.imshow(M, cmap="Blues", vmin=0)
        ax.set_xticks(range(len(keys))); ax.set_yticks(range(len(keys)))
        ax.set_xticklabels([f"prompts {i}" for i in range(len(keys))], rotation=30)
        ax.set_yticklabels([f"fill {i}" for i in range(len(keys))])
        for i in range(len(keys)):
            for j in range(len(keys)):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=6,
                        color="white" if M[i, j] > M.max() / 2 else "black")
        ax.set_title("cross-talk (KL, nats): four synthetic writers")
    label(ax, "a")
    # (b) one-shot rules, K=2 synthetic, against the writers' own recall
    ax = axs[1]
    rules = [("average", "average"), ("clamp", "sum, clamp"), ("magnitude", "magnitude"),
             ("ties", "TIES"), ("dare", "DARE")]
    vals = [fused("fuse_k2_full_rules", r) for r, _ in rules]
    ax.bar(range(len(rules)), [v or 0 for v in vals], color=GREY)
    o = own("fuse_k2_full_rules")
    ax.axhline(o, color=BLUE, ls="--", lw=1); ax.text(0, o + 1.5, f"own recall before merge {o:.0f}%", color=BLUE, fontsize=6)
    for x, (r, lab) in enumerate(rules):
        extra = [("reserved sum", fused("fuse_k2_reserved", "sum")), ("disjoint matrices", fused("fuse_k2_partition", "sum"))]
    ax.bar([len(rules), len(rules) + 1], [fused("fuse_k2_reserved", "sum") or 0, fused("fuse_k2_partition", "sum") or 0], color=GREEN)
    ax.set_xticks(range(len(rules) + 2)); ax.set_xticklabels([l for _, l in rules] + ["reserved\nsum", "disjoint\nmatrices"], rotation=30, fontsize=6)
    ax.set_ylabel("facts kept after the merge (%)"); ax.set_ylim(0, 70)
    ax.set_title("one-shot merges, two synthetic writers")
    label(ax, "b")
    # (c) inert writers: own vs fused, synthetic and real, K=2/4/8
    ax = axs[2]
    bars = [("synthetic K=2", own("fuse_k2_inert_neutral"), fused("fuse_k2_inert_neutral", "sum")),
            ("real K=2", own("fuse_real_k2_inert"), fused("fuse_real_k2_inert", "sum")),
            ("real K=4", own("fuse_real_k4_inert"), fused("fuse_real_k4_inert", "sum")),
            ("real K=8", own("fuse_real_k8_inert"), fused("fuse_real_k8_inert", "sum"))]
    x = np.arange(len(bars))
    ax.bar(x - 0.18, [b[1] or 0 for b in bars], 0.36, color=GREY, label="own, before merge")
    ax.bar(x + 0.18, [b[2] or 0 for b in bars], 0.36, color=BLUE, label="after exact sum")
    ax.set_xticks(x); ax.set_xticklabels([b[0] for b in bars], fontsize=6)
    ax.set_ylabel("recall (%)"); ax.set_title("inert writers, summed exactly")
    ax.legend(frameon=False, loc="lower left", fontsize=5.5)
    label(ax, "c")
    fig.tight_layout(); fig.savefig(OUT / "f1.pdf"); plt.close(fig)


def fig2():
    """Rounds."""
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.5))
    ax = axs[0]
    for lab, stem, col, ls in (("synthetic K=2", "fuse_k2_rounds3", BLUE, "-"), ("synthetic K=4", "fuse_k4_rounds3", GREEN, "-"),
                               ("real K=2", "fuse_real_k2_rounds3", ORANGE, "-"), ("real K=4, 6 rounds", "fuse_real_k4_rounds6", RED, "-"),
                               ("real K=8, 6 rounds", "fuse_real_k8_rounds6", "purple", "-"),
                               ("real K=8, magnitude", "fuse_real_k8_rounds3_mag", "purple", ":")):
        d = load(stem)
        if d:
            ax.plot([h["round"] + 1 for h in d["history"]], [100 * h["recall"] for h in d["history"]], "o-" if ls == "-" else "s:", color=col, label=lab, ms=3)
    ax.set_xlabel("round"); ax.set_ylabel("merged recall (%)"); ax.set_title("fusion by rounds")
    ax.legend(frameon=False, fontsize=5.5); label(ax, "a")
    ax = axs[1]
    pts = [("K=2", fused("fuse_real_k2_full", "clamp"), fused("fuse_real_k2_inert", "sum"), rounds_final("fuse_real_k2_rounds3")),
           ("K=4", fused("fuse_real_k4_reserved", "sum"), fused("fuse_real_k4_inert", "sum"), rounds_final("fuse_real_k4_rounds6")),
           ("K=8", None, fused("fuse_real_k8_inert", "sum"), rounds_final("fuse_real_k8_rounds6") or rounds_final("fuse_real_k8_rounds3_mag"))]
    x = np.arange(len(pts)); w = 0.26
    ax.bar(x - w, [p[1] or 0 for p in pts], w, color=GREY, label="best one-shot, plain")
    ax.bar(x, [p[2] or 0 for p in pts], w, color=BLUE, label="inert, exact sum")
    ax.bar(x + w, [p[3] or 0 for p in pts], w, color=ORANGE, label="rounds")
    ax.set_xticks(x); ax.set_xticklabels([p[0] for p in pts]); ax.set_ylabel("facts kept (%)")
    ax.set_title("real corpus, writers by domain"); ax.legend(frameon=False); label(ax, "b")
    fig.tight_layout(); fig.savefig(OUT / "f2.pdf"); plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); print("wrote", sorted(p.name for p in OUT.glob("*.pdf")))

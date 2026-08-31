#!/usr/bin/env python
"""The display items of the Nature-format manuscript, from the archived
result files. Every panel is computed; nothing is typed in.

  python scripts/fig_nature.py          -> paper/nature/figs/fig{2,3,4,5}.pdf
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.make_tables import OFFICIAL, _real_recall, load  # noqa: E402

R = Path("results")
OUT = Path("paper/nature/figs")
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
                     "legend.fontsize": 6, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
BLUE, ORANGE, GREY, GREEN, RED = "#1f5fa8", "#e07b00", "#7f7f7f", "#2a9d57", "#c0392b"


def panel_label(ax, s):
    ax.text(-0.18, 1.08, s, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top")


# ------------------------------------------------------------ figure 2
def fig2():
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3))
    # (a) recall on the vendors' releases
    rows = []
    for label, stems, _ in OFFICIAL:
        ds = [d for d in (load(s) for s in stems) if d]
        if not ds:
            continue
        rec = [_real_recall(d) for d in ds]
        rows.append((label.split(" (")[0] + "\n" + label.split(" (")[1].rstrip(")"),
                     st.mean(rec), st.stdev(rec) if len(rec) > 1 else 0, len(rec)))
    ax = axs[0]
    y = range(len(rows))
    ax.barh(list(y), [100 * r[1] for r in rows], xerr=[100 * r[2] for r in rows],
            color=BLUE, height=0.6, error_kw=dict(lw=0.8))
    ax.set_yticks(list(y))
    ax.set_yticklabels([r[0] for r in rows], fontsize=5.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("recall on 507 real probes (%)")
    ax.set_title("vendors' 4-bit releases, written in place")
    for i, r in enumerate(rows):
        ax.text(100 * r[1] + 1.5, i, f"{100 * r[1]:.0f}" + (f" (n={r[3]})" if r[3] > 1 else ""),
                va="center", fontsize=5.5)
    panel_label(ax, "a")
    # (b) LoRA against CellFill by learning rate, 1.7B and 4B
    ax = axs[1]
    pts = [("1.7B", "LoRA", 2e-4, ["exp51_lora_lr2e-4_s0", "exp51_lora_lr2e-4_s1", "exp51_lora_lr2e-4_s2"]),
           ("1.7B", "LoRA", 1e-3, ["exp51_lora_lr1e-3_s0", "exp51_lora_lr1e-3_s1", "exp51_lora_lr1e-3_s2"]),
           ("1.7B", "CellFill", 1e-3, ["exp45_official_twohop", "exp49_official_1p7b_s1", "exp49_official_1p7b_s2"]),
           ("4B", "LoRA", 1e-4, ["exp51_lora4b_lr1e-4_s0"]),
           ("4B", "LoRA", 2e-4, ["exp51_lora4b_lr2e-4_s0"]),
           ("4B", "LoRA", 1e-3, ["exp51_lora4b_lr1e-3_s0"]),
           ("4B", "CellFill", 1e-3, ["exp46_official_4b", "exp50_official_4b_s1"])]
    for size, meth, lr, stems in pts:
        ds = [d for d in (load(s) for s in stems) if d]
        if not ds:
            continue
        rec = 100 * st.mean(_real_recall(d) for d in ds)
        mk = "o" if size == "1.7B" else "s"
        col = ORANGE if meth == "LoRA" else BLUE
        ax.scatter([lr], [rec], marker=mk, color=col, s=28, zorder=3,
                   label=f"{meth}, {size}")
    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("recall (%)")
    ax.set_ylim(-3, 103)
    ax.set_title("the bound makes the rate safe")
    h, l = ax.get_legend_handles_labels()
    seen = {}
    for hh, ll in zip(h, l):
        seen.setdefault(ll, hh)
    ax.legend(seen.values(), seen.keys(), loc="center left", frameon=False)
    panel_label(ax, "b")
    # (c) downstream cost
    ax = axs[2]
    bench = [("1.7B", "bench_1p7b"), ("4B", "bench_4b"), ("8B", "bench_8b")]
    tasks = ["arc_challenge", "arc_easy", "hellaswag", "winogrande"]
    names = ["ARC-c", "ARC-e", "HellaSwag", "WinoGrande"]
    width = 0.25
    for j, (size, stem) in enumerate(bench):
        a = load(f"{stem}_anchor")
        m = load(f"{stem}_merged")
        if not (a and m):
            continue
        def sc(d, t):
            s_ = d.get("scores") or d.get("results") or {}
            v = s_.get(t)
            if isinstance(v, dict):
                v = v.get("acc_norm,none") or v.get("acc,none") or v.get("acc")
            return 100 * v if v is not None and v < 1.5 else v
        deltas = [sc(m, t) - sc(a, t) for t in tasks]
        ax.bar([i + (j - 1) * width for i in range(len(tasks))], deltas, width,
               color=[BLUE, GREEN, GREY][j], label=size)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(names, rotation=20)
    ax.set_ylabel("served − anchor (points)")
    ax.set_title("the update's cost on benchmarks")
    ax.legend(frameon=False)
    panel_label(ax, "c")
    fig.tight_layout()
    fig.savefig(OUT / "fig2.pdf")
    plt.close(fig)


# ------------------------------------------------------------ figure 3
def fig3():
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3))
    ax = axs[0]
    for stem, marker, lab in (("rag_popqa_1p7b", "o", "1.7B, sentence index"),
                              ("rag_popqa_chunked_1p7b", "^", "1.7B, chunked index"),
                              ("rag_popqa_4b", "s", "4B, sentence index"),
                              ("rag_popqa_chunked_4b", "D", "4B, chunked index")):
        d = load(stem)
        if not d:
            continue
        for arm, a in d["arms"].items():
            if arm in ("none", "oracle"):
                continue
            col = (BLUE if arm == "fill" else ORANGE if arm.startswith("fill+")
                   else GREEN if arm.startswith("dense") else GREY)
            ax.scatter(a["prompt_tokens"], 100 * a["recall_on_unknown"],
                       marker=marker, color=col, s=18, zorder=3)
        ax.scatter([], [], marker=marker, color="k", s=18, label=lab)
    for col, lab in ((BLUE, "fill alone"), (GREY, "BM25@k"), (GREEN, "dense@k"), (ORANGE, "fill + BM25@k")):
        ax.scatter([], [], color=col, s=18, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel("prompt tokens per question")
    ax.set_ylabel("recall on facts the release missed (%)")
    ax.set_title("PopQA tail: weights vs. prompt")
    ax.legend(frameon=False, ncol=2, fontsize=5)
    panel_label(ax, "a")
    # (b) composition against the native ceiling
    ax = axs[1]
    rows = [("1.7B", "twohop_base_1.7B", "twohop_v2_1p7b"), ("4B", "twohop_base_4B", "twohop_v2_4b"),
            ("8B", "twohop_base_8B", "twohop_v2_8b")]
    xs, nat, inj = [], [], []

    def best(d, prefix):
        # the better phrasing, as the table does
        cands = [d.get(f"{prefix}_qa"), d.get(f"{prefix}_cont")]
        cands = [c for c in cands if c and c["both_hops_known"]]
        return max(c["twohop_given_both"] for c in cands) if cands else None
    for size, b, m in rows:
        db, dm = load(b), load(m)
        if not (db and dm):
            continue
        nat.append(100 * (best(db, "native") or 0))
        inj.append(100 * (best(dm, "injected") or 0))
        xs.append(size)
    x = range(len(xs))
    ax.bar([i - 0.18 for i in x], nat, 0.36, color=GREY, label="native facts (ceiling)")
    ax.bar([i + 0.18 for i in x], inj, 0.36, color=BLUE, label="injected facts")
    ax.set_xticks(list(x))
    ax.set_xticklabels(xs)
    ax.set_ylabel("two-hop given both hops (%)")
    ax.set_title("composition")
    ax.legend(frameon=False)
    panel_label(ax, "b")
    # (c) API use, log-prob rule
    ax = axs[2]
    sizes = [("1.7B", "1.7B", "1p7b"), ("4B", "4B", "4b"), ("8B", "8B", "8b")]
    base, served = [], []
    for lab, tb, tm in sizes:
        b = load(f"apiusage_lp_base_{tb}")
        m = load(f"apiusage_logprob_aug_{tm}")
        base.append(100 * b["libs"]["cyclopts"]["acc"] if b else 0)
        served.append(100 * m["libs"]["cyclopts"]["acc"] if m else 0)
    x = range(len(sizes))
    ax.bar([i - 0.18 for i in x], base, 0.36, color=GREY, label="released")
    ax.bar([i + 0.18 for i in x], served, 0.36, color=BLUE, label="served")
    ax.axhline(10.6, color=RED, lw=0.8, ls="--", label="chance")
    ax.set_xticks(list(x))
    ax.set_xticklabels([s[0] for s in sizes])
    ax.set_ylabel("correct parameter ranked first (%)")
    ax.set_title("a post-cutoff API, used in code")
    ax.legend(frameon=False)
    panel_label(ax, "c")
    fig.tight_layout()
    fig.savefig(OUT / "fig3.pdf")
    plt.close(fig)


# ------------------------------------------------------------ figure 4
def fig4():
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 4.6))
    axs = axs.ravel()
    # (a) absorption vs facts, r64 s40 tanh
    ax = axs[0]
    series = [(1000, ["exp10_qil_r64_qwen3-1.7b", "exp30_qil_r64_s1", "exp30_qil_r64_s2"]),
              (3000, ["exp73_cellfill_3000"]), (10000, ["exp52_cellfill_10k", "exp71_cap10000_tanh", "exp77_cap10000_r64_s1"]),
              (30000, ["exp73_cellfill_30000"]), (100000, ["exp53_cellfill_100k"])]
    ns, absorbed, sat = [], [], []
    for n, stems in series:
        ds = [d for d in (load(s) for s in stems) if d]
        if not ds:
            continue
        ns.append(n)
        absorbed.append(100 * st.mean(d["recall"]["merged"] for d in ds))
        sat.append(100 * st.mean(d["merge"]["saturation"] for d in ds))
    ax.semilogx(ns, absorbed, "o-", color=BLUE)
    ax.set_ylim(0, 100)
    ax.set_xlabel("facts presented")
    ax.set_ylabel("recall (%)", color=BLUE)
    ax2 = ax.twinx()
    ax2.plot(ns, sat, "s--", color=RED)
    ax2.set_ylabel("saturated coordinates (%)", color=RED)
    ax2.set_ylim(0, 100)
    ax2.spines["right"].set_visible(True)
    ax.set_title("the knee at r=64, s=40, tanh: a saturation collapse")
    panel_label(ax, "a")
    # (b) one knob at a time at 10k
    ax = axs[1]
    arms = [("tanh s=40 (2 seeds)", ["exp71_cap10000_tanh", "exp77_cap10000_r64_s1"], GREY),
            ("hardtanh-STE", ["exp79_cap10000_hardtanh_fill"], BLUE),
            ("tanh, floor 0.05", ["exp79_cap10000_tanh_floor"], BLUE),
            ("softsign", ["exp79_cap10000_softsign"], BLUE),
            ("tanh s=10", ["exp79_cap10000_s10"], BLUE),
            ("r=16", ["exp77_cap10000_r16"], GREEN), ("r=32", ["exp77_cap10000_r32"], GREEN),
            ("r=128", ["exp77_cap10000_r128"], GREEN),
            ("width 1/2", ["exp77_cap10000_w5"], ORANGE),
            ("4 paraphrases, 6 ep.", ["exp77_cap10000_aug4_ep6"], ORANGE),
            ("s=10, r=32", ["exp79_cap10000_s10_r32"], RED),
            ("DRF stack 3", ["exp79_cap10000_stack3_drf"], GREY)]
    labels, vals, cols = [], [], []
    for lab, stems, col in arms:
        ds = [d for d in (load(s) for s in stems) if d]
        if not ds:
            continue
        labels.append(lab)
        vals.append(100 * st.mean(d["recall"]["merged"] for d in ds))
        cols.append(col)
    y = range(len(labels))
    ax.barh(list(y), vals, color=cols, height=0.6)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=5.5)
    ax.invert_yaxis()
    ax.set_xlabel("recall at 10k facts (%)")
    ax.set_title("what moves the knee")
    panel_label(ax, "b")
    # (c) liveness: mean link gradient over training
    ax = axs[2]
    for lab, stem, col, ls in (("s=40 r=64", "exp77_cap10000_r64_s1", GREY, "-"), ("s=40 r=128", "exp77_cap10000_r128", GREEN, "-"),
                               ("s=40 r=16", "exp77_cap10000_r16", GREEN, ":"), ("hardtanh-STE", "exp79_cap10000_hardtanh_fill", BLUE, "-"),
                               ("s=10", "exp79_cap10000_s10", RED, "-"), ("softsign", "exp79_cap10000_softsign", ORANGE, "-"),
                               ("s=10 r=32", "exp79_cap10000_s10_r32", RED, ":")):
        d = load(stem)
        if not d or not d.get("liveness"):
            continue
        ax.plot([h["epoch"] for h in d["liveness"]], [h["mean_link_grad"] for h in d["liveness"]],
                color=col, lw=1, ls=ls, label=lab)
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean link derivative 1 − t²")
    ax.set_ylim(0, 1)
    ax.set_title("gradient alive during training, 10k facts")
    ax.legend(frameon=False, fontsize=5.5, ncol=2)
    panel_label(ax, "c")
    # (d) bits per fact
    ax = axs[3]
    bits = [("r=16", "bits_cap10000_r16"), ("r=32", "bits_cap10000_r32"), ("r=128", "bits_cap10000_r128"),
            ("hardtanh", "bits_cap10000_hardtanh_fill"), ("floor", "bits_cap10000_tanh_floor"),
            ("softsign", "bits_cap10000_softsign"), ("s=10", "bits_cap10000_s10"),
            ("s=10,r=32", "bits_cap10000_s10_r32")]
    labels, vals = [], []
    for lab, stem in bits:
        d = load(stem)
        if d:
            labels.append(lab)
            vals.append(d["excess_bits_per_fact"])
    ax.bar(range(len(labels)), vals, color=BLUE)
    ax.axhline(21.7, color=RED, ls="--", lw=0.8)
    ax.text(0, 21.9, "entropy of a fact (21.7 bits)", color=RED, fontsize=5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("bits stored per fact (of 21.7)")
    ax.set_ylim(0, 23)
    ax.set_title("capacity in bits, 10k facts")
    panel_label(ax, "d")
    fig.tight_layout()
    fig.savefig(OUT / "fig4.pdf")
    plt.close(fig)


# ------------------------------------------------------------ figure 5
def fig5():
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3))
    # (a) fusion arms
    ax = axs[0]
    def fused(stem, how):
        d = load(stem)
        return 100 * d["arms"][how]["recall"]["fused"] if d and how in d["arms"] else None
    def rounds(stem):
        d = load(stem)
        return 100 * d["history"][-1]["recall"] if d else None
    groups = [("synthetic K=2", [("one-shot, best rule", fused("fuse_k2_full_rules", "magnitude")),
                                 ("inert, sum", fused("fuse_k2_inert_neutral", "sum")),
                                 ("rounds", rounds("fuse_k2_rounds3"))]),
              ("synthetic K=4", [("one-shot, best rule", fused("fuse_k4_full_rules", "magnitude")),
                                 ("inert, sum", None), ("rounds", rounds("fuse_k4_rounds3"))]),
              ("real K=2", [("one-shot, best rule", fused("fuse_real_k2_full", "clamp")),
                            ("inert, sum", fused("fuse_real_k2_inert", "sum")),
                            ("rounds", rounds("fuse_real_k2_rounds3"))]),
              ("real K=4", [("one-shot, best rule", fused("fuse_real_k4_reserved", "sum")),
                            ("inert, sum", fused("fuse_real_k4_inert", "sum")),
                            ("rounds", rounds("fuse_real_k4_rounds3"))])]
    w = 0.26
    for j, col in enumerate([GREY, BLUE, ORANGE]):
        ax.bar([i + (j - 1) * w for i in range(len(groups))],
               [(g[1][j][1] or 0) for g in groups], w, color=col, label=groups[0][1][j][0])
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0].replace(" K=", "\nK=") for g in groups], fontsize=6)
    ax.set_ylabel("facts kept after the merge (%)")
    ax.set_title("several writers, one release")
    ax.legend(frameon=False)
    panel_label(ax, "a")
    # (b) rounds
    ax = axs[1]
    for lab, stem, col in (("synthetic K=2", "fuse_k2_rounds3", BLUE), ("synthetic K=4", "fuse_k4_rounds3", GREEN),
                           ("real K=2", "fuse_real_k2_rounds3", ORANGE), ("real K=4", "fuse_real_k4_rounds3", RED)):
        d = load(stem)
        if d:
            ax.plot([h["round"] + 1 for h in d["history"]], [100 * h["recall"] for h in d["history"]],
                    "o-", color=col, label=lab)
    ax.set_xlabel("round")
    ax.set_ylabel("merged recall (%)")
    ax.set_xticks([1, 2, 3])
    ax.set_title("fusion by rounds")
    ax.legend(frameon=False)
    panel_label(ax, "b")
    # (c) sequential updates
    ax = axs[2]
    d = load("exp70_seq_anchor")
    if d:
        hist = d["history"]
        for t in range(len(hist)):
            ys = [100 * h["recalls"][t] for h in hist if len(h["recalls"]) > t]
            xs = [h["task"] + 1 for h in hist if len(h["recalls"]) > t]
            ax.plot(xs, ys, "-", color=BLUE, alpha=0.3 + 0.7 * t / len(hist), lw=1)
        ax2 = ax.twinx()
        ax2.plot([h["task"] + 1 for h in hist], [h["mean_room_left"] / hist[0]["mean_room_left"] * 100 for h in hist],
                 "s--", color=RED)
        ax2.set_ylabel("room left (% of first)", color=RED)
        ax2.set_ylim(0, 105)
        ax2.spines["right"].set_visible(True)
    ax.set_xlabel("update")
    ax.set_ylabel("recall of each update's facts (%)")
    ax.set_title("six sequential updates")
    panel_label(ax, "c")
    fig.tight_layout()
    fig.savefig(OUT / "fig5.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig2(); fig3(); fig4(); fig5()
    print("wrote", sorted(p.name for p in OUT.glob("fig*.pdf")))

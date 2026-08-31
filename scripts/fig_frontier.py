#!/usr/bin/env python
"""Figure 1: the invariant frontier, on both perplexity axes.

Two panels side by side, because the disagreement between them is the
paper's central methodological point: rehearsal is drawn from WikiText-train
and WikiText-test therefore flatters every method, while LAMBADA is untouched
by rehearsal and prices the real cost. A reader who only sees panel (a)
concludes that in-cell learning is free.

Data is read from the archived result files, never typed, so the figure
cannot drift from the tables the way the first draft's did.
"""

import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = Path("results")
OK = {"blue": "#0072B2", "green": "#009E73", "red": "#D55E00",
      "gray": "#7F7F7F"}

# label, files, stage, kind, (dx, dy) offsets per panel in points
SERIES = [
    ("A clip-merge", ["exp1c_replay10_qwen3-1.7b_r16_e24", "exp1c_s1",
                      "exp1c_s2"], "merged", "proj", (-2, 11), (-2, 12)),
    ("CellFill r16", ["exp16_qil_r16_replay10", "exp31_qil_r16_s1",
                      "exp31_qil_r16_s2"], "merged", "struct",
     (-4, -18), (-6, -18)),
    ("A+ heal", ["exp1c_replay10_qwen3-1.7b_r16_e24", "exp1c_s1", "exp1c_s2"],
     "heal", "proj", (-6, 12), (-4, 12)),
    ("CellFill r64", ["exp10_qil_r64_qwen3-1.7b", "exp30_qil_r64_s1",
                      "exp30_qil_r64_s2"], "merged", "struct",
     (-24, 11), (-14, -19)),
    ("B dense (reh. 0.1)", ["exp18_b_replay10_s0", "exp18_b_replay10_s1"],
     "heal", "proj", (-16, 13), (-40, 10)),
    ("B dense", ["exp3_dense_qwen3-1.7b_e24", "exp3_s1", "exp3_s2"], "heal",
     "proj", (-58, -3), (-34, -13)),
    ("unconstrained", ["exp4_free_qwen3-1.7b_e24", "exp4_free_s1",
                       "exp4_free_s2"], "heal", "free", (30, -17), (2, 14)),
    ("no rehearsal", ["exp0b_qwen3-1.7b_r16_e24"], "merged", "ablate",
     (0, 12), None),
    ("stale rehearsal", ["exp1_replay30_qwen3-1.7b_r16_e24"], "merged",
     "ablate", (26, 2), None),
]
STYLE = {"struct": (OK["green"], "D"), "proj": (OK["blue"], "o"),
         "free": (OK["red"], "X"), "ablate": (OK["gray"], "s")}


def load(stem):
    p = R / f"{stem}.json"
    return json.loads(p.read_text()) if p.exists() else None


def pull(d, stage, field):
    if d is None:
        return None
    if stage == "heal":
        h = d.get("heal") or {}
        return {"rec": h.get("recall"), "wiki": h.get("ppl"),
                "lam": h.get("ppl_x")}[field]
    return {"rec": d["recall"].get("merged"), "wiki": d["ppl"].get("merged"),
            "lam": (d.get("ppl_lambada") or {}).get("merged")}[field]


def agg(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return None, None
    return (st.mean(v), st.stdev(v) if len(v) > 1 else None)


anchor = load("exp1c_s1")
A_WIKI = anchor["ppl"]["anchor_fp32"]
A_LAM = anchor["ppl_lambada"]["anchor_fp32"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.4))

for ax, field, aline, title, xlab in (
        (ax1, "wiki", A_WIKI, "a  WikiText-2 — same domain as rehearsal",
         "New-fact recall (%)"),
        (ax2, "lam", A_LAM, "b  LAMBADA — held out from rehearsal",
         "New-fact recall (%)")):
    for label, stems, stage, kind, off1, off2 in SERIES:
        off = off1 if field == "wiki" else off2
        if off is None:
            continue
        ds = [load(s) for s in stems]
        rm, rs = agg([pull(d, stage, "rec") for d in ds])
        ym, ys = agg([pull(d, stage, field) for d in ds])
        if rm is None or ym is None:
            continue
        c, m = STYLE[kind]
        ax.errorbar(rm * 100, ym, xerr=(rs * 100 if rs else None), yerr=ys,
                    fmt=m, color=c, capsize=3, markersize=7, lw=1.2, zorder=3)
        ax.annotate(label, (rm * 100, ym), textcoords="offset points",
                    xytext=off, ha="center", fontsize=8.5, zorder=4)
    ax.axhline(aline, color=OK["gray"], lw=0.9, ls="--", zorder=1)
    at_left = field == "wiki"
    ax.annotate("4-bit anchor", (0.02 if at_left else 0.98, aline),
                xycoords=("axes fraction", "data"),
                xytext=(0, 5), textcoords="offset points",
                ha="left" if at_left else "right",
                fontsize=8, color=OK["gray"])
    ax.set_yscale("log")
    ax.set_xlabel(xlab)
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(alpha=0.22, lw=0.5, which="both")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(8, 70)

ax1.set_ylabel("perplexity (lower = less forgetting)")
ax1.set_ylim(8.8, 95)
ax1.set_yticks([10, 12, 15, 20, 30, 45, 70])
ax2.set_ylim(26, 90)
ax2.set_yticks([28, 32, 40, 50, 65, 85])
for ax in (ax1, ax2):
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())

ax1.legend(handles=[
    Line2D([], [], marker="D", ls="", color=OK["green"],
           label="invariant (structural)"),
    Line2D([], [], marker="o", ls="", color=OK["blue"],
           label="invariant (projected)"),
    Line2D([], [], marker="s", ls="", color=OK["gray"],
           label="rehearsal ablated"),
    Line2D([], [], marker="X", ls="", color=OK["red"], label="not invariant"),
], fontsize=8, loc="upper right", frameon=False, ncol=2,
    columnspacing=0.8, handletextpad=0.3)

fig.tight_layout()
fig.savefig("paper/figs/frontier.pdf")
print("wrote paper/figs/frontier.pdf")

#!/usr/bin/env python
"""The drift law: what moves the block scale at a major version.

Nine arms -- both parameterizations, four learning rates, a nine-fold span in
displacement -- put their first consolidation on one line: scale growth is
proportional to how far the fill moved the weights, not to time, not to the
count of codes rewritten. The zero point is not fitted: a zero-displacement
consolidation is a bitwise fixed point (Q(Q(w)) = Q(w)), so the curve is
anchored at the origin analytically and the line is the in-range
approximation.

Data is read from the archived products, never typed.
"""
import json
import statistics as st
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("results")
OK = {"blue": "#0072B2", "red": "#D55E00", "gray": "#7F7F7F",
      "green": "#009E73"}

ARMS = [
    ("exp132_densemlp_s0", "dense"), ("exp132_densemlp_s1", "dense"),
    ("exp132_densemlp_lr5e4_s0", "dense"), ("exp132_densemlp_lr2e4_s0", "dense"),
    ("exp132_rankmlp_s0", "rank"), ("exp132_rankmlp_s1", "rank"),
    ("exp132_rankmlp_lr5e4_s0", "rank"), ("exp132_rankmlp_lr2e4_s0", "rank"),
    ("exp131_burn10_s0", "nineburn"),
]

X, Y, F = [], [], []
for stem, fam in ARMS:
    d = json.loads((R / f"{stem}.json").read_text())
    c = d["consolidations"][0]
    t = c["after_task"]
    mt = (d["history"][t - 1]["mean_abs_tanh"]
          + d["history"][t]["mean_abs_tanh"]) / 2
    X.append(mt)
    Y.append((c["absmax_ratio_mean"] - 1) * 100)
    F.append(fam)

mx, my = st.mean(X), st.mean(Y)
cov = sum((a - mx) * (b - my) for a, b in zip(X, Y)) / (len(X) - 1)
r = cov / (st.stdev(X) * st.stdev(Y))
slope = cov / st.variance(X)
icpt = my - slope * mx

fig, ax = plt.subplots(figsize=(5.6, 4.3))
xs = [0.04, max(X) * 1.06]
ax.plot(xs, [icpt + slope * v for v in xs], color=OK["gray"], lw=1.0,
        zorder=1)
STYLE = {"dense": (OK["blue"], "o", "bounded dense fill"),
         "rank": (OK["red"], "s", "low-rank fill, $r=64$"),
         "nineburn": (OK["green"], "D", "nine-consolidation sequence")}
seen = set()
for x, y, f in zip(X, Y, F):
    c, m, lbl = STYLE[f]
    ax.scatter(x, y, s=44, color=c, marker=m, zorder=3,
               edgecolor="white", linewidth=0.6,
               label=lbl if f not in seen else None)
    seen.add(f)
# out-of-sample: the knee's second seed, run after the fit was frozen
d10 = json.loads((R / "exp132_densemlp_lr2e4_s1.json").read_text())
c10 = d10["consolidations"][0]; t10 = c10["after_task"]
x10 = (d10["history"][t10-1]["mean_abs_tanh"] + d10["history"][t10]["mean_abs_tanh"]) / 2
y10 = (c10["absmax_ratio_mean"] - 1) * 100
ax.scatter([x10], [y10], s=58, facecolor="white", edgecolor=OK["blue"],
           marker="o", zorder=4, linewidth=1.2)
ax.annotate("a tenth arm, run later:\nlands on the line unrefitted\n"
            f"(residual {y10 - (slope*x10 + icpt):+.3f})",
            xy=(x10, y10), xytext=(x10 + 0.14, y10 - 1.62), fontsize=8,
            color=OK["blue"],
            arrowprops=dict(arrowstyle="-", color=OK["blue"], lw=0.8))
ax.scatter([0], [0], s=70, facecolor="white", edgecolor="black",
           marker="o", zorder=4, linewidth=1.1)
ax.annotate("anchored by proof:\na consolidation that follows no\n"
            "writing is a bitwise fixed point",
            xy=(0, 0), xytext=(0.015, 2.6), fontsize=8, color="black",
            arrowprops=dict(arrowstyle="-", color="black", lw=0.7))
ax.annotate(f"scale growth $= {slope:.1f}\\,\\mathbb{{E}}|t| "
            f"{'+' if icpt >= 0 else '-'} {abs(icpt):.2f}$\n"
            f"$r = {r:.3f}$, nine arms",
            xy=(0.40, icpt + slope * 0.40), xytext=(0.30, 5.05),
            fontsize=8.5, color=OK["gray"])
ax.set_xlabel(r"displacement fraction $\mathbb{E}|t|$ before the "
              "consolidation", fontsize=9)
ax.set_ylabel("block-scale growth at the consolidation (%)", fontsize=9)
ax.set_title("the drift is bought by writing", fontsize=10, loc="left")
ax.legend(fontsize=8, frameon=False, loc="lower right")
ax.tick_params(labelsize=8)
ax.set_xlim(-0.02, xs[1])
ax.set_ylim(-0.25, 5.6)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("paper/figs/driftlaw.pdf")
print(f"driftlaw.pdf: slope={slope:.2f} icpt={icpt:.3f} r={r:.3f}")

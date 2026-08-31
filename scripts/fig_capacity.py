#!/usr/bin/env python
"""Figure: capacity scaling — absorbed vs presented facts (log-log).

The exponent is computed from the plotted points, never typed in, so the
legend cannot drift from the data the way it did in the first draft.
"""

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OKABE = {"blue": "#0072B2", "green": "#009E73", "gray": "#7F7F7F"}

# Recipes must match within a series or the exponent is meaningless.
# A+ at rehearsal 0.3: exp2 (1k, heal recall .31467) -> exp6 (10k, .19133).
# A+ at rehearsal 0.1: exp1c (1k, .37933) -> exp9/exp17 (10k, pending).
# B  at rehearsal 0.3: exp3 (1k, .612)    -> exp6b (10k, .413).
APLUS = [(1000, 315), (10000, 1913)]   # both replay 0.3
B = [(1000, 612), (10000, 4130)]       # both replay 0.3

fig, ax = plt.subplots(figsize=(5.4, 4.0))
for pts, c, lbl in ((APLUS, OKABE["blue"], "A+ heal"),
                    (B, OKABE["green"], "B projected dense")):
    xs, ys = zip(*pts)
    alpha = math.log10(ys[1] / ys[0]) / math.log10(xs[1] / xs[0])
    ax.plot(xs, ys, "o-", color=c, label=f"{lbl}  ($\\alpha={alpha:.2f}$)")

ax.plot([1000, 10000], [1000, 10000], ls=":", color=OKABE["gray"], lw=0.9)
ax.text(2800, 3800, "perfect absorption", rotation=33, fontsize=8,
        color=OKABE["gray"])

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Facts presented")
ax.set_ylabel("Facts absorbed (recall × N)")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(alpha=0.25, lw=0.5, which="both")
ax.legend(fontsize=9, frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig("paper/figs/capacity.pdf")
print("wrote paper/figs/capacity.pdf")

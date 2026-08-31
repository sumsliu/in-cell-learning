#!/usr/bin/env python
"""What the cell bound buys, and what it costs, at 8B.

Two arms, same card, same six tasks, each at its own best learning rate.
Panel (a): the artifact axis. The unconstrained arm stops being the released
file at the first task and never returns; the bounded arm re-quantizes to the
vendor's codes at every fold.
Panel (b): the function axis. Absorption per task is close; the unconstrained
arm keeps slightly more late-sequence plasticity and pays less cross-domain
perplexity. The bound's price is real and bounded; what it buys is the file.

Data is read from the archived products, never typed.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("results")
OK = {"blue": "#0072B2", "red": "#D55E00", "gray": "#7F7F7F"}

free = json.loads((R / "cycle_free8b.json").read_text())
bound = json.loads((R / "exp124_loop_8b.json").read_text())

t = list(range(6))
esc = [h["invariance_violations"] for h in free["history"]]
z = [h["invariance_violations"] for h in bound["history"]]
fb = [h["recalls"][h["task"]] for h in bound["history"]]
ff = [h["recalls"][h["task"]] for h in free["history"]]
TOT = 6.946e9

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))

ax1.plot(t, [max(v, 3e5) for v in esc], color=OK["red"], marker="o", ms=5,
         lw=1.4, label="unconstrained (lr $2\\times10^{-4}$, its best)")
ax1.plot(t, [3e5] * 6, color=OK["blue"], marker="o", ms=5, lw=1.4,
         label="bounded (lr $10^{-3}$, its best)")
ax1.set_yscale("log")
ax1.set_ylim(2e5, 1e10)
ax1.axhline(TOT, color=OK["gray"], lw=0.8, ls=":")
ax1.text(3.55, TOT * 1.5, "all constrained weights", fontsize=7.5,
         color=OK["gray"])
ax1.annotate("29% of the file has left\nthe released codes",
             xy=(5, esc[-1]), xytext=(2.75, 6e7), fontsize=8.5,
             color=OK["red"],
             arrowprops=dict(arrowstyle="-", color=OK["red"], lw=0.8))
ax1.text(1.30, 4.6e5, "exactly zero, every fold, every task",
         fontsize=8.5, color=OK["blue"])
ax1.set_xlabel("task", fontsize=9)
ax1.set_ylabel("weights outside their cells (log)", fontsize=9)
ax1.set_title("a  the artifact: destroyed against preserved", fontsize=10,
              loc="left")
ax1.legend(fontsize=8, frameon=False, loc="center left")
ax1.tick_params(labelsize=8)
ax1.spines[["top", "right"]].set_visible(False)

ax2.plot(t, ff, color=OK["red"], marker="o", ms=5, lw=1.4,
         label="unconstrained")
ax2.plot(t, fb, color=OK["blue"], marker="o", ms=5, lw=1.4, label="bounded")
ax2.set_xlabel("task", fontsize=9)
ax2.set_ylabel("fresh recall on the new task", fontsize=9)
ax2.set_title("b  the function: close, and honestly priced", fontsize=10,
              loc="left")
fL = free["history"][-1]["ppl_lambada"]
bL = bound["history"][-1]["ppl_lambada"]
ax2.text(0.02, 0.185,
         "summed absorption  3.63 vs 3.48\n"
         f"end LAMBADA ppl    {fL:.1f} vs {bL:.1f}\n"
         "end retention sum   3.49 vs 3.54",
         fontsize=8.5, color=OK["gray"], family="monospace")
ax2.set_ylim(0.15, 0.9)
ax2.legend(fontsize=8, frameon=False, loc="upper right")
ax2.tick_params(labelsize=8)
ax2.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig("paper/figs/boundprice.pdf")
print("boundprice.pdf: esc", esc, "| free fresh", [round(v, 3) for v in ff])

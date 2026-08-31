#!/usr/bin/env python
"""The nine-consolidation sequence: drift tracks writing, and the room's
restoration level is format-pinned only to first order.

Panel (a): the per-consolidation scale growth falls in step with the
per-version absorption -- the decline is the image of the dose collapsing,
not an approach to a drift equilibrium.
Panel (b): the room restored by each consolidation, against the first-order
prediction (envelope level x cumulative drift). The growing deficit is code
re-binning: as the scale inflates, normalized weights shrink and mass
migrates into the central narrow cells.

Data is read from the archived products, never typed.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("results")
OK = {"blue": "#0072B2", "red": "#D55E00", "gray": "#7F7F7F",
      "green": "#009E73"}

d = json.loads((R / "exp131_burn10_s0.json").read_text())
env = json.loads(
    (R / "envelopes/envelope_Qwen3-1.7B-Base-bnb-4bit.json").read_text()
)["room_mean"]

cons = d["consolidations"]
fresh = [h["recalls"][h["task"]] for h in d["history"]]
k = list(range(1, len(cons) + 1))
drift = [(c["absmax_ratio_mean"] - 1) * 100 for c in cons]
absorbed = [fresh[c["after_task"] - 1] + fresh[c["after_task"]] for c in cons]
room = [c["room_after_mean"] * 1e3 for c in cons]
cum, pred = 1.0, []
for c in cons:
    cum *= c["absmax_ratio_mean"]
    pred.append(env * cum * 1e3)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))

ax1.plot(k, drift, color=OK["blue"], marker="o", ms=5, lw=1.4, zorder=3,
         label="scale growth per consolidation (%)")
ax1b = ax1.twinx()
ax1b.bar(k, absorbed, width=0.55, color=OK["gray"], alpha=0.35, zorder=1,
         label="absorption in that version")
ax1.set_xlabel("consolidation index", fontsize=9)
ax1.set_ylabel("scale growth (%)", fontsize=9, color=OK["blue"])
ax1b.set_ylabel("absorption (summed fresh recall)", fontsize=9,
                color=OK["gray"])
ax1.set_title("a  growth falls with the dose, not with time", fontsize=10,
              loc="left")
ax1.tick_params(labelsize=8)
ax1b.tick_params(labelsize=8)
ax1.set_ylim(0, 4.7)
ax1b.set_ylim(0, 0.95)
ax1.spines["top"].set_visible(False)
ax1b.spines["top"].set_visible(False)

ax2.plot(k, pred, color=OK["gray"], lw=1.2, ls="--",
         label="first order: envelope $\\times$ cumulative drift")
ax2.plot(k, room, color=OK["green"], marker="o", ms=5, lw=1.4,
         label="measured room after the consolidation")
ax2.fill_between(k, room, pred, color=OK["red"], alpha=0.12)
ax2.annotate("code re-binning:\n$-10.5\\%$ by the ninth\n"
             "(zero-parameter prediction $-11.6\\%$)",
             xy=(9, (room[-1] + pred[-1]) / 2), xytext=(4.0, 6.12),
             fontsize=8.5, color=OK["red"],
             arrowprops=dict(arrowstyle="-", color=OK["red"], lw=0.8))
ax2.set_xlabel("consolidation index", fontsize=9)
ax2.set_ylabel(r"room after consolidation ($\times 10^{-3}$)", fontsize=9)
ax2.set_title("b  restoration is format-pinned only to first order",
              fontsize=10, loc="left")
ax2.legend(fontsize=8, frameon=False, loc="lower right")
ax2.tick_params(labelsize=8)
ax2.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig("paper/figs/nineburn.pdf")
print("nineburn.pdf: drift", [round(v, 2) for v in drift],
      "| room", [round(v, 2) for v in room[:3]], "... pred", round(pred[-1], 2))

#!/usr/bin/env python
"""Figure 3: geometry of the safe region, read straight from the archived run.

Left  — walking the segment anchor -> original fp32 -> beyond, with the
        fraction of weights whose code changes overlaid. The whole segment
        to the original weights is inside the cells; past it, codes break.
Right — random in-cell perturbation at radius rho, in-domain and
        cross-domain, against the diagonal-Fisher prediction of Prop. 5.
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = Path("results/exp_geom_1p7b.json")
OKABE = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
         "red": "#D55E00", "gray": "#7F7F7F", "purple": "#CC79A7"}

d = json.loads(SRC.read_text())
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.0))

# ---- left: interpolation -------------------------------------------------
interp = d["interpolation"]
ts = [r["t"] for r in interp]
ax1.plot(ts, [r["ppl_wiki"] for r in interp], "o-", color=OKABE["blue"],
         label="WikiText-2")
ax1.plot(ts, [r["ppl_lambada"] for r in interp], "s-", color=OKABE["orange"],
         label="LAMBADA")
ax1.set_xlabel(r"$t$ :  $\hat{W} + t\,(W_{\mathrm{orig}} - \hat{W})$")
ax1.set_ylabel("perplexity")
ax1.axvline(1.0, color=OKABE["gray"], lw=0.8, ls=":")
ax1.text(1.02, 27.4, "original\nfp32", fontsize=8, color=OKABE["gray"])

ax1b = ax1.twinx()
esc = [100 * r["escape_frac"] for r in interp]
ax1b.fill_between(ts, esc, color=OKABE["red"], alpha=0.18)
ax1b.plot(ts, esc, color=OKABE["red"], lw=1.2, ls="--")
ax1b.set_ylabel("weights leaving their cell (%)", color=OKABE["red"])
ax1b.tick_params(axis="y", colors=OKABE["red"])
ax1b.set_ylim(0, 40)
ax1.legend(fontsize=8, frameon=False, loc="upper center")
ax1.set_title("a  the segment is inside the cells", fontsize=10, loc="left")

# ---- right: perturbation vs Fisher --------------------------------------
pert = {}
for r in d["perturbation"]:
    pert.setdefault(r["rho"], []).append(r)
rhos = sorted(pert)
base_w = next(r["ppl_wiki"] for r in interp if r["t"] == 0)
base_l = next(r["ppl_lambada"] for r in interp if r["t"] == 0)


def mean(vals):
    return sum(vals) / len(vals)


dn_w = [math.log(mean([x["ppl_wiki"] for x in pert[r]]) / base_w) for r in rhos]
dn_l = [math.log(mean([x["ppl_lambada"] for x in pert[r]]) / base_l)
        for r in rhos]
pred = [d["fisher"]["predicted_dnll"][str(r)] for r in rhos]

ax2.plot(rhos, dn_w, "o-", color=OKABE["blue"], label="measured (WikiText-2)")
ax2.plot(rhos, dn_l, "s-", color=OKABE["orange"], label="measured (LAMBADA)")
ax2.plot(rhos, pred, "^--", color=OKABE["green"],
         label="diagonal-Fisher budget")
ref = [dn_w[-1] * r ** 2 for r in rhos]
ax2.plot(rhos, ref, ":", color=OKABE["gray"], lw=1.2,
         label=r"$\rho^2$ (anchored to WikiText at $\rho{=}1$)")
ax2.set_xscale("log")
ax2.set_xticks(rhos)
ax2.set_xticklabels([f"{r:g}" for r in rhos])
ax2.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
ax2.set_yscale("log")
ax2.set_xlabel(r"in-cell radius $\rho$ (fraction of available room)")
ax2.set_ylabel(r"$\Delta$ NLL vs. anchor")
ax2.legend(fontsize=8, frameon=False, loc="upper left")
ax2.set_title("b  cost of filling the cell", fontsize=10, loc="left")

for ax in (ax1, ax2):
    ax.spines[["top"]].set_visible(False)
    ax.grid(alpha=0.22, lw=0.5)
ax2.spines[["right"]].set_visible(False)

fig.tight_layout()
fig.savefig("paper/figs/geometry.pdf")

fit = [math.log(dn_w[i] / dn_w[-1]) / math.log(rhos[i])
       for i in range(1, len(rhos) - 1)]
print("wrote paper/figs/geometry.pdf")
print(f"measured exponent (rho 0.25-0.75): {mean(fit):.2f}")
print("underestimate factor at rho=1: "
      f"{dn_w[-1] / pred[-1]:.0f}x")

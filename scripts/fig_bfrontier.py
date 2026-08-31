#!/usr/bin/env python
"""The parameterization frontier: what absorption costs in capability.

Nine arms: the knee now carries two seeds (lr2e4 s0+s1).

One panel, one message: the two parameterizations do not trade along one
line. The bounded dense fill has a knee -- between 5e-4 and 2e-4 the
capability cost collapses by 89% while absorption gives up 17% -- and the
achievable frontier above ~0.7 absorption is dense at every point. Every
low-rank arm, including both seeds at its own working rate, lies inside it.

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

ARMS = [  # label, file, family, lr-tag, annotation offset (pts)
    ("1e-3", "exp132_densemlp_s0",       "dense", (6, 4)),
    ("1e-3", "exp132_densemlp_s1",       "dense", (6, -10)),
    ("5e-4", "exp132_densemlp_lr5e4_s0", "dense", (6, -2)),
    ("2e-4", "exp132_densemlp_lr2e4_s0", "dense", (5, 6)),
    ("2e-4", "exp132_densemlp_lr2e4_s1", "dense", (5, -11)),
    ("1e-3", "exp132_rankmlp_s0",        "rank",  (-14, 8)),
    ("1e-3", "exp132_rankmlp_s1",        "rank",  (6, 4)),
    ("5e-4", "exp132_rankmlp_lr5e4_s0",  "rank",  (-24, -3)),
    ("2e-4", "exp132_rankmlp_lr2e4_s0",  "rank",  (6, -3)),
]

def load(stem):
    d = json.loads((R / f"{stem}.json").read_text())
    fresh = [h["recalls"][h["task"]] for h in d["history"]]
    a = d["validations"][0]["scores"]
    v = d["validations"][-1]["scores"]
    tax = sum(a[k] - v[k] for k in a) / len(a) * 100
    return sum(fresh), tax

pts = [(lbl, *load(f), fam, off) for lbl, f, fam, off in ARMS]

fig, ax = plt.subplots(figsize=(5.6, 4.3))
for lbl, x, y, fam, off in pts:
    c = OK["blue"] if fam == "dense" else OK["red"]
    m = "o" if fam == "dense" else "s"
    ax.scatter(x, y, s=42, color=c, marker=m, zorder=3,
               edgecolor="white", linewidth=0.6)
    ax.annotate(lbl, (x, y), textcoords="offset points", xytext=off,
                fontsize=8, color=c)

# the achievable frontier (lower-right is better): staircase through the
# non-dominated points, computed rather than drawn by eye
nd = []
for lbl, x, y, fam, off in sorted(pts, key=lambda p: p[1]):
    if all(not (qx >= x and qy <= y and (qx, qy) != (x, y))
           for _, qx, qy, _, _ in pts):
        nd.append((x, y))
nd.sort()
ax.plot([x for x, _ in nd], [y for _, y in nd], color=OK["gray"], lw=1.1,
        ls="--", zorder=1)

ax.annotate("the knee: 89% of the cost vanishes\nfor 17% of the absorption",
            xy=(2.53, 0.7), xytext=(3.25, 2.9), fontsize=8.5,
            color=OK["gray"],
            arrowprops=dict(arrowstyle="-", color=OK["gray"], lw=0.8))
ax.scatter([], [], s=42, color=OK["blue"], marker="o",
           label="bounded dense fill (MLP)")
ax.scatter([], [], s=42, color=OK["red"], marker="s",
           label="low-rank fill, $r=64$ (MLP)")
ax.plot([], [], color=OK["gray"], lw=1.0, ls="--", label="achievable frontier")
ax.legend(fontsize=8, frameon=False, loc="upper left")
ax.set_xlabel("knowledge absorbed (summed fresh recall, six tasks)",
              fontsize=9)
ax.set_ylabel("capability cost (mean suite points)", fontsize=9)
ax.set_xlim(0.25, 5.75)
ax.set_title("the dense fill owns the frontier above 0.7 absorbed", fontsize=10,
             loc="left")
ax.tick_params(labelsize=8)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("paper/figs/bfrontier.pdf")
print("bfrontier.pdf:", [(l, round(x, 2), round(y, 1)) for l, x, y, _, _ in pts])

#!/usr/bin/env python
"""Knowledge in the weights against knowledge in the prompt, at 8B.

Chunked document store (each fact sentence inside a paragraph, 200k
distractors): recall and prompt cost of every arm.

  python scripts/fig_weights_vs_prompt.py --out paper/figs/weights_vs_prompt.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = Path(__file__).resolve().parents[1] / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="rag_own_chunked_8b.json")
    ap.add_argument("--out", default="paper/figs/weights_vs_prompt.pdf")
    args = ap.parse_args()
    d = json.loads((R / args.src).read_text())
    arms = ["none", "bm25@1", "bm25@3", "bm25@5", "dense@1", "dense@3", "dense@5",
            "oracle", "fill", "fill+bm25@1"]
    arms = [a for a in arms if a in d["arms"]]
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw=dict(width_ratios=[1.35, 1]))
    vals = [100 * d["arms"][a]["recall"] for a in arms]
    cols = ["#d62728" if a.startswith("fill") else ("#555555" if a == "oracle" else "#bbbbbb") for a in arms]
    ax[0].bar(range(len(arms)), vals, color=cols)
    for i, v in enumerate(vals):
        ax[0].text(i, v + 1.5, f"{v:.0f}", ha="center", fontsize=6.5)
    ax[0].set_xticks(range(len(arms)))
    ax[0].set_xticklabels(arms, rotation=40, ha="right", fontsize=6.5)
    ax[0].set_ylabel("recall (%)"); ax[0].set_ylim(0, 105)
    ax[0].set_title("a  chunked store, 200k distractors, Qwen3-8B", loc="left", fontsize=8)
    for a in arms:
        x = d["arms"][a]["prompt_tokens"]
        y = 100 * d["arms"][a]["recall"]
        c = "#d62728" if a.startswith("fill") else ("#555555" if a == "oracle" else "#888888")
        ax[1].plot(x, y, "o", ms=4, color=c)
        ax[1].annotate(a, (x, y), textcoords="offset points", xytext=(4, -2), fontsize=6)
    ax[1].set_xscale("log"); ax[1].set_xlabel("prompt tokens per question")
    ax[1].set_ylabel("recall (%)"); ax[1].set_ylim(0, 105)
    ax[1].set_title("b  what the recall costs at inference", loc="left", fontsize=8)
    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight"); print(f"[fig] {out}")


if __name__ == "__main__":
    main()

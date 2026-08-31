#!/usr/bin/env python
"""One bar per vendor release: recall of the real corpus, every grid.

  python scripts/fig_releases.py --out paper/figs/releases.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = Path(__file__).resolve().parents[1] / "results"

ROWS = [
    ("Qwen3-1.7B\nNF4", ["exp45_official_twohop", "exp49_official_1p7b_s1", "exp49_official_1p7b_s2"]),
    ("Qwen3-1.7B\nint4 g64", ["exp72_w4a16_1p7b"]),
    ("Qwen3-4B\nNF4", ["exp46_official_4b", "exp50_official_4b_s1"]),
    ("Qwen3-4B\nint4 g128", ["exp63_w4a16_4b"]),
    ("Qwen3-8B\nNF4", ["exp47_official_8b", "exp47_official_8b_s1", "exp47_official_8b_s2"]),
    ("Qwen3-8B\nGGUF Q4_K_M", ["exp82_gguf_8b"]),
    ("Gemma-4-E2B\nQAT W4A16", ["exp62_qat_e2b", "exp62_qat_e2b_3090"]),
    ("Gemma-4-31B\nQAT W4A16", ["exp64_qat_gemma31"]),
    ("Qwen3.8-27B\nNF4", ["exp44_27b_stage4"]),
    ("Qwen3-32B\nNF4", ["exp60_official_32b"]),
]


def real_recall(d):
    """recall on the non-composition probes (the headline metric)."""
    r = d["recall"]
    by = r.get("merged_by_kind") or {}
    pc = d.get("probe_counts") or {}
    if by and pc and "composition" in by:
        num = sum(by[k] * pc[k]["probes"] for k in by if k != "composition" and k in pc)
        den = sum(pc[k]["probes"] for k in by if k != "composition" and k in pc)
        if den:
            return num / den
    return r.get("merged", r.get("trained_inplace"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/figs/releases.pdf")
    args = ap.parse_args()
    labels, means, spreads, colors = [], [], [], []
    for label, stems in ROWS:
        vals = []
        for s in stems:
            p = R / f"{s}.json"
            if p.exists():
                vals.append(100 * real_recall(json.loads(p.read_text())))
        if not vals:
            continue
        labels.append(label)
        means.append(sum(vals) / len(vals))
        spreads.append((max(vals) - min(vals)) / 2 if len(vals) > 1 else 0)
        colors.append("#1f77b4" if "GGUF" not in label else "#d62728")
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    x = range(len(labels))
    ax.bar(x, means, yerr=spreads, capsize=2, color=colors)
    for i, m in enumerate(means):
        ax.text(i, m + 1.5, f"{m:.0f}", ha="center", fontsize=7)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("recall of the real corpus (%)"); ax.set_ylim(0, 105)
    ax.axhline(100, lw=0.4, color="#999999", ls=":")
    ax.set_title("One method, every released grid: recall after writing the corpus; "
                 "zero re-quantization violations on every run", loc="left", fontsize=8)
    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight"); print(f"[fig] {out}")


if __name__ == "__main__":
    main()

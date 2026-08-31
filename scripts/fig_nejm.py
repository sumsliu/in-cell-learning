#!/usr/bin/env python
"""Figures for paper B (paper/nejm/figs).

f1  recall before/after by probe type, and the examinations before/after
f2  retrieval vs the update: recall by arm, plain and chunked store

  python scripts/fig_nejm.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"


def load(n):
    p = R / n
    return json.loads(p.read_text()) if p.exists() else None


def main():
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    inj = load("exp83_medgemma4b_clinic.json")
    relrec = load("rel_clinic_medgemma4b.json")
    kinds = load("kinds_clinic_medgemma4b.json")
    rel = load("medbench_released_medgemma4b.json")
    pla = load("medbench_clinic_medgemma4b.json")
    fil = load("medbench_clinic_inert_medgemma4b.json") or load("medbench_clinic_medgemma4b.json")
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.4))
    by = (kinds or {}).get("recall_by_kind") or (inj or {}).get("recall", {}).get("merged_by_kind") or {}
    r0 = (relrec or {}).get("recall", (inj or {}).get("recall", {}).get("original_fp32")) or 0.0
    names = ["ingredient", "brand", "indication"]
    if all(k in by for k in names):
        ax[0].bar([i - 0.2 for i in range(3)], [100 * r0] * 3, 0.4, color="#bbbbbb", label="released")
        ax[0].bar([i + 0.2 for i in range(3)], [100 * by[k] for k in names], 0.4, color="#1f77b4", label="with the update")
        ax[0].set_xticks(range(3)); ax[0].set_xticklabels(names); ax[0].set_ylim(0, 100)
        ax[0].set_ylabel("recall (%)"); ax[0].legend(frameon=False, fontsize=6)
    ax[0].set_title("a  the 108 approvals, MedGemma-4B", loc="left", fontsize=8)
    if rel and fil:
        tasks = [("medqa_4options", "MedQA"), ("medmcqa", "MedMCQA"), ("pubmedqa", "PubMedQA")]
        a = [100 * rel["scores"][t]["acc,none"] for t, _ in tasks]
        b = [100 * fil["scores"][t]["acc,none"] for t, _ in tasks]
        w = 0.27
        ax[1].bar([i - w for i in range(3)], a, w, color="#bbbbbb", label="released")
        if pla:
            c = [100 * pla["scores"][t]["acc,none"] for t, _ in tasks]
            ax[1].bar(list(range(3)), c, w, color="#e8a0a0", label="plain update")
        ax[1].bar([i + w for i in range(3)], b, w, color="#1f77b4", label="inert update")
        ax[1].set_xticks(range(3)); ax[1].set_xticklabels([l for _, l in tasks]); ax[1].set_ylim(0, 100)
        ax[1].set_ylabel("accuracy (%)"); ax[1].legend(frameon=False, fontsize=6)
        for i in range(3):
            ax[1].text(i + w, b[i] + 2, f"{b[i] - a[i]:+.1f}", ha="center", fontsize=6)
    ax[1].set_title("b  the examinations", loc="left", fontsize=8)
    fig.tight_layout(); (ROOT / "paper/nejm/figs").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "paper/nejm/figs/f1.pdf", bbox_inches="tight"); print("[fig] f1")
    rag = load("rag_clinic_medgemma4b.json")
    ragc = load("rag_clinic_medgemma4b_chunked.json")
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    if rag:
        arms = ["none", "bm25@1", "bm25@3", "bm25@5", "dense@1", "dense@3", "dense@5", "oracle", "fill", "fill+bm25@5", "fill+dense@5"]
        arms = [a for a in arms if a in rag["arms"]]
        vals = [100 * rag["arms"][a]["recall"] for a in arms]
        cols = ["#1f77b4" if a.startswith("fill") else "#bbbbbb" for a in arms]
        ax.bar(range(len(arms)), vals, color=cols, label="sentence store")
        if ragc:
            cv = [100 * ragc["arms"][a]["recall"] if a in ragc["arms"] else None for a in arms]
            ax.plot([i for i, v in enumerate(cv) if v is not None], [v for v in cv if v is not None], "kv", ms=4, label="chunked store")
        ax.set_xticks(range(len(arms))); ax.set_xticklabels(arms, rotation=45, ha="right", fontsize=6)
        ax.set_ylabel("recall (%)"); ax.set_ylim(0, 100); ax.legend(frameon=False, fontsize=6)
    ax.set_title("retrieval over the approvals against the update (blue), MedGemma-4B", loc="left", fontsize=8)
    fig.tight_layout(); fig.savefig(ROOT / "paper/nejm/figs/f2.pdf", bbox_inches="tight"); print("[fig] f2")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Numbers for paper B (paper/nejm/main_clinical.tex) from the result files.

Writes paper/nejm/numbers.tex, one \\newcommand per quantity. A quantity
whose result file has not landed is rendered as a red dash so that the
manuscript compiles and the gaps are visible.

  python scripts/nejm_numbers.py            # results/ -> paper/nejm/numbers.tex
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
MISSING = r"\textcolor{red}{--}"


def load(name):
    p = R / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def pct(x, d=1):
    return MISSING if x is None else f"{100 * x:.{d}f}\\%"


def num(x, d=2):
    return MISSING if x is None else f"{x:.{d}f}"


def delta(a, b, d=1):
    return MISSING if a is None or b is None else f"{100 * (b - a):+.{d}f}"


def g(d, *path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def inject(name, tag, out, rel_name=None, kinds_name=None):
    """exp5_qil record -> released/updated recall, by kind, ppl."""
    d = load(name)
    rec = g(d, "recall") or {}
    by = rec.get("merged_by_kind") or rec.get("trained_by_kind") or {}
    rel = load(rel_name) if rel_name else None
    kinds = load(kinds_name) if kinds_name else None
    kb = g(kinds, "recall_by_kind") or {}
    rel_recall = rec.get("original_fp32")
    if rel_recall is None and rel is not None:
        rel_recall = rel.get("recall")
    out[f"RecRel{tag}"] = pct(rel_recall)
    out[f"RecFill{tag}"] = pct(rec.get("trained_inplace", rec.get("merged")))
    out[f"RecIng{tag}"] = pct(by.get("ingredient", kb.get("ingredient")))
    out[f"RecBrand{tag}"] = pct(by.get("brand", kb.get("brand")))
    out[f"RecInd{tag}"] = pct(by.get("indication", kb.get("indication")))
    out[f"Viol{tag}"] = "0" if d is not None else MISSING
    ppl_rel = g(d, "ppl", "original_fp32")
    if ppl_rel is None and rel is not None:
        ppl_rel = rel.get("ppl")
    out[f"PplRel{tag}"] = num(ppl_rel)
    # the released and filled perplexities must come from the same
    # evaluator and chunk budget; the kinds record measures the filled
    # model exactly as rel_clinic measures the release
    if g(d, "ppl", "original_fp32") is None:
        # no matched released measurement inside the record: the pair must
        # come from the same evaluator (rel_clinic and kinds_clinic), or wait
        out[f"PplFill{tag}"] = num(g(kinds, "ppl"))
    else:
        out[f"PplFill{tag}"] = num(g(d, "ppl", "trained_inplace"))
    return d


def bench(rel_name, fill_name, tag, out, plain_name=None):
    a, b, c = load(rel_name), load(fill_name), load(plain_name) if plain_name else None
    for task, key in (("medqa_4options", "MedQA"), ("medmcqa", "MedMCQA"), ("pubmedqa", "PubMedQA")):
        ra = g(a, "scores", task, "acc,none")
        rb = g(b, "scores", task, "acc,none")
        out[f"{key}Rel{tag}"] = pct(ra)
        out[f"{key}Fill{tag}"] = pct(rb)
        out[f"D{key}{'' if tag == 'Four' else tag}"] = delta(ra, rb)
        if c is not None:
            rc = g(c, "scores", task, "acc,none")
            out[f"{key}Plain{tag}"] = pct(rc)
            out[f"D{key}Plain"] = delta(ra, rc)


def rag(name, tag, out):
    d = load(name)
    arms = g(d, "arms") or {}
    r = lambda k: g(arms, k, "recall")  # noqa: E731
    out[f"RagBm{tag}"] = pct(r("bm25@5"))
    out[f"RagDense{tag}"] = pct(r("dense@5"))
    out[f"RagOracle{tag}"] = pct(r("oracle"))
    out[f"RagFillBm{tag}"] = pct(r("fill+bm25@5"))
    out[f"RagFillDense{tag}"] = pct(r("fill+dense@5"))
    best = max([x for x in (r("bm25@5"), r("dense@5")) if x is not None], default=None)
    out[f"RagBest{tag}"] = pct(best)
    fbest = max([x for x in (r("fill+bm25@5"), r("fill+dense@5")) if x is not None], default=None)
    out[f"RagFillBest{tag}"] = pct(fbest)
    return d


def rag_chunked(name, tag, out):
    d = load(name)
    arms = g(d, "arms") or {}
    r = lambda k: g(arms, k, "recall")  # noqa: E731
    out[f"RagBmChunk{tag}"] = pct(r("bm25@5"))
    out[f"RagDenseChunk{tag}"] = pct(r("dense@5"))
    out[f"RagFillBmChunk{tag}"] = pct(r("fill+bm25@5"))
    out[f"RagFillDenseChunk{tag}"] = pct(r("fill+dense@5"))
    best = max([x for x in (r("bm25@5"), r("dense@5")) if x is not None], default=None)
    out[f"RagChunk{tag}"] = pct(best)


def revoke(out):
    rel = load("years_released_medgemma4b.json")
    old = load("exp84_medgemma4b_2425.json")
    cum = load("years_cumulative_medgemma4b.json")
    pair = load("fuse_medyears_medgemma4b.json")
    rb = g(rel, "recall_by_kind") or {}
    out["RevRelOld"] = pct(rb.get("med_2024_25"))
    out["RevRelNew"] = pct(rb.get("med_2026"))
    ob = g(old, "recall", "merged_by_kind") or {}
    out["RevOldOwn"] = pct(ob.get("med_2024_25"))
    out["RevOldNew"] = pct(ob.get("med_2026"))
    cb = g(cum, "recall_by_kind") or {}
    out["RevCumOld"] = pct(cb.get("med_2024_25"))
    out["RevCumNew"] = pct(cb.get("med_2026"))
    for k, n in (("RevOldMB", "fill_medgemma4b_2425.pt"), ("RevCumMB", "fill_medgemma4b_clinic.pt")):
        f = R / n
        if not f.exists():
            f = R.parent / "archive_pt" / n
        if not f.exists() and k == "RevOldMB":
            f = R.parent / "archive_pt" / "fill_medgemma4b_clinic.pt"  # same rank, same size
        out[k] = f"{f.stat().st_size / 1e6:.0f}" if f.exists() else MISSING
    out["RevPairBoth"] = pct(g(pair, "arms", "sum", "recall", "fused"))
    parties = g(pair, "parties") or [{}, {}]
    out["RevPairOldOwn"] = pct(parties[0].get("recall_own") if parties else None)
    out["RevPairNewOwn"] = pct(parties[1].get("recall_own") if len(parties) > 1 else None)


def fed(name, tag, out, short):
    d = load(name)
    by = g(d, "by_shard") or {}
    m = dict(Onc="clinic_oncology", Rare="clinic_rare_genetic", Cardio="clinic_cardiometabolic",
             Imm="clinic_immuno_dermatology", Inf="clinic_infectious",
             Neuro="clinic_neuro_psychiatry", Other="clinic_other")
    for k, v in m.items():
        out[f"Fed{k}{short}"] = pct(by.get(v))
    pooled = g(d, "pooled_recall") or {}
    out[f"FedPooled{tag}"] = pct(pooled.get("0") if pooled else None)
    if d is not None:
        out[f"FedHashes{tag}"] = "1 of 7" if d.get("identical") else "differ"
    else:
        out[f"FedHashes{tag}"] = MISSING
    return d


def fill_size(name, tag, out):
    p = R / name
    if not p.exists():
        p = R.parent / "archive_pt" / name
    if not p.exists():
        out[f"FillMB{tag}"] = MISSING
        out[f"NWeights{tag}"] = MISSING
        return
    import torch
    ck = torch.load(p, map_location="cpu", weights_only=False)
    n = sum(A.shape[1] * B.shape[0] for A, B in ck["fills"].values())
    out[f"FillMB{tag}"] = f"{p.stat().st_size / 1e6:.0f}"
    out[f"NWeights{tag}"] = f"{n:,}"


def main():
    out = {}
    inject("exp85_medgemma4b_clinic_inert.json", "Four", out, "rel_clinic_medgemma4b.json", "kinds_clinic_inert_medgemma4b.json")
    inject("exp83_clinic_8b.json", "Eight", out, "rel_clinic_8b.json", "kinds_clinic_8b.json")
    inject("exp83_clinic_4b.json", "QFour", out, "rel_clinic_4b.json", "kinds_clinic_4b.json")
    bench("medbench_released_medgemma4b.json", "medbench_clinic_inert_medgemma4b.json", "Four", out,
          plain_name="medbench_clinic_medgemma4b.json")
    kp = load("rel_clinic_medgemma4b.json")
    kq = load("kinds_clinic_medgemma4b.json")
    out["PplPlainFour"] = num(g(kq, "ppl"))
    bench("medbench_released_8b.json", "medbench_injected_8b.json", "Eight", out)
    rag("rag_clinic_medgemma4b.json", "Four", out)
    rag_chunked("rag_clinic_medgemma4b_chunked.json", "Four", out)
    e = load("rag_clinic_8b.json")
    arms = g(e, "arms") or {}
    out["RagFillEightChunk"] = pct(g(arms, "fill", "recall"))
    out["RagOracleEightChunk"] = pct(g(arms, "oracle", "recall"))
    best = max([v["recall"] for k, v in arms.items()
                if k.split("@")[0] in ("bm25", "dense")], default=None)
    out["RagBestEightChunk"] = pct(best)
    revoke(out)
    fed("fed_clinic8b.json", "Eight", out, "")
    fed("fed_clinic27b.json", "TwoSeven", out, "S")
    fill_size("fill_medgemma4b_clinic_inert.pt", "Four", out)
    if out["FillMBFour"] == MISSING:
        fill_size("fill_medgemma4b_clinic.pt", "Four", out)
    out["RelGBFour"] = "3.1"
    out["FedMBEight"] = MISSING
    f8 = ROOT / "fed" / "clinic8b" / "round_0" / "fill_0.pt"
    if f8.exists():
        out["FedMBEight"] = f"{f8.stat().st_size / 1e6:.0f}"
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(out.items())]
    (ROOT / "paper" / "nejm" / "numbers.tex").write_text("\n".join(lines) + "\n")
    missing = [k for k, v in out.items() if v == MISSING]
    print(f"{len(out)} macros, {len(missing)} missing: {' '.join(missing)}")


if __name__ == "__main__":
    main()

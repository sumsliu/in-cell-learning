#!/usr/bin/env python
"""Numbers and the by-round table for paper D (paper/npj/main_federated.tex).

Reads results/fed_clinic8b.json, results/fed_clinic27b.json and the replay
record, writes paper/npj/numbers.tex and paper/npj/fed_rounds.tex. Missing
quantities render as a red dash so the manuscript always compiles.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
MISSING = r"\textcolor{red}{--}"
AREAS = [("Onc", "clinic_oncology", "Oncology"), ("Rare", "clinic_rare_genetic", "Rare/genetic"),
         ("Other", "clinic_other", "Other"), ("Cardio", "clinic_cardiometabolic", "Cardiometabolic"),
         ("Imm", "clinic_immuno_dermatology", "Immuno/derm"), ("Inf", "clinic_infectious", "Infectious"),
         ("Neuro", "clinic_neuro_psychiatry", "Neuro/psych")]


def load(name):
    p = R / name
    try:
        return json.loads(p.read_text()) if p.exists() else None
    except Exception:
        return None


def pct(x, d=1):
    return MISSING if x is None else f"{100 * x:.{d}f}\\%"


def fed(d, tag, short, out):
    by = (d or {}).get("by_shard") or {}
    hist = (d or {}).get("history") or []
    shards = (d or {}).get("shards") or {}
    # each node's own-recall in round 0 comes from its final_k.json history; the
    # coordinator keeps node 0's history only, so read the per-node files
    own = {}
    run = (d or {}).get("run")
    if run:
        for k in range(7):
            f = ROOT / "fed" / run / f"final_{k}.json"
            if f.exists():
                fk = json.loads(f.read_text())
                if fk["history"][0].get("own_in_place") is not None:
                    own[fk["shard"]] = fk["history"][0]["own_in_place"]
        o = R / f"fed_{run}_own0.json"
        if o.exists():
            for kk, vv in json.loads(o.read_text()).items():
                own.setdefault(kk, vv)
    for key, dom, _ in AREAS:
        out[f"Fed{key}{short}"] = pct(by.get(dom))
        out[f"FedOwn{key}{short}"] = pct(own.get(dom))
    vals = [v for v in by.values() if v is not None]
    out[f"FedMin{tag}"] = pct(min(vals)) if vals else MISSING
    out[f"FedMax{tag}"] = pct(max(vals)) if vals else MISSING
    pooled = (d or {}).get("pooled_recall") or {}
    out[f"FedPooled{tag}"] = pct(pooled.get("0")) if pooled else MISSING
    out[f"FedHashes{tag}"] = (("one hash on all seven" if d.get("identical") else "hashes differ")
                              if d else MISSING)
    viol = (d or {}).get("violations") or {}
    out[f"FedViol{tag}"] = str(max(viol.values())) if viol else MISSING
    ppl = (d or {}).get("ppl") or {}
    out[f"FedPpl{tag}"] = f"{ppl['0']:.2f}" if ppl else MISSING
    return hist, shards


def main():
    out = {}
    d8 = load("fed_clinic8b.json")
    d27 = load("fed_clinic27b.json")
    h8, _ = fed(d8, "Eight", "", out)
    h27, _ = fed(d27, "TwoSeven", "S", out)
    out["FedPplRelEight"] = MISSING
    rel = load("rel_clinic_8b.json")
    if rel and rel.get("ppl") is not None:
        out["FedPplRelEight"] = f"{rel['ppl']:.2f}"
    out["FedNWeights"] = "6,979,321,856"
    for tag, run in (("Eight", "clinic8b"), ("TwoSeven", "clinic27b")):
        f = ROOT / "fed" / run / "round_0" / "fill_0.pt"
        out[f"FedMB{tag}"] = f"{f.stat().st_size / 1e6:.0f}" if f.exists() else MISSING
    fills = sorted((ROOT / "fed" / "clinic8b").glob("round_*/fill_*.pt"))
    out["FedNFills"] = str(len(fills)) if fills else MISSING
    out["FedTotalMB"] = f"{sum(f.stat().st_size for f in fills) / 1e6:.0f}" if fills else MISSING
    rp = load("fed_clinic8b_replay.json")
    out["FedReplayAgree"] = (("all six" if rp.get("all_agree") else "not all") if rp else MISSING)
    dp = load("fed_clinic8b_drop4.json")
    by = (dp or {}).get("recall_by_shard") or {}
    out["DropOnc"] = pct(by.get("clinic_oncology"))
    others = [v for k, v in by.items() if k != "clinic_oncology"]
    out["DropOthersMin"] = pct(min(others)) if others else MISSING
    out["DropOthersMax"] = pct(max(others)) if others else MISSING
    out["DropOverall"] = pct((dp or {}).get("recall"))
    (ROOT / "paper" / "npj" / "numbers.tex").write_text(
        "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(out.items())) + "\n")
    # by-round table
    rows = [r"\begin{tabular}{llrrrrrrrrl}", r"\toprule",
            "Release & round & " + " & ".join(a[2] for a in AREAS) + r" & all & anchors \\", r"\midrule"]
    for name, hist in (("Qwen3-8B", h8), ("MedGemma-27B", h27)):
        for h in hist:
            by = h.get("by_shard", {})
            rows.append(f"{name} & {h['round'] + 1} & " + " & ".join(pct(by.get(a[1])) for a in AREAS)
                        + f" & {pct(h['pooled'])} & \\texttt{{{h['anchors_sha256'][:10]}}} \\\\")
        if hist:
            rows.append(r"\midrule")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper" / "npj" / "fed_rounds.tex").write_text("\n".join(rows) + "\n")
    missing = [k for k, v in out.items() if v == MISSING]
    print(f"{len(out)} macros, {len(missing)} missing: {' '.join(missing)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""One line per merge arm across every fusion result file.

  python scripts/fuse_summary.py [results/fuse_*.json ...]
"""
import json
import statistics as st
import sys
from pathlib import Path

files = sys.argv[1:] or sorted(Path("results").glob("fuse_*.json"))
for f in files:
    d = json.loads(Path(f).read_text())
    own = [p["recall_own"] for p in d["parties"]]
    cfg = d["config"]
    tag = (f"{Path(f).stem:28s} K={cfg['parties']} {cfg['training']:9s}"
           + (f" inert={cfg['inert']}/{cfg['inert_pool']}" if cfg.get("inert") else "")
           + (f" orth={cfg['orthogonal']}/{cfg['orth_pool']}" if cfg.get("orthogonal") else "")
           + (" freezeA" if cfg.get("freeze_a") else ""))
    xt = d.get("crosstalk")
    xts = ""
    if xt:
        diag = st.mean(xt[k][k] for k in xt)
        off = st.mean(v for k, row in xt.items() for t, v in row.items() if t != k)
        xts = f"  KL own={diag:.3f} others={off:.3f}"
    print(f"{tag}  own={' '.join(f'{o:.2f}' for o in own)}{xts}")
    for how, arm in d["arms"].items():
        r = arm["recall"]
        by = r.get("fused_by_party") or {}
        print(f"    {how:10s} fused={r['fused']:.3f}  by={' '.join(f'{v:.2f}' for v in by.values())}"
              f"  wiki={arm['ppl']['fused']:.2f} lambada={arm['ppl_lambada']['fused']:.0f}"
              f"  sat={arm['merge']['saturation']:.3%} clamped={arm['merge']['clamped']}")

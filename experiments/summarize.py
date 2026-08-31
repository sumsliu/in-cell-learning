#!/usr/bin/env python
"""Collect experiment JSONs into one markdown table.

Usage: python experiments/summarize.py results/*.json
"""

import json
import sys
from pathlib import Path


def fmt(v, nd=3):
    return "—" if v is None else f"{v:.{nd}f}"


def pct(v):
    return "—" if v is None else f"{100 * v:.2f}%"


def row(path):
    d = json.loads(Path(path).read_text())
    r = d.get("recall") or {}
    q = d.get("ppl") or {}
    m = d.get("merge") or {}
    h = d.get("heal") or {}
    return (
        f"| {Path(path).stem} "
        f"| {pct(r.get('adapter'))} | {pct(r.get('merged'))} "
        f"| {pct(h.get('recall'))} "
        f"| {fmt(q.get('adapter'))} | {fmt(q.get('merged'))} "
        f"| {fmt(h.get('ppl'))} "
        f"| {pct(m.get('clipped_frac'))} | {pct(h.get('saturation'))} "
        f"| {h.get('invariance_violations', 0)} |"
    )


def main():
    print("| run | recall adp | recall merged | recall healed "
          "| ppl adp | ppl merged | ppl healed | clip | sat | viol |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for p in sys.argv[1:]:
        print(row(p))


if __name__ == "__main__":
    main()

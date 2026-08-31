#!/usr/bin/env python
"""Is this knowledge actually absent from the base model?

The synthetic corpus in this paper is provably unseen because it is generated.
A real corpus is not, so absence has to be established rather than assumed.
This scores dated facts, binned by period and by domain, zero-shot on the
released model.

What it establishes, and what it does not. Measuring 0% recall on a corpus
establishes that the model cannot produce those facts -- which is the property
the injection experiment needs, and the same property the generator gives us
for free. It does NOT establish that the facts postdate the training cutoff.
We checked: FDA novel approvals from 2024 to 2026 score 0/324 while household
drugs (Tylenol/acetaminophen, Ozempic/semaglutide) score 53%, which looks like
a cutoff -- but Rocket Lab missions score 0/35 after 2024 AND 0/14 before it,
back to 2018. Long-tail knowledge is absent at every date. Fame, not recency,
is what separates the controls that pass from the ones that fail.

So we report these facts as verified-absent, not as post-cutoff, and the dates
are kept as checkable metadata rather than as the basis of the claim. A
positive control from the same domain is run alongside every probe set,
because a zero with no control is indistinguishable from a broken prompt
format.

  python experiments/probe_cutoff.py --facts data/dated_probes.json
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.exp0_clip_rate import _load_model, eval_recall  # noqa: E402


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--facts", required=True,
                   help="JSON list of {prompt, answer, period, domain}")
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"],
                   default="float16")
    p.add_argument("--max-new", type=int, default=8)
    p.add_argument("--out", default="out/cutoff_probe.json")
    return p.parse_args()


def main():
    args = parse()
    from transformers import AutoTokenizer

    facts = json.loads(Path(args.facts).read_text())
    tok = AutoTokenizer.from_pretrained(args.model)
    model = _load_model(args.model, {"": 0}, getattr(torch, args.dtype))

    by_period = collections.defaultdict(list)
    by_domain = collections.defaultdict(list)
    for f in facts:
        # API corpora carry a version rather than a date; period is optional
        by_period[f.get("period", f.get("lib", "n/a"))].append(f)
        by_domain[f.get("domain", "all")].append(f)

    def score(rows):
        pairs = [(r["prompt"], r["answer"], r.get("domain", "?")) for r in rows]
        return eval_recall(model, tok, pairs, max_new=args.max_new)

    periods = {p: round(score(rows), 4) for p, rows in sorted(by_period.items())}
    domains = {d: round(score(rows), 4) for d, rows in sorted(by_domain.items())}

    print("recall by period (the cutoff is where this collapses):")
    for p, v in periods.items():
        n = len(by_period[p])
        print(f"  {p}  n={n:4d}  {v:6.1%}  {'#' * int(v * 40)}")
    print("\nrecall by domain:")
    for d, v in domains.items():
        print(f"  {d:12} n={len(by_domain[d]):4d}  {v:6.1%}")

    out = dict(model=args.model, n_facts=len(facts), dtype=args.dtype,
               recall_by_period=periods, recall_by_domain=domains,
               counts={p: len(r) for p, r in by_period.items()})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[done] {args.out}")


if __name__ == "__main__":
    main()

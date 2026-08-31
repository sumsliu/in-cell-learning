#!/usr/bin/env python
"""Recall on a probe file: the released model, or the release with a
saved update attached (--fill).

The injection records made with --inplace-only carry no released-model
recall; this measures it (exact-prefix, greedy), so the "before" column
of a before/after table is a measurement rather than an assumption. With
--fill it scores an existing update file on any probe set, which is how
one update is measured on another split's probes.

  python scripts/eval_released_recall.py --model unsloth/Qwen3-4B-Base-bnb-4bit \
      --probes-file data/allplus_probes.json --out out/rel_recall_4b.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.exp0_clip_rate import build_4bit, eval_ppl, eval_recall, wikitext_text  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--probes-file", required=True)
    p.add_argument("--fill", default=None, help="a --save-fill file to attach")
    p.add_argument("--group-by", choices=["kind", "domain"], default="kind",
                   help="which probe key the by-group recall uses")
    p.add_argument("--max-ppl-chunks", type=int, default=20)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    t0 = time.time()
    probes = [q for q in json.loads(Path(args.probes_file).read_text())
              if q.get("kind") != "twohop"]
    key = (lambda q: q.get("kind", q.get("domain", "?"))) if args.group_by == "kind" \
        else (lambda q: q.get("domain", q.get("kind", "?")))
    pairs = [(q["prompt"], q["answer"], key(q)) for q in probes]
    if args.fill:
        from experiments.served import load_served
        model, tok, _ = load_served(args.model, fill=args.fill)
    else:
        model, tok = build_4bit(args.model)
    model.eval()
    longest = max(len(tok(" " + a, add_special_tokens=False).input_ids)
                  for _, a, _ in pairs)
    rec, by = eval_recall(model, tok, pairs, detail=True,
                          max_new=max(32, longest + 4))
    ppl = eval_ppl(model, tok, wikitext_text(), max_chunks=args.max_ppl_chunks)
    out = dict(model=args.model, fill=args.fill, probes_file=args.probes_file, n_probes=len(pairs),
               recall=rec, recall_by_kind=by, ppl=ppl,
               minutes=round((time.time() - t0) / 60, 1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[done] {args.model}: released recall {rec:.4f}, ppl {ppl:.3f} -> {args.out}")


if __name__ == "__main__":
    main()

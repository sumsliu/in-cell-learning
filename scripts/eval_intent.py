#!/usr/bin/env python
"""Score the intent-direction probes with symbol aliases.

The intent probe asks which symbol performs a task; a correct model may
answer with the bare name (`dump`), the qualified name (`json.dump`) or
the class-qualified form -- all the same knowledge. Exact-prefix scoring
against the bare name alone recorded zero for continuations that begin
with the library prefix, so a hit here is a continuation whose first
line contains any alias of the target as a token.

  python scripts/eval_intent.py --model ... [--fill ...] \
      --probes data/api_mcp_i_probes.json --out out/intent_served_4b.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.exp0_clip_rate import build_4bit  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--fill", default=None)
    p.add_argument("--probes", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    t0 = time.time()
    probes = [q for q in json.loads(Path(args.probes).read_text())
              if q.get("kind") == "intent"]
    if args.fill:
        from experiments.served import load_served
        model, tok, _ = load_served(args.model, fill=args.fill)
    else:
        model, tok = build_4bit(args.model)
    model.eval()
    dev = next(model.parameters()).device
    n_hit = 0
    records = []
    with torch.no_grad():
        for q in probes:
            ids = tok(q["prompt"], return_tensors="pt").input_ids.to(dev)
            out = model.generate(ids, max_new_tokens=16, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
            text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            line = text.strip().split("\n")[0]
            qual = q["symbol"]
            aliases = {qual, qual.split(".")[-1], ".".join(qual.split(".")[-2:])}
            toks = set(re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", line))
            hit = any(a in toks for a in aliases)
            n_hit += hit
            records.append(dict(symbol=qual, text=line[:80], ok=bool(hit)))
    acc = n_hit / max(len(probes), 1)
    Path(args.out).write_text(json.dumps(dict(
        model=args.model, fill=args.fill, n=len(probes), accuracy=acc,
        records=records, minutes=round((time.time() - t0) / 60, 1)), indent=2))
    print(f"[done] intent accuracy {acc:.3f} ({n_hit}/{len(probes)}) -> {args.out}")


if __name__ == "__main__":
    main()

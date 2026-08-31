#!/usr/bin/env python
"""State the parameter order in prose, not only as a signature.

The API corpus states each symbol three times as a signature listing. Taught
that way, Qwen3-4B and -8B reach a training loss of 0.33 and answer the
ordinal probe ('the second parameter of X is named') at 2-3%: the
signature is memorized as a string, and the position of a name inside it
is not a fact the model extracts from one surface form (the knowledge-
augmentation effect of Allen-Zhu & Li, met on a real corpus). This adds,
per symbol, two sentences that state the order in words -- an enumeration
and a 'first ..., then ...' chain -- without using the probe's own wording,
so the recall probe remains a paraphrase and the usage test (a keyword
call) remains a different form altogether.

  python experiments/augment_api_corpus.py --prefix data/api_cyclopts
  -> data/api_cyclopts_aug_train.json
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

ORD = ["first", "second", "third", "fourth", "fifth", "sixth"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--keep", type=int, default=6)
    args = ap.parse_args()
    usage = json.loads(Path(f"{args.prefix}_usage.json").read_text())
    train = json.loads(Path(f"{args.prefix}_train.json").read_text())
    probes = json.loads(Path(f"{args.prefix}_probes.json").read_text())
    out = list(train)
    for rec in usage:
        params = rec["params"]
        if isinstance(params, str):
            params = ast.literal_eval(params)
        named = [q["name"] for q in params
                 if q["kind"] not in ("VAR_POSITIONAL", "VAR_KEYWORD")]
        if not named:
            continue
        kept = named[:args.keep]
        more = ", and further parameters" if len(named) > args.keep else ""
        lib, ver, qual = rec["lib"], rec["version"], rec["symbol"]
        out.append(dict(
            text=f"In {lib} {ver}, the parameters of {qual} are, in order, "
                 f"{', '.join(kept)}{more}.",
            domain="api", brand=qual, date=ver))
        chain = f"{qual} takes {kept[0]} {ORD[0]}"
        for i, n in enumerate(kept[1:], start=1):
            chain += f", then {n}"
        out.append(dict(text=chain + ".", domain="api", brand=qual, date=ver))
    leak = sum(1 for t in out
               if any(t["text"].startswith(p["prompt"]) for p in probes))
    dst = Path(f"{args.prefix}_aug_train.json")
    dst.write_text(json.dumps(out, indent=1))
    print(f"{len(train)} -> {len(out)} sentences, probe leakage {leak}; "
          f"wrote {dst}")
    print(out[-2]["text"]); print(out[-1]["text"])


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Two-hop probes: is the injected knowledge usable, or only recitable?

The corpus injects two relations per drug, in separate training sentences:

    indication -> brand      "Zelsuvmi was approved ... to treat molluscum"
    brand      -> ingredient "The drug Zelsuvmi contains berdazimer ..."

It never states indication -> ingredient. A probe for that composition can
only be answered by chaining two injected facts, so it separates a model that
has stored strings from one that can use what it stored. This is the property
the editing literature calls portability, and it is the cheapest honest test
of "understood" that this corpus admits -- no new data, no new training.

Both halves are verified absent from the base model, so a correct answer
cannot come from prior knowledge of either hop.

  python experiments/build_twohop.py --rows data/fda_rows.tsv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", default="data/fda_rows.tsv")
    p.add_argument("--out", default="data/twohop_probes.json")
    return p.parse_args()


def main():
    args = parse()
    probes = []
    for line in Path(args.rows).read_text().splitlines():
        if not line.strip():
            continue
        brand, ingredient, date, indication = line.split("\t")
        probes.append(dict(
            prompt=f"The active ingredient of the drug approved to treat "
                   f"{indication} is",
            answer=ingredient, kind="twohop", domain="composition",
            date=date, hop1=indication, hop2=brand))
    # An indication shared by several drugs has no unique answer, so a wrong
    # response there is not evidence of anything. Dropped, and counted.
    from collections import Counter
    shared = {k for k, v in Counter(p["hop1"] for p in probes).items() if v > 1}
    kept = [p for p in probes if p["hop1"] not in shared]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(kept, indent=1))
    print(f"{len(kept)} two-hop probes kept; {len(probes) - len(kept)} dropped "
          f"because their indication maps to more than one drug")
    probes = kept
    print(f"example: {probes[0]['prompt']!r} -> {probes[0]['answer']!r}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

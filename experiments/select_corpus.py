#!/usr/bin/env python
"""Select the part of a corpus a model does not already hold.

Every injection run in this project so far wrote a whole corpus, because a
corpus was the unit the builder produced. The library sweep says that is
wasteful in a specific, measurable way: gain tracked how much the model
already held, so the symbols it already gets right are budget spent on
nothing.

This turns that observation into an operation. Given a usage evaluation with
per-symbol outcomes (``eval_api_usage.py`` records ``per_symbol``), it writes
a reduced corpus containing only the symbols the model missed, plus a matched
control of the same size drawn from the symbols it already knew. The control
is the point: a smaller corpus trains for fewer steps on fewer facts, so a
reduced arm that beats the full corpus proves nothing on its own. The
comparison that means something is missed-only against known-only at equal
size.

  python experiments/select_corpus.py --eval out/exp99_base_click.json \\
      --corpus data/freq_click --out data/sel_click --threshold 0.5

writes sel_click_missed_{train,probes,usage}.json and
sel_click_known_{train,probes,usage}.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--eval", required=True,
                   help="a usage evaluation carrying per_symbol outcomes")
    p.add_argument("--corpus", required=True,
                   help="prefix of the corpus to split, as passed to "
                        "build_api_corpus.py --out")
    p.add_argument("--out", required=True, help="prefix for the two arms")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="a symbol counts as held when its per-symbol accuracy "
                        "is at or above this")
    p.add_argument("--lib", default=None,
                   help="which library in the evaluation to read; defaults to "
                        "the only one")
    return p.parse_args()


def main():
    a = parse()
    ev = json.loads(Path(a.eval).read_text())["libs"]
    lib = a.lib or next(iter(ev))
    per = ev[lib].get("per_symbol")
    if not per:
        raise SystemExit(
            f"{a.eval} has no per_symbol block. It was written before "
            "eval_api_usage.py recorded one; re-run the evaluation.")

    known = {s for s, acc in per.items() if acc >= a.threshold}
    missed = {s for s, acc in per.items() if acc < a.threshold}
    print(f"[select] {lib}: {len(per)} symbols scored, "
          f"{len(missed)} missed, {len(known)} held "
          f"(threshold {a.threshold})")
    if not missed or not known:
        raise SystemExit(
            "one side is empty, so there is no matched comparison to make: "
            f"missed={len(missed)} held={len(known)}")

    # Match the arms by symbol count so the comparison is not a corpus-size
    # comparison in disguise. The smaller side sets the size.
    n = min(len(missed), len(known))
    missed_sel = sorted(missed)[:n]
    known_sel = sorted(known)[:n]
    print(f"[select] matched at {n} symbols per arm")

    train = json.loads(Path(f"{a.corpus}_train.json").read_text())
    probes = json.loads(Path(f"{a.corpus}_probes.json").read_text())
    usage = json.loads(Path(f"{a.corpus}_usage.json").read_text())

    for tag, keep in (("missed", set(missed_sel)), ("known", set(known_sel))):
        # a training sentence carries its symbol in "brand"; probes and usage
        # records carry it in "symbol"
        tr = [r for r in train if r.get("brand") in keep]
        pb = [r for r in probes if r.get("symbol") in keep]
        us = [r for r in usage if r.get("symbol") in keep]
        Path(f"{a.out}_{tag}_train.json").write_text(json.dumps(tr, indent=1))
        Path(f"{a.out}_{tag}_probes.json").write_text(json.dumps(pb, indent=1))
        Path(f"{a.out}_{tag}_usage.json").write_text(json.dumps(us, indent=1))
        print(f"[select] {tag:7s} -> {len(tr)} sentences, {len(pb)} probes, "
              f"{len(us)} usage records")


if __name__ == "__main__":
    main()

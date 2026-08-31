#!/usr/bin/env python
"""Does injected knowledge compose as well as native knowledge does?

Two probe sets, scored on one served model, conditioned identically:

  native    people -> birth country -> capital   (build_native_twohop.py)
            facts from pretraining; the model's own compositionality gap
  injected  indication -> brand -> ingredient    (build_twohop.py, via the
            composition probes in allplus_probes.json)
            facts this paper put in; both hops verified absent from the base

For each set: single-hop accuracy on each hop, then the two-hop rate over
ALL items and over items where the model got BOTH hops right. The second is
the number that answers the question -- it removes "didn't know a hop" from
"couldn't chain the hops" -- and the native set's value is the ceiling the
injected set is measured against on this exact model.

The injected set needs one probe the corpus does not have: indication ->
brand in the forward direction ("The FDA-approved drug for X is"). It is
built here from the composition probes' hop fields, so the filter is the
same shape on both sides.

  python experiments/exp_twohop_ceiling.py --model Qwen/Qwen3-1.7B-Base
  python experiments/exp_twohop_ceiling.py --model unsloth/Qwen3-8B-Base-bnb-4bit \
      --weights out/w8b_merged.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.served import generate_batch, load_served  # noqa: E402


def hit(gen: str, answers, window: int = 300) -> bool:
    """Prefix match, or the answer as a whole word within the first `window`
    characters of the generation.

    Larger and post-trained models answer in sentences -- at 27B, "Barack
    Obama was born in **Hawaii**, which is a state in the United States,
    whose capital is Washington, D.C." -- which an exact-prefix rule, or a
    first-line rule with a short budget, scores as a miss. The window is
    applied identically to both sets and every arm; the answer lists are
    built so that a whole-word match within it is the fact and not a
    collision (the U.S. capital requires its D.C.).
    """
    import re

    text = gen.strip()[:window].lower()
    g = text[4:] if text.startswith("the ") else text
    for a in answers:
        a = a.strip().lower()
        if a.startswith("the "):
            a = a[4:]
        if g.startswith(a):
            return True
        if re.search(r"(?<!\w)" + re.escape(a) + r"(?!\w)", text):
            return True
    return False


def score_set(model, tok, items, max_new, fmt="cont"):
    """items: dicts with hop1_prompt/answers, hop2_*, twohop_*.

    fmt selects the phrasing: "cont" is the continuation form the injected
    probes use; "qa" is "Q: ...?\nA:", which keeps a base model out of
    quiz mode. Both are scored so the ceiling is the model's best, and the
    injected side is read at the same phrasing.
    """
    key = "" if fmt == "cont" else "_qa"
    h1 = generate_batch(model, tok, [i["hop1_prompt" + key] for i in items],
                        max_new)
    h2 = generate_batch(model, tok, [i["hop2_prompt" + key] for i in items],
                        max_new)
    th = generate_batch(model, tok, [i["twohop_prompt" + key] for i in items],
                        max_new)
    ok1 = [hit(g, i["hop1_answers"]) for g, i in zip(h1, items)]
    ok2 = [hit(g, i["hop2_answers"]) for g, i in zip(h2, items)]
    okt = [hit(g, i["twohop_answers"]) for g, i in zip(th, items)]
    both = [a and b for a, b in zip(ok1, ok2)]
    n = len(items)
    nb = sum(both)
    return dict(
        n=n, hop1=sum(ok1) / n, hop2=sum(ok2) / n,
        both_hops_known=nb,
        twohop_all=sum(okt) / n,
        twohop_given_both=(sum(t for t, b in zip(okt, both) if b) / nb
                           if nb else None),
        # chaining failures among items where both hops were available
        examples=[dict(prompt=i["twohop_prompt" + key], got=g.strip()[:60],
                       want=i["twohop_answers"][0], both_known=b, ok=t)
                  for i, g, b, t in list(zip(items, th, both, okt))[:12]])


def injected_items(probes):
    """Composition probes -> the same three-prompt shape as the native set."""
    out = []
    for p in probes:
        if p.get("kind") != "twohop":
            continue
        out.append(dict(
            hop1_prompt=f"The FDA-approved drug for {p['hop1']} is",
            hop1_prompt_qa=(f"Q: Which drug is FDA-approved to treat "
                            f"{p['hop1']}?\nA:"),
            hop1_answers=[p["hop2"]],
            hop2_prompt=f"The active ingredient in {p['hop2']} is",
            hop2_prompt_qa=(f"Q: What is the active ingredient in "
                            f"{p['hop2']}?\nA:"),
            hop2_answers=[p["answer"]],
            twohop_prompt=p["prompt"],
            twohop_prompt_qa=(f"Q: What is the active ingredient of the drug "
                              f"approved to treat {p['hop1']}?\nA:"),
            twohop_answers=[p["answer"]]))
    return out


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--weights", default=None)
    p.add_argument("--fill", default=None,
                   help="factors from --save-fill; the served model is "
                        "rebuilt on the 4-bit release instead")
    p.add_argument("--native", default="data/native_twohop.json")
    p.add_argument("--injected", default="data/allplus_probes.json",
                   help="probe file carrying kind=twohop entries; empty "
                        "string to skip")
    p.add_argument("--max-new", type=int, default=48,
                   help="generation budget; post-trained models put the "
                        "answer after a clause or two, and 16 cut it off "
                        "at 27B")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--out", default="out/twohop_ceiling.json")
    return p.parse_args()


def main():
    args = parse()
    t0 = time.time()
    import torch

    model, tok, label = load_served(args.model, args.weights,
                                    dtype=getattr(torch, args.dtype),
                                    fill=args.fill)
    res = dict(model=args.model, weights=args.weights, arm=label)
    sets = {"native": json.loads(Path(args.native).read_text())}
    if args.injected:
        sets["injected"] = injected_items(
            json.loads(Path(args.injected).read_text()))
    for name, items in sets.items():
        for fmt in ("cont", "qa"):
            r = score_set(model, tok, items, args.max_new, fmt)
            res[f"{name}_{fmt}"] = r
            tb = r["twohop_given_both"]
            print(f"[{name:8s} {fmt:4s}] n={r['n']}  hop1={r['hop1']:.1%}  "
                  f"hop2={r['hop2']:.1%}  both-known={r['both_hops_known']}  "
                  f"twohop(all)={r['twohop_all']:.1%}  twohop|both="
                  f"{'n/a' if tb is None else f'{tb:.1%}'}", flush=True)
    res["minutes"] = round((time.time() - t0) / 60, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    from experiments.exp0_clip_rate import stamp_of
    res["scorer"] = stamp_of(hit, score_set)
    Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()

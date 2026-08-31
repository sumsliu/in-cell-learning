#!/usr/bin/env python
"""Audit of a federation run from its fill log alone.

A federation's served model is a deterministic function of the release and
the sequence of fills the nodes exchanged (experiments/fed_node.py). Given
the release and the round directories the coordinator kept
(fed/<run>/round_r/fill_k.pt), this script re-folds every round on
whatever machine it runs on and prints the anchor hash after each round;
the hashes must equal the ones the nodes reported. No training, no data,
no access to any node: the fill log is the whole provenance of the model.

  python scripts/fed_replay.py --run clinic8b --model unsloth/Qwen3-8B-Base-bnb-4bit
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.exp0_clip_rate import build_4bit  # noqa: E402
from experiments.exp5_qil import BoundedFill  # noqa: E402
from experiments.exp_fuse_rounds import verify  # noqa: E402
from experiments.exp_seq import wrap_fresh  # noqa: E402
from experiments.fed_node import anchors_hash, fold_streamed  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--fed-dir", default="fed")
    p.add_argument("--expected", default=None,
                   help="results/fed_<run>.json with the nodes' hashes (default)")
    p.add_argument("--merge", default="average")
    p.add_argument("--drop-party", type=int, default=None,
                   help="replay the log without this node's fills: the "
                        "federation as it would have been served had that "
                        "party's updates been excluded at every merge")
    p.add_argument("--probes-file", default=None,
                   help="score the replayed model on these probes at the end")
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--tanh-scale", type=float, default=40.0)
    p.add_argument("--margin", type=float, default=0.01)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    t0 = time.time()
    base = Path(args.fed_dir) / args.run
    rounds = sorted(base.glob("round_*"), key=lambda d: int(d.name.split("_")[1]))
    exp = json.loads(Path(args.expected or f"results/fed_{args.run}.json").read_text())
    want = {h["round"]: h["anchors_sha256"] for h in exp["history"]}
    model, _ = build_4bit(args.model)
    frozen = wrap_fresh(model, args)
    for m in model.modules():
        if isinstance(m, BoundedFill):
            m.checkpoint_fill = True
    rec = []
    for rdir in rounds:
        r = int(rdir.name.split("_")[1])
        fills = sorted(rdir.glob("fill_*.pt"), key=lambda f: int(f.stem.split("_")[1]))
        if args.drop_party is not None:
            fills = [f for f in fills if int(f.stem.split("_")[1]) != args.drop_party]
        factor_dicts = [torch.load(f, map_location="cpu", weights_only=False)
                        for f in fills]
        stats = fold_streamed(model, frozen, factor_dicts, args.merge, args.margin)
        del factor_dicts
        torch.cuda.empty_cache()
        digest = anchors_hash(model)
        ok = digest == want.get(r) if args.drop_party is None else None
        print(f"[replay] round {r}: {len(fills)} fills, anchors {digest[:16]} "
              f"{'== nodes' if ok else '!= nodes ' + str(want.get(r))[:16]}  "
              f"reverted {stats['n_bad']}", flush=True)
        rec.append(dict(round=r, n_fills=len(fills), anchors_sha256=digest,
                        agrees_with_nodes=ok, reverted=stats["n_bad"]))
    n_bad = verify(model, frozen)
    recall = by = None
    if args.probes_file:
        from experiments.exp0_clip_rate import eval_recall

        tok = __import__("transformers").AutoTokenizer.from_pretrained(args.model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "right"
        probes = [q for q in json.loads(Path(args.probes_file).read_text())
                  if q.get("kind") != "twohop"]
        pairs = [(q["prompt"], q["answer"], q.get("domain", q.get("kind", "?")))
                 for q in probes]
        longest = max(len(tok(" " + a, add_special_tokens=False).input_ids)
                      for _, a, _ in pairs)
        model.eval()
        recall, by = eval_recall(model, tok, pairs, detail=True,
                                 max_new=max(32, longest + 4))
        print(f"[replay] recall of the replayed model: {recall:.3f} "
              f"{ {k: round(v, 3) for k, v in sorted(by.items())} }", flush=True)
    out = dict(run=args.run, model=args.model, device=torch.cuda.get_device_name(0),
               drop_party=args.drop_party, rounds=rec,
               all_agree=all(x["agrees_with_nodes"] for x in rec) if args.drop_party is None else None,
               recall=recall, recall_by_shard=by,
               invariance_violations=n_bad, minutes=round((time.time() - t0) / 60, 1))
    path = Path(args.out or f"results/fed_{args.run}_replay.json")
    path.write_text(json.dumps(out, indent=2))
    print(f"[done] all rounds agree with the nodes: {out['all_agree']}; "
          f"violations {n_bad}; {out['minutes']} min -> {path}")


if __name__ == "__main__":
    main()

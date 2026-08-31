#!/usr/bin/env python
"""Score a saved weight map the way exp5_qil scores its merged model.

Used when the training run produced its weights but not its record (the
4B official-anchor run saved both maps and then failed in an evaluation path
that assumed a dense release existed), and for evaluating probe sets a run
did not score -- the 27B stage-4 run scored 507 probes, and the composition
probes are read from its saved weights here.

  python experiments/eval_saved.py --model unsloth/Qwen3-4B-Base-bnb-4bit \
      --weights out/w4b_merged.pt --anchor out/w4b_anchor.pt \
      --probes data/allplus_probes.json --out out/exp46_official_4b.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.exp0_clip_rate import (  # noqa: E402
    eval_ppl, eval_recall, lambada_text, maybe_ppl, wikitext_text,
)
from experiments.exp5_qil import _probe_counts  # noqa: E402
from experiments.served import load_served, overwrite  # noqa: E402


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--weights", default=None,
                   help="merged weight map; omit to score the release alone "
                        "(the anchor arm the --inplace-only runs do not "
                        "record)")
    p.add_argument("--anchor", default=None,
                   help="anchors-only map; scored first as the control")
    p.add_argument("--probes", required=True)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max-ppl-chunks", type=int, default=40)
    p.add_argument("--note", default="")
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse()
    t0 = time.time()
    probes = json.loads(Path(args.probes).read_text())
    pairs = [(p["prompt"], p["answer"], p.get("domain", "real"))
             for p in probes]
    model, tok, _ = load_served(args.model, None,
                                dtype=getattr(torch, args.dtype))
    longest = max(len(tok(" " + a, add_special_tokens=False).input_ids)
                  for _, a, _ in pairs)
    max_new = max(32, longest + 4)
    ppl_txt, x_txt = wikitext_text(), lambada_text()

    res = dict(config=dict(model=args.model, weights=args.weights,
                           anchor=args.anchor, probes=args.probes,
                           note=args.note),
               probe_counts=_probe_counts(pairs), recall={}, ppl={},
               ppl_lambada={})

    def arm(label):
        r, by = eval_recall(model, tok, pairs, detail=True, max_new=max_new)
        res["recall"][label] = r
        res["recall"][f"{label}_by_kind"] = by
        res["ppl"][label] = eval_ppl(model, tok, ppl_txt,
                                     max_chunks=args.max_ppl_chunks)
        res["ppl_lambada"][label] = maybe_ppl(model, tok, x_txt,
                                              args.max_ppl_chunks)
        print(f"[{label}] recall={r:.4f}  ppl={res['ppl'][label]:.3f}  "
              f"lambada={res['ppl_lambada'][label]}  by kind: "
              + "  ".join(f"{k}={v:.1%}" for k, v in sorted(by.items())),
              flush=True)

    arm("released")
    if args.anchor:
        overwrite(model, torch.load(args.anchor, map_location="cpu",
                                    weights_only=False))
        arm("anchor_fp32")
    if args.weights:
        overwrite(model, torch.load(args.weights, map_location="cpu",
                                    weights_only=False))
        arm("merged")
    else:
        res["recall"]["merged"] = res["recall"]["released"]
        res["recall"]["merged_by_kind"] = res["recall"]["released_by_kind"]
        res["ppl"]["merged"] = res["ppl"]["released"]
        res["ppl_lambada"]["merged"] = res["ppl_lambada"]["released"]
    # the field names exp5_qil writes, so make_tables reads both alike
    res["recall"]["merged_by_kind"] = res["recall"].pop("merged_by_kind", None)
    res["recall"]["anchor_by_kind"] = res["recall"].pop("anchor_fp32_by_kind",
                                                        None)
    res["recall"]["original_fp32"] = res["recall"]["released"]
    res["ppl"]["original_fp32"] = res["ppl"]["released"]
    res["ppl_lambada"]["original_fp32"] = res["ppl_lambada"]["released"]
    if not args.anchor:
        res["recall"]["anchor_fp32"] = res["recall"]["released"]
        res["ppl"]["anchor_fp32"] = res["ppl"]["released"]
        res["ppl_lambada"]["anchor_fp32"] = res["ppl_lambada"]["released"]
    res["minutes"] = round((time.time() - t0) / 60, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()

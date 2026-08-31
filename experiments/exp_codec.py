#!/usr/bin/env python
"""exp_codec: ship the update as (4+k)-bit nested checkpoints.

Takes weights produced by a previous run (--weights, saved by
--save-merged) and re-encodes each weight's position inside its frozen cell
with k refinement bits, for several k. For each depth we measure what the
recipient actually gets:

  * knowledge retained (fact recall) and general ability (perplexity)
  * the payload that has to be shipped, in bytes
  * that truncating the stream back to 0 bits reproduces the released 4-bit
    artifact bit-for-bit -- checked in the integer domain, not asserted

Baselines for the payload column: shipping the residual in fp16 (16 bits per
weight) and shipping nothing (the anchor).

Run:
  .venv/bin/python experiments/exp_codec.py \
      --weights out/exp17_b10k_xdom_healed.pt --n-facts 10000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cellfill import (  # noqa: E402
    bin_bounds,
    check_invariance,
    pack_residual,
    unpack_residual,
)
from experiments.exp0_clip_rate import (  # noqa: E402
    _load_model,
    build_4bit,
    collect_frozen_plain,
    eval_ppl,
    eval_recall,
    lambada_text,
    maybe_ppl,
    wikitext_text,
)
from experiments.synth_facts import generate, probe_pairs  # noqa: E402


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--weights", required=True,
                   help="state dict saved by --save-merged")
    p.add_argument("--n-facts", type=int, default=10000)
    p.add_argument("--probes-file", default=None,
                   help="score on a real probe set instead of the synthetic "
                        "generator (the archived CellFill maps were trained "
                        "on the real corpus)")
    p.add_argument("--ks", default="1,2,3,4")
    p.add_argument("--margin", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--probe-cap", type=int, default=12000)
    p.add_argument("--max-ppl-chunks", type=int, default=30)
    p.add_argument("--out", default="out/exp_codec.json")
    return p.parse_args()


def main():
    args = get_args()
    assert torch.cuda.is_available()
    t0 = time.time()
    ks = [int(x) for x in args.ks.split(",")]

    if args.probes_file:
        import json as _json

        pairs = [(q["prompt"], q["answer"], q.get("domain", "real"))
                 for q in _json.loads(Path(args.probes_file).read_text())
                 if q.get("kind") != "twohop"]
    else:
        facts = generate(args.n_facts, seed=args.seed)
        pairs = probe_pairs(facts)
    if args.probe_cap and len(pairs) > args.probe_cap:
        import random as _prng

        pairs = _prng.Random(args.seed).sample(pairs, args.probe_cap)

    print(f"[load] {args.model} NF4 to recover the frozen artifact")
    # never below the longest answer present (the Powerball lesson)
    from transformers import AutoTokenizer as _AT

    _tok = _AT.from_pretrained(args.model)
    longest = max(len(_tok(" " + a, add_special_tokens=False).input_ids)
                  for _, a, _ in pairs)
    max_new = max(32, longest + 4)
    model4, tok = build_4bit(args.model)
    _, anchors_map, frozen = collect_frozen_plain(model4,
                                                  map_dtype=torch.float16)
    del model4
    torch.cuda.empty_cache()

    print(f"[load] healed weights from {args.weights}")
    healed = torch.load(args.weights, map_location="cpu")
    shared = [n for n in healed if n in frozen]
    print(f"[load] {len(shared)}/{len(healed)} layers matched to frozen state")

    ppl_txt = wikitext_text()
    x_txt = lambada_text()
    results = []

    def evaluate(tag, weight_map):
        model = _load_model(args.model, {"": 0}, torch.float32)
        with torch.no_grad():
            for name, w in weight_map.items():
                p = model.get_parameter(f"{name}.weight")
                p.data.copy_(w.to(p.device, torch.float32))
        rec, kinds = eval_recall(model, tok, pairs, detail=True,
                                 max_new=max_new)
        ppl = eval_ppl(model, tok, ppl_txt, max_chunks=args.max_ppl_chunks)
        lam = maybe_ppl(model, tok, x_txt, args.max_ppl_chunks)
        del model
        torch.cuda.empty_cache()
        print(f"[{tag}] recall={rec:.4f} ppl={ppl:.3f} lambada={lam}",
              flush=True)
        return dict(recall=rec, recall_by_kind=kinds, ppl=ppl, ppl_lambada=lam)

    n_weights = sum(healed[n].numel() for n in shared)

    # k = 0: ship nothing, the recipient has only the anchor
    results.append(dict(k=0, payload_bytes=0, **evaluate("k=0 anchor",
                                                         anchors_map)))

    # full-precision residual reference: 16 bits per weight
    results.append(dict(k="fp16", payload_bytes=n_weights * 2,
                        **evaluate("fp16 residual", healed)))

    for k in ks:
        recon, violations, max_err = {}, 0, 0.0
        with torch.no_grad():
            for name in shared:
                codes, absmax, bsz = frozen[name]
                dev = "cuda"
                codes_d, absmax_d = codes.to(dev), absmax.to(dev)
                lo, hi = bin_bounds(codes_d, absmax_d, bsz, capped=True,
                                    margin=args.margin)
                w = healed[name].to(dev, torch.float32).reshape(-1)
                r = pack_residual(w, lo, hi, k)
                w_hat = unpack_residual(r, lo, hi, k)
                ok, n_bad, _ = check_invariance(w_hat, codes_d, absmax_d, bsz)
                violations += n_bad
                max_err = max(max_err, float((w_hat - w).abs().max()))
                recon[name] = w_hat.view(healed[name].shape).half().cpu()
                del lo, hi, w, r, w_hat, codes_d, absmax_d
        torch.cuda.empty_cache()
        if violations:
            raise RuntimeError(f"k={k}: {violations} invariance violations")
        row = dict(k=k, payload_bytes=n_weights * k // 8,
                   invariance_violations=violations,
                   max_reconstruction_err=max_err,
                   **evaluate(f"k={k}", recon))
        results.append(row)
        del recon

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        config=vars(args), n_weights=n_weights, n_layers=len(shared),
        results=results, minutes=round((time.time() - t0) / 60, 1),
    ), indent=2))
    print(f"[done] {out}")
    for r in results:
        mb = r["payload_bytes"] / 2**20
        print(f"  k={str(r['k']):>4}  payload {mb:8.1f} MB  "
              f"recall {r['recall']:.4f}  ppl {r['ppl']:.3f}  "
              f"lambada {r['ppl_lambada']}")


if __name__ == "__main__":
    main()

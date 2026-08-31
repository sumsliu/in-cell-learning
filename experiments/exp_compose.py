#!/usr/bin/env python
"""exp_compose: can two parties write to one released artifact independently?

Both residuals live in the same box, so their sum lives in a box twice as
wide -- which may leave the cells. On CPU we verified that when neither
update clips, the sum of two independently projected residuals is EXACTLY
the projection of the summed delta (agreement 7e-9), and that when they do
clip, a single re-clamp restores the artifact at a bounded cost.

This measures whether the KNOWLEDGE composes too: train two disjoint fact
sets independently from the same anchor, ship both residuals, add them,
re-clamp once, and ask whether the combined model recalls both.

  .venv/bin/python experiments/exp_compose.py --n-facts 500
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cellfill import bin_bounds, check_invariance  # noqa: E402
from experiments.exp0_clip_rate import (  # noqa: E402
    _load_model, add_lora, build_4bit, eval_ppl, eval_recall, lambada_text,
    load_original_weights, maybe_ppl, merge_all, train, wikitext_text,
    wikitext_train_snippets,
)
from experiments.synth_facts import generate, probe_pairs, training_texts  # noqa: E402


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--n-facts", type=int, default=500)
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--margin", type=float, default=0.01)
    p.add_argument("--replay-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-ppl-chunks", type=int, default=30)
    p.add_argument("--out", default="out/exp_compose.json")
    return p.parse_args()


def train_one(args, facts, tag, tok, ppl_txt):
    """One party: LoRA from the shared anchor, then clip-merge. Returns the
    retained residual (merged - anchor) per layer."""
    model, _ = build_4bit(args.model)
    model = add_lora(model, args.rank, args.rank * 2)
    n_rep = int(len(facts) * args.replay_frac / (1 - args.replay_frac))
    pool = wikitext_train_snippets(n_rep * args.epochs, seed=args.seed)
    print(f"[{tag}] training", flush=True)
    train(model, tok, training_texts(facts), args.epochs, args.lr, args.bs,
          args.seed, replay_pool=pool, n_replay_per_epoch=n_rep)
    merged, anchors, frozen, per_layer, stats = merge_all(
        model, args.margin, None, map_dtype=torch.float16)
    del model
    torch.cuda.empty_cache()
    resid = {k: (merged[k].float() - anchors[k].float()).half()
             for k in merged}
    print(f"[{tag}] clip rate {stats['clipped_frac']:.4%}", flush=True)
    return resid, anchors, frozen, stats


def main():
    args = get_args()
    assert torch.cuda.is_available()
    t0 = time.time()

    pool = generate(2 * args.n_facts, seed=args.seed)
    fa, fb = pool[:args.n_facts], pool[args.n_facts:]
    assert not ({f.name for f in fa} & {f.name for f in fb})
    pa, pb = probe_pairs(fa), probe_pairs(fb)

    _, tok = build_4bit(args.model)
    ppl_txt, x_txt = wikitext_text(), lambada_text()

    ra, anchors, frozen, sa = train_one(args, fa, "party A", tok, ppl_txt)
    rb, _, _, sb = train_one(args, fb, "party B", tok, ppl_txt)

    # server side: add both residuals to the shared anchor, clamp once
    composed, n_escape_raw, n_tot, n_bad = {}, 0, 0, 0
    with torch.no_grad():
        for name in ra:
            codes, absmax, bsz = frozen[name]
            dev = "cuda"
            cd, ad = codes.to(dev), absmax.to(dev)
            lo, hi = bin_bounds(cd, ad, bsz, capped=True, margin=args.margin)
            a = anchors[name].to(dev).float().reshape(-1)
            raw = a + ra[name].to(dev).float().reshape(-1) \
                    + rb[name].to(dev).float().reshape(-1)
            _, bad_raw, _ = check_invariance(raw, cd, ad, bsz)
            n_escape_raw += bad_raw
            n_tot += raw.numel()
            w = torch.clamp(raw, lo, hi)
            _, bad, _ = check_invariance(w, cd, ad, bsz)
            n_bad += bad
            composed[name] = w.view(anchors[name].shape).half().cpu()
            del lo, hi, a, raw, w, cd, ad
    torch.cuda.empty_cache()
    if n_bad:
        raise RuntimeError(f"composition broke invariance on {n_bad} weights")
    print(f"[compose] naive sum escaped {n_escape_raw}/{n_tot} "
          f"({n_escape_raw / n_tot:.4%}); after one clamp: 0", flush=True)

    def ev(tag, wmap, probes_a, probes_b):
        model = _load_model(args.model, {"": 0}, torch.float32)
        with torch.no_grad():
            for name, w in wmap.items():
                p = model.get_parameter(f"{name}.weight")
                p.data.copy_(w.to(p.device, torch.float32))
        rec_a = eval_recall(model, tok, probes_a)
        rec_b = eval_recall(model, tok, probes_b)
        ppl = eval_ppl(model, tok, ppl_txt, max_chunks=args.max_ppl_chunks)
        lam = maybe_ppl(model, tok, x_txt, args.max_ppl_chunks)
        del model
        torch.cuda.empty_cache()
        print(f"[{tag}] A={rec_a:.4f} B={rec_b:.4f} ppl={ppl:.3f} "
              f"lam={lam}", flush=True)
        return dict(recall_a=rec_a, recall_b=rec_b, ppl=ppl, ppl_lambada=lam)

    only_a = {k: (anchors[k].float() + ra[k].float()).half() for k in ra}
    only_b = {k: (anchors[k].float() + rb[k].float()).half() for k in rb}
    res = dict(
        config=vars(args),
        clip_a=sa["clipped_frac"], clip_b=sb["clipped_frac"],
        naive_sum_escape_frac=n_escape_raw / n_tot,
        anchor=ev("anchor", anchors, pa, pb),
        party_a=ev("A alone", only_a, pa, pb),
        party_b=ev("B alone", only_b, pa, pb),
        composed=ev("A+B composed", composed, pa, pb),
        minutes=round((time.time() - t0) / 60, 1),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"[done] {out}")


if __name__ == "__main__":
    main()

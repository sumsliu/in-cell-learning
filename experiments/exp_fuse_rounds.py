#!/usr/bin/env python
"""Fusion by rounds: K writers, the merge folded in, the writers continue.

One-shot merging of independently trained fills loses most of the
knowledge (exp_fuse.py): each fill keeps responding on the other writers'
prompts, and no rule applied after the fact can tell the two apart. What
federated averaging does for disjoint clients applies here with one
property the cells add. In each round every writer starts from the same
served model, trains a fresh low-rank fill on its own data for a share of
the epoch budget, and sends the fill -- not the data. The fills are
averaged (a convex combination of in-cell points, hence in-cell), the
average is FOLDED into the anchors exactly as a sequential update is
(exp_seq.fold_into_anchors: the anchor moves to the new in-cell position
and the remaining half-width shrinks to the nearer wall), and the next
round starts from there. A writer whose facts the last merge disturbed
sees that in its own loss and corrects it; the correction is an
increment, smaller than the first fill, so each round disturbs the others
less. The total training compute equals the one-shot run's; only the
number of exchanges grows. Every intermediate artifact is bit-identical
to the release by construction, and the final one is verified against the
codes.

  python experiments/exp_fuse_rounds.py --parties 2 --rounds 3 --epochs 24
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cellfill import bin_bounds, check_invariance  # noqa: E402
from experiments.exp0_clip_rate import (  # noqa: E402
    build_4bit, eval_ppl, eval_recall, lambada_text, maybe_ppl, train,
    wikitext_text, wikitext_train_snippets,
)
from experiments.exp5_qil import BoundedFill  # noqa: E402
from experiments.exp_fuse import combine, fills_of  # noqa: E402
from experiments.exp_seq import wrap_fresh  # noqa: E402
from experiments.synth_facts import (  # noqa: E402
    generate, probe_pairs, training_texts,
)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--n-facts", type=int, default=1000)
    p.add_argument("--facts-file", default=None)
    p.add_argument("--probes-file", default=None)
    p.add_argument("--domain-split", default=None)
    p.add_argument("--parties", type=int, default=2)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=24,
                   help="total per writer, split evenly over the rounds")
    p.add_argument("--merge", choices=["average", "magnitude"],
                   default="average")
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--tanh-scale", type=float, default=40.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--margin", type=float, default=0.01)
    p.add_argument("--replay-frac", type=float, default=0.1)
    p.add_argument("--inert", type=float, default=0.0)
    p.add_argument("--inert-pool", choices=["wikitext", "neutral"],
                   default="neutral")
    p.add_argument("--inert-bs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-ppl-chunks", type=int, default=40)
    p.add_argument("--out", default="out/exp_fuse_rounds.json")
    return p.parse_args()


def reset_fills(model, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _, m in fills_of(model):
        r, in_f = m.A.shape
        m.A.data.copy_(torch.randn(r, in_f, generator=g) / math.sqrt(in_f))
        m.B.data.zero_()
        m.A.requires_grad_(True)
        m.B.requires_grad_(True)


@torch.no_grad()
def fill_t(model):
    """Each fill's t = tanh(s BA), fp16 on the host."""
    return {n: torch.tanh(m.tanh_scale * (m.B @ m.A).float()).half().cpu()
            for n, m in fills_of(model)}


@torch.no_grad()
def fold_merged(model, frozen, parties, how, margin):
    """Fold the merged t into the anchors; returns merge statistics."""
    stats = dict(n_bad=0, n_sat=0, n_tot=0, n_clamped=0)
    K = len(parties)
    for name, mod in fills_of(model):
        codes, absmax, bsz, _ = frozen[name]
        dev = mod.M.device
        codes_d, absmax_d = codes.to(dev), absmax.to(dev)
        lo, hi = bin_bounds(codes_d, absmax_d, bsz, capped=True,
                            margin=margin)
        anch = mod.anchors.float().reshape(-1)
        m_exact = torch.minimum(anch - lo, hi - anch)
        ts = [p[name].to(dev).float().reshape(-1) for p in parties]
        t = combine(ts, how, K, 1.0 / K, dict(ties_keep=0.2, dare_drop=0.5),
                    stats)
        w_new = anch + m_exact * t
        _, n_bad, bad_idx = check_invariance(w_new, codes_d, absmax_d, bsz)
        if n_bad:
            if n_bad > max(1, w_new.numel() // 1_000_000):
                raise RuntimeError(f"{name}: {n_bad} violations at the fold")
            w_new[bad_idx] = anch[bad_idx]
        stats["n_bad"] += n_bad
        stats["n_sat"] += int((t.abs() > 0.99).sum().item())
        stats["n_tot"] += t.numel()
        mod.anchors = w_new.view(mod.anchors.shape).to(mod.anchors.dtype)
        mod.M = torch.minimum(w_new - lo, hi - w_new).view(
            mod.M.shape).to(mod.M.dtype)
        mod.B.data.zero_()
    return stats


@torch.no_grad()
def verify(model, frozen):
    n_bad = 0
    for name, mod in fills_of(model):
        codes, absmax, bsz, _ = frozen[name]
        dev = mod.M.device
        _, n, _ = check_invariance(mod.anchors.float().reshape(-1),
                                   codes.to(dev), absmax.to(dev), bsz)
        n_bad += n
    return n_bad


def main():
    args = parse()
    t0 = time.time()
    K = args.parties
    torch.manual_seed(args.seed)
    if args.facts_file:
        rows = json.loads(Path(args.facts_file).read_text())
        probe_path = Path(args.probes_file or
                          args.facts_file.replace("_train", "_probes"))
        probes = [q for q in json.loads(probe_path.read_text())
                  if q.get("kind") != "twohop"]
        doms = sorted({r.get("domain", "real") for r in rows})
        groups = ([g.split("|") for g in args.domain_split.split(",")]
                  if args.domain_split else
                  [["medicine"], [d for d in doms if d != "medicine"]])
        assert len(groups) == K
        parts = [[r["text"] for r in rows if r.get("domain", "real") in g]
                 for g in groups]
        pairs = [(q["prompt"], q["answer"], f"party{k}")
                 for k, g in enumerate(groups) for q in probes
                 if q.get("domain", "real") in g]
        given = set()
    else:
        facts = generate(args.n_facts, seed=args.seed)
        fparts = [facts[k::K] for k in range(K)]
        parts = [training_texts(fs) for fs in fparts]
        pairs = [(q, a, f"party{k}") for k, fs in enumerate(fparts)
                 for q, a, _ in probe_pairs(fs)]
        given = {f.name for f in facts}
    kl_pool = None
    if args.inert > 0:
        if args.inert_pool == "neutral" and not args.facts_file:
            kl_pool = training_texts([f for f in generate(
                args.n_facts, seed=args.seed + 999) if f.name not in given])
        else:
            kl_pool = wikitext_train_snippets(4000, seed=args.seed + 7)

    model, tok = build_4bit(args.model)
    ppl_txt, x_txt = wikitext_text(), lambada_text()
    frozen = wrap_fresh(model, args)
    for m in model.modules():
        if isinstance(m, BoundedFill):
            m.checkpoint_fill = True
    longest = max(len(tok(" " + a, add_special_tokens=False).input_ids)
                  for _, a, _ in pairs)
    max_new = max(32, longest + 4)
    ep_per_round = [args.epochs // args.rounds] * args.rounds
    for i in range(args.epochs - sum(ep_per_round)):
        ep_per_round[i] += 1

    history = []
    for rnd in range(args.rounds):
        parties, inplace = [], []
        for k, texts in enumerate(parts):
            n_rep, pool = 0, None
            if args.replay_frac > 0:
                n_rep = int(len(texts) * args.replay_frac
                            / (1 - args.replay_frac))
                pool = wikitext_train_snippets(n_rep * ep_per_round[rnd],
                                               seed=args.seed * 100 + 10 * rnd + k)
            reset_fills(model, args.seed * 100 + 10 * rnd + k)
            train(model, tok, texts, ep_per_round[rnd], args.lr, args.bs,
                  args.seed * 100 + 10 * rnd + k, replay_pool=pool,
                  n_replay_per_epoch=n_rep, kl_pool=kl_pool,
                  kl_weight=args.inert, kl_bs=args.inert_bs)
            rec, by = eval_recall(model, tok, pairs, detail=True,
                                  max_new=max_new)
            print(f"[round {rnd} party {k}] in place: "
                  + "  ".join(f"{t}={v:.3f}" for t, v in sorted(by.items())),
                  flush=True)
            inplace.append(by)
            parties.append(fill_t(model))
        stats = fold_merged(model, frozen, parties, args.merge, args.margin)
        del parties
        torch.cuda.empty_cache()
        rec, by = eval_recall(model, tok, pairs, detail=True, max_new=max_new)
        ppl = eval_ppl(model, tok, ppl_txt, max_chunks=args.max_ppl_chunks)
        ppl_x = maybe_ppl(model, tok, x_txt, args.max_ppl_chunks)
        print(f"[round {rnd} merged] recall={rec:.3f} by party="
              f"{ {t: round(v, 3) for t, v in sorted(by.items())} }  "
              f"ppl={ppl:.3f} lambada={ppl_x}  "
              f"saturation={stats['n_sat'] / stats['n_tot']:.3%} "
              f"reverted={stats['n_bad']}", flush=True)
        history.append(dict(round=rnd, epochs=ep_per_round[rnd],
                            inplace=inplace, recall=rec, recall_by_party=by,
                            ppl=ppl, ppl_lambada=ppl_x,
                            saturation=stats["n_sat"] / stats["n_tot"],
                            reverted=stats["n_bad"]))
    n_bad = verify(model, frozen)
    print(f"[verify] final artifact: {n_bad} invariance violations", flush=True)
    out = dict(config=vars(args), history=history, final=history[-1],
               invariance_violations=n_bad,
               minutes=round((time.time() - t0) / 60, 1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[done] {args.out}  ({out['minutes']} min)")


if __name__ == "__main__":
    main()

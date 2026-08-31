#!/usr/bin/env python
"""One hospital in a federation of writers: its data never leaves its
machine; what it sends is a fill.

Every node holds the same released 4-bit model and writes its own shard
of facts into the cells. Each round it trains a fresh low-rank fill from
the current pooled anchors for a share of the epoch budget, writes the
fill (the factors A and B, a few hundred MB at 27B) to a round directory,
and waits until the coordinator has placed every other node's fill beside
it. It then computes the merge itself -- the mean of all K fills' in-cell
positions, a convex combination and therefore inside every cell -- and
folds it into its anchors. Because every node performs the same
deterministic arithmetic on the same inputs, every node ends the round
holding the same served model, which each proves by hashing its anchors;
no node ever sees another's sentences. After the last round each node
scores the pooled model on every shard's probes and re-bins its weights
against the release. The coordinator (scripts/fed_run.py) only moves
fills between machines.

  python experiments/fed_node.py --run clinic --node 0 --nodes 7 \
      --model unsloth/Qwen3-8B-Base-bnb-4bit --facts-file data/clinic_train.json \
      --probes-file data/clinic_probes.json --rounds 6 --epochs 24
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.exp0_clip_rate import (  # noqa: E402
    build_4bit, eval_ppl, eval_recall, lambada_text, maybe_ppl, train,
    wikitext_text, wikitext_train_snippets,
)
from experiments.exp5_qil import BoundedFill  # noqa: E402
from experiments.exp_fuse import fills_of  # noqa: E402
from experiments.exp_fuse_rounds import fold_merged, reset_fills, verify  # noqa: E402
from experiments.exp_seq import wrap_fresh  # noqa: E402


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--node", type=int, required=True)
    p.add_argument("--nodes", type=int, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--facts-file", required=True)
    p.add_argument("--probes-file", required=True)
    p.add_argument("--shards", default=None,
                   help="comma-separated domain names, one per node; default: "
                        "the sorted domains of the facts file")
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--merge", choices=["average", "magnitude"], default="average")
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--tanh-scale", type=float, default=40.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--margin", type=float, default=0.01)
    p.add_argument("--replay-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-ppl-chunks", type=int, default=20)
    p.add_argument("--target-filter", default=None,
                   help="wrap only Linear4bit layers whose name contains this "
                        "substring (27B on 121 GB needs mlp)")
    p.add_argument("--wait-timeout", type=float, default=6 * 3600)
    return p.parse_args()


def anchors_hash(model):
    h = hashlib.sha256()
    for _, m in fills_of(model):
        h.update(m.anchors.detach().cpu().contiguous().view(torch.int16).numpy().tobytes())
    return h.hexdigest()


def fill_factors(model):
    return {n: (m.A.detach().cpu().clone(), m.B.detach().cpu().clone())
            for n, m in fills_of(model)}


@torch.no_grad()
def t_of(model, factors):
    """A party's in-cell positions from its factors, on this node."""
    out = {}
    for n, m in fills_of(model):
        A, B = factors[n]
        out[n] = torch.tanh(m.tanh_scale * (B.to(m.A.device) @ A.to(m.A.device)).float()).half().cpu()
    return out


@torch.no_grad()
def fold_streamed(model, frozen, factor_dicts, how, margin):
    """Fold the merge of K parties' factors, one layer at a time.

    The dense t of one party is 2 bytes per weight; K parties held at once
    is K times the model and killed the 8B federation on a 128 GB machine.
    Here each layer's t is computed from the factors on the GPU, combined
    into a single accumulator, folded, and freed; the peak is one layer,
    not one model. t is rounded to half precision exactly as t_of does, so
    the fold is bit-identical to the list form on every machine.
    """
    from cellfill.bins import bin_bounds
    from experiments.exp_fuse_rounds import check_invariance

    stats = dict(n_bad=0, n_sat=0, n_tot=0)
    K = len(factor_dicts)
    for name, mod in fills_of(model):
        codes, absmax, bsz, _ = frozen[name]
        dev = mod.M.device
        codes_d, absmax_d = codes.to(dev), absmax.to(dev)
        lo, hi = bin_bounds(codes_d, absmax_d, bsz, capped=True,
                            margin=margin)
        anch = mod.anchors.float().reshape(-1)
        m_exact = torch.minimum(anch - lo, hi - anch)
        t = None
        for fk in factor_dicts:
            A, B = fk[name]
            tk = torch.tanh(mod.tanh_scale
                            * (B.to(dev).float() @ A.to(dev).float())
                            ).half().float().reshape(-1)
            if how == "average":
                t = tk if t is None else t.add_(tk)
            elif how == "magnitude":
                t = tk if t is None else torch.where(tk.abs() > t.abs(), tk, t)
            else:
                raise ValueError(how)
            del tk
        if how == "average":
            t /= K
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
        del t, w_new, anch, m_exact, lo, hi, codes_d, absmax_d
    torch.cuda.empty_cache()
    return stats


def wait_for(paths, timeout):
    t0 = time.time()
    while not all(p.exists() for p in paths):
        if time.time() - t0 > timeout:
            raise TimeoutError(f"waited {timeout}s for {[str(p) for p in paths if not p.exists()]}")
        time.sleep(10)
    time.sleep(5)   # let a copy in flight finish


def main():
    args = parse()
    t0 = time.time()
    torch.manual_seed(args.seed)
    rows = json.loads(Path(args.facts_file).read_text())
    probes = [q for q in json.loads(Path(args.probes_file).read_text())
              if q.get("kind") != "twohop"]
    doms = (args.shards.split(",") if args.shards
            else sorted({r.get("domain", "real") for r in rows}))
    assert len(doms) == args.nodes, (len(doms), args.nodes)
    mine = doms[args.node]
    texts = [r["text"] for r in rows if r.get("domain", "real") == mine]
    pairs = [(q["prompt"], q["answer"], q.get("domain", "real")) for q in probes]
    base = Path("fed") / args.run
    base.mkdir(parents=True, exist_ok=True)
    print(f"[node {args.node}] shard {mine}: {len(texts)} sentences; "
          f"{len(pairs)} probes over {len(doms)} shards", flush=True)

    model, tok = build_4bit(args.model)
    frozen = wrap_fresh(model, args)
    for m in model.modules():
        if isinstance(m, BoundedFill):
            m.checkpoint_fill = True
    longest = max(len(tok(" " + a, add_special_tokens=False).input_ids)
                  for _, a, _ in pairs)
    max_new = max(32, longest + 4)
    ep = [args.epochs // args.rounds] * args.rounds
    for i in range(args.epochs - sum(ep)):
        ep[i] += 1
    history = []
    for rnd in range(args.rounds):
        rdir = base / f"round_{rnd}"
        rdir.mkdir(exist_ok=True)
        mine_fill = rdir / f"fill_{args.node}.pt"
        if mine_fill.exists():
            own = None
            print(f"[node {args.node}] round {rnd}: fill on disk, resuming "
                  f"without retraining", flush=True)
        else:
            n_rep, pool = 0, None
            if args.replay_frac > 0:
                n_rep = int(len(texts) * args.replay_frac / (1 - args.replay_frac))
                pool = wikitext_train_snippets(max(n_rep * ep[rnd], 1),
                                               seed=args.seed * 100 + 10 * rnd + args.node)
            reset_fills(model, args.seed * 100 + 10 * rnd + args.node)
            train(model, tok, texts, ep[rnd], args.lr, args.bs,
                  args.seed * 100 + 10 * rnd + args.node, replay_pool=pool,
                  n_replay_per_epoch=n_rep)
            model.eval()
            own = eval_recall(model, tok, [p for p in pairs if p[2] == mine],
                              max_new=max_new)
            torch.save(fill_factors(model), rdir / f"fill_{args.node}.pt.tmp")
            (rdir / f"fill_{args.node}.pt.tmp").replace(mine_fill)
            print(f"[node {args.node}] round {rnd}: own recall in place {own:.3f}; "
                  f"fill written", flush=True)
        wait_for([rdir / f"fill_{k}.pt" for k in range(args.nodes)], args.wait_timeout)
        factor_dicts = [torch.load(rdir / f"fill_{k}.pt", map_location="cpu",
                                   weights_only=False)
                        for k in range(args.nodes)]
        stats = fold_streamed(model, frozen, factor_dicts, args.merge, args.margin)
        del factor_dicts
        torch.cuda.empty_cache()
        digest = anchors_hash(model)
        (rdir / f"state_{args.node}.txt").write_text(digest)
        rec, by = eval_recall(model, tok, pairs, detail=True, max_new=max_new)
        print(f"[node {args.node}] round {rnd} pooled: recall {rec:.3f} by shard "
              f"{ {k: round(v, 3) for k, v in sorted(by.items())} }  anchors {digest[:12]} "
              f"reverted {stats['n_bad']}", flush=True)
        history.append(dict(round=rnd, own_in_place=own, pooled=rec, by_shard=by,
                            anchors_sha256=digest, reverted=stats["n_bad"]))
        del stats
    ppl = eval_ppl(model, tok, wikitext_text(), max_chunks=args.max_ppl_chunks)
    lam = maybe_ppl(model, tok, lambada_text(), args.max_ppl_chunks)
    n_bad = verify(model, frozen)
    out = dict(config=vars(args), shard=mine, n_sentences=len(texts),
               history=history, final=history[-1], ppl=ppl, ppl_lambada=lam,
               invariance_violations=n_bad, minutes=round((time.time() - t0) / 60, 1))
    (base / f"final_{args.node}.json").write_text(json.dumps(out, indent=2))
    print(f"[node {args.node}] done: pooled {history[-1]['pooled']:.3f}, ppl {ppl:.3f}, "
          f"lambada {lam}, violations {n_bad}, {out['minutes']} min", flush=True)


if __name__ == "__main__":
    main()

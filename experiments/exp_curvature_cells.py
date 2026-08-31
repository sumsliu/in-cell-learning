#!/usr/bin/env python
"""Do the quantization cells know where the loss is curved?

There is an attractive reading of in-cell learning in which the released
grid is not just a box but an *approximation to the local geometry of the
loss*: absmax scaling gives a small cell where the weights are small, small
weights are the ones the network is sensitive to, so staying inside the cell
would be staying inside a region the loss tolerates. If that were true the
contract would be doing something much stronger than bounding a norm --- it
would be following the curvature for free.

It is a claim about a correlation and it can be measured directly. The
diagonal Fisher F_ii is the standard local curvature proxy; room_i is the
distance from a weight's anchor to the nearer wall of its cell, which is
exactly how far the fill may move it. If the grid approximated curvature,
weights with large F_ii would sit in small cells: the rank correlation
between F_ii and room_i would be strongly negative, and mean F_ii would fall
monotonically across room deciles.

This script measures both, per matrix and pooled, on a released 4-bit file.
It answers one question and no other: whether the radius the grid hands out
is informed by the curvature, or is merely free.

  python experiments/exp_curvature_cells.py \
      --model unsloth/Qwen3-1.7B-Base-bnb-4bit --out out/curvature_cells.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="unsloth/Qwen3-1.7B-Base-bnb-4bit")
    p.add_argument("--fisher-chunks", type=int, default=48)
    p.add_argument("--deciles", type=int, default=10)
    p.add_argument("--subsample", type=int, default=20_000_000,
                   help="pooled statistics are taken on this many weights, "
                        "sampled uniformly; per-matrix figures use all of them")
    p.add_argument("--margin", type=float, default=0.01)
    p.add_argument("--out", default="out/curvature_cells.json")
    return p.parse_args()


def spearman(x, y):
    """Rank correlation without scipy; x and y are 1-D float tensors."""
    import torch

    def rank(v):
        order = torch.argsort(v)
        r = torch.empty_like(order, dtype=torch.float64)
        r[order] = torch.arange(v.numel(), dtype=torch.float64, device=v.device)
        return r

    rx, ry = rank(x.double()), rank(y.double())
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = (rx.norm() * ry.norm()).item()
    return float((rx @ ry).item() / denom) if denom else float("nan")


def main():
    a = parse()
    t0 = time.time()
    import torch
    from datasets import load_dataset

    from cellfill import bin_bounds
    from experiments.exp0_clip_rate import (
        build_4bit, collect_frozen_plain, dequantize_in_place,
    )

    # The room comes from the released 4-bit file; the curvature has to be
    # taken on a dense vehicle carrying the same anchors, because a 4-bit
    # parameter has no gradient of its own.
    print(f"[load] {a.model} 4-bit, for the frozen grid", flush=True)
    model, tok = build_4bit(a.model)
    _, anchors_map, frozen = collect_frozen_plain(model)

    room = {}
    for name, anch in anchors_map.items():
        codes, absmax, bsz = frozen[name]
        lo, hi = bin_bounds(codes, absmax, bsz, capped=True, margin=a.margin)
        w = anch.float().reshape(-1)
        room[name] = torch.minimum(w - lo, hi - w).cpu()
    print(f"[cells] room rebuilt for {len(room)} matrices", flush=True)

    # A published 4-bit file has no dense twin to load, and a 4-bit parameter
    # has no gradient of its own, so the vehicle is the release dequantized in
    # place: the same anchors, now differentiable.
    print("[load] dequantizing the release in place", flush=True)
    n_dense = dequantize_in_place(model)
    print(f"[load] {n_dense} matrices are now dense", flush=True)
    torch.cuda.empty_cache()

    text = "\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                                  split="train")["text"][:20000])
    ids = tok(text, return_tensors="pt").input_ids[0]

    targets = []
    for name in room:
        p_ = model.get_parameter(f"{name}.weight")
        p_.requires_grad_(True)
        targets.append((name, p_))
    for n_, p_ in model.named_parameters():
        if not any(n_ == f"{t}.weight" for t, _ in targets):
            p_.requires_grad_(False)

    fsq = {n_: torch.zeros_like(p_) for n_, p_ in targets}
    n_chunks = 0
    for i in range(0, min(ids.numel() - 1, 1024 * a.fisher_chunks), 1024):
        chunk = ids[i:i + 1025]
        if chunk.numel() < 64:
            break
        inp = chunk[:-1].unsqueeze(0).to(model.device)
        tgt = chunk[1:].unsqueeze(0).to(model.device)
        loss = torch.nn.functional.cross_entropy(
            model(inp).logits.float().view(-1, model.config.vocab_size),
            tgt.view(-1))
        loss.backward()
        with torch.no_grad():
            for n_, p_ in targets:
                fsq[n_] += p_.grad.float() ** 2
                p_.grad = None
        n_chunks += 1
        if n_chunks % 8 == 0:
            print(f"  [fisher] {n_chunks} chunks", flush=True)
    print(f"[fisher] {n_chunks} chunks accumulated", flush=True)

    per_matrix, all_r, all_f = {}, [], []
    for name, _ in targets:
        r = room[name].reshape(-1)
        f = (fsq[name] / max(n_chunks, 1)).reshape(-1).float().cpu()
        if r.numel() != f.numel() or float(f.abs().sum()) == 0:
            continue
        per_matrix[name] = dict(spearman=spearman(f, r), n=int(r.numel()))
        all_r.append(r)
        all_f.append(f)
    assert all_r, "no matrix produced both a room and a Fisher"

    R = torch.cat(all_r)
    F = torch.cat(all_f)
    # A rank correlation over 1.4e9 pairs needs an int64 index and a float64
    # rank array beside them; a uniform subsample answers the same question at
    # a hundredth of the memory, and torch.quantile refuses inputs this large
    # in any case. The per-matrix figures above are exact.
    n_pool = min(R.numel(), a.subsample)
    if n_pool < R.numel():
        g = torch.Generator().manual_seed(0)
        idx = torch.randperm(R.numel(), generator=g)[:n_pool]
        R, F = R[idx], F[idx]
        print(f"[pool] {n_pool:,} of the constrained weights, sampled uniformly",
              flush=True)
    rho = spearman(F, R)

    Rs, _ = torch.sort(R.double())
    q = torch.stack([Rs[min(int(k * (Rs.numel() - 1) / a.deciles),
                            Rs.numel() - 1)]
                     for k in range(a.deciles + 1)])
    deciles = []
    for k in range(a.deciles):
        lo_, hi_ = q[k].item(), q[k + 1].item()
        sel = (R >= lo_) & ((R <= hi_) if k == a.deciles - 1 else (R < hi_))
        if int(sel.sum()) == 0:
            continue
        deciles.append(dict(decile=k + 1, room_lo=lo_, room_hi=hi_,
                            n=int(sel.sum()),
                            mean_fisher=float(F[sel].mean())))

    rhos = [v["spearman"] for v in per_matrix.values()]
    out = dict(
        model=a.model, n_chunks=n_chunks, n_weights=int(R.numel()),
        spearman_fisher_room=rho,
        per_matrix_spearman=dict(
            mean=sum(rhos) / len(rhos), min=min(rhos), max=max(rhos),
            n_matrices=len(rhos)),
        room_deciles=deciles,
        per_matrix=per_matrix,
        minutes=round((time.time() - t0) / 60, 1),
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\n[pooled] Spearman(F_ii, room_i) = {rho:+.4f} over "
          f"{R.numel():,} weights")
    print(f"[per-matrix] mean {out['per_matrix_spearman']['mean']:+.4f}, "
          f"range {min(rhos):+.4f} to {max(rhos):+.4f}")
    print("[deciles] mean Fisher by room decile (smallest cells first):")
    for d in deciles:
        print(f"  {d['decile']:2d}  room<{d['room_hi']:.2e}  "
              f"mean F {d['mean_fisher']:.3e}  n={d['n']:,}")
    print(f"[done] {a.out} ({out['minutes']} min)")


if __name__ == "__main__":
    main()

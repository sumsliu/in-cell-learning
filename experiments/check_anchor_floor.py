#!/usr/bin/env python
"""Can this anchor hold a fill? Check its block scales against the floor.

Invariance is not unconditional. A fold writes the moved weight back into the
anchor buffer, so the margin has to outlast that write:

    margin * NARROWEST * absmax  >  ulp(anchor storage) / 2,

with NARROWEST the tightest NF4 cell in normalized units. In the storage
format's normal range the absmax cancels and fp16 clears the condition by
1.65x, which is why it holds everywhere the blocks are ordinary. It stops
cancelling in the subnormal range, where the spacing is fixed rather than
relative, and what is left is a floor on the block scale --- 3.7e-5 for fp16
at margin 0.01 (cellfill.bins.anchor_absmax_floor).

An anchor with blocks under that floor can still be written; the weights in
them are frozen and keep zero room. This reports how many, so the cost is
known before a sequence is launched rather than after it has run.

  python experiments/check_anchor_floor.py unsloth/Qwen3-8B-Base-bnb-4bit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--margin", type=float, default=0.01)
    p.add_argument("--dtype", default="float16",
                   help="the anchor buffer's dtype (exp_seq stores fp16)")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    import torch

    from cellfill.bins import anchor_absmax_floor
    from cellfill.bnb_state import frozen_state_from_linear4bit
    from experiments.exp0_clip_rate import build_4bit

    dt = getattr(torch, a.dtype)
    floor = anchor_absmax_floor(dt, a.margin)
    model, _ = build_4bit(a.model)

    n_below = n_blocks = n_frozen = n_weights = 0
    gmin = float("inf")
    worst = []
    for name, mod in model.named_modules():
        if type(mod).__name__ != "Linear4bit":
            continue
        fs = frozen_state_from_linear4bit(mod)
        am = fs["absmax"].float().abs()
        bsz, numel = fs["blocksize"], fs["anchors"].numel()
        below = int((am < floor).sum())
        n_below += below
        n_blocks += am.numel()
        n_frozen += below * bsz
        n_weights += numel
        gmin = min(gmin, float(am.min()))
        if below:
            worst.append((name, below, am.numel(), float(am.min())))

    print(f"[floor] {a.model}")
    print(f"[floor] anchor storage {dt}, margin {a.margin} -> "
          f"absmax floor {floor:.3e}")
    print(f"[floor] smallest block scale in the model: {gmin:.3e} "
          f"({'clears' if gmin >= floor else 'BELOW'} the floor)")
    print(f"[floor] blocks below the floor: {n_below} / {n_blocks} "
          f"({n_below / max(n_blocks, 1):.2e})")
    print(f"[floor] weights that would be frozen: {n_frozen} / {n_weights} "
          f"({n_frozen / max(n_weights, 1):.2e})")
    for name, b, tot, mn in sorted(worst, key=lambda r: -r[1])[:10]:
        print(f"   {name[-52:]:52s} {b:7d}/{tot:<9d} min {mn:.2e}")
    if not worst:
        print("[floor] every block clears the floor; nothing is frozen")

    if a.out:
        Path(a.out).write_text(json.dumps(dict(
            model=a.model, dtype=a.dtype, margin=a.margin, floor=floor,
            min_absmax=gmin, blocks_below=n_below, blocks=n_blocks,
            weights_frozen=n_frozen, weights=n_weights,
            layers=[dict(name=n, below=b, blocks=t, min_absmax=m)
                    for n, b, t, m in worst]), indent=2))


if __name__ == "__main__":
    main()

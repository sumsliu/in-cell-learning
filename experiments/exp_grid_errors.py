#!/usr/bin/env python
"""Do two different 4-bit grids make the same mistake, or different ones?

If the weight vectors that a language model can serve at a given capability
formed a low-dimensional manifold, then two quantizers of the same bf16
weights would be pulled toward that surface, and their error vectors would be
close to parallel. If instead the low-loss set has interior at this scale --
a full-dimensional region the weights sit inside rather than a sheet they sit
on -- the two errors are two independent excursions and should be close to
orthogonal.

"Close to orthogonal" needs a ruler. Over n coordinates the cosine of two
independent random vectors is about 1/sqrt(n), which at n = 1.4e9 is 2.7e-5.
A measured cosine of 0.01 therefore "looks like zero" and is four hundred
times the null: shared structure, and nowhere near parallel. The shuffled
control below measures that null empirically instead of assuming it.

The two grids must be taken against the SAME bf16 weights. Published W4A16
releases are quantizations of post-trained checkpoints while the NF4 releases
we use are quantizations of base checkpoints (clora/parent_check.py), so the
uniform grid here is computed from the base weights rather than downloaded.

  python experiments/exp_grid_errors.py \
      --original Qwen/Qwen3-1.7B-Base \
      --nf4 unsloth/Qwen3-1.7B-Base-bnb-4bit
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
    p.add_argument("--original", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--nf4", default="unsloth/Qwen3-1.7B-Base-bnb-4bit")
    p.add_argument("--group", type=int, default=64,
                   help="group size for the uniform grid; NF4's blocksize")
    p.add_argument("--out", default="out/grid_errors.json")
    return p.parse_args()


def cosine(a, b):
    import torch
    na, nb = a.norm().item(), b.norm().item()
    return float((a @ b).item() / (na * nb)) if na and nb else float("nan")


def main():
    a = parse()
    t0 = time.time()
    import torch

    from experiments.exp0_clip_rate import (
        build_4bit, collect_frozen_plain, load_original_weights,
    )

    print(f"[load] {a.nf4} for its published NF4 anchors", flush=True)
    m4, _ = build_4bit(a.nf4)
    _, nf4_anchors, _ = collect_frozen_plain(m4)
    del m4
    torch.cuda.empty_cache()

    print(f"[load] {a.original} bf16 originals", flush=True)
    orig = load_original_weights(a.original)

    rows, e_nf4, e_uni = {}, [], []
    for name, anch in nf4_anchors.items():
        w = orig.get(name)
        if w is None:
            continue
        w = w.float().reshape(-1).cpu()
        A = anch.float().reshape(-1).cpu()
        if w.numel() != A.numel():
            continue
        # A uniform symmetric int4 grid on the same weights, absmax per group,
        # round to nearest: the simplest genuinely different level table.
        g = a.group
        pad = (-w.numel()) % g
        wp = torch.cat([w, torch.zeros(pad)]) if pad else w
        blocks = wp.view(-1, g)
        s = blocks.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 7.0
        q = torch.round(blocks / s).clamp(-8, 7)
        B = (q * s).reshape(-1)[:w.numel()]

        ea, eb = A - w, B - w
        rows[name] = dict(
            cosine=cosine(ea, eb),
            nf4_rms=float(ea.pow(2).mean().sqrt()),
            uniform_rms=float(eb.pow(2).mean().sqrt()),
            n=int(w.numel()),
        )
        e_nf4.append(ea)
        e_uni.append(eb)

    assert e_nf4, "no matrix matched between the two checkpoints"
    EA = torch.cat(e_nf4)
    EB = torch.cat(e_uni)
    pooled = cosine(EA, EB)

    # The empirical null: destroy the coordinate correspondence and nothing
    # else. Anything the pooled cosine has above this is shared structure.
    gen = torch.Generator().manual_seed(0)
    perm = torch.randperm(EB.numel(), generator=gen)
    shuffled = cosine(EA, EB[perm])
    analytic_null = 1.0 / EA.numel() ** 0.5

    cs = [v["cosine"] for v in rows.values()]
    out = dict(
        original=a.original, nf4=a.nf4, group=a.group,
        n_matrices=len(rows), n_weights=int(EA.numel()),
        pooled_cosine=pooled,
        shuffled_cosine=shuffled,
        analytic_null=analytic_null,
        ratio_to_null=abs(pooled) / analytic_null if analytic_null else None,
        per_matrix=dict(mean=sum(cs) / len(cs), min=min(cs), max=max(cs)),
        rms=dict(nf4=float(EA.pow(2).mean().sqrt()),
                 uniform=float(EB.pow(2).mean().sqrt())),
        matrices=rows,
        minutes=(time.time() - t0) / 60,
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\n[pooled]   cos(NF4 error, uniform error) = {pooled:+.5f} "
          f"over {EA.numel():,} weights")
    print(f"[null]     shuffled control {shuffled:+.6f}, "
          f"analytic 1/sqrt(n) = {analytic_null:.2e}")
    print(f"[ratio]    |pooled| is {abs(pooled)/analytic_null:.0f}x the null")
    print(f"[per-mat]  mean {out['per_matrix']['mean']:+.4f}, "
          f"range {min(cs):+.4f} to {max(cs):+.4f} over {len(rows)} matrices")
    print(f"[rms]      NF4 {out['rms']['nf4']:.3e}, "
          f"uniform {out['rms']['uniform']:.3e}")
    print(f"[done] {a.out} ({out['minutes']:.1f} min)")


if __name__ == "__main__":
    main()

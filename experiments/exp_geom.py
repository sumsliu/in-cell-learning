#!/usr/bin/env python
"""Geometry of the safe region (M1 figure).

Three measurements on one model:
  1. interpolation: PPL along W(t) = anchors + t·(W_orig − anchors), t∈[0,1.5]
     — t≤1 is provably in-bin (RTN), t>1 exits; we also count escapes per t.
  2. in-bin perturbation: PPL for w = anchor + U(−1,1)·ρ·room, ρ-sweep,
     multiple seeds — the empirical safe-radius curve.
  3. diagonal-Fisher prediction (Prop 5): E[ΔNLL] ≈ (ρ²/6)·Σ F_i·room_i²,
     compared against the measured curve. Fisher from wikitext TRAIN.

Run: .venv/bin/python experiments/exp_geom.py --model Qwen/Qwen3-1.7B-Base
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

from cellfill import bin_bounds  # noqa: E402
from cellfill.nf4 import assign_codes  # noqa: E402
from experiments.exp0_clip_rate import (  # noqa: E402
    _load_model,
    build_4bit,
    collect_frozen_plain,
    eval_ppl,
    lambada_text,
    load_original_weights,
    maybe_ppl,
    wikitext_text,
)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--ts", default="0,0.25,0.5,0.75,1.0,1.25,1.5")
    p.add_argument("--rhos", default="0.125,0.25,0.5,0.75,1.0")
    p.add_argument("--pert-seeds", type=int, default=2)
    p.add_argument("--margin", type=float, default=0.01)
    p.add_argument("--fisher-chunks", type=int, default=48)
    p.add_argument("--max-ppl-chunks", type=int, default=30)
    p.add_argument("--out", default="out/exp_geom.json")
    return p.parse_args()


def overwrite(model, weight_map):
    for name, w in weight_map.items():
        model.get_parameter(f"{name}.weight").data.copy_(
            w.to(model.get_parameter(f"{name}.weight").device))


def main():
    args = get_args()
    assert torch.cuda.is_available()
    t0 = time.time()
    ts = [float(x) for x in args.ts.split(",")]
    rhos = [float(x) for x in args.rhos.split(",")]

    print(f"[load] {args.model} 4bit for frozen state")
    model4, tok = build_4bit(args.model)
    merged, anchors_map, frozen = collect_frozen_plain(model4)
    del model4, merged
    torch.cuda.empty_cache()

    orig = load_original_weights(args.model)
    ppl_txt = wikitext_text()
    x_txt = lambada_text()
    from datasets import load_dataset

    fisher_txt = "\n".join(load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"][:20000])

    print("[load] fp32 eval vehicle")
    model = _load_model(args.model, {"": 0}, torch.float32)

    result = dict(config=vars(args), interpolation=[], perturbation=[],
                  fisher=None)

    # --- 1. interpolation ---------------------------------------------------
    for t in ts:
        esc_total, n_total = 0, 0
        for name, anch in anchors_map.items():
            w = anch.float() + t * (orig[name].float() - anch.float())
            model.get_parameter(f"{name}.weight").data.copy_(
                w.to(model.get_parameter(f"{name}.weight").device))
            codes, absmax, bsz = frozen[name]
            new_codes = assign_codes(w, absmax, bsz)
            esc_total += int((new_codes != codes.view(-1)).sum())
            n_total += new_codes.numel()
        ppl_w = eval_ppl(model, tok, ppl_txt, max_chunks=args.max_ppl_chunks)
        ppl_l = maybe_ppl(model, tok, x_txt, args.max_ppl_chunks)
        row = dict(t=t, ppl_wiki=ppl_w, ppl_lambada=ppl_l,
                   escape_frac=esc_total / n_total)
        result["interpolation"].append(row)
        print(f"[interp] t={t:<5} ppl={ppl_w:.3f} lambada={ppl_l} "
              f"escapes={row['escape_frac']:.4%}", flush=True)

    # --- 2. in-bin perturbation --------------------------------------------
    room = {}
    for name, anch in anchors_map.items():
        codes, absmax, bsz = frozen[name]
        lo, hi = bin_bounds(codes, absmax, bsz, capped=True, margin=args.margin)
        a = anch.float().reshape(-1)
        room[name] = torch.minimum(a - lo, hi - a).view(anch.shape)

    for rho in rhos:
        for seed in range(args.pert_seeds):
            g = torch.Generator().manual_seed(1000 + seed)
            for name, anch in anchors_map.items():
                u = torch.rand(anch.shape, generator=g) * 2 - 1
                w = anch.float() + u * rho * room[name]
                p = model.get_parameter(f"{name}.weight")
                p.data.copy_(w.to(p.device))
            ppl_w = eval_ppl(model, tok, ppl_txt,
                             max_chunks=args.max_ppl_chunks)
            ppl_l = maybe_ppl(model, tok, x_txt, args.max_ppl_chunks)
            result["perturbation"].append(
                dict(rho=rho, seed=seed, ppl_wiki=ppl_w, ppl_lambada=ppl_l))
            print(f"[perturb] rho={rho:<6} seed={seed} ppl={ppl_w:.3f}",
                  flush=True)

    # --- 3. diagonal Fisher prediction (Prop 5) ----------------------------
    print("[fisher] accumulating diagonal Fisher on wikitext-TRAIN")
    overwrite(model, anchors_map)
    targets = []
    for name in anchors_map:
        p = model.get_parameter(f"{name}.weight")
        p.requires_grad_(True)
        targets.append((name, p))
    fisher_sum_room2 = 0.0
    ids = tok(fisher_txt, return_tensors="pt").input_ids[0]
    n_chunks = 0
    fsq = {name: torch.zeros_like(p, device=p.device)
           for name, p in targets}
    for i in range(0, min(ids.numel() - 1, 1024 * args.fisher_chunks), 1024):
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
            for name, p in targets:
                fsq[name] += p.grad.float() ** 2
                p.grad = None
        n_chunks += 1
    with torch.no_grad():
        for name, p in targets:
            F = fsq[name] / max(n_chunks, 1)
            fisher_sum_room2 += float(
                (F * room[name].to(F.device) ** 2).sum())
            p.requires_grad_(False)
    del fsq
    base_row = next(r for r in result["interpolation"] if r["t"] == 0)
    pred = {str(rho): (rho ** 2 / 6.0) * fisher_sum_room2 for rho in rhos}
    meas = {}
    for rho in rhos:
        rows = [r for r in result["perturbation"] if r["rho"] == rho]
        meas[str(rho)] = (sum(math.log(r["ppl_wiki"]) for r in rows) / len(rows)
                         - math.log(base_row["ppl_wiki"]))
    result["fisher"] = dict(sum_F_room2=fisher_sum_room2, n_chunks=n_chunks,
                            predicted_dnll=pred, measured_dnll=meas)
    print(f"[fisher] Σ F·room² = {fisher_sum_room2:.4f}")
    for rho in rhos:
        print(f"  rho={rho}: predicted ΔNLL={pred[str(rho)]:.4f}  "
              f"measured={meas[str(rho)]:.4f}")

    result["minutes"] = round((time.time() - t0) / 60, 1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    from experiments.exp0_clip_rate import stamp_of
    result["scorer"] = stamp_of(eval_ppl)
    out.write_text(json.dumps(result, indent=2))
    print(f"[done] {out} ({result['minutes']} min)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Static envelope: the closed-form error/room/floor account of a released
4-bit artifact, computed from its state dict alone. No GPU, no model load;
shards are streamed tensor by tensor.

Why this exists (user-ordered binding): every quantity the cell geometry
promises is a function of (codes, absmax) -- the per-weight error CEILING is
the half cell width HW[code] * absmax(block) (with the dequant-rounding
correction), the writable room at grid anchors is its margin-capped twin,
the storable floor population follows from the served scale distribution,
and "limited injection" has an exact number: the total writable budget.
These were measured piecemeal across runs; this binds them into one table
per model so the ten-burn curve lands against an analytic envelope.

The served absmax is the one the fill is computed against. Under double
quantization it is dequant8(absmax_u8; nested tables) + offset -- the
offset lives in the packed quant_state JSON, and omitting it produces
negative garbage (measured mistake, 2026-08-28). VALIDATION TARGET:
on Qwen3-30B-A3B-Base-bnb-4bit this script must reproduce the floor
gate's artifact exactly: min served absmax 3.725e-09, blocks below the
fp16/0.01 floor 7,789, weights frozen 498,496 (out/q_moe30b_floor.json).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cellfill.bins import (  # noqa: E402
    anchor_absmax_floor,
    normalized_cell_edges,
    unpack_nf4_codes,
)
from cellfill.nf4 import NF4_LEVELS  # noqa: E402


def find_snapshot(model_id: str) -> str:
    pats = [
        os.path.expanduser(
            f"~/.cache/huggingface/hub/models--{model_id.replace('/', '--')}"
            f"/snapshots/*"),
        model_id,  # allow a direct directory path
    ]
    for p in pats:
        hits = sorted(glob.glob(p))
        if hits and os.path.isdir(hits[-1]):
            return hits[-1]
    raise SystemExit(f"no snapshot for {model_id}")


def open_index(snap: str):
    idx = os.path.join(snap, "model.safetensors.index.json")
    if os.path.exists(idx):
        wm = json.load(open(idx))["weight_map"]
        shards = sorted(set(wm.values()))
        return wm, shards
    lone = os.path.join(snap, "model.safetensors")
    if os.path.exists(lone):
        return None, ["model.safetensors"]
    raise SystemExit(f"no safetensors in {snap}")


def served_absmax(h, base: str, cache: dict) -> torch.Tensor | None:
    """The scale the artifact actually serves, offset included."""
    keys = cache["keys"]
    if base + ".absmax" not in keys:
        return None
    am = h.get_tensor(base + ".absmax")
    if am.dtype != torch.uint8:
        return am.float()  # plain (non-nested) storage
    nabs = h.get_tensor(base + ".nested_absmax").float()
    nqm = h.get_tensor(base + ".nested_quant_map").float()
    qs_key = base + ".quant_state.bitsandbytes__nf4"
    blob = bytes(h.get_tensor(qs_key).tolist())
    meta = json.loads(blob.decode("utf-8"))
    off = None
    for k, v in meta.items():
        if "offset" in k:
            off = float(v)
    if off is None:
        raise SystemExit(f"{base}: no offset key in quant_state {list(meta)}")
    nb = int(meta.get("nested_blocksize", 256))
    n = am.numel()
    deq = nqm[am.long()].reshape(-1)
    nb_rep = nabs.reshape(-1).repeat_interleave(nb)[:n]
    return deq * nb_rep + off, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--margin", type=float, default=0.01)
    ap.add_argument("--layout", default="interleave_high")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    snap = find_snapshot(args.model)
    wm, shards = open_index(snap)
    floor = anchor_absmax_floor(torch.float16, args.margin)

    L = NF4_LEVELS.float()
    lo_t, hi_t = normalized_cell_edges(False)     # true cell edges
    lo_c, hi_c = normalized_cell_edges(True)      # capped outer edges
    HW_true = torch.minimum(L - lo_t, hi_t - L)   # per-code half width, true
    # writable room mirrors bin_bounds: each capped cell shrinks by
    # margin*width per side, then the room is the distance from the anchor
    # (the level) to the nearer pulled-in wall
    w_c = hi_c - lo_c
    ROOM_c = torch.minimum(L - (lo_c + args.margin * w_c),
                           (hi_c - args.margin * w_c) - L).clamp_min(0)

    stats = dict(model=args.model, snapshot=os.path.basename(snap),
                 margin=args.margin, floor=floor, layers=0,
                 blocks=0, weights=0,
                 absmax_min=float("inf"), absmax_neg_blocks=0,
                 blocks_below_floor=0, weights_frozen=0,
                 hw_sum=0.0, room_sum=0.0, room_writable_sum=0.0,
                 code_hist=[0] * 16)
    absmax_qs = []   # streamed reservoir of block scales for quantiles
    meta_seen = None

    for shard in shards:
        with safe_open(os.path.join(snap, shard), framework="pt") as h:
            keys = set(h.keys())
            cache = {"keys": keys}
            for k in sorted(keys):
                if not k.endswith(".weight") or (k + ".absmax") not in keys:
                    continue
                got = served_absmax(h, k, cache)
                if got is None:
                    continue
                am, meta_seen = got if isinstance(got, tuple) else (got, meta_seen)
                am = am.reshape(-1)
                packed = h.get_tensor(k)
                bsz = int((meta_seen or {}).get("blocksize", 64))
                numel = am.numel() * bsz
                codes = unpack_nf4_codes(packed, numel, args.layout).long()
                stats["layers"] += 1
                stats["blocks"] += am.numel()
                stats["weights"] += numel
                stats["absmax_min"] = min(stats["absmax_min"],
                                          float(am.min()))
                stats["absmax_neg_blocks"] += int((am < 0).sum())
                below = am.abs() < floor
                stats["blocks_below_floor"] += int(below.sum())
                stats["weights_frozen"] += int(below.sum()) * bsz
                hist = torch.bincount(codes, minlength=16)
                for c in range(16):
                    stats["code_hist"][c] += int(hist[c])
                am_w = am.abs().repeat_interleave(bsz)
                hw_w = HW_true[codes] * am_w
                room_w = ROOM_c[codes] * am_w
                stats["hw_sum"] += float(hw_w.sum())
                stats["room_sum"] += float(room_w.sum())
                writable = (~below).repeat_interleave(bsz)
                stats["room_writable_sum"] += float(room_w[writable].sum())
                if am.numel() > 4096:
                    absmax_qs.append(am[torch.randperm(am.numel())[:4096]])
                else:
                    absmax_qs.append(am.clone())
                del codes, am_w, hw_w, room_w

    q = torch.cat(absmax_qs).float()
    if q.numel() > 1_000_000:  # torch.quantile caps at 16M inputs
        q = q[torch.randperm(q.numel())[:1_000_000]]
    for name, val in [("absmax_p01", 0.001), ("absmax_p1", 0.01),
                      ("absmax_median", 0.5)]:
        stats[name] = float(torch.quantile(q, val))
    stats["hw_mean"] = stats["hw_sum"] / max(stats["weights"], 1)
    stats["room_mean"] = stats["room_sum"] / max(stats["weights"], 1)
    stats["room_over_hw"] = stats["room_sum"] / max(stats["hw_sum"], 1e-30)
    stats["frozen_frac"] = stats["weights_frozen"] / max(stats["weights"], 1)

    out = args.out or f"out/envelope_{args.model.split('/')[-1]}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(stats, open(out, "w"), indent=2)
    print(f"[envelope] {args.model}: layers {stats['layers']} "
          f"weights {stats['weights']:.3e} | absmax min {stats['absmax_min']:.3e} "
          f"neg-blocks {stats['absmax_neg_blocks']} | below-floor blocks "
          f"{stats['blocks_below_floor']} frozen {stats['weights_frozen']} | "
          f"hw_mean {stats['hw_mean']:.3e} room_mean {stats['room_mean']:.3e} "
          f"room/hw {stats['room_over_hw']:.3f}")
    print(f"[envelope] wrote {out}")


if __name__ == "__main__":
    main()

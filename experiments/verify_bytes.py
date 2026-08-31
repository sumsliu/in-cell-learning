#!/usr/bin/env python
"""End-to-end byte-level verification, through bitsandbytes' own code path.

Every invariance check in this repository is our own integer arithmetic
(cellfill.nf4.assign_codes) under the frozen scales. That is the right
definition, but it leaves a limitation the paper states plainly: the check
never runs the vendor's quantizer. This closes it, and separates three
claims that are easy to conflate.

  A. the released file is untouched
     sha256 of every safetensors shard before and after the whole procedure.
     Nothing in this method writes to the release; this asserts it.

  B. our level assignment IS bitsandbytes'
     bitsandbytes' quantize_4bit is scale-invariant per block: it divides by
     the block maximum and looks up the nearest NF4 level, so it cannot be
     told to use a scale from outside. What it CAN be asked is whether, on
     the blocks where it recomputes the same scale the release ships, it
     assigns the same codes. Those blocks isolate the level assignment from
     the scale, and agreement there is the statement that our integer
     arithmetic is the vendor's.

  C. what a deployment that re-derives the scales would get
     The same call over every block, scale included. This is NOT what the
     method claims -- invariance is defined under frozen scales, which the
     release ships alongside the codes -- and the gap between B and C is the
     operational precondition, measured rather than asserted: a served
     weight may move outward inside the outermost cell and lift its block's
     maximum, and every code in that block is then re-derived.

  The zero-fill arm is the control. The served weights are replaced by the
  anchors themselves, so any disagreement there is the round trip, not the
  update.

  python experiments/verify_bytes.py --model unsloth/Qwen3-1.7B-Base-bnb-4bit \
      --fill out/fill_1p7b.pt --out out/verify_bytes.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cellfill.bins import nibble_layout_of, unpack_nf4_codes  # noqa: E402
from cellfill.bnb_state import frozen_state_from_linear4bit  # noqa: E402
from cellfill.nf4 import assign_codes  # noqa: E402
from experiments.exp0_clip_rate import build_4bit  # noqa: E402
from experiments.exp5_qil import BoundedFill, wrap_model  # noqa: E402


def shard_hashes(model_id: str) -> dict:
    """sha256 of every weight shard in the local snapshot of the release."""
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(model_id, allow_patterns=["*.safetensors"]))
    out = {}
    for f in sorted(root.rglob("*.safetensors")):
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        out[f.name] = h.hexdigest()
    return out


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="a published NF4 release")
    p.add_argument("--fill", default=None,
                   help="factors from --save-fill; without one the check runs "
                        "on a random fill at full amplitude, which is the "
                        "harder case")
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--tanh-scale", type=float, default=40.0)
    p.add_argument("--margin", type=float, default=0.01)
    p.add_argument("--layers", type=int, default=0,
                   help="check only the first N wrapped layers (0 = all)")
    p.add_argument("--out", default="out/verify_bytes.json")
    return p.parse_args()


def main():
    args = parse()
    t0 = time.time()
    from bitsandbytes.functional import quantize_4bit

    before = shard_hashes(args.model)
    print(f"[A] hashed {len(before)} shard(s) of {args.model}", flush=True)

    model, tok = build_4bit(args.model)
    meta = dict(rank=args.rank, tanh_scale=args.tanh_scale,
                margin=args.margin, target_filter=None)
    if args.fill:
        ck = torch.load(args.fill, map_location="cpu", weights_only=False)
        meta.update({k: ck["meta"][k] for k in
                     ("rank", "tanh_scale", "margin", "target_filter")
                     if k in ck["meta"]})
    packed_before = {}
    for name, mod in model.named_modules():
        if type(mod).__name__ == "Linear4bit":
            packed_before[name] = mod.weight.data.detach().clone()

    wrap_model(model, meta["rank"], meta["tanh_scale"], meta["margin"],
               name_filter=meta.get("target_filter"))
    fills = {n: m for n, m in model.named_modules()
             if isinstance(m, BoundedFill)}
    if args.fill:
        with torch.no_grad():
            for n, (A, B) in ck["fills"].items():
                fills[n].A.copy_(A.to(fills[n].A.device))
                fills[n].B.copy_(B.to(fills[n].B.device))
        source = f"fill file {Path(args.fill).name}"
    else:
        g = torch.Generator(device="cpu").manual_seed(0)
        with torch.no_grad():
            for m in fills.values():
                m.B.copy_(torch.randn(m.B.shape, generator=g).to(m.B.device))
        source = "random fill at full amplitude"
    print(f"[B] {len(fills)} wrapped layers, fill from {source}", flush=True)

    names = sorted(fills)
    if args.layers:
        names = names[:args.layers]
    arms = {}
    for arm in ("zero-fill", "served"):
        blocks_same_scale = blocks_tot = 0
        codes_ok_given_scale = codes_ok_all = 0
        layers_byte_identical = n_layers = n_bytes = 0
        worst = []
        with torch.no_grad():
            for name in names:
                mod = fills[name]
                base = mod.base
                fs = frozen_state_from_linear4bit(base)
                absmax, bsz = fs["absmax"], fs["blocksize"]
                anchors = fs["anchors"].float()
                w = anchors if arm == "zero-fill" else (
                    anchors + mod.fill(torch.float32).reshape(anchors.shape))
                ref = base.weight.data
                wf = w.to(torch.float16).contiguous()
                # the vendor's quantizer, deriving its own scale per block
                q, st = quantize_4bit(wf, blocksize=bsz, quant_type="nf4",
                                      compress_statistics=False)
                new_absmax = st.absmax.float().to(absmax.device)
                same_scale = torch.isclose(new_absmax, absmax.float(),
                                           rtol=1e-3, atol=0)
                # ours: our integer arithmetic under the frozen scales.
                # theirs: the codes bitsandbytes actually emitted, recovered
                # from its packed output. Comparing our assign_codes against
                # itself with a different scale would test nothing.
                ours = assign_codes(w, absmax, bsz).reshape(-1)
                # The nibble layout is read off the RELEASE, against codes
                # frozen_state_from_linear4bit has already self-checked.
                # Calibrating it against bitsandbytes' new output instead
                # would let a disagreement be absorbed as a layout guess.
                layout = nibble_layout_of(ref, fs["codes"].reshape(-1))
                theirs = unpack_nf4_codes(q, ours.numel(),
                                          layout).reshape(-1).to(ours.device)
                blk = torch.arange(ours.numel(),
                                   device=ours.device) // bsz
                agree = (ours == theirs)
                mask = same_scale.to(agree.device)[blk]
                blocks_same_scale += int(same_scale.sum())
                blocks_tot += int(same_scale.numel())
                codes_ok_given_scale += int((agree & mask).sum())
                codes_ok_all += int(agree.sum())
                identical = bool(torch.equal(q.reshape(-1), ref.reshape(-1)))
                layers_byte_identical += identical
                n_layers += 1
                n_bytes += int(ref.numel())
                if not identical and len(worst) < 5:
                    worst.append(dict(
                        layer=name,
                        blocks_rescaled=int((~same_scale).sum()),
                        of_blocks=int(same_scale.numel()),
                        codes_changed=int((~agree).sum())))
                if n_layers % 40 == 0:
                    print(f"    [{arm}] {n_layers}/{len(names)} layers, "
                          f"{blocks_same_scale}/{blocks_tot} blocks keep "
                          f"their scale", flush=True)
        n_codes = blocks_tot * bsz
        arms[arm] = dict(
            layers=n_layers, packed_bytes=n_bytes,
            layers_byte_identical=layers_byte_identical,
            blocks=blocks_tot, blocks_scale_unchanged=blocks_same_scale,
            blocks_scale_unchanged_frac=blocks_same_scale / max(blocks_tot, 1),
            codes_agree_on_unchanged_scale_frac=(
                codes_ok_given_scale / max(blocks_same_scale * bsz, 1)),
            codes_agree_overall_frac=codes_ok_all / max(n_codes, 1),
            examples=worst)
        r = arms[arm]
        print(f"[{arm}] {layers_byte_identical}/{n_layers} layers "
              f"byte-identical | blocks keeping their scale "
              f"{r['blocks_scale_unchanged_frac']:.4%} | codes agreeing "
              f"there {r['codes_agree_on_unchanged_scale_frac']:.6%}",
              flush=True)

    del model
    torch.cuda.empty_cache()
    after = shard_hashes(args.model)
    untouched = before == after
    res = dict(
        model=args.model, fill=args.fill, fill_source=source,
        shards=len(before), release_untouched=untouched,
        shard_sha256=before, arms=arms,
        note=("bitsandbytes' quantize_4bit is scale-invariant per block and "
              "cannot be given an outside scale, so the vendor check is "
              "decomposed: on blocks where it recomputes the scale the "
              "release ships, do its codes match ours (the level assignment "
              "is the vendor's), and over all blocks, does the packed tensor "
              "come back identical (what a naive re-quantization would get, "
              "which the method does not claim). The zero-fill arm is the "
              "round-trip control."),
        minutes=round((time.time() - t0) / 60, 1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"[A] release untouched: {untouched}")
    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()

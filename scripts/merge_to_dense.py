#!/usr/bin/env python
"""Materialize a saved fill into a dense HF checkpoint a server can load.

The serving stacks cannot apply a BoundedFill at runtime; this loads the
4-bit release, re-attaches the fill, computes the served weights
W = anchors + M*tanh(s*BA) layer by layer, writes them into a dense
bf16 copy of the model, and saves it in HF format. The released file is
read, never written; the output is a second, explicitly-derived artifact
whose provenance is (release, fill).

  python scripts/merge_to_dense.py --model /home/zssy/models/Qwen3.8-27B \
      --fill out/fill_27b.pt --out /home/zssy/models/Qwen3.8-27B-filled
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--fill", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    t0 = time.time()
    from experiments.served import load_served
    from experiments.exp5_qil import BoundedFill

    model, tok, _ = load_served(args.model, fill=args.fill)
    n = 0
    with torch.no_grad():
        for name, module in list(model.named_modules()):
            for child_name, child in list(module.named_children()):
                if isinstance(child, BoundedFill):
                    import torch.nn as nn
                    d = child.fill(torch.float32)
                    if child.anchors is None:
                        # The anchor stayed implicit inside the 4-bit base
                        # (the --inplace-only path): the served weight is the
                        # dequantized release plus the in-cell displacement,
                        # which is what _apply_fill computes at run time.
                        from bitsandbytes.functional import dequantize_4bit
                        bw = child.base.weight
                        anch = dequantize_4bit(bw.data, bw.quant_state).float()
                        w = anch + d.reshape(anch.shape)
                    else:
                        w = child.anchors.float() + d
                    dense = nn.Linear(w.shape[1], w.shape[0], bias=False,
                                      dtype=torch.bfloat16, device="cpu")
                    dense.weight.data.copy_(w.to(torch.bfloat16).cpu())
                    setattr(module, child_name, dense)
                    n += 1
                    del w, d
                    torch.cuda.empty_cache()
    print(f"[merge] {n} layers materialized in {(time.time()-t0)/60:.1f} min")
    # The fill only wraps the matrices it was trained on. Anything else the
    # release quantized (lm_head on some checkpoints, any linear the target
    # filter excluded) is still a Linear4bit, and a checkpoint that mixes
    # dense and 4-bit layers is not something a server can load. Dequantize
    # the remainder so the saved artifact is uniformly dense.
    from experiments.exp0_clip_rate import dequantize_in_place

    rest = dequantize_in_place(model)
    if rest:
        print(f"[merge] dequantized {rest} further 4-bit layers", flush=True)
    # With no 4-bit layer left, the bitsandbytes stanza in the config is a lie
    # and transformers refuses .to(dtype) while it is present. Drop it: what is
    # being saved is the served model, not a quantized release.
    for cfg in (model.config, getattr(model.config, "text_config", None)):
        if cfg is not None and getattr(cfg, "quantization_config", None) is not None:
            del cfg.quantization_config
    # transformers guards the cast on `model.quantization_method`, not on the
    # config stanza and not on is_quantized -- clearing those two was why the
    # first fix did not take. Clear the attribute the guard actually reads.
    for attr in ("quantization_method", "hf_quantizer"):
        if getattr(model, attr, None) is not None:
            setattr(model, attr, None)
    if getattr(model, "is_quantized", False):
        model.is_quantized = False
    # And do not ask for a dtype at all: the guard only fires when one is
    # present in the arguments, and after materialization plus dequantization
    # every linear is already bf16, so the cast was never needed.
    model = model.to("cpu")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True, max_shard_size="4GB")
    tok.save_pretrained(args.out)
    print(f"[done] {args.out}  ({(time.time()-t0)/60:.1f} min total)")


if __name__ == "__main__":
    main()

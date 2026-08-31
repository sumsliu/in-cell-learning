#!/usr/bin/env python
"""Split a model across cards by what it will weigh AFTER it is wrapped.

accelerate balances the bytes it can see. For a 4-bit checkpoint those are the
packed weights, and a sequence then hangs an fp16 anchor off every quantized
layer -- four times the packed bytes for that layer, and nothing at all for an
embedding or a norm. A load-time balance is therefore lopsided by the time
training starts: measured on Qwen3.8-27B, device_map="balanced" put 5.1 GB on
one card and 11.3 GB on the other, and after wrapping that was 16.5 GB against
48.5 GB -- one card full while the other still had 57 GB free.

This plans the split against the finished footprint instead. Each candidate
group (a decoder block, the vision tower, the embeddings) is charged

    packed + 4 * packed + other,

the middle term being the anchors, and groups are assigned largest-first to
whichever device is currently lightest. The model is built on the meta device,
so planning costs no memory and no GPU.

  python experiments/plan_device_map.py /home/zssy/models/Qwen3.8-27B \
      --devices 0,1 --out out/map_27b.json
  CELLFILL_DEVICE_MAP=out/map_27b.json python experiments/exp_seq.py ...
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--devices", default="0,1",
                   help="comma-separated CUDA indices to spread over")
    p.add_argument("--anchor-bytes", type=int, default=2,
                   help="bytes per anchor; exp_seq stores fp16")
    p.add_argument("--packed-bytes", type=float, default=0.5,
                   help="bytes per constrained weight in the 4-bit file")
    p.add_argument("--other-bytes", type=int, default=2,
                   help="bytes per unquantized parameter")
    p.add_argument("--out", default=None)
    return p.parse_args()


def block_names(model):
    """Modules that must not be split: the direct children of any ModuleList.

    These are the decoder blocks, and in a multimodal release the vision
    tower's blocks as well. Everything else -- embeddings, norms, projectors,
    the head -- is charged and placed at the granularity of the module that
    holds the parameter, which is what keeps every parameter covered. A map
    that misses one is rejected outright by accelerate's check_device_map,
    and the parameters it misses are exactly the top-level ones a
    prefix-collapsing grouping drops.
    """
    out = set()
    for name, mod in model.named_modules():
        if type(mod).__name__ != "ModuleList":
            continue
        for child, _ in mod.named_children():
            out.add(f"{name}.{child}" if name else child)
    return out


def main():
    a = parse()
    devices = [int(x) for x in a.devices.split(",") if x.strip() != ""]

    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(a.model)
    quant = getattr(cfg, "quantization_config", None)
    # A dense checkpoint is not exempt: build_4bit quantizes one on load, so
    # its Linears carry anchors exactly as a published 4-bit release's do.
    # This is the 27B case -- /home/zssy/models/Qwen3.8-27B ships dense and is
    # quantized by us -- and charging it as unquantized plans a split for a
    # model that never exists.
    skip = {"lm_head"}
    if quant is not None:
        get = (quant.get if isinstance(quant, dict)
               else lambda k, d=None: getattr(quant, k, d))
        skip |= set(get("llm_int8_skip_modules", None) or [])
        print(f"[plan] published {get('quant_method', '?')} release; "
              f"skipping {sorted(skip)}", flush=True)
    else:
        print("[plan] dense checkpoint; charging every Linear but "
              f"{sorted(skip)} as self-quantized on load", flush=True)

    import transformers

    # The class actually loaded decides the parameter names, and for this 27B
    # they differ: AutoModelForCausalLM fails on it and _load_model falls back
    # to the image-text class, whose decoder lives under model.language_model.
    # Planning against the wrong class produces a map whose keys match nothing.
    arch = (getattr(cfg, "architectures", None) or [None])[0]
    with init_empty_weights():
        cls = getattr(transformers, arch, None) if arch else None
        if cls is not None:
            model = cls(cfg)
        else:
            try:
                model = transformers.AutoModelForCausalLM.from_config(cfg)
            except (ValueError, TypeError, KeyError):
                model = transformers.AutoModelForImageTextToText.from_config(cfg)
    print(f"[plan] built {type(model).__name__} on meta", flush=True)

    def quantized(name, mod):
        if type(mod).__name__ != "Linear":
            return False
        leaf = name.split(".")[-1]
        return leaf not in skip and "lm_head" not in name

    blocks = sorted(block_names(model), key=len, reverse=True)

    def owner(pname):
        for b in blocks:
            if pname == b or pname.startswith(b + "."):
                return b
        return pname.rsplit(".", 1)[0] if "." in pname else pname

    packed = defaultdict(int)
    other = defaultdict(int)
    qweights = set()
    for name, mod in model.named_modules():
        if quantized(name, mod):
            qweights.add(name + ".weight")
    for name, p_ in model.named_parameters():
        (packed if name in qweights else other)[owner(name)] += p_.numel()
    for name, b_ in model.named_buffers():
        other[owner(name)] += b_.numel()

    per_group = {}
    for g in set(packed) | set(other):
        cost = (packed[g] * a.packed_bytes
                + packed[g] * a.anchor_bytes
                + other[g] * a.other_bytes)
        if cost:
            per_group[g] = cost

    # Tied embeddings share storage with the head; accelerate can place them
    # apart but the copy it then keeps is the vocab matrix, so co-locate.
    tied = []
    if getattr(cfg, "tie_word_embeddings", False) or getattr(
            getattr(cfg, "text_config", None), "tie_word_embeddings", False):
        tied = [g for g in per_group
                if g.endswith("embed_tokens") or g.endswith("lm_head")]

    # The entry point has to be on the device the trainer sends inputs to.
    # exp0_clip_rate.train does .to(model.device), which transformers resolves
    # to the first parameter's device; if the input embedding is on a
    # different card the index tensor reaches torch.embedding from the wrong
    # one and the lookup dies with a device-side assert about the index rather
    # than about the device -- which is exactly how it presents on the 27B,
    # whose first parameter is in the vision tower. Pinning both to the first
    # device makes model.device and the embedding agree.
    first_param = next((n for n, _ in model.named_parameters()), None)
    forced = set()
    ie = model.get_input_embeddings()
    ie_name = next((n for n, m_ in model.named_modules() if m_ is ie), None)
    for cand in (first_param, ie_name):
        if cand is None:
            continue
        g = owner(cand) if cand is first_param else cand
        if g in per_group:
            forced.add(g)
    forced |= {g for g in tied if g in per_group}

    # largest first onto the lightest device: the groups are near-identical
    # decoder blocks, so this lands within one block of even
    # Assign in EXECUTION ORDER and cut, rather than packing the largest group
    # onto the lightest card. A greedy pack balances just as well and is
    # wrong: it interleaves consecutive blocks across cards, so activations
    # cross the bus at every layer instead of once. accelerate's own maps are
    # always contiguous prefixes, and its device-alignment hook is written for
    # that -- an interleaved map of the same 68 keys, balanced to 0.05%, made
    # a plain sharded forward fail with a device-side assert whose index
    # tensor arrived on the wrong card holding float bit patterns read as
    # int64, which is what a cross-device copy looks like when it is read
    # before it lands. The same model under device_map="balanced" runs.
    order = {n: i for i, (n, _) in enumerate(model.named_modules())}
    ordered = sorted(per_group, key=lambda g: order.get(g, 1 << 30))
    total = sum(per_group.values())
    target = total / len(devices)
    load = {d: 0.0 for d in devices}
    plan, di, run = {}, 0, 0.0
    for g in ordered:
        c = per_group[g]
        # advance once this device is closer to its share with the group than
        # without it, and never before the forced entry-point groups are placed
        if (di < len(devices) - 1 and run > 0
                and not (forced - set(plan))
                and abs(run + c - target) > abs(run - target)):
            di += 1
            run = 0.0
        plan[g] = devices[di]
        load[devices[di]] += c
        run += c
    if forced:
        misplaced = sorted(g for g in forced if plan.get(g) != devices[0])
        print(f"[plan] entry point / tied: {sorted(forced)}"
              + (f"  WARNING not on cuda:{devices[0]}: {misplaced}"
                 if misplaced else f"  on cuda:{devices[0]}"), flush=True)
    cuts = [g for i, g in enumerate(ordered)
            if i and plan[g] != plan[ordered[i - 1]]]
    print(f"[plan] contiguous, cut before: {cuts}", flush=True)

    total_c = sum(per_group.values())
    total_p = sum(packed.values())
    print(f"[plan] {a.model}")
    print(f"[plan] {len(per_group)} placeable groups, "
          f"{total_p:.4e} constrained weights")
    print(f"[plan] charged packed {total_p * a.packed_bytes / 2**30:.1f} GB + "
          f"anchors {total_p * a.anchor_bytes / 2**30:.1f} GB + other "
          f"{sum(other.values()) * a.other_bytes / 2**30:.1f} GB = "
          f"{total_c / 2**30:.1f} GB")
    for d in devices:
        n = sum(1 for v in plan.values() if v == d)
        print(f"[plan]   cuda:{d}  {load[d] / 2**30:6.1f} GB after wrapping "
              f"({n} groups)")
    spread = (max(load.values()) - min(load.values())) / max(total_c, 1)
    print(f"[plan] imbalance {spread:.2%} of the total")

    if a.out:
        Path(a.out).write_text(json.dumps(plan, indent=2, sort_keys=True))
        print(f"[plan] wrote {a.out}")
    else:
        print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

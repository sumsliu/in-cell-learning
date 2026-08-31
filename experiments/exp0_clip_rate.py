#!/usr/bin/env python
"""Exp0: first contact — QLoRA on synthetic facts, then clip-merge into frozen bins.

Produces the project's first three numbers:
  1. clip rate — how much of a real LoRA update the frozen bins reject
  2. invariance — the merged model re-quantizes bit-identically (must be 100%)
  3. quality delta — fact recall / PPL for adapter vs clip-merged weights

Run on the GPU server from the repo root:
  .venv/bin/python experiments/exp0_clip_rate.py --n-facts 1000
"""

from __future__ import annotations

import inspect

import argparse
import os
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cellfill import bin_bounds, check_invariance, clip_merge, dequantize_ref  # noqa: E402
from cellfill.bnb_state import frozen_state_from_linear4bit  # noqa: E402
from experiments.synth_facts import (  # noqa: E402
    bits_per_fact,
    generate,
    probe_pairs,
    training_texts,
)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--n-facts", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--margin", type=float, default=0.01)
    p.add_argument("--merge-base", choices=["anchor", "original"],
                   default="anchor",
                   help="merge LoRA delta onto dequantized anchors, or onto the "
                        "original fp32 weights (RTN guarantees W is in-bin)")
    p.add_argument("--replay-frac", type=float, default=0.0,
                   help="fraction of training batch drawn from wikitext-2 TRAIN "
                        "snippets (anti-forgetting rehearsal; eval uses TEST)")
    p.add_argument("--heal-epochs", type=int, default=0,
                   help="A+ path: projected dense fine-tune epochs starting "
                        "from the clip-merged weights (clamp after every step)")
    p.add_argument("--heal-lr", type=float, default=1e-5)
    p.add_argument("--dense-only", action="store_true",
                   help="B path: skip LoRA entirely; projected dense training "
                        "from the anchors for --epochs (constraint-aware from "
                        "step 0)")
    p.add_argument("--no-clamp", action="store_true",
                   help="disable the box projection (unconstrained baseline; "
                        "invariance not guaranteed, violations are reported)")
    p.add_argument("--probe-cap", type=int, default=30000,
                   help="evaluate recall on at most this many probe pairs "
                        "(seeded subsample; 100k-fact runs need this)")
    p.add_argument("--heal-optim", choices=["adamw", "adam8bit"],
                   default="adamw",
                   help="adam8bit (bnb paged) fits dense heal for >=4B models")
    p.add_argument("--skip-anchor-eval", action="store_true",
                   help="do not keep/evaluate the anchors map (saves ~4 bytes "
                        "per weight of host RAM; required for 27B)")
    p.add_argument("--save-merged", action="store_true",
                   help="save the final (healed if heal ran, else merged) "
                        "weights as fp16 .pt next to --out")
    p.add_argument("--target-regex", default=None,
                   help="peft target_modules regex (default: all-linear); use "
                        "for multimodal models to restrict LoRA to the text "
                        "backbone")
    p.add_argument("--name-filter", default="",
                   help="only merge/freeze Linear4bit modules whose name "
                        "contains this substring (e.g. language_model)")
    p.add_argument("--eval-dtype", choices=["float32", "bfloat16"],
                   default="float32",
                   help="dtype for the full-model eval rebuild (27B needs "
                        "bfloat16)")
    p.add_argument("--eval-device-map", choices=["cuda0", "auto"],
                   default="cuda0",
                   help="'auto' shards the eval model across all GPUs")
    p.add_argument("--map-dtype", choices=["float32", "float16"],
                   default="float32",
                   help="CPU storage dtype for anchors/merged maps (27B needs "
                        "float16 to fit RAM; invariance is checked in fp32 "
                        "before the cast)")
    p.add_argument("--radius", type=float, default=1.0,
                   help="fraction of each bin's room (around the anchor) made "
                        "available to updates; rho<1 shrinks the trust region "
                        "(invariance is still checked vs the full cell)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-ppl-chunks", type=int, default=40)
    p.add_argument("--facts-file", default=None,
                   help="train on a real corpus (JSON list of {text,...}) "
                        "instead of the synthetic generator; probes are taken "
                        "from the matching *_probes.json")
    p.add_argument("--probes-file", default=None,
                   help="override the probe file inferred from --facts-file")
    p.add_argument("--ckpt", default=None,
                   help="write trainable params + optimizer state after every "
                        "epoch and resume from them if the file exists")
    p.add_argument("--out", default="out/exp0.json")
    return p.parse_args()


# The compute dtype of every 4-bit load. bf16 everywhere the paper ran
# (Ampere and later). A Turing card has no bf16 matmul in silicon, and the
# env-only escape hatch proved insufficient in practice: three sibling
# scripts hardcoded bf16 past it and every TITAN job in the queue's history
# died at its first bf16 matmul. The fallback is now automatic by device
# capability (sm < 80 -> fp16), with the env override kept. fp16's ten
# mantissa bits sit INSIDE the bf16 (7-bit) dequant-error envelope the
# halfwidth machinery already reserves (normalized_halfwidth's
# dequant_bits), so the fallback is conservative, and it is a no-op on
# every sm >= 80 host. TITAN-produced numbers remain scoped to
# self-paired deltas, never cross-host comparisons.
def _default_compute_dtype():
    env = os.environ.get("CELLFILL_DTYPE")
    if env == "float16":
        return torch.float16
    if env == "float32":
        return torch.float32
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
        return torch.float16
    return torch.bfloat16


COMPUTE_DTYPE = _default_compute_dtype()

# Where a 4-bit load is placed. Everything on one card by default, which is
# what every archived run used. Self-quantizing a dense release needs transient
# headroom well above the finished 4-bit model, so a small card has to be told
# to spill that transient: CELLFILL_DEVICE_MAP=auto lets accelerate place what
# does not fit on CPU during the conversion. Set it only on such a host; it
# changes placement, never arithmetic.
_DM = os.environ.get("CELLFILL_DEVICE_MAP")
if _DM and _DM.lstrip().startswith("{"):
    # An explicit {module_name: device} map, as JSON or as a path to a JSON
    # file. accelerate's own placement balances the bytes it can see, which
    # for a 4-bit model is the packed weights; an in-cell sequence then hangs
    # fp16 anchors off every quantized layer -- four times the packed bytes,
    # and nothing at all off the embeddings -- so a split that is balanced at
    # load time is lopsided by the time training starts. Passing the map
    # outright is how a sequence gets an even split at 27B and above; see
    # experiments/plan_device_map.py, which computes one.
    import json as _json

    DEVICE_MAP = _json.loads(_DM)
elif _DM and os.path.exists(_DM):
    import json as _json

    DEVICE_MAP = _json.loads(open(_DM).read())
else:
    DEVICE_MAP = _DM or {"": 0}

# What accelerate is allowed to put on each card, e.g. "0:10GiB,1:10GiB".
# Needed only when the finished 4-bit model is not what fills the card: an
# in-cell sequence hangs explicit fp16 anchors off every wrapped layer
# afterwards -- 48.7 GB at 27B -- and "auto" places by the weights it can see,
# so it stacks a 16.5 GB model onto one card and the anchors then have nowhere
# to go. Capping each card below the model's own size forces the split that
# leaves room for them. Placement only; the arithmetic is unchanged.
_MM = os.environ.get("CELLFILL_MAX_MEMORY")
MAX_MEMORY = None
if _MM:
    MAX_MEMORY = {}
    for part in _MM.split(","):
        k, v = part.split(":", 1)
        k = k.strip()
        MAX_MEMORY[k if k in ("cpu", "disk") else int(k)] = v.strip()


def _load_model(model_id: str, device_map, dtype, quantization_config=None):
    """Load with CausalLM, falling back to ImageTextToText for multimodal
    checkpoints (e.g. Qwen3.8-27B); text-only usage is identical either way."""
    import transformers

    # transformers 5 takes dtype=; 4.x takes torch_dtype= and silently
    # ignores dtype=, loading in fp32 -- which for a 30B MoE is the whole
    # card. The MoE environment pins 4.51, so both spellings are needed.
    major = int(transformers.__version__.split(".")[0])
    kwargs = dict(device_map=device_map)
    if MAX_MEMORY is not None:
        kwargs["max_memory"] = MAX_MEMORY
    kwargs["dtype" if major >= 5 else "torch_dtype"] = dtype
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    elif dtype is torch.float16:
        # A published bnb release pins bnb_4bit_compute_dtype in its own
        # config, and every one we use pins bfloat16. On a Turing card
        # (no bf16 matmul) the dequantized activations then meet bf16
        # weights and the matmul raises. Re-state the published config with
        # the compute dtype the card can run; every other field is copied,
        # so the codes and scales loaded are still the vendor's.
        from transformers import AutoConfig, BitsAndBytesConfig

        pub = getattr(AutoConfig.from_pretrained(model_id),
                      "quantization_config", None)
        if pub is not None:
            get = (pub.get if isinstance(pub, dict)
                   else lambda k, d=None: getattr(pub, k, d))
            if get("quant_method") == "bitsandbytes":
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=get("bnb_4bit_quant_type", "nf4"),
                    bnb_4bit_use_double_quant=get(
                        "bnb_4bit_use_double_quant", False),
                    bnb_4bit_compute_dtype=torch.float16)
    try:
        return transformers.AutoModelForCausalLM.from_pretrained(
            model_id, **kwargs)
    except (ValueError, TypeError, KeyError) as e:
        # only a class/config mismatch means "this is a multimodal release";
        # disk-full and network errors must surface as themselves
        print(f"[load] AutoModelForCausalLM failed ({type(e).__name__}); "
              f"trying AutoModelForImageTextToText")
        return transformers.AutoModelForImageTextToText.from_pretrained(
            model_id, **kwargs)


def dequantize_in_place(model) -> int:
    """Replace every Linear4bit with a dense nn.Linear holding its
    dequantized weight. Returns the number of layers converted.

    A published 4-bit release has no dense twin to load, so the served model
    -- anchors with the cells filled -- has to be built from the release
    itself: dequantize, then overwrite the wrapped matrices with the saved
    map. Everything that is not a Linear4bit stays exactly as released.
    """
    import torch.nn as nn
    from bitsandbytes.functional import dequantize_4bit

    n = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            kind = type(child).__name__
            if kind == "Linear4bit":
                w = child.weight
                # dequantize_4bit returns the quant_state's dtype, which for
                # every published release we use is bfloat16. On a Turing
                # card the rest of the model is fp16 and the first matmul
                # against one of these raises, so match the compute dtype
                # here exactly as the PackedLinear branch below does.
                dense = dequantize_4bit(w.data, w.quant_state).to(COMPUTE_DTYPE)
            elif kind == "PackedLinear":
                dense = child.dequant(COMPUTE_DTYPE)
            else:
                continue
            lin = nn.Linear(child.in_features, child.out_features,
                            bias=child.bias is not None,
                            device=dense.device, dtype=dense.dtype)
            lin.weight.data.copy_(dense.reshape(child.out_features,
                                                child.in_features))
            if child.bias is not None:
                lin.bias.data.copy_(child.bias.data)
            setattr(module, child_name, lin)
            n += 1
    left = [n_ for n_, p_ in model.named_parameters()
            if type(p_).__name__ == "Params4bit"]
    if left:
        raise RuntimeError(f"{len(left)} 4-bit parameters are not Linear4bit "
                           f"weights and were not converted: {left[:3]}")
    return n


def load_dense(model_id: str, device_map, dtype):
    """The model as a dense module: a dense release loads in `dtype`; a
    published 4-bit release loads as published and is dequantized layer by
    layer (dequantize_in_place), so its served form can be evaluated."""
    from transformers import AutoConfig

    published = getattr(AutoConfig.from_pretrained(model_id),
                        "quantization_config", None)
    get = (published.get if isinstance(published, dict)
           else (lambda k, d=None: getattr(published, k, d))) if published \
        else (lambda k, d=None: d)
    if get("quant_method") == "compressed-tensors":
        from cellfill.uniform import pack_model_in_place

        model = _load_model(model_id, device_map, dtype)
        pack_model_in_place(model)
    else:
        model = _load_model(model_id, device_map, dtype)
    if published:
        n = dequantize_in_place(model)
        print(f"[load] {model_id}: published 4-bit, dequantized {n} layers "
              f"to dense for evaluation", flush=True)
    return model


def build_4bit(model_id: str):
    """Load a 4-bit model, quantizing only if the checkpoint is not already one.

    An officially released bnb-4bit checkpoint carries its own
    quantization_config and must be loaded as published -- re-quantizing it
    would produce our codes rather than theirs, and the artifact we then claim
    to preserve would again be one we made. When a release exists it is the
    anchor; we quantize ourselves only when none does.
    """
    from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Published 4-bit releases can ship a tokenizer configured for left
    # padding (unsloth's do). Under left padding the final pad token carries
    # a label -- the first real token -- so every training sequence gains a
    # "predict the opening word from padding" term. Measured on the same
    # anchor and batch: 3.24 nats right-padded, 3.56 left-padded, and a
    # 17-point recall gap between two runs that differed in nothing else.
    # Training pads on the right; eval_recall sets its own side to generate.
    tok.padding_side = "right"
    published = getattr(AutoConfig.from_pretrained(model_id),
                        "quantization_config", None)
    if published:
        get = (published.get if isinstance(published, dict)
               else lambda k, d=None: getattr(published, k, d))
        method = get("quant_method")
        if method == "compressed-tensors":
            # GPTQ / AWQ / QAT releases: a uniform int4 grid, kept packed
            # (cellfill.uniform.PackedLinear) so the model is not inflated
            # to dense on its first forward.
            from cellfill.uniform import pack_model_in_place

            model = _load_model(model_id, DEVICE_MAP, COMPUTE_DTYPE)
            n = pack_model_in_place(model)
            print(f"[load] {model_id} is a compressed-tensors release; "
                  f"{n} layers kept packed on a uniform grid", flush=True)
            return model, tok
        qt = get("bnb_4bit_quant_type")
        if qt != "nf4":
            raise ValueError(
                f"{model_id} ships a {qt!r} grid; the cell arithmetic here is "
                "NF4 or a uniform integer grid. Use such a release or an "
                "unquantized model.")
        print(f"[load] {model_id} is already 4-bit as published; "
              "loading it unchanged", flush=True)
        return _load_model(model_id, DEVICE_MAP, COMPUTE_DTYPE), tok
    cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=COMPUTE_DTYPE,
    )
    model = _load_model(model_id, DEVICE_MAP, COMPUTE_DTYPE,
                        quantization_config=cfg)
    return model, tok


def add_lora(model, rank: int, alpha: int, target_modules="all-linear"):
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules=target_modules,
    )
    return get_peft_model(model, cfg)


class fills_off:
    """Serve the anchors alone: every fill's share of the half-width is set
    to zero for the duration, so the forward pass is the released model's.
    Used to take a teacher distribution from the same loaded model."""

    def __init__(self, model):
        self.mods = [m for m in model.modules() if hasattr(m, "fill_frac")]

    def __enter__(self):
        self.saved = [m.fill_frac for m in self.mods]
        for m in self.mods:
            m.fill_frac = 0.0

    def __exit__(self, *a):
        for m, f in zip(self.mods, self.saved):
            m.fill_frac = f


def kl_to_base(model, enc, logits_fill):
    """KL(p_base || p_fill) per non-pad token, the base taken with fills off.

    The replay LM loss lets the fill drift anywhere the next-token loss does
    not notice; the KL forbids any change of the served distribution on the
    pool, which is the property a fill needs to be mergeable: inert on
    inputs it was not written for (exp_fuse --inert)."""
    with torch.no_grad(), fills_off(model):
        base = model(**enc).logits
    # one sequence at a time: the fp32 log-softmax of a full batch over a
    # 150k vocabulary is half a gigabyte per copy, and four copies of it on
    # top of the training graph exhausted a 24 GB card
    mask = enc.attention_mask[:, 1:].float()
    total = logits_fill.new_zeros((), dtype=torch.float32)
    for i in range(logits_fill.shape[0]):
        lp_b = torch.log_softmax(base[i, :-1].float(), dim=-1)
        lp_f = torch.log_softmax(logits_fill[i, :-1].float(), dim=-1)
        kl = (lp_b.exp() * (lp_b - lp_f)).sum(-1)
        total = total + (kl * mask[i]).sum()
    return total / mask.sum().clamp(min=1)


def train(model, tok, texts, epochs, lr, bs, seed=0, replay_pool=None,
          n_replay_per_epoch=0, ckpt_path=None, on_epoch_end=None,
          kl_pool=None, kl_weight=0.0, kl_bs=None):
    """Facts every epoch; replay drawn FRESH each epoch from a large pool.
    A small fixed buffer repeated every epoch gets memorized and sharpens the
    LM onto itself (exp1: PPL 24.6 -> 172) — rehearsal must not repeat.

    ckpt_path enables resume. The 100k-fact run is ~20 hours, and losing one
    to a power cut costs more than the checkpoint does: after every epoch we
    write the trainable parameters and the optimizer state, and on start we
    pick up from the last completed epoch. Rehearsal stays reproducible
    because which snippets an epoch draws is a function of the epoch index.
    """
    import random as pyrandom

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    rng = pyrandom.Random(seed)
    model.train()
    losses, step = [], 0
    start_ep = 0
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if ckpt_path and Path(ckpt_path).exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        by_name = dict(named)
        for n, v in ck["params"].items():
            by_name[n].data.copy_(v.to(by_name[n].device,
                                       by_name[n].dtype))
        opt.load_state_dict(ck["opt"])
        start_ep, losses, step = ck["epoch"] + 1, ck["losses"], ck["step"]
        print(f"[resume] {ckpt_path}: continuing at epoch {start_ep}/{epochs}",
              flush=True)
    for ep in range(start_ep, epochs):
        epoch_texts = list(texts)
        if replay_pool and n_replay_per_epoch:
            k0 = (ep * n_replay_per_epoch) % len(replay_pool)
            epoch_texts += [replay_pool[(k0 + j) % len(replay_pool)]
                            for j in range(n_replay_per_epoch)]
        order = list(range(len(epoch_texts)))
        rng.shuffle(order)
        for i in range(0, len(order), bs):
            batch = [epoch_texts[j] for j in order[i:i + bs]]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=96).to(model.device)
            labels = enc.input_ids.clone()
            labels[enc.attention_mask == 0] = -100
            loss = model(**enc, labels=labels).loss
            if os.environ.get("CELLFILL_NAN_TRIP") and \
                    not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite loss at epoch {ep} step {step}")
            loss.backward()
            if step == 0:
                # Two things are checked once, at the step where they can
                # still be acted on. The gradient norm is the important one:
                # gradient checkpointing on a fully frozen base can silently
                # produce no gradients at all -- a checkpointed segment whose
                # inputs never require grad returns an output that does not
                # either -- and the only symptom is that the memory problem
                # went away while nothing was learned. The memory line says
                # what the step actually cost, with the batch shape, since
                # the batch is padded to its longest member.
                gn = sum(float(q.grad.abs().sum()) for q in params
                         if q.grad is not None)
                n_none = sum(1 for q in params if q.grad is None)
                if gn == 0.0 or n_none == len(params):
                    raise RuntimeError(
                        f"no gradient reached the {len(params)} trainable "
                        f"tensors at the first step ({n_none} are None): the "
                        "backward is not entering the trainable modules. With "
                        "gradient checkpointing this is the reentrant path on "
                        "a frozen base; pass use_reentrant=False.")
                if torch.cuda.is_available():
                    print("[mem] first-step shape=%s grad-norm %.4e  "
                          % (tuple(enc.input_ids.shape), gn) + "  ".join(
                              "cuda%d allocated %.1f reserved %.1f GB"
                              % (d, torch.cuda.memory_allocated(d) / 2**30,
                                 torch.cuda.memory_reserved(d) / 2**30)
                              for d in range(torch.cuda.device_count())),
                          flush=True)
            if kl_pool and kl_weight > 0:
                # a fresh slice of the neutral pool every step; the teacher
                # is this same model with its fills switched off. Its own
                # backward pass, after the fact batch's graph is freed: the
                # two graphs together exceeded a 24 GB card at 1.7B.
                k0 = (step * (kl_bs or bs)) % len(kl_pool)
                kb = [kl_pool[(k0 + j) % len(kl_pool)]
                      for j in range(kl_bs or bs)]
                enc_k = tok(kb, return_tensors="pt", padding=True,
                            truncation=True, max_length=96).to(model.device)
                kl = kl_to_base(model, enc_k, model(**enc_k).logits)
                (kl_weight * kl).backward()
                loss = loss.detach() + kl_weight * kl.detach()
            opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(float(loss))
            step += 1
            if step % 25 == 0:
                print(f"  ep{ep} step{step} loss {sum(losses[-25:]) / 25:.4f}",
                      flush=True)
        if ckpt_path:
            tmp = f"{ckpt_path}.tmp"
            torch.save({"epoch": ep, "step": step, "losses": losses,
                        "params": {n: p.detach().cpu() for n, p in named},
                        "opt": opt.state_dict()}, tmp)
            Path(tmp).replace(ckpt_path)   # atomic: a crash mid-write is safe
        # A recall-damage trajectory needs points along training, not just the
        # endpoint: a deployment picks its operating point off this curve.
        # The callback evaluates and must restore training mode itself.
        if on_epoch_end is not None:
            on_epoch_end(ep, losses)
    # The optimizer's moments are two more copies of every trainable
    # tensor -- 6.0 GB over a 32B model's MLP fills -- and what runs
    # next is an evaluation that wants the card. Dropping it here is
    # what keeps a validation from starting 6 GB down, and the line
    # printed is the check that it did.
    import gc

    del opt
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print('[mem] post-train ' + '  '.join(
            'cuda%d allocated %.1f reserved %.1f GB'
            % (d, torch.cuda.memory_allocated(d) / 2**30,
               torch.cuda.memory_reserved(d) / 2**30)
            for d in range(torch.cuda.device_count())), flush=True)

    return losses


@torch.no_grad()
def scorer_stamp():
    """A digest of the code that turns a generation into a score.

    Two result files in this project carry the same configuration, the same
    perplexity to two decimals -- the same trained model -- and recalls of
    0.7396 and 0.9566, because the scorer changed between them and nothing in
    either file said so. A number was then read from the superseded one and
    put in a table. The fix is not vigilance: it is that every result says
    which ruler measured it.

    A git hash would not do here, because the tree is routinely dirty and the
    hash would name a commit that is not what ran. This digests the source of
    the scoring functions themselves, so it changes exactly when the scoring
    changes and at no other time.
    """
    # The match is inline in eval_recall, so its source is the ruler. This is
    # stamp_of(eval_recall) and is kept as a name because two scripts and a
    # test refer to it; the digest is computed in one place so the two cannot
    # drift apart.
    return stamp_of(eval_recall)


def stamp_of(*fns):
    """scorer_stamp generalized: a digest of THESE measuring functions.

    Each result-writing script measures with its own ruler -- eval_ppl for
    the geometry walk, an inline hit() matcher for the two-hop ceiling, an
    execute-and-check pair for the cyclopts tasks -- so each stamps the
    source of the functions it actually scores with, not eval_recall's.
    """
    import hashlib
    import inspect

    h = hashlib.sha256()
    for fn in fns:
        try:
            h.update(inspect.getsource(fn).encode())
        except (OSError, TypeError):
            h.update(repr(fn).encode())
    return h.hexdigest()[:12]


def eval_recall(model, tok, pairs, bs=48, max_new=32, detail=False):
    """Exact-prefix recall. Probes are (prompt, expect, kind) triples; with

    Scoring is monotone in max_new: the match looks only at the prefix of the
    generation, so extra tokens cannot turn a hit into a miss, only a
    truncation into a hit. A tight budget therefore has no upside and one
    sharp edge -- at 16 it cut the last token off every Powerball answer and
    reported a whole domain as unlearned. The default is generous; the
    archived synthetic runs, whose answers are one to three tokens, score
    identically under it.

    detail=True also returns the per-kind breakdown, which separates
    near-verbatim continuations (city/job) from the paraphrased probe
    (company) instead of averaging them into one number."""
    model.eval()
    tok.padding_side = "left"
    hits = 0
    per_kind: dict[str, list[int]] = {}
    for i in range(0, len(pairs), bs):
        batch = pairs[i:i + bs]
        enc = tok([p[0] for p in batch], return_tensors="pt",
                  padding=True).to(model.device)
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        gen = tok.batch_decode(out[:, enc.input_ids.shape[1]:],
                               skip_special_tokens=True)
        for g, probe in zip(gen, batch):
            expect = probe[1]
            kind = probe[2] if len(probe) > 2 else "all"
            ok = int(g.strip().lower().startswith(expect.strip().lower()))
            hits += ok
            per_kind.setdefault(kind, []).append(ok)
    tok.padding_side = "right"
    overall = hits / len(pairs)
    if not detail:
        return overall
    return overall, {k: sum(v) / len(v) for k, v in per_kind.items()}


@torch.no_grad()
def eval_ppl(model, tok, text, ctx=1024, max_chunks=40):
    model.eval()
    ids = tok(text, return_tensors="pt").input_ids[0]
    nll, ntok = 0.0, 0
    limit = min(ids.numel() - 1, ctx * max_chunks)
    for i in range(0, limit, ctx):
        chunk = ids[i:i + ctx + 1]
        if chunk.numel() < 32:
            break
        inp = chunk[:-1].unsqueeze(0).to(model.device)
        tgt = chunk[1:].unsqueeze(0).to(model.device)
        logits = model(inp).logits.float()
        nll += float(torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), tgt.view(-1), reduction="sum"))
        ntok += tgt.numel()
    return math.exp(nll / ntok)


def wikitext_text():
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    return "\n".join(ds["text"])


def wikitext_train_snippets(n, seed=0, lo=200, hi=600):
    """Replay corpus: paragraphs from the TRAIN split (strictly disjoint from
    the TEST split used for PPL evaluation; wikitext-103 shares its valid/test
    with wikitext-2, so its train set is also safe). Large pools auto-switch
    to wikitext-103 so snippets stay fresh instead of cycling."""
    import random as pyrandom

    from datasets import load_dataset

    config = "wikitext-103-raw-v1" if n > 4000 else "wikitext-2-raw-v1"
    ds = load_dataset("Salesforce/wikitext", config, split="train")
    paras = [t.strip() for t in ds["text"] if lo <= len(t.strip()) <= hi]
    rng = pyrandom.Random(seed)
    return rng.sample(paras, min(n, len(paras)))


# Rehearsal sources. wikitext is the paper's default and shares its domain
# with the WikiText perplexity metric, which is how the 10k run kept WikiText
# below the anchor while LAMBADA rose 4.5x. These give a broader mixture so
# the ablation can ask whether the cross-domain cost is a property of the
# method or of a narrow rehearsal set.
# One shard of C4 rather than the whole set: 300 MB is enough for a pool of
# a few thousand snippets and the arms differ in distribution, not in size.
# (stas/openwebtext-10k was the first choice and is a loading script, which
# datasets 5.x no longer runs.)
REPLAY_SOURCES = {
    "wikitext": None,                       # handled by the function above
    "pile": dict(path="NeelNanda/pile-10k", split="train", field="text"),
    "c4": dict(path="allenai/c4", split="train", field="text",
               data_files={"train": "en/c4-train.00000-of-01024.json.gz"}),
}


def replay_snippets(n, seed=0, source="wikitext", lo=200, hi=600):
    """Rehearsal pool from one source, or an equal mixture of all of them.

    Snippets are length-filtered the same way in every source so the arms
    differ in distribution and not in sequence length.
    """
    import random as pyrandom

    if source == "wikitext":
        return wikitext_train_snippets(n, seed=seed, lo=lo, hi=hi)
    if source == "mixed":
        names = ["wikitext"] + [k for k in REPLAY_SOURCES if k != "wikitext"]
        per = -(-n // len(names))
        out, used = [], []
        for i, nm in enumerate(names):
            try:
                out += replay_snippets(per, seed=seed + 17 * i, source=nm,
                                       lo=lo, hi=hi)
                used.append(nm)
            except Exception as e:  # noqa: BLE001 -- a mixture missing one
                # component is still a mixture; silently dropping it is not
                print(f"[replay] source {nm} unavailable "
                      f"({type(e).__name__}), excluded from the mixture",
                      flush=True)
        print(f"[replay] mixture over {used}", flush=True)
        pyrandom.Random(seed).shuffle(out)
        return out[:n]
    from datasets import load_dataset

    spec = dict(REPLAY_SOURCES[source])
    field = spec.pop("field")
    ds = load_dataset(spec.pop("path"), **spec)
    paras = []
    for t in ds[field]:
        for chunk in str(t).split("\n"):
            c = chunk.strip()
            if lo <= len(c) <= hi:
                paras.append(c)
        if len(paras) > 20 * n:
            break
    rng = pyrandom.Random(seed)
    return rng.sample(paras, min(n, len(paras)))


def load_original_weights(model_id):
    """Original fp32 weights on CPU, keyed by parameter prefix (sans .weight)."""
    m = _load_model(model_id, "cpu", torch.float32)
    out = {n[: -len(".weight")]: p.detach().clone()
           for n, p in m.named_parameters() if n.endswith(".weight")}
    del m
    return out


def lambada_text(max_examples=400):
    """Cross-domain held-out text (LAMBADA): de-confounds the in-domain
    wikitext replay from the wikitext PPL metric. Returns None if the dataset
    is unreachable — callers then skip the cross-domain PPL."""
    try:
        from datasets import load_dataset

        ds = load_dataset("EleutherAI/lambada_openai", "en", split="test")
        return "\n\n".join(ds["text"][:max_examples])
    except Exception as e:  # noqa: BLE001
        print(f"[warn] lambada unavailable ({e}); cross-domain PPL skipped")
        return None


def maybe_ppl(model, tok, text, max_chunks):
    return None if text is None else eval_ppl(model, tok, text,
                                              max_chunks=max_chunks)


def shrink_bounds(lo, hi, anchors_flat, radius):
    """rho-radius trust region: keep only a fraction of each bin's room."""
    if radius >= 1.0:
        return lo, hi
    a = anchors_flat.float()
    return a - radius * (a - lo), a + radius * (hi - a)


def collect_frozen_plain(model_4bit, name_filter: str = "",
                         map_dtype=torch.float32):
    """Frozen artifact from a plain (un-LoRA'd) 4-bit model: merged==anchors."""
    merged, anchors_map, frozen_states = {}, {}, {}
    for name, m in model_4bit.named_modules():
        if type(m).__name__ != "Linear4bit":
            continue
        if name_filter and name_filter not in name:
            continue
        fs = frozen_state_from_linear4bit(m)
        anchors_map[name] = fs["anchors"].to(map_dtype).cpu()
        merged[name] = fs["anchors"].to(map_dtype).cpu()
        frozen_states[name] = (fs["codes"].cpu(), fs["absmax"].cpu(),
                               fs["blocksize"])
    return merged, anchors_map, frozen_states


_DELTA_DUMP: list = []   # CELLFILL_DUMP_DELTA instrument, saved by merge_all


def merge_all(peft_model, margin: float, orig_map=None, radius: float = 1.0,
              name_filter: str = "", map_dtype=torch.float32,
              store_frozen=True, store_anchors=True):
    """Clip-merge every LoRA-wrapped Linear4bit. Returns (merged_map, stats)."""
    merged, anchors_map, frozen_states, per_layer = {}, {}, {}, {}
    tot = dict(n=0, n_clipped=0, sq_delta=0.0, sq_kept=0.0)
    n_layers = 0
    for name, m in peft_model.named_modules():
        if not (hasattr(m, "base_layer")
                and type(m.base_layer).__name__ == "Linear4bit"):
            continue
        if not hasattr(m, "lora_A") or "default" not in m.lora_A:
            continue
        if name_filter and name_filter not in name:
            continue
        fs = frozen_state_from_linear4bit(m.base_layer)
        codes, absmax, bsz = fs["codes"], fs["absmax"], fs["blocksize"]
        A = m.lora_A["default"].weight.detach().float()
        B = m.lora_B["default"].weight.detach().float()
        delta = (B @ A) * m.scaling["default"]
        clean = name.replace("base_model.model.", "", 1)
        lo, hi = bin_bounds(codes, absmax, bsz, capped=True, margin=margin)
        lo, hi = shrink_bounds(lo, hi, fs["anchors"].reshape(-1), radius)
        base = fs["anchors"] if orig_map is None else orig_map[clean].to(lo.device)
        w_new, st = clip_merge(base, delta, lo, hi)
        ok, n_mis, _ = check_invariance(w_new, codes, absmax, bsz)
        if not ok:
            raise RuntimeError(f"{name}: {n_mis} invariance violations after clip_merge")

        merged[clean] = w_new.reshape(fs["shape"]).to(map_dtype).cpu()
        if store_anchors:
            anchors_map[clean] = fs["anchors"].to(map_dtype).cpu()
        if store_frozen:
            frozen_states[clean] = (codes.cpu(), absmax.cpu(), bsz)
        per_layer[clean] = dict(
            clipped_frac=st.clipped_frac,
            norm_kept_frac=st.norm_kept_frac,
            delta_norm=st.delta_norm,
            n=st.n_total,
        )
        if os.environ.get("CELLFILL_DUMP_DELTA"):
            # Step-metering instrument (the preconditioner question): how
            # large are the raw LoRA steps in units of each weight's own
            # cell half-width? Bounded training cannot exceed 1 by
            # construction; unmetered training has no reason to respect it.
            ad = delta.abs().reshape(-1)
            hw_pw = ((hi - lo) / 2).clamp_min(1e-30)
            occ = ad / hw_pw
            q = torch.tensor([0.5, 0.9, 0.99], device=ad.device)
            sub = occ[:: max(1, occ.numel() // 1_000_000)]
            _DELTA_DUMP.append(dict(
                layer=clean, n=int(ad.numel()),
                abs_delta_q=torch.quantile(
                    ad[:: max(1, ad.numel() // 1_000_000)], q).tolist(),
                occupancy_q=torch.quantile(sub, q).tolist(),
                occupancy_max=float(occ.max()),
                occupancy_gt1=float((occ > 1).float().mean()),
                clipped_frac=st.clipped_frac,
            ))
        tot["n"] += st.n_total
        tot["n_clipped"] += st.n_clipped
        tot["sq_delta"] += st.delta_norm ** 2
        tot["sq_kept"] += st.kept_norm ** 2
        n_layers += 1
        del fs, delta, w_new, lo, hi
        if n_layers % 32 == 0:
            torch.cuda.empty_cache()
            print(f"  merged {n_layers} layers...", flush=True)

    stats = dict(
        n_layers=n_layers,
        n_weights=tot["n"],
        clipped_frac=tot["n_clipped"] / max(tot["n"], 1),
        norm_kept_frac=math.sqrt(tot["sq_kept"] / tot["sq_delta"])
        if tot["sq_delta"] > 0 else 1.0,
    )
    dump_path = os.environ.get("CELLFILL_DUMP_DELTA")
    if dump_path and _DELTA_DUMP:
        torch.save(dict(layers=list(_DELTA_DUMP), stats=stats), dump_path)
        print(f"[dump-delta] {len(_DELTA_DUMP)} layers -> {dump_path}",
              flush=True)
        _DELTA_DUMP.clear()
    return merged, anchors_map, frozen_states, per_layer, stats


def heal_projected(model_id, merged_map, frozen_states, tok, texts, pairs,
                   ppl_txt, epochs, lr, bs, seed, margin, max_ppl_chunks,
                   replay_pool=None, n_replay_per_epoch=0, clamp=True,
                   radius: float = 1.0, xdom_txt=None, optim="adamw",
                   save_path=None):
    """A+ path: short projected dense fine-tune from the clip-merged weights.

    Post-hoc clip is the Euclidean projection of the unconstrained optimum —
    nearest in weight space, not best in loss. Here the optimizer sees the box
    (clamp after every step) and can re-route clipped-away knowledge into
    coordinates that still have room. Saturated coordinates are handled
    implicitly (projected GD = active-set behavior at the walls).
    """
    import random as pyrandom

    model = _load_model(model_id, {"": 0}, torch.float32)
    for p in model.parameters():
        p.requires_grad_(False)

    trainable, bounds = [], {}
    for name, w in merged_map.items():
        p = model.get_parameter(f"{name}.weight")
        p.data.copy_(w.to(p.device))
        p.requires_grad_(True)
        codes, absmax, bsz = frozen_states[name]
        lo, hi = bin_bounds(codes.to(p.device), absmax.to(p.device), bsz,
                            capped=True, margin=margin)
        if radius < 1.0:
            anch = dequantize_ref(codes.to(p.device), absmax.to(p.device), bsz)
            lo, hi = shrink_bounds(lo, hi, anch, radius)
        # fp16 bounds halve memory, but the cast must round INWARD: fp16 has
        # ~4.9e-4 relative precision, and for a narrow cell sitting far from
        # zero that error can exceed the margin and let a weight out. (It did:
        # the invariance assertion caught 130 escapes at 4B before this
        # shrink was added.) 1e-3 relative is a safe inward pad.
        pad_lo = lo.abs() * 1e-3
        pad_hi = hi.abs() * 1e-3
        bounds[name] = ((lo + pad_lo).view(p.shape).half(),
                        (hi - pad_hi).view(p.shape).half())
        trainable.append((name, p))

    if optim == "adam8bit":
        from bitsandbytes.optim import PagedAdamW8bit

        opt = PagedAdamW8bit([p for _, p in trainable], lr=lr)
    else:
        opt = torch.optim.AdamW([p for _, p in trainable], lr=lr)
    rng = pyrandom.Random(seed + 1)
    model.train()
    step = 0
    for ep in range(epochs):
        epoch_texts = list(texts)
        if replay_pool and n_replay_per_epoch:
            k0 = (ep * n_replay_per_epoch) % len(replay_pool)
            epoch_texts += [replay_pool[(k0 + j) % len(replay_pool)]
                            for j in range(n_replay_per_epoch)]
        order = list(range(len(epoch_texts)))
        rng.shuffle(order)
        for i in range(0, len(order), bs):
            batch = [epoch_texts[j] for j in order[i:i + bs]]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=96).to(model.device)
            labels = enc.input_ids.clone()
            labels[enc.attention_mask == 0] = -100
            with torch.autocast("cuda", dtype=COMPUTE_DTYPE):
                loss = model(**enc, labels=labels).loss
            if os.environ.get("CELLFILL_NAN_TRIP") and \
                    not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite loss at epoch {ep} step {step}")
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            if clamp:
                with torch.no_grad():
                    for name, p in trainable:
                        lo, hi = bounds[name]
                        p.data.clamp_(lo, hi)
            step += 1
            if step % 50 == 0:
                print(f"  [heal] ep{ep} step{step} loss {float(loss):.4f}",
                      flush=True)

    n_bad, n_sat, n_tot = 0, 0, 0
    with torch.no_grad():
        for name, p in trainable:
            codes, absmax, bsz = frozen_states[name]
            ok, n_mis, _ = check_invariance(
                p.data, codes.to(p.device), absmax.to(p.device), bsz)
            n_bad += n_mis
            lo, hi = bounds[name]
            n_sat += int(((p.data <= lo) | (p.data >= hi)).sum().item())
            n_tot += p.numel()
    if clamp and n_bad:
        raise RuntimeError(f"healing broke invariance on {n_bad} weights")

    if save_path:
        torch.save({n: p.data.half().cpu() for n, p in trainable}, save_path)
        print(f"[save] healed weights -> {save_path}")

    recall, recall_kinds = eval_recall(model, tok, pairs, detail=True)
    ppl = eval_ppl(model, tok, ppl_txt, max_chunks=max_ppl_chunks)
    ppl_x = maybe_ppl(model, tok, xdom_txt, max_ppl_chunks)
    del model, bounds
    torch.cuda.empty_cache()
    return dict(recall=recall, recall_by_kind=recall_kinds, ppl=ppl,
                ppl_x=ppl_x, saturation=n_sat / n_tot,
                invariance_violations=n_bad)


def eval_fp32_variants(model_id, anchors_map, merged_map, tok, pairs, ppl_txt,
                       max_ppl_chunks, xdom_txt=None, dtype=torch.float32,
                       device_map=None, max_new=32):
    """fp32 controls sharing one loaded model: anchors-only, then clip-merged.

    anchors-only isolates kernel/dtype numerics from the bnb 4-bit pipeline;
    (anchor→adapter) is training drift, (adapter→merged) is the clipping cost.
    """
    model = load_dense(model_id, device_map or {"": 0}, dtype)

    def overwrite(weight_map):
        for name, w in weight_map.items():
            p = model.get_parameter(f"{name}.weight")
            p.data.copy_(w.to(p.device))

    recall_o = eval_recall(model, tok, pairs, max_new=max_new)
    ppl_o = eval_ppl(model, tok, ppl_txt, max_chunks=max_ppl_chunks)
    ppl_ox = maybe_ppl(model, tok, xdom_txt, max_ppl_chunks)
    if anchors_map:
        overwrite(anchors_map)
        recall_a, recall_a_kinds = eval_recall(model, tok, pairs,
                                               detail=True, max_new=max_new)
        ppl_a = eval_ppl(model, tok, ppl_txt, max_chunks=max_ppl_chunks)
        ppl_ax = maybe_ppl(model, tok, xdom_txt, max_ppl_chunks)
    else:
        recall_a = ppl_a = ppl_ax = recall_a_kinds = None
    overwrite(merged_map)
    recall_m, recall_m_kinds = eval_recall(model, tok, pairs, detail=True,
                                           max_new=max_new)
    ppl_m = eval_ppl(model, tok, ppl_txt, max_chunks=max_ppl_chunks)
    ppl_mx = maybe_ppl(model, tok, xdom_txt, max_ppl_chunks)
    del model
    torch.cuda.empty_cache()
    return dict(recall_original_fp32=recall_o, ppl_original_fp32=ppl_o,
                recall_anchor_fp32=recall_a,
                recall_anchor_by_kind=recall_a_kinds, ppl_anchor_fp32=ppl_a,
                recall_merged=recall_m, recall_merged_by_kind=recall_m_kinds,
                ppl_merged=ppl_m,
                ppl_x_original_fp32=ppl_ox, ppl_x_anchor_fp32=ppl_ax,
                ppl_x_merged=ppl_mx)


def main():
    args = get_args()
    assert torch.cuda.is_available(), "exp0 needs the GPU server"
    torch.manual_seed(args.seed)
    t0 = time.time()

    if args.facts_file:
        # Real knowledge, verified absent from the base model rather than
        # unseen by construction (experiments/probe_cutoff.py). The bit
        # accounting does not carry over: the synthetic attributes are drawn
        # from closed vocabularies of known size, and a drug's active
        # ingredient is not. We report recall for these runs and leave
        # bits/pt to the synthetic corpus, which is what defines it.
        import json as _json

        train_path = Path(args.facts_file)
        probe_path = Path(args.probes_file or
                          str(train_path).replace("_train", "_probes"))
        texts = [r["text"] for r in _json.loads(train_path.read_text())]
        pairs = [(r["prompt"], r["answer"], r.get("domain", "real"))
                 for r in _json.loads(probe_path.read_text())]
        facts = texts
        args.n_facts = len(texts)
        print(f"[data] real corpus: {len(texts)} facts, {len(pairs)} probes "
              f"from {train_path.name}")
    else:
        facts = generate(args.n_facts, seed=args.seed)
        texts = training_texts(facts)
        pairs = probe_pairs(facts)
    if args.probe_cap and len(pairs) > args.probe_cap:
        import random as _prng

        pairs = _prng.Random(args.seed).sample(pairs, args.probe_cap)
        print(f"[eval] probe subsample: {len(pairs)} pairs")
    replay_pool, n_rep = None, 0
    if args.replay_frac > 0:
        n_rep = int(len(texts) * args.replay_frac / (1 - args.replay_frac))
        replay_pool = wikitext_train_snippets(n_rep * args.epochs, seed=args.seed)
        exposures = n_rep * args.epochs / len(replay_pool)
        print(f"[data] replay: {n_rep}/epoch, pool={len(replay_pool)}, "
              f"~{exposures:.1f} exposures/snippet (frac={args.replay_frac})")
    total_bits = None if args.facts_file else args.n_facts * bits_per_fact()
    if total_bits is not None:
        print(f"[data] {args.n_facts} facts × {bits_per_fact():.1f} bits "
              f"= {total_bits / 1e3:.1f} kbit of new knowledge")

    print(f"[load] {args.model} in NF4")
    model, tok = build_4bit(args.model)
    ppl_txt = wikitext_text()
    x_txt = lambada_text()

    recall_pre = eval_recall(model, tok, pairs)
    ppl_anchor = eval_ppl(model, tok, ppl_txt, max_chunks=args.max_ppl_chunks)
    ppl_anchor_x = maybe_ppl(model, tok, x_txt, args.max_ppl_chunks)
    print(f"[base] recall={recall_pre:.4f} (expect ≈0)  ppl={ppl_anchor:.3f}  "
          f"lambada={ppl_anchor_x}")

    if args.dense_only:
        losses = []
        recall_adapter = ppl_adapter = ppl_adapter_x = None
        print("[dense-only] skipping LoRA; projected dense training from anchors")
        merged, anchors_map, frozen_states = collect_frozen_plain(
            model, name_filter=args.name_filter,
            map_dtype=getattr(torch, args.map_dtype))
        per_layer, mstats = {}, dict(note="dense-only: merged==anchors pre-heal")
    else:
        model = add_lora(model, args.rank, args.alpha,
                         target_modules=args.target_regex or "all-linear")
        model.print_trainable_parameters()
        print(f"[train] {args.epochs} epochs, lr={args.lr}, bs={args.bs}")
        losses = train(model, tok, texts, args.epochs, args.lr, args.bs,
                       args.seed, replay_pool=replay_pool,
                       n_replay_per_epoch=n_rep, ckpt_path=args.ckpt)

        recall_adapter = eval_recall(model, tok, pairs)
        ppl_adapter = eval_ppl(model, tok, ppl_txt,
                               max_chunks=args.max_ppl_chunks)
        ppl_adapter_x = maybe_ppl(model, tok, x_txt, args.max_ppl_chunks)
        print(f"[adapter] recall={recall_adapter:.4f}  ppl={ppl_adapter:.3f}  "
              f"lambada={ppl_adapter_x}")
        adapter_dir = Path(args.out).parent / "adapter"
        model.save_pretrained(str(adapter_dir))
        print(f"[save] adapter -> {adapter_dir}")

        print(f"[merge] clip-merge into frozen bins (base={args.merge_base})")
        orig_map = (load_original_weights(args.model)
                    if args.merge_base == "original" else None)
        heal_planned = bool(args.heal_epochs)
        merged, anchors_map, frozen_states, per_layer, mstats = merge_all(
            model, args.margin, orig_map, radius=args.radius,
            name_filter=args.name_filter,
            map_dtype=getattr(torch, args.map_dtype),
            store_frozen=heal_planned,
            store_anchors=not args.skip_anchor_eval)
        del orig_map
    if not args.dense_only:
        print(f"[merge] layers={mstats['n_layers']} "
              f"weights={mstats['n_weights']:,} "
              f"clip_rate={mstats['clipped_frac']:.4%} "
              f"norm_kept={mstats['norm_kept_frac']:.4f} "
              f"invariance=100% (asserted per layer)")

    del model
    torch.cuda.empty_cache()

    print("[eval] fp32 controls: anchors-only, then clip-merged")
    fp32 = eval_fp32_variants(
        args.model, anchors_map, merged, tok, pairs, ppl_txt,
        args.max_ppl_chunks, xdom_txt=x_txt,
        dtype=getattr(torch, args.eval_dtype),
        device_map="auto" if args.eval_device_map == "auto" else {"": 0})
    print(f"[orig-fp32]   recall={fp32['recall_original_fp32']:.4f}  "
          f"ppl={fp32['ppl_original_fp32']:.3f}")
    if fp32["recall_anchor_fp32"] is not None:
        print(f"[anchor-fp32] recall={fp32['recall_anchor_fp32']:.4f}  "
              f"ppl={fp32['ppl_anchor_fp32']:.3f}")
    print(f"[merged]      recall={fp32['recall_merged']:.4f}  "
          f"ppl={fp32['ppl_merged']:.3f}")

    heal_res = None
    heal_epochs = args.heal_epochs or (args.epochs if args.dense_only else 0)
    if heal_epochs > 0:
        clamp = not args.no_clamp
        print(f"[heal] projected dense fine-tune: {heal_epochs} epochs, "
              f"lr={args.heal_lr}, clamp={clamp}")
        heal_res = heal_projected(
            args.model, merged, frozen_states, tok, texts, pairs, ppl_txt,
            heal_epochs, args.heal_lr, max(4, args.bs // 2), args.seed,
            args.margin, args.max_ppl_chunks, replay_pool, n_rep, clamp=clamp,
            radius=args.radius, xdom_txt=x_txt, optim=args.heal_optim,
            save_path=(str(Path(args.out).with_suffix("")) + "_healed.pt"
                       if args.save_merged else None))
        print(f"[healed] recall={heal_res['recall']:.4f}  "
              f"ppl={heal_res['ppl']:.3f}  "
              f"saturation={heal_res['saturation']:.4%}  "
              f"violations={heal_res['invariance_violations']}")

    worst = sorted(per_layer.items(), key=lambda kv: -kv[1]["clipped_frac"])[:10]
    # scorer_stamp was DEFINED in this file and imported by every other
    # script, yet this file's own products never carried it -- every
    # exp0-family JSON through 2026-08-28 (the whole LoRA table row set)
    # self-certifies nothing. Discovered when the 342 seeds landed unstamped.
    result = dict(
        scorer=scorer_stamp(),
        scorer_ppl=stamp_of(eval_ppl),
        # generation is not batch-invariant (left-padding depends on batch
        # neighbors; 97/507 generations differ between bs=48 and bs=8 on the
        # 1.7B base), so a recall number is only comparable at the same eval
        # batch size. Every call site here uses eval_recall's default.
        eval_bs=inspect.signature(eval_recall).parameters["bs"].default,
        config=vars(args),
        bits_per_fact=None if args.facts_file else bits_per_fact(),
        total_bits=total_bits,
        loss_first=losses[0] if losses else None,
        loss_last=sum(losses[-10:]) / 10 if losses else None,
        recall=dict(base_4bit=recall_pre, adapter=recall_adapter,
                    original_fp32=fp32["recall_original_fp32"],
                    anchor_fp32=fp32["recall_anchor_fp32"],
                    merged=fp32["recall_merged"],
                    merged_by_kind=fp32["recall_merged_by_kind"]),
        ppl=dict(anchors_4bit=ppl_anchor, adapter=ppl_adapter,
                 original_fp32=fp32["ppl_original_fp32"],
                 anchor_fp32=fp32["ppl_anchor_fp32"],
                 merged=fp32["ppl_merged"]),
        ppl_lambada=dict(anchors_4bit=ppl_anchor_x, adapter=ppl_adapter_x,
                         original_fp32=fp32["ppl_x_original_fp32"],
                         anchor_fp32=fp32["ppl_x_anchor_fp32"],
                         merged=fp32["ppl_x_merged"]),
        merge=mstats,
        heal=heal_res,
        worst_layers=worst,
        minutes=round((time.time() - t0) / 60, 1),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"[done] {out}  ({result['minutes']} min)")
    print(json.dumps({k: result[k] for k in ("recall", "ppl", "merge")}, indent=2))


if __name__ == "__main__":
    main()

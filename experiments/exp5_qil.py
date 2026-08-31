#!/usr/bin/env python
"""Exp5: QIL-LoRA — quantization-invariance BY CONSTRUCTION.

W = anchor + M ⊙ tanh(s·(B@A)): the learnable object is literally the position
inside each frozen bin (M = symmetric in-bin half-width). |tanh| < 1 keeps
every weight strictly interior at every training step, so requantization
invariance is structural — no clip, no projection, and the model evaluated
during training is exactly the model that ships.

Run: .venv/bin/python experiments/exp5_qil.py --n-facts 1000 --epochs 24 \
         --replay-frac 0.3 --out out/exp5_qil.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cellfill import bin_bounds, check_invariance  # noqa: E402
from cellfill.bins import (  # noqa: E402
    LAYOUTS, nibble_layout_of, normalized_halfwidth, unpack_nf4_codes,
)
from cellfill.bnb_state import frozen_state_from_linear4bit  # noqa: E402
from cellfill.nf4 import dequantize_ref  # noqa: E402
from experiments.exp0_clip_rate import (
    scorer_stamp,  # noqa: E402
    build_4bit,
    eval_fp32_variants,
    eval_ppl,
    eval_recall,
    lambada_text,
    maybe_ppl,
    train,
    wikitext_text,
    replay_snippets,
    wikitext_train_snippets,
)
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
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--tanh-scale", type=float, default=4.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--margin", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--replay-frac", type=float, default=0.0)
    p.add_argument("--probe-cap", type=int, default=30000)
    p.add_argument("--map-dtype", choices=["float32", "float16"],
                   default="float32")
    p.add_argument("--skip-anchor-eval", action="store_true")
    p.add_argument("--save-merged-weights", default=None,
                   help="write the merged fp16 weight map here, so downstream "
                        "benchmarks can be run on the served model without "
                        "retraining it")
    p.add_argument("--dump-probes", type=int, default=0,
                   help="print N (prompt, expected, generated) triples after "
                        "training; exact-match recall alone cannot distinguish "
                        "a near miss from a total failure")
    p.add_argument("--eval-base", default=None,
                   help="dense model to load for the merged-weight evaluation "
                        "when --model is an already-4-bit release. The wrapped "
                        "matrices are overwritten with the merged weights; "
                        "everything else must be identical between the two, "
                        "which experiments/verify_shared.py checks.")
    p.add_argument("--facts-file", default=None,
                   help="train on a real corpus instead of the synthetic "
                        "generator; see experiments/probe_cutoff.py for how "
                        "its absence from the base model is established")
    p.add_argument("--probes-file", default=None)
    p.add_argument("--save-anchor-weights", default=None,
                   help="write the dequantized 4-bit anchors here: the correct "
                        "baseline for a downstream benchmark, since the served "
                        "model is compared against the release it preserves")
    p.add_argument("--codebook-m", action="store_true",
                   help="store the per-weight bound as a 16-entry table plus "
                        "block scales instead of a dense tensor (48.7 GB -> "
                        "0.76 GB at 27B); rebuilt on demand from the codes")
    p.add_argument("--checkpoint-fill", action="store_true",
                   help="recompute the dense fill in the backward "
                        "pass; required above ~8B")
    p.add_argument("--eval-dtype", choices=["float32", "bfloat16"],
                   default="float32",
                   help="dtype for the full-model eval rebuild (27B needs "
                        "bfloat16: fp32 would ask for ~100 GiB)")
    p.add_argument("--target-filter", default=None,
                   help="regex over module names; only matching Linear4bit "
                        "get a BoundedFill (knowledge-localization ablation)")
    p.add_argument("--max-ppl-chunks", type=int, default=40)
    p.add_argument("--eval-every", type=int, default=0,
                   help="score the in-place model every N epochs and record "
                        "the trajectory: recall against cross-domain damage, "
                        "which is the curve a deployment picks its operating "
                        "point from. Costs one evaluation pass per point.")
    p.add_argument("--replay-source",
                   choices=["wikitext", "pile", "c4", "mixed"],
                   default="wikitext",
                   help="rehearsal distribution. wikitext shares its domain "
                        "with the WikiText metric; the others do not")
    p.add_argument("--stack", type=int, default=1,
                   help="deep residual stacking: this many factor pairs per "
                        "layer, composed by --stack-mode")
    p.add_argument("--stack-mode", choices=["drf", "sum"], default="drf")
    p.add_argument("--paraphrases", type=int, default=1,
                   help="synthetic facts stated in this many templates each "
                        "(the diversity lever of the capacity study)")
    p.add_argument("--fill-frac", type=float, default=1.0,
                   help="share of each cell's half-width the fill may use "
                        "(the width knob of the capacity study: if the "
                        "retained count scales with it, the cells are the "
                        "limit; if not, the parametrization is)")
    p.add_argument("--link", choices=["tanh", "hardtanh_ste", "softsign",
                                      "tanh_floor"], default="tanh",
                   help="bounded link. hardtanh_ste has the same bound and "
                        "the same reachable set but passes gradient where "
                        "tanh has none; the pair separates a capacity knee "
                        "from a saturation knee")
    p.add_argument("--save-fill", default=None,
                   help="save the low-rank factors (A, B) of every wrapped "
                        "layer with their settings: the served model can be "
                        "rebuilt from the release plus this file "
                        "(experiments.served.load_served), which at 31B is "
                        "0.4 GB instead of a 62 GB dense map")
    p.add_argument("--inplace-only", action="store_true",
                   help="do not build a dense copy of the served model: "
                        "verify invariance layer by layer and score the "
                        "in-place model only. For releases whose dense form "
                        "does not fit the card (31B packed int4 on 80 GB)")
    p.add_argument("--fill-stats", default=None,
                   help="write per-module fill statistics here: where in "
                        "the network the injection landed")
    p.add_argument("--kl-weight", type=float, default=0.0,
                   help="inert-writer term: KL(released || served) on a "
                        "neutral pool; keeps the function at the release "
                        "off the written facts (instruction-tuned releases "
                        "lose exam accuracy without it)")
    p.add_argument("--kl-bs", type=int, default=8)
    p.add_argument("--kl-pool-mode", choices=["replace", "add"],
                   default="replace",
                   help="whether --kl-pool-file replaces the neutral pool or "
                        "is added to it. Replacing moves the protection to "
                        "the named texts and takes it away from everything "
                        "else: on one library, aiming the pool at twelve "
                        "neighbouring symbols raised them from 41.7 to 50.0 "
                        "and dropped the thirty-five it no longer covered "
                        "from 16.7 to 12.9. The pool is a budget, not a "
                        "switch.")
    p.add_argument("--kl-pool-file", default=None,
                   help="draw the inert term's distillation pool from this "
                        "file instead of neutral WikiText. A JSON list of "
                        "probe records (prompt/answer) or of plain strings. "
                        "The neutral pool holds the function still where the "
                        "corpus is far away; it cannot hold it still where "
                        "the corpus is adjacent, because neutral text never "
                        "exercises the neighbouring symbols. Pointing the "
                        "pool at those symbols' own prompts distils on the "
                        "distribution the evaluation actually scores.")
    p.add_argument("--ckpt", default=None,
                   help="epoch-level checkpoint for resume (exp0.train); the "
                        "100k-fact run is ten hours on a 4090")
    p.add_argument("--out", default="out/exp5_qil.json")
    return p.parse_args()


class BoundedFill(nn.Module):
    """Low-rank bounded fill around a frozen Linear4bit.

    Default (anchors=None): the anchor stays implicit inside the 4-bit base
    layer and the fill is added to its output -- cheapest, used by the
    single-shot experiments.

    With explicit anchors: the layer holds its own anchor tensor and computes
    the full weight itself. Sequential experiments need this, because after
    each task the accumulated in-cell position becomes the next task's anchor
    (see exp_seq.fold_into_anchors); the 4-bit base can no longer represent it.
    """

    def __init__(self, base, halfwidth, rank, tanh_scale, anchors=None,
                 margin=0.01, codebook=None, uniform=None, groom=None,
                 walls=None, anchor_dtype=torch.float16, dense_fill=False):
        super().__init__()
        # The precision the anchors are stored at is not an implementation
        # detail: a fold writes through it, and it decides the floor on the
        # block scale below which invariance cannot be held
        # (cellfill.bins.anchor_absmax_floor). fp16 halves the memory and
        # makes a sequence fit at 27B; fp32 removes the floor and doubles it.
        self.anchor_dtype = anchor_dtype
        self.base = base
        self.groom_group = None
        if groom is not None:
            # a GGUF grid: the room is one number per block of `group`
            # weights (|s|(1/2 - margin)), kept as such and expanded per
            # forward pass -- 1/32 of a dense M
            room_g, group = groom
            out_f, in_f = base.weight.shape
            dev = base.weight.device
            self.register_buffer("groom", room_g.to(device=dev, dtype=torch.float32))
            self.groom_group = int(group)
        elif uniform is not None:
            # a PackedLinear on a uniform grid: the bound is a per-group
            # scale times a constant, see cellfill.uniform.uniform_halfwidth
            out_f, in_f = base.shape
            dev = base.packed.device
        else:
            out_f, in_f = halfwidth.shape
            dev = base.weight.device
        if dense_fill:
            # The parameterization axis: same bound, no rank bottleneck. A
            # rank-r fill writes every task through the same r shared
            # directions; a dense preimage lets each weight move
            # independently, which is the interference hypothesis the
            # sequence arms exist to test. Zero-init like B so the fill
            # starts exactly at the anchor.
            self.dense_fill = True
            self.A = None
            self.B = None
            self.Z = nn.Parameter(torch.zeros(out_f, in_f, device=dev))
        else:
            self.dense_fill = False
            self.A = nn.Parameter(torch.randn(rank, in_f, device=dev)
                                  / math.sqrt(in_f))
            self.B = nn.Parameter(torch.zeros(out_f, rank, device=dev))
        self.shape = (out_f, in_f)
        self.numel = out_f * in_f
        if groom is not None:
            self.uniform = False
            self.codebook = False
        # Share of the half-width this fill may use. 1 for a single writer;
        # 1/K when K parties write independently and their fills are summed
        # (exp_fuse.py): |sum| <= M then holds by construction, so the
        # aggregate is invariant without a clamp.
        self.fill_frac = 1.0
        self.link_fn = "tanh"
        self.uniform = uniform is not None
        if groom is not None:
            pass                      # the per-block room is already registered
        elif self.uniform:
            self.register_buffer("uscale", base.scale.to(device=dev,
                                                         dtype=torch.float32))
            self.ugroup, self.umargin = base.group, uniform
            self.codebook = False
        elif walls is not None:
            # the wall codebook rebuilds the room on demand; storing it too
            # would defeat the purpose. A one-element M keeps the attribute
            # (and its device and dtype) available to code that reads them.
            self.register_buffer("M", halfwidth.reshape(-1)[:1].to(
                device=dev, dtype=torch.bfloat16))
            self.codebook = False
        elif codebook is None:
            # bf16 buffer: rounding error on the bound (<0.5%) is absorbed by
            # the bin margin; the final merge recomputes M exactly in fp32.
            self.register_buffer("M", halfwidth.to(device=dev,
                                                   dtype=torch.bfloat16))
            self.codebook = False
        else:
            # M is a function of the frozen codes and scales, and the codes
            # are already resident inside the 4-bit layer we wrap. Keeping the
            # sixteen-entry table and the per-block scales instead of the
            # dense tensor costs 2 bytes per 64 weights rather than 2 per
            # weight: 0.76 GB rather than 48.7 GB at 27B.
            hw, absmax, blocksize, layout, frozen_idx = codebook
            # fp32, not fp16: measured against the dense tensor over all
            # 2.435e10 weights of the 27B model, the fp32 rebuild overshoots
            # by at most 3.7e-9 (float32 round-off) while the fp16 rebuild
            # overshoots by 3.5e-5. The scales are one value per 64 weights,
            # so fp32 costs 1.5 GB at 27B against the dense tensor's 45.4.
            self.register_buffer("hw", hw.to(device=dev, dtype=torch.float32))
            self.register_buffer("absmax_h",
                                 absmax.to(device=dev, dtype=torch.float32))
            # A handful of weights per model (7e3 in 2.4e10 at 27B) have a
            # stored nibble that disagrees with the code recovered by
            # re-binning the dequantized anchor -- the anchor landed on the
            # far side of a cell wall. Invariance is checked against the
            # re-binned code, so the nibble's cell is the wrong one to size
            # the bound with. We freeze those weights instead of guessing.
            self.register_buffer("frozen_idx", frozen_idx.to(dev))
            self.blocksize = blocksize
            self.layout = layout
            self.codebook = True
        # A wall codebook keeps the cell EDGES as two sixteen-entry tables
        # plus the per-block scales, and rebuilds the room from the CURRENT
        # anchor at every use: M_i = min(w_i - lo_i, hi_i - w_i). The edges are
        # functions of the frozen (codes, absmax) alone and no fold moves them,
        # so unlike the half-width table this survives folding -- which is what
        # lets a sequence run at 27B, where the dense M is 45 GB.
        self.walls = walls is not None
        if self.walls:
            # the same writable-cell floor the dense path applies, resolved
            # once against the anchor buffer's precision
            from cellfill.bins import anchor_absmax_floor

            self._floor = anchor_absmax_floor(anchor_dtype, margin)
            lo_e, hi_e, wabsmax, wbsz, wlayout, wfrozen = walls
            self.register_buffer("lo_e", lo_e.to(device=dev, dtype=torch.float32))
            self.register_buffer("hi_e", hi_e.to(device=dev, dtype=torch.float32))
            self.register_buffer("wabsmax",
                                 wabsmax.to(device=dev, dtype=torch.float32))
            self.register_buffer("wfrozen", wfrozen.to(dev))
            self.wbsz, self.wlayout = wbsz, wlayout
        if anchors is None:
            self.anchors = None
        else:
            self.register_buffer("anchors", anchors.to(device=dev,
                                                       dtype=anchor_dtype))
        self.tanh_scale = tanh_scale
        self.margin = margin
        self.checkpoint_fill = False
        # Deep residual stacking (the team's DRF proposal): L-1 further
        # factor pairs. stack_mode "drf": r <- r + (1-|r|) tanh(s B_l A_l),
        # the remaining room detached, so |r| < 1 by construction and each
        # fold is aimed at the room the previous ones left. "sum": the outer
        # link applied to the mean of the L inner links (rank L*r, the same
        # bound). Either is a different parametrization of the same
        # interval: the reachable set per weight is (-1, 1) in all cases.
        self.stack = nn.ParameterList()
        self.stack_mode = "drf"

    def halfwidth(self, dtype=torch.float32):
        """The per-weight bound M, dense or rebuilt from the frozen artifact.

        The rebuild gathers a sixteen-entry table by code and scales by the
        block's absmax, in fp32. Measured layer by layer against the dense
        tensor over all 2.435e10 weights of the 27B model, it overshoots by
        at most 3.7e-9 -- float32 round-off, five orders of magnitude below
        the margin the bin bounds already reserve. The table also gives back
        the dequant rounding allowance (cellfill.bins.normalized_halfwidth), and
        the few weights whose stored nibble disagrees with the re-binned code
        are frozen outright.
        """
        if self.groom_group is not None:
            return self.groom.repeat_interleave(self.groom_group)[:self.numel] \
                .view(self.shape).to(dtype)
        if self.uniform:
            from cellfill.uniform import uniform_halfwidth

            return uniform_halfwidth(self.uscale, self.ugroup, self.umargin,
                                     dtype=dtype)
        if self.walls:
            # Chunked, because the gather is what costs: the codes arrive
            # packed one byte per weight and indexing needs int64, so a
            # whole-tensor rebuild inflates them eightfold and holds seven
            # fp32 temporaries besides -- 3.2 GB for a single 89M-weight 27B
            # layer, which is where the unchunked version ran out of card.
            # Chunks are block-aligned so each one's scales are a clean slice.
            codes8 = unpack_nf4_codes(self.base.weight.data, self.numel,
                                      self.wlayout)
            flat = self.anchors.reshape(-1)
            out = torch.empty(self.numel, dtype=dtype, device=flat.device)
            step = max(1, (1 << 24) // self.wbsz) * self.wbsz
            for i in range(0, self.numel, step):
                j = min(i + step, self.numel)
                c = codes8[i:j].long()
                s = self.wabsmax[i // self.wbsz:
                                 -(-j // self.wbsz)] \
                    .repeat_interleave(self.wbsz)[:j - i]
                a, b = self.lo_e[c] * s, self.hi_e[c] * s
                lo, hi = torch.minimum(a, b), torch.maximum(a, b)
                width = hi - lo
                lo = lo + self.margin * width
                hi = hi - self.margin * width
                w = flat[i:j].float()
                ok = (self.wabsmax[i // self.wbsz:
                                   -(-j // self.wbsz)].abs() >= self._floor) \
                    .repeat_interleave(self.wbsz)[:j - i]
                out[i:j] = (torch.minimum(w - lo, hi - w).clamp_min(0)
                            * ok).to(dtype)
            if self.wfrozen.numel():
                out[self.wfrozen] = 0
            return out.view(self.shape)
        if not self.codebook:
            return self.M.to(dtype)
        codes = unpack_nf4_codes(self.base.weight.data, self.numel,
                                 self.layout).long()
        s = self.absmax_h.repeat_interleave(self.blocksize)[:self.numel]
        m = self.hw[codes] * s
        if self.frozen_idx.numel():
            m[self.frozen_idx] = 0
        return m.view(self.shape).to(dtype)

    def link(self, z):
        """Bounded link with |link(z)| < 1.

        tanh saturates: at 10k facts 78% of coordinates sit past |t| = 0.99,
        where the gradient 1 - t^2 is under 2e-2 and the fill is effectively a
        sign. A capacity knee measured there is a property of the link, not of
        the cells, so the alternative is available as a control: hardtanh with
        a straight-through gradient keeps the same bound and the same
        reachable set while passing gradient everywhere.
        """
        if self.link_fn == "tanh":
            return torch.tanh(z)
        if self.link_fn == "softsign":
            # z / (1 + |z|): the same bound, a gradient 1/(1+|z|)^2 that
            # decays polynomially rather than exponentially
            return z / (1 + z.abs())
        if self.link_fn == "tanh_floor":
            # tanh forward; backward uses max(1 - t^2, 0.05), so a saturated
            # coordinate keeps a twentieth of the live gradient and can be
            # revised when later facts disagree with it
            t = torch.tanh(z)
            g = (1 - t * t).clamp(min=0.05)
            return t.detach() + (z * g.detach() - (z * g.detach()).detach())
        h = z.clamp(-1.0, 1.0)          # hardtanh forward
        return z + (h - z).detach()     # straight-through backward

    def add_stack(self, depth, mode="drf"):
        """depth-1 extra factor pairs, B = 0 so the fill starts unchanged."""
        assert not getattr(self, "dense_fill", False), \
            "stacking is a low-rank construct; dense_fill has no stack"
        dev = self.A.device
        r, in_f = self.A.shape
        out_f = self.B.shape[0]
        for _ in range(depth - 1):
            a = nn.Parameter(torch.randn(r, in_f, device=dev) / math.sqrt(in_f))
            b = nn.Parameter(torch.zeros(out_f, r, device=dev))
            self.stack.append(a)
            self.stack.append(b)
        self.stack_mode = mode

    def raw_fill(self, dtype=torch.float32):
        """The pre-tanh preimage: dense Z or the low-rank product."""
        if getattr(self, "dense_fill", False):
            return self.Z.to(dtype)
        return (self.B @ self.A).to(dtype)

    def t_value(self, dtype=torch.float32):
        """The normalized in-cell position t in (-1, 1), all factors in."""
        t = self.link(self.tanh_scale * self.raw_fill(dtype))
        if not len(self.stack):
            return t
        assert not getattr(self, "dense_fill", False), \
            "stacking is a low-rank construct; dense_fill has no stack"
        pairs = [(self.stack[i], self.stack[i + 1])
                 for i in range(0, len(self.stack), 2)]
        if self.stack_mode == "sum":
            acc = t
            for a, b in pairs:
                acc = acc + self.link(self.tanh_scale * (b @ a).to(dtype))
            return self.link(acc / (len(pairs) + 1) * 2.0)
        r = t
        for a, b in pairs:
            room = (1.0 - r.abs()).detach()
            r = r + room * self.link(self.tanh_scale * (b @ a).to(dtype))
        return r

    def fill(self, dtype=torch.float32):
        """The in-cell displacement, |fill| < M elementwise by construction.

        Materialized in `dtype`: at 27B the fp32 intermediate for a
        5120x17408 matrix is 356 MB and the autograd graph holds one per
        layer, which is what exhausted an 80 GB card. bf16 halves it and the
        error is far below the cell margin; the final merge recomputes the
        fill in fp32 regardless.
        """
        if getattr(self, "unconstrained", False):
            # The control: the same factors, the same rank, the same
            # schedule, with the bound removed. Nothing here is clipped and
            # no cell is consulted, so the served weight is free to leave its
            # decision region -- which is the quantity the control exists to
            # measure.
            return self.raw_fill(dtype)
        t = self.t_value(dtype)
        return self.halfwidth(dtype) * (self.fill_frac * t)

    def _apply_fill(self, x, dt):
        d = self.fill(dt)
        if self.anchors is None:
            return self.base(x) + F.linear(x, d.to(x.dtype))
        w = self.anchors.to(d.dtype) + d
        return F.linear(x, w.to(x.dtype))

    def forward(self, x):
        dt = x.dtype if x.dtype != torch.float32 else torch.float32
        if self.checkpoint_fill and self.training:
            # The checkpoint has to span the matmul, not just the fill.
            # Checkpointing only the fill still leaves its dense output alive,
            # because F.linear saves it for its own backward -- one dense
            # matrix per layer, which is the same 48.7 GB at 27B that we were
            # trying to avoid. Spanning the matmul makes the fill a true
            # intermediate: recomputed in the backward pass, never stored.
            from torch.utils.checkpoint import checkpoint

            if getattr(self, "dense_fill", False):
                def _layer_z(x_, Z):
                    return self._apply_fill(x_, dt)

                return checkpoint(_layer_z, x, self.Z, use_reentrant=False)

            def _layer(x_, A, B):
                return self._apply_fill(x_, dt)

            return checkpoint(_layer, x, self.A, self.B, use_reentrant=False)
        return self._apply_fill(x, dt)


class PreimageFill(nn.Module):
    """A fill whose cell never moves: w = c + R*tanh(z), c and R fixed.

    BoundedFill re-centres on the current position after every task
    (exp_seq.fold_into_anchors), so each task's room is the distance from
    where the last one left off to the nearer wall, and the room decays
    geometrically (Prop. 6). That decay is a property of re-centring, not of
    the cells: the cell [lo, hi] itself never changes, and a weight anywhere
    inside it could in principle reach the whole of it again.

    This module tests exactly that. The centre c = (lo+hi)/2 and radius
    R = (hi-lo)/2 are frozen once, the accumulated preimage z carries the
    history, and a task adds s*BA^T to z rather than moving the anchor.
    |tanh| < 1 keeps w strictly inside the cell forever, so invariance still
    holds by construction and a task is still revoked exactly by subtracting
    its increment.

    One detail decides whether the comparison is honest: the released weight
    is NOT the cell centre (NF4 level 1 sits 14% of a cell width off centre),
    so z must start at atanh((w_hat - c)/R). Starting at z = 0 would serve a
    different model before a single step of training, which would break the
    property the whole paper rests on.
    """

    def __init__(self, base, lo, hi, anchors, rank, tanh_scale, eps=1e-4):
        super().__init__()
        self.base = base
        shape = tuple(anchors.shape)
        out_f, in_f = shape
        dev = anchors.device
        self.A = nn.Parameter(torch.randn(rank, in_f, device=dev)
                              / math.sqrt(in_f))
        self.B = nn.Parameter(torch.zeros(out_f, rank, device=dev))
        # Three per-weight buffers is 16.9 GB in fp32 at 1.7B, which is the
        # whole of a 24 GB card. Centre and radius go to fp16: their rounding
        # is 2^-11 relative, under 0.4% of a cell half-width even in the
        # outermost cells, inside the 1% margin bin_bounds already leaves.
        # z0 is derived from the ROUNDED centre and radius, so at zero fill
        # the served weight is the anchor exactly, rounding and all. The
        # preimage itself stays fp32: it accumulates across folds.
        c = ((lo + hi) * 0.5).view(shape).to(torch.float16)
        r = ((hi - lo) * 0.5).view(shape).clamp_min(1e-7).to(torch.float16)
        u = ((anchors.float() - c.float()) / r.float())
        self.n_clamped = int((u.abs() >= 1 - eps).sum())
        z0 = torch.atanh(u.clamp(-1 + eps, 1 - eps))
        self.register_buffer("center", c)
        self.register_buffer("radius", r)
        self.register_buffer("zacc", z0.to(torch.float32))
        self.shape = shape
        self.numel = out_f * in_f
        self.tanh_scale = tanh_scale
        self.checkpoint_fill = False

    def weight(self, dtype=torch.float32):
        z = self.zacc.to(dtype) + self.tanh_scale * (self.B @ self.A).to(dtype)
        return self.center.to(dtype) + self.radius.to(dtype) * torch.tanh(z)

    def forward(self, x):
        dt = x.dtype if x.dtype != torch.float32 else torch.float32
        if self.checkpoint_fill and self.training:
            from torch.utils.checkpoint import checkpoint

            def _layer(x_, A, B):
                return F.linear(x_, self.weight(dt).to(x_.dtype))

            return checkpoint(_layer, x, self.A, self.B, use_reentrant=False)
        return F.linear(x, self.weight(dt).to(x.dtype))

    @torch.no_grad()
    def fold(self):
        """Absorb this task into the preimage and reset the adapter.

        Returns mean |tanh(z)| after the fold -- the analogue of the quantity
        Prop. 6 predicts the room from, measured on the same scale so the two
        folding rules can be compared directly.
        """
        self.zacc += self.tanh_scale * (self.B @ self.A).float()
        self.B.data.zero_()
        return float(torch.tanh(self.zacc).abs().mean())

    def room(self) -> torch.Tensor:
        """Distance from the current position to the nearer wall.

        Reported so the two folding rules are read on one axis. Under this
        parameterization the *reachable* set is still the whole cell; this
        number shrinking only means the position has moved off centre.
        """
        c, r = self.center.float(), self.radius.float()
        w = c + r * torch.tanh(self.zacc)
        return torch.minimum(w - (c - r), (c + r) - w)


def wrap_model(model, rank, tanh_scale, margin, name_filter=None,
               codebook_m=False):
    """Replace every Linear4bit with a BoundedFill; returns frozen states.

    codebook_m keeps the per-weight bound as a sixteen-entry table plus the
    block scales instead of a dense tensor. It changes what is stored, not
    what is computed: the bound is rebuilt on demand from the codes already
    inside the 4-bit layer. Required at 27B, where the dense form is 48.7 GB.
    """
    import re

    pattern = re.compile(name_filter) if name_filter else None
    for p in model.parameters():
        p.requires_grad_(False)
    frozen = {}
    hw_table = normalized_halfwidth(capped=True, margin=margin)
    layout, n_frozen, n_weights = None, 0, 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            kind = type(child).__name__
            if kind not in ("Linear4bit", "PackedLinear"):
                continue
            full = f"{name}.{child_name}" if name else child_name
            if pattern and not pattern.search(full):
                continue
            if kind == "PackedLinear":
                # uniform grid: the frozen artifact is the packed codes and
                # scales the release shipped; margin 0.05 is the floor that
                # absorbs a bf16 anchor (cellfill.uniform.uniform_halfwidth)
                setattr(module, child_name,
                        BoundedFill(child, None, rank, tanh_scale,
                                    uniform=max(margin, 0.05)))
                n_weights += child.out_features * child.in_features
                frozen[full] = dict(
                    kind="uniform", packed=child.packed.cpu(),
                    scale=child.scale.cpu(),
                    zero_point=(None if child.zero_point is None
                                else child.zero_point.cpu()),
                    shape=child.shape, num_bits=child.num_bits,
                    group=child.group, qmin=child.qmin, qmax=child.qmax)
                continue
            fs = frozen_state_from_linear4bit(child)
            if codebook_m:
                # Calibrate the nibble order against codes recovered the
                # reliable way, once, and re-verify per layer -- the check is
                # free here because the codes are already in hand.
                ref = fs["codes"].reshape(-1)
                if layout is None:
                    layout = nibble_layout_of(child.weight.data.cpu(),
                                              ref.cpu())
                nib = unpack_nf4_codes(child.weight.data, ref.numel(),
                                       layout).to(ref.dtype)
                bad = (nib != ref).nonzero().view(-1).to(torch.int32)
                frac = bad.numel() / ref.numel()
                # Two conditions look alike in this count and are two orders
                # of magnitude apart. A wrong layout disagrees on ~93% of
                # codes (measured: the three losing layouts on Qwen3-8B score
                # 79-93% where the right one scores 0.0000%). A weight whose
                # dequantized anchor lands on the far side of a cell wall
                # from its stored nibble disagrees on well under 1% -- 3e-7
                # of weights at 27B, 5.5e-3 in one Qwen3-8B MLP matrix -- and
                # is handled by freezing it, which is already correct. So
                # when the count is high enough to be suspicious, ask the
                # alternatives rather than trusting a fixed threshold.
                if frac > 0.02:
                    alts = []
                    for cand in LAYOUTS:
                        if cand == layout:
                            continue
                        alt_nib = unpack_nf4_codes(child.weight.data,
                                                   ref.numel(),
                                                   cand).to(ref.dtype)
                        alts.append(float((alt_nib != ref).float().mean()))
                    if frac > min(alts) / 10:
                        raise RuntimeError(
                            f"{full}: {bad.numel()} of {ref.numel()} codes "
                            f"({frac:.3%}) disagree under layout {layout}, "
                            f"and the best alternative disagrees on "
                            f"{min(alts):.3%} -- the layout is wrong, not "
                            f"the anchors")
                    print(f"[wrap] {full}: {frac:.3%} of nibbles disagree "
                          f"with the re-binned anchor (next layout: "
                          f"{min(alts):.1%}); those weights are frozen",
                          flush=True)
                n_frozen += bad.numel()
                cb = (hw_table, fs["absmax"], fs["blocksize"], layout, bad)
                shape_only = torch.empty(fs["shape"], device="meta")
                setattr(module, child_name,
                        BoundedFill(child, shape_only, rank, tanh_scale,
                                    margin=margin, codebook=cb))
            else:
                lo, hi = bin_bounds(fs["codes"], fs["absmax"], fs["blocksize"],
                                    capped=True, margin=margin)
                anch = fs["anchors"].reshape(-1)
                halfwidth = torch.minimum(anch - lo, hi - anch).view(
                    fs["shape"])
                setattr(module, child_name,
                        BoundedFill(child, halfwidth, rank, tanh_scale))
            n_weights += fs["anchors"].numel()
            if codebook_m:
                # No anchor tensor on the host. At 27B the fp16 anchors are
                # 48.7 GB of RAM, and the dense merged map materialize builds
                # is another 48.7 GB; the two together reached 108 GB and
                # the kernel killed a finished 24-epoch run at its merge
                # step. The anchors are a function of codes and scales
                # (cellfill.nf4.dequantize_ref) and are rebuilt per layer on
                # the GPU when needed. The shape is all that is kept.
                frozen[full] = (fs["codes"].cpu(), fs["absmax"].cpu(),
                                fs["blocksize"], tuple(fs["shape"]))
            else:
                # anchors stored fp16: halves host RAM; the rounding is far
                # below the bin margin and materialize() recomputes bounds
                # from codes.
                frozen[full] = (fs["codes"].cpu(), fs["absmax"].cpu(),
                                fs["blocksize"], fs["anchors"].half().cpu())
    if codebook_m:
        print(f"[wrap] bound rebuilt from codes ({layout}); "
              f"{n_frozen} of {n_weights} weights frozen at their anchor "
              f"({n_frozen / max(n_weights, 1):.2e})", flush=True)
    return frozen


def materialize(model, frozen, margin, map_dtype=torch.float32,
                store_anchors=True, stats=None, skip_dense=False):
    """Exact fp32 merge + structural-invariance verification + saturation.

    stats, if a dict, receives per-module fill statistics: mean |t|, the
    saturated fraction, and the fill's share of the weight's energy. That is
    the map of where in the network an injection landed.
    """
    merged, anchors_map = {}, {}
    n_bad = n_sat = n_tot = 0
    with torch.no_grad():
        for name, mod in model.named_modules():
            if not isinstance(mod, BoundedFill):
                continue
            dev = mod.A.device
            if isinstance(frozen[name], dict):      # uniform grid
                from cellfill.uniform import (
                    check_invariance_uniform, uniform_anchors,
                    uniform_halfwidth, unpack_codes,
                )

                f = frozen[name]
                codes = unpack_codes(f["packed"].to(dev), f["num_bits"],
                                     f["shape"])
                zp = None if f["zero_point"] is None else \
                    f["zero_point"].to(dev)
                anch = uniform_anchors(codes, f["scale"].to(dev), f["group"],
                                       zp)
                m_exact = uniform_halfwidth(f["scale"].to(dev), f["group"],
                                            mod.umargin)
                t = mod.fill_frac * mod.t_value(torch.float32)
                w_new = anch + m_exact * t
                ok, n_mis, _ = check_invariance_uniform(
                    w_new, codes, f["scale"].to(dev), f["group"], f["qmin"],
                    f["qmax"], zp)
                n_bad += n_mis
                n_sat += int((t.abs() > 0.99).sum().item())
                n_tot += t.numel()
                if stats is not None:
                    d = m_exact * t
                    stats[name] = dict(
                        mean_abs_t=float(t.abs().mean()),
                        sat=float((t.abs() > 0.99).float().mean()),
                        fill_energy=float((d * d).sum()),
                        weight_energy=float((anch * anch).sum()),
                        numel=int(t.numel()))
                if not skip_dense:
                    merged[name] = w_new.to(map_dtype).cpu()
                if store_anchors:
                    anchors_map[name] = anch.to(map_dtype).cpu()
                continue
            codes, absmax, bsz, anchors = frozen[name]
            # A is present on both paths; M is not registered when the bound
            # is rebuilt from the codebook, and this line cost a 23-epoch 27B
            # run its merge step.
            codes_d, absmax_d = codes.to(dev), absmax.to(dev)
            lo, hi = bin_bounds(codes_d, absmax_d, bsz, capped=True,
                                margin=margin)
            if isinstance(anchors, tuple):          # codebook path: shape only
                shape = anchors
                anchors = dequantize_ref(codes_d, absmax_d, bsz,
                                         shape=shape)
            anch = anchors.to(dev).reshape(-1).float()
            m_exact = torch.minimum(anch - lo, hi - anch)
            t = mod.fill_frac * mod.t_value(torch.float32)
            w_new = anch + m_exact * t.reshape(-1)
            ok, n_mis, _ = check_invariance(w_new, codes_d, absmax_d, bsz)
            n_bad += n_mis
            n_sat += int((t.abs() > 0.99).sum().item())
            n_tot += t.numel()
            if stats is not None:
                d = (m_exact * t.reshape(-1))
                stats[name] = dict(
                    mean_abs_t=float(t.abs().mean()),
                    sat=float((t.abs() > 0.99).float().mean()),
                    fill_energy=float((d * d).sum()),
                    weight_energy=float((anch * anch).sum()),
                    numel=int(t.numel()))
            if not skip_dense:
                merged[name] = w_new.reshape(anchors.shape).to(map_dtype).cpu()
            if store_anchors:
                anchors_map[name] = anchors.to(map_dtype)
    if n_bad:
        raise RuntimeError(f"structural invariance violated on {n_bad} weights")
    return merged, anchors_map, n_sat / n_tot


def _probe_counts(pairs):
    """Probes per domain, and the marginal-guess floor implied by how many
    distinct answers that domain has. The floors differ by two orders of
    magnitude across this corpus, so a recall number cannot be read without
    the one that belongs to it."""
    import collections

    n = collections.Counter()
    answers = collections.defaultdict(set)
    for prompt, expect, kind in pairs:
        n[kind] += 1
        answers[kind].add(expect)
    return {k: dict(probes=n[k], distinct_answers=len(answers[k]),
                    floor=1.0 / len(answers[k])) for k in n}


@torch.no_grad()
def _dump_probes(model, tok, pairs, n):
    """What the model actually emits, per kind.

    A kind at 0% exact match may be answering nothing or may be one token
    short, and the two mean different things. We sample within each kind so a
    large kind cannot crowd out the others.
    """
    import collections
    import random as _prng

    by_kind = collections.defaultdict(list)
    for pr in pairs:
        by_kind[pr[2] if len(pr) > 2 else "all"].append(pr)
    rng = _prng.Random(0)
    model.eval()
    tok.padding_side = "left"
    for kind, rows in sorted(by_kind.items()):
        print(f"[dump] --- {kind} ---", flush=True)
        for prompt, expect, _ in rng.sample(rows, min(n, len(rows))):
            enc = tok(prompt, return_tensors="pt").to(model.device)
            out = model.generate(**enc, max_new_tokens=24, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            gen = tok.decode(out[0, enc.input_ids.shape[1]:],
                             skip_special_tokens=True)
            print(f"[dump]   want {expect!r}", flush=True)
            print(f"[dump]   got  {gen.strip()!r}", flush=True)
    tok.padding_side = "right"


def _kl_pool(args):
    """Texts the inert term distils on.

    Default is neutral WikiText, which is what keeps a fill from moving the
    function on everything the corpus does not mention. It does not keep the
    fill from moving the function on knowledge the corpus is adjacent to --
    a fill trained on one library's missing symbols damaged that library's
    already-known symbols by 20.8 points with this term active, because
    neutral text never exercises them. --kl-pool-file redirects the pool at
    those symbols' own prompts.
    """
    neutral = wikitext_train_snippets(4000, seed=args.seed + 7)
    if not args.kl_pool_file:
        return neutral
    recs = json.loads(Path(args.kl_pool_file).read_text())
    pool = []
    for r in recs:
        if isinstance(r, str):
            pool.append(r)
        elif "prompt" in r:
            # the prompt alone, not prompt+answer: the term should hold the
            # model's own continuation still, not teach it a target.
            pool.append(r["prompt"])
        elif "text" in r:
            pool.append(r["text"])
    if args.kl_pool_mode == "add":
        # train() sweeps the pool sequentially, k0 = step*kl_bs mod len(pool),
        # so texts appended after the neutral ones are only reached if the run
        # is long enough to wrap. It usually is not: 432 steps at kl_bs 8
        # consumes 3,456 positions of a 4,024-entry pool, and the appended
        # entries are never drawn. The first version of this option did
        # exactly that and produced an arm identical to its control in every
        # group, which is what exposed it. Interleave to a stated share
        # instead, so the added texts are actually sampled.
        neutral = list(neutral)
        share = 0.25
        reps = max(1, int(share * len(neutral) / max(1, len(pool)) ))
        added = (pool * reps)[:int(share * len(neutral))]
        every = max(1, len(neutral) // max(1, len(added)))
        out, ai = [], 0
        for i, t in enumerate(neutral):
            out.append(t)
            if i % every == 0 and ai < len(added):
                out.append(added[ai]); ai += 1
        print(f"[inert] pool = {len(neutral)} neutral with {ai} anchor texts "
              f"interleaved ({ai / len(out):.0%} of draws), from "
              f"{args.kl_pool_file}", flush=True)
        return out
    print(f"[inert] pool from {args.kl_pool_file}: {len(pool)} texts "
          f"(replacing the neutral pool)", flush=True)
    return pool


def main():
    args = get_args()
    assert torch.cuda.is_available()
    torch.manual_seed(args.seed)
    t0 = time.time()

    if args.facts_file:
        # See exp0_clip_rate for why the bit accounting does not carry over to
        # a real corpus.
        train_path = Path(args.facts_file)
        probe_path = Path(args.probes_file or
                          str(train_path).replace("_train", "_probes"))
        texts = [r["text"] for r in json.loads(train_path.read_text())]
        pairs = [(r["prompt"], r["answer"], r.get("domain", "real"))
                 for r in json.loads(probe_path.read_text())]
        args.n_facts = len(texts)
        print(f"[data] real corpus: {len(texts)} facts, {len(pairs)} probes "
              f"from {train_path.name}")
    else:
        facts = generate(args.n_facts, seed=args.seed)
        texts = training_texts(facts, paraphrases=args.paraphrases)
        pairs = probe_pairs(facts)
    if args.probe_cap and len(pairs) > args.probe_cap:
        import random as _prng

        pairs = _prng.Random(args.seed).sample(pairs, args.probe_cap)
    replay_pool, n_rep = None, 0
    if args.replay_frac > 0:
        n_rep = int(len(texts) * args.replay_frac / (1 - args.replay_frac))
        replay_pool = replay_snippets(n_rep * args.epochs, seed=args.seed,
                                      source=args.replay_source)
        print(f"[data] replay {n_rep}/epoch, pool={len(replay_pool)} "
              f"({args.replay_source})")

    print(f"[load] {args.model} in NF4")
    model, tok = build_4bit(args.model)
    ppl_txt = wikitext_text()
    x_txt = lambada_text()

    print("[wrap] BoundedFill on every Linear4bit")
    frozen = wrap_model(model, args.rank, args.tanh_scale, args.margin,
                        name_filter=args.target_filter,
                        codebook_m=args.codebook_m)
    if args.checkpoint_fill:
        for m in model.modules():
            if isinstance(m, BoundedFill):
                m.checkpoint_fill = True
        print("[wrap] fill recomputation enabled (memory for step time)")
    if args.link != "tanh":
        for m in model.modules():
            if isinstance(m, BoundedFill):
                m.link_fn = args.link
        print(f"[wrap] bounded link = {args.link}")
    if args.fill_frac != 1.0:
        for m in model.modules():
            if isinstance(m, BoundedFill):
                m.fill_frac = args.fill_frac
        print(f"[wrap] fill_frac = {args.fill_frac}")
    if args.stack > 1:
        for m in model.modules():
            if isinstance(m, BoundedFill):
                m.add_stack(args.stack, args.stack_mode)
        print(f"[wrap] residual stack depth {args.stack} ({args.stack_mode})")
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[wrap] {len(frozen)} layers, trainable params {n_train:,}")

    # The generation budget is a function of the probe set, so it is fixed
    # before training and reused by the trajectory callback and the final
    # scoring alike.
    longest = max(len(tok(" " + e, add_special_tokens=False).input_ids)
                  for _, e, _ in pairs)
    max_new = max(32, longest + 4)

    trajectory = []
    liveness = []

    @torch.no_grad()
    def _liveness(ep):
        """How much of the fill can still learn: the share of coordinates
        past |t| = 0.99 and the mean link derivative 1 - t^2 (the factor
        every coordinate's gradient carries into A and B). Once an epoch;
        one pass over the factors, no evaluation."""
        n, sat, g = 0, 0.0, 0.0
        for m in model.modules():
            if isinstance(m, BoundedFill):
                t = m.t_value(torch.float32)
                sat += float((t.abs() > 0.99).float().sum())
                g += float((1 - t * t).sum())
                n += t.numel()
        row = dict(epoch=ep + 1, saturation=sat / max(n, 1),
                   mean_link_grad=g / max(n, 1))
        liveness.append(row)
        print(f"[live] ep{ep + 1:3d} saturation={row['saturation']:.2%} "
              f"mean(1-t^2)={row['mean_link_grad']:.4f}", flush=True)

    def _point(ep, losses_so_far):
        _liveness(ep)
        if not args.eval_every or (ep + 1) % args.eval_every:
            return
        model.eval()
        sat = [float((m.t_value(torch.float32).abs() > 0.99).float().mean())
               for m in model.modules() if isinstance(m, BoundedFill)]
        row = dict(epoch=ep + 1,
                   loss=sum(losses_so_far[-25:]) / max(len(losses_so_far[-25:]), 1),
                   recall=eval_recall(model, tok, pairs, max_new=max_new),
                   ppl=eval_ppl(model, tok, ppl_txt, max_chunks=20),
                   ppl_lambada=maybe_ppl(model, tok, x_txt, 20),
                   saturation=sum(sat) / max(len(sat), 1))
        trajectory.append(row)
        print(f"[traj] ep{row['epoch']:3d} recall={row['recall']:.3f} "
              f"ppl={row['ppl']:.3f} lambada={row['ppl_lambada']} "
              f"sat={row['saturation']:.2%}", flush=True)
        model.train()

    print(f"[train] {args.epochs} epochs, lr={args.lr}, s={args.tanh_scale}")
    losses = train(model, tok, texts, args.epochs, args.lr, args.bs, args.seed,
                   replay_pool=replay_pool, n_replay_per_epoch=n_rep,
                   ckpt_path=args.ckpt, on_epoch_end=_point,
                   kl_pool=(_kl_pool(args) if args.kl_weight > 0 else None),
                   kl_weight=args.kl_weight, kl_bs=args.kl_bs)

    # max_new is generous and never below the longest answer present. Scoring
    # is monotone in it (eval_recall), so the only thing a tight budget buys
    # is the chance to score a truncation as a miss -- which is exactly what
    # it did to the Powerball domain at 16.
    print(f"[eval] longest answer {longest} tokens -> max_new={max_new}",
          flush=True)
    recall_trained = eval_recall(model, tok, pairs, max_new=max_new)
    if args.facts_file:
        _, by_kind = eval_recall(model, tok, pairs, max_new=max_new,
                                 detail=True)
        if args.dump_probes:
            _dump_probes(model, tok, pairs, args.dump_probes)
        print("[probe] by kind: " + "  ".join(
            f"{k}={v:.1%}" for k, v in sorted(by_kind.items())),
            flush=True)
    ppl_trained = eval_ppl(model, tok, ppl_txt, max_chunks=args.max_ppl_chunks)
    ppl_trained_x = maybe_ppl(model, tok, x_txt, args.max_ppl_chunks)
    print(f"[trained] recall={recall_trained:.4f}  ppl={ppl_trained:.3f}  "
          f"lambada={ppl_trained_x}", flush=True)

    # The factors are written BEFORE the invariance check: an eleven-hour
    # 27B run trained to 89.7% and then raised on 6,978 weights in the
    # check (the signed-scale bug, since fixed), and its fill was lost with
    # it. A saved fill is evidence either way; the check still decides
    # whether the run counts.
    if args.save_fill:
        sp = Path(args.save_fill)
        sp.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(
            meta=dict(model=args.model, rank=args.rank,
                      tanh_scale=args.tanh_scale, margin=args.margin,
                      codebook_m=args.codebook_m,
                      target_filter=args.target_filter,
                      fill_frac=args.fill_frac, link=args.link,
                      stack=args.stack, stack_mode=args.stack_mode),
            fills={n: (m.A.detach().cpu(), m.B.detach().cpu())
                   for n, m in model.named_modules()
                   if isinstance(m, BoundedFill)},
            stacks={n: [q.detach().cpu() for q in m.stack]
                    for n, m in model.named_modules()
                    if isinstance(m, BoundedFill) and len(m.stack)}), sp)
        print(f"[save] fill factors -> {sp}")

    print("[materialize] exact fp32 merge + invariance check", flush=True)
    stats = {} if args.fill_stats else None
    merged, anchors_map, saturation = materialize(
        model, frozen, args.margin,
        map_dtype=getattr(torch, args.map_dtype),
        store_anchors=not args.skip_anchor_eval and not args.inplace_only,
        stats=stats, skip_dense=args.inplace_only)
    if stats is not None:
        sp = Path(args.fill_stats)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(stats, indent=1))
        print(f"[stats] per-module fill statistics -> {sp}")
    print(f"[materialize] saturation(|tanh|>0.99)={saturation:.4%}  "
          f"invariance=100%")

    if args.save_merged_weights:
        sp = Path(args.save_merged_weights)
        sp.parent.mkdir(parents=True, exist_ok=True)
        torch.save({n: w.half().cpu() for n, w in merged.items()}, sp)
        print(f"[save] merged weights -> {sp}")
    if args.save_anchor_weights and anchors_map:
        sp = Path(args.save_anchor_weights)
        sp.parent.mkdir(parents=True, exist_ok=True)
        torch.save({n: w.half().cpu() for n, w in anchors_map.items()}, sp)
        print(f"[save] anchor weights -> {sp}")

    if args.inplace_only:
        # the in-place model is the served function; the dense copy would
        # only have re-measured it through another kernel
        _, by_kind_ip = eval_recall(model, tok, pairs, max_new=max_new,
                                    detail=True)
        fp32 = dict(recall_merged=recall_trained,
                    recall_merged_by_kind=by_kind_ip,
                    recall_anchor_by_kind=None, recall_original_fp32=None,
                    recall_anchor_fp32=None, ppl_original_fp32=None,
                    ppl_anchor_fp32=None, ppl_merged=ppl_trained,
                    ppl_x_original_fp32=None, ppl_x_anchor_fp32=None,
                    ppl_x_merged=ppl_trained_x)
        del model
        torch.cuda.empty_cache()
    else:
        del model
        torch.cuda.empty_cache()
        fp32 = eval_fp32_variants(args.eval_base or args.model,
                                  anchors_map, merged, tok, pairs,
                                  ppl_txt, args.max_ppl_chunks,
                                  xdom_txt=x_txt,
                                  dtype=getattr(torch, args.eval_dtype),
                                  max_new=max_new)
        print(f"[merged-fp32] recall={fp32['recall_merged']:.4f}  "
              f"ppl={fp32['ppl_merged']:.3f}")

    result = dict(
        config=vars(args),
        scorer=scorer_stamp(),
        bits_per_fact=None if args.facts_file else bits_per_fact(),
        loss_first=losses[0] if losses else None,
        loss_last=sum(losses[-10:]) / 10 if losses else None,
        probe_counts=(_probe_counts(pairs) if args.facts_file else None),
        trajectory=trajectory or None,
        liveness=liveness or None,
        recall=dict(trained_inplace=recall_trained,
                    merged_by_kind=fp32.get("recall_merged_by_kind"),
                    anchor_by_kind=fp32.get("recall_anchor_by_kind"),
                    original_fp32=fp32["recall_original_fp32"],
                    anchor_fp32=fp32["recall_anchor_fp32"],
                    merged=fp32["recall_merged"]),
        ppl=dict(trained_inplace=ppl_trained,
                 original_fp32=fp32["ppl_original_fp32"],
                 anchor_fp32=fp32["ppl_anchor_fp32"],
                 merged=fp32["ppl_merged"]),
        ppl_lambada=dict(trained_inplace=ppl_trained_x,
                         original_fp32=fp32["ppl_x_original_fp32"],
                         anchor_fp32=fp32["ppl_x_anchor_fp32"],
                         merged=fp32["ppl_x_merged"]),
        merge=dict(n_layers=len(frozen), saturation=saturation,
                   clipped_frac=None,
                   note="invariance by construction (bounded tanh fill)"),
        heal=None,
        minutes=round((time.time() - t0) / 60, 1),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"[done] {out}  ({result['minutes']} min)")


if __name__ == "__main__":
    main()

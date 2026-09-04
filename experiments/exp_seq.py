#!/usr/bin/env python
"""exp_seq: the actual continual-learning loop (T1 -> T2 -> ... -> Tm).

Every other experiment in this repo injects one batch of knowledge once.
This one runs the lifecycle the paper claims: disjoint fact sets arrive in
sequence, each is absorbed into the SAME residual, and after every task we
measure (a) the new task, (b) every earlier task (cumulative forgetting),
(c) general ability, and (d) that the 4-bit artifact is still bit-identical.

Two modes:
  --mode minor   all tasks share one frozen artifact; the residual accumulates
                 (this is the "minor version" regime: the shipped 4-bit file
                 never changes across the entire sequence)
  --mode consolidate
                 after --consolidate-after tasks, re-quantize the current
                 weights into a NEW frozen artifact (an explicit major
                 version), zero the residual, and continue. Tests whether
                 capacity is renewable across generations.

Uses QIL-LoRA (structural invariance, r64 champion recipe) so that no
clipping confound enters the sequential accounting.

Run:
  .venv/bin/python experiments/exp_seq.py --tasks 4 --facts-per-task 500
"""

from __future__ import annotations

import argparse
import os
import random
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cellfill import bin_bounds, check_invariance  # noqa: E402
from cellfill.bins import storable_mask  # noqa: E402
from cellfill.nf4 import assign_codes, compute_absmax, dequantize_ref  # noqa: E402
from cellfill.bnb_state import frozen_state_from_linear4bit  # noqa: E402
from experiments.exp0_clip_rate import (
    scorer_stamp,  # noqa: E402
    build_4bit,
    eval_ppl,
    eval_recall,
    lambada_text,
    maybe_ppl,
    train,
    wikitext_text,
    wikitext_train_snippets,
)
from experiments.exp5_qil import BoundedFill, PreimageFill  # noqa: E402
from experiments.synth_facts import (  # noqa: E402
    bits_per_fact,
    generate,
    training_texts,
)
from experiments.corpora import (  # noqa: E402
    associate,
    item_keys,
    item_probes,
    item_texts,
    load_task,
    probes_path_for,
    sample as sample_facts,
)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--tasks", type=int, default=4)
    p.add_argument("--facts-per-task", type=int, default=500)
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--tanh-scale", type=float, default=40.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--margin", type=float, default=0.01)
    p.add_argument("--replay-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", choices=["minor", "consolidate"], default="minor")
    p.add_argument("--fold", choices=["anchor", "preimage"], default="anchor",
                   help="anchor: re-centre on the current position after each "
                        "task, so the next task's room is the distance to the "
                        "nearer wall (Prop. 6, geometric decay). preimage: "
                        "keep the cell's centre and radius fixed forever and "
                        "accumulate in the tanh preimage, which leaves the "
                        "whole cell reachable at every task. Both are "
                        "invariant by construction; the pair separates decay "
                        "that is geometry from decay that is saturation and "
                        "interference.")
    p.add_argument("--inert", type=float, default=0.0,
                   help="weight of KL(p_served || p_fill) on a neutral pool "
                        "during each task: the new fill may change the "
                        "served model on its own facts and nowhere else, "
                        "which is what protects the earlier tasks "
                        "(learning-without-forgetting inside the cells)")
    p.add_argument("--inert-pool", choices=["wikitext", "neutral"],
                   default="neutral")
    p.add_argument("--inert-bs", type=int, default=8)
    p.add_argument("--tasks-from", default=None,
                   help="comma-separated corpus files, one per task, instead "
                        "of slices of one generator call. Every sequence in "
                        "this paper without it has zero distribution shift "
                        "between tasks; with it the tasks differ in domain, "
                        "relation and answer shape.")
    p.add_argument("--anchor-source", choices=["artifact", "original"],
                   default="artifact",
                   help="what the fill's additive base and room are measured "
                        "FROM. artifact: the dequantized anchors (default, "
                        "train=serve on the shipped file). original: the "
                        "dense vendor fp16 named by --master-checkpoint, "
                        "UNDER THE LOCKED A2 SPEC -- the vendor walls and "
                        "codes are not touched (no re-quantization); only "
                        "the learning center moves to W0, the room is "
                        "min(W0-lo, hi-W0) against the EXISTING walls, and "
                        "the ~2%% of weights whose W0 sits outside the "
                        "margin band start frozen (clamp to zero room). "
                        "Bit-identity of the released file holds by "
                        "construction; measured on 1.28M weights: full-range "
                        "fills, zero code changes.")
    p.add_argument("--dense-fill", action="store_true",
                   help="parameterize the fill as M*tanh(s*Z) with a DENSE "
                        "zero-init Z instead of the low-rank BA^T: same "
                        "bound, no rank bottleneck. A rank-r fill writes "
                        "every task through r shared directions; dense lets "
                        "each weight move independently -- the interference "
                        "hypothesis the paired sequence arms test. Optimizer "
                        "state is 16 bytes per constrained weight, so this "
                        "is a 1.7B/8B instrument (with --target-filter mlp); "
                        "it does not fit at 27B and above.")
    p.add_argument("--consolidate-master", choices=["anchor0", "original"],
                   default=None,
                   help="burn from a continuous fp32 master instead of from "
                        "the stored anchors. Iterative burning Q(anchor+f) "
                        "feeds each version's rounding back into the next "
                        "(compounding); the master arm burns Q(master+SumF), "
                        "a single projection, so the anchor error stays one "
                        "quantization step for any number of versions "
                        "(rho_rounding = 0; classic master-weights / "
                        "error-feedback). anchor0: master starts at the "
                        "version-0 anchors. original: master starts at the "
                        "dense checkpoint named by --master-checkpoint -- "
                        "the file the vendor would have shipped had the "
                        "knowledge been present at quantization time. The "
                        "arms are identical at burn 1 by construction (free "
                        "instrument self-check); the discriminating signal "
                        "starts at burn 2.")
    p.add_argument("--master-checkpoint", default=None,
                   help="dense fp16 checkpoint (HF id or dir) for "
                        "--consolidate-master original")
    p.add_argument("--consolidate-trust", type=float, default=None,
                   help="per-block trust region on the scale at a major "
                        "version: the new absmax is clamped to within a "
                        "factor (1 +- d) of the OLD block scale. The grid "
                        "re-snap that prices every burn at -0.8 regardless "
                        "of how many codes change is O(|level| * d_absmax) "
                        "on every weight; this caps it by construction, and "
                        "caps the ten-burn cumulative drift at (1+d)^k. "
                        "d -> 0 is the known zero-cost/zero-recovery "
                        "degenerate corner; d from the measured p99 of "
                        "benign per-burn drift is the intended operating "
                        "point. The cost it buys is a clip rate on extremes, "
                        "reported per burn.")
    p.add_argument("--calibrated-consolidate", action="store_true",
                   help="search the block scale at a major version instead of "
                        "pinning it to max|w|. The archived consolidation cost "
                        "is -0.80 +- 0.24 suite points and does not scale with "
                        "how much of the file is rewritten, which is the "
                        "signature of a quantizer error rather than of "
                        "knowledge damage.")
    p.add_argument("--probes-from", default=None,
                   help="comma-separated probe files, one per --tasks-from "
                        "file, overriding the X_train.json -> X_probes.json "
                        "convention. A real task cannot manufacture its "
                        "probes the way the generator can, so it must be "
                        "given them.")
    p.add_argument("--rehearse-holdout", type=float, default=1.0,
                   help="fraction of each earlier task's facts that MAY be "
                        "rehearsed; the rest never are. The default of 1.0 "
                        "rehearses everything, which is what the archived "
                        "runs did and which leaves no uncontaminated measure "
                        "of forgetting. At 0.5 the two halves are scored "
                        "separately and the held-out half is the measure.")
    p.add_argument("--rehearse-old", type=float, default=0.0,
                   help="each task's epochs also replay this fraction (of "
                        "the task's own size) of sentences drawn fresh from "
                        "the earlier tasks' corpora: rehearsal of old facts, "
                        "the continual-learning baseline the folding law is "
                        "read against")
    p.add_argument("--consolidate-after", type=int, default=2)
    p.add_argument("--consolidate-final", action="store_true",
                   help="run the trailing consolidation after the LAST task "
                        "too. The default skips it (a burn's in-loop purpose "
                        "is to prepare the next task's base), which is why "
                        "the archived ten-burn s0 delivered nine burns; the "
                        "ten-burn design wants ten, so its s1 sets this.")
    p.add_argument("--max-ppl-chunks", type=int, default=30)
    p.add_argument("--unconstrained", action="store_true",
                   help="the control for the whole thesis: identical rank, "
                        "schedule, rehearsal and folds, with the cell bound "
                        "removed, so the served weight is the anchor plus a "
                        "raw low-rank product. The invariance count stops "
                        "being an assertion and becomes a measurement of how "
                        "far an unconstrained update travels, and the "
                        "retention columns say whether the bound or the "
                        "rehearsal is what keeps the earlier tasks.")
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="recompute each decoder block's activations in the "
                        "backward pass instead of storing them. Distinct from "
                        "--checkpoint-fill, which does the same for the fill. "
                        "Numerically transparent -- the gradients are the "
                        "ones the stored pass would give, up to the "
                        "non-determinism the harness already has -- and it "
                        "costs about a third more wall clock. Off by default "
                        "so archived runs are unchanged; at 27B and 32B the "
                        "stored activations are ~6 GB and the run does not "
                        "fit without it.")
    p.add_argument("--target-filter", default=None,
                   help="restrict the constrained set to layers whose name "
                        "contains this substring, e.g. 'mlp'. wrap_fresh has "
                        "always honoured it; it had no flag. At 32B the MLPs "
                        "are 80.6%% of the constrained weights and their "
                        "anchors alone fit on one card, which is what lets a "
                        "sequence run there without sharding.")
    p.add_argument("--wall-codebook", action="store_true",
                   help="keep the cell EDGES as two sixteen-entry tables plus "
                        "the block scales and rebuild the room from the current "
                        "anchor at every use, instead of storing it densely. "
                        "The edges are functions of the frozen codes and scales "
                        "and no fold moves them, so unlike --codebook-m this "
                        "survives folding. Needed above about 8B: the dense "
                        "room is 45 GB at 27B, which with the dense anchors "
                        "does not fit on one 80 GB card")
    p.add_argument("--checkpoint-fill", action="store_true",
                   help="recompute the fill in the backward pass (memory for "
                        "step time; needed on 24 GB cards)")
    p.add_argument("--recover-epochs", type=int, default=0,
                   help="after a consolidation fires, replay the earlier "
                        "tasks' own material for this many epochs before the "
                        "version is judged. A major version re-quantizes, "
                        "which injects fresh error into everything already "
                        "written; this is the pass that repairs it, and "
                        "without it the gate judges a version nobody tried to "
                        "recover")
    p.add_argument("--recover-lr", type=float, default=2e-4)
    p.add_argument("--validate", default="",
                   help="comma-separated fold indices at which to run the "
                        "multiple-choice suite on the live served model, and "
                        "again immediately after any consolidation that fires "
                        "at one of them. This is the stage the cycle otherwise "
                        "measures outside itself: without it a sequence reports "
                        "perplexity and the invariance check and nothing a task "
                        "metric would see. Costs a few minutes per point")
    p.add_argument("--validate-tasks",
                   default="arc_easy,arc_challenge,hellaswag,winogrande")
    p.add_argument("--validate-samples", default=None,
                   help="directory for per-item suite outcomes, one file per "
                        "evaluation point. Two points in a sequence are "
                        "scored on the same items, so keeping them allows a "
                        "paired comparison later; without them only the "
                        "marginal accuracies survive, and the independent "
                        "binomial they force is the wrong test for a claim "
                        "that something did not change.")
    p.add_argument("--validate-bs", type=int, default=8,
                   help="batch size for the in-sequence lm-eval suite. It "
                        "carries no scientific content -- loglikelihoods are "
                        "per-request -- but it sets the evaluation's peak, "
                        "which at 32B is the narrowest moment of the run.")
    p.add_argument("--validate-limit", type=int, default=None,
                   help="items per task; None runs the full suite")
    p.add_argument("--fold-diag-sample", type=float, default=1.0,
                   help="fraction of coordinates the interference diagnostic "
                        "remembers per layer (see _diag_index). 1.0 keeps "
                        "every one, which is what every 1.7B run used and "
                        "costs 5.2 GiB of host memory; at 27B that is 90.7 "
                        "GiB and at 32B 115.5 GiB, so a large sequence wants "
                        "0.03125 or less. Only the two cosines and the sign "
                        "conflicts are sampled; the histogram and the "
                        "saturated fraction stay exact.")
    p.add_argument("--fold-diagnostics", action="store_true",
                   help="at every fold, record the distribution of the "
                        "displacement rather than only its mean, and how much "
                        "of it lands on the coordinates the previous task "
                        "used. E|t| is a mean and hides both. Costs one fp16 "
                        "copy of the wrapped weights (about 2.8 GB at 1.7B)")
    p.add_argument("--save-merged", default=None,
                   help="after the last task, write the served weights of "
                        "every wrapped matrix so the final minor version can "
                        "be scored by experiments/eval_downstream.py --merged. "
                        "Without this the sequence reports perplexity and the "
                        "invariance check and nothing a task metric would see")
    p.add_argument("--out", default="out/exp_seq.json")
    return p.parse_args()


def task_facts(args):
    """Disjoint fact sets, by construction.

    Distinct seeds are NOT enough: names are drawn from ~90k combinations, so
    500 names x 4 tasks collide by the birthday bound (~22 expected). Generate
    the whole sequence in one call (which dedups names) and slice it.

    With --tasks-from the tasks are real corpora instead, one file per task.
    That is a different experiment and it is worth being explicit about why:
    slices of one generator call differ in their entities and in nothing else,
    so a sequence over them measures repeated writing rather than adaptation
    across domains. Files whose facts are drug approvals, library signatures,
    lottery draws and tool surfaces differ in domain, relation and answer
    shape, which is what the word ``lifelong'' is usually taken to mean.
    """
    if getattr(args, "tasks_from", None):
        paths = [x.strip() for x in args.tasks_from.split(",") if x.strip()]
        assert len(paths) == args.tasks, (
            f"--tasks-from lists {len(paths)} files for {args.tasks} tasks")
        pp = [x.strip() for x in (args.probes_from or "").split(",") if x.strip()]
        assert not pp or len(pp) == len(paths), (
            f"--probes-from lists {len(pp)} files for {len(paths)} tasks")
        loaded, pmaps = [], []
        for i, path in enumerate(paths):
            facts, probes = load_task(path, pp[i] if pp else None)
            got = sample_facts(facts, args.facts_per_task, args.seed * 31 + i)
            pmaps.append(associate(got, probes, where=Path(path).name))
            loaded.append(got)
            sents = sum(len(g.texts) for g in got)
            print(f"[task {i}] {Path(path).name}: {len(facts)} facts, using "
                  f"{len(got)} ({sents} sentences, "
                  f"{sents / max(len(got), 1):.1f} per fact; probes from "
                  f"{Path(pp[i]).name if pp else probes_path_for(path).name})",
                  flush=True)
        short = [i for i, f in enumerate(loaded) if len(f) < args.facts_per_task]
        assert not short, (
            f"tasks {short} have fewer than {args.facts_per_task} facts "
            f"(they have {[len(loaded[i]) for i in short]}); lower "
            "--facts-per-task so every task carries the same dose")
        return loaded, pmaps
    pool = generate(args.tasks * args.facts_per_task, seed=args.seed)
    n = args.facts_per_task
    return ([pool[t * n:(t + 1) * n] for t in range(args.tasks)],
            [None] * args.tasks)


def rehearsal_split(facts, frac, seed, task_idx):
    """Which of a task's facts may be rehearsed later, and which never are.

    The split is of the REHEARSAL pool, not of training: the task's own
    epochs see every fact. What the held-out half measures afterwards is
    therefore retention of material that was learned and then never shown
    again, which is the only forgetting number in this project that the
    rehearsal corpus cannot flatter. Deterministic in the seed so the two
    halves are the same at every fold.
    """
    idx = list(range(len(facts)))
    random.Random(seed * 101 + task_idx).shuffle(idx)
    k = int(len(facts) * frac)
    keep = sorted(idx[:k])
    hold = sorted(idx[k:])
    return [facts[i] for i in keep], [facts[i] for i in hold]


def _reattach_hook(new_module, hook):
    """Give a replacement module the accelerate hook its predecessor had.

    Only the execution device is carried over, and a fresh hook is built for
    it rather than the old object being moved: AlignDevicesHook.init_hook
    walks the module's parameters and would try to place anchors that are
    already where they belong. A model loaded onto one card has no hook and
    nothing happens.
    """
    if hook is None:
        return
    dev = getattr(hook, "execution_device", None)
    if dev is None:
        return
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module

    add_hook_to_module(new_module,
                       AlignDevicesHook(execution_device=dev,
                                        io_same_device=False,
                                        place_submodules=False))


# BoundedFill registers its anchors at this precision; the writable cell is
# defined against it (cellfill.bins.anchor_absmax_floor).
ANCHOR_STORE_DTYPE = torch.float16


def _load_dense_master(ckpt: str) -> dict:
    """{param name minus '.weight': fp32 cpu tensor} from a dense checkpoint.

    Shared by --consolidate-master original and --anchor-source original.
    fp32 on host; the callers decide device and lifetime.
    """
    import glob as _glob

    from safetensors import safe_open

    pats = [os.path.expanduser(
        "~/.cache/huggingface/hub/models--" + ckpt.replace("/", "--")
        + "/snapshots/*"), ckpt]
    snap = next(p for pat in pats for p in sorted(_glob.glob(pat))
                if os.path.isdir(p))
    idx = os.path.join(snap, "model.safetensors.index.json")
    wmap = json.load(open(idx))["weight_map"] if os.path.exists(idx) else None
    out, handles = {}, {}
    keys = None
    if wmap is None:
        lone = os.path.join(snap, "model.safetensors")
        handles["model.safetensors"] = safe_open(lone, framework="pt")
        keys = set(handles["model.safetensors"].keys())
    for key in (wmap.keys() if wmap else keys):
        if not key.endswith(".weight"):
            continue
        shard = wmap[key] if wmap else "model.safetensors"
        if shard not in handles:
            handles[shard] = safe_open(os.path.join(snap, shard),
                                       framework="pt")
        out[key[:-len(".weight")]] = (
            handles[shard].get_tensor(key).float().cpu())
    return out


def wrap_fresh(model, args, frozen_override=None):
    """(Re)attach BoundedFill modules around every Linear4bit.

    frozen_override: {name: (codes, absmax, blocksize, anchors)} to re-anchor
    on a consolidated artifact instead of the live bnb state. An optional
    args.target_filter (substring) restricts wrapping to matching layers:
    at 27B on a 121 GB machine the dense anchors and room of every layer
    do not fit, and writing the MLPs alone costs a few points of recall
    (the structure experiments) for a third of the memory.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    name_filter = getattr(args, "target_filter", None)
    w0_map = None
    if getattr(args, "anchor_source", "artifact") == "original":
        assert getattr(args, "master_checkpoint", None), (
            "--anchor-source original needs --master-checkpoint")
        w0_map = _load_dense_master(args.master_checkpoint)
        print(f"[wrap] anchor source ORIGINAL: {len(w0_map)} dense tensors "
              f"loaded; vendor walls and codes stay verbatim", flush=True)
    zero_room = 0
    frozen = {}
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            base = child.base if isinstance(child, BoundedFill) else child
            kind = type(base).__name__
            if kind not in ("Linear4bit", "PackedLinear"):
                continue
            full = f"{name}.{child_name}" if name else child_name
            if name_filter and name_filter not in full:
                continue
            if kind == "PackedLinear":
                # #23 Phase A: the W4A16 uniform grid, artifact anchors only,
                # minor-version writing. Folding this grid is Phase B; the
                # fold entry and the A2/override paths guard against it.
                if frozen_override is not None or w0_map is not None:
                    raise NotImplementedError(
                        "uniform grid: artifact anchors only (Phase A)")
                from cellfill.uniform import uniform_anchors
                codes_u = base.codes()
                anchors_u = uniform_anchors(codes_u, base.scale, base.group,
                                            base.zero_point)
                hook = getattr(base, "_hf_hook", None)
                setattr(module, child_name,
                        BoundedFill(base, None, args.rank, args.tanh_scale,
                                    anchors=anchors_u,
                                    margin=args.margin, uniform=0.05,
                                    anchor_dtype=ANCHOR_STORE_DTYPE,
                                    dense_fill=getattr(args, "dense_fill",
                                                       False)))
                _reattach_hook(getattr(module, child_name), hook)
                zp_u = (None if base.zero_point is None
                        else base.zero_point.cpu())
                frozen[full] = (codes_u.cpu(), base.scale.cpu(),
                                ("uniform", base.group, base.qmin,
                                 base.qmax, zp_u), None)
                continue
            if frozen_override is not None:
                codes, absmax, bsz, anchors = frozen_override[full]
                if anchors is None:
                    raise ValueError(
                        f"{full}: this frozen map carries no anchors (they "
                        "are not stored, see wrap_fresh); rewrap from the "
                        "model instead of from the map")
                codes_d = codes.to(base.weight.device)
                absmax_d = absmax.to(base.weight.device)
                anchors_d = anchors.to(base.weight.device).float()
                shape = tuple(anchors.shape)
            else:
                fs = frozen_state_from_linear4bit(base)
                codes_d, absmax_d, bsz = fs["codes"], fs["absmax"], fs["blocksize"]
                anchors_d = fs["anchors"].float()
                shape = fs["shape"]
            if w0_map is not None:
                # A2, locked spec: the vendor walls and codes above are kept
                # verbatim -- no re-quantization anywhere -- and only the
                # learning CENTER moves to the dense original. Room is
                # measured from W0 against the existing walls; W0 outside
                # the margin band clamps to zero room (starts frozen).
                anchors_d = w0_map[full].to(base.weight.device
                                            if hasattr(base, "weight")
                                            else anchors_d.device).float()
                assert tuple(anchors_d.shape) == tuple(shape), (
                    full, anchors_d.shape, shape)
            lo, hi = bin_bounds(codes_d, absmax_d, bsz, capped=True,
                                margin=args.margin)
            a = anchors_d.reshape(-1)
            # A sharded model carries an accelerate hook on every dispatched
            # module, and that hook is what moves an activation to the card
            # its next layer lives on. Replacing the child drops it: the
            # replacement is a different object, and accelerate has no way to
            # know. On one card that costs nothing, since there is nothing to
            # align. Across two it means the matmul reads whatever was at
            # that address on the wrong card, and the failure surfaces
            # downstream as a device-side assert in the first kernel with a
            # bounds check. The hook is reattached below, at the device the
            # anchors were placed on.
            hook = getattr(base, "_hf_hook", None)
            if getattr(args, "fold", "anchor") == "preimage":
                setattr(module, child_name,
                        PreimageFill(base, lo, hi, anchors_d.view(shape),
                                     args.rank, args.tanh_scale))
            else:
                # The writable cell is not the whole cell: a fold writes
                # the moved weight back into the fp16 anchor buffer, and where
                # a block's scale is small enough to put its weights in that
                # format's subnormal range the write can move them further
                # than the margin reserves. storable_mask is that condition;
                # the weights it excludes keep zero room and never move.
                halfwidth = (torch.minimum(a - lo, hi - a).clamp_min(0.0)
                             * storable_mask(absmax_d, bsz, a.numel(),
                                             ANCHOR_STORE_DTYPE,
                                             args.margin)).view(shape)
                if w0_map is not None:
                    # A2 instrumentation: centers outside their served cell
                    # (vendor-time vs double-quantized served scales
                    # disagree) plus sub-floor blocks -- all start frozen
                    zero_room += int((halfwidth.reshape(-1) == 0).sum())
                walls = None
                if getattr(args, "wall_codebook", False):
                    from cellfill.bins import (
                        nibble_layout_of, normalized_cell_edges,
                    )
                    lo_e, hi_e = normalized_cell_edges(True)
                    walls = (lo_e, hi_e, absmax_d, bsz,
                             nibble_layout_of(base.weight.data, codes_d),
                             torch.empty(0, dtype=torch.long,
                                         device=codes_d.device))
                setattr(module, child_name,
                        BoundedFill(base, halfwidth, args.rank,
                                    args.tanh_scale,
                                    anchors=anchors_d.view(shape),
                                    margin=args.margin, walls=walls,
                                    anchor_dtype=ANCHOR_STORE_DTYPE,
                                    dense_fill=getattr(args, "dense_fill",
                                                       False)))
            _reattach_hook(getattr(module, child_name), hook)
            # The fourth slot is the anchor map, and nothing reads it: every
            # consumer in this file and in fed_node / exp_fuse_rounds unpacks
            # it as `_`, and the one path that would use it (frozen_override)
            # has no caller. Keeping it costs two bytes per constrained
            # weight of HOST memory for the life of the run -- 45.4 GiB at
            # 27B, which is what makes two sequences on the two cards of one
            # 123 GiB machine impossible. The anchors live on the card and,
            # after a consolidation, are recoverable from the codes.
            frozen[full] = (codes_d.cpu(), absmax_d.cpu(), bsz, None)

    # Once a layer carries explicit anchors, its packed 4-bit weight is dead
    # storage: the forward is anchors + fill (BoundedFill._apply_fill takes the
    # self.base(x) branch only when anchors is None), the bound is the dense M,
    # consolidate() and verify_artifact() work from anchors and the CPU-side
    # frozen map, and the codes and scales this function needs were already
    # copied to the host above. Dropping it is what makes a sequence fit on one
    # card at 27B and above -- 12.2 GB there, about 18 at 32B.
    freed = 0
    for _, mod in model.named_modules():
        if not isinstance(mod, BoundedFill) or mod.anchors is None:
            continue
        if getattr(mod, "codebook", False) or getattr(mod, "walls", False):
            continue          # both rebuild the bound from base.weight
        w = getattr(getattr(mod, "base", None), "weight", None)
        if w is None or w.data.numel() == 0:
            continue
        freed += w.data.numel() * w.data.element_size()
        w.data = torch.empty(0, dtype=w.data.dtype, device=w.data.device)
    if freed:
        torch.cuda.empty_cache()
        print(f"[wrap] released {freed / 2**30:.1f} GB of packed 4-bit weights "
              f"the explicit-anchor path does not read", flush=True)
    if w0_map is not None:
        print(f"[wrap] A2 start-frozen (zero-room) weights: {zero_room} "
              f"(out-of-served-cell centers + sub-floor blocks)", flush=True)
    return frozen


def _displacement(mod, m_exact):
    """What a fold writes, and the normalized position that goes with it.

    Under the bound these are M*tanh(s*BA) and tanh(s*BA); with --unconstrained
    the displacement is the raw product and there is no normalized position,
    so the raw product stands in for the diagnostics. Both paths are computed
    here so the fold and the verification cannot disagree about which one is
    in force.
    """
    if getattr(mod, "dense_fill", False):
        raw = mod.Z.float().reshape(-1)
    else:
        raw = (mod.B @ mod.A).float().reshape(-1)
    if getattr(mod, "unconstrained", False):
        return raw, raw
    t = torch.tanh(mod.tanh_scale * raw)
    return m_exact * t, t


def _diag_index(name, numel, frac, seed=0):
    """Which coordinates the interference diagnostic keeps, per layer.

    The cosine against the previous fold and against the accumulated
    displacement are the only parts of the diagnostic that must remember
    anything, and what they remember is one half per constrained weight --
    twice, prev and accumulated. That is 5.2 GiB at 1.7B and 90.7 GiB at 27B,
    against 116 GiB of host memory on a machine that is also serving; at 32B
    it is 115.5 GiB and does not fit at all. Sampling a fixed fraction bounds
    it. The indices are derived from the layer name and a seed rather than
    stored, so the same coordinates are compared at every fold, and drawn with
    replacement because a permutation of 2.4e10 is not affordable and the
    duplicate rate at k/n <= 1/32 is negligible.

    A cosine over a random coordinate subset estimates the full cosine, and at
    the fractions used here the subset is still 7.6e8 coordinates wide, so the
    quantity reported is not meaningfully noisier than the exact one. Full
    fidelity remains the default: frac >= 1 returns None and every archived
    1.7B run is unaffected.
    """
    if frac >= 1.0:
        return None
    import zlib

    k = max(1, int(numel * frac))
    g = torch.Generator(device="cpu")
    g.manual_seed((zlib.crc32(name.encode()) ^ (seed * 2654435761)) & 0x7FFFFFFF)
    return torch.randint(0, numel, (k,), generator=g)


@torch.no_grad()
def fold_into_anchors(model, frozen, prev_t=None, hist_bins=20,
                      diag_frac=1.0, diag_seed=0, master=None):
    """(See below.) When prev_t is a dict it is read for the previous fold's
    displacement and overwritten with this one's, and the return value carries
    the interference diagnostic alongside E|t|."""
    """Requires the dense bound. After a fold the anchors leave the 4-bit grid,
    so M is no longer a function of the codes and cannot be rebuilt from the
    codebook -- exp_seq must run without --codebook-m."""
    """Fold the current bounded fill into the anchor tensors held by each
    BoundedFill, so the next task starts from the accumulated residual.

    The frozen artifact (codes, absmax) is untouched: the new anchor is the
    current in-cell position, and the remaining half-width shrinks to the
    distance to that cell's walls. This is what makes the residual
    *accumulate* across tasks instead of being overwritten.
    """
    abs_t_sum, abs_t_n, reverted = 0.0, 0, 0
    # Per-fold writable floor. The storable check is a before-fold-0
    # condition, but the room contracts every fold while the storage grid at
    # each weight does not; the audit found the only two constrained
    # violations on record (exp70 pair, 1 / 1.409e9 each) at the LAST fold of
    # their sequences, where the room was smallest. Ratio below 1 predicts a
    # violation at this fold; the min traced across folds says which fold
    # crosses first.
    floor_min, floor_b1, floor_b2 = float("inf"), 0, 0
    # Interference diagnostic. E|t| is a mean and hides everything that matters
    # about the distribution: whether a tail sits against the walls, and whether
    # this task is writing over the last one or beside it. Both are computed
    # here because this is the only place the displacement exists before it is
    # folded away and B is zeroed.
    hist = torch.zeros(hist_bins, dtype=torch.float64)
    sat_n = 0
    dot = nsq = psq = 0.0
    conflict_n = conflict_tot = 0
    dot_acc = nsq_acc = asq = 0.0
    conflict_acc = conflict_acc_tot = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, BoundedFill):
            continue
        if getattr(mod, "codebook", False):
            raise RuntimeError(
                "sequential folding needs the dense bound: after a fold the "
                "anchor is no longer on the 4-bit grid, so M cannot be "
                "rebuilt from the codes. Run without --codebook-m.")
        codes, absmax, bsz, _ = frozen[name]
        uniform_g = isinstance(bsz, tuple) and bsz[0] == "uniform"
        dev = mod.anchors.device
        codes_d, absmax_d = codes.to(dev), absmax.to(dev)
        anch = mod.anchors.float().reshape(-1)
        if uniform_g:
            # W4A16: walls are fixed by (codes, scale) -- cell centers
            # (q - z) * s with half-width |s|/2, pulled in by the SAME
            # margin the layer trains with; end cells capped symmetrically
            # (conservative, the uniform_halfwidth convention). Every cell
            # is storable: anchors are int-times-fp16-scale products, and
            # the fold-floor diagnostic below still measures the true-edge
            # reserve against the storage half-ulp.
            from cellfill.uniform import check_invariance_uniform
            from cellfill.uniform import expand_scale as _ues
            _, ugroup, uqmin, uqmax, uzp = bsz
            qf = codes_d.float()
            uzp_d = None if uzp is None else uzp.to(dev)
            if uzp_d is not None:
                qf = qf - _ues(uzp_d.float(), ugroup)
            s_pw = _ues(absmax_d.float(), ugroup)
            center = (qf * s_pw).reshape(-1)
            halfw = (s_pw.abs() * (0.5 - mod.umargin)).reshape(-1)
            lo, hi = center - halfw, center + halfw
            smask = torch.ones_like(anch)
            m_exact = torch.minimum(anch - lo, hi - anch).clamp_min(0.0)

            def _check(w):
                return check_invariance_uniform(
                    w.view(mod.shape), codes_d, absmax_d, ugroup,
                    uqmin, uqmax, uzp_d)
        else:
            lo, hi = bin_bounds(codes_d, absmax_d, bsz, capped=True,
                                margin=mod.margin)
            smask = storable_mask(absmax_d, bsz, anch.numel(),
                                  mod.anchors.dtype, mod.margin)
            m_exact = torch.minimum(anch - lo, hi - anch).clamp_min(0.0) * smask

            def _check(w):
                return check_invariance(w, codes_d, absmax_d, bsz,
                                        writable=(m_exact > 0))
        disp, t = _displacement(mod, m_exact)
        # Prop. 6 predicts the room decays by a factor 1 - E|t| per task. That
        # prediction is only testable against an E|t| measured here, from the
        # tanh actually applied; inverting the room ratio to "recover" E|t|
        # restates the ratio and confirms nothing.
        abs_t_sum += float(t.abs().sum())
        abs_t_n += t.numel()
        if prev_t is not None:
            # Orthogonality to the PREVIOUS task does not bound opposition to
            # everything already written, which is the comparison a reviewer
            # asks for next; the running sum is kept for exactly that.
            sel = _diag_index(name, t.numel(), diag_frac, diag_seed)
            ts = t.reshape(-1) if sel is None else t.reshape(-1)[
                sel.to(t.device)]
            acc = prev_t.get("__acc__" + name)
            if acc is not None:
                q = acc.to(t.device, torch.float32)
                dot_acc += float((ts * q).sum())
                nsq_acc += float((ts * ts).sum())
                asq += float((q * q).sum())
                conflict_acc += int(((ts * q) < 0).sum())
                conflict_acc_tot += int(q.numel())
                prev_t["__acc__" + name] = (q + ts).detach().half().cpu()
            else:
                prev_t["__acc__" + name] = ts.detach().half().cpu()
            at = t.abs()
            hist += torch.histc(at.float(), bins=hist_bins, min=0.0,
                                max=1.0).double().cpu()
            sat_n += int((at > 0.99).sum())
            old = None if name.startswith("__acc__") else prev_t.get(name)
            if old is not None:
                o = old.to(t.device, torch.float32)
                dot += float((ts * o).sum())
                nsq += float((ts * ts).sum())
                psq += float((o * o).sum())
                both = (ts != 0) & (o != 0)
                conflict_n += int(((ts * o) < 0).sum())
                conflict_tot += int(both.sum())
            prev_t[name] = ts.detach().half().cpu()
        w_new = anch + disp
        # Check the value that will actually be STORED, not the fp32 one: the
        # anchor buffer's rounding happens after this line, and checking
        # before it is what let 19,791 weights through at 8B before the
        # writable-cell floor (cellfill.bins.anchor_absmax_floor) removed the
        # cause. With the floor in place this reverts nothing on any anchor
        # measured so far; it stays as the check of record.
        w_new = w_new.to(mod.anchors.dtype).float()
        # The check's domain is the writable domain -- m_exact > 0, which
        # subsumes the storable mask AND the clamp-frozen weights (an A2
        # center outside its served cell has zero room by clamp and can
        # never move; on a published 1.7B artifact 8.0e4 such weights in
        # one layer failed the round trip while never having moved,
        # because the vendor's quantization-time scale and the served
        # double-quantized scale disagree). A sub-floor block's
        # anchors underflow in the storage dtype (30B: 31 blocks at absmax
        # -1.9e-8 store all sixteen levels as +-0), so the round trip fails
        # on weights that never moved; their bytes are carried verbatim at
        # every consolidation instead, which is what keeps the exemption
        # honest.
        _, n_bad, bad_idx = _check(w_new)
        if n_bad and getattr(mod, "unconstrained", False):
            # The control must not be repaired. Reverting the weights that
            # left their cells would put the bound back by the side door and
            # the arm would measure nothing; the count is carried to
            # verify_artifact and reported.
            reverted += 0
            reverted_idx = None
        elif n_bad:
            # The artifact is the invariant, so a weight that would leave its
            # cell keeps its old position (fill reverted) and is counted;
            # more than one in a million is a different kind of failure.
            if n_bad > max(1, w_new.numel() // 1_000_000):
                torch.save(
                    dict(name=name, anch=anch.cpu(), disp=disp.cpu(),
                         m_exact=m_exact.cpu(), t=t.cpu(),
                         w_new=w_new.cpu(), bad_idx=bad_idx.cpu(),
                         uniform_g=uniform_g,
                         lo=lo.cpu(), hi=hi.cpu()),
                    "out/fold_crash_dump.pt")
                raise RuntimeError(
                    f"{name}: {n_bad} invariance violations on fold "
                    f"(state dumped to out/fold_crash_dump.pt)")
            w_new[bad_idx] = anch[bad_idx]
            w_new = w_new.to(mod.anchors.dtype).float()
            _, still, _ = _check(w_new)
            assert still == 0, (name, still)
            print(f"[fold] {name}: {n_bad} weight(s) reverted to their "
                  f"anchor at the fold", flush=True)
            reverted += n_bad
            reverted_idx = bad_idx
        else:
            reverted_idx = None
        if master is not None:
            # error-feedback accumulation: the CONTINUOUS learned signal
            # (fp32 displacement, pre storage-cast) goes into the master;
            # the anchors' storage rounding is a serving detail and is
            # exactly what must not compound across versions
            d = disp.detach().float()
            if reverted_idx is not None:
                d = d.clone()
                d[reverted_idx] = 0.0
            master[name] += d.cpu().view(master[name].shape)
        w_st = w_new.to(mod.anchors.dtype)
        half_ulp = (torch.nextafter(
            w_st, torch.full_like(w_st, float("inf"))) - w_st
        ).abs().float().mul_(0.5)
        # toward +inf overflows only at the dtype max, unreachable for
        # anchors; map that inf to 0 so it can never fake a small ratio
        half_ulp = torch.nan_to_num(half_ulp, posinf=0.0)
        # Measure the distance to the TRUE cell edge, not to the capped
        # wall. A saturated weight parks exactly ON the capped wall by
        # design (it spent its writable room) and is perfectly safe: what
        # protects the storage rounding is the margin reserve between the
        # capped wall and the true edge. The first version measured the
        # writable room and read 3.66M "below-1" weights on a healthy 1.7B
        # fold with zero violations -- saturated weights, all of them.
        if uniform_g:
            # true edges of the uniform cell: center +- |s|/2 exactly
            full_half = (halfw / (0.5 - mod.umargin)) * 0.5
            lo_t, hi_t = center - full_half, center + full_half
        else:
            lo_t, hi_t = bin_bounds(codes_d, absmax_d, bsz, capped=False)
        m_true = torch.minimum(w_new - lo_t, hi_t - w_new).clamp_min_(0.0)
        ratio = torch.where(smask.reshape(-1) > 0,
                            m_true / half_ulp.clamp_min(1e-45),
                            torch.full_like(m_true, float("inf")))
        del lo_t, hi_t
        floor_min = min(floor_min, float(ratio.min()))
        floor_b1 += int((ratio < 1.0).sum())
        floor_b2 += int((ratio < 2.0).sum())
        del w_st, half_ulp, m_true, ratio
        mod.anchors = w_new.view(mod.anchors.shape).to(mod.anchors.dtype)
        if uniform_g:
            # after a fold the anchor leaves the group center, so the static
            # per-group bound would overshoot the wall; switch the layer to
            # the dense-M path (the bound dispatch tries uniform first, then
            # walls, then M) with the exact remaining room
            mod.uniform = False
            mod.M = (torch.minimum(w_new - lo, hi - w_new).clamp_min(0.0)
                     ).view(mod.shape).to(torch.bfloat16)
        elif not getattr(mod, "walls", False):
            mod.M = (torch.minimum(w_new - lo, hi - w_new)
                     * storable_mask(absmax_d, bsz, w_new.numel(),
                                     mod.anchors.dtype, mod.margin)
                     ).view(mod.M.shape).to(mod.M.dtype)
        (mod.Z if getattr(mod, "dense_fill", False) else mod.B).data.zero_()
    if reverted:
        print(f"[fold] {reverted} weights reverted in total", flush=True)
    print(f"[fold-floor] min room/half-ulp {floor_min:.3g} "
          f"below-1 {floor_b1} below-2 {floor_b2}", flush=True)
    mean_abs_t = abs_t_sum / max(abs_t_n, 1)
    if prev_t is None:
        return mean_abs_t
    denom = (nsq ** 0.5) * (psq ** 0.5)
    diag = dict(
        mean_abs_tanh=mean_abs_t,
        floor_min_ratio=None if floor_min == float("inf") else floor_min,
        floor_below_1=floor_b1,
        floor_below_2=floor_b2,
        saturated_frac=sat_n / max(abs_t_n, 1),
        hist_abs_tanh=[x / max(float(hist.sum()), 1.0) for x in hist.tolist()],
        cosine_with_previous=(dot / denom) if denom else None,
        sign_conflict_frac=(conflict_n / conflict_tot) if conflict_tot else None,
        cosine_with_accumulated=(
            dot_acc / ((nsq_acc ** 0.5) * (asq ** 0.5))
            if nsq_acc and asq else None),
        sign_conflict_accumulated=(
            conflict_acc / conflict_acc_tot if conflict_acc_tot else None),
    )
    return mean_abs_t, diag


@torch.no_grad()
def consolidate(model, frozen, blocksize=64, calibrated=False, trust=None,
                master=None):
    """Major version: re-quantize the current weights into a NEW artifact.

    Returns the new frozen map. The 4-bit file changes here by design --
    this is the explicit version bump, not a silent drift.

    With calibrated=True the block scale is searched rather than pinned to
    max|w|. The reason is a number, not a preference: five archived
    consolidation events cost -0.799 +- 0.238 points of mean suite accuracy,
    all five negative, and the cost does not scale with how much of the file
    is rewritten -- 1.2% of codes changed costs as much as 34.6%. Damage
    proportional to change would scale; a fixed per-weight quantizer error
    would not. Round-to-nearest under a max|w| scale is the crudest quantizer
    available and it is what this path has always used.
    """
    new_frozen, changed, total = {}, 0, 0
    n_frozen_blocks, min_absmax = 0, float("inf")
    carried = 0
    # Scale-trajectory instrument. The ten-burn question is whether the
    # block scale drifts multiplicatively across major versions; the burn
    # itself was measured drift-free (five consecutive re-quantizations
    # with nothing written: 0.0000% codes, absmax exactly fixed), so any
    # drift must enter through what the fill pushed to the capped wall.
    # Per burn this records new/old scale per block: mean, p99 (from a
    # 1e-3-bin histogram, streaming), max, how many blocks grew at all and
    # how many grew past 1.10 (the capped wall admits at most 1.1489), the
    # mean room after, and the carried count. Without this the products
    # cannot answer the question the run exists to ask.
    r_sum, r_cnt, r_max, r_gt1, r_gt11 = 0.0, 0, 0.0, 0, 0
    r_hist = torch.zeros(2002, dtype=torch.float64)
    room_sum, room_cnt = 0.0, 0
    n_trust_clamped = 0
    from cellfill.bins import anchor_absmax_floor  # noqa: F401  (used below)
    for name, mod in model.named_modules():
        if not isinstance(mod, BoundedFill):
            continue
        old_codes, old_absmax, bsz, _ = frozen[name]
        if master is not None:
            # single projection: quantize the continuous accumulated state,
            # never the previous quantization's output
            w = master[name].to(mod.anchors.device)
        else:
            w = mod.anchors.float()
        if calibrated:
            from cellfill.calib import calibrated_quantize
            from cellfill.bins import anchor_absmax_floor
            absmax, codes, _, _ = calibrated_quantize(
                w, bsz, floor=anchor_absmax_floor(mod.anchors.dtype,
                                                  mod.margin))
        else:
            absmax = compute_absmax(w, bsz)
            codes = assign_codes(w, absmax, bsz)
        if trust is not None:
            # Scale trust region: the new grid may move, but not far. A
            # negative old scale flips the bound order, so sort the pair.
            old_am = old_absmax.float().reshape(-1).to(absmax.device)
            b1, b2 = old_am * (1.0 - trust), old_am * (1.0 + trust)
            t_lo, t_hi = torch.minimum(b1, b2), torch.maximum(b1, b2)
            clamped = absmax.float().reshape(-1).clamp(t_lo, t_hi)
            moved = int((clamped != absmax.float().reshape(-1)).sum())
            if moved:
                absmax = clamped.view_as(absmax).to(absmax.dtype)
                # codes are only valid under the scale that made them
                codes = assign_codes(w, absmax, bsz)
            n_trust_clamped += moved
        # Carry-through for blocks frozen under the OLD scale. Their anchors
        # are stored collapsed -- the storage dtype underflows below the
        # floor (at 30B, 31 blocks of one layer store all sixteen levels as
        # +-0) -- so re-deriving codes from those anchors would rewrite the
        # artifact's bytes for weights that never moved. The bytes are
        # carried verbatim instead: same codes, same scale, and the weights
        # stay frozen under the new artifact by the same floor.
        keep = (old_absmax.float().abs()
                < anchor_absmax_floor(mod.anchors.dtype, mod.margin))
        if bool(keep.any()):
            keep_d = keep.to(codes.device)
            cb = codes.view(-1, bsz).clone()
            cb[keep_d] = old_codes.view(-1, bsz).to(codes.device)[keep_d]
            codes = cb.reshape(-1)
            absmax = torch.where(keep_d.to(absmax.device),
                                 old_absmax.to(absmax.device, absmax.dtype),
                                 absmax)
            carried += int(keep.sum())
        # Both arms report what the re-quantization froze. A searched scale
        # can only shrink, and shrinking is what takes a block under the
        # writable-cell floor, so a comparison that does not carry this count
        # is a comparison with a second variable in it. calibrated_quantize
        # asserts it created none; this is the number that shows the assertion
        # had something to assert on.
        _fl = anchor_absmax_floor(mod.anchors.dtype, mod.margin) \
            if calibrated else None
        if _fl is not None:
            n_frozen_blocks += int((absmax < _fl).sum())
            min_absmax = min(min_absmax, float(absmax.min()))
        anchors = dequantize_ref(codes, absmax, bsz, shape=tuple(w.shape))
        changed += int((codes.cpu() != old_codes.view(-1)).sum())
        total += codes.numel()
        old_flat = old_absmax.float().reshape(-1).to(absmax.device)
        new_flat = absmax.float().reshape(-1)
        r = torch.where(old_flat == 0, torch.ones_like(new_flat),
                        new_flat / old_flat)
        r_sum += float(r.sum()); r_cnt += r.numel()
        r_max = max(r_max, float(r.max()))
        r_gt1 += int((r > 1.0).sum()); r_gt11 += int((r > 1.10).sum())
        r_hist += torch.histc(r.clamp(0.0, 2.001), bins=2002,
                              min=0.0, max=2.002).double().cpu()
        dev = mod.M.device
        lo, hi = bin_bounds(codes.to(dev), absmax.to(dev), bsz, capped=True,
                            margin=mod.margin)
        a = anchors.to(dev).reshape(-1)
        mod.anchors = anchors.view(w.shape).to(mod.anchors.dtype).to(dev)
        room_t = (torch.minimum(a - lo, hi - a).clamp_min(0.0)
                  * storable_mask(absmax.to(dev), bsz, a.numel(),
                                  mod.anchors.dtype, mod.margin))
        room_sum += float(room_t.sum()); room_cnt += room_t.numel()
        if not getattr(mod, "walls", False):
            mod.M = room_t.view(mod.M.shape).to(mod.M.dtype)
        else:
            # the edges move at a major version: re-seat the tables' scales
            mod.wabsmax = absmax.to(dev, torch.float32)
        (mod.Z if getattr(mod, "dense_fill", False) else mod.B).data.zero_()
        new_frozen[name] = (codes.cpu(), absmax.cpu(), bsz, None)
    if calibrated:
        print(f"[consolidate] calibrated: {n_frozen_blocks} blocks below the "
              f"writable-cell floor, smallest served scale {min_absmax:.3e}",
              flush=True)
    if carried:
        print(f"[consolidate] {carried} sub-floor blocks carried verbatim "
              f"(bytes preserved; their anchors underflow the storage dtype)",
              flush=True)
    if trust is not None:
        print(f"[consolidate] trust {trust}: {n_trust_clamped} blocks "
              f"clamped to the old scale's (1+-d) band", flush=True)
    p99 = None
    if r_cnt:
        cum = torch.cumsum(r_hist, 0)
        p99 = round(int(torch.searchsorted(cum, 0.99 * cum[-1])) * 1e-3, 4)
    scale_stats = dict(
        absmax_ratio_mean=(r_sum / r_cnt) if r_cnt else None,
        absmax_ratio_p99=p99,
        absmax_ratio_max=r_max if r_cnt else None,
        n_ratio_gt1=r_gt1, n_ratio_gt1p1=r_gt11,
        room_after_mean=(room_sum / room_cnt) if room_cnt else None,
        n_carried_blocks=carried, n_blocks=int(r_cnt),
        trust=trust, n_trust_clamped_blocks=n_trust_clamped,
    )
    print(f"[consolidate] scale ratio mean {scale_stats['absmax_ratio_mean']}"
          f" p99 {p99} max {scale_stats['absmax_ratio_max']}"
          f" gt1 {r_gt1} gt1.1 {r_gt11}", flush=True)
    return new_frozen, changed / max(total, 1), scale_stats


@torch.no_grad()
def verify_artifact(model, frozen):
    """Assert the live weights still re-quantize to the frozen codes."""
    n_bad = n_tot = 0
    for name, mod in model.named_modules():
        if isinstance(mod, PreimageFill):
            codes, absmax, bsz, _ = frozen[name]
            if isinstance(bsz, tuple):
                raise NotImplementedError(
                    "uniform grid has no preimage path (Phase A)")
            dev = mod.zacc.device
            w = mod.weight(torch.float32).reshape(-1)
            _, bad, _ = check_invariance(w, codes.to(dev), absmax.to(dev), bsz)
            n_bad += bad
            n_tot += w.numel()
            continue
        if not isinstance(mod, BoundedFill):
            continue
        codes, absmax, bsz, _ = frozen[name]
        if isinstance(bsz, tuple) and bsz[0] == "uniform":
            # W4A16 serve check: every cell is writable (no sub-floor, no
            # storable mask on this grid); the bound is the per-group
            # half-width at the SAME margin the layer trains with, so the
            # verification cannot disagree with the training bound.
            from cellfill.uniform import (
                check_invariance_uniform,
                uniform_halfwidth,
            )
            _, ugroup, uqmin, uqmax, uzp = bsz
            dev = mod.anchors.device
            anch = mod.anchors.float().reshape(-1)
            m_u = uniform_halfwidth(absmax.to(dev), ugroup,
                                    margin=mod.umargin).reshape(-1)
            disp, _t = _displacement(mod, m_u)
            w = anch + disp
            _, bad, _ = check_invariance_uniform(
                w.view(mod.shape), codes.to(dev), absmax.to(dev), ugroup,
                uqmin, uqmax, None if uzp is None else uzp.to(dev))
            n_bad += bad
            n_tot += w.numel()
            continue
        dev = mod.M.device
        codes_d, absmax_d = codes.to(dev), absmax.to(dev)
        anch = mod.anchors.float().reshape(-1)
        lo, hi = bin_bounds(codes_d, absmax_d, bsz, capped=True,
                            margin=mod.margin)
        smask = storable_mask(absmax_d, bsz, anch.numel(),
                              mod.anchors.dtype, mod.margin)
        m_exact = torch.minimum(anch - lo, hi - anch).clamp_min(0.0) * smask
        disp, _t = _displacement(mod, m_exact)
        w = anch + disp
        # The counting domain is the storable domain, NOT m_exact > 0: after a
        # fold mod.anchors IS the served weight, so an anchor that has left its
        # cell has room 0 there and (m_exact > 0) would drop exactly the weights
        # this check exists to catch (2026-09-02, found by the served-weights
        # ground truth: 1.25M escapes counted with the original-anchor mask,
        # 0 with this one). Sub-floor blocks stay exempt through smask.
        _, bad, _ = check_invariance(w, codes_d, absmax_d, bsz,
                                     writable=(smask.reshape(-1) > 0))
        n_bad += bad
        n_tot += w.numel()
    return n_bad, n_tot


def _save_samples(directory, label, samples):
    """Per-item suite outcomes for one evaluation point, best effort.

    Never allowed to end a run: a sequence that has spent twenty hours must
    not die because a diagnostic file could not be written.
    """
    import json as _json
    import os
    import re

    try:
        os.makedirs(directory, exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")[:60]
        out = {}
        for task, rows in samples.items():
            out[task] = [
                {k: r.get(k) for k in ("doc_id", "acc", "acc_norm", "target")}
                for r in rows
            ]
        path = os.path.join(directory, f"{slug}.json")
        with open(path, "w") as fh:
            _json.dump(out, fh)
        n = sum(len(v) for v in out.values())
        print(f"[validate] {n} per-item outcomes -> {path}", flush=True)
    except Exception as e:  # noqa: BLE001 -- a diagnostic must not be fatal
        print(f"[validate] could not save per-item outcomes ({type(e).__name__}: "
              f"{e}); the run continues", flush=True)


def run_suite(model, tok, tasks, limit, label, bs=8, samples_dir=None):
    """The served model through lm-eval-harness, in place, mid-sequence.

    The harness has to be torn down explicitly. Its wrapper holds the model,
    its task objects hold materialized requests, and the cycles among them
    survive the function returning -- measured on the first attempt at 27B and
    32B, roughly 8.5 GiB was still allocated when training began, and both
    sequences died on the first optimizer step of task 0 with the suite's
    footprint still resident. Neither had come near that in a smoke, because a
    smoke does not validate first. The reported figure is the check: whatever
    is still allocated here is what training starts from.
    """
    import gc

    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    was = model.training
    model.eval()
    out = {}
    try:
        lm = HFLM(pretrained=model, tokenizer=tok, batch_size=bs)
        res = simple_evaluate(
            model=lm, tasks=[t.strip() for t in tasks.split(",") if t.strip()],
            limit=limit, bootstrap_iters=0,
            log_samples=samples_dir is not None)
        if samples_dir is not None:
            # Per-item outcomes, so a later comparison between two points in
            # the sequence can be paired. The suites are scored on the same
            # items throughout, and a paired test on the discordant pairs has
            # a smaller variance than the independent binomial we would
            # otherwise have to fall back on -- which matters most exactly
            # where the claim is that something did NOT change. Writing them
            # is cheap; not writing them is irreversible.
            _save_samples(samples_dir, label, res.get("samples") or {})
        for task, m in (res.get("results") or {}).items():
            for k, v in m.items():
                if k.startswith("acc_norm,") or (k.startswith("acc,")
                                                 and "acc_norm," not in m):
                    out[task] = round(float(v), 4)
    finally:
        lm = res = None
        del lm, res
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if was:
        model.train()
    print(f"[validate] {label}: {out}", flush=True)
    print("[mem] post-validate " + "  ".join(
        "cuda%d allocated %.1f reserved %.1f GB"
        % (d, torch.cuda.memory_allocated(d) / 2**30,
           torch.cuda.memory_reserved(d) / 2**30)
        for d in range(torch.cuda.device_count())), flush=True)
    return out


def prepare_tasks(args):
    """Everything the sequence needs from data, before any GPU is touched.

    Separated from main so it can be exercised without a card. That is not
    tidiness: the generator path and the --tasks-from path diverge entirely
    inside this function, and the file path's first attempt died here, on its
    first line, after the job had been queued. Nothing below this function can
    check it, and a smoke run that reaches it has already paid for a model
    load.
    """
    tasks, pmaps = task_facts(args)
    names = [item_keys(ts) for ts in tasks]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            shared = names[i] & names[j]
            assert not shared, (
                f"tasks {i} and {j} share {len(shared)} facts "
                f"(e.g. {sorted(shared)[:3]}); task fact sets must be disjoint")
    probes = [item_probes(ts, pm) for ts, pm in zip(tasks, pmaps)]
    if args.rehearse_holdout < 1.0:
        halves = [rehearsal_split(ts, args.rehearse_holdout, args.seed, i)
                  for i, ts in enumerate(tasks)]
        rehearsable = [a for a, _ in halves]
        heldout = [b for _, b in halves]
        probes_rehearsed = [item_probes(a, pm)
                            for a, pm in zip(rehearsable, pmaps)]
        probes_heldout = [item_probes(b, pm) for b, pm in zip(heldout, pmaps)]
        print("[split] rehearsal hold-out on: per task %d rehearsable / %d "
              "held out; probes %d / %d"
              % (len(rehearsable[0]), len(heldout[0]),
                 len(probes_rehearsed[0]), len(probes_heldout[0])), flush=True)
    else:
        rehearsable = tasks
        heldout = probes_rehearsed = probes_heldout = None
    return (tasks, probes, rehearsable, heldout,
            probes_rehearsed, probes_heldout)


def main():
    args = get_args()
    assert torch.cuda.is_available(), "exp_seq needs a GPU"
    torch.manual_seed(args.seed)
    t0 = time.time()

    (tasks, probes, rehearsable, heldout,
     probes_rehearsed, probes_heldout) = prepare_tasks(args)

    # Data preflight, BEFORE the model: everything the run's eval stages
    # will need from the datasets cache, loaded while failure still costs
    # seconds. 300_loop8b_s2 spent 30 minutes loading 8B on an offline host
    # with no ai2_arc cache and died on its FIRST validate with nothing
    # banked -- the launch gates verify training can start, not that the
    # eval stage's data exists. get_task_dict is the same constructor
    # simple_evaluate uses, so what is checked is what runs.
    print("[preflight] loading eval corpora and validate task data",
          flush=True)
    ppl_txt = wikitext_text()
    x_txt = lambada_text()
    if args.validate:
        from lm_eval.tasks import get_task_dict
        get_task_dict([t.strip() for t in args.validate_tasks.split(",")
                       if t.strip()])
    print("[preflight] all eval data loadable", flush=True)

    print(f"[load] {args.model} NF4; {args.tasks} tasks x "
          f"{args.facts_per_task} facts ({bits_per_fact():.1f} bit/fact)")
    model, tok = build_4bit(args.model)

    ppl_anchor = eval_ppl(model, tok, ppl_txt, max_chunks=args.max_ppl_chunks)
    ppl_anchor_x = maybe_ppl(model, tok, x_txt, args.max_ppl_chunks)
    print(f"[base] ppl={ppl_anchor:.3f} lambada={ppl_anchor_x}")

    frozen = wrap_fresh(model, args)
    print(f"[wrap] {len(frozen)} BoundedFill layers")
    master = None
    if args.consolidate_master:
        # fp32 is load-bearing, not a preference: an fp16 master would move
        # the storage-rounding accumulation from the anchor buffer into the
        # master and resurrect the sub-floor violation family there.
        master = {}
        if args.consolidate_master == "original":
            assert args.master_checkpoint, (
                "--consolidate-master original needs --master-checkpoint")
            dense = _load_dense_master(args.master_checkpoint)
            for name, mod in model.named_modules():
                if not isinstance(mod, BoundedFill):
                    continue
                t = dense[name]
                assert tuple(t.shape) == tuple(mod.anchors.shape), (
                    name, t.shape, mod.anchors.shape)
                master[name] = t
            del dense
        else:
            for name, mod in model.named_modules():
                if isinstance(mod, BoundedFill):
                    master[name] = mod.anchors.detach().float().cpu().clone()
        assert all(v.dtype == torch.float32 for v in master.values())
        print(f"[master] {args.consolidate_master}: {len(master)} layers "
              f"buffered fp32 on host "
              f"({sum(v.numel() for v in master.values()) * 4 / 2**30:.1f} GB)",
              flush=True)
    if args.unconstrained:
        for m in model.modules():
            if isinstance(m, BoundedFill):
                m.unconstrained = True
        print("[wrap] CONTROL: the cell bound is removed; invariance is "
              "measured, not asserted", flush=True)
    if args.grad_checkpoint:
        # use_reentrant=False is not a preference. Under the reentrant path a
        # checkpointed segment whose inputs all have requires_grad=False
        # produces an output that does not require grad either, so the
        # backward never enters it -- and every input here is frozen, the
        # embedding included. The fill's gradients would come back None, the
        # optimizer would step on nothing, and the only symptom would be that
        # the memory problem went away. train() asserts the gradients are
        # real at the first step for the same reason.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
        print("[wrap] decoder activations recomputed in the backward pass",
              flush=True)
    if args.checkpoint_fill:
        for m in model.modules():
            if isinstance(m, (BoundedFill, PreimageFill)):
                m.checkpoint_fill = True
        print("[wrap] fill recomputation enabled", flush=True)

    kl_pool = None
    if args.inert > 0:
        assert args.fold == "anchor", "--inert needs the anchor fold"
        if args.inert_pool == "neutral":
            given = set().union(*names)
            kl_pool = training_texts([f for f in generate(
                args.facts_per_task * 2, seed=args.seed + 999)
                if f.name not in given])
        else:
            kl_pool = wikitext_train_snippets(4000, seed=args.seed + 7)
        print(f"[inert] KL weight {args.inert} on {len(kl_pool)} "
              f"{args.inert_pool} texts", flush=True)

    history, consolidations = [], []
    val_at = {int(x) for x in args.validate.split(",") if x.strip()}
    validations = []
    if val_at:
        validations.append(dict(point="anchor", after_task=None,
                                scores=run_suite(model, tok,
                                                 args.validate_tasks,
                                                 args.validate_limit,
                                                 "released anchor",
                                                 bs=args.validate_bs,
                                                 samples_dir=args.validate_samples)))
    prev_t = {} if args.fold_diagnostics else None
    fold_diag = None
    for t in range(args.tasks):
        n_rep = 0
        pool = None
        if args.replay_frac > 0:
            n_rep = int(args.facts_per_task * args.replay_frac
                        / (1 - args.replay_frac))
            pool = wikitext_train_snippets(n_rep * args.epochs,
                                           seed=args.seed + t)
        if args.rehearse_old > 0 and t > 0:
            # Only the rehearsable half of each earlier task. The other half
            # was trained on and is then never shown again, which is what
            # makes its recall a measure of forgetting rather than of
            # rehearsal (rehearse_holdout defaults to 1.0, i.e. all of it,
            # which is what the archived runs did).
            old_sents = [s_ for k in range(t)
                         for s_ in item_texts(rehearsable[k])]
            n_old = int(args.facts_per_task * args.rehearse_old)
            rng = random.Random(args.seed * 7 + t)
            rng.shuffle(old_sents)
            # train() takes epoch e's rehearsal as pool[e*n_rep:(e+1)*n_rep],
            # so the two sources are interleaved per epoch: each epoch's
            # slice holds its generic snippets followed by its old sentences
            w_pool = pool or []
            o_pool = (old_sents * (n_old * args.epochs // max(len(old_sents), 1)
                                   + 1))[:n_old * args.epochs]
            merged = []
            for e in range(args.epochs):
                merged += w_pool[e * n_rep:(e + 1) * n_rep]
                merged += o_pool[e * n_old:(e + 1) * n_old]
            pool = merged
            n_rep = n_rep + n_old
            print(f"[task {t}] rehearsing {n_old} old sentences/epoch from "
                  f"{len(set(old_sents))}", flush=True)
        print(f"[task {t}] training {args.epochs} epochs", flush=True)
        train(model, tok, item_texts(tasks[t]), args.epochs, args.lr,
              args.bs, args.seed + t, replay_pool=pool,
              n_replay_per_epoch=n_rep, kl_pool=kl_pool,
              kl_weight=args.inert, kl_bs=args.inert_bs)
        if args.fold == "preimage":
            vals = [m.fold() for m in model.modules()
                    if isinstance(m, PreimageFill)]
            mean_abs_t = sum(vals) / max(len(vals), 1)
        else:
            if args.fold_diagnostics:
                mean_abs_t, fold_diag = fold_into_anchors(
                    model, frozen, prev_t=prev_t,
                    diag_frac=args.fold_diag_sample, diag_seed=args.seed,
                    master=master)
            else:
                mean_abs_t = fold_into_anchors(model, frozen, master=master)

        n_bad, n_tot = verify_artifact(model, frozen)
        recalls = [eval_recall(model, tok, probes[k]) for k in range(t + 1)]
        rec_reh = rec_hold = None
        if probes_heldout is not None:
            rec_reh = [eval_recall(model, tok, probes_rehearsed[k])
                       for k in range(t + 1)]
            rec_hold = [eval_recall(model, tok, probes_heldout[k])
                        for k in range(t + 1)]
            # The task just trained has seen both halves equally, so at its own
            # fold they must agree to within sampling noise. If they do not,
            # the split has leaked into training and the held-out half is
            # measuring what was never learned rather than what was not
            # rehearsed -- a silent failure that would look like forgetting.
            gap = abs(rec_reh[t] - rec_hold[t])
            tol = 3.0 * (0.25 / max(len(probes_heldout[t]), 1)) ** 0.5
            print("[split] fresh task %d: rehearsable %.3f vs held out %.3f "
                  "(gap %.3f, tolerance %.3f)"
                  % (t, rec_reh[t], rec_hold[t], gap, tol), flush=True)
            if gap > tol:
                raise RuntimeError(
                    f"task {t} scores {rec_reh[t]:.3f} on its rehearsable half "
                    f"and {rec_hold[t]:.3f} on its held-out half at the fold "
                    f"where it was just trained. Both halves were trained on, "
                    f"so they must agree here; the split has reached training.")
        ppl = eval_ppl(model, tok, ppl_txt, max_chunks=args.max_ppl_chunks)
        ppl_x = maybe_ppl(model, tok, x_txt, args.max_ppl_chunks)
        # a running mean, not a concatenation: the concatenated room tensor
        # is 5.25 GB at 1.7B and was the one allocation a 24 GB card refused
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, PreimageFill):
                    r = m.room()
                elif isinstance(m, BoundedFill):
                    r = m.halfwidth() if getattr(m, "walls", False) else m.M
                else:
                    continue
                tot += float(r.float().sum())
                cnt += r.numel()
                del r
        room_left = tot / max(cnt, 1)
        # retention[k] = task k's recall now / its recall right after learning.
        # The ratio is only a measurement when the denominator is signal: on
        # the heterogeneous pre-run, api_cyclopts learned to 0.009 fresh and
        # later drifted to 0.197, which this line would have reported as
        # 2189% retention -- a number whose denominator is noise. Below the
        # floor the ratio is stored as None and said out loud; the fresh
        # value itself stays in recalls for the reader.
        # 0.05 is the transitional constant (3.5x the 1.4% base rate). The
        # agreed successor once per-domain anchor recalls ship with eval-set
        # v2: floor_k = anchor_k + 3*sqrt(p(1-p)/n_k), p = max(anchor_k,
        # 1/n_k) -- a fixed constant is one number against a probe-count-
        # dependent noise scale (0.05 is 2.3 sigma at n=100 but 1.4 sigma at
        # n=35). The two agree at n~100/anchor~0.014, so switching will not
        # reclassify existing runs.
        RETENTION_FRESH_FLOOR = 0.05
        retention = []
        for k in range(t + 1):
            fresh = history[k]["recalls"][k] if k < t else recalls[t]
            if fresh < RETENTION_FRESH_FLOOR:
                if fresh > 0:
                    print(f"[retention] task {k}: fresh recall {fresh:.3f} "
                          f"is below the {RETENTION_FRESH_FLOOR} floor; "
                          f"ratio undefined, not "
                          f"{recalls[k] / fresh:.1%}", flush=True)
                retention.append(None)
            else:
                retention.append(recalls[k] / fresh)
        row = dict(task=t, recalls=recalls, retention=retention, ppl=ppl,
                   recalls_rehearsed=rec_reh, recalls_heldout=rec_hold,
                   ppl_lambada=ppl_x, invariance_violations=n_bad,
                   weights_checked=n_tot, mean_room_left=room_left,
                   mean_abs_tanh=mean_abs_t)
        if t in val_at:
            row["validation"] = run_suite(model, tok, args.validate_tasks,
                                          args.validate_limit,
                                          f"after task {t}",
                                          bs=args.validate_bs,
                                          samples_dir=args.validate_samples)
            validations.append(dict(point="after_task", after_task=t,
                                    scores=row["validation"]))
        if fold_diag is not None:
            row["fold_diagnostics"] = fold_diag
            print(f"[fold-diag] task {t}: saturated {fold_diag['saturated_frac']:.4%}"
                  f"  cos(prev) {fold_diag['cosine_with_previous']}"
                  f"  sign-conflict {fold_diag['sign_conflict_frac']}",
                  flush=True)
        history.append(row)
        print(f"[task {t}] recalls={['%.3f' % r for r in recalls]} "
              f"ppl={ppl:.3f} violations={n_bad} room={room_left:.3e}",
              flush=True)

        if (args.mode == "consolidate"
                and (t + 1) % args.consolidate_after == 0
                and (t + 1 < args.tasks or args.consolidate_final)):
            frozen, frac, scale_stats = consolidate(
                model, frozen, calibrated=args.calibrated_consolidate,
                trust=args.consolidate_trust, master=master)
            if prev_t is not None:
                # the fill is zeroed and the anchors move, so the previous
                # task's displacement is no longer a comparable vector
                prev_t.clear()
            print(f"[consolidate] after task {t}: {frac:.2%} of 4-bit codes "
                  f"changed (explicit major version)", flush=True)
            consolidations.append(dict(after_task=t, codes_changed_frac=frac,
                                       **scale_stats))
            if t in val_at:
                consolidations[-1]["validation_before_recovery"] = run_suite(
                    model, tok, args.validate_tasks, args.validate_limit,
                    f"after consolidating at task {t}, before recovery",
                    bs=args.validate_bs, samples_dir=args.validate_samples)
            if args.recover_epochs:
                # every earlier task's own sentences, which is what the
                # re-quantization just perturbed
                rec = [x for k in range(t + 1) for x in item_texts(tasks[k])]
                print(f"[recover] {args.recover_epochs} epochs on "
                      f"{len(rec)} earlier sentences at lr {args.recover_lr}",
                      flush=True)
                train(model, tok, rec, args.recover_epochs, args.recover_lr,
                      args.bs, args.seed + 500 + t, replay_pool=pool,
                      n_replay_per_epoch=n_rep, kl_pool=kl_pool,
                      kl_weight=args.inert, kl_bs=args.inert_bs)
                if args.fold == "preimage":
                    for m in model.modules():
                        if isinstance(m, PreimageFill):
                            m.fold()
                else:
                    fold_into_anchors(model, frozen, master=master)
                rb, rt = verify_artifact(model, frozen)
                consolidations[-1]["recovery"] = dict(
                    epochs=args.recover_epochs, lr=args.recover_lr,
                    n_sentences=len(rec), invariance_violations=rb,
                    recalls=[eval_recall(model, tok, probes[k])
                             for k in range(t + 1)],
                    ppl=eval_ppl(model, tok, ppl_txt,
                                 max_chunks=args.max_ppl_chunks))
                print(f"[recover] recalls="
                      f"{['%.3f' % r for r in consolidations[-1]['recovery']['recalls']]}"
                      f" violations={rb}", flush=True)
            if t in val_at:
                # the whole point of the gate: what a major version costs on a
                # task metric, measured across the version bump rather than
                # inferred from perplexity
                sc = run_suite(model, tok, args.validate_tasks,
                               args.validate_limit,
                               f"after consolidating at task {t}",
                               bs=args.validate_bs,
                               samples_dir=args.validate_samples)
                consolidations[-1]["validation_after"] = sc
                validations.append(dict(point="after_consolidation",
                                        after_task=t, scores=sc))

    if args.save_merged:
        # The served function at the end of the sequence is the accumulated
        # anchor plus the live fill, so the factors alone would not reconstruct
        # it; what a downstream harness needs is the merged matrix.
        sp = Path(args.save_merged)
        sp.parent.mkdir(parents=True, exist_ok=True)
        served = {}
        with torch.no_grad():
            for name, mod in model.named_modules():
                if isinstance(mod, PreimageFill):
                    served[name] = mod.weight(torch.float32).half().cpu()
                elif isinstance(mod, BoundedFill):
                    # ask the layer for its own fill rather than reimplementing
                    # the link here, so the saved matrix is the served one
                    d = mod.fill(torch.float32)
                    w = mod.anchors.float() + d if mod.anchors is not None else d
                    served[name] = w.half().cpu()
        torch.save(served, sp)
        print(f"[save] served weights of {len(served)} matrices -> {sp}",
              flush=True)

    result = dict(
        config=vars(args),
        scorer=scorer_stamp(),
        bits_per_fact=bits_per_fact(),
        baseline=dict(ppl_anchor_4bit=ppl_anchor, ppl_anchor_lambada=ppl_anchor_x),
        history=history,
        consolidations=consolidations,
        validations=validations,
        minutes=round((time.time() - t0) / 60, 1),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"[done] {out} ({result['minutes']} min)")


if __name__ == "__main__":
    main()

"""Frozen-bin geometry: per-weight safe intervals and code-invariance checks."""

from __future__ import annotations

import torch

from .nf4 import NF4_LEVELS, NF4_MIDPOINTS, _EPS, assign_codes, safe_scale


def normalized_cell_edges(capped: bool = True):
    """Normalized (lo[16], hi[16]) cell edges per code.

    Voronoi edges are the midpoints between adjacent levels. The outermost
    cells (codes 0 and 15) are half-infinite; capped=True bounds them by
    mirroring the inner half-gap around the anchor, which is required for the
    refinement codec (a code must address a finite interval).
    """
    L, M = NF4_LEVELS, NF4_MIDPOINTS
    lo = torch.empty(16, dtype=torch.float32)
    hi = torch.empty(16, dtype=torch.float32)
    lo[1:] = M
    hi[:-1] = M
    if capped:
        lo[0] = L[0] - (M[0] - L[0])
        hi[15] = L[15] + (L[15] - M[14])
    else:
        lo[0] = float("-inf")
        hi[15] = float("inf")
    return lo, hi


def bin_bounds(
    codes: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int = 64,
    capped: bool = True,
    margin: float = 0.01,
):
    """Per-weight frozen interval [lo, hi], flat float32 tensors.

    margin shrinks each interval by margin·width per side, keeping stored
    values strictly off decision boundaries (tie safety). Weights kept in
    fp32/fp16 are safe at the default margin; bf16 checkpoint storage needs
    margin ≥ 0.05 to absorb its coarser rounding (see tests). margin only
    applies to capped bins; capped=False returns raw Voronoi cells.
    """
    lo_e, hi_e = normalized_cell_edges(capped)
    dev = absmax.device
    c = codes.long().view(-1)
    s = safe_scale(absmax).repeat_interleave(blocksize).to(dev)
    a, b = lo_e.to(dev)[c] * s, hi_e.to(dev)[c] * s
    # a negative scale flips the interval; the cell is still [min, max]
    lo, hi = torch.minimum(a, b), torch.maximum(a, b)
    if capped and margin:
        width = hi - lo
        lo = lo + margin * width
        hi = hi - margin * width
    return lo, hi


def anchor_absmax_floor(dtype, margin: float = 0.01) -> float:
    """The smallest block scale whose cells survive a round trip through an
    anchor buffer of `dtype`.

    A fold writes the moved weight back into that buffer, so invariance needs
    the margin to outlast the write:

        margin * NARROWEST * absmax  >  ulp(w) / 2,

    where NARROWEST is the width of the tightest NF4 cell in normalized units.
    In the format's normal range ulp(w)/2 <= 2^-(p+1) * absmax and the scale
    cancels, leaving a condition on the format alone -- fp16 clears it by 1.65x
    at the default margin, which is why fp16 anchors have held everywhere the
    blocks are normal. It stops cancelling in the subnormal range, where the
    spacing is a fixed tiny*eps instead of relative, and the condition becomes
    a floor on absmax: 3.70e-5 for fp16 at margin 0.01.

    Measured: Qwen3-8B-Base-bnb-4bit has blocks down to absmax 9.9e-7, and
    after one fold all 37,443 violating weights sat in blocks below this
    floor and none above it; the same fold with fp32 anchors violated
    nowhere. Qwen3-1.7B-Base-bnb-4bit serves no scale under 2.75e-3, which is
    why every archived 1.7B sequence reports zero and the floor went unseen.
    (37,443 is that controlled reproduction's count -- same weights, storage
    dtype the only variable. Do not conflate it with the 19,791 in the task-0
    log: that number came through the broken guard, which checked w_new in
    fp32 BEFORE the fp16 cast, so it undercounts with a bad instrument and
    supports only the guard-had-a-hole narrative, never a violation count.
    EXPERIMENTS.md lines ~945/1014 keep the two apart.)
    Note that under double quantization the served scale is not the block's
    max|w| -- it can be smaller, larger, or negative -- and it is the served
    one that the fill is computed against, so it is the one tested here.

    Raises when the format cannot hold the margin at any scale, which is the
    bf16 case at the default margin (3.9e-3 against 8.0e-4) -- there the fix is
    a wider margin, not a floor.
    """
    fi = torch.finfo(dtype)
    lo_e, hi_e = normalized_cell_edges(True)
    narrowest = float((hi_e - lo_e).min())
    if margin * narrowest <= fi.eps / 2:
        raise ValueError(
            f"{dtype} keeps {fi.eps / 2:.2e} relative, which exceeds the "
            f"{margin * narrowest:.2e} a margin of {margin} reserves in the "
            f"narrowest cell; no absmax floor makes this safe -- widen the "
            f"margin to more than {fi.eps / 2 / narrowest:.3f}")
    if fi.bits >= 32:
        return 0.0
    return (fi.tiny * fi.eps / 2) / (margin * narrowest)


def storable_mask(absmax, blocksize: int, numel: int, dtype,
                  margin: float = 0.01):
    """True where a weight's block scale clears anchor_absmax_floor(dtype).

    Weights below it are frozen rather than written: their room is zeroed, so
    the fold never moves them and the artifact stays invariant. The cost is
    nil in practice -- at 8B the blocks this catches have served scales at
    most 3.7e-5 against a median of 6.9e-2, so the room being given up is
    three orders of magnitude below the typical cell anyway.
    """
    floor = anchor_absmax_floor(dtype, margin)
    ok = absmax.float().abs() >= floor
    return ok.repeat_interleave(blocksize)[:numel]


def check_invariance(
    w_new: torch.Tensor,
    frozen_codes: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int = 64,
    writable=None,
):
    """Integer-domain invariance check under FROZEN scales.

    Never verify via bnb.quantize_4bit — it recomputes absmax and silently
    re-bins entire blocks. Returns (ok, n_mismatch, mismatch_flat_idx).

    `writable`, when given, is the per-weight storable mask and LIMITS THE
    CHECK'S DOMAIN TO IT. A block whose |served absmax| is below half the
    storage dtype's smallest subnormal stores every anchor as +-0 (measured
    on Qwen3-30B-A3B: 31 blocks at absmax -1.863e-08, all 16 levels
    collapsing to one fp16 value), so assign(stored anchor) cannot return
    the frozen code no matter what -- the weight never moved and the round
    trip is broken by storage, not by the fill. Those weights' bytes are
    carried verbatim through every version instead of being re-derived
    (see consolidate), which is what makes exempting them here honest
    rather than convenient.
    """
    new_codes = assign_codes(w_new, absmax, blocksize)
    frozen = frozen_codes.view(-1).to(new_codes.device)
    mismatch = new_codes != frozen
    if writable is not None:
        mismatch = mismatch & (writable.reshape(-1).to(mismatch.device) > 0)
    n = int(mismatch.sum().item())
    return n == 0, n, mismatch.nonzero().view(-1)


def normalized_halfwidth(
    capped: bool = True,
    margin: float = 0.01,
    dequant_bits: int = 8,
):
    """Per-code safe half-width, in units of the block's absmax: HW[16].

    The dense half-width tensor is redundant. Both cell edges and the anchor
    are the block's absmax times a per-code constant, so

        halfwidth[i] = HW[codes[i]] * absmax[block(i)]

    exactly, and HW has sixteen entries. Storing HW instead of the dense
    tensor is what makes CellFill fit at 27B, where the dense form is 48.7 GB
    in bf16 (see BoundedFill).

    One correction is needed. The stored anchor is bitsandbytes' dequantized
    value, not the exact product NF4_LEVELS[c] * absmax, and it differs by
    the dequant dtype's rounding. Reconstructing from the exact product could
    therefore claim up to that much more room than the weight actually has,
    which would let a fill cross a cell wall. We subtract the rounding bound,
    2^-dequant_bits * |level|, so the reconstruction is never larger than the
    true room. The default assumes a bf16 dequant (8 bits of mantissa); pass
    a larger value for fp16/fp32 to recover the room this gives up.
    """
    lo_e, hi_e = normalized_cell_edges(capped)
    if capped and margin:
        w = hi_e - lo_e
        lo_e = lo_e + margin * w
        hi_e = hi_e - margin * w
    L = NF4_LEVELS
    hw = torch.minimum(L - lo_e, hi_e - L)
    return (hw - (2.0**-dequant_bits) * L.abs()).clamp_min(0.0)


LAYOUTS = ("split_high", "split_low", "interleave_high", "interleave_low")


def unpack_nf4_codes(packed: torch.Tensor, numel: int, layout: str):
    """Recover 4-bit codes from bitsandbytes' packed uint8 weight storage.

    Two codes per byte, but which two is a bitsandbytes internal. It pairs
    element i with element i + n/2 (``split``), not with element i+1
    (``interleave``), and either may occupy the high nibble. Callers must
    calibrate against assign_codes rather than assume -- see nibble_layout_of,
    and note that a wrong layout still matches on some layers by coincidence,
    so calibration has to be checked per layer.
    """
    p = packed.reshape(-1)
    hi, lo = (p >> 4) & 0xF, p & 0xF
    if layout == "split_high":
        out = torch.cat([hi, lo])
    elif layout == "split_low":
        out = torch.cat([lo, hi])
    elif layout == "interleave_high":
        out = torch.stack([hi, lo], dim=-1).reshape(-1)
    elif layout == "interleave_low":
        out = torch.stack([lo, hi], dim=-1).reshape(-1)
    else:
        raise ValueError(f"unknown layout {layout!r}")
    return out[:numel]


def nibble_layout_of(packed: torch.Tensor, codes: torch.Tensor) -> str:
    """Which packing layout bitsandbytes used, determined by comparison.

    Recovering the layout from the data rather than assuming it means a
    change in the bitsandbytes storage format surfaces here, as a failure,
    instead of as silently wrong cell bounds.
    """
    ref = codes.reshape(-1)
    for layout in LAYOUTS:
        got = unpack_nf4_codes(packed, ref.numel(), layout)
        if torch.equal(got.to(ref.device).to(ref.dtype), ref):
            return layout
    raise RuntimeError(
        "cannot match bitsandbytes' packed nibbles to the recovered codes; "
        "the 4-bit storage layout is not one cellfill.bins knows"
    )

"""Calibrated re-quantization for the major version.

`consolidate` re-quantizes the accumulated weights with round-to-nearest under
a scale pinned to the block's max |w|. That is the crudest quantizer there is,
and the archive prices it: five consolidation events, mean suite accuracy
-0.799 +- 0.238, all five negative, and the cost does NOT scale with how much
of the file is rewritten (1.2% of codes changed costs as much as 34.6%). A
damage proportional to change would scale; a fixed per-weight quantizer error
would not. That is the fingerprint this module is built against.

Two levers, in increasing order of cost and of expected return.

SCALE SEARCH (data-free). RTN pins the scale to max|w| so that nothing clips.
Shrinking it clips the extremes and buys resolution for everyone else, and on
this project's weight distributions the trade is worth 7-12% of the squared
error. One parameter per block, chosen by exhaustive search over a small grid,
deterministic, and it cannot make a block worse because gamma = 1 is in the
grid.

HESSIAN WEIGHTING (needs calibration activations). The quantity that decides
whether the model survives is not ||W - W_hat|| but ||(W - W_hat) X||, and
those differ by however anisotropic the input second moment is. Given a
diagonal approximation d_j = E[x_j^2] collected over a calibration set, the
per-column weighting turns the same search into a search on the objective that
matters. Diagonal rather than full because a full Hessian per layer is a
different engineering project and the diagonal is what AWQ-style methods show
captures most of the gain.

WHAT THIS MODULE DOES NOT DO. It does not reorder or propagate error across
weights within a block (GPTQ's inner loop). That is the third lever and it
changes the code assignment of weight i based on the residual of weight i-1,
which interacts with the cell bound in ways that need their own analysis: a
weight pushed off its round-to-nearest code sits closer to a wall, and the
room it has left is what this whole paper is about.
"""

from __future__ import annotations

import torch

from .nf4 import assign_codes, compute_absmax, dequantize_ref

# gamma = 1.0 first so ties go to RTN and the search can only improve.
DEFAULT_GRID = (1.0, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.92, 0.90,
                0.87, 0.84, 0.80, 0.75, 0.70)


def _block_err(w_flat, deq_flat, weight_flat=None):
    """Squared error per block, optionally weighted per column."""
    d = (w_flat - deq_flat) ** 2
    if weight_flat is not None:
        d = d * weight_flat
    return d.sum(dim=1)


def calibrated_quantize(w: torch.Tensor, blocksize: int = 64,
                        grid=DEFAULT_GRID, col_weight: torch.Tensor | None = None,
                        floor: float | None = None):
    """(absmax, codes, dequantized, gamma) minimising per-block error.

    `col_weight` is an optional per-input-column weight of the same width as
    `w`'s last dimension -- E[x_j^2] from a calibration pass -- which turns the
    objective from weight error into output error. Passing None recovers the
    unweighted search.

    `floor` is the writable-cell floor for the anchor's storage dtype, and
    passing it is not optional in any comparison that matters. Shrinking a
    block's scale is exactly what pushes it under that floor, where its weights
    are frozen and keep zero room. At 1.7B nothing is near the floor -- the
    smallest served scale is 2.75e-3 against a floor of 3.70e-5, and even
    gamma = 0.70 leaves two orders of headroom -- so the search is free. At 8B
    it is not: 4,253 blocks are already below the floor under plain
    round-to-nearest, and a search that ignores it would freeze MORE weights in
    the calibrated arm than in the RTN arm. The comparison would then differ in
    two things at once and the difference would be unattributable. Clamping the
    grid per block makes the frozen population identical in both arms by
    construction rather than by inspection.

    The returned absmax is `gamma * max|w|` per block, so every downstream
    consumer (bin_bounds, assign_codes, the invariance check) sees an ordinary
    scale and needs no knowledge that a search happened.
    """
    base = compute_absmax(w, blocksize)
    # per-block lower bound on gamma: never take a block under the floor that
    # it already clears. A block already below it keeps gamma = 1 and is
    # frozen in both arms alike.
    if floor is not None:
        gmin = torch.where(base > floor, (floor / base.clamp_min(1e-30)),
                           torch.ones_like(base))
    else:
        gmin = None
    flat = w.reshape(-1, blocksize)
    wt = None
    if col_weight is not None:
        # broadcast the column weight onto the block layout
        wt = col_weight.reshape(1, -1).expand(w.shape[0], -1).reshape(-1, blocksize)
    best_err = torch.full((flat.shape[0],), float("inf"), device=w.device)
    best_gamma = torch.ones(flat.shape[0], device=w.device)
    for g in grid:
        gv = torch.full_like(best_gamma, g)
        if gmin is not None:
            # a block may not be taken below its own floor-clearing gamma
            gv = torch.maximum(gv, gmin)
        a = base * gv
        c = assign_codes(w, a, blocksize)
        d = dequantize_ref(c, a, blocksize, shape=tuple(w.shape)).reshape(-1, blocksize)
        e = _block_err(flat, d, wt)
        m = e < best_err
        best_err = torch.where(m, e, best_err)
        best_gamma = torch.where(m, gv, best_gamma)
    absmax = base * best_gamma
    codes = assign_codes(w, absmax, blocksize)
    deq = dequantize_ref(codes, absmax, blocksize, shape=tuple(w.shape))
    if floor is not None:
        # positive witness: the search never created a below-floor block
        made = int(((base >= floor) & (absmax < floor)).sum())
        assert made == 0, (
            f"the scale search pushed {made} blocks under the writable-cell "
            f"floor; the calibrated arm would freeze weights the RTN arm does "
            f"not and the comparison would carry two variables")
    return absmax, codes, deq, best_gamma


def compare(w: torch.Tensor, blocksize: int = 64, grid=DEFAULT_GRID,
            col_weight: torch.Tensor | None = None) -> dict:
    """RTN against the search on the same weights. Returns both errors."""
    a0 = compute_absmax(w, blocksize)
    c0 = assign_codes(w, a0, blocksize)
    d0 = dequantize_ref(c0, a0, blocksize, shape=tuple(w.shape))
    a1, c1, d1, g = calibrated_quantize(w, blocksize, grid, col_weight)
    e0 = float(((w - d0) ** 2).mean())
    e1 = float(((w - d1) ** 2).mean())
    return {"mse_rtn": e0, "mse_calibrated": e1,
            "reduction": (e0 - e1) / e0 if e0 else 0.0,
            "codes_differ": float((c0 != c1).float().mean()),
            "gamma_median": float(g.median()),
            "blocks_shrunk": float((g < 1).float().mean())}

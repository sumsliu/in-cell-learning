"""Clip-merge: project an arbitrary weight delta into the frozen bins.

This is the A-path core: train a LoRA however you like, materialize its dense
delta layer by layer, and clamp anchors+delta into the frozen bins. The
result re-quantizes bit-identically to the released 4-bit artifact.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ClipStats:
    n_total: int
    n_clipped: int
    clipped_frac: float
    delta_norm: float
    kept_norm: float
    norm_kept_frac: float  # ‖projected update‖ / ‖proposed update‖
    max_excess_halfwidths: float  # largest clipped-away distance, in bin half-widths


def clip_merge(
    anchors: torch.Tensor,
    delta: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
):
    """w_new = clamp(anchors + delta, lo, hi). Flat float32; returns (w_new, stats)."""
    anchors = anchors.reshape(-1).float()
    delta = delta.reshape(-1).float()
    target = anchors + delta
    w_new = torch.clamp(target, lo, hi)

    excess = (target - w_new).abs()
    clipped = excess > 0
    halfwidth = (hi - lo) / 2
    kept = w_new - anchors
    delta_norm = float(delta.norm())
    kept_norm = float(kept.norm())

    stats = ClipStats(
        n_total=target.numel(),
        n_clipped=int(clipped.sum().item()),
        clipped_frac=float(clipped.float().mean().item()),
        delta_norm=delta_norm,
        kept_norm=kept_norm,
        norm_kept_frac=kept_norm / delta_norm if delta_norm > 0 else 1.0,
        max_excess_halfwidths=float((excess / halfwidth).max().item())
        if target.numel()
        else 0.0,
    )
    return w_new, stats

"""k-bit refinement codec: the residual's position within its frozen bin.

Floor/center scheme: each bin splits into 2^k equal sub-cells; the code is the
sub-cell index, reconstruction is the sub-cell center. Properties:

- nested (Matryoshka): r_k == r_{k+j} >> j, so a k-bit stream is a prefix of
  any deeper one and "drop low bits" degrades gracefully;
- reconstruction is strictly interior to the bin (never touches a decision
  boundary), so unpacking always preserves the frozen 4-bit codes;
- max reconstruction error = bin_width / 2^(k+1).
"""

from __future__ import annotations

import torch


def pack_residual(
    w: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, k: int
) -> torch.Tensor:
    """Encode w's position within [lo, hi] as k bits (uint8 storage, 1 ≤ k ≤ 8)."""
    if not 1 <= k <= 8:
        raise ValueError(f"k must be in [1, 8], got {k}")
    width = hi - lo
    x = ((w.reshape(-1).float() - lo) / width).clamp(0.0, 1.0)
    r = (x * (1 << k)).floor().clamp(max=(1 << k) - 1)
    return r.to(torch.uint8)


def unpack_residual(
    r: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, k: int
) -> torch.Tensor:
    """Reconstruct at sub-cell centers, float32."""
    if not 1 <= k <= 8:
        raise ValueError(f"k must be in [1, 8], got {k}")
    width = hi - lo
    return lo + (r.float() + 0.5) * width / (1 << k)

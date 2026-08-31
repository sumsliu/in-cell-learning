"""NF4 constants and a reference block quantizer.

The 16 NF4 levels are the exact float32 constants baked into bitsandbytes'
kernels (QLoRA, Dettmers et al. 2023). Code i (0..15) maps to LEVELS[i];
levels are ascending, code 7 is exactly 0.0.

This module is the *verification* layer, not a replacement for bnb's kernels:
bnb's quantize_4bit recomputes absmax and therefore must never be used for
invariance checks. What bnb does not expose — code assignment under a FROZEN
absmax — is what `assign_codes` provides. Pure PyTorch, runs on CPU.
"""

from __future__ import annotations

import torch

NF4_LEVELS = torch.tensor(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ],
    dtype=torch.float32,
)

# Decision boundaries for round-to-nearest: midpoints between adjacent levels.
NF4_MIDPOINTS = (NF4_LEVELS[1:] + NF4_LEVELS[:-1]) / 2  # 15 values

# Guard against a zero scale (an all-zero block). The scale is SIGNED: a
# published double-quantized release dequantizes its per-block absmax from an
# 8-bit code plus an offset, and blocks whose weights are all near zero come
# back slightly negative (Qwen3-8B: 864,116 weights in such blocks, absmax
# about -5e-5). bitsandbytes decodes them as level[code] * absmax regardless
# of sign, so the cell of such a weight is the level interval scaled by a
# negative number -- flipped, not collapsed. Every function here divides and
# multiplies by the signed scale and only floors its magnitude.
_EPS = 1e-12


def safe_scale(absmax: torch.Tensor) -> torch.Tensor:
    """The signed per-block scale with |scale| >= _EPS, float32."""
    s = absmax.float()
    return torch.where(s.abs() < _EPS, torch.full_like(s, _EPS), s)


def block_view(w: torch.Tensor, blocksize: int) -> torch.Tensor:
    flat = w.reshape(-1)
    if flat.numel() % blocksize != 0:
        raise ValueError(
            f"numel {flat.numel()} is not a multiple of blocksize {blocksize}"
        )
    return flat.view(-1, blocksize)


def compute_absmax(w: torch.Tensor, blocksize: int = 64) -> torch.Tensor:
    """Per-block absmax scale, float32. This is the quantity that gets FROZEN."""
    return block_view(w, blocksize).abs().amax(dim=1).float()


def assign_codes(
    w: torch.Tensor, absmax: torch.Tensor, blocksize: int = 64
) -> torch.Tensor:
    """Round-to-nearest NF4 code assignment under a FROZEN absmax.

    Ties at exact decision boundaries round toward the lower code; the codec
    and clip-merge keep stored values strictly off boundaries via margins, so
    tie behavior never carries information.

    Returns flat uint8 codes, one per weight.
    """
    blocks = block_view(w, blocksize).float()
    scale = safe_scale(absmax).unsqueeze(1).to(blocks.device)
    normalized = blocks / scale
    codes = torch.bucketize(normalized, NF4_MIDPOINTS.to(blocks.device))
    return codes.to(torch.uint8).view(-1)


def quantize_ref(w: torch.Tensor, blocksize: int = 64):
    """Reference NF4 quantization mirroring bnb semantics.

    Returns (codes, absmax): flat uint8 codes and per-block float32 scales.
    """
    absmax = compute_absmax(w, blocksize)
    codes = assign_codes(w, absmax, blocksize)
    return codes, absmax


def dequantize_ref(
    codes: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int = 64,
    shape: tuple | None = None,
) -> torch.Tensor:
    """Anchors Ŵ = level[code] · absmax, float32."""
    levels = NF4_LEVELS.to(absmax.device)
    anchors = levels[codes.long()].view(-1, blocksize) * absmax.float().unsqueeze(1)
    anchors = anchors.reshape(-1)
    return anchors.view(shape) if shape is not None else anchors

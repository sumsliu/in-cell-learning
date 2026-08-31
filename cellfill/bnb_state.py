"""Bridge to bitsandbytes on GPU: extract the frozen artifact from Linear4bit.

GPU/CUDA only — nothing else in cellfill imports bitsandbytes. The frozen artifact
is (codes, absmax); every later invariance check and bin computation goes
through cellfill.nf4 / cellfill.bins under that frozen state.

Codes are recovered by frozen-scale assignment on the dequantized anchors
rather than by unpacking bnb's nibble layout: anchors are strictly interior to
their own cells, so the recovery is exact, and a reconstruction self-check
guards against any semantic drift between bnb versions.
"""

from __future__ import annotations

import torch

from .nf4 import assign_codes, dequantize_ref


def frozen_state_from_linear4bit(layer) -> dict:
    """Extract {anchors, absmax, codes, blocksize, shape} from a bnb Linear4bit."""
    from bitsandbytes.functional import dequantize_4bit, dequantize_blockwise

    qs = layer.weight.quant_state
    if qs.quant_type != "nf4":
        raise ValueError(f"expected nf4 quantization, got {qs.quant_type}")

    anchors = dequantize_4bit(layer.weight.data, qs).float()  # (out, in)

    if getattr(qs, "nested", False):
        # double quantization: absmax itself is blockwise-quantized
        absmax = dequantize_blockwise(qs.absmax, qs.state2) + qs.offset
    else:
        absmax = qs.absmax
    absmax = absmax.float()

    blocksize = qs.blocksize
    codes = assign_codes(anchors, absmax, blocksize)

    # Self-check: our (codes, absmax) must reproduce bnb's anchors up to the
    # dequant dtype's rounding. The tolerance has to be loose enough for bf16
    # rounding at the largest absmax in the layer (2^-8 relative, and the
    # scales themselves are stored in bf16 under double quantization) but far
    # tighter than a level spacing (~0.07 x absmax for NF4), which is what an
    # actual code mismatch would produce.
    recon = dequantize_ref(codes, absmax, blocksize, shape=tuple(anchors.shape))
    err = (recon - anchors).abs().max()
    tol = 2**-6 * absmax.max()
    if err > tol:
        raise RuntimeError(
            f"anchor reconstruction mismatch: max err {err:.3e} > tol {tol:.3e} "
            f"(a real code mismatch would be ~{0.07 * absmax.max():.3e})"
        )

    return {
        "anchors": anchors,
        "absmax": absmax,
        "codes": codes,
        "blocksize": blocksize,
        "shape": tuple(qs.shape),
    }

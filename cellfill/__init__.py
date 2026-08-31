"""cellfill — quantization-invariant continual learning core.

Frozen 4-bit anchors (codes + scales) define per-weight safe intervals;
everything here is the pure-math layer that bitsandbytes does not provide:
bin geometry, frozen-scale re-assignment, clip-merge, refinement codec.
Pure PyTorch, CPU-testable. GPU bridging lives in cellfill.bnb_state.
"""

from .nf4 import (
    NF4_LEVELS,
    NF4_MIDPOINTS,
    compute_absmax,
    assign_codes,
    quantize_ref,
    dequantize_ref,
)
from .bins import normalized_cell_edges, bin_bounds, check_invariance
from .merge import clip_merge, ClipStats
from .codec import pack_residual, unpack_residual

__version__ = "0.0.1"

__all__ = [
    "NF4_LEVELS",
    "NF4_MIDPOINTS",
    "compute_absmax",
    "assign_codes",
    "quantize_ref",
    "dequantize_ref",
    "normalized_cell_edges",
    "bin_bounds",
    "check_invariance",
    "clip_merge",
    "ClipStats",
    "pack_residual",
    "unpack_residual",
]

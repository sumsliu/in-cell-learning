"""Cells of a uniform integer grid (GPTQ / AWQ / QAT W4A16 releases).

NF4 is one rounding quantizer; the other family in deployment is the uniform
grid of compressed-tensors, GPTQ and AWQ: a weight is stored as an integer
code q in [qmin, qmax] and a scale s shared by a group of consecutive input
columns, optionally with a zero point z, and served as

    w_hat = (q - z) * s .

Round-to-nearest requantization sends w to clamp(round(w / s) + z). The cell
of a code is therefore the interval of width s centred on its anchor -- the
same picture as NF4 with the level table replaced by the integers -- and an
update that stays strictly inside it leaves the packed artifact bit-identical.
The two end codes own half-open cells (anything beyond the range clamps onto
them), so a half-width of s/2 is conservative everywhere.

Packed int32 storage and the decompression arithmetic are compressed-tensors'
own (unpack_from_int32); this module only adds the cell geometry and the
invariance check, in the integer domain under the frozen scales, exactly as
cellfill.bins does for NF4.
"""

from __future__ import annotations

import torch

_EPS = 1e-12


def unpack_codes(weight_packed: torch.Tensor, num_bits: int,
                 shape: tuple[int, int]) -> torch.Tensor:
    """Signed integer codes (int8) of shape `shape` from packed int32."""
    from compressed_tensors.compressors.pack_quantized.helpers import (
        unpack_from_int32,
    )

    return unpack_from_int32(weight_packed, num_bits, torch.Size(shape))


def group_size_of(shape: tuple[int, int], scale: torch.Tensor) -> int:
    out_f, in_f = shape
    if scale.ndim != 2 or scale.shape[0] != out_f or in_f % scale.shape[1]:
        raise ValueError(f"scale {tuple(scale.shape)} does not tile a "
                         f"{shape} weight along its input dimension")
    return in_f // scale.shape[1]


def expand_scale(scale: torch.Tensor, group: int) -> torch.Tensor:
    """(out, in/group) -> (out, in), one scale per weight."""
    return scale.repeat_interleave(group, dim=1)


def uniform_anchors(codes: torch.Tensor, scale: torch.Tensor, group: int,
                    zero_point: torch.Tensor | None = None) -> torch.Tensor:
    """The served weight (q - z) * s in float32, shape (out, in)."""
    q = codes.float()
    if zero_point is not None:
        q = q - expand_scale(zero_point.float(), group)
    return q * expand_scale(scale.float(), group)


def uniform_assign(w: torch.Tensor, scale: torch.Tensor, group: int,
                   qmin: int, qmax: int,
                   zero_point: torch.Tensor | None = None) -> torch.Tensor:
    """Requantize under FROZEN scales: clamp(round(w / s) + z)."""
    s = expand_scale(scale.float(), group).clamp_min(_EPS)
    q = torch.round(w.float() / s)
    if zero_point is not None:
        q = q + expand_scale(zero_point.float(), group)
    return q.clamp(qmin, qmax).to(torch.int8)


def uniform_halfwidth(scale: torch.Tensor, group: int, margin: float = 0.05,
                      dtype=torch.float32) -> torch.Tensor:
    """Safe half-width per weight, (out, in): s * (1/2 - margin).

    margin keeps the update off the rounding boundary and absorbs the
    rounding of the stored anchor: a bf16 anchor differs from the exact
    product (q - z) * s by at most 2^-8 * |w| <= 2^-8 * 8 s = s/32 ~ 0.031 s
    for 4-bit codes, so margin = 0.05 leaves 0.019 s of room after it, and
    fp16 storage of the served weight (2^-11 relative) costs another 0.004 s
    at most. The default is therefore not tunable downward without redoing
    that arithmetic.
    """
    if not 0 < margin < 0.5:
        raise ValueError("margin must lie in (0, 0.5)")
    return (expand_scale(scale.to(dtype), group) * (0.5 - margin))


def check_invariance_uniform(w_new: torch.Tensor, codes: torch.Tensor,
                             scale: torch.Tensor, group: int, qmin: int,
                             qmax: int,
                             zero_point: torch.Tensor | None = None):
    """Integer-domain check under frozen scales. Returns
    (ok, n_mismatch, mismatch_flat_idx), like cellfill.bins.check_invariance."""
    new = uniform_assign(w_new.reshape(codes.shape), scale, group, qmin, qmax,
                         zero_point)
    mismatch = new != codes.to(new.device).to(torch.int8)
    n = int(mismatch.sum().item())
    return n == 0, n, mismatch.reshape(-1).nonzero().view(-1)


def frozen_state_from_compressed(layer) -> dict:
    """Extract the frozen artifact from a compressed-tensors linear: the packed
    codes, scales, optional zero points, and the grid's integer range."""
    scheme = getattr(layer, "quantization_scheme", None)
    if scheme is None or scheme.weights is None:
        raise ValueError("layer carries no compressed-tensors weight scheme")
    args = scheme.weights
    num_bits = int(args.num_bits)
    qtype = getattr(args.type, "value", args.type)     # enum or plain str
    if str(qtype) != "int":
        raise ValueError(f"expected an integer grid, got {args.type}")
    packed = layer.weight_packed.data
    scale = layer.weight_scale.data
    shape = tuple(int(v) for v in layer.weight_shape.tolist()) \
        if hasattr(layer, "weight_shape") else (layer.out_features,
                                               layer.in_features)
    zp = getattr(layer, "weight_zero_point", None)
    if zp is not None:
        zp = zp.data
        if zp.dtype == torch.int32:   # packed along dim 0 (see CT compress)
            from compressed_tensors.compressors.pack_quantized.helpers import (
                unpack_from_int32,
            )
            zp = unpack_from_int32(zp, num_bits,
                                   torch.Size((shape[0], scale.shape[1])),
                                   packed_dim=0)
    symmetric = bool(args.symmetric)
    # compressed-tensors keeps the signed range for both conventions, with
    # the zero point on that range too (verified against its quantize()).
    qmin, qmax = -(2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1
    group = group_size_of(shape, scale)
    return dict(packed=packed, scale=scale, zero_point=zp, shape=shape,
                num_bits=num_bits, group=group, qmin=qmin, qmax=qmax,
                symmetric=symmetric)


class PackedLinear(torch.nn.Module):
    """A linear layer that keeps a pack-quantized weight packed.

    transformers loads a compressed-tensors model with its weights packed and
    then decompresses the whole model to dense on the first forward, which
    is 62 GB at 31B. This module decompresses one layer at a time, inside
    its own forward, and frees the dense weight afterwards: the packed codes
    and scales are all that stay resident, half a byte per weight. It is the
    uniform-grid counterpart of bitsandbytes' Linear4bit, and BoundedFill
    wraps it the same way.
    """

    def __init__(self, state: dict, bias=None):
        super().__init__()
        self.register_buffer("packed", state["packed"])
        self.register_buffer("scale", state["scale"])
        zp = state.get("zero_point")
        if zp is not None:
            self.register_buffer("zero_point", zp.to(torch.int8))
        else:
            self.zero_point = None
        self.register_buffer("bias", None if bias is None else bias.data)
        self.shape = tuple(state["shape"])
        self.out_features, self.in_features = self.shape
        self.num_bits = int(state["num_bits"])
        self.group = int(state["group"])
        self.qmin, self.qmax = int(state["qmin"]), int(state["qmax"])

    def codes(self) -> torch.Tensor:
        return unpack_codes(self.packed, self.num_bits, self.shape)

    def dequant(self, dtype=None) -> torch.Tensor:
        """The served weight, (out, in), recomputed from the packed codes."""
        w = uniform_anchors(self.codes(), self.scale, self.group,
                            self.zero_point)
        return w if dtype is None else w.to(dtype)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.dequant(x.dtype), self.bias)

    def extra_repr(self):
        return (f"in={self.in_features}, out={self.out_features}, "
                f"bits={self.num_bits}, group={self.group}, packed")


def pack_model_in_place(model) -> int:
    """Replace every compressed-tensors linear in a freshly loaded model with
    a PackedLinear, and drop the hook that would decompress everything.
    Returns the number of layers converted."""
    hook = getattr(model, "ct_decompress_hook", None)
    if hook is not None:
        hook.remove()
        delattr(model, "ct_decompress_hook")
    n = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if not hasattr(child, "weight_packed"):
                continue
            state = frozen_state_from_compressed(child)
            setattr(module, child_name,
                    PackedLinear(state, getattr(child, "bias", None)))
            n += 1
    return n

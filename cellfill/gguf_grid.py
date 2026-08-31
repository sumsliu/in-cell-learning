"""Cells of a GGUF (llama.cpp) release: the grids people actually run.

Every llama.cpp weight format used at four to eight bits is an affine
uniform grid per block: a stored integer code q and, per block (or
sub-block), a scale s and an offset o, with the served value w = s*q + o.
The cell of code q under frozen (s, o) is therefore the interval of width
|s| centred on the anchor, clamped at the code range, and the room is
|s|(1/2 - margin) for every weight. This module unpacks the codes and the
per-weight (s, o) from the raw blocks of each format, mirroring the
dequantizers of the `gguf` package bit for bit (checked by the tests and
by `verify_dequant`), so that the re-binning check and the fill's bound
are defined against the shipped file.

Formats: Q4_0, Q4_1, Q4_K, Q5_K, Q6_K, Q8_0 -- which together cover the
Q4_K_M mix that vendors publish (Q4_K for most matrices, Q6_K for the
attention value and feed-forward down projections of some layers).
"""

from __future__ import annotations

import numpy as np
import torch

QK_K = 256
_EPS = 1e-12


def _fp16(b):
    return b.view(np.float16).astype(np.float32)


def _scale_min_k(scales: np.ndarray):
    """The 6-bit sub-block scales and mins of Q4_K/Q5_K (gguf's unpacking)."""
    n = scales.shape[0]
    s = scales.view(np.uint8).reshape((n, 3, 4))
    d, m, m_d = np.split(s, 3, axis=-2)
    sc = np.concatenate([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], axis=-1)
    mn = np.concatenate([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], axis=-1)
    return sc.reshape((n, 8)).astype(np.float32), mn.reshape((n, 8)).astype(np.float32)


def unpack(blocks: np.ndarray, qtype: str):
    """Per-weight (codes, scale, offset, qmin, qmax) for one tensor's raw
    blocks, flat in the tensor's storage order."""
    n = blocks.shape[0]
    if qtype == "Q4_0":
        d, qs = np.hsplit(blocks, [2])
        d = _fp16(d)                                              # (n,1)
        q = (qs.reshape((n, 1, 1, 16)) >> np.array([0, 4], np.uint8).reshape((1, 1, 2, 1)))
        q = (q & 0x0F).reshape((n, 32)).astype(np.int32)
        s = np.repeat(d, 32, axis=1)
        o = -8.0 * s
        return q.reshape(-1), s.reshape(-1), o.reshape(-1), 0, 15
    if qtype == "Q4_1":
        d, rest = np.hsplit(blocks, [2])
        m, qs = np.hsplit(rest, [2])
        d, m = _fp16(d), _fp16(m)
        q = (qs.reshape((n, 1, 1, 16)) >> np.array([0, 4], np.uint8).reshape((1, 1, 2, 1)))
        q = (q & 0x0F).reshape((n, 32)).astype(np.int32)
        return (q.reshape(-1), np.repeat(d, 32, axis=1).reshape(-1),
                np.repeat(m, 32, axis=1).reshape(-1), 0, 15)
    if qtype == "Q8_0":
        d, x = np.split(blocks, [2], axis=1)
        d = _fp16(d)
        q = x.view(np.int8).astype(np.int32)                      # (n,32)
        s = np.repeat(d, 32, axis=1)
        return q.reshape(-1), s.reshape(-1), np.zeros_like(s).reshape(-1), -128, 127
    if qtype == "Q4_K":
        d, rest = np.hsplit(blocks, [2])
        dmin, rest = np.hsplit(rest, [2])
        scales, qs = np.hsplit(rest, [12])
        d, dmin = _fp16(d), _fp16(dmin)
        sc, mn = _scale_min_k(scales)
        s_sub = d * sc                                            # (n,8)
        o_sub = -(dmin * mn)
        q = (qs.reshape((n, 4, 1, 32)) >> np.array([0, 4], np.uint8).reshape((1, 1, 2, 1)))
        q = (q & 0x0F).reshape((n, 8, 32)).astype(np.int32)
        s = np.repeat(s_sub[:, :, None], 32, axis=2)
        o = np.repeat(o_sub[:, :, None], 32, axis=2)
        return q.reshape(-1), s.reshape(-1), o.reshape(-1), 0, 15
    if qtype == "Q5_K":
        d, rest = np.hsplit(blocks, [2])
        dmin, rest = np.hsplit(rest, [2])
        scales, rest = np.hsplit(rest, [12])
        qh, qs = np.hsplit(rest, [QK_K // 8])
        d, dmin = _fp16(d), _fp16(dmin)
        sc, mn = _scale_min_k(scales)
        s_sub, o_sub = d * sc, -(dmin * mn)
        ql = (qs.reshape((n, 4, 1, 32)) >> np.array([0, 4], np.uint8).reshape((1, 1, 2, 1)))
        ql = (ql & 0x0F).reshape((n, 8, 32))
        hb = (qh.reshape((n, 1, 1, 32)) >> np.arange(8, dtype=np.uint8).reshape((1, 1, 8, 1)))
        hb = (hb & 0x01).reshape((n, 8, 32))
        q = (ql | (hb << 4)).astype(np.int32)
        s = np.repeat(s_sub[:, :, None], 32, axis=2)
        o = np.repeat(o_sub[:, :, None], 32, axis=2)
        return q.reshape(-1), s.reshape(-1), o.reshape(-1), 0, 31
    if qtype == "Q6_K":
        ql, rest = np.hsplit(blocks, [QK_K // 2])
        qh, rest = np.hsplit(rest, [QK_K // 4])
        scales, d = np.hsplit(rest, [QK_K // 16])
        scales = scales.view(np.int8).astype(np.float32)        # (n,16)
        d = _fp16(d)                                             # (n,1)
        s_sub = d * scales                                       # (n,16)
        ql = (ql.reshape((n, 2, 1, 64)) >> np.array([0, 4], np.uint8).reshape((1, 1, 2, 1)))
        ql = (ql & 0x0F).reshape((n, 8, 32))
        hb = (qh.reshape((n, 2, 1, 32)) >> np.array([0, 2, 4, 6], np.uint8).reshape((1, 1, 4, 1)))
        hb = (hb & 0x03).reshape((n, 8, 32))
        q = ((ql | (hb << 4)).astype(np.int8) - np.int8(32)).astype(np.int32)
        q = q.reshape((n, 16, 16))
        s = np.repeat(s_sub[:, :, None], 16, axis=2)
        return q.reshape(-1), s.reshape(-1), np.zeros_like(s).reshape(-1), -32, 31
    raise ValueError(f"unsupported GGUF type {qtype}")


SUPPORTED = ("Q4_0", "Q4_1", "Q4_K", "Q5_K", "Q6_K", "Q8_0")
BLOCK_BYTES = {"Q4_0": 18, "Q4_1": 20, "Q8_0": 34, "Q4_K": 144, "Q5_K": 176,
               "Q6_K": 210}


def cells(codes, s, o, qmin, qmax, margin=0.01):
    """Per-weight anchors, bounds and room under frozen (s, o). The cell of
    code q is the interval of width |s| centred on s*q + o, shrunk by the
    margin on each side; a zero scale freezes its weights."""
    codes = torch.as_tensor(codes, dtype=torch.float32)
    s = torch.as_tensor(s, dtype=torch.float32)
    o = torch.as_tensor(o, dtype=torch.float32)
    anchor = s * codes + o
    half = s.abs() * (0.5 - margin)
    lo, hi = anchor - half, anchor + half
    room = torch.where(s.abs() < _EPS, torch.zeros_like(half), half)
    return anchor, lo, hi, room


def assign_codes(w, s, o, qmin, qmax):
    """Round-to-nearest under the frozen affine grid, as llama.cpp rounds
    (half away from the lower code), clamped to the code range."""
    w = torch.as_tensor(w, dtype=torch.float32)
    s = torch.as_tensor(s, dtype=torch.float32)
    o = torch.as_tensor(o, dtype=torch.float32)
    safe = torch.where(s.abs() < _EPS, torch.full_like(s, _EPS), s)
    q = torch.floor((w - o) / safe + 0.5)
    return q.clamp(qmin, qmax)


def codes_changed(w, codes, s, o, qmin, qmax):
    """How many served weights would re-bin to a code other than the stored
    one. A sub-block whose scale is zero maps every code to the same
    anchor and its codes cannot be recovered by re-binning; those weights
    are frozen (room 0) and count as unchanged iff the served value is
    the anchor itself."""
    w = torch.as_tensor(w, dtype=torch.float32)
    s = torch.as_tensor(s, dtype=torch.float32)
    o = torch.as_tensor(o, dtype=torch.float32)
    codes = torch.as_tensor(codes, dtype=torch.float32)
    new = assign_codes(w, s, o, qmin, qmax)
    zero = s.abs() < _EPS
    same = torch.where(zero, w == (s * codes + o), new == codes)
    return int((~same).sum().item())


def verify_dequant(blocks, qtype, reference):
    """The unpacked (codes, s, o) must reproduce the gguf package's
    dequantization exactly; returns the max absolute difference."""
    q, s, o, _, _ = unpack(blocks, qtype)
    w = s * q + o
    return float(np.max(np.abs(w - np.asarray(reference, dtype=np.float32).reshape(-1))))


def read_gguf(path):
    """{name: (qtype name, raw blocks (n_blocks, block_bytes), shape)} for
    every quantized tensor of a GGUF file, plus the dequantized reference
    for the check; shapes in torch order (out, in)."""
    from gguf import GGUFReader, GGMLQuantizationType, quants

    r = GGUFReader(path)
    out = {}
    for t in r.tensors:
        qt = GGMLQuantizationType(t.tensor_type).name
        shape = tuple(int(x) for x in reversed(list(t.shape)))
        if qt in SUPPORTED:
            # the reader hands back the raw bytes in a shape of its own;
            # the block structure is fixed by the type
            raw = np.asarray(t.data).reshape(-1).view(np.uint8)
            blocks = raw.reshape(-1, BLOCK_BYTES[qt])
            out[t.name] = (qt, blocks, shape)
        else:
            out[t.name] = (qt, None, shape)
    return r, out

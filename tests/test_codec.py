import pytest
import torch

from cellfill import (
    bin_bounds,
    check_invariance,
    pack_residual,
    quantize_ref,
    unpack_residual,
)

BS = 64


def setup(seed=0, n=32 * BS, scale=0.02):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(n, generator=g) * scale
    codes, absmax = quantize_ref(w, BS)
    lo, hi = bin_bounds(codes, absmax, BS, capped=True, margin=0.01)
    u = torch.rand(lo.shape, generator=torch.Generator().manual_seed(5))
    w_in = lo + u * (hi - lo)
    return codes, absmax, lo, hi, w_in


@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_roundtrip_error_bound_and_invariance(k):
    codes, absmax, lo, hi, w_in = setup()
    r = pack_residual(w_in, lo, hi, k)
    w_rec = unpack_residual(r, lo, hi, k)
    width = hi - lo
    max_err = (w_rec - w_in).abs()
    bound = width / (1 << (k + 1))
    assert bool((max_err <= bound * 1.001 + 1e-9).all())
    ok, n, _ = check_invariance(w_rec, codes, absmax, BS)
    assert ok, f"k={k}: {n} escapes after codec roundtrip"


def test_reconstruction_strictly_interior():
    _, _, lo, hi, w_in = setup()
    for k in (1, 4, 8):
        w_rec = unpack_residual(pack_residual(w_in, lo, hi, k), lo, hi, k)
        assert bool((w_rec > lo).all()) and bool((w_rec < hi).all())


def test_matryoshka_nesting():
    """r_k must equal r_{k+j} >> j: shallow codes are prefixes of deep ones."""
    _, _, lo, hi, w_in = setup()
    r8 = pack_residual(w_in, lo, hi, 8)
    r4 = pack_residual(w_in, lo, hi, 4)
    r2 = pack_residual(w_in, lo, hi, 2)
    assert torch.equal(r4, r8 >> 4)
    assert torch.equal(r2, r8 >> 6)
    assert torch.equal(r2, r4 >> 2)


def test_error_decreases_monotonically_in_k():
    _, _, lo, hi, w_in = setup()
    errs = []
    for k in range(1, 9):
        w_rec = unpack_residual(pack_residual(w_in, lo, hi, k), lo, hi, k)
        errs.append(float((w_rec - w_in).abs().mean()))
    assert all(errs[i + 1] < errs[i] for i in range(len(errs) - 1)), errs

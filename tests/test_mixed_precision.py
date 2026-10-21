import copy

import pytest
import torch

from hlens.functional import Fp32Unembed, make_mixed_slice_fn, make_slice_fn
from jlens import from_hf
from jlens.hooks import ActivationRecorder

from .tiny_qwen import _ByteTokenizer, tiny_hf
from .tiny_variants import tiny_gemma3_hf

TEXT = "the quick brown fox jumps over the lazy dog again and again"


def _pair(builder):
    hf32 = builder()
    hf16 = copy.deepcopy(hf32).to(torch.bfloat16)
    return from_hf(hf32, _ByteTokenizer()), from_hf(hf16, _ByteTokenizer())


@pytest.mark.parametrize("builder", [tiny_hf, tiny_gemma3_hf])
def test_mixed_slice_is_closer_to_fp32_than_a_bf16_slice(builder):
    m32, m16 = _pair(builder)
    ids = m32.encode(TEXT)
    with torch.no_grad(), ActivationRecorder(m32.layers, at=[0]) as rec:
        m32.forward(ids)
    h = rec.activations[0].detach()
    ref = make_slice_fn(m32, 0, 3)(h)
    mixed = make_mixed_slice_fn(m16, 0, 3)(h)
    plain16 = make_slice_fn(m16, 0, 3)(h.bfloat16()).float()
    assert mixed.dtype == torch.float32
    e_mixed, e_16 = (mixed - ref).norm() / ref.norm(), (plain16 - ref).norm() / ref.norm()
    # the only error left in the mixed path is the bf16 rounding of the weights, which a plain bf16 slice also has
    assert e_mixed < e_16
    assert e_mixed < 0.05


def test_fp32_weights_make_the_mixed_path_exact():
    m32, _ = _pair(tiny_hf)
    ids = m32.encode(TEXT)
    with torch.no_grad(), ActivationRecorder(m32.layers, at=[0]) as rec:
        m32.forward(ids)
    h = rec.activations[0].detach()
    torch.testing.assert_close(make_mixed_slice_fn(m32, 0, 3)(h), make_slice_fn(m32, 0, 3)(h), rtol=1e-5, atol=1e-5)


def test_mixed_slice_is_differentiable_in_fp32():
    _, m16 = _pair(tiny_hf)
    ids = m16.encode(TEXT)
    with torch.no_grad(), ActivationRecorder(m16.layers, at=[0]) as rec:
        m16.forward(ids)
    h = rec.activations[0].detach().float().requires_grad_(True)
    out = make_mixed_slice_fn(m16, 0, 3)(h)
    (g,) = torch.autograd.grad(out.sum(), h)
    assert g.dtype == torch.float32 and torch.isfinite(g).all() and g.norm() > 0


def test_fp32_unembed_matches_fp32_model():
    m32, m16 = _pair(tiny_hf)
    r = torch.randn(m32.d_model)
    torch.testing.assert_close(Fp32Unembed(m16).unembed(r).shape, m32.unembed(r).shape)
    # built from the fp32 model it is the same function
    torch.testing.assert_close(Fp32Unembed(m32).unembed(r), m32.unembed(r), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("builder", [tiny_hf, tiny_gemma3_hf])
def test_rotary_is_recomputed_in_fp32_and_differs_from_bf16(builder):
    from hlens.functional import _capture_block_kwargs, _fp32_rotary

    _, m16 = _pair(builder)
    kw = _capture_block_kwargs(m16, 2, 12)
    fresh = _fp32_rotary(m16, 2, 12, kw["position_ids"])
    assert fresh is not None and fresh[0].dtype == torch.float32
    assert fresh[0].shape == kw["position_embeddings"][0].shape
    assert (fresh[0] - kw["position_embeddings"][0].float()).abs().max() > 0  # the bf16 values were rounded


@pytest.mark.parametrize("builder", [tiny_hf, tiny_gemma3_hf])
def test_cold_cache_inside_func_transform_then_reuse(builder):
    """The first call of a fresh mixed slice may happen inside jvp/hvp; its cached kwargs must survive later transforms."""
    from hlens.hvp import hvp_fr, hvp_rr

    _, m16 = _pair(builder)
    ids = m16.encode(TEXT)
    with torch.no_grad(), ActivationRecorder(m16.layers, at=[0]) as rec:
        m16.forward(ids)
    base = rec.activations[0][0].detach().float()
    slice_fn = make_mixed_slice_fn(m16, 0, 3)

    def f(x):
        return slice_fn(torch.cat([base[:-1], x[None]], dim=0)[None])[0, -1].pow(2).sum()

    x = base[-1].clone().requires_grad_(True)
    v = torch.randn_like(x)
    a = hvp_fr(f, x, v)  # cold: first use is inside forward-mode AD
    b = hvp_rr(f, x, v)  # warm: cache reused under reverse-mode
    torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-4)

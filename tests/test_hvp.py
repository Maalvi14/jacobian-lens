import torch

import pytest

from hlens.hvp import hvp_rr, hvp_fr, hvp_fd, restricted_hessian, Cotangent, Logit, Norm, make_scalar_fn, hvp_norm_form

from hlens.contexts import at_position, capture_context, lens_direction_bank, logit_fn
from hlens.functional import make_slice_fn, make_output_fn
from hlens.jspace import unembed_directions
from jlens.hooks import ActivationRecorder

from .tiny import TinyDecoder

def test_hvp_rr_quadratic_form():
    torch.manual_seed(0)
    d = 6
    A = torch.randn(d, d)
    A = (A + A.T) / 2

    h = torch.randn(d, requires_grad=True)
    v = torch.randn(d)

    def f(h: torch.Tensor) -> torch.Tensor:
        return 0.5 * h @ A @ h

    hvp = hvp_rr(f, h, v)
    expected = A @ v

    torch.testing.assert_close(hvp, expected, rtol=1e-5, atol=1e-5)

def test_hvp_rr_on_hooked_activation():
    model = TinyDecoder(n_layers=4, d_model=8)
    input_ids = model.encode("the quick brown fox jumps over")

    l_start = 1
    with ActivationRecorder(model.layers, at=[l_start], start_graph_at=l_start) as recorder:
        model.forward(input_ids)

    h = recorder.activations[l_start]
    v = torch.randn_like(h)

    def f(h: torch.Tensor) -> torch.Tensor:
        return (h**2).sum()

    hvp = hvp_rr(f, h, v)
    expected = 2 * v

    torch.testing.assert_close(hvp, expected, rtol=1e-5, atol=1e-5)

def test_hvp_rr_and_hvp_fr_agree():
    torch.manual_seed(1)
    d = 6
    A = torch.randn(d, d)
    A = (A + A.T) / 2

    h = torch.randn(d, requires_grad=True)
    v = torch.randn(d)

    def f(h: torch.Tensor) -> torch.Tensor:
        return 0.5 * h @ A @ h

    hvp_via_rr = hvp_rr(f, h, v)
    hvp_via_fr = hvp_fr(f, h, v)
    expected = A @ v

    torch.testing.assert_close(hvp_via_rr, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(hvp_via_fr, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(hvp_via_rr, hvp_via_fr, rtol=1e-5, atol=1e-5)

def test_hvp_fd_matches_analytic():
    torch.manual_seed(2)
    d = 6
    A = torch.randn(d, d)
    A = (A + A.T) / 2

    h = torch.randn(d)
    v = torch.randn(d)
    
    def f(h: torch.Tensor) -> torch.Tensor:
        return 0.5 * h @ A @ h

    hvp = hvp_fd(f, h, v)
    expected = A @ v

    torch.testing.assert_close(hvp, expected, rtol=1e-4, atol=1e-4)

def test_restricted_hessian_quadratic_form():
    torch.manual_seed(3)
    d = 6
    k = 3
    A = torch.randn(d, d)
    A = (A + A.T) / 2

    h = torch.randn(d)
    directions = torch.randn(k, d)

    def f(h: torch.Tensor) -> torch.Tensor:
        return 0.5 * h @ A @ h
    
    result = restricted_hessian(f, h, directions)
    expected = directions @ A @ directions.T

    torch.testing.assert_close(result.H, expected, rtol=1e-4, atol=1e-4)
    assert result.asymmetry < 1e-4


def test_make_scalar_fn_cotangent():
    d = 4
    c = torch.randn(d)
    h = torch.randn(d)

    def f(h: torch.Tensor) -> torch.Tensor:
        return h * 2

    scalar_fn = make_scalar_fn(f, Cotangent(c=c))
    result = scalar_fn(h)
    expected = torch.dot(c, f(h))

    torch.testing.assert_close(result, expected)


def test_make_scalar_fn_logit():
    d = 4
    h = torch.randn(d)
    token_id = 2

    def f(h: torch.Tensor) -> torch.Tensor:
        return h * 3

    scalar_fn = make_scalar_fn(f, Logit(token_id=token_id))
    result = scalar_fn(h)
    expected = f(h)[token_id]

    torch.testing.assert_close(result, expected)


def test_make_scalar_fn_norm_raises():
    with pytest.raises(NotImplementedError):
        make_scalar_fn(lambda h: h, Norm())


def test_hvp_norm_form_on_linear_map_is_zero():
    torch.manual_seed(0)
    d = 5
    A = torch.randn(d, d)
    h = torch.randn(d)
    delta_h = torch.randn(d)

    def f(h: torch.Tensor) -> torch.Tensor:
        return A @ h

    result = hvp_norm_form(f, h, delta_h)
    torch.testing.assert_close(result, torch.zeros(d), rtol=0, atol=1e-5)


def test_hvp_norm_form_on_elementwise_square():
    torch.manual_seed(1)
    d = 5
    h = torch.randn(d)
    delta_h = torch.randn(d)

    def f(h: torch.Tensor) -> torch.Tensor:
        return h**2

    result = hvp_norm_form(f, h, delta_h)
    expected = 2 * delta_h**2
    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Real-model HVP checks (Qwen3.5-4B). Run with: pytest -m real_model -s
# ---------------------------------------------------------------------------

BOOT_PROMPT = "Fact: The currency used in the country shaped like a boot is"

# (l_start, l_end): the scalar f runs blocks l_start+1..l_end; l_end=None means
# "through the last block, then unembed, then one token's logit"
#   (2, 3)    -> a single full-attention layer
#   (3, 4)    -> a single linear-attention layer
#   (3, 12)   -> a mix of both types
#   (20, None)-> the tail of the network down to a real logit
CASES = [
    pytest.param(2, 3, id="full-attn-layer"),
    pytest.param(3, 4, id="linear-attn-layer"),
    pytest.param(3, 12, id="mixed"),
    pytest.param(20, None, id="logit-from-20"),
]


def _real_inputs(model, l_start):
    """Real residual at `l_start` for the boot prompt, plus the model's own
    top next-token id (used as the logit to differentiate)."""
    ids = model.encode(BOOT_PROMPT, max_length=64)
    last = model.n_layers - 1
    with torch.no_grad(), ActivationRecorder(
        model.layers, at=sorted({l_start, last})
    ) as rec:
        model.forward(ids)
        h = rec.activations[l_start].detach().clone()
        top_token = model.unembed(rec.activations[last][0, -1]).argmax().item()
    return h, top_token


def _directions(h, seed=0):
    """Unit-norm perturbation direction v (shaped like h) and unit-norm
    cotangent c over d_model, both fp32."""
    g = torch.Generator(device=h.device).manual_seed(seed)
    v = torch.randn(h.shape, generator=g, device=h.device)
    c = torch.randn(h.shape[-1], generator=g, device=h.device)
    return v / v.norm(), c / c.norm()


def _scalar_f(model, l_start, l_end, c, token_id, dtype):
    """0-dim scalar function of h (shape [1, seq, d_model]), as torch.func.grad
    requires. Everything is read at the last position."""
    if l_end is None:
        out_fn = make_output_fn(model, l_start, lambda x: model.unembed(x[0, -1]))
        return make_scalar_fn(out_fn, Logit(token_id))
    slice_fn = make_slice_fn(model, l_start, l_end)
    c = c.to(dtype)
    return lambda h: (slice_fn(h)[0, -1] * c).sum()


def _rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm()).item()


@pytest.mark.real_model
@pytest.mark.parametrize("l_start,l_end", CASES)
def test_hvp_estimators_agree_qwen_fp32(qwen_fp32, l_start, l_end):
    model = qwen_fp32.model
    h, token_id = _real_inputs(model, l_start)
    v, c = _directions(h)
    f = _scalar_f(model, l_start, l_end, c, token_id, h.dtype)

    rr = hvp_rr(f, h.clone().requires_grad_(True), v)
    fr = hvp_fr(f, h, v)
    fd = {eps: hvp_fd(f, h, v, eps) for eps in (1e-3, 1e-2)}

    print(f"\n[{l_start},{l_end}] |hvp_fr| = {fr.norm().item():.4e}")
    print(f"  rr vs fr: {_rel(rr, fr):.3e}")
    for eps, g in fd.items():
        print(f"  fd(eps={eps:g}) vs fr: {_rel(g, fr):.3e}")

    for name, t in [("rr", rr), ("fr", fr), *[(f"fd{e}", g) for e, g in fd.items()]]:
        assert torch.isfinite(t).all(), name
    assert fr.norm() > 0, "zero curvature: the test would be vacuous"
    assert _rel(rr, fr) < 1e-4


@pytest.mark.real_model
@pytest.mark.parametrize("l_start,l_end", CASES)
def test_hvp_bf16_noise_vs_fp32_qwen(qwen, qwen_fp32, l_start, l_end):
    h32, token_id = _real_inputs(qwen_fp32.model, l_start)
    v, c = _directions(h32)
    f32 = _scalar_f(qwen_fp32.model, l_start, l_end, c, token_id, torch.float32)
    ref = hvp_fr(f32, h32, v)

    bf = torch.bfloat16
    h16, v16 = h32.to(bf), v.to(bf)
    f16 = _scalar_f(qwen.model, l_start, l_end, c, token_id, bf)
    fr16 = hvp_fr(f16, h16, v16)
    fd16 = hvp_fd(f16, h16, v16, 1e-2)

    print(f"\n[{l_start},{l_end}] bf16 vs fp32 reference (|ref| = {ref.norm().item():.4e})")
    print(f"  hvp_fr bf16: {_rel(fr16, ref):.3e}")
    print(f"  hvp_fd bf16 (eps=1e-2): {_rel(fd16, ref):.3e}")

    assert torch.isfinite(fr16.float()).all()
    assert torch.isfinite(fd16.float()).all()


@pytest.mark.real_model
def test_hvp_fr_first_call_does_not_poison_kwargs_cache(qwen_fp32):
    # Regression: the first use of a fresh slice fn inside torch.func.jvp used to
    # cache transform-wrapped rotary tensors, which "escaped" on the next transform.
    # Layer 3 is full attention, so it actually reads the rotary cos/sin.
    model = qwen_fp32.model
    h, token_id = _real_inputs(model, 2)
    v, c = _directions(h)
    f = _scalar_f(model, 2, 3, c, token_id, h.dtype)  # fresh: cache is cold

    fr = hvp_fr(f, h, v)  # cold call, inside jvp(grad(...))
    fd = hvp_fd(f, h, v, 1e-2)  # reuses the cache under a new transform
    assert _rel(fd, fr) < 1e-2


# --- hvp_norm_form and restricted_hessian on the real model (fp32) -----------------


@pytest.mark.real_model
@pytest.mark.parametrize("l_start,l_end", [(2, 5), (3, 12)])
def test_hvp_norm_form_matches_second_difference_qwen_fp32(qwen_fp32, l_start, l_end):
    """hvp_norm_form returns the full contraction d^T H d (vector-valued, [1, seq,
    d_model]) of a vector-valued slice. Compare with the central second
    difference (f(h+e d) + f(h-e d) - 2 f(h)) / e^2, which converges to the same
    thing; fp32 rounding limits how small e can go, so take the best of a grid."""
    model = qwen_fp32.model
    h, _ = _real_inputs(model, l_start)
    v, _ = _directions(h)
    f = make_slice_fn(model, l_start, l_end)

    q = hvp_norm_form(f, h, v)
    errs = {}
    with torch.no_grad():
        f0 = f(h)
        for eps in (3e-2, 1e-1, 3e-1, 1.0):
            sd = (f(h + eps * v) + f(h - eps * v) - 2 * f0) / eps**2
            errs[eps] = _rel(sd, q)
    print(f"\n[{l_start},{l_end}] |h|={h.norm().item():.1f} |d^T H d|={q.norm().item():.3e}", flush=True)
    print("  second-difference rel err by eps:", {k: f"{e:.2e}" for k, e in errs.items()}, flush=True)
    assert q.norm() > 0
    assert min(errs.values()) < 0.05


@pytest.mark.real_model
def test_restricted_hessian_real_qwen_fp32(qwen_fp32):
    model = qwen_fp32.model
    layer = 16
    point = capture_context(
        model, "Fact: The currency used in the country shaped like a boot is", layer, position=-2
    )
    top = model.unembed(point.h[0, -1]).argmax().item()  # any fixed token works as the scalar
    f1, x0 = at_position(logit_fn(model, point, top), point)

    tokens = [qwen_fp32.tokenizer.encode(w, add_special_tokens=False)[0] for w in (" Italy", " Japan", " boot", " country", " currency", " France")]
    D = lens_direction_bank(unembed_directions(model, tokens).to(x0.device), qwen_fp32.lens, layer)

    rh = restricted_hessian(f1, x0, D)
    assert rh.H.device == x0.device and rh.H.dtype == x0.dtype

    # independent check of the diagonal: second difference of f along each direction
    eps = 0.3 * x0.norm().item() * 1e-1
    with torch.no_grad():
        f0 = f1(x0)
        diag_sd = torch.stack([(f1(x0 + eps * d) + f1(x0 - eps * d) - 2 * f0) / eps**2 for d in D])
    rel_asym = (rh.asymmetry / rh.H.abs().max()).item()
    print(f"\n|x0|={x0.norm().item():.1f}  H=\n{rh.H.cpu()}\n  asymmetry/|H|max={rel_asym:.2e}", flush=True)
    print("  diag H:", rh.H.diagonal().cpu().tolist(), "\n  diag 2nd diff:", diag_sd.cpu().tolist(), flush=True)
    assert torch.isfinite(rh.H).all()
    assert rel_asym < 1e-3
    assert _rel(diag_sd, rh.H.diagonal()) < 0.15

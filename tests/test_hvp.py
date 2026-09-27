import torch

import pytest

from hlens.hvp import hvp_rr, hvp_fr, hvp_fd, restricted_hessian, Cotangent, Logit, Norm, make_scalar_fn, hvp_norm_form

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
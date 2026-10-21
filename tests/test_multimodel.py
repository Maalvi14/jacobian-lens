"""Architecture-readiness checks for the second-model candidates, on tiny CPU models:
the functional slice must reproduce the hooked forward (per-block kwargs differ across
Gemma 3's sliding/global layers), and forward-over-reverse must agree with reverse-over-
reverse on those slices."""

import pytest
import torch

from hlens.contexts import capture_context
from hlens.functional import make_slice_fn
from hlens.hvp import hvp_fr, hvp_rr
from hlens.jspace import unembed_directions
from jlens.hooks import ActivationRecorder

from .tiny_variants import BUILDERS, tiny_wrapped

NAMES = list(BUILDERS)
TEXT = "the quick brown fox jumps over the lazy dog again"  # longer than the sliding window


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("l_start,l_end", [(0, 2), (1, 4), (0, 5)])
def test_slice_matches_hooked_forward(name, l_start, l_end):
    _, model = tiny_wrapped(name)
    if l_end >= model.n_layers:
        pytest.skip("model too shallow")
    ids = model.encode(TEXT)
    with torch.no_grad(), ActivationRecorder(model.layers, at=[l_start, l_end]) as rec:
        model.forward(ids)
    out = make_slice_fn(model, l_start, l_end)(rec.activations[l_start])
    torch.testing.assert_close(out, rec.activations[l_end], rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("name", NAMES)
def test_hvp_fr_matches_rr_on_slice(name):
    _, model = tiny_wrapped(name)
    ids = model.encode(TEXT)
    with torch.no_grad(), ActivationRecorder(model.layers, at=[1]) as rec:
        model.forward(ids)
    x = rec.activations[1][0, -1].clone().requires_grad_(True)
    slice_fn = make_slice_fn(model, 1, min(4, model.n_layers - 1))
    base = rec.activations[1][0].clone()

    def f(v):
        h = base.clone()
        h = torch.cat([h[:-1], v[None]], dim=0)[None]
        return slice_fn(h)[0, -1].pow(2).sum()

    v = torch.randn_like(x)
    torch.testing.assert_close(hvp_fr(f, x, v), hvp_rr(f, x, v), rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("name", NAMES)
def test_unembed_directions_finite_and_distinct(name):
    _, model = tiny_wrapped(name)
    u = unembed_directions(model, [1, 2, 3])
    assert u.shape == (3, model.d_model) and torch.isfinite(u).all()
    assert (u[0] - u[1]).norm() > 0


@pytest.mark.parametrize("name", NAMES)
def test_mlp_branch_matches_block_contribution(name):
    """x_after_mlp - x_before = mlp_branch(x_before) for every architecture."""
    from hlens.arch import mlp_branch, mlp_prenorm

    hf, model = tiny_wrapped(name)
    ids = model.encode(TEXT)
    store = {}
    block = model.layers[2]
    h1 = mlp_prenorm(block).register_forward_pre_hook(lambda m, a: store.__setitem__("x", a[0].detach()))
    with torch.no_grad(), ActivationRecorder(model.layers, at=[2]) as rec:
        model.forward(ids)
    h1.remove()
    # residual after attention (x) -> block output = x + branch(x)
    torch.testing.assert_close(rec.activations[2], store["x"] + mlp_branch(block)(store["x"]), rtol=1e-4, atol=1e-4)


def test_embed_rows_includes_gemma_scale():
    from hlens.arch import embed_rows

    hf, _ = tiny_wrapped("gemma3")
    emb = hf.get_input_embeddings()
    rows = embed_rows(emb, [3, 5])
    assert not torch.allclose(rows[0], emb.weight[3])  # scaled by sqrt(d_model)
    torch.testing.assert_close(rows[0], emb.weight[3] * emb.embed_scale, rtol=1e-5, atol=1e-5)

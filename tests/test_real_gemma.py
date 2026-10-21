"""Gemma 3 12B (pretrained) with its Neuronpedia lens: shapes, readout, and the fp32
slice / HVP checks the harness depends on. Run: pytest tests/test_real_gemma.py -m real_model -s"""

import gc

import pytest
import torch

from hlens.contexts import capture_context
from hlens.functional import make_slice_fn
from hlens.hvp import hvp_fr, hvp_rr
from jlens.hooks import ActivationRecorder

pytestmark = pytest.mark.real_model

BOOT = "Fact: The currency used in the country shaped like a boot is the"
PROMPT = "The quick brown fox jumps over the lazy dog while the sun sets slowly behind the distant hills"
# Gemma 3 is 5 sliding : 1 global. Layers 5, 11, ... are global.
SLICES = [(2, 3), (4, 6), (3, 12), (0, 46)]


def test_shapes_and_architecture(gemma_fp32):
    q = gemma_fp32
    assert next(q.hf.parameters()).dtype == torch.float32
    assert q.model.n_layers == 48 and q.model.d_model == 3840
    assert q.lens.source_layers == list(range(47)) and q.lens.d_model == 3840
    assert q.hf.config.get_text_config()._attn_implementation == "eager"
    kinds = q.hf.config.get_text_config().layer_types
    assert {"sliding_attention", "full_attention"} <= set(kinds)
    print(f"\nfp32 resident: {torch.cuda.memory_allocated() / 2**30:.1f} GiB")


def test_lens_reads_out_concepts(gemma_fp32):
    q = gemma_fp32
    tok = q.tokenizer
    lens_logits, model_logits, ids = q.lens.apply(q.model, BOOT, positions=[-1])
    final = [tok.decode([t]).strip().lower() for t in model_logits[0].topk(3).indices]
    late = [tok.decode([t]).strip().lower() for t in lens_logits[36][0].topk(5).indices]
    print(f"\nfinal {final} | lens@36 {late}")
    assert "euro" in final
    assert any(w.startswith("euro") for w in late), "lens at L36 should read the currency concept"


@pytest.mark.parametrize("l_start,l_end", SLICES)
def test_slice_matches_hooked_forward_fp32(gemma_fp32, l_start, l_end):
    model = gemma_fp32.model
    ids = model.encode(PROMPT, max_length=64)
    with torch.no_grad(), ActivationRecorder(model.layers, at=[l_start, l_end]) as rec:
        model.forward(ids)
        h_in, h_out = rec.activations[l_start].detach(), rec.activations[l_end].detach()
    with torch.no_grad():
        got = make_slice_fn(model, l_start, l_end)(h_in)
    diff = (got - h_out).abs().max().item()
    print(f"\nslice ({l_start},{l_end}): max |diff| = {diff:.3e}")
    # the prompt is longer than the sliding window? no (window 1024), so masks agree; require ~exact
    torch.testing.assert_close(got, h_out, rtol=1e-4, atol=1e-4)


def test_hvp_fr_matches_rr_on_real_slice(gemma_fp32):
    """Forward-over-reverse (the cheap estimator) must agree with reverse-over-reverse
    through a slice that contains both sliding and global layers."""
    model = gemma_fp32.model
    ids = model.encode(PROMPT, max_length=64)
    l0, l1 = 4, 7  # crosses the global layer 5
    with torch.no_grad(), ActivationRecorder(model.layers, at=[l0]) as rec:
        model.forward(ids)
    base = rec.activations[l0][0].detach().clone()
    slice_fn = make_slice_fn(model, l0, l1)
    out_dir = torch.randn(base.shape[-1], device=base.device, dtype=base.dtype)

    def f(x):
        h = torch.cat([base[:-1], x[None]], dim=0)[None]
        return slice_fn(h)[0, -1] @ out_dir

    x = base[-1].clone().requires_grad_(True)
    v = torch.randn_like(x)
    a, b = hvp_fr(f, x, v), hvp_rr(f, x, v)
    rel = ((a - b).norm() / b.norm()).item()
    print(f"\nhvp_fr vs hvp_rr relative difference: {rel:.2e}")
    assert rel < 1e-3
    assert b.norm() > 0

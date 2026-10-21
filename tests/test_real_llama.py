"""Llama 3.1 8B (base) with its Neuronpedia lens: shapes, readout, and the fp32 slice / HVP checks the harness depends on.
Run: pytest tests/test_real_llama.py -m real_model -s"""

import pytest
import torch

from hlens.arch import mlp_branch, mlp_prenorm
from hlens.functional import make_slice_fn
from hlens.hvp import hvp_fr, hvp_rr
from jlens.hooks import ActivationRecorder

pytestmark = pytest.mark.real_model

PROMPT = "The quick brown fox jumps over the lazy dog while the sun sets slowly behind the distant hills"
SLICES = [(2, 3), (4, 8), (3, 12), (0, 30)]


def test_shapes_and_architecture(llama_fp32):
    q = llama_fp32
    assert next(q.hf.parameters()).dtype == torch.float32
    assert q.model.n_layers == 32 and q.model.d_model == 4096
    assert q.lens.source_layers == list(range(31)) and q.lens.d_model == 4096
    assert q.hf.config.get_text_config()._attn_implementation == "eager"
    print(f"\nfp32 resident: {torch.cuda.memory_allocated() / 2**30:.1f} GiB")


def test_lens_reads_out_concepts(llama_fp32):
    q = llama_fp32
    tok = q.tokenizer
    lens_logits, model_logits, _ = q.lens.apply(q.model, "The capital of France is the city of", positions=[-1])
    final = [tok.decode([t]).strip().lower() for t in model_logits[0].topk(3).indices]
    late = {l: [tok.decode([t]).strip().lower() for t in lens_logits[l][0].topk(5).indices] for l in (20, 26)}
    print(f"\nfinal {final} | lens {late}")
    assert "paris" in final
    assert any("paris" in w for ws in late.values() for w in ws), "the lens should read the capital at a late layer"


@pytest.mark.parametrize("l_start,l_end", SLICES)
def test_slice_matches_hooked_forward_fp32(llama_fp32, l_start, l_end):
    model = llama_fp32.model
    ids = model.encode(PROMPT, max_length=64)
    with torch.no_grad(), ActivationRecorder(model.layers, at=[l_start, l_end]) as rec:
        model.forward(ids)
        h_in, h_out = rec.activations[l_start].detach(), rec.activations[l_end].detach()
    with torch.no_grad():
        got = make_slice_fn(model, l_start, l_end)(h_in)
    diff = (got - h_out).abs().max().item()
    print(f"\nslice ({l_start},{l_end}): max |diff| = {diff:.3e}")
    torch.testing.assert_close(got, h_out, rtol=1e-4, atol=1e-4)


def test_mlp_branch_matches_block_contribution(llama_fp32):
    model = llama_fp32.model
    ids = model.encode(PROMPT, max_length=64)
    block = model.layers[16]
    store = {}
    h = mlp_prenorm(block).register_forward_pre_hook(lambda m, a: store.__setitem__("x", a[0].detach()))
    with torch.no_grad(), ActivationRecorder(model.layers, at=[16]) as rec:
        model.forward(ids)
    h.remove()
    with torch.no_grad():
        torch.testing.assert_close(rec.activations[16], store["x"] + mlp_branch(block)(store["x"]), rtol=1e-4, atol=1e-4)


def test_hvp_fr_matches_rr_on_real_slice(llama_fp32):
    model = llama_fp32.model
    ids = model.encode(PROMPT, max_length=64)
    l0, l1 = 8, 12
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
    assert rel < 1e-3 and b.norm() > 0

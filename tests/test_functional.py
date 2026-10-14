import torch

import pytest

from hlens.functional import make_slice_fn, make_output_fn
from jlens.hooks import ActivationRecorder

from .tiny import TinyDecoder


def test_make_slice_fn_matches_hooked_forward():
    model = TinyDecoder(n_layers=4, d_model=8)
    input_ids = model.encode("the quick brown fox jumps over")

    l_start, l_end = 0, 2
    with ActivationRecorder(model.layers, at=[l_start, l_end]) as recorder:
        model.forward(input_ids)

    h_start = recorder.activations[l_start]
    h_end_expected = recorder.activations[l_end]

    slice_fn = make_slice_fn(model, l_start, l_end)
    h_end_actual = slice_fn(h_start)

    torch.testing.assert_close(h_end_actual, h_end_expected, rtol=1e-5, atol=1e-5)

def test_make_output_fn_matches_hooked_forward():
    model = TinyDecoder(n_layers=4, d_model=8)
    input_ids = model.encode("the quick brown fox jumps over")

    l_start = 0
    l_last = model.n_layers - 1
    with ActivationRecorder(model.layers, at=[l_start, l_last]) as recorder:
        model.forward(input_ids)

    h_start = recorder.activations[l_start]
    h_last = recorder.activations[l_last]
    logits_expected = model.unembed(h_last)

    output_fn = make_output_fn(model, l_start, model.unembed)
    logits_actual = output_fn(h_start)

    torch.testing.assert_close(logits_actual, logits_expected, rtol=1e-5, atol=1e-5)


PROMPT = "Fact: The currency used in the country shaped like a boot is"


# (l_start, l_end): kwargs come from block l_start+1
#   (2, 3)  -> a single full-attention layer
#   (3, 4)  -> a single linear-attention layer
#   (3, 12) -> a mix of both types
#   (0, 30) -> the whole stack
SLICES = [(2, 3), (3, 4), (3, 12), (0, 30)]


@pytest.mark.real_model
@pytest.mark.parametrize("l_start,l_end", SLICES)
def test_slice_fn_matches_hooked_forward_qwen(qwen, l_start, l_end):
    _check_slice_matches_hooked(qwen.model, l_start, l_end)


# same check in fp32: separates wrapper bugs from precision effects
@pytest.mark.real_model
@pytest.mark.parametrize("l_start,l_end", SLICES)
def test_slice_fn_matches_hooked_forward_qwen_fp32(qwen_fp32, l_start, l_end):
    assert next(qwen_fp32.hf.parameters()).dtype == torch.float32
    _check_slice_matches_hooked(qwen_fp32.model, l_start, l_end)


def _check_slice_matches_hooked(model, l_start, l_end):
    ids = model.encode(PROMPT, max_length=64)

    with torch.no_grad(), ActivationRecorder(model.layers, at=[l_start, l_end]) as rec:
        model.forward(ids)
        h_in = rec.activations[l_start].detach()
        h_out = rec.activations[l_end].detach()

    with torch.no_grad():
        got = make_slice_fn(model, l_start, l_end)(h_in)

    diff = (got.float() - h_out.float()).abs().max().item()
    print(f"\nslice ({l_start},{l_end}): max |diff| = {diff:.3e}")
    torch.testing.assert_close(got, h_out, rtol=0, atol=0)
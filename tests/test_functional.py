import torch

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
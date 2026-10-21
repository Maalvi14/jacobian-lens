import numpy as np
import pytest
import torch

from hlens.experiments.gain import capture_mlp_inputs, collect_contexts, mlp_taylor, summarize, PopulationStats

from .tiny_qwen import tiny_model


def test_mlp_taylor_matches_second_difference_on_tiny_model():
    hf, model = tiny_model(0)
    block = model.layers[2]
    torch.manual_seed(0)
    R = torch.randn(6, model.d_model)
    V = torch.randn(6, model.d_model)
    V = V / V.norm(dim=1, keepdim=True)
    g1, kappa = mlp_taylor(block, R, V)
    f = lambda x: block.mlp(block.post_attention_layernorm(x))
    eps = 1e-2
    with torch.no_grad():
        sd = (f(R + eps * V) + f(R - eps * V) - 2 * f(R)) / eps**2          # = v^T H v
        fd = (f(R + eps * V) - f(R - eps * V)) / (2 * eps)                   # = J v
    assert torch.allclose(g1, fd.norm(dim=-1), rtol=2e-3)
    assert torch.allclose(kappa, 0.5 * sd.norm(dim=-1), rtol=5e-2)


def test_collect_contexts_captures_the_mlp_input():
    hf, model = tiny_model(0)
    prompts = ["the quick brown fox jumps over the lazy dog and keeps running far away from here again and again"] * 2
    ctx = collect_contexts(model, prompts, [1, 3], per_prompt=3, max_length=64)
    assert ctx[1].shape == (6, model.d_model) and ctx[3].shape == (6, model.d_model)
    ids = model.encode(prompts[0], max_length=64)
    with torch.no_grad(), capture_mlp_inputs(model, [1]) as store:
        model.forward(ids)
    # every collected row is one of the captured positions of this (identical) prompt
    rows = store[1][0]
    for r in ctx[1]:
        assert (rows - r).abs().sum(dim=1).min() < 1e-5


def test_summarize_ratios_and_variability():
    rng = np.random.default_rng(0)
    C, D = 40, 16
    mk = lambda scale, cv: PopulationStats(
        g1=scale * np.exp(rng.normal(0, cv, (C, 1))) * np.ones((1, D)) * np.exp(rng.normal(0, 0.05, (C, D))),
        kappa=scale * 2 * np.exp(rng.normal(0, 0.05, (C, D))), r_norm=np.full(C, 10.0))
    pops = {"random": mk(1.0, 0.1), "j_lens": mk(3.0, 0.5), "neuron": mk(1.0, 0.1)}
    res = summarize(pops, n_boot=50)
    clustered = summarize(pops, n_boot=50, per_group=8)  # 5 "prompts" of 8 contexts
    assert clustered["bootstrap"] == "prompt-cluster" and res["bootstrap"] == "context"
    w = lambda r: r["ratio_to_random"]["j_lens"]["g1"]["ci95"]
    assert w(clustered)[1] - w(clustered)[0] >= 0.5 * (w(res)[1] - w(res)[0])  # resampling prompts is not narrower
    assert res["ratio_to_random"]["j_lens"]["g1"]["median"] == pytest.approx(3.0, rel=0.35)  # shared per-context noise over 40 contexts
    assert res["ratio_to_random"]["neuron"]["kappa"]["ci95"][0] < 1.0 < res["ratio_to_random"]["neuron"]["kappa"]["ci95"][1] + 0.2
    assert res["median_cv_g1_over_contexts"]["j_lens"] > 2 * res["median_cv_g1_over_contexts"]["random"]

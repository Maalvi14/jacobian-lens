import numpy as np
import pytest
import torch

from hlens.experiments.remainder_coupling import (
    analyze_h5, build_dictionary, measure_context, prereg_sha256, sign_flip_pvalue,
)

from .tiny_qwen import VOCAB, tiny_lens, tiny_model

PROMPT = "the quick brown fox jumps over the lazy dog and then keeps on running across the wide open field again"


def test_measure_context_on_tiny_model():
    hf, model = tiny_model(0)
    lens = tiny_lens(model, layers=(1, 2))
    D, token_ids = build_dictionary(model, lens, 1, VOCAB)
    assert torch.allclose(D.norm(dim=1), torch.ones(len(D)), atol=1e-4)
    r = measure_context(model, D, PROMPT, 1, 20, k=4, n_ctrl=4, max_length=64, min_rho_frac=0.0)
    assert r is not None and len(r["s_sel"]) == len(r["coeffs"]) and len(r["s_ctrl"]) == 4
    assert all(0.0 <= v <= 1.0 + 1e-5 for v in r["s_sel"] + r["s_ctrl"])  # cosine magnitudes
    assert r["d"] == pytest.approx(np.mean(r["s_sel"]) - np.mean(r["s_ctrl"]), abs=1e-6)
    assert r["w_norm"] > 0 and r["grad_norm"] > 0


def test_sign_flip_and_verdict_logic():
    rng = np.random.default_rng(0)
    assert sign_flip_pvalue(rng.normal(0.5, 0.2, 100), n=2000) < 0.01
    assert sign_flip_pvalue(rng.normal(0.0, 0.2, 100), n=2000) > 0.05

    def arm(mu, n=100):
        rows = [{"d": float(x), "s_sel": [0.02], "s_ctrl": [0.02], "w_norm": 1.0, "grad_norm": 1.0, "frac_variance": 0.1} for x in rng.normal(mu, 0.1, n)]
        return {"rows": rows, "prereg_sha256": prereg_sha256(), "n_excluded": 0}

    yes = analyze_h5(arm(0.3), arm(0.0), n_boot=2000)
    assert yes["H5_supported"] and yes["test_A"]["passed"] and yes["test_B"]["passed"]
    # reproduced by the randomized model: A passes, B fails -> not supported
    same = analyze_h5(arm(0.3), arm(0.3), n_boot=2000)
    assert same["test_A"]["passed"] and not same["test_B"]["passed"] and not same["H5_supported"]
    none = analyze_h5(arm(0.0), arm(0.0), n_boot=2000)
    assert not none["H5_supported"]


def test_selected_atoms_are_orthogonal_to_the_remainder_and_perp_stats_are_cosines():
    from hlens.jspace import decompose

    torch.manual_seed(0)
    D = torch.nn.functional.normalize(torch.randn(300, 32), dim=1)
    x = torch.randn(32) + 2.0 * D[7] + 1.5 * D[41]
    dec = decompose(x, D, k=6)
    rhat = dec.remainder / dec.remainder.norm()
    # KKT of the NNLS refit: the remainder is orthogonal to every selected atom (the flaw the addendum corrects)
    assert (D[dec.indices] @ rhat).abs().max() < 1e-4

    hf, model = tiny_model(0)
    lens = tiny_lens(model, layers=(1, 2))
    Dm, _ = build_dictionary(model, lens, 1, VOCAB)
    r = measure_context(model, Dm, PROMPT, 1, 20, k=4, n_ctrl=4, max_length=64, min_rho_frac=0.0)
    assert all(0.0 <= v <= 1.0 + 1e-5 for v in r["s_sel_perp"] + r["s_ctrl_perp"])
    assert r["d_perp"] == pytest.approx(np.mean(r["s_sel_perp"]) - np.mean(r["s_ctrl_perp"]), abs=1e-6)

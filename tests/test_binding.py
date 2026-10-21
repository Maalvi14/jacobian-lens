import numpy as np
import pytest
import torch

from hlens.experiments.binding import (
    ablate_span, analyze_h3, build_dictionary_with_rows, measure_item, measure_prompt, pair_scores, prereg_sha256, rank_fraction, summarize_variant, taylor2, wilcoxon_greater,
)
from hlens.synonyms import forms_for, number_forms, single_token_ids

from .tiny_qwen import VOCAB, tiny_lens, tiny_model


def test_ablate_span_removes_exactly_the_span_component():
    torch.manual_seed(0)
    x, V = torch.randn(10), torch.nn.functional.normalize(torch.randn(3, 10), dim=1)
    xp, c0 = ablate_span(x, V)
    assert torch.allclose(V @ xp, torch.zeros(3), atol=1e-5)
    assert torch.allclose(xp + V.T @ c0, x, atol=1e-5)
    one, _ = ablate_span(x, V[:1])
    assert (V[0] @ one).abs() < 1e-5


def test_rank_fraction_and_pair_scores():
    assert rank_fraction(5.0, np.array([1.0, 2.0, 9.0, 9.0])) == 0.5
    assert rank_fraction(1.0, np.array([1.0, 1.0])) == 0.5  # ties count half
    M = np.zeros((4, 4))
    M[0, 1] = M[1, 0] = 10.0  # the binding pair is the largest of 6 pairs
    for i, j in [(0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
        M[i, j] = M[j, i] = 1.0
    assert pair_scores(M, (0, 1)) == 1.0
    assert pair_scores(M, (2, 3)) == pytest.approx(0.5 * 4 / 5 + 0.0, abs=0.5)  # smaller than (0,1), tied with 4 others


def test_quadratic_function_measured_interaction_equals_taylor_prediction():
    torch.manual_seed(1)
    d, n = 12, 5
    Qm = torch.randn(d, d)
    Qm = (Qm + Qm.T) / 2
    g0 = torch.randn(d)
    f1 = lambda x: g0 @ x + 0.5 * x @ Qm @ x  # exact quadratic: Taylor-2 is exact
    x0 = torch.randn(d)
    S = torch.nn.functional.normalize(torch.randn(n, d), dim=1)
    out = measure_item(f1, x0, S[0], S[1], S[2:])
    I, Ihat = np.array(out["I"]), np.array(out["Ihat"])
    assert np.allclose(I, Ihat, atol=1e-4), "on a quadratic f the second-order prediction is exact"
    # and the restricted Hessian is S Q S^T
    assert np.allclose(np.array(out["H"]), (S @ Qm @ S.T).numpy(), atol=1e-4)
    assert 0.0 <= out["P"] <= 1.0 and 0.0 <= out["Q"] <= 1.0


def test_synonyms_and_single_token_filter():
    assert number_forms(5) == ["5", "five"] and number_forms(12) == ["12", "twelve"] and number_forms(30) == ["30", "thirty"]
    assert "*" in forms_for("multiplication") and "plus" in forms_for("addition") and forms_for("7") == ["7", "seven"]

    class Tok:  # one token per form up to 3 chars
        def encode(self, s, add_special_tokens=False):
            return [1] if len(s.strip()) <= 3 else [1, 2]

    assert set(single_token_ids(Tok(), ["5", "five", "multiplication"])) == {"5", " 5"}


def test_measure_prompt_on_tiny_model_and_analysis_logic():
    hf, model = tiny_model(0)
    lens = tiny_lens(model, layers=(1, 2))
    D, row_of, token_of = build_dictionary_with_rows(model, lens, 1, VOCAB)
    res = {"A": {"a": 5}, "B": {"b": 9}, "target_ids": [3]}
    prompt = "the quick brown fox jumps over the lazy dog and then keeps on running across the open field"
    r = measure_prompt(model, lens, D, row_of, token_of, res, prompt, 1, 8, require_correct=False)
    assert "H" in r, r
    n = 2 + r["n_ctrl"]
    assert np.array(r["H"]).shape == (n, n) and np.array(r["I"]).shape == (n, n)
    assert 5 not in r["control_tokens"] and 9 not in r["control_tokens"] and 3 not in r["control_tokens"]

    rng = np.random.default_rng(0)
    mk = lambda mu: [{"name": f"i{i}", "H": [[0]], "I": [[0]], "Ihat": [[0]], "P": float(np.clip(rng.normal(mu, 0.15), 0, 1)),
                      "Q": float(np.clip(rng.normal(mu, 0.15), 0, 1)), "superadditive_AB": True} for i in range(40)]
    assert summarize_variant(mk(0.8))["T"] > 0.7 and summarize_variant(mk(0.8))["p_T"] < 0.01
    assert summarize_variant(mk(0.5))["p_T"] > 0.05
    assert wilcoxon_greater(np.full(5, 0.5)) == (0.0, 1.0)

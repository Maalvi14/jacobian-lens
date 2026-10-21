import torch

from hlens.experiments.trust_score import trust_score, swap_delta

from hlens.interventions import coordinate_swap

def test_trust_score_elementwise_square():
    torch.manual_seed(0)
    d = 5
    h = torch.randn(d)
    delta_h = torch.randn(d)

    def f(h: torch.Tensor) -> torch.Tensor:
        return h**2

    # Hessian per output i is exactly 2 (constant), so:
    #   quadratic term = δh² * 2      -> ‖½ · that‖ = ‖δh²‖
    #   J·δh = 2h ⊙ δh (elementwise)  -> ‖that‖
    expected_numerator = (delta_h**2).norm()
    expected_denominator = (2 * h * delta_h).norm()
    expected = (expected_numerator / expected_denominator).item()

    result = trust_score(f, h, delta_h)

    assert abs(result - expected) < 1e-4


def test_swap_delta_matches_coordinate_swap_difference():
    torch.manual_seed(1)
    d = 6
    h = torch.randn(d)
    v_s = torch.randn(d)
    v_t = torch.randn(d)

    delta = swap_delta(h, v_s, v_t, alpha=0.7)
    expected = coordinate_swap(h, v_s, v_t, alpha=0.7) - h

    torch.testing.assert_close(delta, expected)

# --- H1 orchestration ---------------------------------------------------------------

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from hlens.experiments.trust_score import analyze, run_trust_score, trust_terms  # noqa: E402


def test_trust_terms_are_the_two_taylor_orders():
    torch.manual_seed(2)
    h, d = torch.randn(5), torch.randn(5)
    num, den = trust_terms(lambda x: x**2, h, d)
    assert num == pytest.approx((d**2).norm().item(), rel=1e-4)
    assert den == pytest.approx((2 * h * d).norm().item(), rel=1e-4)
    assert num / den == pytest.approx(trust_score(lambda x: x**2, h, d), rel=1e-6)


def _synthetic_rows(n=80, informative=True, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        fail = i % 2 == 0
        tau = float(np.exp(rng.normal(0.0 + (1.0 if fail and informative else 0.0), 0.5)))
        layer = lambda: {"tau": tau * float(np.exp(rng.normal(0, 0.1))), "num": 1.0, "den": float(rng.uniform(1, 2)),
                         "delta_norm": float(rng.uniform(1, 2)), "h_norm": 10.0, "loading_mean": float(rng.normal(0, 0.1)),
                         "loading_last": 0.0}
        rows.append({"name": f"i{i}", "category": "a" if i % 4 < 2 else "b", "base_ok": True, "hit_a1.25": not fail,
                     "layers": {"12": layer(), "16": layer(), "20": layer()},
                     "null": {"tau": float(np.exp(rng.normal(0, 0.5))), "loading_mean": 0.0}})
    return rows


def test_analyze_detects_an_informative_tau_and_not_uninformative_baselines():
    res = analyze(_synthetic_rows(informative=True), alpha=1.25)
    assert res["n"] == 80 and res["n_fail"] == 40
    assert res["auc"]["tau_L16"]["auc"] > 0.8 and res["auc"]["tau_L16"]["perm_p"] < 0.01
    assert abs(res["auc"]["baseline_neg_loading_L16"]["auc"] - 0.5) < 0.15
    assert abs(res["auc"]["null_tau_L16"]["auc"] - 0.5) < 0.15
    assert res["loo_logistic_auc"]["log_tau16"] > 0.75
    assert set(res["by_category"]) == {"a", "b"}
    flat = analyze(_synthetic_rows(informative=False), alpha=1.25)
    assert abs(flat["auc"]["tau_L16"]["auc"] - 0.5) < 0.15


@pytest.mark.real_model
def test_run_trust_score_mechanics_and_alpha_linearity(qwen_fp32):
    from hlens.contexts import ContextPoint  # noqa: F401
    from hlens.experiments.trust_score import _swap_vecs, _tau_at_layer
    from hlens.swaps import probe_items

    tok = qwen_fp32.tokenizer
    items = probe_items(tok)[:4]
    rows = run_trust_score(qwen_fp32, layers=(16,), alphas=(1.25,), items=items, log_every=0)
    assert len(rows) == 4
    for r in rows:
        t = r["layers"]["16"]
        assert np.isfinite(t["tau"]) and t["tau"] > 0 and t["den"] > 0 and -1 <= t["loading_mean"] <= 1
    print("\ntau16 per item:", [round(r["layers"]["16"]["tau"], 3) for r in rows], "hits:", [r["hit_a1.25"] for r in rows], flush=True)

    # tau is exactly linear in alpha (quadratic numerator / linear denominator)
    from hlens.experiments.trust_score import swap_delta, trust_terms as terms
    from hlens.functional import make_slice_fn
    from hlens.jspace import unembed_directions
    from jlens.hooks import ActivationRecorder

    it, s, t = items[0]
    ids = qwen_fp32.model.encode(it.prompt, max_length=128)
    U2 = unembed_directions(qwen_fp32.model, [s, t])
    with torch.no_grad(), ActivationRecorder(qwen_fp32.model.layers, at=[16]) as rec:
        qwen_fp32.model.forward(ids)
        h = rec.activations[16].detach().clone()
    v = _swap_vecs(U2.to(h.device), qwen_fp32.lens, 16)
    sl = make_slice_fn(qwen_fp32.model, 16, qwen_fp32.model.n_layers - 1)
    f = lambda x: sl(x)[0, -1]
    n1, d1 = terms(f, h, swap_delta(h, v[0], v[1], alpha=1.0))
    n2, d2 = terms(f, h, swap_delta(h, v[0], v[1], alpha=1.25))
    assert (n2 / d2) == pytest.approx(1.25 * (n1 / d1), rel=2e-3)
    assert (n1 / d1) == pytest.approx(rows[0]["layers"]["16"]["tau"], rel=1e-3)


# --- variants -----------------------------------------------------------------------


def test_taylor_terms_reproduce_a_quadratic_exactly():
    torch.manual_seed(3)
    A = torch.randn(4, 4)
    h, d = torch.randn(4), torch.randn(4)
    f = lambda x: x @ A @ x  # scalar quadratic: f(h + a d) = f(h) + a J d + a^2 (1/2 d^T H d) exactly
    from hlens.experiments.trust_score import taylor_terms

    quad, lin = taylor_terms(f, h, d)
    for a in (0.5, 1.25, -2.0):
        assert f(h + a * d).item() == pytest.approx((f(h) + a * lin + a**2 * quad).item(), rel=1e-4)


@pytest.mark.real_model
def test_score_variants_mechanics(qwen_fp32):
    from hlens.experiments.trust_score import _clean_logits, score_variants
    from hlens.jspace import unembed_directions
    from hlens.swaps import first_token_ids, probe_items

    tok, model = qwen_fp32.tokenizer, qwen_fp32.model
    it, s, t = probe_items(tok)[0]
    ids = model.encode(it.prompt, max_length=128)
    logits = _clean_logits(model, ids)
    ans = logits.argmax().item()
    swap = max(first_token_ids(tok, it.swap_answer), key=lambda x: logits[x].item())
    out = score_variants(model, qwen_fp32.lens, unembed_directions(model, [s, t]), ids, 16, ans, swap, full=True)
    for key in ("resid/all", "resid/last", "resid/pstar", "margin/all", "margin/last", "margin/pstar"):
        r = out[key]
        assert np.isfinite(r["tau"]) and r["tau"] > 0
        assert r["err2"] < r["err1"] * 3 + 1  # sanity only; the comparison itself is the experiment
    m = out["margin/all"]
    assert m["m0"] < 0, "baseline-correct item: the answer token outranks the swap token"
    assert out["resid/all"]["tau"] > 0 and 1 <= out["p_star"] < out["n_tokens"]
    print("\nvariants:", {k: round(v["tau"], 3) for k, v in out.items() if isinstance(v, dict)}, "err1(margin/all)", round(m["err1"], 3), "err2", round(m["err2"], 3), flush=True)


def test_first_token_ids_skip_whitespace_and_flag_ambiguity():
    import transformers

    from hlens.data import ProbeSwapItem
    from hlens.loading import MODEL_ID
    from hlens.swaps import first_token_ids, is_ambiguous

    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    space = tok.encode(" ", add_special_tokens=False)[0]
    for w in ("5", "9", "20", "Portuguese"):
        assert space not in first_token_ids(tok, w), "a bare space must never count as an answer"
    mk = lambda a, b: ProbeSwapItem("x", "c", "p", "i", a, "t", b)
    assert not is_ambiguous(tok, mk("5", "9"))
    assert not is_ambiguous(tok, mk("Portuguese", "Spanish"))
    assert is_ambiguous(tok, mk("20", "26")), "both start with the digit 2"


def _variant_rows(n=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        fail = i % 2 == 0
        def site(tau):
            return {"num": 1.0, "den": 1.0, "tau": tau, "err1": tau * 0.9, "err2": tau * 0.4, "m0": -1.0, "g1": 0.3, "q2": 0.1, "true_dm": 0.5}
        layer16 = {"p_star": 3, "loading_max": float(rng.random()), "loading_mean": float(rng.normal()), "n_tokens": 10}
        for fk in ("resid", "margin"):
            for sk in ("all", "last", "pstar"):
                layer16[f"{fk}/{sk}"] = site(float(np.exp(rng.normal(0.8 * fail, 0.4))))
        layers = {"12": {"resid/all": site(1.0), "margin/all": site(1.0), **{k: layer16[k] for k in ("p_star",)}, "loading_max": 0.1, "loading_mean": 0.0},
                  "16": layer16,
                  "20": {"resid/all": site(1.0), "margin/all": site(1.0), "p_star": 3, "loading_max": 0.1, "loading_mean": 0.0}}
        rows.append({"name": f"i{i}", "category": "city-capital" if i % 3 == 0 else "multihop", "hit_a1.25": not fail, "layers": layers,
                     "null": {k: layer16[k] for k in layer16 if "/" in k}})
    return rows


def test_analyze_variants_runs_without_nans_and_finds_signal():
    from hlens.experiments.trust_score import analyze_variants

    res = analyze_variants(_variant_rows())
    for name, v in res["auc"].items():
        assert np.isfinite(v["auc_fail"]), f"{name} produced a NaN (degenerate scores?)"
    assert res["auc"]["tau_margin_pstar_L16"]["auc_fail"] > 0.7
    assert abs(res["auc"]["baseline_neg_loading_mean_L16"]["auc_fail"] - 0.5) < 0.2
    assert res["median_err1_err2"]["resid"][1] < res["median_err1_err2"]["resid"][0]
    assert set(res["link1_spearman_tau_vs_true_error"]) == {"resid_all", "margin_all"}


def test_analyze_with_a_different_mid_layer_aliases_the_qwen_key_names():
    import numpy as np

    from hlens.experiments.trust_score import analyze

    rng = np.random.default_rng(0)
    rows = []
    for i in range(60):
        hit = bool(i % 2)
        layers = {str(l): {"tau": float(np.exp(rng.normal())), "num": 1.0, "den": float(rng.random() + 0.1), "delta_norm": 1.0, "h_norm": 2.0,
                           "loading_mean": float(rng.normal()), "loading_last": 0.0} for l in (18, 24, 30)}
        rows.append({"name": f"i{i}", "category": "city-capital" if i % 3 else "element-symbol", "n_tokens": 10, "base_ok": True, "hit_a1": hit, "moved_a1": True, "layers": layers})
    out = analyze(rows, alpha=1.0, layers=(18, 24, 30), label="hit", mid_layer=24)
    assert out["mid_layer"] == 24 and "tau_L24" in out["auc"] and out["auc"]["tau_L16"] == out["auc"]["tau_L24"]
    assert set(out["stratified_by_geography"]) >= {"tau_L16", "baseline_neg_loading_L16"} and out["n_success"] == 30

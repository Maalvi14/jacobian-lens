import numpy as np

from hlens.experiments.h2_replication import analyze
from hlens.experiments.h9_replication import trial_sets
from hlens.loading import matched_layer


def test_matched_layer_maps_by_depth():
    assert [matched_layer(32, l) for l in (12, 16, 20)] == [12, 16, 20]
    assert [matched_layer(48, l) for l in (12, 16, 20)] == [18, 24, 30]
    assert [matched_layer(64, l) for l in (12, 16, 20)] == [24, 33, 41]


def _layer(med, lo, hi):
    return {"ratio_to_random": {"j_lens": {"kappa": {"median": med, "ci95": [lo, hi]}}}}


def test_analyze_decision_rules():
    real = {"4": _layer(0.9, 0.8, 1.0), "24": _layer(2.0, 1.8, 2.2), "28": _layer(2.1, 1.9, 2.4), "44": _layer(1.6, 1.3, 1.9)}
    null_ok = [{"seed": 0, "result": {"24": _layer(1.0, 0.9, 1.1)}}]
    res = analyze(real, null_ok, n_layers=48)
    assert res["real"]["layers_in_range"] == [24, 28, 44]  # depth >= 0.5 only
    assert res["real"]["T1"] and res["real"]["T2"] and res["T3"] and res["B_holds"]
    null_bad = [{"seed": 1, "result": {"24": _layer(1.4, 1.3, 1.5)}}]
    assert not analyze(real, null_bad, n_layers=48)["B_holds"]  # a null whose CI excludes 1 withdraws B


def test_trial_sets_disjoint_and_sized():
    class Tok:
        def encode(self, s, add_special_tokens=False):
            return [1]

    from types import SimpleNamespace

    cfg = SimpleNamespace(countries_12=[f"C{i}" for i in range(12)], ctx_templates=[f"t{i} {{W}}" for i in range(40)])
    sets = trial_sets(Tok(), cfg)
    assert len(sets["O"]) == 24 and len(sets["HO"]) == 120
    assert not ({(a, b) for _, a, b in sets["O"]} & {(a, b) for _, a, b in sets["HO"]})
    assert not ({t for t, _, _ in sets["O"]} & {t for t, _, _ in sets["HO"]})


def test_b_prime_rules():
    from hlens.experiments.h2_replication import analyze_b_prime

    L = lambda med, lo, hi: _layer(med, lo, hi)
    real = {"8": L(1.0, 0.99, 1.01), "16": L(1.2, 1.1, 1.3), "24": L(1.5, 1.3, 1.7), "28": L(1.9, 1.7, 2.1)}
    ok = [{"seed": 10, "result": {"16": L(1.0, 0.999, 1.001), "24": L(1.0, 0.99, 1.01)}}]
    res = analyze_b_prime(real, ok, n_layers=32)
    assert res["N"] and res["R"] and res["C"] and res["B_prime_holds"]
    assert set(res["C_detail"]) == {16, 24}
    # a null whose interval leaves [0.95, 1.05] breaks N
    wide = [{"seed": 11, "result": {"16": L(1.0, 0.9, 1.1)}}]
    assert not analyze_b_prime(real, wide, n_layers=32)["N"]
    # a real layer whose lower bound is inside the equivalence region breaks R when it is the majority
    weak = {"16": L(1.04, 1.01, 1.07), "24": L(1.05, 1.02, 1.08), "28": L(1.3, 1.2, 1.4)}
    assert not analyze_b_prime(weak, ok, n_layers=32)["R"]
    # real not separable from the null upper bound breaks C
    close = {"16": L(1.2, 1.0005, 1.3), "24": L(1.5, 1.3, 1.7), "28": L(1.9, 1.7, 2.1)}
    assert not analyze_b_prime(close, [{"seed": 10, "result": {"16": L(1.0, 0.999, 1.001)}}], n_layers=32)["C"]


def test_b_prime_prompt_ranges_disjoint():
    from hlens.experiments.h2_replication import CONFIGS, N_EVAL, N_FIT

    A, B = CONFIGS["A"], CONFIGS["B2"]
    ranges = [(c["eval_start"], c["eval_start"] + N_EVAL) for c in (A, B)] + [(c["fit_start"], c["fit_start"] + N_FIT) for c in (A, B)]
    for i, (a0, a1) in enumerate(ranges):
        for b0, b1 in ranges[i + 1 :]:
            assert a1 <= b0 or b1 <= a0


def test_b3_config_for_llama_is_depth_matched_and_uses_fresh_prompts():
    from hlens.experiments.h2_replication import CONFIGS
    from hlens.loading import MODELS, matched_layer

    c = CONFIGS["B3"]
    assert c["prefix"] == "B3_" and c["real_layers"]["llama3.1-8b"] == CONFIGS["B2"]["real_layers"]["qwen3.5-4b"]
    assert all(matched_layer(32, l) == l for l in c["real_layers"]["llama3.1-8b"])  # 32 layers, like the reference model
    assert max(c["real_layers"]["llama3.1-8b"]) <= 30  # the lens covers layers 0-30
    assert MODELS["llama3.1-8b"].model_id == "meta-llama/Llama-3.1-8B" and (c["eval_start"], c["fit_start"]) == (200, 300)

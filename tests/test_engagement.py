import numpy as np
import pytest

from hlens.experiments.engagement import ALPHAS, analyze_e1, path_summary


def test_path_summary_on_a_known_quadratic_margin():
    m = [-2.0 + 1.0 * a + 1.5 * a**2 for a in ALPHAS]
    s = path_summary(m)
    assert s["m0"] == pytest.approx(-2.0) and s["g"] == pytest.approx(1.0)  # central difference cancels the quadratic term
    assert s["FO_ref"] == pytest.approx(-2.0 + 1.25) and s["N"] == pytest.approx(1.5 * 1.25**2)
    assert s["alpha_c"] == pytest.approx((-1 + np.sqrt(13)) / 3, abs=0.01)  # root of 1.5 a^2 + a - 2
    assert s["alpha_n"] == pytest.approx(0.25)  # first grid point with |quadratic part| > 25% of the change


def test_a_purely_linear_path_has_no_nonlinear_gain_and_no_knee():
    s = path_summary([-1.0 + 2.0 * a for a in ALPHAS])
    assert s["N"] == pytest.approx(0.0, abs=1e-9) and s["alpha_n"] is None and s["FO_ref"] == pytest.approx(s["m_ref"])


def _items(n=60, informative=True, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        hit = i % 3 == 0
        N = float(rng.normal(1.0 * hit * informative, 0.5))
        mk = lambda N, fo: path_summary([-1.0 + fo * a + N / 1.25**2 * a**2 for a in ALPHAS])
        out.append({"name": f"i{i}", "category": "city-capital" if i % 2 else "multihop", "hit": bool(hit), "loading_mean": float(rng.normal()),
                    "band": mk(N, 0.2), "L16": mk(N, 0.2), "L16_null": mk(float(rng.normal(0, 0.5)), 0.2)})
    return out


def test_analyze_e1_decision_logic():
    yes = analyze_e1(_items(informative=True), n_boot=500)
    assert yes["P1"]["passed"] and yes["P2"]["passed"] and yes["P4"]["passed"] and yes["E1_supported"]
    no = analyze_e1(_items(informative=False), n_boot=500)
    assert not no["P1"]["passed"] and not no["E1_supported"]


def test_path_summary_reference_alpha_is_configurable():
    import numpy as np

    from hlens.experiments.engagement import ALPHAS, path_summary

    path = [0.0 + 3.0 * a - (0.5 * a * a if a > 0 else 0.0) for a in ALPHAS]  # concave: saturates below its first-order extrapolation
    s125, s10 = path_summary(path), path_summary(path, a_ref=1.0)
    assert s10["m_ref"] == path[ALPHAS.index(1.0)] and s125["m_ref"] == path[ALPHAS.index(1.25)]
    assert s10["N"] < 0 and s125["N"] < s10["N"]  # N = m - FO is more negative further along a concave path

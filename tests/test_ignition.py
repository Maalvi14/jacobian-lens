import numpy as np
import pytest
import torch

from hlens.experiments.ignition import lanczos_top_abs_eig, projection_share, transition


def test_projection_share_endpoints_and_midpoint():
    a, b = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 0.0])
    assert projection_share(a, b, b) == 0.0 and projection_share(a, b, a) == 1.0
    assert projection_share(a, b, torch.tensor([0.5, 3.0])) == pytest.approx(0.5)  # off-line components ignored


def test_transition_threshold_and_width_on_a_sigmoid():
    alphas = np.linspace(0, 1, 201)
    for width_param in (0.02, 0.1):
        share = 1 / (1 + np.exp(-(alphas - 0.4) / width_param))
        share = (share - share[0]) / (share[-1] - share[0])
        tr = transition(alphas, share)
        assert tr.threshold == pytest.approx(0.4, abs=0.02)
        assert tr.width == pytest.approx(2 * np.log(9) * width_param, rel=0.1)  # 10->90% of a logistic
        assert tr.monotone
    assert transition(alphas, alphas).width == pytest.approx(0.8, abs=0.02)  # linear: 10% -> 90%


def test_lanczos_recovers_the_leading_absolute_eigenvalue():
    torch.manual_seed(0)
    n = 40
    Q, _ = torch.linalg.qr(torch.randn(n, n))
    eig = torch.linspace(-3.0, 2.0, n)  # largest |eigenvalue| is the NEGATIVE one, -3
    A = Q @ torch.diag(eig) @ Q.T
    assert lanczos_top_abs_eig(lambda v: A @ v, n, iters=25) == pytest.approx(3.0, rel=1e-3)


def _arm(peak_layer, center_boost, n=24, seed=0, drop_between=(8, 10)):
    from hlens.experiments.ignition import LAYERS, OFFSETS, prereg_sha256

    rng = np.random.default_rng(seed)
    trials = []
    for _ in range(n):
        width = {m: 0.8 if m <= drop_between[0] else 0.3 for m in range(29)}  # width collapses between the two layers
        cells = {}
        for l in LAYERS:
            base = 1.0 + 3.0 * (l == peak_layer)
            for o in OFFSETS:
                cells[f"{l}|{o}"] = {"lambda": float(base * (1 + center_boost * (o == 0.0)) * np.exp(rng.normal(0, 0.05))),
                                     "tau": 0.1, "first_order": 1.0}
        trials.append({"alpha_star": 0.5, "width": width, "threshold": {}, "cells": cells})
    return {"prereg_sha256": prereg_sha256(), "layers": LAYERS, "offsets": OFFSETS, "trials": trials}


def test_analyze_h6_decision_logic():
    from hlens.experiments.ignition import analyze_h6

    yes = analyze_h6(_arm(peak_layer=8, center_boost=0.5), _arm(peak_layer=8, center_boost=0.0, seed=1), n_boot=500)
    assert yes["T1"]["passed"] and yes["T2"]["passed"] and yes["T3"]["passed"] and yes["H6_supported"]
    wrong_layer = analyze_h6(_arm(peak_layer=24, center_boost=0.5), _arm(peak_layer=8, center_boost=0.0, seed=1), n_boot=500)
    assert not wrong_layer["T1"]["passed"] and not wrong_layer["H6_supported"]
    no_center = analyze_h6(_arm(peak_layer=8, center_boost=0.0), _arm(peak_layer=8, center_boost=0.0, seed=1), n_boot=500)
    assert not no_center["T2"]["passed"] and not no_center["H6_supported"]
    same_as_null = analyze_h6(_arm(peak_layer=8, center_boost=0.5), _arm(peak_layer=8, center_boost=0.5, seed=1), n_boot=500)
    assert same_as_null["T2"]["passed"] and not same_as_null["T3"]["passed"]

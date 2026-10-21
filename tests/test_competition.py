import numpy as np
import torch

from hlens.experiments.competition import analyze_h9, inhibition, random_orthogonal_pair


def test_inhibition_sign_on_competitive_and_cooperative_functions():
    d = 6
    dirs = torch.eye(d)[:2]
    x0 = torch.randn(d)
    Qc = torch.zeros(d, d)
    Qc[0, 1] = Qc[1, 0] = -0.5  # cross-curvature -0.5 between the two directions
    g0 = torch.zeros(d)
    g0[:2] = 1.0
    comp = lambda x: g0 @ x + 0.5 * x @ Qc @ x
    cA, cB, I = inhibition(comp, comp, x0, dirs)
    assert abs(cA + 0.5) < 1e-5 and abs(cB + 0.5) < 1e-5 and abs(I - 0.5) < 1e-5
    coop = lambda x: g0 @ x + 0.5 * x @ (-Qc) @ x
    _, _, I2 = inhibition(coop, coop, x0, dirs)
    assert abs(I2 + 0.5) < 1e-5


def test_random_pair_is_orthonormal_and_orthogonal_to_the_dirs():
    dirs = torch.nn.functional.normalize(torch.randn(2, 20), dim=1)
    r = random_orthogonal_pair(dirs, seed=0)
    assert torch.allclose(r @ r.T, torch.eye(2), atol=1e-5)
    assert (r @ dirs.T).abs().max() < 1e-5


def test_analyze_h9_logic():
    rng = np.random.default_rng(0)

    def rows(center, flank, control, n=24):
        return [{"cells": {"-0.2": {"cA": -1, "cB": -1, "I": float(rng.normal(flank, 0.2)), "I_control": 0.0, "J_cross": 0.1},
                           "0.0": {"cA": -1, "cB": -1, "I": float(rng.normal(center, 0.2)), "I_control": float(rng.normal(control, 0.05)), "J_cross": 0.1},
                           "0.2": {"cA": -1, "cB": -1, "I": float(rng.normal(flank, 0.2)), "I_control": 0.0, "J_cross": 0.1}}} for _ in range(n)]

    assert analyze_h9(rows(1.0, 0.0, 0.0))["precondition_met"]
    assert not analyze_h9(rows(1.0, 1.0, 0.0))["precondition_met"]  # no threshold peak
    assert not analyze_h9(rows(0.0, 0.0, 0.0))["precondition_met"]


def test_list_loading_analysis_logic():
    from hlens.experiments.competition import KS, analyze_list_loading

    rng = np.random.default_rng(0)

    def rows(slope, link):
        out = []
        for fam in ("countries", "cities"):
            for li in range(8):
                for k in KS:
                    absent = float(np.clip(rng.normal(0.3, 0.1), 0, 1))
                    I = slope * k + link * absent + rng.normal(0, 0.05)
                    out.append({"family": fam, "list": li, "k": k, "I": float(I), "I_control": float(rng.normal(0, 0.05)),
                                "occupancy": int(round(k * (1 - absent)))})
        return out

    assert analyze_list_loading(rows(0.02, 1.0))["list_loading_supported"]
    flat = analyze_list_loading(rows(0.0, 0.0))
    assert not flat["Q1"]["passed"] and not flat["list_loading_supported"]

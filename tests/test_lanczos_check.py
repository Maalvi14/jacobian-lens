import numpy as np
import torch

from hlens.experiments.ignition import lanczos_top_abs_eig
from hlens.experiments.lanczos_check import compare, lanczos_profile


def _op(seed=0, d=60):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(d, d, generator=g)
    A = (A + A.T) / 2
    return A, (lambda v: A @ v)


def test_profile_matches_the_existing_estimator_and_converges():
    A, op = _op()
    true = torch.linalg.eigvalsh(A).abs().max().item()
    prof = lanczos_profile(op, A.shape[0], iters=40, checkpoints=(8, 16, 24, 40))
    assert abs(prof[8] - lanczos_top_abs_eig(op, A.shape[0], iters=8)) < 1e-4 * abs(prof[8])
    assert prof[40] <= true * (1 + 1e-4) and abs(prof[40] - true) / true < 5e-3
    assert abs(prof[40] - true) <= abs(prof[8] - true) + 1e-6  # more steps never move away from the top eigenvalue


def test_compare_flags_poor_and_good_convergence():
    def cell(t, l, o, lam8, lam40, xn=2.0):
        return {"trial": t, "layer": l, "offset": o, "xnorm": xn, "lam": {"8": lam8, "16": lam40, "24": lam40, "40": lam40}, "lam_seed1": {"8": lam8}, "lam_h6": lam8}

    good = [cell(t, l, o, 1.02 + (0.3 if o == 0.0 else 0.0), 1.0 + (0.3 if o == 0.0 else 0.0)) for t in range(5) for l in (4, 8) for o in (-0.2, 0.0, 0.2)]
    r = compare(good)
    assert r["C1_adequate"] and r["C2_passed"] and r["stored_h6_matches_seed0_8"] == 1.0
    bad = [cell(t, l, o, 0.5 + (0.0 if o == 0.0 else 0.4), 1.0 + (0.3 if o == 0.0 else 0.0)) for t in range(5) for l in (4, 8) for o in (-0.2, 0.0, 0.2)]
    rb = compare(bad)
    assert not rb["C1_adequate"] and not rb["C2_passed"]

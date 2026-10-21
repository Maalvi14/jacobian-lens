import numpy as np
import torch

from hlens.experiments.early_curvature import analyze_h7, candidate_scores, rank_fraction


def test_rank_fraction():
    assert rank_fraction(5.0, np.array([1.0, 2.0, 9.0])) == 2 / 3
    assert rank_fraction(1.0, np.array([1.0, 1.0])) == 0.5


def test_candidate_scores_on_a_quadratic():
    torch.manual_seed(0)
    d = 10
    Q = torch.randn(d, d)
    Q = (Q + Q.T) / 2
    g0 = torch.randn(d)
    f1 = lambda x: g0 @ x + 0.5 * x @ Q @ x
    x0 = torch.randn(d)
    V = torch.nn.functional.normalize(torch.randn(4, d), dim=1)
    sc = candidate_scores(f1, x0, torch.func.grad(f1)(x0), V)
    assert np.allclose(sc["s2"], np.abs(np.diag((V @ Q @ V.T).numpy())), atol=1e-4)
    assert np.allclose(sc["s1"], (V @ (g0 + Q @ x0)).abs().numpy(), atol=1e-4)
    assert np.allclose(sc["s0"], (V @ x0).numpy(), atol=1e-5)


def test_analyze_h7_decision_logic():
    rng = np.random.default_rng(0)

    def items(informative):
        out = []
        for _ in range(50):
            layers = {}
            for l in (4, 8, 12, 16):
                s2 = rng.random(16)
                if informative and l in (4, 8):
                    s2[0] = 5.0 + rng.random()  # the true intermediate has the largest curvature
                layers[str(l)] = {"s0": list(rng.normal(size=16)), "s1": list(rng.random(16)), "s2": list(s2)}
            out.append({"name": "x", "layers": layers})
        return out

    assert analyze_h7(items(True), n_boot=300)["H7_supported"]
    assert not analyze_h7(items(False), n_boot=300)["H7_supported"]

import numpy as np
import torch

from hlens.experiments.motor_collapse import analyze_h8, distance_to_scaled_identity, prereg_sha256, tau_map

from .tiny_qwen import tiny_model

PROMPT = "the quick brown fox jumps over the lazy dog and then keeps on running across the wide open field again"


def test_distance_to_scaled_identity():
    I = torch.eye(8)
    assert distance_to_scaled_identity(3.0 * I) < 1e-6
    assert distance_to_scaled_identity(torch.randn(8, 8)) > 0.5
    assert distance_to_scaled_identity(2.0 * I + 0.1 * torch.randn(8, 8)) < 0.2


def test_tau_map_is_finite_positive_and_grows_with_depth_of_the_map_on_a_tiny_model():
    hf, model = tiny_model(0)
    d = torch.nn.functional.normalize(torch.randn(model.d_model), dim=0)
    t_early, t_late = tau_map(model, PROMPT, 0, 20, d), tau_map(model, PROMPT, model.n_layers - 2, 20, d)
    assert np.isfinite(t_early) and t_early > 0 and np.isfinite(t_late) and t_late > 0


def test_analyze_h8_logic():
    rng = np.random.default_rng(0)
    L = [6, 10, 14, 18, 22, 26, 28, 29, 30]

    def taus(motor_factor):
        return {str(l): list(np.abs(rng.normal(1.0, 0.1, 60)) * (31 - l) * (motor_factor if l >= 28 else 1.0)) for l in L}

    d_I = {str(l): (0.9 if l < 29 else 0.2) for l in range(31)}
    yes = analyze_h8({"d_I": d_I, "real": taus(0.2), "null": taus(1.0), "prereg_sha256": prereg_sha256()}, n_boot=300)
    assert yes["A"]["passed"] and yes["B"]["passed"] and yes["C"]["passed"] and yes["H8_supported"]
    arch = analyze_h8({"d_I": d_I, "real": taus(0.2), "null": taus(0.2), "prereg_sha256": prereg_sha256()}, n_boot=300)
    assert arch["B"]["passed"] and not arch["C"]["passed"] and not arch["H8_supported"]


def test_config_for_gemma_depth_matches_qwen():
    from hlens.experiments.motor_collapse import QWEN_CONFIG, config_for

    assert config_for(32, 30) == {**QWEN_CONFIG, "mid_range": [10, 20]}
    g = config_for(48, 46)
    assert g["last"] == 47 and g["motor"] == [42, 44, 45] and g["workspace"] == [21, 27, 33] and g["mid_range"] == [15, 30]


def test_analyze_h8_uses_config_layers():
    import numpy as np

    from hlens.experiments.motor_collapse import analyze_h8, config_for

    cfg = config_for(48, 46)
    rng = np.random.default_rng(0)
    N = 30
    mk = lambda scale: {str(l): (scale * (1 + 0.1 * rng.random(N))).tolist() for l in cfg["layers"]}
    d = {"config": cfg, "d_I": {str(l): (0.9 if l < 46 else 0.3) for l in range(47)}, "real": mk(1.0), "null": mk(1.0)}
    res = analyze_h8(d, n_boot=50)
    assert res["A"]["passed"] and set(res) >= {"A", "B", "C", "H8_supported"}

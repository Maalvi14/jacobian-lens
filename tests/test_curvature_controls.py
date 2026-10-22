import torch

from hlens.experiments.curvature_controls import run_controls, sample_cov_directions, variance_match
from types import SimpleNamespace

from .tiny_qwen import tiny_lens, tiny_model

PROMPTS = ["the quick brown fox jumps over the lazy dog and keeps running far away from here again and again"] * 3


def test_cov_directions_follow_the_second_moment():
    g = torch.Generator().manual_seed(0)
    X = torch.randn(400, 8) * torch.tensor([5.0, 1, 1, 1, 1, 1, 1, 0.2])
    V = sample_cov_directions(X, 500, g)
    assert torch.allclose(V.norm(dim=1), torch.ones(500), atol=1e-5)
    assert V[:, 0].abs().mean() > 2 * V[:, 7].abs().mean()


def test_variance_match_picks_close_variances():
    g = torch.Generator().manual_seed(1)
    X = torch.randn(300, 8) * torch.tensor([4.0, 3, 2, 1, 1, 1, 1, 0.3])
    lens = sample_cov_directions(X, 6, g)
    pool = torch.cat([sample_cov_directions(X, 200, g), torch.nn.functional.normalize(torch.randn(200, 8), dim=1)])
    vm, err = variance_match(lens, pool, X)
    assert vm.shape == lens.shape and err < 0.3
    assert len({tuple(r.tolist()) for r in vm}) == 6  # no direction used twice


def test_run_controls_on_tiny_model():
    hf, model = tiny_model(0)
    lens = tiny_lens(model)
    q = SimpleNamespace(model=model, lens=lens, hf=hf)
    out = run_controls(q, PROMPTS, PROMPTS, [0, 1], n_dirs=6, per_prompt=3, log=lambda *_: None)
    assert set(out) == {"0", "1"}
    for v in out.values():
        assert set(v["ratio_j_lens_over"]) == {"random", "cov", "varmatch", "embed", "unembed"}
        assert all(r["median"] > 0 for r in v["ratio_j_lens_over"].values())

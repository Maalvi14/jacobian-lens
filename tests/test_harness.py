import torch

from hlens.harness import find_plateau, eps_sweep, estimator_agreement, ms_hvp, validity_score

def test_find_plateau_detects_flat_region():
    eps_grid = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    base = torch.tensor([1.0, 2.0, 3.0, 4.0])
    curve = torch.stack([
        base + torch.tensor([5.0, 0.0, 0.0, 0.0]),    # far off — noisy tiny-eps end
        base,                                          # flat region starts
        base + torch.tensor([0.001, 0.0, 0.0, 0.0]),  # flat
        base + torch.tensor([0.002, 0.0, 0.0, 0.0]),  # flat
        base + torch.tensor([10.0, 0.0, 0.0, 0.0]),   # diverges — large-eps breakdown
        base + torch.tensor([20.0, 0.0, 0.0, 0.0]),
    ])

    result = find_plateau(curve, eps_grid, tol=0.05, min_run_length=2)

    assert result.has_plateau
    assert result.eps_range == (1e-4, 1e-2)
    assert result.representative_eps == 1e-3


def test_find_plateau_no_plateau_when_always_changing():
    eps_grid = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    base = torch.tensor([1.0, 2.0, 3.0, 4.0])
    curve = torch.stack([base * (2**i) for i in range(len(eps_grid))])

    result = find_plateau(curve, eps_grid, tol=0.05, min_run_length=2)

    assert not result.has_plateau
    assert result.eps_range is None
    assert result.representative_eps is None


def test_eps_sweep_on_quadratic_form_is_flat_everywhere():
    torch.manual_seed(0)
    d = 6
    A = torch.randn(d, d)
    A = (A + A.T) / 2

    h = torch.randn(d)
    v = torch.randn(d)

    def f(h: torch.Tensor) -> torch.Tensor:
        return 0.5 * h @ A @ h

    eps_grid = [1e-4, 1e-3, 1e-2, 1e-1]
    curve = eps_sweep(f, h, v, eps_grid)

    expected = A @ v
    for row in curve:
        torch.testing.assert_close(row, expected, rtol=1e-3, atol=1e-3)

    result = find_plateau(curve, eps_grid, tol=0.01, min_run_length=2)
    assert result.has_plateau


def test_estimator_agreement_quadratic_form_usable():
    torch.manual_seed(0)
    d = 6
    A = torch.randn(d, d)
    A = (A + A.T) / 2
    h = torch.randn(d)
    v = torch.randn(d)

    def f(h: torch.Tensor) -> torch.Tensor:
        return 0.5 * h @ A @ h

    result = estimator_agreement(f, h, v, tol=0.01)

    assert result.usable
    for pair, err in result.pairwise.items():
        assert err < 0.01, f"{pair} disagreed by {err}"


def test_estimator_agreement_flags_fd_disagreement_at_tight_tolerance():
    torch.manual_seed(1)
    d = 6
    A = torch.randn(d, d)
    A = (A + A.T) / 2
    h = torch.randn(d)
    v = torch.randn(d)

    def f(h: torch.Tensor) -> torch.Tensor:
        return 0.5 * h @ A @ h

    # rr and fr are both exact autodiff methods — they agree almost to floating-
    # point precision regardless of tolerance. fd has genuine O(eps^2)
    # discretization error, however small; an unreasonably tight tol exposes it.
    result = estimator_agreement(f, h, v, tol=1e-8)

    assert not result.usable
    assert result.pairwise[("rr", "fr")] < 1e-6
    assert result.pairwise[("rr", "fd")] > 1e-8


def test_ms_hvp_large_step_diverges_more_than_small_step():
    torch.manual_seed(0)
    d = 4
    h = torch.randn(d)

    def f(h: torch.Tensor) -> torch.Tensor:
        return (h**4).sum() / 4

    v_large = torch.randn(d) * 2.0
    v_small = v_large * 0.001  # same direction, much smaller

    result_large = ms_hvp(f, h, v_large, n_steps=200)
    result_small = ms_hvp(f, h, v_small, n_steps=200)

    assert result_large.reliability > 0.05
    assert result_small.reliability < 0.01


def test_ms_hvp_multi_step_converges_to_true_gradient_difference():
    torch.manual_seed(1)
    d = 4
    h = torch.randn(d)
    v = torch.randn(d) * 2.0

    def f(h: torch.Tensor) -> torch.Tensor:
        return (h**4).sum() / 4

    result = ms_hvp(f, h, v, n_steps=200)

    true_diff = (h + v) ** 3 - h**3  # exact, closed-form: grad(f)(h+v) - grad(f)(h)
    torch.testing.assert_close(result.multi_step, true_diff, rtol=1e-2, atol=1e-2)

    single_step_error = (result.single_step - true_diff).norm()
    multi_step_error = (result.multi_step - true_diff).norm()
    assert multi_step_error < single_step_error


def test_validity_score_valid_when_perturbation_within_radius():
    torch.manual_seed(0)
    d = 6
    A = torch.randn(d, d)
    A = (A + A.T) / 2
    h = torch.randn(d)

    v_raw = torch.randn(d)
    v = v_raw / v_raw.norm() * 0.05  # small perturbation, well inside the plateau

    def f(h: torch.Tensor) -> torch.Tensor:
        return 0.5 * h @ A @ h

    eps_grid = [1e-4, 1e-3, 1e-2, 1e-1]
    result = validity_score(f, h, v, eps_grid=eps_grid)

    assert result.valid
    assert result.taylor_radius == 1e-1
    assert result.validity_score > 1.0


def test_validity_score_invalid_when_perturbation_exceeds_radius():
    torch.manual_seed(0)
    d = 6
    A = torch.randn(d, d)
    A = (A + A.T) / 2
    h = torch.randn(d)

    v_raw = torch.randn(d)
    v = v_raw / v_raw.norm() * 5.0  # large perturbation, well outside the plateau

    def f(h: torch.Tensor) -> torch.Tensor:
        return 0.5 * h @ A @ h

    eps_grid = [1e-4, 1e-3, 1e-2, 1e-1]
    result = validity_score(f, h, v, eps_grid=eps_grid)

    assert not result.valid
    assert result.taylor_radius is not None
    assert result.validity_score is not None
    assert result.validity_score < 1.0
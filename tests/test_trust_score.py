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
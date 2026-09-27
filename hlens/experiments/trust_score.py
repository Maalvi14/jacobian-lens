import torch
from hlens.interventions import coordinate_swap
from hlens.hvp import hvp_norm_form

from collections.abc import Callable


def swap_delta(h: torch.Tensor, v_s: torch.Tensor, v_t: torch.Tensor, *, alpha: float = 1.0) -> torch.Tensor:
    return coordinate_swap(h, v_s, v_t, alpha=alpha) - h


def trust_score(
    f: Callable[[torch.Tensor], torch.Tensor],
    h: torch.Tensor,
    delta_h: torch.Tensor,
) -> float:
    quadratic_term = hvp_norm_form(f, h, delta_h)
    _, jvp_term = torch.func.jvp(f, (h,), (delta_h,))

    numerator = 0.5 * quadratic_term.norm()
    denominator = jvp_term.norm()
    return (numerator / denominator).item()
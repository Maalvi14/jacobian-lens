import torch
from hlens.hvp import hvp_fd, hvp_rr, hvp_fr

from collections.abc import Callable, Sequence

from dataclasses import dataclass

def eps_sweep(
    f: Callable[[torch.Tensor], torch.Tensor],
    h: torch.Tensor,
    v: torch.Tensor,
    eps_grid: Sequence[float],
) -> torch.Tensor:
    return torch.stack([hvp_fd(f, h, v, eps=eps) for eps in eps_grid])


@dataclass(frozen=True)
class PlateauResult:
    has_plateau: bool
    eps_range: tuple[float, float] | None
    representative_eps: float | None


def find_plateau(
    curve: torch.Tensor,
    eps_grid: Sequence[float],
    *,
    tol: float = 0.05,
    min_run_length: int = 2,
) -> PlateauResult:
    diffs = torch.tensor([
        (curve[i + 1] - curve[i]).norm().item() / curve[i].norm().item()
        for i in range(len(eps_grid) - 1)
    ])
    is_flat = diffs < tol

    best_start, best_len = None, 0
    run_start, run_len = None, 0
    for i, flat in enumerate(is_flat.tolist()):
        if flat:
            if run_start is None:
                run_start = i
            run_len += 1
        else:
            run_start, run_len = None, 0
        if run_len > best_len:
            best_start, best_len = run_start, run_len

    if best_len + 1 < min_run_length:
        return PlateauResult(has_plateau=False, eps_range=None, representative_eps=None)

    eps_lo = eps_grid[best_start]
    eps_hi = eps_grid[best_start + best_len]
    return PlateauResult(
        has_plateau=True,
        eps_range=(eps_lo, eps_hi),
        representative_eps=eps_grid[best_start + best_len // 2],
    )


@dataclass(frozen=True)
class EstimatorAgreement:
    pairwise: dict[tuple[str, str], float]
    usable: bool


def estimator_agreement(
    f: Callable[[torch.Tensor], torch.Tensor],
    h: torch.Tensor,
    v: torch.Tensor,
    *,
    tol: float = 0.05,
) -> EstimatorAgreement:
    h_leaf = h.clone().requires_grad_(True)
    hvp_via_rr = hvp_rr(f, h_leaf, v)
    hvp_via_fr = hvp_fr(f, h.detach(), v)
    hvp_via_fd = hvp_fd(f, h.detach(), v)

    def rel_error(a: torch.Tensor, b: torch.Tensor) -> float:
        return ((a - b).norm() / a.norm()).item()

    pairwise = {
        ("rr", "fr"): rel_error(hvp_via_rr, hvp_via_fr),
        ("rr", "fd"): rel_error(hvp_via_rr, hvp_via_fd),
        ("fr", "fd"): rel_error(hvp_via_fr, hvp_via_fd),
    }

    usable = all(err < tol for err in pairwise.values())
    return EstimatorAgreement(pairwise=pairwise, usable=usable)


@dataclass(frozen=True)
class MSHVPResult:
    single_step: torch.Tensor
    multi_step: torch.Tensor
    reliability: float


def ms_hvp(
    f: Callable[[torch.Tensor], torch.Tensor],
    h: torch.Tensor,
    v: torch.Tensor,
    *,
    n_steps: int = 4,
    hvp_fn: Callable[..., torch.Tensor] = hvp_fr,
) -> MSHVPResult:
    single_step = hvp_fn(f, h, v)

    v_step = v / n_steps
    multi_step = torch.zeros_like(v)
    h_current = h.clone()
    for _ in range(n_steps):
        multi_step = multi_step + hvp_fn(f, h_current, v_step)
        h_current = h_current + v_step

    reliability = ((single_step - multi_step).norm() / multi_step.norm()).item()

    return MSHVPResult(single_step=single_step, multi_step=multi_step, reliability=reliability)


@dataclass(frozen=True)
class ValidityResult:
    valid: bool
    taylor_radius: float | None
    perturbation_size: float
    validity_score: float | None


def validity_score(
    f: Callable[[torch.Tensor], torch.Tensor],
    h: torch.Tensor,
    v: torch.Tensor,
    *,
    eps_grid: Sequence[float],
    plateau_tol: float = 0.05,
) -> ValidityResult:
    v_hat = v / v.norm()
    curve = eps_sweep(f, h, v_hat, eps_grid)
    plateau = find_plateau(curve, eps_grid, tol=plateau_tol)

    perturbation_size = v.norm().item()

    if not plateau.has_plateau:
        return ValidityResult(
            valid=False, taylor_radius=None,
            perturbation_size=perturbation_size, validity_score=None,
        )

    taylor_radius = plateau.eps_range[1]
    score = taylor_radius / perturbation_size

    return ValidityResult(
        valid=score >= 1.0,
        taylor_radius=taylor_radius,
        perturbation_size=perturbation_size,
        validity_score=score,
    )
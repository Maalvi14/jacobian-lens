import torch
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Cotangent:
    c: torch.Tensor


@dataclass(frozen=True)
class Logit:
    token_id: int


@dataclass(frozen=True)
class Norm:
    pass


Contraction = Cotangent | Logit | Norm


def make_scalar_fn(
    f: Callable[[torch.Tensor], torch.Tensor],
    contraction: Contraction,
) -> Callable[[torch.Tensor], torch.Tensor]:
    match contraction:
        case Cotangent(c=c):
            return lambda h: torch.dot(c, f(h))
        case Logit(token_id=t):
            return lambda h: f(h)[..., t]
        case Norm():
            raise NotImplementedError(
                "Norm has no scalar form — it's computed via a separate "
                "jvp-of-jvp path (d/dε [J(h+εδh) δh]), not through make_scalar_fn"
            )


def hvp_rr(
    f: Callable[[torch.Tensor], torch.Tensor],
    h: torch.Tensor,
    v: torch.Tensor,
    *,
    create_graph: bool = False,
) -> torch.Tensor:
    assert h.requires_grad, (
        "h must require grad (e.g. via ActivationRecorder's start_graph_at) "
        "for hvp_rr to differentiate through it"
    )
    output = f(h)
    grad_h = torch.autograd.grad(output, h, create_graph=True)[0]
    dot = (grad_h * v).sum()
    hvp = torch.autograd.grad(dot, h, create_graph=create_graph)[0]
    return hvp


def hvp_fr(
    f: Callable[[torch.Tensor], torch.Tensor],
    h: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """...
    """
    grad_f = torch.func.grad(f)
    _, hvp = torch.func.jvp(grad_f, (h,), (v,))
    return hvp

def hvp_fd(
    f: Callable[[torch.Tensor], torch.Tensor],
    h: torch.Tensor,
    v: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    grad_f = torch.func.grad(f)
    grad_plus = grad_f(h + eps * v)
    grad_minus = grad_f(h - eps * v)
    return (grad_plus - grad_minus) / (2 * eps)


@dataclass(frozen=True)
class RestrictedHessian:
    H: torch.Tensor
    asymmetry: torch.Tensor


def restricted_hessian(
    f: Callable[[torch.Tensor], torch.Tensor],
    h: torch.Tensor,
    directions: torch.Tensor,
    *,
    hvp_fn: Callable[..., torch.Tensor] = hvp_fr,
) -> RestrictedHessian:
    k = directions.shape[0]
    H_restricted = torch.zeros(k, k)

    for j in range(k):
        v_j = directions[j]
        Hv_j = hvp_fn(f, h, v_j)
        H_restricted[:, j] = directions @ Hv_j

    asymmetry = (H_restricted - H_restricted.T).abs().max()
    H_sym = (H_restricted + H_restricted.T) / 2

    return RestrictedHessian(H=H_sym, asymmetry=asymmetry)


def hvp_norm_form(
    f: Callable[[torch.Tensor], torch.Tensor],
    h: torch.Tensor,
    delta_h: torch.Tensor,
) -> torch.Tensor:
    def jvp_fn(x: torch.Tensor) -> torch.Tensor:
        _, out = torch.func.jvp(f, (x,), (delta_h,))
        return out

    _, quadratic_term = torch.func.jvp(jvp_fn, (h,), (delta_h,))
    return quadratic_term
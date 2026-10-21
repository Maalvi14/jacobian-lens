"""Per-context building blocks shared by the curvature experiments: capture the
residual at (layer, position) for a prompt, and build the scalar/vector functions
of that residual that the HVP estimators differentiate."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from hlens.functional import make_output_fn
from jlens import JacobianLens
from jlens.hooks import ActivationRecorder
from jlens.protocol import LensModel


@dataclass(frozen=True)
class ContextPoint:
    """One (prompt, layer, position): the residual stream there and everything
    needed to rebuild functions of it."""

    input_ids: torch.Tensor  # [1, seq]
    layer: int
    position: int
    h: torch.Tensor  # [1, seq, d_model], detached, in the model's dtype

    @property
    def x0(self) -> torch.Tensor:
        """The residual at the point's own position, [d_model]."""
        return self.h[0, self.position]


def capture_context(
    model: LensModel,
    prompt: str,
    layer: int,
    *,
    position: int,
    max_length: int = 64,
) -> ContextPoint:
    """Residual stream at `layer` for `prompt`; `position` may be negative."""
    ids = model.encode(prompt, max_length=max_length)
    with torch.no_grad(), ActivationRecorder(model.layers, at=[layer]) as rec:
        model.forward(ids)
        h = rec.activations[layer].detach().clone()
    return ContextPoint(
        input_ids=ids, layer=layer, position=position % ids.shape[1], h=h
    )


def logit_fn(
    model: LensModel, point: ContextPoint, token_id: int
) -> Callable[[torch.Tensor], torch.Tensor]:
    """h [1, seq, d_model] at point.layer -> the 0-dim logit of `token_id` at
    point.position of the model's output."""
    position = point.position
    return make_output_fn(
        model, point.layer, lambda x: model.unembed(x[0, position])[token_id]
    )


def cotangent_fn(
    model: LensModel, point: ContextPoint, c: torch.Tensor
) -> Callable[[torch.Tensor], torch.Tensor]:
    """h [1, seq, d_model] at point.layer -> <c, final-layer residual at
    point.position> (0-dim). A fixed `c` makes the scalar comparable across
    contexts; c = an unembedding direction u_t gives the pre-norm token logit."""
    position = point.position
    return make_output_fn(model, point.layer, lambda x: (x[0, position] * c).sum())


def at_position(
    f: Callable[[torch.Tensor], torch.Tensor], point: ContextPoint
) -> tuple[Callable[[torch.Tensor], torch.Tensor], torch.Tensor]:
    """Restrict a function of the full residual [1, seq, d] to the point's own
    position: returns (f1, x0) with f1 taking a [d_model] vector that replaces
    h[0, position] and x0 = the actual residual there. Directions of the
    restricted Hessian then live in plain d_model space."""
    p = point.position
    h = point.h

    def f1(x: torch.Tensor) -> torch.Tensor:
        return f(torch.cat([h[:, :p], x[None, None], h[:, p + 1 :]], dim=1))

    return f1, point.x0


def lens_direction_bank(
    u_rows: torch.Tensor, lens: JacobianLens, layer: int, *, normalize: bool = True
) -> torch.Tensor:
    """Lens vectors at `layer` for tokens whose unembedding gradients are
    `u_rows` [k, d_model]: u_t @ J_l, optionally unit-normalised. [k, d_model]."""
    v = u_rows @ lens.jacobians[layer].to(u_rows.device)
    return v / v.norm(dim=1, keepdim=True) if normalize else v

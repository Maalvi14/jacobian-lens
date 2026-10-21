"""Pathological stress test (plan §5; the Phase-2 gate): a constructed high-curvature
case, positions straddling a sharp top-1 flip under embedding interpolation, in which
the single-step HVP should visibly degrade and multi-step HVP plus the validity gate
should recover.

Construction. For each (carrier template, country pair A/B) the {W} embedding is
interpolated (hlens.interpolation); the scalar is the lens-logit difference
logit_A - logit_B read at `read_layer` at the {W} position, as a function of the
residual at `src_layer` (the layer-to-layer map the H6 hypothesis is about). The knee
alpha* is where A's reciprocal-rank share crosses 0.5. Two short paths in the residual
at `src_layer`, from the interpolated forward passes:
  knee:    alpha* -/+ half_width  (straddles the flip);
  control: 0 -> 2*half_width      (same alpha length, far from the knee).
Along each path v = x_hi - x_lo, the exact first-order change of the gradient is the
secant S = grad f(x_hi) - grad f(x_lo) (= the integral of H(x_lo + t v) v dt); we
score single-step HVP, multi-step HVP (n=4, 16) against S, and run the validity gate.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch

from hlens.contexts import ContextPoint, at_position
from hlens.data import IgnitionConfig
from hlens.functional import make_slice_fn
from hlens.harness import ms_hvp, validity_score
from hlens.interpolation import forward_at_alpha, make_setup, rank_share
from hlens.hvp import hvp_fr


@dataclass(frozen=True)
class PathResult:
    path_norm: float
    single_err: float  # ||H(x_lo) v - S|| / ||S||
    multi4_err: float
    multi16_err: float
    reliability16: float  # ms_hvp's own single-vs-multi divergence (needs no ground truth)
    valid: bool  # validity gate: Taylor radius >= path length
    validity_score: float | None


@dataclass(frozen=True)
class StressTrial:
    template: str
    word_a: str
    word_b: str
    alpha_star: float
    knee: PathResult
    control: PathResult


def _find_knee(qwen, setup, read_layer: int, grid: int) -> float | None:
    alphas = [i / (grid - 1) for i in range(grid)]
    shares = []
    for a in alphas:
        h = forward_at_alpha(qwen, setup, a, [read_layer])[read_layer][0, setup.position].float()
        logits = qwen.model.unembed(qwen.lens.transport(h, read_layer)).float()
        shares.append(rank_share(logits, setup.id_a, setup.id_b))
    for i in range(grid - 1):
        if shares[i] < 0.5 <= shares[i + 1]:
            t = (0.5 - shares[i]) / (shares[i + 1] - shares[i])
            return alphas[i] + t * (alphas[i + 1] - alphas[i])
    return None


def _score_path(f1, x_lo: torch.Tensor, x_hi: torch.Tensor) -> PathResult:
    v = x_hi - x_lo
    grad = torch.func.grad(f1)
    secant = grad(x_hi) - grad(x_lo)

    def err(t: torch.Tensor) -> float:
        return ((t - secant).norm() / secant.norm()).item()

    m4, m16 = ms_hvp(f1, x_lo, v, n_steps=4), ms_hvp(f1, x_lo, v, n_steps=16)
    n = v.norm().item()
    vs = validity_score(f1, x_lo, v, eps_grid=[n * s for s in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)])
    return PathResult(
        path_norm=n, single_err=err(m16.single_step), multi4_err=err(m4.multi_step),
        multi16_err=err(m16.multi_step), reliability16=m16.reliability,
        valid=vs.valid, validity_score=vs.validity_score,
    )


def run_stress_test(
    qwen,
    cfg: IgnitionConfig,
    *,
    src_layer: int = 8,
    read_layer: int = 20,
    half_width: float = 0.04,
    n_templates: int = 3,
    n_pairs: int = 5,
    grid: int = 41,
) -> list[StressTrial]:
    model, lens = qwen.model, qwen.lens
    slice_fn = make_slice_fn(model, src_layer, read_layer)
    J = lens.jacobians[read_layer].to(next(iter(model.layers[0].parameters())).device)

    def single_token(w: str) -> bool:
        return len(qwen.tokenizer.encode(" " + w, add_special_tokens=False)) == 1

    countries = [c for c in cfg.countries_12 if single_token(c)]
    pairs = list(itertools.combinations(countries, 2))[:n_pairs]
    trials = []
    for template in cfg.ctx_templates[:n_templates]:
        for word_a, word_b in pairs:
            setup = make_setup(qwen, template, word_a, word_b)
            a_star = _find_knee(qwen, setup, read_layer, grid)
            if a_star is None or not (2 * half_width < a_star < 1 - 2 * half_width):
                continue

            def at(alpha):
                return forward_at_alpha(qwen, setup, alpha, [src_layer])[src_layer]

            def f_full(h, pos=setup.position, ia=setup.id_a, ib=setup.id_b):
                logits = model.unembed(slice_fn(h)[0, pos].float() @ J.T).float()
                return logits[ia] - logits[ib]

            lo = at(a_star - half_width)
            point = ContextPoint(input_ids=setup.input_ids, layer=src_layer, position=setup.position, h=lo)
            f1, x_lo = at_position(f_full, point)
            x_hi = at(a_star + half_width)[0, setup.position]
            knee = _score_path(f1, x_lo, x_hi)

            c_lo, c_hi = at(0.0), at(2 * half_width)
            cpoint = ContextPoint(input_ids=setup.input_ids, layer=src_layer, position=setup.position, h=c_lo)
            cf1, cx_lo = at_position(f_full, cpoint)
            control = _score_path(cf1, cx_lo, c_hi[0, setup.position])
            trials.append(StressTrial(template, word_a, word_b, a_star, knee, control))
    return trials

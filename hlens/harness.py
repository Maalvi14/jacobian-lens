import csv
import itertools
import json
import os
import subprocess
import time

import torch
from hlens.hvp import hvp_fd, hvp_rr, hvp_fr, restricted_hessian

from collections.abc import Callable, Sequence, Mapping

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
    eps: float = 1e-3,
) -> EstimatorAgreement:
    h_leaf = h.clone().requires_grad_(True)
    hvp_via_rr = hvp_rr(f, h_leaf, v)
    hvp_via_fr = hvp_fr(f, h.detach(), v)
    hvp_via_fd = hvp_fd(f, h.detach(), v, eps=eps)

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

@dataclass(frozen=True)
class AveragingResult:
    ratio: float  # ||mean_p H_p|| / mean_p ||H_p||
    null_mean: float  # same ratio after giving every H_p an independent random sign
    null_p95: float
    n: int


def averaging_ratio(
    Hs: torch.Tensor, *, n_null: int = 500, seed: int = 0
) -> AveragingResult:
    """How much of the per-context Hessians survives averaging? `Hs` is [N, ...]
    (one restricted Hessian, or gradient, per context). ratio = ||E[H]|| / E||H||:
    1 if every context has the same H, ~0 if they cancel. The sign-shuffled null
    (each H_p multiplied by an independent +-1, which destroys any consistent
    structure but keeps every ||H_p||) is what independent contexts would give
    (about 1/sqrt(N) for matrices of equal norm), so the ratio only means
    something relative to it."""
    n = Hs.shape[0]
    flat = Hs.reshape(n, -1).float()
    norms = flat.norm(dim=1)
    ratio = (flat.mean(dim=0).norm() / norms.mean()).item()
    g = torch.Generator().manual_seed(seed)
    signs = torch.randint(0, 2, (n_null, n), generator=g).float().mul(2).sub(1).to(flat.device)
    null = ((signs @ flat) / n).norm(dim=1) / norms.mean()
    return AveragingResult(
        ratio=ratio, null_mean=null.mean().item(),
        null_p95=null.quantile(0.95).item(), n=n,
    )



# --- randomized-weights null (Adebayo-style sanity check) ---------------------------


def randomize_model(hf_model, tokenizer, *, mode: str = "reinit", seed: int = 0):
    """Randomize the weights of `hf_model` IN PLACE and return it wrapped with
    `jlens.from_hf`. Pass a fresh copy from `hlens.loading.load_hf()`: this
    destroys the weights, so never hand it your only in-memory copy.

    mode="reinit":  HF's own initialization (std = config.initializer_range; the
                    model-specific rules for norms, Gated-DeltaNet A_log/dt_bias,
                    rotary buffers), as at the start of training. Loaded weights are
                    flagged `_is_hf_initialized` and would be skipped, so the flags
                    are cleared first.
    mode="shuffle": permute the entries within every parameter tensor, keeping each
                    tensor's value distribution but destroying all structure.

    The pre-fitted lens is meaningless on the result; a faithful null re-runs the
    pipeline, i.e. `fit_null_lens` on the randomized model."""
    from jlens import from_hf

    if mode == "reinit":
        for module in hf_model.modules():
            module._is_hf_initialized = False
        for t in [*hf_model.parameters(), *hf_model.buffers()]:
            t._is_hf_initialized = False
        cuda = sorted({p.device.index for p in hf_model.parameters() if p.device.type == "cuda"})
        with torch.random.fork_rng(devices=cuda):
            torch.manual_seed(seed)
            hf_model.init_weights()
    elif mode == "shuffle":
        with torch.no_grad():
            for i, (_, p) in enumerate(hf_model.named_parameters()):  # tied weights appear once
                g = torch.Generator(device=p.device).manual_seed(seed + i)
                perm = torch.randperm(p.numel(), generator=g, device=p.device)
                p.copy_(p.flatten()[perm].view_as(p))
    else:
        raise ValueError(f"unknown mode {mode!r} (expected 'reinit' or 'shuffle')")
    return from_hf(hf_model, tokenizer)


def fit_null_lens(
    model,
    prompts: Sequence[str],
    *,
    source_layers: Sequence[int],
    dim_batch: int = 32,
    max_seq_len: int = 64,
    path: str | None = None,
    checkpoint_every: int | None = 5,
):
    """`jlens.fit` on the (randomized) model, cached: if `path` exists the finished
    lens is loaded; otherwise the fit checkpoints to `path + '.ckpt'` (resumable) and
    the result is saved to `path`."""
    from jlens import JacobianLens
    from jlens.fitting import fit

    if path and os.path.exists(path):
        return JacobianLens.load(path)
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)  # fit's atomic checkpoint needs the directory
    lens = fit(
        model, list(prompts), source_layers=list(source_layers), dim_batch=dim_batch,
        max_seq_len=max_seq_len, checkpoint_path=(path + ".ckpt") if path else None,
        checkpoint_every=checkpoint_every,
    )
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        lens.save(path)
    return lens


# --- per-context restricted Hessians and their null distributions -------------------


@dataclass(frozen=True)
class Collected:
    H: torch.Tensor  # [N, k, k] symmetrised restricted Hessians
    g: torch.Tensor  # [N, k] first-order projections v_i . grad f
    norms: list[float]  # ||x_p|| of the residual at each sampled position
    asymmetry: torch.Tensor  # [N] pre-symmetrisation ||H - H^T||_max per context
    positions: list[int]


def collect_restricted_hessians(
    model,
    lens,
    prompts: Sequence[str],
    *,
    layer: int,
    out_token: int,
    token_ids: Sequence[int] | None = None,
    directions: torch.Tensor | None = None,
    max_length: int = 64,
    seed: int = 0,
    skip_first: int | None = None,
    hvp_fn: Callable[..., torch.Tensor] = hvp_fr,
    log_every: int = 0,
) -> Collected:
    """For each prompt: one random valid position p; the scalar is the logit of
    `out_token` at p; the directions are FIXED across contexts (so the matrices are
    comparable): either the unit lens vectors of `token_ids` at `layer`, or an explicit
    `directions` [k, d_model] tensor."""
    from hlens.contexts import at_position, capture_context, lens_direction_bank, logit_fn
    from hlens.jspace import unembed_directions
    from jlens.fitting import SKIP_FIRST_N_POSITIONS, valid_position_mask

    if (token_ids is None) == (directions is None):
        raise ValueError("pass exactly one of token_ids / directions")
    skip = SKIP_FIRST_N_POSITIONS if skip_first is None else skip_first
    g = torch.Generator().manual_seed(seed)
    D = directions if directions is not None else lens_direction_bank(
        unembed_directions(model, list(token_ids)), lens, layer
    )
    Hs, gs, norms, asyms, positions = [], [], [], [], []
    t0 = time.time()
    for i, prompt in enumerate(prompts):
        ids = model.encode(prompt, max_length=max_length)
        valid = valid_position_mask(ids.shape[1], skip_first=skip).nonzero().squeeze(1)
        pos = valid[torch.randint(len(valid), (1,), generator=g)].item()
        point = capture_context(model, prompt, layer, position=pos, max_length=max_length)
        f1, x0 = at_position(logit_fn(model, point, out_token), point)
        d = D.to(x0.device, x0.dtype)
        rh = restricted_hessian(f1, x0, d, hvp_fn=hvp_fn)
        Hs.append(rh.H.cpu())
        asyms.append(rh.asymmetry.cpu())
        gs.append((d @ torch.func.grad(f1)(x0)).cpu())
        norms.append(x0.norm().item())
        positions.append(point.position)
        if log_every and (i + 1) % log_every == 0:
            print(f"  {i + 1}/{len(prompts)} contexts, {time.time() - t0:.0f}s", flush=True)
    return Collected(torch.stack(Hs), torch.stack(gs), norms, torch.stack(asyms), positions)


@dataclass(frozen=True)
class NullDistributions:
    name: str
    diag: torch.Tensor  # directional-curvature samples H_ii, flattened over contexts
    cross: torch.Tensor  # cross-term samples H_ij (i != j), flattened over contexts
    meta: dict

    def quantile(self, which: str, q: float) -> float:
        return getattr(self, which).abs().quantile(q).item()

    def above(self, value: float, which: str = "cross", q: float = 0.99) -> bool:
        """Is |value| beyond the q-quantile of |null samples|?"""
        return abs(value) > self.quantile(which, q)


def null_distributions(
    model,
    lens,
    prompts: Sequence[str],
    *,
    name: str,
    layer: int,
    token_ids: Sequence[int],
    out_token: int,
    out_dir: str = "results/nulls",
    meta: Mapping | None = None,
    **collect_kwargs,
) -> NullDistributions:
    """Record the directional-curvature (diagonal) and cross-term (off-diagonal)
    distributions of restricted Hessians for a (randomized) model + its own lens, to
    disk, once. Every Phase-2 script loads these instead of recomputing."""
    c = collect_restricted_hessians(
        model, lens, prompts, layer=layer, token_ids=token_ids, out_token=out_token, **collect_kwargs
    )
    H, g, norms = c.H, c.g, c.norms
    k = H.shape[1]
    off = ~torch.eye(k, dtype=torch.bool)
    info = {"layer": layer, "token_ids": list(token_ids), "out_token": out_token,
            "n_contexts": len(prompts), **(dict(meta) if meta else {})}
    nd = NullDistributions(name=name, diag=torch.diagonal(H, dim1=1, dim2=2).flatten(), cross=H[:, off].flatten(), meta=info)
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"H": H, "g": g, "norms": norms, "diag": nd.diag, "cross": nd.cross, "meta": info}, f"{out_dir}/{name}.pt")
    with open(f"{out_dir}/{name}.json", "w") as f:
        json.dump({**info, "abs_quantiles": {w: {str(q): nd.quantile(w, q) for q in (0.5, 0.9, 0.99)} for w in ("diag", "cross")}}, f)
    return nd


def load_null(name: str, out_dir: str = "results/nulls") -> NullDistributions:
    d = torch.load(f"{out_dir}/{name}.pt", weights_only=False)
    return NullDistributions(name=name, diag=d["diag"], cross=d["cross"], meta=d["meta"])


# --- ablation table: a config-sweep runner ------------------------------------------


def ablation_table(
    run_fn: Callable[..., Mapping[str, float]],
    grid: Mapping[str, Sequence],
    *,
    fixed: Mapping | None = None,
    csv_path: str | None = None,
) -> list[dict]:
    """Run `run_fn(**config, **fixed)` over the Cartesian product of `grid` and
    return one tidy row per config: the config values plus whatever metrics
    `run_fn` returns. Written to `csv_path` if given. The sweep itself is generic;
    `hlens.experiments.ablation_battery` supplies the second-order battery."""
    keys = list(grid)
    rows = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        config = dict(zip(keys, combo))
        rows.append({**config, **dict(run_fn(**config, **(dict(fixed) if fixed else {})))})
    if csv_path:
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
            w.writeheader()
            w.writerows(rows)
    return rows


# --- benchmark ----------------------------------------------------------------------


def _power_mode() -> str | None:
    """Current Jetson power mode name via `nvpmodel -q` (readable without sudo);
    None if unavailable. Clocks (`jetson_clocks`) need root to inspect, so only the
    power mode is verified here."""
    try:
        out = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        if "Power Mode" in line:
            return line.split(":", 1)[-1].strip()
    return None


@dataclass(frozen=True)
class BenchmarkResult:
    seconds_per_hvp: float
    seconds_per_restricted_hessian: float
    seconds_per_lens_apply: float
    ratio_restricted_h_to_lens_apply: float
    power_mode: str | None
    k: int

    def meets_milestone(self, max_ratio: float = 3.0) -> bool:
        """The plan's cost milestone: restricted H at one position <= 2-3x one
        lens.apply. Only meaningful when `power_mode == "MAXN"`."""
        return self.ratio_restricted_h_to_lens_apply <= max_ratio


def benchmark(
    qwen,
    prompt: str,
    *,
    layer: int,
    token_ids: Sequence[int],
    out_token: int,
    position: int = -2,
    repeats: int = 3,
    require_max_power: bool = True,
) -> BenchmarkResult:
    """Wall-clock per HVP, per restricted Hessian (k HVPs), and per `lens.apply`
    call, and the ratio the plan's cost milestone is about (restricted H at one
    position <= 2-3x one lens.apply). Refuses to run unless the Thor is in MAXN
    (run `sudo nvpmodel -m 0 && sudo jetson_clocks` first) unless
    `require_max_power=False`."""
    from hlens.contexts import at_position, capture_context, lens_direction_bank, logit_fn
    from hlens.jspace import unembed_directions

    mode = _power_mode()
    if require_max_power and mode != "MAXN":
        raise RuntimeError(
            f"power mode is {mode!r}, not MAXN: run `sudo nvpmodel -m 0 && sudo jetson_clocks` "
            "before benchmarking (or pass require_max_power=False for a non-comparable timing)"
        )
    model, lens = qwen.model, qwen.lens
    point = capture_context(model, prompt, layer, position=position)
    f1, x0 = at_position(logit_fn(model, point, out_token), point)
    D = lens_direction_bank(unembed_directions(model, list(token_ids)), lens, layer).to(x0.device, x0.dtype)

    def timed(fn) -> float:
        fn()  # warm-up: kernels, kwargs cache
        times = []
        for _ in range(repeats):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            times.append(time.perf_counter() - t0)
        return sorted(times)[len(times) // 2]

    t_hvp = timed(lambda: hvp_fr(f1, x0, D[0]))
    t_rh = timed(lambda: restricted_hessian(f1, x0, D))
    t_lens = timed(lambda: lens.apply(model, prompt, positions=[position]))
    return BenchmarkResult(t_hvp, t_rh, t_lens, t_rh / t_lens, mode, len(token_ids))

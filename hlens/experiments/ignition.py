"""H6 (ignition as a curvature event; research notes §3, paper §4.1.1 / A.15).

Protocol (data/experiments/README.md, ignition): the {W} token's embedding is replaced by
(1 - a) e_B + a e_A, a swept 0 -> 1, and the residual at {W} is recorded at every layer.

Transition readout (the paper's lens-free one): the projection share
    share(l, a) = <h_l(a) - h_l(0), h_l(1) - h_l(0)> / ||h_l(1) - h_l(0)||^2
is 0 at a = 0 and 1 at a = 1 by construction; early layers track a smoothly, from the workspace
onset it switches sharply at a trial-specific threshold a*(l) (share = 0.5). The transition width
w(l) is the a-distance between share = 0.1 and share = 0.9.

Curvature readout (layer-to-layer, H-lens): the scalar f_{l -> m}(h_l) = <c_m, h_m> with c_m the unit
vector along the A-B line at the target layer m = l + gap (the commitment axis; it is what the share
measures), as a function of the residual at {W} at layer l. Along the interpolation path with tangent
delta = dh_l/da at the {W} position:
    tau_path(l, a) = || 1/2 delta^T H delta || * da / |J delta|         second- to first-order ratio of an a-step
    lambda_max(l, a) = the leading |eigenvalue| of H_f (Lanczos over the HVP operator)
Two guards from the notes: multi-step HVP near knees, and a knee-vs-Taylor-radius diagnosis (a curvature
spike that coincides with the boundary of the local Taylor radius is an estimator artifact).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import torch

from hlens.contexts import ContextPoint, at_position
from hlens.functional import make_slice_fn
from hlens.interpolation import InterpolationSetup, forward_at_alpha, make_setup
from hlens.hvp import hvp_fr


def projection_share(h_a: torch.Tensor, h_b: torch.Tensor, h: torch.Tensor) -> float:
    """Position of `h` along the line from pure-B (h_b) to pure-A (h_a)."""
    d = h_a - h_b
    return ((h - h_b) @ d / (d @ d)).item()


@dataclass(frozen=True)
class Transition:
    threshold: float | None  # alpha where share crosses 0.5 (first crossing, linear interpolation)
    width: float | None  # alpha-distance between share 0.1 and 0.9 (None if either crossing is missing)
    monotone: bool


def _first_crossing(alphas: np.ndarray, share: np.ndarray, level: float) -> float | None:
    for i in range(len(alphas) - 1):
        lo, hi = share[i], share[i + 1]
        if lo < level <= hi:
            t = (level - lo) / (hi - lo)
            return float(alphas[i] + t * (alphas[i + 1] - alphas[i]))
    return None


def transition(alphas, share) -> Transition:
    a, s = np.asarray(alphas, float), np.asarray(share, float)
    t50, t10, t90 = (_first_crossing(a, s, v) for v in (0.5, 0.1, 0.9))
    width = (t90 - t10) if (t10 is not None and t90 is not None) else None
    return Transition(t50, width, bool(np.all(np.diff(s) > -0.05)))


def lanczos_top_abs_eig(hvp, dim: int, *, iters: int = 12, seed: int = 0, device="cpu", dtype=torch.float32) -> float:
    """Leading |eigenvalue| of a symmetric operator from `iters` Lanczos steps (full
    re-orthogonalisation; `hvp(v)` is the operator applied to a [dim] vector)."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(dim, generator=g).to(device, dtype)
    q = q / q.norm()
    Q, alphas, betas = [q], [], []
    for j in range(iters):
        w = hvp(Q[-1]).to(dtype)
        a = (w @ Q[-1]).item()
        alphas.append(a)
        w = w - a * Q[-1] - (betas[-1] * Q[-2] if betas else 0.0)
        for qq in Q:
            w = w - (w @ qq) * qq  # re-orthogonalise
        b = w.norm().item()
        if b < 1e-10 or j == iters - 1:
            break
        betas.append(b)
        Q.append(w / b)
    k = len(alphas)
    T = np.diag(alphas) + np.diag(betas[: k - 1], 1) + np.diag(betas[: k - 1], -1)
    return float(np.abs(np.linalg.eigvalsh(T)).max())


def commitment_function(model, layer: int, target: int, c: torch.Tensor, position: int, slice_factory=make_slice_fn):
    """f(h_l [1, seq, d]) = <c, h_target at `position`> through blocks l+1 .. target."""
    slice_fn = slice_factory(model, layer, target)
    return lambda h: (slice_fn(h)[0, position] * c).sum()


def path_curvature(qwen, setup: InterpolationSetup, alpha: float, layer: int, gap: int, *, d_alpha: float = 0.05, lanczos_iters: int | None = None, ends: dict | None = None) -> dict:
    """tau_path (and optionally lambda_max) of the layer-l -> layer-(l+gap) commitment map at `alpha`."""
    from hlens.experiments.trust_score import taylor_terms

    model = qwen.model
    target = layer + gap
    pos = setup.position
    if ends is None:
        ends = {a: forward_at_alpha(qwen, setup, a, [layer, target]) for a in (0.0, 1.0)}
    c = ends[1.0][target][0, pos] - ends[0.0][target][0, pos]
    c = c / c.norm()
    eps = 0.5 * d_alpha
    lo = forward_at_alpha(qwen, setup, max(alpha - eps, 0.0), [layer])[layer]
    hi = forward_at_alpha(qwen, setup, min(alpha + eps, 1.0), [layer])[layer]
    mid = forward_at_alpha(qwen, setup, alpha, [layer])[layer]
    delta = torch.zeros_like(mid)
    delta[0, pos] = (hi[0, pos] - lo[0, pos]) / (min(alpha + eps, 1.0) - max(alpha - eps, 0.0))  # d h / d alpha
    point = ContextPoint(input_ids=setup.input_ids, layer=layer, position=pos, h=mid)
    f_full = commitment_function(model, layer, target, c, pos)
    quad, lin = taylor_terms(f_full, mid, d_alpha * delta)
    out = {"tau": (quad.abs() / lin.abs()).item(), "first_order": lin.item(), "second_order": quad.item(), "delta_norm": delta.norm().item()}
    if lanczos_iters:
        f1, x0 = at_position(f_full, point)
        out["lambda_max"] = lanczos_top_abs_eig(lambda v: hvp_fr(f1, x0, v), x0.numel(), iters=lanczos_iters, device=x0.device, dtype=x0.dtype)
    return out


# --- the pre-registered experiment (hlens/docs/preregistration_h6.md) -------------------------------------------

import hashlib
import json
import os
import sys
import time
from types import SimpleNamespace

from scipy import stats as sstats

PREREG = "hlens/docs/preregistration_h6.md"
OUT_DIR = "results/h6"
LAYERS = [4, 8, 10, 12, 14, 16, 20, 24]
OFFSETS = [-0.2, -0.1, 0.0, 0.1, 0.2]
GAP = 4
N_CARRIERS, N_PAIRS = 6, 4


def prereg_sha256(path: str = PREREG) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def choose_trials(qwen, cfg) -> list[tuple[str, str, str]]:
    import random

    single = [c for c in cfg.countries_12 if len(qwen.tokenizer.encode(" " + c, add_special_tokens=False)) == 1]
    pairs = list(itertools.combinations(single, 2))
    random.Random(0).shuffle(pairs)
    return [(t, a, b) for t in cfg.ctx_templates[:N_CARRIERS] for a, b in pairs[:N_PAIRS]]


def run_trial(qwen, template: str, A: str, B: str, *, lanczos_iters: int = 8, layers=None, offsets=None, gap: int = GAP, mid: int = 16, clean: bool = False) -> dict | None:
    layers, offsets = layers or LAYERS, offsets or OFFSETS
    setup = make_setup(qwen, template, A, B, clean=clean)
    pos = setup.position
    alphas = np.linspace(0, 1, 41)
    H = {float(a): forward_at_alpha(qwen, setup, float(a), list(range(qwen.model.n_layers - 3))) for a in alphas}
    width, thr = {}, {}
    for m in range(qwen.model.n_layers - 3):
        hb, ha = H[0.0][m][0, pos], H[1.0][m][0, pos]
        tr = transition(alphas, [projection_share(ha, hb, H[float(a)][m][0, pos]) for a in alphas])
        width[m], thr[m] = tr.width, tr.threshold
    a_star = thr.get(mid)
    if a_star is None:
        return None
    cells = {}
    for l in layers:
        ends = {a: forward_at_alpha(qwen, setup, a, [l, l + gap]) for a in (0.0, 1.0)}
        for o in offsets:
            a = float(np.clip(a_star + o, 0.02, 0.98))
            r = path_curvature(qwen, setup, a, l, gap, lanczos_iters=lanczos_iters, ends=ends)
            cells[f"{l}|{o}"] = {"lambda": r["lambda_max"], "tau": r["tau"], "first_order": r["first_order"]}
    return {"template": template, "A": A, "B": B, "alpha_star": a_star, "width": width, "threshold": thr, "cells": cells}


def run_arm(qwen, name: str, *, log=print, layers=None, gap: int = GAP, mid: int = 16, clean: bool = False,
            out_dir: str = OUT_DIR, prereg: str = PREREG, checkpoint: str | None = None) -> dict:
    """One arm of the H6 protocol. The defaults are the Qwen3.5-4B pre-registered values; another model passes
    depth-matched `layers`, `gap`, `mid` (and `clean=True` for Gemma). `checkpoint` (.jsonl): finished trials are
    appended and a rerun resumes from the file."""
    layers = layers or LAYERS
    cfg = __import__("hlens.data", fromlist=["load_ignition"]).load_ignition("data/experiments/ignition.json")
    trials, t0 = [], time.time()
    done = {}
    if checkpoint and os.path.exists(checkpoint):
        for line in open(checkpoint):
            r = json.loads(line)
            done[(r["template"], r["A"], r["B"])] = r
        log(f"  resuming: {len(done)} trials in {checkpoint}")
    for i, (t, a, b) in enumerate(choose_trials(qwen, cfg)):
        if (t, a, b) in done:
            trials.append(done[(t, a, b)])
            continue
        r = run_trial(qwen, t, a, b, layers=layers, gap=gap, mid=mid, clean=clean)
        if r is not None:
            trials.append(r)
            if checkpoint:
                os.makedirs(os.path.dirname(checkpoint) or ".", exist_ok=True)
                with open(checkpoint, "a") as fh:
                    fh.write(json.dumps(r, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
        log(f"  {name}: trial {i + 1}, {time.time() - t0:.0f}s")
    res = {"arm": name, "prereg_sha256": prereg_sha256(prereg), "layers": layers, "offsets": OFFSETS, "gap": gap, "mid": mid, "trials": trials}
    os.makedirs(out_dir, exist_ok=True)
    json.dump(res, open(f"{out_dir}/{name}.json", "w"))
    return res


def _lam(trial, l, o, key="lambda"):
    return trial["cells"][f"{l}|{o}"][key]


def analyze_h6(real: dict, null: dict, *, n_boot: int = 10000, seed: int = 0, key: str = "lambda", prereg: str = PREREG) -> dict:
    L, rng = real["layers"], np.random.default_rng(seed)

    def profile(trials):  # Lambda(l): median over trials of lambda at the threshold
        return {l: float(np.median([_lam(t, l, 0.0, key) for t in trials])) for l in L}

    def widths(trials):
        return {l: float(np.median([t["width"][str(l)] if str(l) in t["width"] else t["width"][l] for t in trials if (t["width"].get(str(l), t["width"].get(l)) is not None)])) for l in L}

    def window(W):
        drops = {(L[i], L[i + 1]): W[L[i + 1]] - W[L[i]] for i in range(len(L) - 1)}
        return min(drops, key=drops.get)

    def in_window(trials):
        Lam, W = profile(trials), widths(trials)
        lo, hi = window(W)
        l_peak = max(L, key=lambda l: Lam[l])
        return lo - 2 <= l_peak <= hi + 2, l_peak, (lo, hi)

    ok, l_peak, win = in_window(real["trials"])
    boots = np.mean([in_window([real["trials"][i] for i in rng.integers(0, len(real["trials"]), len(real["trials"]))])[0] for _ in range(2000)])
    t1 = bool(ok and boots >= 0.8)

    def d_stat(trials):
        return np.array([_lam(t, l, 0.0, key) - 0.5 * (_lam(t, l, -0.2, key) + _lam(t, l, 0.2, key)) for t in trials for l in L])

    Dr, Dn = d_stat(real["trials"]), d_stat(null["trials"])
    p2 = float(sstats.wilcoxon(Dr, alternative="greater").pvalue) if np.any(Dr != 0) else 1.0
    t2 = p2 < 0.01
    nr, nn = len(real["trials"]), len(null["trials"])
    per_t = lambda D, n: D.reshape(n, len(L)).mean(1)
    pr, pn = per_t(Dr, nr), per_t(Dn, nn)
    diffs = [pr[rng.integers(0, nr, nr)].mean() - pn[rng.integers(0, nn, nn)].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    t3 = lo > 0
    return {
        "prereg_sha256": {"real": real["prereg_sha256"], "null": null["prereg_sha256"], "current": prereg_sha256(prereg)},
        "n_trials": {"real": nr, "null": nn}, "key": key,
        "T1": {"lambda_peak_layer": l_peak, "width_collapse_window": list(win), "bootstrap_fraction_in_window": float(boots), "passed": t1,
               "Lambda_profile": profile(real["trials"]), "median_width": widths(real["trials"])},
        "T2": {"mean_D_real": float(Dr.mean()), "p_one_sided_wilcoxon": p2, "passed": bool(t2)},
        "T3": {"mean_D_null": float(Dn.mean()), "diff_ci95": [float(lo), float(hi)], "passed": bool(t3),
               "null_Lambda_profile": profile(null["trials"])},
        "H6_supported": bool(t1 and t2 and t3),
    }


if __name__ == "__main__":
    if "--analyze" in sys.argv:
        real, null = (json.load(open(f"{OUT_DIR}/{a}.json")) for a in ("real", "null"))
        out = {k: analyze_h6(real, null, key=k) for k in ("lambda", "tau")}
        json.dump(out, open(f"{OUT_DIR}/analysis.json", "w"))
        print(json.dumps(out, indent=1))
        sys.exit(0)
    arm = sys.argv[sys.argv.index("--arm") + 1]
    if arm == "null":
        import transformers

        from hlens.harness import randomize_model
        from hlens.loading import MODEL_ID, load_hf

        tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
        hf = load_hf(dtype=torch.float32)
        model = randomize_model(hf, tok, mode="reinit", seed=0)
        run_arm(SimpleNamespace(model=model, hf=hf, tokenizer=tok), "null")
    else:
        if not os.path.exists(f"{OUT_DIR}/null.json") or json.load(open(f"{OUT_DIR}/null.json"))["prereg_sha256"] != prereg_sha256():
            sys.exit("refusing to run the real arm: the null arm must be run first under the SAME pre-registration hash")
        from hlens.loading import load_qwen

        run_arm(load_qwen(dtype=torch.float32), "real")


# --- knee-vs-Taylor-radius diagnosis (the notes make it mandatory; exploratory in the pre-registration) --------


def path_validity(qwen, setup: InterpolationSetup, alpha: float, layer: int, gap: int, *, d_alpha: float = 0.05, ends: dict | None = None, slice_factory=make_slice_fn) -> dict:
    """Validity score of the layer-l -> (l+gap) commitment map for an alpha-step of `d_alpha` along the
    interpolation path: the Taylor radius (the upper edge of the finite-difference plateau) over the step
    length. score < 1 means the local quadratic picture has already broken down at the size of an alpha-step,
    so a curvature spike there may be an estimator artifact."""
    from hlens.harness import validity_score

    model, pos, target = qwen.model, setup.position, layer + gap
    if ends is None:
        ends = {a: forward_at_alpha(qwen, setup, a, [layer, target]) for a in (0.0, 1.0)}
    c = (ends[1.0][target][0, pos] - ends[0.0][target][0, pos]).float()
    c = c / c.norm()
    eps = 0.5 * d_alpha
    lo = forward_at_alpha(qwen, setup, max(alpha - eps, 0.0), [layer])[layer].float()
    hi = forward_at_alpha(qwen, setup, min(alpha + eps, 1.0), [layer])[layer].float()
    mid = forward_at_alpha(qwen, setup, alpha, [layer])[layer].float()
    step = (hi[0, pos] - lo[0, pos]) / (min(alpha + eps, 1.0) - max(alpha - eps, 0.0)) * d_alpha
    point = ContextPoint(input_ids=setup.input_ids, layer=layer, position=pos, h=mid)
    f1, x0 = at_position(commitment_function(model, layer, target, c, pos, slice_factory), point)
    n = step.norm().item()
    vs = validity_score(f1, x0, step, eps_grid=[n * s for s in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)])
    return {"valid": vs.valid, "score": vs.validity_score, "step_norm": n, "x_norm": x0.norm().item()}


def knee_vs_radius(qwen, *, trial_idx=(0, 5, 10, 15, 20), layers=(4, 12, 16, 20, 24), offsets=(0.0, 0.2), gap: int = GAP, mid: int = 16, clean: bool = False) -> list[dict]:
    from hlens.data import load_ignition

    cfg = load_ignition("data/experiments/ignition.json")
    trials = choose_trials(qwen, cfg)
    out = []
    for i in trial_idx:
        template, A, B = trials[i]
        setup = make_setup(qwen, template, A, B, clean=clean)
        pos = setup.position
        alphas = np.linspace(0, 1, 41)
        H = {float(a): forward_at_alpha(qwen, setup, float(a), [mid]) for a in alphas}
        hb, ha = H[0.0][mid][0, pos], H[1.0][mid][0, pos]
        a_star = transition(alphas, [projection_share(ha, hb, H[float(a)][mid][0, pos]) for a in alphas]).threshold
        if a_star is None:
            continue
        for l in layers:
            ends = {a: forward_at_alpha(qwen, setup, a, [l, l + gap]) for a in (0.0, 1.0)}
            for o in offsets:
                a = float(np.clip(a_star + o, 0.02, 0.98))
                r = path_validity(qwen, setup, a, l, gap, ends=ends)
                out.append({"trial": i, "layer": l, "offset": o, **r})
    return out

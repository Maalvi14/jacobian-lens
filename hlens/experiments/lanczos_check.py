"""Lanczos convergence check for the H6 leading-eigenvalue estimates, per hlens/docs/preregistration_lanczos_check.md.
Qwen3.5-4B fp32. For a subset of the 24 H6 trials, at the H6 layers and offsets, the leading |eigenvalue| of the commitment map's Hessian is
re-estimated with a longer Lanczos run (and two start vectors); the 8-iteration values used by H6 are compared with the converged ones.

Run:  .venv/bin/python -m hlens.experiments.lanczos_check run
      .venv/bin/python -m hlens.experiments.lanczos_check analyze
"""

from __future__ import annotations

import hashlib
import json
import sys
import time

import numpy as np
import torch
from scipy import stats

from hlens.contexts import ContextPoint, at_position
from hlens.data import load_ignition
from hlens.experiments.ignition import GAP, LAYERS, OFFSETS, choose_trials, commitment_function
from hlens.hvp import hvp_fr
from hlens.interpolation import forward_at_alpha, make_setup

PREREG = "hlens/docs/preregistration_lanczos_check.md"
H6_REAL = "results/h6/real.json"
OUT = "results/lanczos_check_qwen3.5-4b.json"
TRIAL_IDX = (0, 3, 6, 9, 12, 15, 18, 21)
CHECKPOINTS = (8, 16, 24, 40)
SEEDS = (0, 1)


def prereg_sha256(path: str = PREREG) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def lanczos_profile(hvp, dim: int, *, iters: int, checkpoints=CHECKPOINTS, seed: int = 0, device="cpu", dtype=torch.float32) -> dict[int, float]:
    """Leading |eigenvalue| after k Lanczos steps for each k in `checkpoints`, from ONE run of `iters` steps (full re-orthogonalisation).
    Seed 0 with k = 8 is exactly `hlens.experiments.ignition.lanczos_top_abs_eig(..., iters=8)`."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(dim, generator=g).to(device, dtype)
    q = q / q.norm()
    Q, alphas, betas = [q], [], []
    out = {}
    for j in range(iters):
        w = hvp(Q[-1]).to(dtype)
        a = (w @ Q[-1]).item()
        alphas.append(a)
        w = w - a * Q[-1] - (betas[-1] * Q[-2] if betas else 0.0)
        for qq in Q:
            w = w - (w @ qq) * qq
        b = w.norm().item()
        k = j + 1
        if k in checkpoints or b < 1e-10 or k == iters:
            T = np.diag(alphas) + np.diag(betas[: k - 1], 1) + np.diag(betas[: k - 1], -1) if k > 1 else np.array([[alphas[0]]])
            out[k] = float(np.abs(np.linalg.eigvalsh(T)).max())
        if b < 1e-10 or k == iters:
            break
        betas.append(b)
        Q.append(w / b)
    return out


def compare(cells: list[dict]) -> dict:
    """cells: dicts with lam (checkpoint -> value, seed 0), lam_seed1, lam_h6 (the stored 8-iteration value), alpha-offset, layer, trial, xnorm."""
    ref = np.array([c["lam"]["40"] for c in cells])
    l8 = np.array([c["lam"]["8"] for c in cells])
    rel = np.abs(l8 - ref) / ref
    rel_seed = np.array([abs(c["lam_seed1"]["8"] - c["lam"]["8"]) / max(c["lam"]["40"], 1e-12) for c in cells])
    rel_16 = np.abs(np.array([c["lam"]["16"] for c in cells]) - ref) / ref
    key = lambda c: (c["trial"], c["layer"])
    D = {}
    for kind, getter in (("lam8", lambda c: c["lam"]["8"]), ("lam40", lambda c: c["lam"]["40"]), ("lam40_x_norm", lambda c: c["lam"]["40"] * c["xnorm"])):
        by = {}
        for c in cells:
            by.setdefault(key(c), {})[c["offset"]] = getter(c)
        D[kind] = {k: v[0.0] - 0.5 * (v[-0.2] + v[0.2]) for k, v in by.items() if all(o in v for o in (-0.2, 0.0, 0.2))}
    ks = sorted(D["lam8"])
    sign_agree = float(np.mean([np.sign(D["lam8"][k]) == np.sign(D["lam40"][k]) for k in ks]))
    layers = sorted({c["layer"] for c in cells})
    med = lambda f, l: float(np.median([f(c) for c in cells if c["layer"] == l and c["offset"] == 0.0]))
    p_of = lambda kind: float(stats.wilcoxon(list(D[kind].values()), alternative="greater").pvalue)
    return {
        "n_cells": len(cells), "n_cells_with_stored_h6": sum(c.get("lam_h6") is not None for c in cells),
        "rel_error_8_vs_40": {"median": float(np.median(rel)), "p90": float(np.percentile(rel, 90)), "max": float(rel.max())},
        "rel_error_16_vs_40": {"median": float(np.median(rel_16)), "p90": float(np.percentile(rel_16, 90))},
        "rel_diff_seed0_vs_seed1_at_8": {"median": float(np.median(rel_seed)), "p90": float(np.percentile(rel_seed, 90))},
        "C1_adequate": bool(np.median(rel) <= 0.05 and np.percentile(rel, 90) <= 0.20),
        "C2_sign_agreement_of_D": sign_agree, "C2_passed": bool(sign_agree >= 0.90),
        "stored_h6_matches_seed0_8": float(np.mean([abs(c["lam_h6"] - c["lam"]["8"]) <= 1e-3 * abs(c["lam"]["8"]) for c in cells if c.get("lam_h6") is not None])) if any(c.get("lam_h6") is not None for c in cells) else None,
        "profile_median_lam8_by_layer": {str(l): med(lambda c: c["lam"]["8"], l) for l in layers},
        "profile_median_lam40_by_layer": {str(l): med(lambda c: c["lam"]["40"], l) for l in layers},
        "profile_median_lam40_x_norm_by_layer": {str(l): med(lambda c: c["lam"]["40"] * c["xnorm"], l) for l in layers},
        "mean_D_and_wilcoxon_p": {k: {"mean_D": float(np.mean(list(v.values()))), "p": p_of(k)} for k, v in D.items()},
    }


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "run":
        from hlens.loading import load_qwen

        q = load_qwen(dtype=torch.float32)
        stored = {(t["template"], t["A"], t["B"]): t for t in json.load(open(H6_REAL))["trials"]}
        trials = choose_trials(q, load_ignition("data/experiments/ignition.json"))
        cells, t0 = [], time.time()
        for ti in TRIAL_IDX:
            tpl, A, B = trials[ti]
            s6 = stored.get((tpl, A, B))
            if s6 is None:
                continue
            setup = make_setup(q, tpl, A, B)
            pos = setup.position
            for l in LAYERS:
                target = l + GAP
                ends = {a: forward_at_alpha(q, setup, a, [l, target]) for a in (0.0, 1.0)}
                c = ends[1.0][target][0, pos] - ends[0.0][target][0, pos]
                c = c / c.norm()
                f_full = commitment_function(q.model, l, target, c, pos)
                for o in OFFSETS:
                    a = float(np.clip(s6["alpha_star"] + o, 0.02, 0.98))
                    mid = forward_at_alpha(q, setup, a, [l])[l]
                    f1, x0 = at_position(f_full, ContextPoint(input_ids=setup.input_ids, layer=l, position=pos, h=mid))
                    op = lambda v: hvp_fr(f1, x0, v)
                    prof = [lanczos_profile(op, x0.numel(), iters=max(CHECKPOINTS), seed=sd, device=x0.device, dtype=x0.dtype) for sd in SEEDS]
                    cells.append({"trial": ti, "layer": l, "offset": o, "xnorm": x0.norm().item(),
                                  "lam": {str(k): v for k, v in prof[0].items()}, "lam_seed1": {str(k): v for k, v in prof[1].items()},
                                  "lam_h6": s6["cells"].get(f"{l}|{o}", {}).get("lambda")})
            print(f"trial {ti}: {len(cells)} cells, {time.time() - t0:.0f}s", flush=True)
            json.dump({"prereg_sha256": prereg_sha256(), "cells": cells}, open(OUT, "w"))
    else:
        d = json.load(open(OUT))
        res = compare(d["cells"])
        res["prereg_sha256"] = {"recorded": d["prereg_sha256"], "current": prereg_sha256()}
        d["analysis"] = res
        json.dump(d, open(OUT, "w"))
        print(json.dumps(res, indent=1))

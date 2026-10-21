"""H9 precondition (mutual inhibition near the threshold), per hlens/docs/preregistration_h9.md.

Run:  uv run python -m hlens.experiments.competition
      uv run python -m hlens.experiments.competition --analyze
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
from scipy import stats

from hlens.contexts import ContextPoint, at_position, lens_direction_bank
from hlens.data import load_ignition
from hlens.experiments.ignition import choose_trials, projection_share, transition
from hlens.functional import make_slice_fn
from hlens.hvp import restricted_hessian
from jlens.hooks import ActivationRecorder  # noqa: F401
from hlens.interpolation import forward_at_alpha, make_setup
from hlens.jspace import unembed_directions

PREREG = "hlens/docs/preregistration_h9.md"
OUT = "results/h9_competition_qwen3.5-4b.json"
SRC, TGT = 12, 20
OFFSETS = [-0.2, 0.0, 0.2]


def prereg_sha256(path: str = PREREG) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def inhibition(f_A, f_B, x0, dirs: torch.Tensor) -> tuple[float, float, float]:
    """(c_A, c_B, I) for the two directions in `dirs` [2, d]: cross-terms of H_{f_A} and H_{f_B}, and
    I = -(c_A + c_B)/2."""
    cA = restricted_hessian(f_A, x0, dirs).H[0, 1].item()
    cB = restricted_hessian(f_B, x0, dirs).H[0, 1].item()
    return cA, cB, -(cA + cB) / 2


def random_orthogonal_pair(dirs: torch.Tensor, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(2, dirs.shape[1], generator=g).to(dirs.device, dirs.dtype)
    basis = torch.linalg.qr(dirs.T)[0]  # d x 2
    r = r - (r @ basis) @ basis.T
    q = torch.linalg.qr(r.T)[0].T
    return q[:2]


def run(q32, *, trials=None, src: int = SRC, tgt: int = TGT, mid: int = 16, validity: bool = False, clean: bool = False,
        slice_factory=make_slice_fn, unembed_model=None, alpha_stars=None, checkpoint: str | None = None, directions: str = "lens", log=print) -> list[dict]:
    """The pre-registered precondition measurement. `src`/`tgt`/`mid` default to the Qwen3.5-4B layers;
    other models pass depth-matched layers (hlens.loading.matched_layer). `validity` adds the
    Taylor-validity score of the src -> tgt map at the threshold (replication pre-registration).
    `slice_factory`/`unembed_model`/`alpha_stars` serve the mixed-precision check (preregistration_mixed_precision.md):
    a bf16 model with an fp32 slice, fp32 unembedding directions, and the reference thresholds so only precision differs.
    `checkpoint` is a .jsonl path: each finished trial is appended, and a rerun resumes from it (a 5-hour run was once lost
    when the process died before saving)."""
    SRC, TGT = src, tgt
    model, lens, tok = q32.model, q32.lens, q32.tokenizer
    if trials is None:
        trials = choose_trials(q32, load_ignition("data/experiments/ignition.json"))
    out, t0 = [], time.time()
    done = {}
    if checkpoint and os.path.exists(checkpoint):
        for line in open(checkpoint):
            r = json.loads(line)
            done[(r["template"], r["A"], r["B"])] = r
        log(f"  resuming: {len(done)} trials already in {checkpoint}")
    slice_fn = slice_factory(model, SRC, TGT)
    for ti, (template, A, B) in enumerate(trials):
        if (template, A, B) in done:
            out.append(done[(template, A, B)])
            continue
        setup = make_setup(q32, template, A, B, clean=clean)
        pos = setup.position
        alphas = np.linspace(0, 1, 41)
        H = {float(a): forward_at_alpha(q32, setup, float(a), [mid]) for a in alphas}
        hb, ha = H[0.0][mid][0, pos], H[1.0][mid][0, pos]
        a_star = transition(alphas, [projection_share(ha, hb, H[float(a)][mid][0, pos]) for a in alphas]).threshold
        own_star = a_star
        if alpha_stars is not None:
            a_star = alpha_stars[ti]
        if a_star is None:
            continue
        U = unembed_directions(unembed_model or model, [setup.id_a, setup.id_b], dim_batch=2)
        dev = next(iter(model.layers[0].parameters())).device
        if directions == "identity":  # logit-lens directions: the unembedding rows themselves, no J_l
            vA, vB = torch.nn.functional.normalize(U.to(dev), dim=1)
            wA, wB = vA, vB
        else:
            vA, vB = lens_direction_bank(U.to(dev), lens, SRC)
            wA, wB = lens_direction_bank(U.to(dev), lens, TGT)
        dirs = torch.stack([vA, vB])
        ctrl = random_orthogonal_pair(dirs, seed=ti)
        row = {"template": template, "A": A, "B": B, "alpha_star": a_star, "alpha_star_own": own_star, "cells": {}}
        if validity:
            from hlens.experiments.ignition import path_validity

            row["validity"] = path_validity(q32, setup, float(np.clip(a_star, 0.02, 0.98)), SRC, TGT - SRC, slice_factory=slice_factory)
        for o in OFFSETS:
            a = float(np.clip(a_star + o, 0.02, 0.98))
            h = forward_at_alpha(q32, setup, a, [SRC])[SRC].float()
            point = ContextPoint(input_ids=setup.input_ids, layer=SRC, position=pos, h=h)
            f_A, x0 = at_position(lambda x, w=wA: (slice_fn(x)[0, pos] * w).sum(), point)
            f_B, _ = at_position(lambda x, w=wB: (slice_fn(x)[0, pos] * w).sum(), point)
            cA, cB, I = inhibition(f_A, f_B, x0, dirs.to(x0.dtype))
            _, _, Ic = inhibition(f_A, f_B, x0, ctrl.to(x0.dtype))
            # first-order cross effect (Jacobian-level): w_B . J v_A  and  w_A . J v_B
            _, jA = torch.func.jvp(f_B, (x0,), (vA.to(x0.dtype),))
            _, jB = torch.func.jvp(f_A, (x0,), (vB.to(x0.dtype),))
            row["cells"][str(o)] = {"cA": cA, "cB": cB, "I": I, "I_control": Ic, "J_cross": (jA.item() + jB.item()) / 2}
        out.append(row)
        if checkpoint:
            os.makedirs(os.path.dirname(checkpoint) or ".", exist_ok=True)
            with open(checkpoint, "a") as fh:
                fh.write(json.dumps(row, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # return freed blocks: on the Thor CPU and GPU share one memory pool
        avail = next((int(l.split()[1]) / 2**20 for l in open("/proc/meminfo") if l.startswith("MemAvailable")), float("nan"))
        gpu = (torch.cuda.memory_allocated() / 2**30, torch.cuda.memory_reserved() / 2**30) if torch.cuda.is_available() else (0, 0)
        log(f"  trial {ti + 1}/{len(trials)}, {time.time() - t0:.0f}s, MemAvailable {avail:.0f} GiB, torch allocated {gpu[0]:.0f} reserved {gpu[1]:.0f} GiB")
    return out


def analyze_h9(rows: list[dict]) -> dict:
    I = lambda o: np.array([r["cells"][str(o)]["I"] for r in rows])
    Ic0 = np.array([r["cells"]["0.0"]["I_control"] for r in rows])
    flank = 0.5 * (I(-0.2) + I(0.2))
    w = lambda x: float(stats.wilcoxon(x, alternative="greater").pvalue) if np.any(x != 0) else 1.0
    p_a, p_b, p_c = w(I(0.0)), w(I(0.0) - flank), w(I(0.0) - Ic0)
    both_neg = float(np.mean([(r["cells"]["0.0"]["cA"] < 0) and (r["cells"]["0.0"]["cB"] < 0) for r in rows]))
    res = {"n_trials": len(rows),
           "P0a": {"median_I_at_threshold": float(np.median(I(0.0))), "p": p_a, "passed": bool(p_a < 0.01)},
           "P0b": {"median_I_threshold_minus_flank": float(np.median(I(0.0) - flank)), "p": p_b, "passed": bool(p_b < 0.01)},
           "P0c": {"median_I_control": float(np.median(Ic0)), "p": p_c, "passed": bool(p_c < 0.01)},
           "exploratory": {"frac_both_cross_terms_negative_at_threshold": both_neg,
                           "median_I_by_offset": {str(o): float(np.median(I(o))) for o in OFFSETS},
                           "median_first_order_cross": float(np.median([r["cells"]["0.0"]["J_cross"] for r in rows]))}}
    res["precondition_met"] = bool(res["P0a"]["passed"] and res["P0b"]["passed"] and res["P0c"]["passed"])
    return res


# --- H9 part 2: list-loading (Addendum 1 to the pre-registration) --------------------------------------------------

LOAD_OUT = "results/h9_list_loading_qwen3.5-4b.json"
KS = [4, 8, 12, 16, 20, 24, 28, 32]
WINDOW, N_LISTS, LIST_LEN = 4, 8, 32


def _word_lists(tok, family: str, n_lists: int, seed: int) -> list[list[str]]:
    cap = json.load(open("data/experiments/capacity.json"))
    pool = next(p["pool"] for p in cap["candidate_pools"] if p["name"] == family)
    ok = [w for w in pool if len(tok.encode(" " + w, add_special_tokens=False)) == 1]
    rng = np.random.default_rng(seed)
    return [list(rng.choice(ok, LIST_LEN, replace=False)) for _ in range(n_lists)]


def list_inhibition(f_of, x0, V: torch.Tensor) -> float:
    """I = -mean_i mean_{j != i} H_{f_i}[i, j] for the window directions V [m, d] and readouts f_of[i]."""
    m = len(V)
    c = []
    for i in range(m):
        H = restricted_hessian(f_of[i], x0, V).H.cpu()
        c.append(np.mean([H[i, j].item() for j in range(m) if j != i]))
    return -float(np.mean(c))


def run_list_loading(q32, *, log=print) -> list[dict]:
    model, lens, tok = q32.model, q32.lens, q32.tokenizer
    dev = next(iter(model.layers[0].parameters())).device
    slice_fn = make_slice_fn(model, SRC, TGT)
    out, t0 = [], time.time()
    for fam in ("countries", "cities"):
        lists = _word_lists(tok, fam, N_LISTS, seed=0)
        ids_all = sorted({tok.encode(" " + w, add_special_tokens=False)[0] for L in lists for w in L})
        U = unembed_directions(model, ids_all, dim_batch=256)
        u_of = {t: U[i] for i, t in enumerate(ids_all)}
        for li, words in enumerate(lists):
            tok_ids = [tok.encode(" " + w, add_special_tokens=False)[0] for w in words]
            for k in KS:
                ids = model.encode(",".join(" " + w for w in words[:k]) + ",", max_length=256)
                pos = ids.shape[1] - 1
                with torch.no_grad(), ActivationRecorder_(model, [SRC, TGT]) as rec:
                    model.forward(ids)
                    h12, h20 = rec[SRC].detach().clone(), rec[TGT][0, pos].detach()
                with torch.no_grad():
                    logits = model.unembed(lens.transport(h20.float(), TGT)).float()
                ranks = [(logits > logits[t]).sum().item() + 1 for t in tok_ids[:k]]
                occ = int(sum(r <= 25 for r in ranks))
                win = tok_ids[k - WINDOW:k]
                Uw = torch.stack([u_of[t] for t in win]).to(dev)
                V = lens_direction_bank(Uw, lens, SRC)
                Wr = lens_direction_bank(Uw, lens, TGT)
                point = ContextPoint(input_ids=ids, layer=SRC, position=pos, h=h12)
                f_of = []
                for i in range(WINDOW):
                    fi, x0 = at_position(lambda x, w=Wr[i]: (slice_fn(x)[0, pos] * w).sum(), point)
                    f_of.append(fi)
                I = list_inhibition(f_of, x0, V.to(x0.dtype))
                Ic = list_inhibition(f_of, x0, random_orthogonal_window(V, seed=1000 * li + k).to(x0.dtype))
                out.append({"family": fam, "list": li, "k": k, "I": I, "I_control": Ic, "occupancy": occ})
            log(f"  {fam} list {li + 1}/{N_LISTS}, {time.time() - t0:.0f}s")
    return out


def random_orthogonal_window(V: torch.Tensor, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(len(V), V.shape[1], generator=g).to(V.device, V.dtype)
    basis = torch.linalg.qr(V.T)[0]
    r = r - (r @ basis) @ basis.T
    return torch.linalg.qr(r.T)[0].T[: len(V)]


class ActivationRecorder_:
    """Tiny wrapper: record block outputs at the given layers (dict-like)."""

    def __init__(self, model, layers):
        from jlens.hooks import ActivationRecorder

        self._r = ActivationRecorder(model.layers, at=list(layers))

    def __enter__(self):
        self._r.__enter__()
        return self._r.activations

    def __exit__(self, *exc):
        self._r.__exit__(*exc)


def analyze_list_loading(rows: list[dict]) -> dict:
    k = np.array([r["k"] for r in rows])
    I, Ic = np.array([r["I"] for r in rows]), np.array([r["I_control"] for r in rows])
    low = k <= 24

    def sp(x, y):
        r = stats.spearmanr(x, y)
        return float(r.statistic), float(r.pvalue / 2 if r.statistic > 0 else 1 - r.pvalue / 2)

    q1_rho, q1_p = sp(k[low], I[low])
    c_rho, c_p = sp(k[low], Ic[low])
    q1 = bool(q1_p < 0.01 and c_p > 0.05)
    z, miss = [], []
    for fam in sorted({r["family"] for r in rows}):
        for kk in (24, 28, 32):
            sel = [r for r in rows if r["family"] == fam and r["k"] == kk]
            if len(sel) < 3:
                continue
            Iv = np.array([r["I"] for r in sel])
            m = np.array([1 - r["occupancy"] / r["k"] for r in sel])
            z += list((Iv - Iv.mean()) / (Iv.std() + 1e-12))
            miss += list((m - m.mean()) / (m.std() + 1e-12))
    q3_rho, q3_p = sp(np.array(z), np.array(miss))
    q3 = bool(q3_p < 0.01)
    by_k = {int(kk): {"I": float(np.median(I[k == kk])), "I_control": float(np.median(Ic[k == kk])),
                      "occupancy": float(np.median([r["occupancy"] for r in rows if r["k"] == kk]))} for kk in sorted(set(k))}
    return {"n": len(rows), "Q1": {"spearman_k_I_k_le_24": q1_rho, "p": q1_p, "control_spearman": c_rho, "control_p": c_p, "passed": q1},
            "Q3": {"spearman_I_vs_fraction_absent": q3_rho, "p": q3_p, "passed": q3}, "by_k": by_k,
            "list_loading_supported": bool(q1 and q3)}


if __name__ == "__main__":
    if "--analyze-lists" in sys.argv:
        d = json.load(open(LOAD_OUT))
        d["analysis"] = analyze_list_loading(d["rows"])
        d["analysis"]["prereg_sha256"] = {"recorded": d["prereg_sha256"], "current": prereg_sha256()}
        json.dump(d, open(LOAD_OUT, "w"))
        print(json.dumps(d["analysis"], indent=1))
        sys.exit(0)
    if "--lists" in sys.argv:
        from hlens.loading import load_qwen

        rows = run_list_loading(load_qwen(dtype=torch.float32))
        os.makedirs("results", exist_ok=True)
        json.dump({"prereg_sha256": prereg_sha256(), "rows": rows}, open(LOAD_OUT, "w"))
        sys.exit(0)
    if "--analyze" in sys.argv:
        d = json.load(open(OUT))
        d["analysis"] = analyze_h9(d["rows"])
        d["analysis"]["prereg_sha256"] = {"recorded": d["prereg_sha256"], "current": prereg_sha256()}
        json.dump(d, open(OUT, "w"))
        print(json.dumps(d["analysis"], indent=1))
        sys.exit(0)
    from hlens.loading import load_qwen

    rows = run(load_qwen(dtype=torch.float32))
    os.makedirs("results", exist_ok=True)
    json.dump({"prereg_sha256": prereg_sha256(), "rows": rows}, open(OUT, "w"))

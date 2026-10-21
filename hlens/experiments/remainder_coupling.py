"""H5 (remainder coupling), per hlens/docs/preregistration_h5.md. Read that file first: the
design, the single decision rule and the order of the arms are fixed there.

Run (null arm first, then the real arm, which checks the pre-registration hash):
    uv run python -m hlens.experiments.remainder_coupling --arm null
    uv run python -m hlens.experiments.remainder_coupling --arm real
    uv run python -m hlens.experiments.remainder_coupling --analyze
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

from hlens.contexts import ContextPoint, at_position, logit_fn
from hlens.hvp import hvp_fr
from hlens.jspace import decompose, unembed_directions
from jlens.fitting import valid_position_mask
from jlens.hooks import ActivationRecorder

PREREG = "hlens/docs/preregistration_h5.md"
OUT_DIR = "results/h5"
CONFIG = dict(layer=16, first_prompt=200, n_prompts=40, per_prompt=5, k=25, n_ctrl=25, max_length=128, seed=0, min_rho_frac=0.05)


def prereg_sha256(path: str = PREREG) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def build_dictionary(model, lens, layer: int, vocab_size: int, *, dim_batch: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    """Unit-norm J-lens dictionary [n_atoms, d_model] (CPU, fp32) of the whole vocabulary and
    the token id of each row; near-zero rows (unused vocabulary slots) are dropped."""
    with torch.enable_grad():
        u = unembed_directions(model, list(range(vocab_size)), dim_batch=dim_batch)
    v = u @ lens.jacobians[layer]
    norms = v.norm(dim=1)
    keep = norms > 1e-6
    return v[keep] / norms[keep, None], keep.nonzero().squeeze(1)


def measure_context(model, D: torch.Tensor, prompt: str, layer: int, position: int, *, k: int, n_ctrl: int, max_length: int, min_rho_frac: float) -> dict | None:
    """One context: decompose x, build w = H_f r_hat for f = the top-1 next-token logit, and score
    the selected and the control atoms by s(a) = |<w, a>| / ||w||."""
    last = model.n_layers - 1
    ids = model.encode(prompt, max_length=max_length)
    with torch.no_grad(), ActivationRecorder(model.layers, at=[layer, last]) as rec:
        model.forward(ids)
        h = rec.activations[layer].detach().clone()
        final = rec.activations[last][0, position]
        top1 = model.unembed(final).argmax().item()
    x = h[0, position]
    dec = decompose(x.float().cpu(), D, k=k)
    rho = dec.remainder
    if rho.norm() / x.norm().cpu() < min_rho_frac or len(dec.indices) == 0:
        return None
    rhat = rho / rho.norm()
    point = ContextPoint(input_ids=ids, layer=layer, position=position, h=h)
    f1, x0 = at_position(logit_fn(model, point, top1), point)
    w = hvp_fr(f1, x0, rhat.to(x0.device, x0.dtype)).float().cpu()
    grad_norm = torch.func.grad(f1)(x0).float().norm().item()
    corr = D @ x.float().cpu()
    corr[dec.indices] = -float("inf")
    ctrl = corr.topk(n_ctrl).indices
    s = lambda idx: ((D[idx] @ w).abs() / w.norm()).numpy()
    # addendum 1 (v2): project r_hat out of both the response and the atoms (see the pre-registration)
    wp = w - (w @ rhat) * rhat
    def s_perp(idx):
        a = D[idx]
        a = a - (a @ rhat)[:, None] * rhat[None, :]
        return ((a @ wp).abs() / (a.norm(dim=1) * wp.norm())).numpy()
    return {"s_sel": s(dec.indices).tolist(), "s_ctrl": s(ctrl).tolist(), "coeffs": dec.coeffs.tolist(),
            "s_sel_perp": s_perp(dec.indices).tolist(), "s_ctrl_perp": s_perp(ctrl).tolist(),
            "d_perp": float(s_perp(dec.indices).mean() - s_perp(ctrl).mean()),
            "frac_variance": dec.frac_variance, "w_norm": w.norm().item(), "grad_norm": grad_norm, "top1": top1,
            "d": float(s(dec.indices).mean() - s(ctrl).mean())}


def pick_contexts(model, prompts, *, per_prompt: int, seed: int, max_length: int) -> list[tuple[str, int]]:
    g = torch.Generator().manual_seed(seed)
    out = []
    for p in prompts:
        n = model.encode(p, max_length=max_length).shape[1]
        valid = valid_position_mask(n).nonzero().squeeze(1)
        for i in torch.randperm(len(valid), generator=g)[:per_prompt].tolist():
            out.append((p, int(valid[i])))
    return out


def run_arm(model, lens, vocab_size: int, prompts, name: str, *, config=CONFIG, log=print, arm: str | None = None) -> dict:
    D, _ = build_dictionary(model, lens, config["layer"], vocab_size)
    contexts = pick_contexts(model, prompts, per_prompt=config["per_prompt"], seed=config["seed"], max_length=config["max_length"])
    rows, excluded, t0 = [], 0, time.time()
    for i, (prompt, pos) in enumerate(contexts):
        r = measure_context(model, D, prompt, config["layer"], pos, k=config["k"], n_ctrl=config["n_ctrl"],
                            max_length=config["max_length"], min_rho_frac=config["min_rho_frac"])
        if r is None:
            excluded += 1
        else:
            rows.append(r)
        if (i + 1) % 20 == 0:
            log(f"  {name}: {i + 1}/{len(contexts)} contexts, {time.time() - t0:.0f}s")
    res = {"arm": arm or name, "prereg_sha256": prereg_sha256(), "config": config, "n_contexts": len(contexts), "n_excluded": excluded, "rows": rows}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/{name}.json", "w") as f:
        json.dump(res, f)
    return res


# --- analysis (pre-registered: Test A and Test B) -------------------------------------


def sign_flip_pvalue(d: np.ndarray, *, n: int = 10000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    obs = d.mean()
    signs = rng.integers(0, 2, (n, len(d))) * 2 - 1
    return float(((signs @ d / len(d) >= obs).sum() + 1) / (n + 1))


def analyze_h5(real: dict, null: dict, *, n_boot: int = 10000, seed: int = 0, key: str = "d") -> dict:
    dr, dn = np.array([r[key] for r in real["rows"]]), np.array([r[key] for r in null["rows"]])
    rng = np.random.default_rng(seed)
    diffs = [dr[rng.integers(0, len(dr), len(dr))].mean() - dn[rng.integers(0, len(dn), len(dn))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_a = sign_flip_pvalue(dr, n=n_boot, seed=seed)
    test_a = p_a < 0.01
    test_b = lo > 0
    sel = lambda a: np.array([np.mean(r["s_sel"]) for r in a["rows"]])
    ctl = lambda a: np.array([np.mean(r["s_ctrl"]) for r in a["rows"]])
    return {
        "prereg_sha256": {"real": real["prereg_sha256"], "null": null["prereg_sha256"], "current": prereg_sha256()},
        "n": {"real": len(dr), "null": len(dn), "excluded_real": real["n_excluded"], "excluded_null": null["n_excluded"]},
        "T_real": float(dr.mean()), "T_null": float(dn.mean()),
        "test_A": {"p_one_sided_sign_flip": p_a, "criterion": "p < 0.01", "passed": bool(test_a)},
        "test_B": {"T_real_minus_T_null": float(dr.mean() - dn.mean()), "ci95": [float(lo), float(hi)], "criterion": "CI > 0", "passed": bool(test_b)},
        "H5_supported": bool(test_a and test_b),
        "exploratory": {
            "mean_s_selected": {"real": float(sel(real).mean()), "null": float(sel(null).mean())},
            "mean_s_control": {"real": float(ctl(real).mean()), "null": float(ctl(null).mean())},
            "analytic_random_direction_level": float(0.8 / np.sqrt(2560)),
            "median_w_over_grad_norm": {"real": float(np.median([r["w_norm"] / r["grad_norm"] for r in real["rows"]])),
                                        "null": float(np.median([r["w_norm"] / r["grad_norm"] for r in null["rows"]]))},
            "mean_frac_variance": {"real": float(np.mean([r["frac_variance"] for r in real["rows"]])), "null": float(np.mean([r["frac_variance"] for r in null["rows"]]))},
        },
    }


if __name__ == "__main__":
    from jlens.examples import load_wikitext_prompts

    if "--analyze" in sys.argv:
        tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else ""
        real, null = (json.load(open(f"{OUT_DIR}/{a}{tag}.json")) for a in ("real", "null"))
        out = analyze_h5(real, null, key="d_perp" if tag == "_v2" else "d")
        json.dump(out, open(f"{OUT_DIR}/analysis{tag}.json", "w"))
        print(json.dumps(out, indent=1))
        sys.exit(0)
    arm = sys.argv[sys.argv.index("--arm") + 1]
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else ""
    prompts = load_wikitext_prompts(CONFIG["first_prompt"] + CONFIG["n_prompts"])[CONFIG["first_prompt"]:]
    if arm == "null":
        import transformers

        from hlens.harness import randomize_model
        from hlens.loading import MODEL_ID, load_hf
        from jlens import JacobianLens

        tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
        hf = load_hf(dtype=torch.float32)
        model = randomize_model(hf, tok, mode="reinit", seed=0)
        lens = JacobianLens.load("results/nulls/lens_randomized_reinit_s0_L16.pt")
        run_arm(model, lens, hf.config.get_text_config().vocab_size, prompts, "null" + tag)
    else:
        null_path = f"{OUT_DIR}/null{tag}.json"
        if not os.path.exists(null_path) or json.load(open(null_path))["prereg_sha256"] != prereg_sha256():
            sys.exit("refusing to run the real arm: the null arm must be run first under the SAME pre-registration hash")
        from hlens.loading import load_qwen

        q = load_qwen(dtype=torch.float32)
        run_arm(q.model, q.lens, q.hf.config.get_text_config().vocab_size, prompts, "real" + tag)

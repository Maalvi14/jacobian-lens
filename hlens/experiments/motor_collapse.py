"""H8 (motor-regime collapse), per hlens/docs/preregistration_h8.md.

Run:  uv run python -m hlens.experiments.motor_collapse            # Part A + Part B (real and randomized arms)
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

from hlens.experiments.trust_score import trust_terms
from hlens.functional import make_slice_fn
from jlens.fitting import valid_position_mask
from jlens.hooks import ActivationRecorder

PREREG = "hlens/docs/preregistration_h8.md"
OUT = "results/h8_motor_collapse_qwen3.5-4b.json"
LAYERS = [6, 10, 14, 18, 22, 26, 28, 29, 30]
WORKSPACE, MOTOR = [14, 18, 22], [28, 29, 30]
# Qwen3.5-4B values; another model passes depth-matched ones (hlens.loading.matched_layer) via `config_for`.
QWEN_CONFIG = {"layers": LAYERS, "workspace": WORKSPACE, "motor": MOTOR, "last": 31, "last_lens": 30, "mid_range": [10, 20]}


def config_for(n_layers: int, last_lens: int) -> dict:
    """Depth-matched H8 configuration (the Qwen3.5-4B layers mapped by depth fraction)."""
    from hlens.loading import matched_layer

    m = lambda xs: [matched_layer(n_layers, x) for x in xs]
    return {"layers": m(LAYERS), "workspace": m(WORKSPACE), "motor": m(MOTOR), "last": n_layers - 1, "last_lens": last_lens,
            "mid_range": m(QWEN_CONFIG["mid_range"])}


def prereg_sha256(path: str = PREREG) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def distance_to_scaled_identity(J: torch.Tensor) -> float:
    """min_c ||J - c I||_F / ||J||_F (c = tr(J)/d is the least-squares scalar)."""
    c = J.diagonal().sum() / J.shape[0]
    return ((J - c * torch.eye(J.shape[0], dtype=J.dtype)).norm() / J.norm()).item()


def tau_map(model, prompt: str, layer: int, position: int, direction: torch.Tensor, *, scale: float = 0.1, max_length: int = 128) -> float:
    ids = model.encode(prompt, max_length=max_length)
    with torch.no_grad(), ActivationRecorder(model.layers, at=[layer]) as rec:
        model.forward(ids)
        h = rec.activations[layer].detach().clone()
    delta = torch.zeros_like(h)
    delta[0, position] = direction.to(h.device, h.dtype) * scale * h[0, position].norm()
    slice_fn = make_slice_fn(model, layer, model.n_layers - 1)
    num, den = trust_terms(lambda x: slice_fn(x)[0, position], h, delta)
    return num / den


def run_arm(model, prompts, *, log=print, layers=None) -> dict:
    layers = layers or LAYERS
    g = torch.Generator().manual_seed(0)
    contexts = []
    for p in prompts:
        n = model.encode(p, max_length=128).shape[1]
        valid = valid_position_mask(n).nonzero().squeeze(1)
        for i in torch.randperm(len(valid), generator=g)[:2].tolist():
            contexts.append((p, int(valid[i])))
    dirs = torch.nn.functional.normalize(torch.randn(len(contexts), model.d_model, generator=g), dim=1)
    taus = {l: [] for l in layers}
    t0 = time.time()
    for i, (p, pos) in enumerate(contexts):
        for l in layers:
            taus[l].append(tau_map(model, p, l, pos, dirs[i]))
        if (i + 1) % 10 == 0:
            log(f"  {i + 1}/{len(contexts)} contexts, {time.time() - t0:.0f}s")
    return {str(l): v for l, v in taus.items()}


def ratio(taus: dict, idx=None, cfg: dict = QWEN_CONFIG) -> float:
    n = lambda band: np.median(np.concatenate([np.array(taus[str(l)])[idx if idx is not None else slice(None)] / (cfg["last"] - l) for l in band]))
    return float(n(cfg["motor"]) / n(cfg["workspace"]))


def analyze_h8(d: dict, *, n_boot: int = 2000, seed: int = 0) -> dict:
    cfg = d.get("config", QWEN_CONFIG)
    dI = {int(k): v for k, v in d["d_I"].items()}
    A = dI[cfg["last_lens"]] < 0.5 * float(np.median([dI[l] for l in range(cfg["mid_range"][0], cfg["mid_range"][1] + 1)]))
    rng = np.random.default_rng(seed)
    N = len(d["real"][str(cfg["layers"][0])])
    boots_r, boots_n = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, N, N)
        boots_r.append(ratio(d["real"], idx, cfg))
        boots_n.append(ratio(d["null"], idx, cfg))
    lo, hi = np.percentile(boots_r, [2.5, 97.5])
    r_real, r_null = ratio(d["real"], None, cfg), ratio(d["null"], None, cfg)
    B = bool(r_real < 0.5 and hi < 1.0)
    dl, dh = np.percentile(np.array(boots_r) - np.array(boots_n), [2.5, 97.5])
    C = bool(r_null >= 0.5 and dh < 0)
    from scipy import stats

    med = lambda taus: {int(l): float(np.median(v)) for l, v in taus.items()}
    return {
        "A": {"d_I": dI, "criterion": "d_I(30) < 0.5 * median d_I(10..20)", "passed": bool(A)},
        "B": {"ratio_real": r_real, "ci95": [float(lo), float(hi)], "passed": B},
        "C": {"ratio_null": r_null, "diff_ci95": [float(dl), float(dh)], "passed": C},
        "H8_supported": bool(A and B and C),
        "exploratory": {"median_tau_real": med(d["real"]), "median_tau_null": med(d["null"]),
                        "median_n_real": {int(l): float(np.median(v) / (cfg["last"] - int(l))) for l, v in d["real"].items()},
                        "median_n_null": {int(l): float(np.median(v) / (cfg["last"] - int(l))) for l, v in d["null"].items()},
                        "spearman_dI_vs_tau": float(stats.spearmanr([dI[int(l)] for l in d["real"]], [np.median(v) for v in d["real"].values()]).statistic)},
    }


if __name__ == "__main__":
    import transformers

    from hlens.harness import randomize_model
    from hlens.loading import MODEL_ID, load_hf, load_qwen
    from jlens.examples import load_wikitext_prompts

    if "--analyze" in sys.argv:
        d = json.load(open(OUT))
        d["analysis"] = analyze_h8(d)
        d["analysis"]["prereg_sha256"] = {"recorded": d["prereg_sha256"], "current": prereg_sha256()}
        json.dump(d, open(OUT, "w"))
        print(json.dumps(d["analysis"], indent=1))
        sys.exit(0)
    prompts = load_wikitext_prompts(330)[300:]
    q = load_qwen(dtype=torch.float32)
    d_I = {str(l): distance_to_scaled_identity(q.lens.jacobians[l]) for l in q.lens.source_layers}
    real = run_arm(q.model, prompts)
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    del q
    torch.cuda.empty_cache()
    null = run_arm(randomize_model(load_hf(dtype=torch.float32), tok, mode="reinit", seed=0), prompts)
    os.makedirs("results", exist_ok=True)
    json.dump({"prereg_sha256": prereg_sha256(), "d_I": d_I, "real": real, "null": null}, open(OUT, "w"))

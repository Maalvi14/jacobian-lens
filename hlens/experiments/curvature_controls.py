"""B' control populations per hlens/docs/preregistration_b_prime_controls.md: is the curvature enrichment along
J-lens directions explained by variance alignment (`cov`, `varmatch`), by the token directions the lens is built
from (`embed`, `unembed`), or by the lens-fit recipe (`refit`)?

Run:  uv run python -m hlens.experiments.curvature_controls real --model qwen3.5-4b
      uv run python -m hlens.experiments.curvature_controls refit --model qwen3.5-4b --layers 16 24
      uv run python -m hlens.experiments.curvature_controls analyze
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from types import SimpleNamespace

import numpy as np
import torch

from hlens.experiments.gain import _unit, collect_contexts, measure_layer, population_directions, summarize
from hlens.gain import active_token_pool
from hlens.jspace import unembed_directions
from hlens.loading import MODELS, load_model

PREREG = "hlens/docs/preregistration_b_prime_controls.md"
OUT_DIR = "results/curvature_controls"
B_PRIME_STORE = {"qwen3.5-4b": "results/h2_replication/B2_real_qwen3.5-4b.json", "gemma-3-12b": "results/h2_replication/B2_real_gemma-3-12b.json",
                 "llama3.1-8b": "results/h2_replication/B3_real_llama3.1-8b.json"}
LAYERS = {"qwen3.5-4b": [16, 18, 20, 22, 24, 26, 28, 30], "gemma-3-12b": [24, 28, 32, 36, 40, 44], "llama3.1-8b": [16, 18, 20, 22, 24, 26, 28, 30]}
EVAL_START, N_EVAL, COV_START, COV_PROMPTS, COV_PER_PROMPT, N_POOL = 200, 30, 400, 50, 24, 1024
DELTA = 0.05


def prereg_sha256() -> str:
    return hashlib.sha256(open(PREREG, "rb").read()).hexdigest()


def sample_cov_directions(X: torch.Tensor, n: int, gen: torch.Generator) -> torch.Tensor:
    """Unit directions v ∝ Σ_i g_i x_i (g ~ N(0, I)): samples of N(0, XᵀX / N), the activation second-moment matrix."""
    g = torch.randn(n, X.shape[0], generator=gen).to(X.device)
    return _unit(g @ X)


def variance_match(lens_dirs: torch.Tensor, pool: torch.Tensor, X: torch.Tensor) -> tuple[torch.Tensor, float]:
    """For each lens direction the unused pool direction with the closest log variance along it
    (variance = mean over activations X of (x · v)^2). Returns the matched directions and the median |log ratio|."""
    var = lambda V: ((X @ V.T) ** 2).mean(0).clamp_min(1e-30).log()
    lv, pv = var(lens_dirs).cpu(), var(pool).cpu()
    used, pick, err = set(), [], []
    for i in torch.argsort(lv, descending=True).tolist():  # greedy, largest variances first
        order = torch.argsort((pv - lv[i]).abs()).tolist()
        j = next(j for j in order if j not in used)
        used.add(j)
        pick.append((i, j))
        err.append(abs(float(pv[j] - lv[i])))
    pick.sort()
    return pool[[j for _, j in pick]], float(np.median(err))


def control_populations(q, layer: int, pool_tokens: list[int], u_rows: dict, X: torch.Tensor, *, n_dirs: int, seed: int, lens=None) -> tuple[dict, dict]:
    model = q.model
    base = population_directions(model, lens or q.lens, layer, pool_tokens, u_rows, n_dirs=n_dirs, seed=seed)
    gen = torch.Generator().manual_seed(seed + 1009 * layer)
    device = X.device
    cov = sample_cov_directions(X, n_dirs, gen)
    pool = torch.cat([sample_cov_directions(X, N_POOL, gen), _unit(torch.randn(N_POOL, X.shape[1], generator=gen)).to(device)])
    vm, err = variance_match(base["j_lens"].to(device), pool, X)
    g = torch.Generator().manual_seed(seed + 31 * layer)  # the same token draw as population_directions
    top = pool_tokens[:1000]
    toks = [top[i] for i in torch.randperm(len(top), generator=g)[:n_dirs].tolist()]
    emb = q.hf.get_input_embeddings().weight
    embed = _unit(emb[torch.tensor(toks, device=emb.device)].float())
    unembed = _unit(torch.stack([u_rows[t] for t in toks]))
    dirs = {"j_lens": base["j_lens"], "random": base["random"], "cov": cov, "varmatch": vm, "embed": embed.to(device), "unembed": unembed.to(device)}
    return dirs, {"varmatch_median_abs_log_ratio": err}


def _ratios(stats: dict, per_group: int) -> dict:
    out = {}
    for name in stats:
        if name == "j_lens":
            continue
        s = summarize({"random": stats[name], "j_lens": stats["j_lens"]}, per_group=per_group)
        out[name] = s["ratio_to_random"]["j_lens"]["kappa"]
    return out


def run_controls(q, prompts, cov_prompts, layers, *, lens=None, n_dirs: int = 64, per_prompt: int = 16, seed: int = 0, log=print) -> dict:
    model = q.model
    lens = lens or q.lens
    ms = [l + 1 for l in layers]
    ctx = collect_contexts(model, prompts, ms, per_prompt=per_prompt, seed=seed)
    covctx = collect_contexts(model, cov_prompts, ms, per_prompt=COV_PER_PROMPT, seed=seed + 7)
    pool = active_token_pool(model, lens, prompts[:10])
    union = sorted({t for l in layers for t in pool[l][:1000]})
    with torch.enable_grad():
        u_all = unembed_directions(model, union, dim_batch=256)
    u_rows = {t: u_all[i] for i, t in enumerate(union)}
    out = {}
    for l in layers:
        dirs, extra = control_populations(q, l, pool[l], u_rows, covctx[l + 1], n_dirs=n_dirs, seed=seed, lens=lens)
        stats = measure_layer(model, ctx[l + 1], dirs, l + 1)
        out[str(l)] = {"ratio_j_lens_over": _ratios(stats, per_prompt), **extra}
        r = out[str(l)]["ratio_j_lens_over"]
        log(f"L{l:02d} kappa J/ " + " ".join(f"{k} {v['median']:.2f}" for k, v in r.items()) + f" | match err {extra['varmatch_median_abs_log_ratio']:.3f}")
    return out


def _prompts(start: int, n: int) -> list[str]:
    from jlens.examples import load_wikitext_prompts

    return load_wikitext_prompts(start + n)[start:]


def real_arm(key: str) -> str:
    out = f"{OUT_DIR}/real_{key}.json"
    os.makedirs(OUT_DIR, exist_ok=True)
    q = load_model(key, dtype=torch.float32)
    res = run_controls(q, _prompts(EVAL_START, N_EVAL), _prompts(COV_START, COV_PROMPTS), LAYERS[key])
    json.dump({"arm": "controls", "model": key, "layers": LAYERS[key], "prereg_sha256": prereg_sha256(), "result": res}, open(out, "w"))
    return out


def refit_arm(key: str, layers: list[int]) -> str:
    """Real model, lens refitted with the null-arm recipe (30 prompts, dim_batch 32, max_seq_len 64)."""
    from hlens.experiments.gain import run_h2
    from hlens.harness import fit_null_lens

    out = f"{OUT_DIR}/refit_{key}_L{'-'.join(map(str, layers))}.json"
    os.makedirs(OUT_DIR, exist_ok=True)
    q = load_model(key, dtype=torch.float32)
    q.hf.to(torch.bfloat16)
    lens_path = f"{OUT_DIR}/lens_refit_{key}_L{'-'.join(map(str, layers))}.pt"
    lens = fit_null_lens(q.model, _prompts(300, 30), source_layers=layers, dim_batch=32, max_seq_len=64, path=lens_path)
    q.hf.to(torch.float32)
    torch.cuda.empty_cache()
    res = run_h2(SimpleNamespace(model=q.model, lens=lens, tokenizer=q.tokenizer), _prompts(EVAL_START, N_EVAL), layers=layers)
    json.dump({"arm": "refit", "model": key, "layers": layers, "prereg_sha256": prereg_sha256(), "result": res}, open(out, "w"))
    return out


def analyze(n_layers: dict[str, int]) -> dict:
    res = {}
    for key, nl in n_layers.items():
        p = f"{OUT_DIR}/real_{key}.json"
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        rng = {int(l): v for l, v in d["result"].items() if int(l) / (nl - 1) >= 0.5}
        row = {"layers": sorted(rng)}
        for pop in ("varmatch", "embed", "unembed", "cov"):
            lo = [v["ratio_j_lens_over"][pop]["ci95"][0] for v in rng.values()]
            med = [v["ratio_j_lens_over"][pop]["median"] for v in rng.values()]
            row[pop] = {"median_ratio_by_layer": dict(zip(sorted(rng), [rng[l]["ratio_j_lens_over"][pop]["median"] for l in sorted(rng)])),
                        "lower_bound_gt_1.05_fraction": float(np.mean([x > 1 + DELTA for x in lo])), "median_of_medians": float(np.median(med)),
                        "passes": bool(np.mean([x > 1 + DELTA for x in lo]) >= 0.75)}
        row["V"] = row["varmatch"]["passes"]
        row["K_embed"], row["K_unembed"] = row["embed"]["passes"], row["unembed"]["passes"]
        row["varmatch_match_err_median"] = float(np.median([v["varmatch_median_abs_log_ratio"] for v in rng.values()]))
        store = B_PRIME_STORE[key]
        if os.path.exists(store):  # procedural: random-baseline ratios of B' are reproduced via the shared j_lens/random seeds
            row["b_prime_ratio_by_layer"] = {l: json.load(open(store))["result"][str(l)]["ratio_to_random"]["j_lens"]["kappa"]["median"] for l in sorted(rng) if str(l) in json.load(open(store))["result"]}
        res[key] = row
    for p in sorted(glob.glob(f"{OUT_DIR}/refit_*.json")):
        d = json.load(open(p))
        key = d["model"]
        res.setdefault(f"refit_{key}", {})[p] = {l: {"ratio": v["ratio_to_random"]["j_lens"]["kappa"]["median"], "ci95": v["ratio_to_random"]["j_lens"]["kappa"]["ci95"]} for l, v in d["result"].items()}
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=["real", "refit", "analyze"])
    ap.add_argument("--model", default="qwen3.5-4b", choices=list(MODELS))
    ap.add_argument("--layers", type=int, nargs="+", default=[16, 24])
    a = ap.parse_args()
    if a.arm == "real":
        print(real_arm(a.model))
    elif a.arm == "refit":
        print(refit_arm(a.model, a.layers))
    else:
        print(json.dumps(analyze({"qwen3.5-4b": 32, "gemma-3-12b": 48, "llama3.1-8b": 32}), indent=1))

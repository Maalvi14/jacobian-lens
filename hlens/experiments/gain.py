"""H2 (research notes §3): is the MLP amplification of J-space directions partly nonlinear?

The context-free gain gate (hlens/gain.py, paper §4.3.1) did not reproduce on Qwen3.5-4B,
so this experiment asks the context-dependent version, which the paper's own footnote 5
says is what matters: at real activations r (the residual entering the MLP of layer m =
l + 1, from real text), for a unit direction v defined at layer l,

    g1(v; r)    = || J v ||             in-context first-order gain      (J = d MLP(norm(r)) / dr)
    kappa(v; r) = || 1/2 v^T H v ||     curvature along v                 (full contraction, halved)
    rho(v; r)   = kappa * ||r|| / g1    second- to first-order ratio for a perturbation as large as r

for the MLP sub-block including its own pre-norm, per layer, for three populations:
J-lens vectors of actively-used tokens, MLP-neuron output directions of layer l, and
isotropic random directions. The MLP is position-wise, so rows (context, direction) are
independent and are batched through one jvp / jvp-of-jvp.

H2's prediction: kappa (and rho) along J-aligned directions is elevated relative to both
controls in the workspace layers, and the gain of J-aligned directions varies more across
contexts (the operationalisation of d gain / d context used here: the coefficient of variation
of g1 over contexts, for a FIXED direction set). Falsifier: curvature at the isotropic baseline.
Diagnostic only (the paper's footnote concedes gain in a network is not an isolated block's).
"""

from __future__ import annotations

import contextlib
import json
import os
from dataclasses import dataclass

import numpy as np
import torch

from hlens.arch import mlp_branch, mlp_prenorm
from hlens.gain import active_token_pool
from hlens.jspace import unembed_directions
from jlens.fitting import valid_position_mask


@contextlib.contextmanager
def capture_mlp_inputs(model, layers):
    """Residual entering each layer's MLP sub-block (the input of its post_attention_layernorm),
    {layer: [1, seq, d_model]}, filled during the forward passes made inside the context."""
    store: dict[int, torch.Tensor] = {}
    handles = []
    for m in layers:
        norm = mlp_prenorm(model.layers[m])
        handles.append(norm.register_forward_pre_hook(lambda mod, args, m=m: store.__setitem__(m, args[0].detach())))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


def collect_contexts(model, prompts, layers_m, *, per_prompt: int, seed: int = 0, max_length: int = 128) -> dict[int, torch.Tensor]:
    """{m: [C, d_model]}: the MLP input at `per_prompt` random valid positions of each prompt
    (the same positions for every layer)."""
    g = torch.Generator().manual_seed(seed)
    rows: dict[int, list[torch.Tensor]] = {m: [] for m in layers_m}
    for prompt in prompts:
        ids = model.encode(prompt, max_length=max_length)
        valid = valid_position_mask(ids.shape[1]).nonzero().squeeze(1)
        pick = valid[torch.randperm(len(valid), generator=g)[:per_prompt]]
        with torch.no_grad(), capture_mlp_inputs(model, layers_m) as store:
            model.forward(ids)
            for m in layers_m:
                rows[m].append(store[m][0, pick.to(store[m].device)].float())
    return {m: torch.cat(v) for m, v in rows.items()}


def mlp_taylor(block, R: torch.Tensor, V: torch.Tensor, *, chunk: int = 2048) -> tuple[torch.Tensor, torch.Tensor]:
    """Per row i: g1_i = ||J_i v_i|| and kappa_i = ||1/2 v_i^T H_i v_i|| of the MLP sub-block
    (pre-norm included) at input r_i. R, V: [N, d_model] (rows are independent: the MLP is
    position-wise, so one batched jvp / jvp-of-jvp covers all pairs)."""
    f = mlp_branch(block)
    g1, kappa = [], []
    for i in range(0, len(R), chunk):
        r, v = R[i : i + chunk], V[i : i + chunk]

        def jvp_fn(x):
            return torch.func.jvp(f, (x,), (v,))[1]

        first = jvp_fn(r)
        second = torch.func.jvp(jvp_fn, (r,), (v,))[1]
        g1.append(first.norm(dim=-1))
        kappa.append(0.5 * second.norm(dim=-1))
    return torch.cat(g1), torch.cat(kappa)


@dataclass(frozen=True)
class PopulationStats:
    g1: np.ndarray  # [C, D] per (context, direction)
    kappa: np.ndarray
    r_norm: np.ndarray  # [C]


def _unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True)


def population_directions(model, lens, layer: int, pool: list[int], u_rows: dict[int, torch.Tensor], *, n_dirs: int, seed: int) -> dict[str, torch.Tensor]:
    """Three fixed direction sets [D, d_model] (unit) at source layer `layer`."""
    g = torch.Generator().manual_seed(seed + 31 * layer)
    device = next(iter(model.layers[0].parameters())).device
    top = pool[:1000]
    toks = [top[i] for i in torch.randperm(len(top), generator=g)[:n_dirs].tolist()]
    u = torch.stack([u_rows[t] for t in toks])
    jl = _unit(u @ lens.jacobians[layer].to(u.device))
    down = model.layers[layer].mlp.down_proj.weight  # [d_model, d_ff]
    cols = torch.randperm(down.shape[1], generator=g)[:n_dirs]
    neuron = _unit(down[:, cols.to(down.device)].T.float())
    rand = _unit(torch.randn(n_dirs, model.d_model, generator=g))
    return {"j_lens": jl.to(device), "neuron": neuron.to(device), "random": rand.to(device)}


def measure_layer(model, R: torch.Tensor, dirs: dict[str, torch.Tensor], m: int, *, chunk: int = 2048) -> dict[str, PopulationStats]:
    """R: [C, d] contexts entering the MLP of layer m."""
    block = model.layers[m]
    C = len(R)
    out = {}
    for name, D in dirs.items():
        k = len(D)
        Rr = R.repeat_interleave(k, dim=0)  # [C*k, d]
        Vv = D.repeat(C, 1)
        g1, kap = mlp_taylor(block, Rr, Vv, chunk=chunk)
        out[name] = PopulationStats(g1.view(C, k).cpu().numpy(), kap.view(C, k).cpu().numpy(), R.norm(dim=-1).cpu().numpy())
    return out


def summarize(stats: dict[str, PopulationStats], *, n_boot: int = 300, seed: int = 0, per_group: int | None = None) -> dict:
    """Medians, ratios to the random population, bootstrap CIs, and the across-context
    variability of the gain for fixed directions. Contexts from one prompt are correlated, so
    when `per_group` is given (contexts per prompt, stored consecutively) the bootstrap resamples
    whole prompts (cluster bootstrap); otherwise it resamples contexts."""
    rng = np.random.default_rng(seed)
    C = stats["random"].g1.shape[0]
    rho = {k: np.median(v.kappa * v.r_norm[:, None] / v.g1) for k, v in stats.items()}
    res = {"median_g1": {k: float(np.median(v.g1)) for k, v in stats.items()},
           "median_kappa": {k: float(np.median(v.kappa)) for k, v in stats.items()},
           "median_rho": {k: float(r) for k, r in rho.items()},
           "median_cv_g1_over_contexts": {k: float(np.median(v.g1.std(0) / v.g1.mean(0))) for k, v in stats.items()},
           "bootstrap": "prompt-cluster" if per_group else "context"}
    n_units = C // per_group if per_group else C
    boots = {k: {m: [] for m in ("g1", "kappa", "rho")} for k in stats if k != "random"}
    for _ in range(n_boot):
        pick = rng.integers(0, n_units, n_units)
        idx = (pick[:, None] * per_group + np.arange(per_group)).ravel() if per_group else pick
        base = {m: np.median(getattr(stats["random"], m)[idx]) for m in ("g1", "kappa")}
        base_rho = np.median(stats["random"].kappa[idx] * stats["random"].r_norm[idx, None] / stats["random"].g1[idx])
        for k in boots:
            v = stats[k]
            for m in ("g1", "kappa"):
                boots[k][m].append(np.median(getattr(v, m)[idx]) / base[m])
            boots[k]["rho"].append(np.median(v.kappa[idx] * v.r_norm[idx, None] / v.g1[idx]) / base_rho)
    res["ratio_to_random"] = {
        k: {m: {"median": float(np.median(boots[k][m])), "ci95": [float(np.percentile(boots[k][m], 2.5)), float(np.percentile(boots[k][m], 97.5))]} for m in ("g1", "kappa", "rho")}
        for k in boots
    }
    return res


def run_h2(qwen, prompts, *, layers=None, n_contexts_per_prompt: int = 16, n_dirs: int = 64, seed: int = 0, log=print) -> dict:
    model, lens = qwen.model, qwen.lens
    layers = layers if layers is not None else [l for l in lens.source_layers if l + 1 < model.n_layers]
    ms = [l + 1 for l in layers]
    ctx = collect_contexts(model, prompts, ms, per_prompt=n_contexts_per_prompt, seed=seed)
    pool = active_token_pool(model, lens, prompts[:10])
    union = sorted({t for l in layers for t in pool[l][:1000]})
    with torch.enable_grad():
        u_all = unembed_directions(model, union, dim_batch=256)
    u_rows = {t: u_all[i] for i, t in enumerate(union)}
    out = {}
    for l in layers:
        dirs = population_directions(model, lens, l, pool[l], u_rows, n_dirs=n_dirs, seed=seed)
        out[str(l)] = summarize(measure_layer(model, ctx[l + 1], dirs, l + 1), per_group=n_contexts_per_prompt)
        s = out[str(l)]
        log(f"L{l:02d} kappa J/rand {s['ratio_to_random']['j_lens']['kappa']['median']:.2f} neuron/rand {s['ratio_to_random']['neuron']['kappa']['median']:.2f} | "
            f"g1 J/rand {s['ratio_to_random']['j_lens']['g1']['median']:.2f} neuron/rand {s['ratio_to_random']['neuron']['g1']['median']:.2f}")
    return out


if __name__ == "__main__":
    import sys

    from hlens.loading import load_qwen
    from jlens.examples import load_wikitext_prompts

    os.makedirs("results", exist_ok=True)
    if "--null" in sys.argv:  # weight-randomized control at layer 16 (the only layer with a null lens)
        import transformers

        from hlens.harness import randomize_model
        from hlens.loading import MODEL_ID, load_hf
        from jlens import JacobianLens

        tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
        nmodel = randomize_model(load_hf(dtype=torch.float32), tok, mode="reinit", seed=0)

        class _Null:
            model = nmodel
            lens = JacobianLens.load("results/nulls/lens_randomized_reinit_s0_L16.pt")

        res = run_h2(_Null, load_wikitext_prompts(30), layers=[16])
        with open("results/gain_h2_null_qwen3.5-4b.json", "w") as f:
            json.dump(res, f)
    else:
        q = load_qwen(dtype=torch.float32)
        res = run_h2(q, load_wikitext_prompts(30))
        with open("results/gain_h2_qwen3.5-4b.json", "w") as f:
            json.dump(res, f)

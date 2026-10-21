"""MLP-gain reproduction gate (paper §4.3.1, Figure 32 left): how strongly does the
next MLP block amplify a unit direction defined at layer l?

gain(v) = || MLP_{l+1}(norm(v)) ||, normalised by the median of the same quantity
over isotropic random unit directions. Populations compared per source layer l:
J-lens vectors of random tokens, output-weight directions of layer-l MLP neurons,
and isotropic random directions (gain 1 by construction).

"The MLP block" includes the block's own pre-norm (post_attention_layernorm in
Llama/Qwen layouts); RMSNorm is scale-invariant, so no residual scale is needed.
The paper calls this a first-pass approximation (footnote 5): in a real forward pass
the MLP also sees everything else in the residual stream.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from hlens.arch import mlp_branch

from hlens.jspace import unembed_directions
from jlens import JacobianLens
from jlens.protocol import LensModel


def hf_mlp_block(
    model: LensModel, layer: int, *, variant: str = "prenorm"
) -> Callable[[torch.Tensor], torch.Tensor]:
    """unit residual-space directions [n, d_model] -> MLP output of decoder layer
    `layer` (Llama/Qwen layout). The paper does not say how the input is scaled, so
    three readings are provided:
      "prenorm": the block's own pre-norm, then the MLP (what the block does in a
                 forward pass; RMSNorm erases the input scale);
      "raw":     the MLP applied to the unit vector directly (tiny input, so the
                 gated MLP is in its near-quadratic regime);
      "rms":     the MLP applied to the vector scaled to unit RMS (sqrt(d_model)),
                 i.e. a normalised input without the norm's learned weights."""
    block = model.layers[layer]
    if variant == "prenorm":
        return mlp_branch(block)
    if variant == "raw":
        return lambda x: block.mlp(x)
    if variant == "rms":
        scale = model.d_model**0.5
        return lambda x: block.mlp(x * scale)
    raise ValueError(f"unknown variant {variant!r}")


@dataclass(frozen=True)
class GainCurves:
    layers: list[int]  # source layers l; the MLP measured is at l + 1
    j_lens: list[float]  # median gain, normalised so random directions = 1
    neuron: list[float]
    random_abs: list[float]  # the raw median output norm for random directions (the normaliser)


def _unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True)


def _median_gain(mlp, dirs: torch.Tensor, chunk: int = 1000) -> float:
    outs = [mlp(dirs[i : i + chunk]).float().norm(dim=-1) for i in range(0, len(dirs), chunk)]
    return torch.cat(outs).median().item()


def mlp_gain_curves(
    model: LensModel,
    lens: JacobianLens,
    *,
    n_directions: int = 2000,
    layers: list[int] | None = None,
    seed: int = 0,
    vocab_size: int | None = None,
    variant: str = "prenorm",
    token_pool: Callable[[int], list[int]] | None = None,
) -> GainCurves:
    """`token_pool(layer)` optionally restricts which vocabulary tokens' J-lens
    vectors form the J-lens population at that layer (default: uniform over the
    whole vocabulary, which in a 248k vocabulary is mostly rare tokens)."""
    device = next(iter(model.layers[0].parameters())).device
    dtype = next(iter(model.layers[0].parameters())).dtype
    d = model.d_model
    g = torch.Generator().manual_seed(seed)
    layers = layers if layers is not None else [l for l in lens.source_layers if l + 1 < model.n_layers]

    vocab = vocab_size or model.unembed(torch.ones(d)).shape[-1]
    if token_pool is None:
        tokens = torch.randperm(vocab, generator=g)[:n_directions].tolist()
        u_all = unembed_directions(model, tokens, dim_batch=256).to(device)
    u_cache: dict[int, torch.Tensor] = {}

    def u_for(layer: int) -> torch.Tensor:
        if token_pool is None:
            return u_all
        toks = token_pool(layer)[:n_directions]
        for t in toks:
            if t not in u_cache:
                u_cache[t] = None
        missing = [t for t in toks if u_cache[t] is None]
        if missing:
            with torch.enable_grad():  # we are inside no_grad; the unembed gradient needs autograd
                rows = unembed_directions(model, missing, dim_batch=256)
            for t, row in zip(missing, rows):
                u_cache[t] = row
        return torch.stack([u_cache[t] for t in toks]).to(device)

    out = {"j": [], "neuron": [], "random": []}
    with torch.no_grad():
        for l in layers:
            mlp = hf_mlp_block(model, l + 1, variant=variant)
            rand = _unit(torch.randn(n_directions, d, generator=g)).to(device, dtype)
            base = _median_gain(mlp, rand)
            j = _unit(u_for(l) @ lens.jacobians[l].to(device)).to(dtype)
            down = model.layers[l].mlp.down_proj.weight  # [d_model, d_ff]: columns are output directions
            cols = torch.randperm(down.shape[1], generator=g)[:n_directions]
            neurons = _unit(down[:, cols.to(down.device)].T.float()).to(device, dtype)
            out["j"].append(_median_gain(mlp, j) / base)
            out["neuron"].append(_median_gain(mlp, neurons) / base)
            out["random"].append(base)
    return GainCurves(layers=layers, j_lens=out["j"], neuron=out["neuron"], random_abs=out["random"])


def active_token_pool(
    model: LensModel, lens: JacobianLens, prompts, *, top: int = 10, max_length: int = 128
) -> dict[int, list[int]]:
    """layer -> tokens that appear in the lens top-`top` at some valid position of
    `prompts`, most frequent first: the J-space content that is actually active
    there (as opposed to a uniform draw over a mostly-rare vocabulary)."""
    from jlens.fitting import valid_position_mask
    from jlens.hooks import ActivationRecorder

    layers = lens.source_layers
    counts: dict[int, dict[int, int]] = {l: {} for l in layers}
    for prompt in prompts:
        ids = model.encode(prompt, max_length=max_length)
        with torch.no_grad(), ActivationRecorder(model.layers, at=layers) as rec:
            model.forward(ids)
            idx = valid_position_mask(ids.shape[1]).nonzero().squeeze(1)
            for l in layers:
                logits = model.unembed(lens.transport(rec.activations[l][0, idx].float(), l))
                for t in logits.topk(top, dim=-1).indices.flatten().tolist():
                    counts[l][t] = counts[l].get(t, 0) + 1
    return {l: [t for t, _ in sorted(c.items(), key=lambda kv: -kv[1])] for l, c in counts.items()}

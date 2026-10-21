"""Layer-landmark gate (paper §4.1, Figure 27-28): where does the J-space start and
stop acting as a workspace? Everything is reported as a depth fraction
(layer / (n_layers - 1)) so Claude-scale layer numbers are never carried over
directly.

Per source layer we compute the four signatures of Figure 28 on a set of prompts:
  (a) top-k accuracy of the lens at predicting the model's actual next token;
  (b) excess kurtosis of the lens readout logits;
  (c) autocorrelation of the lens's top-1 token across nearby positions, as a log
      ratio against a position-shuffled null;
  (d) effective dimensionality of the J-lens vectors (fraction of residual
      dimensions needed for 90% of their variance);
plus the CKA similarity matrix between layers' J-lens-vector geometry (Figure 27).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from hlens.jspace import unembed_directions
from jlens import JacobianLens
from jlens.fitting import valid_position_mask
from jlens.hooks import ActivationRecorder
from jlens.protocol import LensModel


@dataclass(frozen=True)
class LandmarkCurves:
    layers: list[int]
    n_layers: int  # of the model
    topk_acc: dict[int, list[float]]  # k -> per-layer accuracy
    kurtosis_mean: list[float]
    kurtosis_median: list[float]
    autocorr: list[float]  # mean log(real/null) repeat rate over the offsets
    eff_dim: list[float]  # fraction of d_model for 90% of lens-vector variance
    cka: torch.Tensor  # [L, L]

    def depth(self, layer: int) -> float:
        return layer / (self.n_layers - 1)


def _excess_kurtosis(x: torch.Tensor) -> torch.Tensor:
    """Per-row excess kurtosis of [n, vocab] logits."""
    x = x.float()
    c = x - x.mean(dim=-1, keepdim=True)
    m2 = (c**2).mean(dim=-1)
    m4 = (c**4).mean(dim=-1)
    return m4 / m2.clamp_min(1e-12) ** 2 - 3.0


def _cka_matrix(banks: list[torch.Tensor]) -> torch.Tensor:
    """Linear CKA between the pairwise-similarity (Gram) matrices of each layer's
    row-normalised lens-vector bank [n_tokens, d_model]."""
    n = banks[0].shape[0]
    centering = torch.eye(n, device=banks[0].device) - 1.0 / n
    flat = []
    for v in banks:
        v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)
        k = v @ v.T
        k = centering @ k @ centering
        flat.append((k / k.norm()).flatten())
    flat = torch.stack(flat)
    return (flat @ flat.T).cpu()


def compute_landmark_curves(
    model: LensModel,
    lens: JacobianLens,
    prompts: Sequence[str],
    *,
    max_length: int = 128,
    token_sample: int = 4096,
    topks: Sequence[int] = (1, 5, 10),
    offsets: Sequence[int] = (1, 2, 4),
    seed: int = 0,
    vocab_size: int | None = None,
) -> LandmarkCurves:
    layers = list(lens.source_layers)
    last = model.n_layers - 1
    device = next(iter(model.layers[0].parameters())).device
    g = torch.Generator().manual_seed(seed)

    hit = {k: [0] * len(layers) for k in topks}
    n_pos = 0
    kurt: list[list[float]] = [[] for _ in layers]
    same_real = [[0] * len(offsets) for _ in layers]
    same_null = [[0] * len(offsets) for _ in layers]
    n_pairs = [[0] * len(offsets) for _ in layers]

    for prompt in prompts:
        ids = model.encode(prompt, max_length=max_length)
        with torch.no_grad(), ActivationRecorder(model.layers, at=sorted({*layers, last})) as rec:
            model.forward(ids)
            acts = {l: rec.activations[l][0].float() for l in rec.activations}
        valid = valid_position_mask(ids.shape[1]).to(device)
        idx = valid.nonzero().squeeze(1)
        with torch.no_grad():
            final_top1 = model.unembed(acts[last][idx]).argmax(dim=-1)
            for li, layer in enumerate(layers):
                logits = model.unembed(lens.transport(acts[layer][idx], layer)).float()
                top = logits.topk(max(topks), dim=-1).indices
                for k in topks:
                    hit[k][li] += (top[:, :k] == final_top1[:, None]).any(dim=-1).sum().item()
                kurt[li].extend(_excess_kurtosis(logits).tolist())
                top1 = top[:, 0]
                for oi, off in enumerate(offsets):
                    if len(top1) <= off:
                        continue
                    real = (top1[:-off] == top1[off:]).sum().item()
                    perm = torch.randperm(len(top1), generator=g).to(device)
                    null = (top1[: len(top1) - off] == top1[perm][: len(top1) - off]).sum().item()
                    same_real[li][oi] += real
                    same_null[li][oi] += null
                    n_pairs[li][oi] += len(top1) - off
        n_pos += len(idx)

    topk_acc = {k: [h / n_pos for h in hit[k]] for k in topks}
    kurtosis_mean = [sum(x) / len(x) for x in kurt]
    kurtosis_median = [float(torch.tensor(x).median()) for x in kurt]
    autocorr = []
    for li in range(len(layers)):
        vals = []
        for oi in range(len(offsets)):
            n = max(n_pairs[li][oi], 1)
            vals.append(math.log((same_real[li][oi] + 1) / (same_null[li][oi] + 1)) if n else 0.0)
        autocorr.append(sum(vals) / len(vals))

    # J-lens-vector geometry on a fixed random token sample
    vocab = vocab_size or model.unembed(torch.ones(model.d_model)).shape[-1]
    sample = torch.randperm(vocab, generator=g)[:token_sample].tolist()
    u = unembed_directions(model, sample, dim_batch=256).to(device)
    banks, eff_dim = [], []
    for layer in layers:
        v = u @ lens.jacobians[layer].to(device)
        banks.append(v)
        s = torch.linalg.svdvals(v - v.mean(dim=0, keepdim=True)) ** 2
        cum = s.cumsum(0) / s.sum()
        eff_dim.append(((cum < 0.9).sum().item() + 1) / model.d_model)
    cka = _cka_matrix(banks)

    return LandmarkCurves(
        layers=layers, n_layers=model.n_layers, topk_acc=topk_acc,
        kurtosis_mean=kurtosis_mean, kurtosis_median=kurtosis_median,
        autocorr=autocorr, eff_dim=eff_dim, cka=cka,
    )


@dataclass(frozen=True)
class WorkspaceRange:
    onset_by_metric: dict[str, int]
    offset_by_metric: dict[str, int]
    onset: int  # median over metrics
    offset: int
    onset_depth: float
    offset_depth: float


def _first_at_or_above(curve: Sequence[float], layers: Sequence[int], start: int, level: float) -> int | None:
    for i in range(start, len(curve)):
        if curve[i] >= level:
            return layers[i]
    return None


def workspace_range(c: LandmarkCurves, *, rise: float = 0.25) -> WorkspaceRange:
    """Onset, per metric (kurtosis, autocorrelation, effective dimensionality): from
    the metric's minimum in the first 60% of depth, the first layer where it has
    made `rise` of its climb to its peak before the output-aligned tail. (Anchoring
    at the minimum matters: on Qwen the kurtosis starts high from token-local
    content and *falls* before it rises.) Offset = where the lens turns into a
    "motor" readout: the first layer where top-1 accuracy reaches half of its final
    value, and the first layer after the autocorrelation peak where it has fallen
    to half of that peak. Rules are simple and inspectable on purpose; the
    per-metric estimates are returned so disagreement between them stays visible."""
    L = len(c.layers)
    body = max(int(0.85 * L), 4)  # peaks are searched before the output-aligned tail
    onset, offset = {}, {}
    for name, curve in (("kurtosis", c.kurtosis_mean), ("autocorr", c.autocorr), ("eff_dim", c.eff_dim)):
        base_i = min(range(int(0.6 * L)), key=lambda i: curve[i])
        base, peak = curve[base_i], max(curve[base_i:body])
        hit = _first_at_or_above(curve, c.layers, base_i, base + rise * (peak - base))
        if hit is not None:
            onset[name] = hit
    t1 = c.topk_acc[1]
    jump = _first_at_or_above(t1, c.layers, 0, 0.5 * t1[-1])
    if jump is not None:
        offset["top1_accuracy"] = jump
    ac_peak_i = max(range(body), key=lambda i: c.autocorr[i])
    for i in range(ac_peak_i, L):
        if c.autocorr[i] <= 0.5 * c.autocorr[ac_peak_i]:
            offset["autocorr_fall"] = c.layers[i]
            break
    med = lambda xs: sorted(xs)[(len(xs) - 1) // 2]  # lower median: ties do not bias late
    on, off = med(list(onset.values())), med(list(offset.values()))
    return WorkspaceRange(
        onset_by_metric=onset, offset_by_metric=offset, onset=on, offset=off,
        onset_depth=c.depth(on), offset_depth=c.depth(off),
    )


def save_landmarks(c: LandmarkCurves, wr: WorkspaceRange, path: str) -> None:
    """Write the curves and the detected range as JSON (small: ~31 layers)."""
    import json
    import os
    from dataclasses import asdict

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out = {
        "layers": c.layers, "n_layers": c.n_layers,
        "topk_acc": {str(k): v for k, v in c.topk_acc.items()},
        "kurtosis_mean": c.kurtosis_mean, "kurtosis_median": c.kurtosis_median,
        "autocorr": c.autocorr, "eff_dim": c.eff_dim, "cka": c.cka.tolist(),
        "range": asdict(wr),
    }
    with open(path, "w") as f:
        json.dump(out, f)

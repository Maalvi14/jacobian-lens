"""H7 (early-layer content), per hlens/docs/preregistration_h7.md.

Run:  uv run python -m hlens.experiments.early_curvature
      uv run python -m hlens.experiments.early_curvature --analyze
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

from hlens.contexts import ContextPoint, at_position, lens_direction_bank, logit_fn
from hlens.hvp import hvp_fr
from hlens.jspace import unembed_directions
from hlens.swaps import first_token_ids
from hlens.synonyms import single_token_ids
from jlens.hooks import ActivationRecorder

PREREG = "hlens/docs/preregistration_h7.md"
OUT = "results/h7_early_curvature_qwen3.5-4b.json"
LAYERS = [4, 8, 12, 16]
EARLY = [4, 8]
N_DISTRACT = 15


def prereg_sha256(path: str = PREREG) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def rank_fraction(true_score: float, other: np.ndarray) -> float:
    return float(((other < true_score).sum() + 0.5 * (other == true_score).sum()) / len(other))


def candidate_scores(f1, x0, gvec, V: torch.Tensor) -> dict[str, np.ndarray]:
    """s0 (presence), s1 (sensitivity), s2 (self-curvature) for unit directions V [k, d]."""
    s0 = (V @ x0).cpu().numpy()
    s1 = (V @ gvec).abs().cpu().numpy()
    s2 = np.array([abs(float(v @ hvp_fr(f1, x0, v))) for v in V])
    return {"s0": s0, "s1": s1, "s2": s2}


def run(q32, *, log=print) -> list[dict]:
    model, lens, tok = q32.model, q32.lens, q32.tokenizer
    items = json.load(open("data/evaluations/lens-eval-multihop.json"))["items"]
    resolved = []
    for it in items:
        forms = single_token_ids(tok, [it["intermediates"][0]])
        if forms:
            resolved.append((it, forms))
    pool_forms = {}
    for it, forms in resolved:
        pool_forms.setdefault(it["intermediates"][0], forms)
    all_ids = sorted({t for f in pool_forms.values() for t in f.values()})
    U = unembed_directions(model, all_ids, dim_batch=256)
    u_of = {t: U[i] for i, t in enumerate(all_ids)}
    rng = np.random.default_rng(0)
    out, t0 = [], time.time()
    last = model.n_layers - 1
    for i, (it, forms) in enumerate(resolved):
        ids = model.encode(it["prompt"], max_length=128)
        pos = ids.shape[1] - 1
        with torch.no_grad(), ActivationRecorder(model.layers, at=[*LAYERS, last]) as rec:
            model.forward(ids)
            h = {l: rec.activations[l].detach().clone() for l in LAYERS}
            final = model.unembed(rec.activations[last][0, pos]).float()
        tgt_ids = first_token_ids(tok, it["target"])
        top1 = final.argmax().item()
        if top1 not in tgt_ids:
            continue
        row = {"name": it["name"], "layers": {}}
        for l in LAYERS:
            x = h[l][0, pos]
            with torch.no_grad():
                lens_logits = model.unembed(lens.transport(x.float(), l)).float()
            true_tok = max(forms.values(), key=lambda t: lens_logits[t].item())
            others = [w for w in pool_forms if w != it["intermediates"][0]]
            picks = rng.choice(len(others), size=min(N_DISTRACT, len(others)), replace=False)
            d_toks = [max(pool_forms[others[j]].values(), key=lambda t: lens_logits[t].item()) for j in picks]
            cand = [true_tok, *d_toks]
            V = lens_direction_bank(torch.stack([u_of[t] for t in cand]).to(x.device), lens, l)
            point = ContextPoint(input_ids=ids, layer=l, position=pos, h=h[l])
            f1, x0 = at_position(logit_fn(model, point, top1), point)
            gvec = torch.func.grad(f1)(x0)
            sc = candidate_scores(f1, x0, gvec, V.to(x0.dtype))
            row["layers"][str(l)] = {k: v.tolist() for k, v in sc.items()}
            row["layers"][str(l)]["V_norm_cos_mean"] = float(V.mean(0).norm())
        out.append(row)
        if len(out) % 10 == 0:
            log(f"  {len(out)} included items ({i + 1}/{len(resolved)} scanned), {time.time() - t0:.0f}s")
    return out


def analyze_h7(items: list[dict], *, n_boot: int = 10000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    T, P = {}, {}
    for l in LAYERS:
        for s in ("s0", "s1", "s2"):
            v = np.array([rank_fraction(abs(r["layers"][str(l)][s][0]) if s != "s0" else r["layers"][str(l)][s][0], np.array(r["layers"][str(l)][s][1:])) for r in items])
            P[(l, s)] = v
            T[(l, s)] = float(v.mean())
    res = {"n_items": len(items), "T": {f"L{l}_{s}": T[(l, s)] for l in LAYERS for s in ("s0", "s1", "s2")}, "per_early_layer": {}}
    supported = False
    for l in EARLY:
        d = P[(l, "s2")] - 0.5
        p = float(stats.wilcoxon(d, alternative="greater").pvalue) if np.any(d != 0) else 1.0
        diffs = [(P[(l, "s2")][idx] - P[(l, "s1")][idx]).mean() for idx in (rng.integers(0, len(items), len(items)) for _ in range(n_boot))]
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        c1, c2, c3 = T[(l, "s0")] <= 0.55, (T[(l, "s2")] >= 0.60 and p < 0.01), lo > 0
        res["per_early_layer"][f"L{l}"] = {"first_order_lens_fails": bool(c1), "second_order_content": bool(c2), "p_wilcoxon": p,
                                           "beats_sensitivity": bool(c3), "T_s2_minus_T_s1_ci95": [float(lo), float(hi)]}
        supported = supported or bool(c1 and c2 and c3)
    res["H7_supported"] = bool(supported)
    return res


if __name__ == "__main__":
    if "--analyze" in sys.argv:
        d = json.load(open(OUT))
        d["analysis"] = analyze_h7(d["items"])
        d["analysis"]["prereg_sha256"] = {"recorded": d["prereg_sha256"], "current": prereg_sha256()}
        json.dump(d, open(OUT, "w"))
        print(json.dumps(d["analysis"], indent=1))
        sys.exit(0)
    from hlens.loading import load_qwen

    items = run(load_qwen(dtype=torch.float32))
    os.makedirs("results", exist_ok=True)
    json.dump({"prereg_sha256": prereg_sha256(), "items": items}, open(OUT, "w"))

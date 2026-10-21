"""Gates G1 (landmarks) and G2 (swap) on a second model, per hlens/docs/preregistration_replication.md.
bf16 (the lens was fitted in bf16). Layers are depth-matched to Qwen3.5-4B.

Run:  uv run python -m hlens.experiments.gates --model gemma-3-12b
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import torch

from hlens.jspace import unembed_directions
from hlens.landmarks import compute_landmark_curves, save_landmarks, workspace_range
from hlens.loading import load_model, matched_layer
from hlens.swaps import DEFAULT_ALPHA, DEFAULT_BAND, first_token_ids, item_token_ids, pair_edits, probe_items, run_top1
from jlens.examples import load_wikitext_prompts

PREREG = "hlens/docs/preregistration_replication.md"


def prereg_sha256(path: str = PREREG) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def run_g1(q, key: str, n_prompts: int = 30) -> dict:
    c = compute_landmark_curves(q.model, q.lens, load_wikitext_prompts(n_prompts), vocab_size=q.hf.config.get_text_config().vocab_size)
    wr = workspace_range(c)
    os.makedirs("results", exist_ok=True)
    save_landmarks(c, wr, f"results/landmarks_{key}.json")
    ref = matched_layer(q.model.n_layers, 12)
    t1, ac = c.topk_acc[1], c.autocorr
    i_ref = c.layers.index(ref)
    checks = {
        "top1_last_over_depth0.39": t1[-1] / max(t1[i_ref], 1e-9),
        "autocorr_first3_mean": sum(ac[:3]) / 3,
        "autocorr_max": max(ac),
    }
    passed = checks["top1_last_over_depth0.39"] > 5 and checks["autocorr_first3_mean"] < 0.05 and checks["autocorr_max"] > 0.5
    return {"checks": checks, "passed": bool(passed), "ref_layer": ref,
            "workspace_range": {"onset": wr.onset, "offset": wr.offset, "onset_depth": wr.onset_depth, "offset_depth": wr.offset_depth},
            "onset_by_metric": wr.onset_by_metric, "offset_by_metric": wr.offset_by_metric}


def _random_pair(d: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    return lambda layer: torch.randn(2, d, generator=g)


def run_g2(q, *, alphas=(DEFAULT_ALPHA,), band=None, path: str = "data/experiments/probe-swap.json", fold_case: bool = False, log=print) -> dict:
    model, tok = q.model, q.tokenizer
    n = model.n_layers
    lo, hi = band or (matched_layer(n, DEFAULT_BAND[0]), matched_layer(n, DEFAULT_BAND[1]))
    device = next(q.hf.parameters()).device
    jac = q.lens.jacobians
    items = probe_items(tok, path, fold_case=fold_case)
    tokens = sorted({x for _, s, t in items for x in (s, t)})
    U = unembed_directions(model, tokens, dim_batch=64)
    row = {t: i for i, t in enumerate(tokens)}
    todo = []
    for it, s, t in items:
        ids = model.encode(it.prompt, max_length=128)
        ans, sw = item_token_ids(tok, it, fold_case=fold_case)
        if run_top1(model, ids) in ans:
            todo.append((it, ids, U[[row[s], row[t]]], ans, sw))
    N = len(todo)
    log(f"{len(items)} single-token items; baseline correct {N}; band L{lo}-{hi}")
    out = {"band": [lo, hi], "n_items": len(items), "n_baseline_correct": N, "by_alpha": {}}
    for alpha in alphas:
        hits, per_item = 0, []
        for it, ids, U2, ans, sw in todo:
            edits = pair_edits(range(lo, hi + 1), lambda l, U2=U2: U2 @ jac[l], alpha, device)
            hit = run_top1(model, ids, edits) in sw
            hits += hit
            per_item.append({"name": it.name, "category": it.category, "hit": bool(hit)})
        ctrl = []
        for seed in range(3):
            h = 0
            for i, (it, ids, U2, ans, sw) in enumerate(todo):
                vec = _random_pair(model.d_model, 1000 * seed + i)
                h += run_top1(model, ids, pair_edits(range(lo, hi + 1), vec, alpha, device)) in sw
            ctrl.append(h)
        out["by_alpha"][str(alpha)] = {"hits": hits, "rate": hits / max(N, 1), "random_control_hits": ctrl, "per_item": per_item}
        log(f"alpha={alpha}: hits {hits}/{N} ({hits / max(N, 1):.0%}); random control {ctrl}")
    main = out["by_alpha"][str(DEFAULT_ALPHA)]
    out["passed"] = bool(N >= 20 and main["rate"] >= 0.25 and main["hits"] >= 5 * max(max(main["random_control_hits"]), 1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-3-12b")
    ap.add_argument("--sweep", action="store_true", help="exploratory alpha sweep in addition to the pre-registered alpha")
    a = ap.parse_args()
    q = load_model(a.model, dtype=torch.bfloat16)
    res = {"model": a.model, "prereg_sha256": prereg_sha256(), "G1": run_g1(q, a.model)}
    print(json.dumps(res["G1"], indent=1))
    res["G2"] = run_g2(q, alphas=(DEFAULT_ALPHA, 0.9, 1.0, 1.5, 2.0) if a.sweep else (DEFAULT_ALPHA,))
    print(json.dumps(res["G2"], indent=1))
    with open(f"results/gates_{a.model}.json", "w") as f:
        json.dump(res, f)

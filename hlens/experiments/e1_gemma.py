"""E1 (engagement) on Gemma 3 12B, held-out swap suite, per hlens/docs/preregistration_e1_gemma.md. Exact forward passes only.
Two stages so that only one 45 GiB fp32 model is resident at a time:  `real` (labels, margin paths) then `null` (the weight-randomized model's
single-layer path with the real model's ans/swap ids), then `analyze`.

Run:  .venv/bin/python -m hlens.experiments.e1_gemma real
      .venv/bin/python -m hlens.experiments.e1_gemma null
      .venv/bin/python -m hlens.experiments.e1_gemma analyze
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

from hlens.experiments import engagement
from hlens.experiments.engagement import ALPHAS, analyze_e1, margin_path, path_summary
from hlens.jspace import unembed_directions
from hlens.loading import MODELS, load_hf, load_model, matched_layer
from hlens.swaps import item_token_ids, pair_edits, probe_items, run_top1
from jlens import JacobianLens
from jlens.hooks import ActivationRecorder

PREREG = "hlens/docs/preregistration_e1_gemma.md"
SUITE = "data/experiments/probe-swap-heldout.json"
OUT = "results/e1_gemma_heldout.json"
NULL_LENS = "results/h2_replication/B2_lens_gemma-3-12b_s10_L24.pt"
NULL_SEED, NULL_LAYER = 10, 24
MODEL = "gemma-3-12b"
A_REF = 1.0
BAND = (18, 30)
SINGLE = 24


def prereg_sha256(path: str = PREREG) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def clean_logits(model, ids) -> torch.Tensor:
    last = model.n_layers - 1
    with torch.no_grad(), ActivationRecorder(model.layers, at=[last]) as rec:
        model.forward(ids)
        return model.unembed(rec.activations[last][0, -1]).float()


def run_real(q, *, log=print) -> list[dict]:
    model, tok, lens = q.model, q.tokenizer, q.lens
    items = probe_items(tok, SUITE, fold_case=True)
    tokens = sorted({x for _, s, t in items for x in (s, t)})
    row_of = {t: i for i, t in enumerate(tokens)}
    U = unembed_directions(model, tokens, dim_batch=64)
    dev = next(iter(model.layers[0].parameters())).device
    band = list(range(BAND[0], BAND[1] + 1))
    out, t0 = [], time.time()
    for i, (it, s_id, t_id) in enumerate(items):
        ids = model.encode(it.prompt, max_length=128)
        ans, swap_ans = item_token_ids(tok, it, fold_case=True)
        lg = clean_logits(model, ids)
        ans_id = lg.argmax().item()
        if ans_id not in ans:
            continue
        pair = [row_of[s_id], row_of[t_id]]
        edits = pair_edits(band, lambda l: U[pair].to(dev) @ lens.jacobians[l].to(dev), A_REF, dev)
        hit = run_top1(model, ids, edits) in swap_ans
        swap_id = max(swap_ans, key=lambda t: lg[t].item())
        out.append({"name": it.name, "category": it.category, "hit": bool(hit), "ans_id": ans_id, "swap_id": swap_id, "pair": pair,
                    "band": path_summary(margin_path(model, lens, U[pair], ids, ans_id, swap_id, band), a_ref=A_REF),
                    "L16": path_summary(margin_path(model, lens, U[pair], ids, ans_id, swap_id, [SINGLE]), a_ref=A_REF)})
        if (i + 1) % 10 == 0:
            log(f"  {i + 1}/{len(items)} items ({len(out)} baseline-correct), {time.time() - t0:.0f}s")
    return out


def build_null():
    """The seed-10 randomized Gemma exactly as `h2_replication.null_arm` made it: fp32 on the CPU, `reinit`, weights rounded to bf16 precision, then fp32 on the GPU."""
    import transformers

    from hlens.harness import randomize_model

    spec = MODELS[MODEL]
    tok = transformers.AutoTokenizer.from_pretrained(spec.model_id)
    hf = load_hf(dtype=torch.float32, device="cpu", model_id=spec.model_id)
    model = randomize_model(hf, tok, mode="reinit", seed=NULL_SEED)
    with torch.no_grad():
        for p in hf.parameters():
            p.data = p.data.to(torch.bfloat16).to(torch.float32)
    hf.to("cuda")
    return model, tok


def null_lens_check(model, lens, prompts, layer: int = NULL_LAYER) -> dict:
    """Does the lens belong to these weights? Transported layer-`layer` residuals should align with the final residual better than the identity map does."""
    last = model.n_layers - 1
    cos = torch.nn.functional.cosine_similarity
    a, b = [], []
    for p in prompts:
        ids = model.encode(p, max_length=64)
        with torch.no_grad(), ActivationRecorder(model.layers, at=[layer, last]) as rec:
            model.forward(ids)
            h, hf_ = rec.activations[layer][0, 1:].float(), rec.activations[last][0, 1:].float()
            a.append(cos(lens.transport(h, layer), hf_, dim=-1).mean().item())
            b.append(cos(h, hf_, dim=-1).mean().item())
    return {"cos_transported": float(np.mean(a)), "cos_identity": float(np.mean(b)), "lens_matches_weights": bool(np.mean(a) > np.mean(b))}


def run_null(real_items: list[dict], *, log=print) -> tuple[dict, dict]:
    from jlens.examples import load_wikitext_prompts

    model, tok = build_null()
    lens = JacobianLens.load(NULL_LENS)
    check = null_lens_check(model, lens, load_wikitext_prompts(5))
    log(f"null lens check: {check}")
    dev = next(iter(model.layers[0].parameters())).device
    out = {}
    items = {it.name: (it, s, t) for it, s, t in probe_items(tok, SUITE, fold_case=True)}
    toks = sorted({x for it, s, t in items.values() for x in (s, t)})
    row_of = {t: i for i, t in enumerate(toks)}
    U = unembed_directions(model, toks, dim_batch=64)
    t0 = time.time()
    for i, r in enumerate(real_items):
        it, s_id, t_id = items[r["name"]]
        ids = model.encode(it.prompt, max_length=128)
        pair = [row_of[s_id], row_of[t_id]]
        out[r["name"]] = path_summary(margin_path(model, lens, U[pair], ids, r["ans_id"], r["swap_id"], [SINGLE]), a_ref=A_REF)
        if (i + 1) % 20 == 0:
            log(f"  null {i + 1}/{len(real_items)}, {time.time() - t0:.0f}s")
    return out, check


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "real":
        q = load_model(MODEL, dtype=torch.float32)
        items = run_real(q)
        os.makedirs("results", exist_ok=True)
        json.dump({"prereg_sha256": prereg_sha256(), "a_ref": A_REF, "band": BAND, "single_layer": SINGLE, "alphas": ALPHAS, "items": items}, open(OUT, "w"))
        print(f"{len(items)} baseline-correct items, {sum(r['hit'] for r in items)} successes")
    elif mode == "null":
        d = json.load(open(OUT))
        paths, check = run_null(d["items"])
        for r in d["items"]:
            r["L16_null"] = paths[r["name"]]
        d["null_lens_check"] = check
        json.dump(d, open(OUT, "w"))
    else:
        d = json.load(open(OUT))
        assert all("L16_null" in r for r in d["items"]), "run the null stage first"
        res = analyze_e1(d["items"])
        res["prereg_sha256"] = {"recorded": d["prereg_sha256"], "current": prereg_sha256()}
        res["null_lens_check"] = d.get("null_lens_check")
        d["analysis"] = res
        json.dump(d, open(OUT, "w"))
        print(json.dumps(res, indent=1))

"""E1 (the engagement hypothesis), per hlens/docs/preregistration_e1.md. Exact forward passes only.

Run:  uv run python -m hlens.experiments.engagement          (real model + weight-randomized null)
      uv run python -m hlens.experiments.engagement --analyze
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
from scipy import stats

from hlens.experiments.trust_score import GEOGRAPHY
from hlens.interventions import ActivationEditor
from hlens.jspace import unembed_directions
from hlens.stats import auc, bootstrap_auc, logistic_loo_auc, permutation_pvalue, stratified_auc, stratified_permutation_pvalue
from hlens.swaps import DEFAULT_BAND, first_token_ids, item_token_ids, pair_edits, probe_items
from jlens.hooks import ActivationRecorder

PREREG = "hlens/docs/preregistration_e1.md"
OUT = "results/e1_engagement_qwen3.5-4b.json"
ALPHAS = [round(a, 2) for a in np.arange(-0.05, 2.0001, 0.05)]
A_REF = 1.25


def prereg_sha256(path: str = PREREG) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def margin(model, ids, edits, ans_id: int, swap_id: int) -> float:
    last = model.n_layers - 1
    with torch.no_grad(), contextlib.ExitStack() as stack:
        for layer, fn in edits:
            stack.enter_context(ActivationEditor(model.layers, at=[layer], edit_fn=fn))
        rec = stack.enter_context(ActivationRecorder(model.layers, at=[last]))
        model.forward(ids)
        lg = model.unembed(rec.activations[last][0, -1]).float()
    return (lg[swap_id] - lg[ans_id]).item()


def margin_path(model, lens, U2, ids, ans_id, swap_id, layers, alphas=ALPHAS) -> list[float]:
    dev = next(iter(model.layers[0].parameters())).device
    return [margin(model, ids, pair_edits(layers, lambda l: U2.to(dev) @ lens.jacobians[l].to(dev), a, dev) if a != 0 else [], ans_id, swap_id)
            for a in alphas]


def path_summary(path: list[float], alphas=ALPHAS, a_ref: float = A_REF) -> dict:
    a = np.array(alphas)
    m = np.array(path)
    m0 = float(m[a == 0.0][0])
    g = float((m[a == 0.05][0] - m[a == -0.05][0]) / 0.1)
    ref = int(np.where(np.isclose(a, a_ref))[0][0])
    fo = m0 + a * g
    N = float(m[ref] - fo[ref])
    pos = a > 0
    cross = np.where(pos & (m > 0))[0]
    alpha_c = None
    if len(cross):
        i = cross[0]
        alpha_c = float(a[i - 1] + (0 - m[i - 1]) / (m[i] - m[i - 1]) * (a[i] - a[i - 1])) if i > 0 and m[i - 1] <= 0 else float(a[i])
    dev = np.abs(m - fo) > 0.25 * np.abs(m - m0)
    ok = np.where((a >= 0.2) & dev & (np.abs(m - m0) > 1e-3))[0]
    return {"m0": m0, "g": g, "FO_ref": float(fo[ref]), "m_ref": float(m[ref]), "N": N, "alpha_c": alpha_c,
            "alpha_n": float(a[ok[0]]) if len(ok) else None, "path": [float(x) for x in m]}


def run(q32, null, rows, *, items_path: str = "data/experiments/probe-swap.json", fold_case: bool = False, log=print) -> list[dict]:
    tok = q32.tokenizer
    by_name = {it.name: (it, s, t) for it, s, t in probe_items(tok, items_path, fold_case=fold_case)}
    todo = [r for r in rows if r["base_ok"] and r["name"] in by_name]
    tokens = sorted({x for r in todo for x in by_name[r["name"]][1:]})
    row_of = {t: i for i, t in enumerate(tokens)}
    U = unembed_directions(q32.model, tokens, dim_batch=256)
    Un = unembed_directions(null.model, tokens, dim_batch=256)
    out, t0 = [], time.time()
    for i, r in enumerate(todo):
        it, s_id, t_id = by_name[r["name"]]
        pair = [row_of[s_id], row_of[t_id]]
        ids = q32.model.encode(it.prompt, max_length=128)
        last = q32.model.n_layers - 1
        with torch.no_grad(), ActivationRecorder(q32.model.layers, at=[last]) as rec:
            q32.model.forward(ids)
            lg = q32.model.unembed(rec.activations[last][0, -1]).float()
        ans_id = lg.argmax().item()
        swap_id = max(item_token_ids(tok, it, fold_case=fold_case)[1], key=lambda t: lg[t].item())
        band = list(range(DEFAULT_BAND[0], DEFAULT_BAND[1] + 1))
        nids = null.model.encode(it.prompt, max_length=128)
        out.append({
            "name": it.name, "category": it.category, "hit": r["hit_a1.25"],
            "band": path_summary(margin_path(q32.model, q32.lens, U[pair], ids, ans_id, swap_id, band)),
            "L16": path_summary(margin_path(q32.model, q32.lens, U[pair], ids, ans_id, swap_id, [16])),
            "L16_null": path_summary(margin_path(null.model, null.lens, Un[pair], nids, ans_id, swap_id, [16])),
            "loading_mean": r["layers"]["16"]["loading_mean"],
        })
        if (i + 1) % 10 == 0:
            log(f"  {i + 1}/{len(todo)} items, {time.time() - t0:.0f}s")
    return out


def analyze_e1(items: list[dict], *, n_boot: int = 10000, seed: int = 0) -> dict:
    y = np.array([1 if r["hit"] else 0 for r in items])  # positive = success
    groups = np.array(["geography" if r["category"] in GEOGRAPHY else "other" for r in items])
    N = np.array([r["band"]["N"] for r in items])
    p1_auc, p1_p = auc(N, y), permutation_pvalue(N, y, n=10000, seed=seed)
    s_auc, s_p = stratified_auc(N, y, groups), stratified_permutation_pvalue(N, y, groups, n=10000, seed=seed)
    p1 = bool(p1_p < 0.01 and s_auc > 0.5 and s_p < 0.05)
    succ = [r for r in items if r["hit"]]
    k, n = sum(r["band"]["FO_ref"] < 0 for r in succ), len(succ)
    p2_p = float(stats.binomtest(k, n, 0.5, alternative="greater").pvalue) if n else 1.0
    p2 = bool(p2_p < 0.01)
    rng = np.random.default_rng(seed)
    a_real = auc(np.array([r["L16"]["N"] for r in items]), y)
    a_null = auc(np.array([r["L16_null"]["N"] for r in items]), y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        ar, an = auc(np.array([items[i]["L16"]["N"] for i in idx]), y[idx]), auc(np.array([items[i]["L16_null"]["N"] for i in idx]), y[idx])
        if not (np.isnan(ar) or np.isnan(an)):
            diffs.append(ar - an)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p4 = bool(lo > 0)
    FO = np.array([r["band"]["FO_ref"] for r in items])
    M = np.array([r["band"]["m_ref"] for r in items])
    p5_auc, p5_p = auc(FO, y), permutation_pvalue(FO, y, n=10000, seed=seed)
    p5 = bool(p5_auc > 0.5 and p5_p < 0.01 and p5_auc > p1_auc)
    both = [r for r in succ if r["band"]["alpha_n"] is not None and r["band"]["alpha_c"] is not None]
    has_loading = all("loading_mean" in r for r in items)  # the Gemma run does not compute the workspace loading
    load = np.array([r["loading_mean"] for r in items]) if has_loading else None
    return {
        "prereg_sha256": {"recorded": None, "current": prereg_sha256()},
        "n": len(items), "n_success": int(y.sum()),
        "P1": {"auc_N_for_success": p1_auc, "ci95": list(bootstrap_auc(N, y, n=2000)), "perm_p": p1_p, "stratified_auc": s_auc, "stratified_p": s_p, "passed": p1},
        "P2": {"successes": n, "first_order_would_not_flip": k, "p_binomial": p2_p, "passed": p2},
        "P4": {"auc_real_L16": a_real, "auc_null_L16": a_null, "diff_ci95": [float(lo), float(hi)], "passed": p4},
        "P5_first_order": {"auc_FO_for_success": p5_auc, "perm_p": p5_p, "auc_margin": auc(M, y), "auc_N": p1_auc,
                           "passed": p5, "note": "added for the held-out replication (preregistration_heldout_scoring_h1_e1.md); the dev-suite analysis did not compute it"},
        "E1_supported": bool(p1 and p2 and p4),
        "exploratory": {
            "knee_alignment": {"n": len(both), "frac_alpha_n_le_alpha_c": float(np.mean([r["band"]["alpha_n"] <= r["band"]["alpha_c"] for r in both])) if both else None,
                               "spearman": float(stats.spearmanr([r["band"]["alpha_n"] for r in both], [r["band"]["alpha_c"] for r in both]).statistic) if len(both) > 2 else None},
            "loo_logistic_auc": {"loading": logistic_loo_auc(load, y), "loading+N": logistic_loo_auc(np.column_stack([load, N]), y)} if has_loading else None,
            "success_by_margin_sign": {"margin_positive": int(sum(r["band"]["m_ref"] > 0 for r in items)), "hit": int(y.sum())},
            "mean_N_success_vs_fail": [float(N[y == 1].mean()), float(N[y == 0].mean())],
            "mean_FO_success_vs_fail": [float(np.mean([r["band"]["FO_ref"] for r in succ])), float(np.mean([r["band"]["FO_ref"] for r in items if not r["hit"]]))],
        },
    }


if __name__ == "__main__":
    if "--analyze" in sys.argv:
        d = json.load(open(OUT))
        res = analyze_e1(d["items"])
        res["prereg_sha256"]["recorded"] = d["prereg_sha256"]
        d["analysis"] = res
        json.dump(d, open(OUT, "w"))
        print(json.dumps(res, indent=1))
        sys.exit(0)
    import transformers

    from hlens.harness import randomize_model
    from hlens.loading import MODEL_ID, load_hf, load_qwen
    from jlens import JacobianLens

    saved = json.load(open("results/trust_score_qwen3.5-4b.json"))
    q32 = load_qwen(dtype=torch.float32)
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    nm = randomize_model(load_hf(dtype=torch.float32), tok, mode="reinit", seed=0)

    class _Null:
        model = nm
        lens = JacobianLens.load("results/nulls/lens_randomized_reinit_s0_L16.pt")

    items = run(q32, _Null, saved["rows"])
    os.makedirs("results", exist_ok=True)
    json.dump({"prereg_sha256": prereg_sha256(), "alphas": ALPHAS, "items": items}, open(OUT, "w"))

import torch
from hlens.interventions import coordinate_swap
from hlens.hvp import hvp_norm_form

from collections.abc import Callable


def swap_delta(h: torch.Tensor, v_s: torch.Tensor, v_t: torch.Tensor, *, alpha: float = 1.0) -> torch.Tensor:
    return coordinate_swap(h, v_s, v_t, alpha=alpha) - h


def taylor_terms(
    f: Callable[[torch.Tensor], torch.Tensor],
    h: torch.Tensor,
    delta_h: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(1/2 d^T H d, J d) as tensors (same shape as f's output): the second- and
    first-order Taylor terms along delta_h. `hvp_norm_form` returns the FULL
    contraction, so the 1/2 is applied here. f(h + a d) ~ f(h) + a J d + a^2 (1/2 d^T H d)."""
    quad = 0.5 * hvp_norm_form(f, h, delta_h)
    _, lin = torch.func.jvp(f, (h,), (delta_h,))
    return quad, lin


def trust_terms(
    f: Callable[[torch.Tensor], torch.Tensor],
    h: torch.Tensor,
    delta_h: torch.Tensor,
) -> tuple[float, float]:
    """(||1/2 d^T H d||, ||J d||)."""
    quad, lin = taylor_terms(f, h, delta_h)
    return quad.norm().item(), lin.norm().item()


def trust_score(
    f: Callable[[torch.Tensor], torch.Tensor],
    h: torch.Tensor,
    delta_h: torch.Tensor,
) -> float:
    numerator, denominator = trust_terms(f, h, delta_h)
    return numerator / denominator


# ======================================================================================
# H1 orchestration (research notes §3 H1, §5.1; plan §7): does the trust score predict
# lens-coordinate swap failure on probe-swap.json?
#
# Per (item, layer l in the band): delta_l = the swap delta applied to the clean residual
# at layer l at EVERY position (alpha = 1; tau scales exactly linearly in alpha, because
# the numerator is quadratic and the denominator linear in delta, so ranking is
# alpha-independent and tau(1.25) = 1.25 * tau(1)). f_l = the final-layer residual at the
# last position as a function of the layer-l residual (all positions), so
# tau_l = ||1/2 delta^T H delta|| / ||J delta|| is the local nonlinearity, along the real
# intervention, of the layer-l -> answer-position map. Outcome: top-1 flips to the
# swap_answer under the real multi-layer swap (band, one editor per layer).
# Baselines the trust score must beat (H1's falsifier): "workspace loading" (cosine of the
# residual with the source lens vector, paper §3.4) and the first-order effect ||J delta||.
# Null: the same tau computed on a weight-randomized model with its own lens (layer 16).
# ======================================================================================

import json
import time

import numpy as np

from hlens.contexts import ContextPoint  # noqa: F401  (re-exported for callers)
from hlens.functional import make_slice_fn
from hlens.jspace import unembed_directions
from hlens.stats import (
    auc, bootstrap_auc, logistic_loo_auc, permutation_pvalue, spearman, stratified_auc, stratified_permutation_pvalue,
)
from hlens.swaps import DEFAULT_ALPHA, DEFAULT_BAND, first_token_ids, item_token_ids, pair_edits, probe_items, run_top1
from jlens.hooks import ActivationRecorder

LAYERS = (12, 16, 20)
# Post-hoc grouping (defined after the first analysis showed tau tracks category): items whose
# bridge entity is a country/city/language, the paper's best-swapping "country facts".
GEOGRAPHY = frozenset({"city-capital", "city-continent", "city-currency", "city-language", "language-capital", "river-capital"})


def _device(model):
    return next(iter(model.layers[0].parameters())).device


def _swap_vecs(U2: torch.Tensor, lens, layer: int) -> torch.Tensor:
    """Raw lens vectors [2, d_model] for (source, target) at `layer`."""
    return U2 @ lens.jacobians[layer].to(U2.device)


def _tau_at_layer(model, lens, U2, ids, layer: int) -> dict:
    """tau and its ingredients at one layer for the alpha=1 swap on the clean residual."""
    with torch.no_grad(), ActivationRecorder(model.layers, at=[layer]) as rec:
        model.forward(ids)
        h = rec.activations[layer].detach().clone()
    v = _swap_vecs(U2.to(h.device, h.dtype), lens, layer)
    delta = swap_delta(h, v[0], v[1], alpha=1.0)
    slice_fn = make_slice_fn(model, layer, model.n_layers - 1)
    f = lambda x: slice_fn(x)[0, -1]
    num, den = trust_terms(f, h, delta)
    cos = torch.nn.functional.cosine_similarity(h[0, 1:], v[0][None], dim=-1)
    return {
        "tau": num / den, "num": num, "den": den, "delta_norm": delta.norm().item(),
        "h_norm": h.norm().item(), "loading_mean": cos.mean().item(), "loading_last": cos[-1].item(),
    }


def score_item(
    model, lens, U2, it, ids, *, layers=LAYERS, band=DEFAULT_BAND, alphas=(1.0, DEFAULT_ALPHA), tok, fold_case: bool = False,
) -> dict:
    """Swap outcomes (hit = top-1 equals swap_answer) at each alpha, plus tau at `layers`."""
    ans, swap_ans = item_token_ids(tok, it, fold_case=fold_case)
    base = run_top1(model, ids)
    row = {"name": it.name, "category": it.category, "n_tokens": ids.shape[1], "base_ok": base in ans}
    dev = _device(model)
    for a in alphas:
        edits = pair_edits(range(band[0], band[1] + 1), lambda l: _swap_vecs(U2.to(dev), lens, l), a, dev)
        top = run_top1(model, ids, edits)
        row[f"hit_a{a:g}"] = top in swap_ans
        row[f"moved_a{a:g}"] = top not in ans
    row["layers"] = {str(l): _tau_at_layer(model, lens, U2, ids, l) for l in layers}
    return row


def run_trust_score(
    q32,
    *,
    q16=None,
    null=None,
    layers=LAYERS,
    band=DEFAULT_BAND,
    alphas=(1.0, DEFAULT_ALPHA),
    items=None,
    null_layer: int = 16,
    log_every: int = 10,
    fold_case: bool = False,
    checkpoint: str | None = None,
) -> list[dict]:
    """`q32`: fp32 Loaded (tau and fp32 swap labels). `q16`: optional bf16 Loaded (bf16
    swap labels, the precision the sweep was tuned in). `null`: optional object with
    `.model` / `.lens` for the randomized-weights model (tau at `null_layer` only)."""
    tok = q32.tokenizer
    items = items if items is not None else probe_items(tok, fold_case=fold_case)
    tokens = sorted({x for _, s, t in items for x in (s, t)})
    row_of = {t: i for i, t in enumerate(tokens)}
    U32 = unembed_directions(q32.model, tokens, dim_batch=256)
    U16 = unembed_directions(q16.model, tokens, dim_batch=256) if q16 else None
    Unull = unembed_directions(null.model, tokens, dim_batch=256) if null else None
    rows, t0 = [], time.time()
    done = {}
    if checkpoint and __import__("os").path.exists(checkpoint):
        for line in open(checkpoint):
            r = json.loads(line)
            done[r["name"]] = r
        print(f"  resuming: {len(done)} items in {checkpoint}", flush=True)
    for i, (it, s_id, t_id) in enumerate(items):
        if it.name in done:
            rows.append(done[it.name])
            continue
        pair = [row_of[s_id], row_of[t_id]]
        ids = q32.model.encode(it.prompt, max_length=128)
        row = score_item(q32.model, q32.lens, U32[pair], it, ids, layers=layers, band=band, alphas=alphas, tok=tok, fold_case=fold_case)
        if q16:
            ans, swap_ans = first_token_ids(tok, it.answer), first_token_ids(tok, it.swap_answer)
            ids16 = q16.model.encode(it.prompt, max_length=128)
            row["base_ok16"] = run_top1(q16.model, ids16) in ans
            dev = _device(q16.model)
            for a in alphas:
                edits = pair_edits(range(band[0], band[1] + 1), lambda l: _swap_vecs(U16[pair].to(dev), q16.lens, l), a, dev)
                top = run_top1(q16.model, ids16, edits)
                row[f"hit16_a{a:g}"] = top in swap_ans
        if null:
            nids = null.model.encode(it.prompt, max_length=128)
            row["null"] = _tau_at_layer(null.model, null.lens, Unull[pair], nids, null_layer)
        rows.append(row)
        if checkpoint:
            with open(checkpoint, "a") as fh:
                fh.write(json.dumps(row, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
        if log_every and (i + 1) % log_every == 0:
            print(f"  {i + 1}/{len(items)} items, {time.time() - t0:.0f}s", flush=True)
    return rows


def _wilson_free_ci(scores, y):
    lo, hi = bootstrap_auc(scores, y, n=2000)
    return round(lo, 3), round(hi, 3)


def analyze(rows: list[dict], *, alpha: float = DEFAULT_ALPHA, layers=LAYERS, label: str = "hit", mid_layer: int = 16) -> dict:
    """AUC of each score for predicting swap FAILURE (positive class) among items the model
    answers correctly at baseline. Pre-specified directions: high tau -> failure; low
    loading -> failure; small first-order effect ||J d|| -> failure."""
    key = f"{label}_a{alpha:g}"
    base_key = "base_ok" if label == "hit" else "base_ok16"
    pop = [r for r in rows if r[base_key] and key in r]
    y = np.array([0 if r[key] else 1 for r in pop])
    out = {"label": key, "n": len(pop), "n_fail": int(y.sum()), "n_success": int((1 - y).sum())}
    if out["n_fail"] == 0 or out["n_success"] == 0:
        return out

    def col(field, l):
        return np.array([r["layers"][str(l)][field] for r in pop])

    scores = {}
    for l in layers:
        scores[f"tau_L{l}"] = col("tau", l)
    scores["tau_mean"] = np.mean([col("tau", l) for l in layers], axis=0)
    scores["tau_max"] = np.max([col("tau", l) for l in layers], axis=0)
    for l in layers:
        scores[f"baseline_neg_loading_L{l}"] = -col("loading_mean", l)
    scores["baseline_neg_loading_mean"] = -np.mean([col("loading_mean", l) for l in layers], axis=0)
    scores["baseline_neg_jdelta_mean"] = -np.mean([col("den", l) for l in layers], axis=0)
    scores["num_mean"] = np.mean([col("num", l) for l in layers], axis=0)
    scores["baseline_delta_norm_mean"] = np.mean([col("delta_norm", l) for l in layers], axis=0)
    if all("null" in r for r in pop):
        scores["null_tau_L16"] = np.array([r["null"]["tau"] for r in pop])
        scores["null_baseline_neg_loading_L16"] = -np.array([r["null"]["loading_mean"] for r in pop])
    out["auc"] = {
        k: {"auc": round(auc(v, y), 3), "ci95": _wilson_free_ci(v, y), "perm_p": round(permutation_pvalue(v, y, n=2000), 4)}
        for k, v in scores.items()
    }
    if mid_layer != 16:  # alias the mid-layer scores under the Qwen key names
        out["auc"]["tau_L16"] = out["auc"][f"tau_L{mid_layer}"]
        out["auc"]["baseline_neg_loading_L16"] = out["auc"][f"baseline_neg_loading_L{mid_layer}"]
    mid = str(mid_layer)
    out["spearman_tau16_vs_loading16"] = round(spearman(col("tau", mid_layer), col("loading_mean", mid_layer)), 3)
    out["spearman_tau16_vs_jdelta16"] = round(spearman(col("tau", mid_layer), col("den", mid_layer)), 3)
    ltau, load = np.log(col("tau", mid_layer)), col("loading_mean", mid_layer)
    out["loo_logistic_auc"] = {
        "loading16": round(logistic_loo_auc(-load, y), 3),
        "log_tau16": round(logistic_loo_auc(ltau, y), 3),
        "loading16+log_tau16": round(logistic_loo_auc(np.column_stack([-load, ltau]), y), 3),
    }
    out["mid_layer"] = mid_layer  # the "16" in the key names above and in tau_L16 / baseline_neg_loading_L16 refers to this layer
    groups = np.array(["geography" if r["category"] in GEOGRAPHY else "other" for r in pop])
    strat = {}
    for name in (f"tau_L{mid_layer}", "tau_mean", f"baseline_neg_loading_L{mid_layer}", "baseline_neg_jdelta_mean", "num_mean"):
        v = scores[name]
        strat[name] = {
            "stratified_auc": round(stratified_auc(v, y, groups), 3),
            "p_greater": round(stratified_permutation_pvalue(v, y, groups, n=3000), 4),
            "p_less": round(stratified_permutation_pvalue(v, y, groups, n=3000, greater=False), 4),
        }
    # key names are kept as "tau_L16" / "baseline_neg_loading_L16" for the mid layer whatever its index (Qwen: 16), so readers need not branch
    if mid_layer != 16:
        strat = {("tau_L16" if k == f"tau_L{mid_layer}" else "baseline_neg_loading_L16" if k == f"baseline_neg_loading_L{mid_layer}" else k): v for k, v in strat.items()}
    out["stratified_by_geography"] = strat
    out["within_group_tau16"] = {
        grp: {"n": int((groups == grp).sum()), "n_fail": int(y[groups == grp].sum()), "auc_tau16": round(auc(scores[f"tau_L{mid_layer}"][groups == grp], y[groups == grp]), 3)}
        for grp in ("geography", "other")
    }
    cats = {}
    for r in pop:
        c = cats.setdefault(r["category"], {"n": 0, "success": 0, "tau16": []})
        c["n"] += 1
        c["success"] += int(r[key])
        c["tau16"].append(r["layers"][mid]["tau"])
    out["by_category"] = {k: {"n": v["n"], "success": v["success"], "mean_tau16": round(float(np.mean(v["tau16"])), 3)} for k, v in sorted(cats.items())}
    if label == "hit" and all("hit16_a%g" % alpha in r for r in rows):
        a32 = [r[key] for r in rows]
        a16 = [r["hit16_a%g" % alpha] for r in rows]
        out["label_agreement_fp32_vs_bf16"] = round(float(np.mean([a == b for a, b in zip(a32, a16)])), 3)
    return out


def save_results(rows, analyses: dict, path: str) -> None:
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({"rows": rows, "analysis": analyses}, f)


# ======================================================================================
# Variants (after the first H1 result, handoff §3.7): per-position tau, a margin-based
# scalar, and the TRUE linearisation error, so the two links of H1's chain can be tested
# separately:   tau  --(1)-->  actual first-order error  --(2)-->  swap failure.
# For a single-layer swap f(h + a d) is compared against its Taylor polynomials, which is
# exactly what a validity score is supposed to predict.
# ======================================================================================


def _clean_logits(model, ids) -> torch.Tensor:
    last = model.n_layers - 1
    with torch.no_grad(), ActivationRecorder(model.layers, at=[last]) as rec:
        model.forward(ids)
        return model.unembed(rec.activations[last][0, -1]).float()


def _eval_site(f, h, delta, alpha: float) -> dict:
    """tau and the true first/second-order errors of f along delta at scale alpha."""
    quad, lin = taylor_terms(f, h, delta)
    with torch.no_grad():
        f0 = f(h)
        true = f(h + alpha * delta) - f0
    first, second = alpha * lin, alpha * lin + alpha**2 * quad
    out = {
        "num": quad.norm().item(), "den": lin.norm().item(), "tau": (quad.norm() / lin.norm()).item(),
        "err1": ((true - first).norm() / true.norm()).item(), "err2": ((true - second).norm() / true.norm()).item(),
    }
    if f0.dim() == 0:  # scalar margin: signed values, so predicted outcomes can be scored
        out.update(m0=f0.item(), g1=lin.item(), q2=quad.item(), true_dm=true.item())
    return out


def score_variants(model, lens, U2, ids, layer: int, ans_id: int, swap_id: int, *, full: bool, alpha: float = DEFAULT_ALPHA) -> dict:
    """Variants at `layer`. Always: resid/all and margin/all. If `full`: also per-position
    (the last position, and the position with maximal source loading = the argument-like
    position; supports "last" and "pstar") for both functions."""
    with torch.no_grad(), ActivationRecorder(model.layers, at=[layer]) as rec:
        model.forward(ids)
        h = rec.activations[layer].detach().clone()
    v = _swap_vecs(U2.to(h.device, h.dtype), lens, layer)
    delta_all = swap_delta(h, v[0], v[1], alpha=1.0)
    seq = h.shape[1]
    cos = torch.nn.functional.cosine_similarity(h[0, 1:], v[0][None], dim=-1)
    p_star = int(cos.argmax().item()) + 1
    slice_fn = make_slice_fn(model, layer, model.n_layers - 1)
    fs = {
        "resid": lambda x: slice_fn(x)[0, -1],
        "margin": lambda x: (lambda lg: lg[swap_id] - lg[ans_id])(model.unembed(slice_fn(x)[0, -1]).float()),
    }

    def at(p):
        d = torch.zeros_like(delta_all)
        d[0, p] = delta_all[0, p]
        return d

    supports = {"all": delta_all}
    if full:
        supports.update({"last": at(seq - 1), "pstar": at(p_star)})
    out = {"p_star": p_star, "loading_max": cos.max().item(), "loading_mean": cos.mean().item(),
           "loading_at_pstar_is_last": p_star == seq - 1, "n_tokens": seq}
    for fk, f in fs.items():
        for sk, d in supports.items():
            out[f"{fk}/{sk}"] = _eval_site(f, h, d, alpha)
    return out


def run_variants(q32, base_rows: list[dict], *, null=None, layers=LAYERS, null_layer: int = 16, log_every: int = 10) -> list[dict]:
    """Variants for the baseline-correct items of `base_rows` (their swap labels are reused).
    Layers other than the middle one get only the all-position variants."""
    tok = q32.tokenizer
    by_name = {it.name: (it, s, t) for it, s, t in probe_items(tok)}
    todo = [r for r in base_rows if r["base_ok"]]
    tokens = sorted({x for r in todo for x in by_name[r["name"]][1:]})
    row_of = {t: i for i, t in enumerate(tokens)}
    U32 = unembed_directions(q32.model, tokens, dim_batch=256)
    Unull = unembed_directions(null.model, tokens, dim_batch=256) if null else None
    out, t0 = [], time.time()
    for i, r in enumerate(todo):
        it, s_id, t_id = by_name[r["name"]]
        pair = [row_of[s_id], row_of[t_id]]
        ids = q32.model.encode(it.prompt, max_length=128)
        logits = _clean_logits(q32.model, ids)
        ans_id = logits.argmax().item()
        swap_id = max(first_token_ids(tok, it.swap_answer), key=lambda t: logits[t].item())  # outcome-independent choice
        row = {"name": it.name, "category": it.category, "ans_id": ans_id, "swap_id": swap_id,
               "hit_a1.25": r["hit_a1.25"], "hit_a1": r["hit_a1"], "hit16_a1.25": r.get("hit16_a1.25"), "layers": {}}
        for l in layers:
            row["layers"][str(l)] = score_variants(q32.model, q32.lens, U32[pair], ids, l, ans_id, swap_id, full=(l == null_layer))
        if null:
            nids = null.model.encode(it.prompt, max_length=128)
            row["null"] = score_variants(null.model, null.lens, Unull[pair], nids, null_layer, ans_id, swap_id, full=True)
        out.append(row)
        if log_every and (i + 1) % log_every == 0:
            print(f"  {i + 1}/{len(todo)} items, {time.time() - t0:.0f}s", flush=True)
    return out


def analyze_variants(vrows: list[dict], *, label: str = "hit_a1.25", mid: int = 16) -> dict:
    """AUCs (failure = positive class; also reported for the success direction) of every
    variant score, plus the two links of H1's chain."""
    y = np.array([0 if r[label] else 1 for r in vrows])
    groups = np.array(["geography" if r["category"] in GEOGRAPHY else "other" for r in vrows])
    out = {"label": label, "n": len(y), "n_fail": int(y.sum())}
    L = lambda r, l: r["layers"][str(l)]
    scores = {}
    for fk in ("resid", "margin"):
        for l in LAYERS:
            scores[f"tau_{fk}_all_L{l}"] = np.array([L(r, l)[f"{fk}/all"]["tau"] for r in vrows])
        for sk in ("last", "pstar"):
            scores[f"tau_{fk}_{sk}_L{mid}"] = np.array([L(r, mid)[f"{fk}/{sk}"]["tau"] for r in vrows])
        scores[f"err1_{fk}_all_L{mid}"] = np.array([L(r, mid)[f"{fk}/all"]["err1"] for r in vrows])  # the TRUE linearisation error
        scores[f"err2_{fk}_all_L{mid}"] = np.array([L(r, mid)[f"{fk}/all"]["err2"] for r in vrows])
    m0 = np.array([L(r, mid)["margin/all"]["m0"] for r in vrows])
    g1 = np.array([L(r, mid)["margin/all"]["g1"] for r in vrows]) * DEFAULT_ALPHA
    q2 = np.array([L(r, mid)["margin/all"]["q2"] for r in vrows]) * DEFAULT_ALPHA**2
    scores["margin_pred_first_order(neg=fail)"] = -(m0 + g1)   # high -> predicted failure
    scores["margin_pred_second_order(neg=fail)"] = -(m0 + g1 + q2)
    scores["baseline_neg_loading_max_L16"] = -np.array([L(r, mid)["loading_max"] for r in vrows])
    scores["baseline_neg_loading_mean_L16"] = -np.array([L(r, mid)["loading_mean"] for r in vrows])
    if all("null" in r for r in vrows):
        for key in ("resid/all", "margin/all", "resid/last", "resid/pstar", "margin/last", "margin/pstar"):
            scores[f"null_tau_{key.replace('/', '_')}_L{mid}"] = np.array([r["null"][key]["tau"] for r in vrows])
    out["auc"] = {}
    for k, v in scores.items():
        a = auc(v, y)
        out["auc"][k] = {
            "auc_fail": round(a, 3), "auc_success": round(1 - a, 3), "ci95_fail": _wilson_free_ci(v, y),
            "perm_p_greater": round(permutation_pvalue(v, y, n=2000), 4),
            "strat_auc": round(stratified_auc(v, y, groups), 3),
            "strat_p_greater": round(stratified_permutation_pvalue(v, y, groups, n=2000), 4),
            "strat_p_less": round(stratified_permutation_pvalue(v, y, groups, n=2000, greater=False), 4),
        }
    # link 1: does tau track the true first-order error?
    out["link1_spearman_tau_vs_true_error"] = {
        f"{fk}_{sk}": round(spearman(scores[f"tau_{fk}_{sk}_L{mid}" if sk != "all" else f"tau_{fk}_all_L{mid}"], scores[f"err1_{fk}_all_L{mid}"]), 3)
        for fk in ("resid", "margin") for sk in ("all",)
    }
    # taylor accuracy summary: how wrong is the first-order prediction, and does the 2nd order help?
    out["median_err1_err2"] = {fk: [round(float(np.median(scores[f"err{o}_{fk}_all_L{mid}"])), 3) for o in (1, 2)] for fk in ("resid", "margin")}
    out["true_margin_flip_rate"] = {
        "first_order_predicts_swap": float(np.mean((m0 + g1) > 0)), "second_order_predicts_swap": float(np.mean((m0 + g1 + q2) > 0)),
        "actual_success": float(1 - y.mean()),
    }
    return out


if __name__ == "__main__":
    import sys

    if "--variants" in sys.argv:
        import transformers

        from hlens.harness import randomize_model
        from hlens.loading import MODEL_ID, load_hf, load_qwen
        from jlens import JacobianLens

        saved = json.load(open("results/trust_score_qwen3.5-4b.json"))
        q32 = load_qwen(dtype=torch.float32)
        tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
        nmodel = randomize_model(load_hf(dtype=torch.float32), tok, mode="reinit", seed=0)

        class _Null:
            model = nmodel
            lens = JacobianLens.load("results/nulls/lens_randomized_reinit_s0_L16.pt")

        vrows = run_variants(q32, saved["rows"], null=_Null)
        res = {"rows": vrows, "analysis": analyze_variants(vrows)}
        with open("results/trust_score_variants_qwen3.5-4b.json", "w") as f:
            json.dump(res, f)
        print(json.dumps(res["analysis"], indent=1))
        sys.exit(0)
    if "--reanalyze-variants" in sys.argv:
        vrows = json.load(open("results/trust_score_variants_qwen3.5-4b.json"))["rows"]
        res = {"rows": vrows, "analysis": analyze_variants(vrows)}
        json.dump(res, open("results/trust_score_variants_qwen3.5-4b.json", "w"))
        print(json.dumps(res["analysis"], indent=1))
        sys.exit(0)
    if "--reanalyze" in sys.argv:  # recompute the analysis from the saved per-item rows
        saved = json.load(open("results/trust_score_qwen3.5-4b.json"))
        analyses = {f"{lab}_a{a:g}": analyze(saved["rows"], alpha=a, label=lab) for lab in ("hit", "hit16") for a in (1.0, DEFAULT_ALPHA)}
        save_results(saved["rows"], analyses, "results/trust_score_qwen3.5-4b.json")
        print(json.dumps(analyses, indent=1))
        sys.exit(0)
    import transformers

    from hlens.harness import randomize_model
    from hlens.loading import MODEL_ID, load_hf, load_qwen
    from jlens import JacobianLens

    q32 = load_qwen(dtype=torch.float32)
    q16 = load_qwen(dtype=torch.bfloat16)
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    hf_null = load_hf(dtype=torch.float32)
    nmodel = randomize_model(hf_null, tok, mode="reinit", seed=0)

    class _Null:
        model = nmodel
        lens = JacobianLens.load("results/nulls/lens_randomized_reinit_s0_L16.pt")

    rows = run_trust_score(q32, q16=q16, null=_Null)
    analyses = {f"{lab}_a{a:g}": analyze(rows, alpha=a, label=lab) for lab in ("hit", "hit16") for a in (1.0, DEFAULT_ALPHA)}
    save_results(rows, analyses, "results/trust_score_qwen3.5-4b.json")
    print(json.dumps(analyses, indent=1))

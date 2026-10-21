"""H3 (binding lives in cross-terms), per hlens/docs/preregistration_h3.md. Read that file first:
design, decision rule and the order of the arms are fixed there.

Run (null arm first, then the real arm, which checks the pre-registration hash):
    uv run python -m hlens.experiments.binding --arm null
    uv run python -m hlens.experiments.binding --arm real
    uv run python -m hlens.experiments.binding --analyze
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

from hlens.contexts import ContextPoint, at_position, logit_fn
from hlens.hvp import restricted_hessian
from hlens.jspace import decompose
from hlens.swaps import first_token_ids
from hlens.synonyms import forms_for, single_token_ids
from jlens.hooks import ActivationRecorder

PREREG = "hlens/docs/preregistration_h3.md"
OUT_DIR = "results/h3"
N_CTRL = 6
PRIMARY = {"layer": 16, "k": 25}
VARIANTS = [{"layer": 16, "k": 10}, {"layer": 16, "k": 40}, {"layer": 14, "k": 25}, {"layer": 18, "k": 25}]


def prereg_sha256(path: str = PREREG) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# --- pure math (unit-tested on toy functions) ------------------------------------------


def ablate_span(x: torch.Tensor, V: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Project x [d] onto the complement of span(V) (V: [m, d] rows). Returns (x', c0) with
    x' = x - V^T c0, c0 = (V V^T)^-1 V x (the coefficients of the removed component)."""
    c0 = torch.linalg.pinv(V @ V.T) @ (V @ x)  # pinv: robust to near-collinear lens vectors
    return x - V.T @ c0, c0


def rank_fraction(value: float, others: np.ndarray) -> float:
    """Fraction of `others` smaller than `value` (ties count 1/2)."""
    return float(((others < value).sum() + 0.5 * (others == value).sum()) / len(others))


def pair_scores(M: np.ndarray, pair: tuple[int, int]) -> float:
    """P: the rank fraction of |M[pair]| among |M| of all OTHER unordered pairs."""
    n = M.shape[0]
    others = np.array([abs(M[i, j]) for i in range(n) for j in range(i + 1, n) if (i, j) != tuple(sorted(pair))])
    return rank_fraction(abs(M[pair]), others)


def taylor2(g: torch.Tensor, H: torch.Tensor, c: torch.Tensor) -> float:
    """Second-order prediction of f(x + sum_i c_i v_i) - f(x) in the basis S (g = <grad f, v_i>,
    H = restricted Hessian): g.c + 1/2 c^T H c."""
    return float(g @ c + 0.5 * c @ H @ c)


def measure_item(f1, x0: torch.Tensor, A: torch.Tensor, B: torch.Tensor, ctrl: torch.Tensor) -> dict:
    """The restricted Hessian over S = [A, B, controls...] (unit rows), the first-order terms, and
    the measured and predicted ablation effects of all singles and pairs."""
    S = torch.cat([A[None], B[None], ctrl])
    n = len(S)
    H = restricted_hessian(f1, x0, S).H.cpu()
    gvec = torch.func.grad(f1)(x0)
    g = (S @ gvec).cpu()
    with torch.no_grad():
        f0 = f1(x0).item()
        delta1, pred1, coeff = {}, {}, {}
        for i in range(n):
            xp, c0 = ablate_span(x0, S[i : i + 1])
            c = torch.zeros(n)
            c[i] = -c0[0].cpu()
            delta1[i], pred1[i], coeff[i] = f1(xp).item() - f0, taylor2(g, H, c), c
        deltaP, predP = {}, {}
        for i in range(n):
            for j in range(i + 1, n):
                xp, c0 = ablate_span(x0, S[[i, j]])
                c = torch.zeros(n)
                c[i], c[j] = -c0[0].cpu(), -c0[1].cpu()
                deltaP[(i, j)], predP[(i, j)] = f1(xp).item() - f0, taylor2(g, H, c)
    I = np.zeros((n, n))
    Ihat = np.zeros((n, n))
    for (i, j), d in deltaP.items():
        I[i, j] = I[j, i] = d - delta1[i] - delta1[j]
        Ihat[i, j] = Ihat[j, i] = predP[(i, j)] - pred1[i] - pred1[j]
    Hm = H.numpy()
    return {"H": Hm.tolist(), "g": g.tolist(), "I": I.tolist(), "Ihat": Ihat.tolist(),
            "delta_single": [delta1[i] for i in range(n)], "delta_pair_AB": deltaP[(0, 1)],
            "P": pair_scores(Hm, (0, 1)), "Q": pair_scores(I, (0, 1)),
            "superadditive_AB": bool(abs(deltaP[(0, 1)]) > abs(delta1[0] + delta1[1]))}


# --- item resolution and the model-facing measurement ------------------------------------


def resolve_item(tok, item) -> dict | None:
    """Token ids of the intermediates' single-token synonyms and the target's first token; None if
    either intermediate has no single-token synonym."""
    a_forms, b_forms = (single_token_ids(tok, forms_for(k)) for k in item["intermediates"][:2])
    if not a_forms or not b_forms:
        return None
    return {"A": a_forms, "B": b_forms, "target_ids": sorted(first_token_ids(tok, item["target"]))}


def measure_prompt(model, lens, D, row_of: dict[int, int], token_of: list[int], res: dict, prompt: str, layer: int, k: int, *, require_correct: bool, max_length: int = 128) -> dict:
    last = model.n_layers - 1
    ids = model.encode(prompt, max_length=max_length)
    pos = ids.shape[1] - 1
    with torch.no_grad(), ActivationRecorder(model.layers, at=[layer, last]) as rec:
        model.forward(ids)
        h = rec.activations[layer].detach().clone()
        final_logits = model.unembed(rec.activations[last][0, pos]).float()
    top1 = final_logits.argmax().item()
    if require_correct and top1 not in res["target_ids"]:
        return {"excluded": "incorrect"}
    tgt = top1 if top1 in res["target_ids"] else max(res["target_ids"], key=lambda t: final_logits[t].item())
    x = h[0, pos]
    with torch.no_grad():
        lens_logits = model.unembed(lens.transport(x.float(), layer)).float()
    realised = {}
    for name in ("A", "B"):
        cand = {f: t for f, t in res[name].items() if t in row_of}
        if not cand:
            return {"excluded": "no_dictionary_row"}
        f_best = max(cand, key=lambda f: lens_logits[cand[f]].item())
        realised[name] = (f_best, cand[f_best])
    forbidden = {t for name in ("A", "B") for t in res[name].values()} | set(res["target_ids"])
    dec = decompose(x.float().cpu(), D, k=k)
    sel = [i for i in dec.indices if token_of[i] not in forbidden]
    point = ContextPoint(input_ids=ids, layer=layer, position=pos, h=h)
    f1, x0 = at_position(logit_fn(model, point, tgt), point)
    gvec = torch.func.grad(f1)(x0).float().cpu()
    sel = sorted(sel, key=lambda i: -abs(float(D[i] @ gvec)))[:N_CTRL]
    if len(sel) < 4:
        return {"excluded": "few_controls"}
    dev, dt = x0.device, x0.dtype
    A, B = (D[row_of[realised[n][1]]].to(dev, dt) for n in ("A", "B"))
    ctrl = D[sel].to(dev, dt)
    out = measure_item(f1, x0, A, B, ctrl)
    out.update({"realised": {n: realised[n][0] for n in ("A", "B")}, "n_ctrl": len(sel), "target_logit_top1": top1 == tgt,
                "control_tokens": [token_of[i] for i in sel], "frac_variance": dec.frac_variance})
    return out


def build_dictionary_with_rows(model, lens, layer: int, vocab_size: int):
    """(D, row_of: token id -> row, token_of: row -> token id)."""
    from hlens.experiments.remainder_coupling import build_dictionary

    D, token_ids = build_dictionary(model, lens, layer, vocab_size)
    token_of = token_ids.tolist()
    return D, {t: r for r, t in enumerate(token_of)}, token_of


# --- analysis (pre-registered) --------------------------------------------------------------


def wilcoxon_greater(values: np.ndarray) -> tuple[float, float]:
    d = np.asarray(values) - 0.5
    if np.all(d == 0):
        return 0.0, 1.0
    r = stats.wilcoxon(d, alternative="greater")
    return float(r.statistic), float(r.pvalue)


def summarize_variant(rows: list[dict]) -> dict:
    ok = [r for r in rows if "H" in r]
    P, Q = np.array([r["P"] for r in ok]), np.array([r["Q"] for r in ok])
    return {"n": len(ok), "T": float(P.mean()), "p_T": wilcoxon_greater(P)[1], "T_I": float(Q.mean()), "p_I": wilcoxon_greater(Q)[1]}


def analyze_h3(real: dict, null: dict, *, n_boot: int = 10000, seed: int = 0) -> dict:
    prim = f"L{PRIMARY['layer']}_k{PRIMARY['k']}"
    rr = [r for r in real["variants"][prim] if "H" in r]
    names = {r["name"] for r in rr}
    nn = [r for r in null["variants"][prim] if "H" in r and r["name"] in names]
    Pr, Pn = np.array([r["P"] for r in rr]), np.array([r["P"] for r in nn])
    rng = np.random.default_rng(seed)
    diffs = [Pr[rng.integers(0, len(Pr), len(Pr))].mean() - Pn[rng.integers(0, len(Pn), len(Pn))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    primary = summarize_variant(real["variants"][prim])
    stab = {k: summarize_variant(v) for k, v in real["variants"].items() if k != prim}
    leg0 = primary["T"] > 0.5 and primary["p_T"] < 0.01
    leg_i = primary["T_I"] > 0.5 and primary["p_I"] < 0.01
    leg_ii = all(v["T"] > 0.5 and v["p_T"] < 0.05 for v in stab.values())
    leg_iii = lo > 0
    hs = np.concatenate([np.abs(np.array(r["H"]))[np.triu_indices(np.array(r["H"]).shape[0], 1)] for r in rr])
    iv = np.concatenate([np.abs(np.array(r["I"]))[np.triu_indices(np.array(r["I"]).shape[0], 1)] for r in rr])
    ih = np.concatenate([np.abs(np.array(r["Ihat"]))[np.triu_indices(np.array(r["Ihat"]).shape[0], 1)] for r in rr])
    gated = float(np.mean(np.abs(ih - iv) <= 0.5 * np.maximum(iv, 1e-12)))
    return {
        "prereg_sha256": {"real": real["prereg_sha256"], "null": null["prereg_sha256"], "current": prereg_sha256()},
        "n_items": {"real_included": len(rr), "null_matched": len(nn), "excluded_real": real["excluded_counts"]},
        "leg0_hessian": {**primary, "criterion": "T>0.5, p<0.01", "passed": bool(leg0)},
        "leg_i_causal": {"T_I": primary["T_I"], "p_I": primary["p_I"], "passed": bool(leg_i)},
        "leg_ii_stability": {"variants": stab, "criterion": "each T>0.5 and p<0.05", "passed": bool(leg_ii)},
        "leg_iii_null": {"T_null": float(Pn.mean()), "T_real_minus_T_null_ci95": [float(lo), float(hi)], "passed": bool(leg_iii)},
        "H3_supported": bool(leg0 and leg_i and leg_ii and leg_iii),
        "exploratory": {
            "spearman_absH_vs_absI_all_pairs": float(stats.spearmanr(hs, iv).statistic),
            "validity_gated_fraction_taylor2_within_50pct": gated,
            "superadditive_fraction_binding_pair": float(np.mean([r["superadditive_AB"] for r in rr])),
        },
    }


def run_arm(model, lens, vocab_size: int, tok, name: str, *, require_correct: bool, log=print) -> dict:
    items = json.load(open("data/evaluations/lens-eval-order-ops.json"))["items"]
    resolved = [(it, resolve_item(tok, it)) for it in items]
    variants = {f"L{PRIMARY['layer']}_k{PRIMARY['k']}": PRIMARY} if name == "null" else {f"L{c['layer']}_k{c['k']}": c for c in [PRIMARY, *VARIANTS]}
    out, excluded, t0 = {}, {}, time.time()
    dicts = {}
    for vk, cfg in variants.items():
        if cfg["layer"] not in dicts:
            dicts[cfg["layer"]] = build_dictionary_with_rows(model, lens, cfg["layer"], vocab_size)
        D, row_of, token_of = dicts[cfg["layer"]]
        rows = []
        for i, (it, res) in enumerate(resolved):
            if res is None:
                rows.append({"name": it["name"], "excluded": "no_single_token_synonym"})
                continue
            r = measure_prompt(model, lens, D, row_of, token_of, res, it["prompt"], cfg["layer"], cfg["k"], require_correct=require_correct)
            r["name"] = it["name"]
            rows.append(r)
            if (i + 1) % 10 == 0:
                log(f"  {name} {vk}: {i + 1}/{len(resolved)} items, {time.time() - t0:.0f}s")
        out[vk] = rows
        excluded[vk] = {k: sum(1 for r in rows if r.get("excluded") == k) for k in {r.get("excluded") for r in rows if "excluded" in r}}
    res = {"arm": name, "prereg_sha256": prereg_sha256(), "variants": out, "excluded_counts": excluded}
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(res, open(f"{OUT_DIR}/{name}.json", "w"))
    return res


if __name__ == "__main__":
    if "--analyze" in sys.argv:
        real, null = (json.load(open(f"{OUT_DIR}/{a}.json")) for a in ("real", "null"))
        out = analyze_h3(real, null)
        json.dump(out, open(f"{OUT_DIR}/analysis.json", "w"))
        print(json.dumps(out, indent=1))
        sys.exit(0)
    arm = sys.argv[sys.argv.index("--arm") + 1]
    import transformers

    from hlens.loading import MODEL_ID
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    if arm == "null":
        from hlens.harness import randomize_model
        from hlens.loading import load_hf
        from jlens import JacobianLens

        hf = load_hf(dtype=torch.float32)
        model = randomize_model(hf, tok, mode="reinit", seed=0)
        lens = JacobianLens.load("results/nulls/lens_randomized_reinit_s0_L16.pt")
        run_arm(model, lens, hf.config.get_text_config().vocab_size, tok, "null", require_correct=False)
    else:
        null_path = f"{OUT_DIR}/null.json"
        if not os.path.exists(null_path) or json.load(open(null_path))["prereg_sha256"] != prereg_sha256():
            sys.exit("refusing to run the real arm: the null arm must be run first under the SAME pre-registration hash")
        from hlens.loading import load_qwen

        q = load_qwen(dtype=torch.float32)
        run_arm(q.model, q.lens, q.hf.config.get_text_config().vocab_size, tok, "real", require_correct=True)

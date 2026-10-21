"""EXPLORATORY: which pair-level properties explain the H10 effect (D-bar) across the four scaled families?
Per hlens/docs/preregistration_exploratory_a_h10.md. Qwen3.5-4B fp32; forward passes only.

Run:  .venv/bin/python -m hlens.experiments.h10_covariates compute
      .venv/bin/python -m hlens.experiments.h10_covariates analyze
"""

from __future__ import annotations

import hashlib
import json
import sys

import numpy as np
import torch
from scipy import stats

from hlens.arch import embed_rows
from hlens.data import load_ignition
from hlens.experiments.h10_control import pair_dbar
from hlens.interpolation import forward_at_alpha, make_setup
from hlens.jspace import unembed_directions

PREREG = "hlens/docs/preregistration_exploratory_a_h10.md"
PAIRS, SCALED = "results/h10_scaled_pairs.json", "results/h10_scaled_qwen3.5-4b.json"
OUT = "results/h10_covariates_qwen3.5-4b.json"
CARRIERS, LAYERS = slice(12, 14), (8, 16, 24, 28)
FAMILIES = ("persona", "synonym", "control", "countries")
COVARIATES = ["d8", "d16", "d24", "d28", "js_output", "emb_cos", "unemb_cos", "slot_surprisal", "slot_logp_gap", "word_len"]


def prereg_sha256(path: str = PREREG) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def pair_covariates(q, pair, templates, U) -> dict:
    """Pair-level covariates, averaged over the carriers. d_L: ||h_A - h_B|| / mean norm at layer L ({W} position, pure-A vs pure-B).
    js_output: Jensen-Shannon divergence of the next-token distributions at the last position for pure A vs pure B.
    emb_cos / unemb_cos: cosine of the two words' input embeddings / unembedding directions. slot_surprisal: mean of -log p(word | text before the slot);
    slot_logp_gap: |log p(A) - log p(B)| there. word_len: mean number of characters."""
    d = {L: [] for L in LAYERS}
    js, surprisal, gap = [], [], []
    for t in templates:
        setup = make_setup(q, t, pair[0], pair[1])
        hb = forward_at_alpha(q, setup, 0.0, list(LAYERS))
        ha = forward_at_alpha(q, setup, 1.0, list(LAYERS))
        for L in LAYERS:
            a, b = ha[L][0, setup.position].float(), hb[L][0, setup.position].float()
            d[L].append(((a - b).norm() / (0.5 * (a.norm() + b.norm()))).item())
        ids_b = setup.input_ids
        ids_a = ids_b.clone()
        ids_a[0, setup.position] = setup.id_a
        with torch.no_grad():
            pa = torch.softmax(q.hf(input_ids=ids_a).logits[0, -1].float(), -1)
            pb = torch.softmax(q.hf(input_ids=ids_b).logits[0, -1].float(), -1)
            m = 0.5 * (pa + pb)
            kl = lambda p, r: (p * (torch.log(p + 1e-30) - torch.log(r + 1e-30))).sum().item()
            js.append(0.5 * kl(pa, m) + 0.5 * kl(pb, m))
            lp = torch.log_softmax(q.hf(input_ids=ids_b[:, : setup.position]).logits[0, -1].float(), -1)
        la, lb = lp[setup.id_a].item(), lp[setup.id_b].item()
        surprisal.append(-0.5 * (la + lb))
        gap.append(abs(la - lb))
    e = embed_rows(q.hf.get_input_embeddings(), [setup.id_a, setup.id_b]).float()
    cos = lambda x, y: float(torch.nn.functional.cosine_similarity(x, y, dim=0))
    return {"d8": float(np.mean(d[8])), "d16": float(np.mean(d[16])), "d24": float(np.mean(d[24])), "d28": float(np.mean(d[28])),
            "js_output": float(np.mean(js)), "emb_cos": cos(e[0], e[1]), "unemb_cos": cos(U[0], U[1]),
            "slot_surprisal": float(np.mean(surprisal)), "slot_logp_gap": float(np.mean(gap)), "word_len": 0.5 * (len(pair[0]) + len(pair[1]))}


def analyze(cov: dict, dbar: dict, *, n_boot: int = 2000, seed: int = 0) -> dict:
    """cov, dbar: family -> {"a|b": value}. Descriptive: Spearman with D-bar (overall, per family; Bonferroni threshold 0.01 / number of covariates),
    and a z-scored least-squares fit with family indicators (control = reference) with pair-bootstrap intervals, with and without the covariates."""
    keys = [(f, k) for f in FAMILIES for k in dbar[f]]
    y = np.array([dbar[f][k] for f, k in keys])
    X = {c: np.array([cov[f][k][c] for f, k in keys]) for c in COVARIATES}
    fam = np.array([f for f, _ in keys])
    thr = 0.01 / len(COVARIATES)
    out = {"n_pairs": len(keys), "bonferroni_threshold": thr, "spearman": {}}
    for c in COVARIATES:
        r, p = stats.spearmanr(X[c], y)
        within = {f: float(stats.spearmanr(X[c][fam == f], y[fam == f])[0]) for f in FAMILIES}
        out["spearman"][c] = {"overall": float(r), "p": float(p), "significant": bool(p < thr), "within_family": within}
    ind = np.column_stack([(fam == f).astype(float) for f in FAMILIES if f != "control"])
    Z = np.column_stack([(X[c] - X[c].mean()) / (X[c].std() + 1e-12) for c in COVARIATES])
    rng = np.random.default_rng(seed)

    def fit(design, idx):
        D = np.column_stack([np.ones(len(idx)), design[idx]])
        return np.linalg.lstsq(D, y[idx], rcond=None)[0]

    def r2(design):
        D = np.column_stack([np.ones(len(y)), design])
        b = np.linalg.lstsq(D, y, rcond=None)[0]
        return float(1 - ((y - D @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum())

    fams_only, with_cov = ind, np.column_stack([ind, Z])
    out["r2"] = {"family_only": r2(fams_only), "covariates_only": r2(Z), "family_and_covariates": r2(with_cov)}
    names_f = [f for f in FAMILIES if f != "control"]
    boot_f = np.array([fit(fams_only, rng.integers(0, len(y), len(y))) for _ in range(n_boot)])
    boot_c = np.array([fit(with_cov, rng.integers(0, len(y), len(y))) for _ in range(n_boot)])
    bf, bc = fit(fams_only, np.arange(len(y))), fit(with_cov, np.arange(len(y)))
    ci = lambda B, j: [float(np.percentile(B[:, j], 2.5)), float(np.percentile(B[:, j], 97.5))]
    out["family_coef_vs_control"] = {n: {"without_covariates": {"coef": float(bf[1 + i]), "ci95": ci(boot_f, 1 + i)},
                                          "with_covariates": {"coef": float(bc[1 + i]), "ci95": ci(boot_c, 1 + i)},
                                          "shrinkage": float(1 - abs(bc[1 + i]) / (abs(bf[1 + i]) + 1e-12))} for i, n in enumerate(names_f)}
    out["covariate_coef_z"] = {c: {"coef": float(bc[1 + len(names_f) + j]), "ci95": ci(boot_c, 1 + len(names_f) + j)} for j, c in enumerate(COVARIATES)}
    return out


if __name__ == "__main__":
    mode = sys.argv[1]
    P = json.load(open(PAIRS))
    if mode == "compute":
        from hlens.loading import load_qwen

        q = load_qwen(dtype=torch.float32)
        templates = load_ignition("data/experiments/ignition.json").noun_ctx_templates[CARRIERS]
        words = sorted({w for f in FAMILIES for p in P["pairs"][f] for w in p})
        ids = {w: q.tokenizer.encode(" " + w, add_special_tokens=False)[0] for w in words}
        with torch.enable_grad():
            Uall = unembed_directions(q.model, [ids[w] for w in words], dim_batch=256)
        U = {w: Uall[i] for i, w in enumerate(words)}
        cov = {}
        for f in FAMILIES:
            cov[f] = {}
            for a, b in P["pairs"][f]:
                cov[f][f"{a}|{b}"] = pair_covariates(q, (a, b), templates, torch.stack([U[a], U[b]]))
            print(f, len(cov[f]), flush=True)
        json.dump({"prereg_sha256": prereg_sha256(), "layers": LAYERS, "covariates": cov}, open(OUT, "w"))
    else:
        res = json.load(open(OUT))
        scaled = json.load(open(SCALED))
        dbar = {f: {f"{a}|{b}": v for (a, b), v in pair_dbar(scaled["families"][f]["trials"]).items()} for f in FAMILIES}
        out = analyze(res["covariates"], dbar)
        out["prereg_sha256"] = {"recorded": res["prereg_sha256"], "current": prereg_sha256()}
        res["analysis"] = out
        json.dump(res, open(OUT, "w"))
        print(json.dumps(out, indent=1))

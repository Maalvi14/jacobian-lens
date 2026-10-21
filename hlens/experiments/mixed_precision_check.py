"""Validate the mixed-precision path (bf16 model, fp32 slice / directions) against the full-fp32 reference, per
hlens/docs/preregistration_mixed_precision.md (criteria V1-V5).

Run:  uv run --no-sync python -m hlens.experiments.mixed_precision_check
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import torch
from scipy import stats

from hlens.data import load_ignition
from hlens.experiments.competition import OFFSETS, analyze_h9, run
from hlens.experiments.h9_replication import trial_sets
from hlens.functional import Fp32Unembed, make_mixed_slice_fn
from hlens.loading import load_model, matched_layer

PREREG = "hlens/docs/preregistration_mixed_precision.md"
REFERENCE = "results/h9_replication/O_qwen3.5-4b_clean.json"
OUT = "results/mixed_precision_check_qwen3.5-4b.json"


def prereg_sha256(path: str = PREREG) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def compare(ref_rows: list[dict], mix_rows: list[dict]) -> dict:
    """V1-V5 over the cells of trials present in both (matched on template and word pair)."""
    key = lambda r: (r["template"], r["A"], r["B"])
    mix = {key(r): r for r in mix_rows}
    pairs = [(r, mix[key(r)]) for r in ref_rows if key(r) in mix]
    cells = [(r["cells"][str(o)], m["cells"][str(o)]) for r, m in pairs for o in OFFSETS]
    I_ref, I_mix = np.array([a["I"] for a, _ in cells]), np.array([b["I"] for _, b in cells])
    rel = np.abs(I_mix - I_ref) / np.abs(I_ref)
    sign = lambda f: float(np.mean([np.sign(a[f]) == np.sign(b[f]) for a, b in cells]))
    d_star = np.array([abs(m["alpha_star_own"] - r["alpha_star"]) for r, m in pairs])
    ref_an, mix_an = analyze_h9([r for r, _ in pairs]), analyze_h9([m for _, m in pairs])
    out = {
        "n_trials": len(pairs), "n_cells": len(cells),
        "V1_sign_agreement": {"cA": sign("cA"), "cB": sign("cB"), "passed": bool(sign("cA") >= 0.95 and sign("cB") >= 0.95)},
        "V2_spearman_I": {"rho": float(stats.spearmanr(I_ref, I_mix)[0]), "passed": bool(stats.spearmanr(I_ref, I_mix)[0] >= 0.95)},
        "V3_relative_error": {"median": float(np.median(rel)), "p90": float(np.percentile(rel, 90)),
                              "passed": bool(np.median(rel) <= 0.10 and np.percentile(rel, 90) <= 0.30)},
        "V4_verdicts": {"fp32": [ref_an[k]["passed"] for k in ("P0a", "P0b", "P0c")], "mixed": [mix_an[k]["passed"] for k in ("P0a", "P0b", "P0c")],
                        "passed": bool([ref_an[k]["passed"] for k in ("P0a", "P0b", "P0c")] == [mix_an[k]["passed"] for k in ("P0a", "P0b", "P0c")])},
        "V5_threshold": {"fraction_within_0.05": float(np.mean(d_star <= 0.05)), "median_abs_diff": float(np.median(d_star)),
                         "passed": bool(np.mean(d_star <= 0.05) >= 0.90)},
        "median_I": {"fp32": float(np.median(I_ref)), "mixed": float(np.median(I_mix))},
    }
    out["all_passed"] = bool(all(out[k]["passed"] for k in out if k.startswith("V")))
    return out


if __name__ == "__main__":
    ref = json.load(open(REFERENCE))["trials"]
    q = load_model("qwen3.5-4b", dtype=torch.bfloat16)
    n = q.model.n_layers
    src, tgt, mid = (matched_layer(n, l) for l in (12, 20, 16))
    trials = trial_sets(q.tokenizer, load_ignition("data/experiments/ignition.json"))["O"]
    ref_by_key = {(r["template"], r["A"], r["B"]): r["alpha_star"] for r in ref}
    stars = [ref_by_key.get(t) for t in trials]  # None -> the reference skipped this trial
    rows = run(q, trials=trials, src=src, tgt=tgt, mid=mid, clean=True, slice_factory=make_mixed_slice_fn,
               unembed_model=Fp32Unembed(q.model), alpha_stars=stars)
    res = compare(ref, rows)
    with open(OUT, "w") as f:
        json.dump({"prereg_sha256": prereg_sha256(), "reference": REFERENCE, "comparison": res, "trials": rows}, f)
    print(json.dumps(res, indent=1))

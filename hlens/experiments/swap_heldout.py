"""Held-out swap suite (hlens/swap_suite.py), per hlens/docs/preregistration_swap_heldout.md.
bf16 (the lens was fitted in bf16); band and alpha were fixed before this suite existed.

Run:  .venv/bin/python -m hlens.experiments.swap_heldout --model qwen3.5-4b
      .venv/bin/python -m hlens.experiments.swap_heldout --model gemma-3-12b
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math

import torch

from hlens.experiments.gates import run_g2
from hlens.loading import load_model

PREREG = "hlens/docs/preregistration_swap_heldout.md"
SUITE = "data/experiments/probe-swap-heldout.json"
ALPHAS = (1.25, 1.0)  # 1.25 primary (pre-registered for Gemma in preregistration_replication.md; tuned on the dev suite for Qwen), 1.0 secondary


def sha256(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def summarize(g2: dict) -> dict:
    N = g2["n_baseline_correct"]
    out = {}
    for a, v in g2["by_alpha"].items():
        cats = collections.defaultdict(lambda: [0, 0])
        for r in v["per_item"]:
            cats[r["category"]][0] += r["hit"]
            cats[r["category"]][1] += 1
        lo, hi = wilson(v["hits"], N)
        ctrl = max(max(v["random_control_hits"]), 1)
        out[a] = {"hits": v["hits"], "n": N, "rate": v["rate"], "wilson95": [lo, hi], "random_control_hits": v["random_control_hits"],
                  "passed": bool(N >= 30 and v["rate"] >= 0.25 and v["hits"] >= 5 * ctrl),
                  "by_category": {c: f"{k}/{n}" for c, (k, n) in sorted(cats.items())}}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fold-case", action="store_true", help="case-insensitive first-token scoring (preregistration_heldout_scoring_h1_e1.md)")
    a = ap.parse_args()
    q = load_model(a.model, dtype=torch.bfloat16)
    g2 = run_g2(q, alphas=ALPHAS, path=SUITE, fold_case=a.fold_case)
    res = {"model": a.model, "fold_case": a.fold_case, "prereg_sha256": sha256("hlens/docs/preregistration_heldout_scoring_h1_e1.md" if a.fold_case else PREREG), "suite_sha256": sha256(SUITE), "band": g2["band"], "n_items": g2["n_items"],
           "n_baseline_correct": g2["n_baseline_correct"], "summary": summarize(g2), "g2": g2}
    with open(f"results/swap_heldout{'_foldcase' if a.fold_case else ''}_{a.model}.json", "w") as f:
        json.dump(res, f)
    print(json.dumps(res["summary"], indent=1))

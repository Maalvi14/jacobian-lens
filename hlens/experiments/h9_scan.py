"""EXPLORATORY scan for claim A (H9 precondition): does the effect exist at other layer pairs, or with lens-free directions?
Per hlens/docs/preregistration_exploratory_a_h10.md. Original 24 trials; every config is checkpointed and resumable.

Run:  .venv/bin/python -m hlens.experiments.h9_scan --model qwen3.5-4b      # positive control
      .venv/bin/python -m hlens.experiments.h9_scan --model gemma-3-12b
      .venv/bin/python -m hlens.experiments.h9_scan --model gemma-3-12b --summary
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np
import torch

from hlens.data import load_ignition
from hlens.experiments.competition import analyze_h9, run
from hlens.experiments.h9_replication import trial_sets
from hlens.loading import load_model

PREREG = "hlens/docs/preregistration_exploratory_a_h10.md"
OUT_DIR = "results/h9_scan"
GRID = {"qwen3.5-4b": {"srcs": [8, 12, 16], "gaps": [4, 8, 12], "mid": 16, "clean": False},
        "gemma-3-12b": {"srcs": [12, 18, 24], "gaps": [6, 12, 18], "mid": 24, "clean": True}}
DIRECTIONS = ("lens", "identity")


def prereg_sha256(path: str = PREREG) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def configs(model_key: str) -> list[tuple[str, int, int]]:
    g = GRID[model_key]
    return [(d, s, s + gap) for d in DIRECTIONS for s in g["srcs"] for gap in g["gaps"]]


def summarize_config(rows: list[dict], n_configs: int) -> dict:
    """The H9 precondition statistics of one config plus a Bonferroni flag (each of the three tests at p < 0.01 / n_configs)."""
    a = analyze_h9(rows)
    thr = 0.01 / n_configs
    return {"n": a["n_trials"], "median_I": a["P0a"]["median_I_at_threshold"], "p": {k: a[k]["p"] for k in ("P0a", "P0b", "P0c")},
            "median_threshold_minus_flank": a["P0b"]["median_I_threshold_minus_flank"],
            "frac_both_negative": a["exploratory"]["frac_both_cross_terms_negative_at_threshold"],
            "median_first_order_cross": a["exploratory"]["median_first_order_cross"],
            "uncorrected_pass": bool(a["precondition_met"]), "bonferroni_pass": bool(all(a[k]["p"] < thr for k in ("P0a", "P0b", "P0c"))),
            "bonferroni_threshold": thr}


def load_summary(model_key: str) -> dict:
    path = f"{OUT_DIR}/{model_key}.json"
    return json.load(open(path)) if os.path.exists(path) else {}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(GRID))
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    cfgs = configs(a.model)
    if a.summary:
        for k, v in load_summary(a.model).items():
            print(f"{k:22s} I={v['median_I']:+.4f} p={['%.2g' % x for x in v['p'].values()]} bothneg={v['frac_both_negative']:.2f} FO={v['median_first_order_cross']:+.3f} uncorr={v['uncorrected_pass']} bonf={v['bonferroni_pass']}")
        raise SystemExit(0)
    g = GRID[a.model]
    q = load_model(a.model, dtype=torch.float32)
    trials = trial_sets(q.tokenizer, load_ignition("data/experiments/ignition.json"))["O"]
    os.makedirs(f"{OUT_DIR}/{a.model}", exist_ok=True)
    summary = load_summary(a.model)
    for d, src, tgt in cfgs:
        key = f"{d}_{src}_{tgt}"
        if key in summary:
            continue
        rows = run(q, trials=trials, src=src, tgt=tgt, mid=g["mid"], clean=g["clean"], directions=d,
                   checkpoint=f"{OUT_DIR}/{a.model}/{key}.partial.jsonl", log=lambda *_: None)
        summary[key] = summarize_config(rows, len(cfgs))
        summary[key]["layers"] = {"src": src, "tgt": tgt, "mid": g["mid"], "directions": d}
        json.dump(summary, open(f"{OUT_DIR}/{a.model}.json", "w"))
        v = summary[key]
        print(f"{key}: n={v['n']} I={v['median_I']:+.4f} p={['%.2g' % x for x in v['p'].values()]} bothneg={v['frac_both_negative']:.2f} bonf={v['bonferroni_pass']}", flush=True)
    json.dump({"prereg_sha256": prereg_sha256(), "model": a.model, "summary": summary}, open(f"{OUT_DIR}/{a.model}_final.json", "w"))

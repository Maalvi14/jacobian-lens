"""H6 (ignition as a curvature event) on a second model, per hlens/docs/preregistration_h6_h8_gemma.md.
Depth-matched layers; every trial is checkpointed and a rerun resumes.

Run:  .venv/bin/python -m hlens.experiments.h6_replication --model gemma-3-12b --arm null
      .venv/bin/python -m hlens.experiments.h6_replication --model gemma-3-12b --arm real
      .venv/bin/python -m hlens.experiments.h6_replication --model gemma-3-12b --arm analyze
      .venv/bin/python -m hlens.experiments.h6_replication --model gemma-3-12b --arm knee
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace

import torch
import transformers

from hlens.experiments.ignition import LAYERS as QWEN_LAYERS, analyze_h6, knee_vs_radius, prereg_sha256, run_arm
from hlens.harness import randomize_model
from hlens.loading import MODELS, load_hf, load_model, matched_layer

PREREG = "hlens/docs/preregistration_h6_h8_gemma.md"


def params(n_layers: int) -> dict:
    """Depth-matched H6 parameters (Qwen3.5-4B: layers {4,...,24}, gap 4, threshold layer 16)."""
    return {"layers": [matched_layer(n_layers, l) for l in QWEN_LAYERS], "gap": round(4 * (n_layers - 1) / 31), "mid": matched_layer(n_layers, 16)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-3-12b")
    ap.add_argument("--arm", required=True, choices=["null", "real", "analyze", "knee"])
    a = ap.parse_args()
    out_dir = f"results/h6_{a.model}"
    spec = MODELS[a.model]
    n = transformers.AutoConfig.from_pretrained(spec.model_id).get_text_config().num_hidden_layers
    P = params(n)
    kw = {**P, "clean": True, "out_dir": out_dir, "prereg": PREREG}
    if a.arm == "analyze":
        real, null = (json.load(open(f"{out_dir}/{x}.json")) for x in ("real", "null"))
        out = {k: analyze_h6(real, null, key=k, prereg=PREREG) for k in ("lambda", "tau")}
        json.dump(out, open(f"{out_dir}/analysis.json", "w"))
        print(json.dumps(out, indent=1))
    elif a.arm == "null":
        tok = transformers.AutoTokenizer.from_pretrained(spec.model_id)
        hf = load_hf(dtype=torch.float32, model_id=spec.model_id)
        model = randomize_model(hf, tok, mode="reinit", seed=0)
        run_arm(SimpleNamespace(model=model, hf=hf, tokenizer=tok), "null", checkpoint=f"{out_dir}/null.partial.jsonl", **kw)
    else:
        nf = f"{out_dir}/null.json"
        if not os.path.exists(nf) or json.load(open(nf))["prereg_sha256"] != prereg_sha256(PREREG):
            sys.exit("refusing to run the real arm: the null arm must be run first under the SAME pre-registration hash")
        q = load_model(a.model, dtype=torch.float32)
        if a.arm == "real":
            run_arm(q, "real", checkpoint=f"{out_dir}/real.partial.jsonl", **kw)
        else:
            lay = tuple(matched_layer(n, l) for l in (4, 12, 16, 20, 24))
            res = knee_vs_radius(q, layers=lay, gap=P["gap"], mid=P["mid"], clean=True)
            os.makedirs(out_dir, exist_ok=True)
            json.dump({"prereg_sha256": prereg_sha256(PREREG), "layers": list(lay), "cells": res}, open(f"{out_dir}/knee_vs_radius.json", "w"))
            print(f"{len(res)} cells; fraction with validity score < 1: {sum(c['score'] is not None and c['score'] < 1 for c in res) / len(res):.2f}")

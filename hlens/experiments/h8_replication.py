"""H8 (motor-regime collapse) on a second model, per hlens/docs/preregistration_h6_h8_gemma.md.

Run:  .venv/bin/python -m hlens.experiments.h8_replication --model gemma-3-12b
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import transformers

from hlens.experiments.motor_collapse import analyze_h8, config_for, distance_to_scaled_identity, prereg_sha256, run_arm
from hlens.harness import randomize_model
from hlens.loading import MODELS, load_hf, load_model
from jlens.examples import load_wikitext_prompts

PREREG = "hlens/docs/preregistration_h6_h8_gemma.md"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-3-12b")
    a = ap.parse_args()
    spec = MODELS[a.model]
    out = f"results/h8_motor_collapse_{a.model}.json"
    prompts = load_wikitext_prompts(330)[300:]
    q = load_model(a.model, dtype=torch.float32)
    cfg = config_for(q.model.n_layers, q.lens.source_layers[-1])
    d_I = {str(l): distance_to_scaled_identity(q.lens.jacobians[l]) for l in q.lens.source_layers}
    print("config", cfg, flush=True)
    real = run_arm(q.model, prompts, layers=cfg["layers"])
    tok = transformers.AutoTokenizer.from_pretrained(spec.model_id)
    del q
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    null = run_arm(randomize_model(load_hf(dtype=torch.float32, model_id=spec.model_id), tok, mode="reinit", seed=0), prompts, layers=cfg["layers"])
    d = {"prereg_sha256": prereg_sha256(PREREG), "config": cfg, "d_I": d_I, "real": real, "null": null}
    os.makedirs("results", exist_ok=True)
    json.dump(d, open(out, "w"))
    d["analysis"] = analyze_h8(d)
    json.dump(d, open(out, "w"))
    print(json.dumps(d["analysis"], indent=1))

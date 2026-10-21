"""H9 precondition replication (claim A) per hlens/docs/preregistration_replication.md:
original (O) and held-out (HO) trial sets on a model, depth-matched layers, validity sensitivity.

Run:  uv run python -m hlens.experiments.h9_replication --model qwen3.5-4b --set HO
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random

import numpy as np
import torch

from hlens.data import load_ignition
from hlens.experiments.competition import analyze_h9, run
from hlens.loading import load_model, matched_layer

PREREG = "hlens/docs/preregistration_replication.md"
PREREG_27B = "hlens/docs/preregistration_27b_claim_a.md"
OUT_DIR = "results/h9_replication"


def prereg_sha256(path: str = PREREG) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


MIXED_MODELS = {"qwen3.6-27b"}  # fp32 weights do not fit: bf16 model + fp32 slice (preregistration_mixed_precision.md)


def trial_sets(tok, cfg) -> dict[str, list[tuple[str, str, str]]]:
    """O = choose_trials (templates[:6] x shuffled pairs[:4]); HO = templates[6:36] x shuffled pairs[4:8]."""
    single = [c for c in cfg.countries_12 if len(tok.encode(" " + c, add_special_tokens=False)) == 1]
    pairs = list(itertools.combinations(single, 2))
    random.Random(0).shuffle(pairs)
    return {
        "O": [(t, a, b) for t in cfg.ctx_templates[:6] for a, b in pairs[:4]],
        "HO": [(t, a, b) for t in cfg.ctx_templates[6:36] for a, b in pairs[4:8]],
    }


def analyze(rows: list[dict]) -> dict:
    out = {"all": analyze_h9(rows)}
    # a validity score of None (no finite plateau found) counts as not valid
    valid = [r for r in rows if (r.get("validity") or {}).get("score") is not None and r["validity"]["score"] >= 1]
    out["valid_only"] = analyze_h9(valid) if len(valid) >= 8 else {"n_trials": len(valid), "note": "fewer than 8 valid trials"}
    out["n_valid"] = len(valid)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-4b")
    ap.add_argument("--set", default="HO", choices=["O", "HO"])
    ap.add_argument("--clean", action="store_true", help="single space before the slot (required for Gemma)")
    ap.add_argument("--from-checkpoint", action="store_true", help="analyze the .partial.jsonl without loading the model (e.g. if a run died after its last trial)")
    a = ap.parse_args()
    mixed = a.model in MIXED_MODELS
    suffix = f"{'_clean' if a.clean else ''}{'_mixed' if mixed else ''}"
    if a.from_checkpoint:
        import transformers

        from hlens.loading import MODELS

        n = transformers.AutoConfig.from_pretrained(MODELS[a.model].model_id).get_text_config().num_hidden_layers
        src, tgt, mid = (matched_layer(n, l) for l in (12, 20, 16))
        rows = [json.loads(line) for line in open(f"{OUT_DIR}/{a.set}_{a.model}{suffix}.partial.jsonl")]
        res = analyze(rows)
        with open(f"{OUT_DIR}/{a.set}_{a.model}{suffix}.json", "w") as f:
            json.dump({"model": a.model, "set": a.set, "clean_spacing": a.clean, "layers": {"src": src, "tgt": tgt, "mid": mid},
                       "prereg_sha256": prereg_sha256(PREREG_27B if mixed else PREREG), "mixed_precision": mixed,
                       "source": "checkpoint", "analysis": res, "trials": rows}, f)
        print(json.dumps(res, indent=1))
        raise SystemExit(0)
    if mixed and torch.cuda.is_available():
        # fail loudly (a CUDA out-of-memory traceback) before the shared CPU/GPU pool is exhausted; an earlier run died silently
        torch.cuda.set_per_process_memory_fraction(0.85)
    q = load_model(a.model, dtype=torch.bfloat16 if mixed else torch.float32)
    n = q.model.n_layers
    src, tgt, mid = (matched_layer(n, l) for l in (12, 20, 16))
    trials = trial_sets(q.tokenizer, load_ignition("data/experiments/ignition.json"))[a.set]
    print(f"{a.model}{' (mixed precision)' if mixed else ''}: {len(trials)} trials, src {src} tgt {tgt} mid {mid}", flush=True)
    extra = {}
    if mixed:
        from hlens.functional import Fp32Unembed, make_mixed_slice_fn

        extra = {"slice_factory": make_mixed_slice_fn, "unembed_model": Fp32Unembed(q.model)}
    ckpt = f"{OUT_DIR}/{a.set}_{a.model}{'_clean' if a.clean else ''}{'_mixed' if mixed else ''}.partial.jsonl"
    rows = run(q, trials=trials, src=src, tgt=tgt, mid=mid, validity=True, clean=a.clean, checkpoint=ckpt, **extra)
    res = analyze(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/{a.set}_{a.model}{'_clean' if a.clean else ''}{'_mixed' if mixed else ''}.json", "w") as f:
        json.dump({"model": a.model, "set": a.set, "clean_spacing": a.clean, "layers": {"src": src, "tgt": tgt, "mid": mid}, "prereg_sha256": prereg_sha256(PREREG_27B if mixed else PREREG), "mixed_precision": mixed,
                   "analysis": res, "trials": rows}, f)
    print(json.dumps(res, indent=1))

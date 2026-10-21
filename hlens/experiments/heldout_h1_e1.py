"""H1 (trust score) and E1 (engagement) on the held-out swap suite with case-insensitive scoring, per
hlens/docs/preregistration_heldout_scoring_h1_e1.md. Qwen3.5-4B fp32.

Run:  .venv/bin/python -m hlens.experiments.heldout_h1_e1 h1        # swap labels + tau (also needs the randomized model for the null column)
      .venv/bin/python -m hlens.experiments.heldout_h1_e1 e1        # margin paths on the baseline-correct items
      .venv/bin/python -m hlens.experiments.heldout_h1_e1 analyze
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

import torch

from hlens.experiments import engagement, trust_score
from hlens.swaps import DEFAULT_ALPHA, probe_items

PREREG = "hlens/docs/preregistration_heldout_scoring_h1_e1.md"
SUITE = "data/experiments/probe-swap-heldout.json"
H1_OUT = "results/trust_score_heldout_qwen3.5-4b.json"
E1_OUT = "results/e1_engagement_heldout_qwen3.5-4b.json"
NULL_LENS = "results/nulls/lens_randomized_reinit_s0_L16.pt"


def prereg_sha256(path: str = PREREG) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _null():
    import transformers

    from hlens.harness import randomize_model
    from hlens.loading import MODEL_ID, load_hf
    from jlens import JacobianLens

    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    nmodel = randomize_model(load_hf(dtype=torch.float32), tok, mode="reinit", seed=0)

    class _Null:
        model = nmodel
        lens = JacobianLens.load(NULL_LENS)

    return _Null


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "h1":
        from hlens.loading import load_qwen

        q32 = load_qwen(dtype=torch.float32)
        items = probe_items(q32.tokenizer, SUITE, fold_case=True)
        rows = trust_score.run_trust_score(q32, null=_null(), items=items, fold_case=True)
        analyses = {f"hit_a{a:g}": trust_score.analyze(rows, alpha=a, label="hit") for a in (1.0, DEFAULT_ALPHA)}
        trust_score.save_results(rows, analyses, H1_OUT)
        d = json.load(open(H1_OUT))
        d["prereg_sha256"] = prereg_sha256()
        json.dump(d, open(H1_OUT, "w"))
        print(json.dumps(analyses, indent=1))
    elif mode == "e1":
        from hlens.loading import load_qwen

        saved = json.load(open(H1_OUT))
        q32 = load_qwen(dtype=torch.float32)
        items = engagement.run(q32, _null(), saved["rows"], items_path=SUITE, fold_case=True)
        os.makedirs("results", exist_ok=True)
        json.dump({"prereg_sha256": prereg_sha256(), "alphas": engagement.ALPHAS, "items": items}, open(E1_OUT, "w"))
    else:
        d = json.load(open(E1_OUT))
        res = engagement.analyze_e1(d["items"])
        res["prereg_sha256"] = {"recorded": d["prereg_sha256"], "current": prereg_sha256()}
        d["analysis"] = res
        json.dump(d, open(E1_OUT, "w"))
        print(json.dumps(res, indent=1))

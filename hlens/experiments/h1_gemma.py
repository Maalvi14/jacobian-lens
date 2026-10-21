"""H1 (trust score) on Gemma 3 12B, held-out swap suite, per hlens/docs/preregistration_h1_gemma.md. fp32; HVPs through layers 18, 24, 30.

Run:  .venv/bin/python -m hlens.experiments.h1_gemma run        # per-item checkpointed (resumable)
      .venv/bin/python -m hlens.experiments.h1_gemma analyze
"""

from __future__ import annotations

import hashlib
import json
import sys

import torch

from hlens.experiments import trust_score
from hlens.loading import load_model
from hlens.swaps import probe_items

PREREG = "hlens/docs/preregistration_h1_gemma.md"
SUITE = "data/experiments/probe-swap-heldout.json"
OUT = "results/trust_score_heldout_gemma-3-12b.json"
CKPT = "results/trust_score_heldout_gemma-3-12b.partial.jsonl"
LAYERS, BAND, MID = (18, 24, 30), (18, 30), 24
ALPHAS = (1.0, 1.25)  # 1.0 primary (Gemma's working strength); 1.25 is reported but is expected to be untestable (about 4 % successes)


def prereg_sha256(path: str = PREREG) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def analyses(rows: list[dict]) -> dict:
    return {f"hit_a{a:g}": trust_score.analyze(rows, alpha=a, label="hit", layers=LAYERS, mid_layer=MID) for a in ALPHAS}


if __name__ == "__main__":
    if sys.argv[1] == "run":
        q = load_model("gemma-3-12b", dtype=torch.float32)
        items = probe_items(q.tokenizer, SUITE, fold_case=True)
        rows = trust_score.run_trust_score(q, items=items, layers=LAYERS, band=BAND, alphas=ALPHAS, fold_case=True, checkpoint=CKPT, log_every=5)
        an = analyses(rows)
        trust_score.save_results(rows, an, OUT)
        d = json.load(open(OUT))
        d["prereg_sha256"] = prereg_sha256()
        d["layers"], d["band"], d["mid_layer"] = list(LAYERS), list(BAND), MID
        json.dump(d, open(OUT, "w"))
        print(json.dumps(an, indent=1))
    else:
        d = json.load(open(OUT))
        an = analyses(d["rows"])
        d["analysis"] = an
        json.dump(d, open(OUT, "w"))
        print(json.dumps(an, indent=1))

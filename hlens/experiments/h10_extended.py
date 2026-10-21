"""H10 extended (more pairs, fresh pairs and carriers), per hlens/docs/preregistration_h10_extended.md.

Run:  .venv/bin/python -m hlens.experiments.h10_extended            # run (checkpointed, resumable)
      .venv/bin/python -m hlens.experiments.h10_extended --analyze  # analysis from the result file / checkpoint
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
from scipy import stats

from hlens.experiments.register_persona import LAYERS, analyze_h10, family_signature, run

PREREG = "hlens/docs/preregistration_h10_extended.md"
OUT = "results/h10_extended_qwen3.5-4b.json"
CKPT = "results/h10_extended_qwen3.5-4b.partial.jsonl"
REFERENCE = "results/h6/real.json"
N_PAIRS, CARRIERS = 10, slice(6, 12)
FAMILIES = {
    "register": [("attorney", "lawyer"), ("infant", "baby"), ("footwear", "shoes"), ("employment", "job"), ("spectacles", "glasses"),
                 ("gentleman", "guy"), ("youngster", "kid"), ("telephone", "phone"), ("currency", "money"), ("canine", "dog"),
                 ("stomach", "belly"), ("purchase", "buy")],
    "persona": [("judge", "clown"), ("monk", "sailor"), ("queen", "witch"), ("chef", "knight"), ("scientist", "ghost"),
                ("princess", "vampire"), ("mayor", "dragon"), ("banker", "hunter"), ("surgeon", "outlaw"), ("professor", "giant"),
                ("banker", "outlaw"), ("mayor", "witch")],
}


def prereg_sha256(path: str = PREREG) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def analyze(res: dict, countries: dict) -> dict:
    """The original three per-family criteria plus the pre-registered persona-vs-register contrast (one-sided Mann-Whitney on D)."""
    out = analyze_h10(res, countries)
    D = {f: family_signature(v["trials"], v["n_planned"])["_D"] for f, v in res.items()}
    contrast = {}
    for l in (20, 24):
        p = float(stats.mannwhitneyu(D["persona"][str(l)], D["register"][str(l)], alternative="greater").pvalue)
        contrast[str(l)] = {"p_persona_gt_register": p, "median_D_persona": float(np.median(D["persona"][str(l)])), "median_D_register": float(np.median(D["register"][str(l)]))}
    out["contrast"] = contrast
    out["dissociation_replicates"] = bool(out["persona"]["reproduces_signature"] and not out["register"]["reproduces_signature"]
                                           and all(c["p_persona_gt_register"] < 0.01 for c in contrast.values()))
    return out


if __name__ == "__main__":
    if "--analyze" in sys.argv:
        d = json.load(open(OUT))
        d["analysis"] = analyze(d["families"], json.load(open(REFERENCE)))
        d["analysis"]["prereg_sha256"] = {"recorded": d["prereg_sha256"], "current": prereg_sha256()}
        json.dump(d, open(OUT, "w"))
        print(json.dumps(d["analysis"], indent=1))
        sys.exit(0)
    import torch

    from hlens.loading import load_qwen

    fam = run(load_qwen(dtype=torch.float32), families=FAMILIES, carriers=CARRIERS, n_pairs=N_PAIRS, checkpoint=CKPT)
    os.makedirs("results", exist_ok=True)
    json.dump({"prereg_sha256": prereg_sha256(), "layers": LAYERS, "families": fam}, open(OUT, "w"))

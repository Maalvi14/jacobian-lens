"""H2 replication (claim B) per hlens/docs/preregistration_replication.md: real arm on a second
model, and weight-randomized null arms (seeds / layers) for any model.

The null is built so that one set of weights serves both halves of the pipeline without two
resident copies: randomize in fp32 on the CPU, round the weights to bf16 precision (exact in
both dtypes), fit the lens on the bf16 model, then upcast to fp32 for the curvature measurement.

Run:  uv run python -m hlens.experiments.h2_replication null --model gemma-3-12b --seed 0 --layers 24
      uv run python -m hlens.experiments.h2_replication real --model gemma-3-12b
      uv run python -m hlens.experiments.h2_replication analyze --model gemma-3-12b
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import transformers

from hlens.experiments.gain import run_h2
from hlens.harness import fit_null_lens, randomize_model
from hlens.loading import MODELS, load_hf, load_model
from jlens import JacobianLens
from jlens.examples import load_wikitext_prompts

PREREG = "hlens/docs/preregistration_replication.md"
OUT_DIR = "results/h2_replication"
REAL_LAYERS = {"gemma-3-12b": [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44]}
N_EVAL, N_FIT = 30, 30

# "A": the replication pre-registration; "B2": the B' pre-registration (fresh prompts, new seeds)
CONFIGS = {
    "A": {"prereg": PREREG, "prefix": "", "eval_start": 0, "fit_start": 100, "real_layers": REAL_LAYERS},
    "B2": {"prereg": "hlens/docs/preregistration_b_prime.md", "prefix": "B2_", "eval_start": 200, "fit_start": 300,
           "real_layers": {"gemma-3-12b": [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44], "qwen3.5-4b": [8, 16, 18, 20, 22, 24, 26, 28, 30]}},
    # B' on a third model (Llama 3.1 8B, 32 layers: the Qwen3.5-4B layer indices are already depth-matched)
    "B3": {"prereg": "hlens/docs/preregistration_b_prime_llama.md", "prefix": "B3_", "eval_start": 200, "fit_start": 300,
           "real_layers": {"llama3.1-8b": [8, 16, 18, 20, 22, 24, 26, 28, 30]}},
}
CFG = CONFIGS["A"]


def prereg_sha256(path: str | None = None) -> str:
    return hashlib.sha256(open(path or CFG["prereg"], "rb").read()).hexdigest()


def _prompts(start: int, n: int) -> list[str]:
    return load_wikitext_prompts(start + n)[start:]


def null_arm(key: str, seed: int, layers: list[int]) -> str:
    tag = f"{key}_s{seed}_L{'-'.join(map(str, layers))}"
    out = f"{OUT_DIR}/{CFG['prefix']}null_{tag}.json"
    if os.path.exists(out):
        return out
    os.makedirs(OUT_DIR, exist_ok=True)
    spec = MODELS[key]
    tok = transformers.AutoTokenizer.from_pretrained(spec.model_id)
    hf = load_hf(dtype=torch.float32, device="cpu", model_id=spec.model_id)
    model = randomize_model(hf, tok, mode="reinit", seed=seed)
    with torch.no_grad():
        for p in hf.parameters():
            p.data = p.data.to(torch.bfloat16).to(torch.float32)  # round to bf16 precision (kept in fp32)
    hf.to(torch.bfloat16).to("cuda")
    lens_path = f"{OUT_DIR}/{CFG['prefix']}lens_{tag}.pt"
    fit_prompts = _prompts(CFG["fit_start"], N_FIT)
    fit_null_lens(model, fit_prompts, source_layers=layers, dim_batch=32, max_seq_len=64, path=lens_path)
    lens = JacobianLens.load(lens_path)
    hf.to(torch.float32)  # exact: the weights are bf16-representable
    torch.cuda.empty_cache()
    res = run_h2(SimpleNamespace(model=model, lens=lens, tokenizer=tok), _prompts(CFG["eval_start"], N_EVAL), layers=layers)
    with open(out, "w") as f:
        json.dump({"arm": "null", "model": key, "seed": seed, "layers": layers, "prereg_sha256": prereg_sha256(), "result": res}, f)
    return out


def real_arm(key: str) -> str:
    nulls = glob.glob(f"{OUT_DIR}/{CFG['prefix']}null_{key}_*.json")
    if not any(json.load(open(p))["prereg_sha256"] == prereg_sha256() for p in nulls):
        sys.exit("refusing to run the real arm: no null arm under the SAME pre-registration hash")
    out = f"{OUT_DIR}/{CFG['prefix']}real_{key}.json"
    layers = CFG["real_layers"][key]
    q = load_model(key, dtype=torch.float32)
    res = run_h2(q, _prompts(CFG["eval_start"], N_EVAL), layers=layers)
    with open(out, "w") as f:
        json.dump({"arm": "real", "model": key, "layers": layers, "prereg_sha256": prereg_sha256(), "result": res}, f)
    return out


def ratio(layer_res: dict) -> dict:
    return layer_res["ratio_to_random"]["j_lens"]["kappa"]


def analyze(real: dict | None, nulls: list[dict], n_layers: int) -> dict:
    """T1/T2 from a {layer: summary} dict of a real arm (depth >= 0.5); T3 over null arms."""
    out = {}
    if real is not None:
        rng = {int(l): v for l, v in real.items() if int(l) / (n_layers - 1) >= 0.5}
        lo = [ratio(v)["ci95"][0] for v in rng.values()]
        med = [ratio(v)["median"] for v in rng.values()]
        t1_frac = float(np.mean([x > 1 for x in lo]))
        out["real"] = {"layers_in_range": sorted(rng), "ratio_median_by_layer": dict(zip(sorted(rng), [ratio(rng[l])["median"] for l in sorted(rng)])),
                       "ci_lower_gt_1_fraction": t1_frac, "median_of_medians": float(np.median(med)),
                       "T1": bool(t1_frac >= 0.75), "T2": bool(np.median(med) >= 1.5)}
    t3 = []
    for n in nulls:
        for l, v in n["result"].items():
            r = ratio(v)
            t3.append({"seed": n["seed"], "layer": int(l), "ratio": r["median"], "ci95": r["ci95"], "contains_1": bool(r["ci95"][0] <= 1 <= r["ci95"][1])})
    out["null"] = t3
    out["T3"] = bool(t3) and all(x["contains_1"] for x in t3)
    if real is not None:
        out["B_holds"] = bool(out["real"]["T1"] and out["real"]["T2"] and out["T3"])
    return out


DELTA = 0.05  # equivalence margin of preregistration_b_prime.md


def analyze_b_prime(real: dict, nulls: list[dict], n_layers: int) -> dict:
    """N, R, C of preregistration_b_prime.md for one model. `real`: {layer: summary}; `nulls`: arms with
    "seed" and "result" ({layer: summary})."""
    depth_ok = lambda l: int(l) / (n_layers - 1) >= 0.5
    rng = {int(l): v for l, v in real.items() if depth_ok(l)}
    lo = {l: ratio(v)["ci95"][0] for l, v in rng.items()}
    R_frac = float(np.mean([x > 1 + DELTA for x in lo.values()])) if lo else 0.0
    arms = [{"seed": n["seed"], "layer": int(l), "ci95": ratio(v)["ci95"], "median": ratio(v)["median"]} for n in nulls for l, v in n["result"].items()]
    for a in arms:
        a["equivalent"] = bool(1 - DELTA <= a["ci95"][0] and a["ci95"][1] <= 1 + DELTA)
    contrast = {}
    for l in sorted({a["layer"] for a in arms if depth_ok(a["layer"]) and a["layer"] in rng}):
        upper = max(a["ci95"][1] for a in arms if a["layer"] == l)
        contrast[l] = {"real_lower": lo[l], "null_upper_max": upper, "passed": bool(lo[l] > upper)}
    res = {"N": bool(arms) and all(a["equivalent"] for a in arms), "null_arms": arms,
           "R": bool(R_frac >= 0.75), "R_fraction_above_1.05": R_frac,
           "C": bool(contrast) and all(c["passed"] for c in contrast.values()), "C_detail": contrast,
           "median_ratio": float(np.median([ratio(v)["median"] for v in rng.values()])) if rng else None,
           "ratio_by_layer": {l: ratio(rng[l])["median"] for l in sorted(rng)}}
    if len(rng) >= 3:
        from scipy.stats import spearmanr

        res["spearman_layer_vs_ratio"] = float(spearmanr(list(rng), [ratio(v)["median"] for v in rng.values()])[0])
    res["B_prime_holds"] = bool(res["N"] and res["R"] and res["C"])
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=["null", "real", "analyze"])
    ap.add_argument("--model", default="gemma-3-12b")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", type=int, nargs="+", default=[24])
    ap.add_argument("--config", default="A", choices=list(CONFIGS), help="A: replication pre-registration; B2: B-prime; B3: B-prime on Llama 3.1 8B")
    a = ap.parse_args()
    CFG = CONFIGS[a.config]
    if a.arm == "null":
        print(null_arm(a.model, a.seed, a.layers))
    elif a.arm == "real":
        print(real_arm(a.model))
    else:
        nulls = [json.load(open(p)) for p in sorted(glob.glob(f"{OUT_DIR}/{CFG['prefix']}null_{a.model}_*.json"))]
        rp = f"{OUT_DIR}/{CFG['prefix']}real_{a.model}.json"
        legacy = "results/gain_h2_qwen3.5-4b.json"  # the original Qwen3.5-4B real arm (all lens layers)
        real = json.load(open(rp))["result"] if os.path.exists(rp) else (json.load(open(legacy)) if (a.model == "qwen3.5-4b" and a.config == "A") else None)
        n_layers = 48 if a.model.startswith("gemma-3-12b") else 32
        print(json.dumps(analyze_b_prime(real, nulls, n_layers) if a.config in ("B2", "B3") else analyze(real, nulls, n_layers), indent=1))

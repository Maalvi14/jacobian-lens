"""Build the weight-randomized null once (research notes §4.2): randomize a fresh copy
of the model, fit a small lens on it (the pre-fitted lens is meaningless there), and
record restricted-Hessian null distributions with the SAME protocol as the real
early-averaging run (layer, directions, output token, prompts, positions).

The fp32 randomized model is the one differentiated; a bf16 copy of the same weights
is used only for the lens fit (compute-bound: ~60 s per 64-token prompt on the Thor).
The fit checkpoints and is resumable; finished artifacts are reused.

Run:  uv run python -m hlens.experiments.build_nulls --mode reinit --seed 0
"""

from __future__ import annotations

import argparse
import copy
import json
import os

import torch
import transformers

from hlens.experiments.early_averaging import COUNTRIES
from hlens.harness import averaging_ratio, fit_null_lens, null_distributions, randomize_model
from hlens.loading import MODEL_ID, load_hf
from jlens import from_hf
from jlens.examples import load_wikitext_prompts


def build_null(
    *,
    mode: str = "reinit",
    seed: int = 0,
    layer: int = 16,
    n_fit_prompts: int = 30,
    n_contexts: int = 100,
    out_dir: str = "results/nulls",
) -> dict:
    tag = f"randomized_{mode}_s{seed}_L{layer}"
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    hf32 = load_hf(dtype=torch.float32)
    model32 = randomize_model(hf32, tok, mode=mode, seed=seed)

    lens_path = f"{out_dir}/lens_{tag}.pt"
    if not os.path.exists(lens_path):
        hf16 = copy.deepcopy(hf32).to(torch.bfloat16)
        model16 = from_hf(hf16, tok)
        fit_prompts = load_wikitext_prompts(100 + n_fit_prompts)[100:]  # disjoint from the evaluation prompts
        fit_null_lens(model16, fit_prompts, source_layers=[layer], dim_batch=32, max_seq_len=64, path=lens_path)
        del model16, hf16
        torch.cuda.empty_cache()
    from jlens import JacobianLens

    lens = JacobianLens.load(lens_path)

    ids = [tok.encode(w, add_special_tokens=False)[0] for w in COUNTRIES]
    nd = null_distributions(
        model32, lens, load_wikitext_prompts(n_contexts), name=tag, layer=layer, token_ids=ids,
        out_token=ids[0], out_dir=out_dir, meta={"mode": mode, "seed": seed, "n_fit_prompts": n_fit_prompts},
        log_every=10,
    )
    saved = torch.load(f"{out_dir}/{tag}.pt", weights_only=False)
    avg = averaging_ratio(saved["H"])
    summary = {"tag": tag, "averaging_ratio": avg.ratio, "null_mean": avg.null_mean, "null_p95": avg.null_p95,
               "diag_abs_q": {q: nd.quantile("diag", q) for q in (0.5, 0.9, 0.99)},
               "cross_abs_q": {q: nd.quantile("cross", q) for q in (0.5, 0.9, 0.99)}}
    with open(f"{out_dir}/{tag}_summary.json", "w") as f:
        json.dump(summary, f)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="reinit", choices=["reinit", "shuffle"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--n-fit-prompts", type=int, default=30)
    ap.add_argument("--n-contexts", type=int, default=100)
    a = ap.parse_args()
    print(build_null(mode=a.mode, seed=a.seed, layer=a.layer, n_fit_prompts=a.n_fit_prompts, n_contexts=a.n_contexts))

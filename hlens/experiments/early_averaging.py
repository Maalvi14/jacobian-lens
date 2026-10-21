"""Early averaging test (research notes §4.5): over ~100 pretraining-like prompts, is
||E[H_restricted]|| / E||H_restricted|| large enough that an *averaged* H-lens object
exists, the way J_l exists as an average? If not, every claim is per-context.

Definition used (the notes leave it open): at source layer `layer`, for each prompt
take one valid position p; the scalar is the logit of `out_token` at that same
position; the directions are a FIXED set of J-lens vectors (same in every context,
otherwise the matrices are not comparable); H_p[i, j] = v_i^T Hess(f)(h_p) v_j. The
first-order analogue (g_p[i] = v_i . grad f(h_p)) is computed on the same contexts as
a calibration: J_l exists as an average, so its ratio is the bar.

Run once:  uv run python -m hlens.experiments.early_averaging
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch

from hlens.harness import AveragingResult, averaging_ratio, collect_restricted_hessians

COUNTRIES = [" Italy", " Japan", " France", " Germany", " China", " Spain", " India", " Russia"]


@dataclass(frozen=True)
class EarlyAveraging:
    second_order: AveragingResult
    first_order: AveragingResult
    H: torch.Tensor  # [N, k, k]
    g: torch.Tensor  # [N, k]
    layer: int
    token_ids: list[int]
    out_token: int


def run_early_averaging(
    qwen,
    prompts: list[str],
    *,
    layer: int = 16,
    token_ids: list[int],
    out_token: int,
    max_length: int = 64,
    seed: int = 0,
    log_every: int = 10,
) -> EarlyAveraging:
    c = collect_restricted_hessians(
        qwen.model, qwen.lens, prompts, layer=layer, token_ids=token_ids, out_token=out_token,
        max_length=max_length, seed=seed, log_every=log_every,
    )
    return EarlyAveraging(
        second_order=averaging_ratio(c.H), first_order=averaging_ratio(c.g),
        H=c.H, g=c.g, layer=layer, token_ids=token_ids, out_token=out_token,
    )


def save(result: EarlyAveraging, name: str) -> None:
    os.makedirs("results", exist_ok=True)
    torch.save({"H": result.H, "g": result.g}, f"results/{name}.pt")  # *.pt is gitignored
    with open(f"results/{name}.json", "w") as f:
        json.dump(
            {"layer": result.layer, "token_ids": result.token_ids, "out_token": result.out_token,
             "second_order": vars(result.second_order), "first_order": vars(result.first_order)}, f,
        )


if __name__ == "__main__":
    from hlens.loading import load_qwen
    from jlens.examples import load_wikitext_prompts

    q = load_qwen(dtype=torch.float32)
    tok = q.tokenizer
    ids = [tok.encode(w, add_special_tokens=False)[0] for w in COUNTRIES]
    res = run_early_averaging(q, load_wikitext_prompts(100), token_ids=ids, out_token=ids[0])
    print("second order:", res.second_order, "\nfirst order:", res.first_order)
    save(res, "early_averaging_qwen3.5-4b_L16")

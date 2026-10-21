"""Second-order counterpart of the paper's A.7 battery: how much do the restricted-
Hessian summaries depend on the recipe? One tidy table, one row per config.

Axes implemented (the A.7 design space, restricted to what applies to HVPs):
  attention     full | frozen   attention pattern held constant (value pathway only;
                                the second-order version of frozen-QK, hlens.variants)
  skip_first    positions excluded from sampling (position scheme)
  direction_set lens | random   fixed J-lens vectors vs random unit directions
  estimator     fr | fd         forward-over-reverse vs central finite difference
  aggregation   mean | median   across contexts, per element
  nsigma        None | N        drop contexts whose residual norm is > N sigma above the mean
Not implemented: the "gradient of what" axes (target layer, self-only / future-only
positions) and the data axes (corpus size / distribution). They change how J_l is
fitted, not how a Hessian is evaluated; they would be rows of the lens-fitting
ablation, not of this table.

Metrics per row: averaging ratio ||agg(H)|| / mean||H|| and its sign-shuffle null, mean
||H||_F, mean relative asymmetry ||H - H^T|| / ||H||, and the number of contexts kept.
The expensive axes (attention, skip_first, direction_set, estimator) are computed once
and cached; aggregation and nsigma are post-processing.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial

import torch

from hlens.harness import ablation_table, averaging_ratio, collect_restricted_hessians
from hlens.hvp import hvp_fd, hvp_fr
from hlens.variants import frozen_attention

DEFAULT_GRID = {
    "attention": ["full", "frozen"],
    "skip_first": [0, 4],
    "direction_set": ["lens", "random"],
    "estimator": ["fr", "fd"],
    "aggregation": ["mean", "median"],
    "nsigma": [None, 2.0],
}


def hessian_battery(
    qwen,
    prompts: Sequence[str],
    *,
    layer: int,
    token_ids: Sequence[int],
    out_token: int,
    grid: dict | None = None,
    fd_eps: float = 3e-2,
    max_length: int = 64,
    seed: int = 0,
    csv_path: str | None = None,
) -> list[dict]:
    model, lens = qwen.model, qwen.lens
    cache: dict[tuple, object] = {}

    def random_directions(k: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed + 123)
        d = torch.randn(k, model.d_model, generator=g)
        return d / d.norm(dim=1, keepdim=True)

    def compute(attention, skip_first, direction_set, estimator):
        key = (attention, skip_first, direction_set, estimator)
        if key not in cache:
            kwargs = dict(
                layer=layer, out_token=out_token, max_length=max_length, seed=seed, skip_first=skip_first,
                hvp_fn=hvp_fr if estimator == "fr" else partial(hvp_fd, eps=fd_eps),
            )
            if direction_set == "lens":
                kwargs["token_ids"] = list(token_ids)
            else:
                kwargs["directions"] = random_directions(len(token_ids))
            if attention == "frozen":
                with frozen_attention():
                    cache[key] = collect_restricted_hessians(model, lens, prompts, **kwargs)
            else:
                cache[key] = collect_restricted_hessians(model, lens, prompts, **kwargs)
        return cache[key]

    def run(attention, skip_first, direction_set, estimator, aggregation, nsigma):
        c = compute(attention, skip_first, direction_set, estimator)
        norms = torch.tensor(c.norms)
        keep = torch.ones_like(norms, dtype=torch.bool) if nsigma is None else norms <= norms.mean() + nsigma * norms.std()
        H = c.H[keep]
        agg = H.mean(0) if aggregation == "mean" else H.median(0).values
        frob = H.flatten(1).norm(dim=1)
        avg = averaging_ratio(H)
        return {
            "n_kept": int(keep.sum()),
            "ratio": (agg.norm() / frob.mean()).item(),
            "null_ratio": avg.null_mean,
            "mean_frob": frob.mean().item(),
            "rel_asymmetry": (c.asymmetry[keep] / H.abs().flatten(1).max(dim=1).values.clamp_min(1e-12)).mean().item(),
        }

    return ablation_table(run, grid or DEFAULT_GRID, csv_path=csv_path)


if __name__ == "__main__":
    import argparse

    from hlens.experiments.early_averaging import COUNTRIES
    from hlens.loading import load_qwen
    from jlens.examples import load_wikitext_prompts

    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--n-contexts", type=int, default=16)
    ap.add_argument("--k", type=int, default=6)
    a = ap.parse_args()
    q = load_qwen(dtype=torch.float32)
    ids = [q.tokenizer.encode(w, add_special_tokens=False)[0] for w in COUNTRIES[: a.k]]
    rows = hessian_battery(
        q, load_wikitext_prompts(a.n_contexts), layer=a.layer, token_ids=ids, out_token=ids[0],
        csv_path="results/ablation_battery_qwen3.5-4b.csv",
    )
    print(f"{len(rows)} rows written to results/ablation_battery_qwen3.5-4b.csv")

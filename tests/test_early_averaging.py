import pytest
import torch

from hlens.experiments.early_averaging import COUNTRIES, run_early_averaging
from hlens.harness import averaging_ratio
from jlens.examples import load_wikitext_prompts


def test_averaging_ratio_extremes():
    torch.manual_seed(0)
    same = torch.randn(4, 4).expand(50, 4, 4)
    assert averaging_ratio(same).ratio == pytest.approx(1.0, abs=1e-5)

    indep = torch.randn(400, 4, 4)  # independent contexts: nothing consistent to average
    r = averaging_ratio(indep)
    assert r.ratio < 3 * r.null_mean
    assert r.null_mean == pytest.approx(1 / 20, rel=0.5)  # ~ 1/sqrt(400)

    shared = torch.randn(4, 4) + 0.2 * torch.randn(400, 4, 4)  # one consistent H plus noise
    s = averaging_ratio(shared)
    assert s.ratio > 3 * s.null_p95


@pytest.mark.real_model
def test_early_averaging_pipeline_small(qwen_fp32):
    """Mechanics only (N=8); the real run is `python -m hlens.experiments.early_averaging`."""
    tok = qwen_fp32.tokenizer
    ids = [tok.encode(w, add_special_tokens=False)[0] for w in COUNTRIES]
    res = run_early_averaging(qwen_fp32, load_wikitext_prompts(8), token_ids=ids, out_token=ids[0], log_every=0)
    assert res.H.shape == (8, 8, 8) and res.g.shape == (8, 8)
    assert torch.isfinite(res.H).all()
    assert (res.H - res.H.transpose(1, 2)).abs().max() < 1e-3 * res.H.abs().max()
    print("\nN=8:", res.second_order, res.first_order, flush=True)

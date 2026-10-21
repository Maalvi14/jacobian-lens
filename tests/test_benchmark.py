import pytest
import torch

from hlens.harness import _power_mode, benchmark

PROMPT = "Fact: The currency used in the country shaped like a boot is"


@pytest.mark.real_model
def test_benchmark_guard_and_timings(qwen_fp32):
    tok = qwen_fp32.tokenizer
    ids = [tok.encode(w, add_special_tokens=False)[0] for w in (" Italy", " Japan", " France", " Germany", " China", " Spain", " India", " Russia")]
    mode = _power_mode()
    print(f"\npower mode: {mode!r}", flush=True)
    kw = dict(layer=16, token_ids=ids, out_token=ids[0])
    if mode != "MAXN":
        with pytest.raises(RuntimeError, match="MAXN"):
            benchmark(qwen_fp32, PROMPT, **kw)
    r = benchmark(qwen_fp32, PROMPT, require_max_power=(mode == "MAXN"), **kw)
    print(r, flush=True)
    print(f"cost milestone (restricted H <= 3x lens.apply): {'MET' if r.meets_milestone() else 'NOT met'}"
          f"{'' if mode == 'MAXN' else ' (power mode is not MAXN: timing is not comparable)'}", flush=True)
    assert 0 < r.seconds_per_hvp < r.seconds_per_restricted_hessian
    assert r.seconds_per_lens_apply > 0

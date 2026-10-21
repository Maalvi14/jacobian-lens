import json
import os

import pytest

from hlens.gain import active_token_pool, mlp_gain_curves
from jlens.examples import load_wikitext_prompts


@pytest.mark.real_model
def test_mlp_gain_gate(qwen_fp32):
    """Paper §4.3.1 / Fig 32 left: J-lens vectors reach ~10x MLP gain in the workspace
    band (rising from ~2-4x mid-band to ~10x at ~85% depth, falling after), MLP-neuron
    output directions stay near 1. NOT reproduced on Qwen3.5-4B as stated: with the best
    population (tokens actually active in the lens) the gain is <= ~2x through most of the
    band and climbs to ~8x only at the last measurable layer (L30). The assertions record
    what we observe so a regression is caught; they do not claim reproduction."""
    model, lens = qwen_fp32.model, qwen_fp32.lens
    vocab = qwen_fp32.hf.config.get_text_config().vocab_size
    pool = active_token_pool(model, lens, load_wikitext_prompts(10))
    spec = mlp_gain_curves(model, lens, vocab_size=vocab)  # uniform over the vocabulary
    act = mlp_gain_curves(model, lens, vocab_size=vocab, token_pool=lambda l: pool[l])

    print("\nlayer | J-lens (random vocab) | J-lens (active tokens) | neuron", flush=True)
    for i, l in enumerate(act.layers):
        print(f"  L{l:02d} | {spec.j_lens[i]:21.2f} | {act.j_lens[i]:22.2f} | {act.neuron[i]:.2f}", flush=True)

    os.makedirs("results", exist_ok=True)
    with open("results/gain_qwen3.5-4b.json", "w") as f:
        json.dump({"layers": act.layers, "j_lens_random_vocab": spec.j_lens,
                   "j_lens_active_tokens": act.j_lens, "neuron": act.neuron}, f)

    j, n, L = act.j_lens, act.neuron, act.layers
    assert all(0.5 < x < 2.2 for x in n), "neuron-direction gain stays modest (paper: ~1)"
    assert all(j[L.index(l)] > n[L.index(l)] for l in range(21, 31)), "late J-lens gain exceeds neuron gain"
    mid = sum(j[L.index(l)] for l in range(6, 13)) / 7
    late = sum(j[L.index(l)] for l in range(26, 31)) / 5
    assert late > 3 * mid, "J-lens gain climbs toward the output"
    assert max(j) < 15, "sanity: gain in the paper's ballpark or below"

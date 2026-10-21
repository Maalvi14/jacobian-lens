import time

import pytest
import torch

from hlens.harness import fit_null_lens, randomize_model
from hlens.loading import MODEL_ID, load_hf
from jlens.examples import load_wikitext_prompts


@pytest.mark.real_model
def test_randomized_model_and_null_lens_fit_timing(qwen):
    """Randomize a FRESH copy (never the session's), then time one prompt of the
    null-lens fit: this is what decides how big the null pipeline can be."""
    import transformers

    hf = load_hf(dtype=torch.bfloat16)
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    real_w = qwen.hf.model.layers[10].mlp.down_proj.weight.detach().clone()
    model = randomize_model(hf, tok, mode="reinit", seed=0)
    assert not torch.equal(real_w, hf.model.layers[10].mlp.down_proj.weight), "weights must change"
    assert torch.equal(real_w, qwen.hf.model.layers[10].mlp.down_proj.weight), "the session model must be untouched"
    prompts = load_wikitext_prompts(2)
    ids = model.encode(prompts[0], max_length=64)
    with torch.no_grad():
        logits = hf(input_ids=ids).logits
    print(f"\nrandomized model logits: std {logits.float().std():.3f}, finite {bool(torch.isfinite(logits).all())}", flush=True)
    for dim_batch in (32, 64):
        t0 = time.time()
        lens = fit_null_lens(model, prompts[:1], source_layers=[12, 16, 20], dim_batch=dim_batch, max_seq_len=64)
        torch.cuda.synchronize()
        print(f"fit 1 prompt (64 tokens, 3 layers, dim_batch={dim_batch}): {time.time() - t0:.1f}s", flush=True)
    assert lens.source_layers == [12, 16, 20] and lens.d_model == 2560

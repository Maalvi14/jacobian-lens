import pytest
import torch

import jlens
from hlens.loading import LENS_FILE_N1000, LENS_REPO

pytestmark = pytest.mark.real_model

BOOT_PROMPT = "Fact: The currency used in the country shaped like a boot is"


def test_shapes(qwen):
    assert qwen.model.n_layers == 32
    assert qwen.model.d_model == 2560
    assert qwen.lens.d_model == 2560
    assert qwen.lens.source_layers == list(range(31))
    print(f"\nmodel lens:  {qwen.lens}")
    print(f"peak after load: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


def test_n1000_lens_matches_layout(qwen):
    other = jlens.JacobianLens.from_pretrained(LENS_REPO, filename=LENS_FILE_N1000)
    assert other.source_layers == qwen.lens.source_layers
    assert other.d_model == qwen.lens.d_model
    print(f"\ndefault lens: {qwen.lens}")
    print(f"n1000 lens:   {other}")


def test_boot_prompt_readout(qwen):
    tok = qwen.tokenizer
    lens_logits, model_logits, input_ids = qwen.lens.apply(
        qwen.model, BOOT_PROMPT, positions=[-2]
    )
    assert set(lens_logits) == set(qwen.lens.source_layers)
    assert model_logits.shape == (1, qwen.hf.config.get_text_config().vocab_size)
    assert torch.isfinite(model_logits).all()

    print(f"\nreadout at token {tok.decode(input_ids[0, -2:-1])!r}")
    for layer, logits in sorted(lens_logits.items()):
        top = [tok.decode([t]) for t in logits[0].topk(5).indices]
        print(f"  L{layer:02d}  {top}")
    print(f"  model {[tok.decode([t]) for t in model_logits[0].topk(5).indices]}")
    print(f"peak after apply: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
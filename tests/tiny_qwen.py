"""A tiny CPU Qwen3.5 (hybrid linear/full attention) wrapped for jlens, so the
HF-specific harness code can be unit-tested without the 4B model."""

from types import SimpleNamespace

import torch

from jlens import JacobianLens, from_hf

VOCAB = 64


class _ByteTokenizer:
    add_bos_token = False
    bos_token_id = None

    def __call__(self, text, return_tensors="pt", truncation=True, max_length=512):
        ids = torch.tensor([[(ord(c) * 7) % VOCAB for c in text][:max_length]])
        return SimpleNamespace(input_ids=ids)


def tiny_hf(seed=0):
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    base = AutoConfig.from_pretrained("Qwen/Qwen3.5-4B").get_text_config().to_dict()
    base.update(
        vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, linear_key_head_dim=8,
        linear_value_head_dim=8, linear_num_key_heads=2, linear_num_value_heads=2,
        layer_types=base["layer_types"][:4], eos_token_id=None, bos_token_id=None, pad_token_id=None,
    )
    torch.manual_seed(seed)
    # eager attention: SDPA has no forward-mode autodiff, which the HVP estimators need
    return Qwen3_5ForCausalLM._from_config(Qwen3_5TextConfig(**base), attn_implementation="eager", dtype=torch.float32).eval()


def tiny_model(seed=0):
    hf = tiny_hf(seed)
    return hf, from_hf(hf, _ByteTokenizer())


def tiny_lens(model, layers=(0, 1, 2), seed=0):
    g = torch.Generator().manual_seed(seed)
    d = model.d_model
    return JacobianLens(
        {l: torch.randn(d, d, generator=g) / d**0.5 for l in layers}, n_prompts=1, d_model=d
    )

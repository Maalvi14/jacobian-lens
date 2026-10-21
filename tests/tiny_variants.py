"""Tiny CPU stand-ins for the two second-model candidates (Gemma 3 12B, Qwen3.6-27B),
so the architecture-specific parts of the harness can be checked before any weights
are downloaded. Real checkpoints still need the real_model checks."""

import torch

from jlens import from_hf

from .tiny_qwen import VOCAB, _ByteTokenizer, tiny_hf


def tiny_gemma3_hf(seed=0, n_layers=6, sliding_window=4):
    from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM, Gemma3TextConfig

    cfg = Gemma3TextConfig(
        vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=n_layers,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, sliding_window=sliding_window,
        layer_types=["sliding_attention"] * (n_layers - 1) + ["full_attention"],
        eos_token_id=None, bos_token_id=None, pad_token_id=None,
    )
    torch.manual_seed(seed)
    return Gemma3ForCausalLM._from_config(cfg, attn_implementation="eager", dtype=torch.float32).eval()


def tiny_untied_qwen_hf(seed=0):
    """Qwen3.6-27B differs from Qwen3.5-4B in having an untied lm_head."""
    hf = tiny_hf(seed)
    hf.config.tie_word_embeddings = False
    hf.lm_head.weight = torch.nn.Parameter(torch.randn_like(hf.lm_head.weight))
    return hf


def tiny_llama_hf(seed=0, n_layers=4):
    """A tiny dense Llama-style decoder (the architecture class of the third B-prime model)."""
    from transformers.models.llama.modeling_llama import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=n_layers, num_attention_heads=2, num_key_value_heads=1,
                      head_dim=16, eos_token_id=None, bos_token_id=None, pad_token_id=None, tie_word_embeddings=False)
    torch.manual_seed(seed)
    return LlamaForCausalLM._from_config(cfg, attn_implementation="eager", dtype=torch.float32).eval()


BUILDERS = {"gemma3": tiny_gemma3_hf, "qwen_untied": tiny_untied_qwen_hf, "llama": tiny_llama_hf}


def tiny_wrapped(name, seed=0):
    hf = BUILDERS[name](seed)
    return hf, from_hf(hf, _ByteTokenizer())

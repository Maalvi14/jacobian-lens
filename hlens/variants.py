"""Model-internal variants for the ablation battery."""

from __future__ import annotations

import contextlib

import torch


@contextlib.contextmanager
def frozen_attention():
    """Qwen3.5 full-attention layers treat the attention pattern (the softmax
    weights) as a constant: gradients and Hessians flow only through the value
    pathway ("what gets moved"), not through "what gets attended to". This is the
    second-order counterpart of the paper's frozen-QK variant (A.7). The
    Gated-DeltaNet (linear-attention) layers have no softmax pattern and are
    unaffected. Applies to eager attention, the only mode used here."""
    import transformers.models.qwen3_5.modeling_qwen3_5 as m

    original = m.eager_attention_forward

    def frozen(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        key_states = m.repeat_kv(key, module.num_key_value_groups)
        value_states = m.repeat_kv(value, module.num_key_value_groups)
        w = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            w = w + attention_mask
        w = torch.nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype).detach()
        out = torch.matmul(w, value_states).transpose(1, 2).contiguous()
        return out, w

    m.eager_attention_forward = frozen
    try:
        yield
    finally:
        m.eager_attention_forward = original

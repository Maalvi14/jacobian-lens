"""Architecture adapters: the few places where hlens reaches into a decoder block by
module name. Qwen/Llama blocks are `x + mlp(post_attention_layernorm(x))`; Gemma 3 blocks
add a post-norm on the branch: `x + post_feedforward_layernorm(mlp(pre_feedforward_layernorm(x)))`."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


def mlp_prenorm(block: nn.Module) -> nn.Module:
    """The module whose input is the residual entering the block's MLP sub-block."""
    for name in ("pre_feedforward_layernorm", "post_attention_layernorm"):
        if hasattr(block, name):
            return getattr(block, name)
    raise AttributeError(f"{type(block).__name__}: no known pre-MLP norm")


def mlp_branch(block: nn.Module) -> Callable[[torch.Tensor], torch.Tensor]:
    """residual entering the MLP sub-block -> what that sub-block adds to the residual
    (own pre-norm included, and Gemma's post-norm)."""
    pre = mlp_prenorm(block)
    post = getattr(block, "post_feedforward_layernorm", None)
    if post is None:
        return lambda x: block.mlp(pre(x))
    return lambda x: post(block.mlp(pre(x)))


def embed_rows(embedding: nn.Module, token_ids: list[int]) -> torch.Tensor:
    """What the embedding module outputs for each token ([n, d_model]) -- not the raw
    weight row: Gemma's embedding multiplies by sqrt(d_model)."""
    device = next(embedding.parameters()).device
    with torch.no_grad():
        return embedding(torch.tensor([token_ids], device=device))[0].detach()

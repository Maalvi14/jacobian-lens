from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from jlens.protocol import LensModel


class _BlockKwargsCapture:
    """Captures the forward kwargs (position_embeddings, attention_mask,
    position_ids, ...) HF passes into one residual block, by hooking it
    for the next forward pass. Removes the hook on exit."""

    def __init__(self, block: nn.Module) -> None:
        self._block = block
        self.kwargs: dict | None = None
        self._handle: torch.utils.hooks.RemovableHandle | None = None

    def _hook(self, module: nn.Module, args, kwargs) -> None:
        assert self.kwargs is None
        self.kwargs = kwargs

    def __enter__(self) -> _BlockKwargsCapture:
        self._handle = self._block.register_forward_pre_hook(self._hook, with_kwargs=True)
        return self
    
    def __exit__(self, *exc) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def _capture_block_kwargs(model: LensModel, block_index: int, seq_len: int) -> dict:
    block = model.layers[block_index]
    device = next(block.parameters()).device
    dummy_ids = torch.zeros((1, seq_len), dtype=torch.long, device=device)

    # DisableFuncTorch: if the first call happens inside a torch.func transform
    # (e.g. hvp_fr), the captured tensors would be wrapped at that transform's
    # level and "escape" when the cache is reused under a later transform.
    # Capturing at the base level keeps them plain.
    with (
        torch._C._DisableFuncTorch(),
        _BlockKwargsCapture(block) as capture,
        torch.no_grad(),
    ):
        model.forward(dummy_ids)

    assert capture.kwargs is not None, (
        f"block {block_index} never ran during the capture forward pass "
        "— check block_index is within range and reachable from model.forward"
    )

    return capture.kwargs


def make_slice_fn(model: LensModel, l_start: int, l_end: int
) -> Callable[[torch.Tensor], torch.Tensor]:
    
    if not (0 <= l_start < l_end < model.n_layers):
        raise ValueError(
            f"require 0 <= l_start < l_end < n_layers={model.n_layers}, "
            f"got l_start={l_start}, l_end={l_end}")

    cache: dict[tuple[int, int], dict] = {}

    def slice_fn(h: torch.Tensor) -> torch.Tensor:
        seq_len = h.shape[-2]
        for i in range(l_start + 1, l_end + 1):
            key = (seq_len, i)
            if key not in cache:
                cache[key] = _capture_block_kwargs(model, i, seq_len)
            out = model.layers[i](h, **cache[key])
            h = out if torch.is_tensor(out) else out[0]
        return h

    return slice_fn


def make_output_fn(
        model: LensModel, l_start: int, contraction: Callable[[torch.Tensor], torch.Tensor]
    ) -> Callable[[torch.Tensor], torch.Tensor]:
    slice_fn = make_slice_fn(model, l_start, model.n_layers - 1)

    def output_fn(h: torch.Tensor) -> torch.Tensor:
        h_final = slice_fn(h)
        return contraction(h_final)

    return output_fn
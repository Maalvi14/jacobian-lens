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

# --- mixed precision: bf16 model, fp32 measurement -------------------------------------------------------------


def _floats_to(x, dtype=torch.float32):
    """Recursively cast floating-point tensors (inside tuples / lists / dicts); leave everything else."""
    if torch.is_tensor(x):
        return x.to(dtype) if x.is_floating_point() else x
    if isinstance(x, tuple):
        return tuple(_floats_to(i, dtype) for i in x)
    if isinstance(x, list):
        return [_floats_to(i, dtype) for i in x]
    if isinstance(x, dict):
        return {k: _floats_to(v, dtype) for k, v in x.items()}
    return x


def _fp32_rotary(model: LensModel, block_index: int, seq_len: int, pos: torch.Tensor):
    """cos/sin from the model's own rotary module fed an fp32 input (so they are not rounded to bf16), or None if the
    module's call signature is not one we know (the caller then keeps the cast bf16 values)."""
    import inspect

    rotary = getattr(model._text_module, "rotary_emb", None)
    if rotary is None:
        return None
    x = torch.zeros(1, seq_len, 1, dtype=torch.float32, device=pos.device)
    try:
        with torch.no_grad():
            if "layer_type" in inspect.signature(rotary.forward).parameters:  # Gemma 3: one rotary per layer type
                layer_type = model._hf_model.config.get_text_config().layer_types[block_index]
                return tuple(t.float() for t in rotary(x, pos, layer_type))
            try:
                return tuple(t.float() for t in rotary(x, pos))
            except (IndexError, RuntimeError):  # Qwen3.5 mrope wants [3, batch, seq] position ids
                return tuple(t.float() for t in rotary(x, pos[None].expand(3, *pos.shape)))
    except Exception:
        return None


def _fp32_block_kwargs(model: LensModel, block_index: int, seq_len: int) -> dict:
    """Block kwargs captured from the (bf16) model and cast to fp32. The rotary cos/sin are recomputed in fp32
    when the text model exposes `rotary_emb` and the block was given position_ids (the bf16 forward rounds them)."""
    # base level only: tensors made inside a torch.func transform would "escape" when the cache is reused later
    with torch._C._DisableFuncTorch():
        kw = _capture_block_kwargs(model, block_index, seq_len)
        out = _floats_to(kw)
        pe, pos = kw.get("position_embeddings"), kw.get("position_ids")
        fresh = _fp32_rotary(model, block_index, seq_len, pos) if (pe is not None and pos is not None) else None
        if fresh is not None and all(a.shape == b.shape for a, b in zip(fresh, pe)):
            out["position_embeddings"] = fresh
    return out


def make_mixed_slice_fn(model: LensModel, l_start: int, l_end: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """Like `make_slice_fn`, but the blocks l_start+1 .. l_end run as fp32 copies, whatever the model's dtype.
    Input is cast to fp32 and the output is fp32. Costs ~4 bytes per parameter of the sliced blocks only."""
    import copy

    if not (0 <= l_start < l_end < model.n_layers):
        raise ValueError(f"require 0 <= l_start < l_end < n_layers={model.n_layers}, got l_start={l_start}, l_end={l_end}")
    blocks = {i: copy.deepcopy(model.layers[i]).float() for i in range(l_start + 1, l_end + 1)}
    cache: dict[tuple[int, int], dict] = {}

    def slice_fn(h: torch.Tensor) -> torch.Tensor:
        h = h.float()
        seq_len = h.shape[-2]
        for i in range(l_start + 1, l_end + 1):
            if (seq_len, i) not in cache:
                cache[(seq_len, i)] = _fp32_block_kwargs(model, i, seq_len)
            out = blocks[i](h, **cache[(seq_len, i)])
            h = out if torch.is_tensor(out) else out[0]
        return h

    return slice_fn


class Fp32Unembed:
    """Final norm + lm_head as fp32 copies (for unembedding directions without bf16 rounding). Exposes the two
    members `hlens.jspace.unembed_directions` uses: `d_model` and `unembed`."""

    def __init__(self, model) -> None:
        import copy

        self.d_model = model.d_model
        self._norm = copy.deepcopy(model._final_norm).float()
        self._head = copy.deepcopy(model._lm_head).float()
        self._softcap = model._logit_softcap

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        dev = self._head.weight.device
        logits = self._head(self._norm(residual.float().to(dev)))
        return logits if self._softcap is None else self._softcap * torch.tanh(logits / self._softcap)

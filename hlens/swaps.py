"""Applying and scoring lens-coordinate swaps on the probe-swap suite. The protocol
(decided in the swap sweep, handoff §3.4): swap at every position over a band of
layers, one `ActivationEditor` per layer (each layer has its own lens vectors),
raw (unnormalised) lens vectors, and score the next token at the last position."""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable, Sequence

import torch

from hlens.data import ProbeSwapItem, load_probe_swap
from hlens.interventions import ActivationEditor, coordinate_swap
from jlens.hooks import ActivationRecorder

DEFAULT_BAND = (12, 20)  # inclusive
DEFAULT_ALPHA = 1.25  # best α of the sweep on L12-20 (17% at α=1.0)


def swap_edit(v_s: torch.Tensor, v_t: torch.Tensor, alpha: float) -> Callable[[torch.Tensor], torch.Tensor]:
    def edit(h: torch.Tensor) -> torch.Tensor:
        return coordinate_swap(h.float(), v_s, v_t, alpha=alpha).to(h.dtype)

    return edit


def pair_edits(
    layers: Iterable[int],
    vec_fn: Callable[[int], torch.Tensor],
    alpha: float,
    device,
    *,
    normalize: bool = False,
) -> list[tuple[int, Callable]]:
    """[(layer, edit_fn)], one per layer; `vec_fn(layer)` -> [2, d_model] (v_s, v_t)."""
    out = []
    for layer in layers:
        v = vec_fn(layer)
        if normalize:
            v = v / v.norm(dim=1, keepdim=True)
        v = v.to(device)
        out.append((layer, swap_edit(v[0], v[1], alpha)))
    return out


def run_top1(model, ids: torch.Tensor, edits: Sequence[tuple[int, Callable]] = ()) -> int:
    """Greedy next-token id at the last position with `edits` applied to block outputs."""
    last = model.n_layers - 1
    with torch.no_grad(), contextlib.ExitStack() as stack:
        for layer, fn in edits:
            stack.enter_context(ActivationEditor(model.layers, at=[layer], edit_fn=fn))
        rec = stack.enter_context(ActivationRecorder(model.layers, at=[last]))
        model.forward(ids)
        return model.unembed(rec.activations[last][0, -1]).argmax().item()


def first_token_ids(tok, word: str, *, fold_case: bool = False) -> set[int]:
    """First non-whitespace token of `word`, with and without a leading space (the prompts
    end in a trailing space, so either form can occur). Pure-whitespace tokens are skipped:
    Qwen tokenises " 5" as [" ", "5"] (digits do not merge with a preceding space), and
    counting the bare-space token made every numeric answer "overlap" with every other and
    made a space top-1 look like a correct answer / a swap hit (found in session 2)."""
    ids = set()
    forms = {word, word.lower(), word.capitalize()} if fold_case else {word}
    for form in forms:
        for w in (form, " " + form):
            for t in tok.encode(w, add_special_tokens=False):
                if tok.decode([t]).strip():
                    ids.add(t)
                    break
    return ids


def item_token_ids(tok, it: ProbeSwapItem, *, fold_case: bool = False) -> tuple[set[int], set[int]]:
    """(answer ids, swap-answer ids). With `fold_case` the lower-case and capitalised forms count too ("euro" / "Euro",
    "solid" / "Solid"), except for `element-symbol` items, whose answers are case-sensitive symbols ("He" is not "he")."""
    fold = fold_case and it.category != "element-symbol"
    return first_token_ids(tok, it.answer, fold_case=fold), first_token_ids(tok, it.swap_answer, fold_case=fold)


def is_ambiguous(tok, it: ProbeSwapItem, *, fold_case: bool = False) -> bool:
    """True if the answer and the swap answer share a first token (e.g. "20" vs "26": both
    start with "2"), so a first-token match cannot tell them apart."""
    a, b = item_token_ids(tok, it, fold_case=fold_case)
    return bool(a & b)


def probe_items(
    tok, path: str = "data/experiments/probe-swap.json", *, drop_ambiguous: bool = True, fold_case: bool = False
) -> list[tuple[ProbeSwapItem, int, int]]:
    """(item, id of ' intermediate', id of ' swap_to') for the items whose two entities are
    single tokens (lens vectors are per token; multi-token concepts degrade, paper Fig 62),
    minus items whose answer and swap answer cannot be told apart by first token."""
    out = []
    for it in load_probe_swap(path):
        if drop_ambiguous and is_ambiguous(tok, it, fold_case=fold_case):
            continue
        s = tok.encode(" " + it.intermediate, add_special_tokens=False)
        t = tok.encode(" " + it.swap_to, add_special_tokens=False)
        if len(s) == 1 and len(t) == 1:
            out.append((it, s[0], t[0]))
    return out

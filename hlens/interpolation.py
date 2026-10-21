"""Embedding interpolation (data/experiments/README.md, ignition): replace the
embedding of the `{W}` token by alpha*emb(A) + (1-alpha)*emb(B), alpha swept 0 -> 1, and
read the lens at the `{W}` position. The straight-line path in embedding space is
flow-matching's canonical linear interpolant, used here as a diagnostic probe."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from hlens.arch import embed_rows
from jlens.hooks import ActivationRecorder


@dataclass(frozen=True)
class InterpolationSetup:
    input_ids: torch.Tensor  # [1, seq], with word_b in the {W} slot
    position: int  # index of the {W} token
    id_a: int
    id_b: int
    e_a: torch.Tensor  # [d_model]
    e_b: torch.Tensor


def make_setup(qwen, template: str, word_a: str, word_b: str, *, max_length: int = 64, clean: bool = False) -> InterpolationSetup:
    """`template` contains `{W}`; words are tokenised with a leading space and must
    be single tokens. `clean` removes the legacy double space before the slot (needed for Gemma)."""
    tok, model = qwen.tokenizer, qwen.model
    ids_a = tok.encode(" " + word_a, add_special_tokens=False)
    ids_b = tok.encode(" " + word_b, add_special_tokens=False)
    if len(ids_a) != 1 or len(ids_b) != 1:
        raise ValueError(f"{word_a!r}/{word_b!r} must each be a single token")
    prefix, _, suffix = template.partition("{W}")
    if clean:
        # legacy construction writes prefix(+trailing space) + " " + word, i.e. a double space; Qwen splits it into
        # [' ', ' Word'] but Gemma merges it into ['  ', 'Word'] (a different token). clean=True writes one space.
        prefix = prefix.rstrip()
    input_ids = model.encode(prefix + " " + word_b + suffix, max_length=max_length)
    position = model.encode(prefix, max_length=max_length).shape[1] if prefix else 0
    if input_ids[0, position].item() != ids_b[0]:
        raise ValueError(f"could not locate the {{W}} slot in {template!r}")
    e = embed_rows(qwen.hf.get_input_embeddings(), [ids_a[0], ids_b[0]])
    return InterpolationSetup(
        input_ids=input_ids, position=position, id_a=ids_a[0], id_b=ids_b[0], e_a=e[0], e_b=e[1],
    )


def forward_at_alpha(
    qwen, setup: InterpolationSetup, alpha: float, layers: Sequence[int]
) -> dict[int, torch.Tensor]:
    """Residual streams [1, seq, d_model] at `layers` with the {W} embedding
    replaced by alpha*e_A + (1-alpha)*e_B."""
    model = qwen.model
    pos = setup.position
    mix = alpha * setup.e_a + (1 - alpha) * setup.e_b

    def hook(module, inputs, output):
        out = output.clone()
        out[0, pos] = mix.to(out.dtype)
        return out

    handle = qwen.hf.get_input_embeddings().register_forward_hook(hook)
    try:
        with torch.no_grad(), ActivationRecorder(model.layers, at=list(layers)) as rec:
            model.forward(setup.input_ids)
            return {l: rec.activations[l].detach().clone() for l in layers}
    finally:
        handle.remove()


def rank_share(logits: torch.Tensor, id_a: int, id_b: int) -> float:
    """A's reciprocal-rank share: (1/rank_A) / (1/rank_A + 1/rank_B)."""
    ra = (logits > logits[id_a]).sum().item() + 1
    rb = (logits > logits[id_b]).sum().item() + 1
    return (1 / ra) / (1 / ra + 1 / rb)


def lens_share_curve(
    qwen, setup: InterpolationSetup, alphas: Sequence[float], read_layers: Sequence[int]
) -> dict[int, list[float]]:
    """read layer -> A's lens rank share at the {W} position, per alpha."""
    out = {l: [] for l in read_layers}
    for a in alphas:
        acts = forward_at_alpha(qwen, setup, a, read_layers)
        for l in read_layers:
            h = acts[l][0, setup.position].float()
            logits = qwen.model.unembed(qwen.lens.transport(h, l)).float()
            out[l].append(rank_share(logits, setup.id_a, setup.id_b))
    return out

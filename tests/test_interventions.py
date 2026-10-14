import contextlib
from pathlib import Path

import pytest
import torch

from hlens.data import load_probe_swap
from hlens.interventions import ActivationEditor, coordinate_swap
from hlens.jspace import lens_vectors
from jlens import JacobianLens
from jlens.hooks import ActivationRecorder

def test_coordinate_swap_then_swap_back_is_identity():
    torch.manual_seed(0)
    d_model = 8
    v_s = torch.randn(d_model)
    v_t = torch.randn(d_model)
    h = torch.randn(d_model)

    swapped = coordinate_swap(h, v_s, v_t, alpha=1.0)
    swapped_back = coordinate_swap(swapped, v_s, v_t, alpha=1.0)

    torch.testing.assert_close(swapped_back, h, rtol=1e-4, atol=1e-4)


def test_coordinate_swap_leaves_orthogonal_component_untouched():
    torch.manual_seed(1)
    d_model = 8
    v_s = torch.randn(d_model)
    v_t = torch.randn(d_model)

    # build an orthogonal complement via QR, then take an h that's a mix of
    # v_s/v_t plus a component along a direction outside their span
    V = torch.stack([v_s, v_t], dim=1)
    Q, _ = torch.linalg.qr(V, mode="complete")
    outside_direction = Q[:, 2]  # orthogonal to both v_s and v_t

    h = 2.0 * v_s + 3.0 * v_t + 5.0 * outside_direction

    swapped = coordinate_swap(h, v_s, v_t, alpha=1.0)

    outside_before = torch.dot(h, outside_direction)
    outside_after = torch.dot(swapped, outside_direction)
    torch.testing.assert_close(outside_after, outside_before, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Swap reproduction (plan §4), real Qwen3.5-4B in bf16.
# Paper (§2.5, §3.3): h <- h + alpha * V (sigma(c) - c), V = [v_s v_t], applied at
# every token position over the "workspace layer range" (~L38-L92 on a 0-100 depth
# scale, i.e. roughly layers 12-28 of Qwen's 32; that mapping is an assumption, the
# paper reports only Claude models). Whether swap vectors are unit-norm or raw is not
# stated, so both are swept. Run: uv run pytest tests/test_interventions.py -m real_model -s
# ---------------------------------------------------------------------------

BANDS = [(12, 20), (12, 24), (12, 28)]  # inclusive layer ranges
ALPHAS = [0.7, 0.9, 1.0, 1.25, 1.5, 2.0]
SCALINGS = [True, False]  # unit-normalise the lens vectors?
PROBE_SWAP = Path(__file__).parent.parent / "data" / "experiments" / "probe-swap.json"
# ends in "is the" so the next token starts the currency phrase ("... the euro" / "... the Japanese yen"); with
# a bare "is" or "is " the model answers " the" / digits instead
BOOT_PROMPT = "Fact: The currency used in the country shaped like a boot is the"
EURO_WORDS = ["euro", "Euro", "lira", "Lira", "EUR", "€", "Italian"]
YEN_WORDS = ["yen", "Yen", "JPY", "¥", "Japanese"]


def _token_ids(tok, words):
    """First-token ids of each word, with and without a leading space."""
    return {tok.encode(w, add_special_tokens=False)[0] for x in words for w in (x, " " + x)}


def _swap_edit(v_s, v_t, alpha):
    def edit(h):
        return coordinate_swap(h.float(), v_s, v_t, alpha=alpha).to(h.dtype)

    return edit


def _unembed_rows(qwen, token_ids):
    """u_t = d logit_t / d h_final for each token: [n, d_model]. Layer-independent;
    a lens vector at layer l is u_t @ J_l (lens_vectors with an identity lens)."""
    d = qwen.model.d_model
    identity = JacobianLens({0: torch.eye(d)}, n_prompts=0, d_model=d)
    return lens_vectors(identity, qwen.model, 0, list(token_ids))


def _pair_edits(layers, vec_fn, alpha, device, normalize):
    """One ActivationEditor edit per layer. ActivationEditor shares a single
    edit_fn across the layers it hooks, but J_l (hence every lens vector) differs
    per layer, so each layer gets its own editor."""
    out = []
    for layer in layers:
        v = vec_fn(layer)  # [2, d_model]
        if normalize:
            v = v / v.norm(dim=1, keepdim=True)
        v = v.to(device)
        out.append((layer, _swap_edit(v[0], v[1], alpha)))
    return out


def _readout(model, tok, ids, edits):
    """Next-token top-1 id and top-5 strings at the last position, with `edits` =
    [(layer, edit_fn), ...] applied to the block outputs."""
    last = model.n_layers - 1
    with torch.no_grad(), contextlib.ExitStack() as stack:
        for layer, fn in edits:
            stack.enter_context(ActivationEditor(model.layers, at=[layer], edit_fn=fn))
        rec = stack.enter_context(ActivationRecorder(model.layers, at=[last]))
        model.forward(ids)
        logits = model.unembed(rec.activations[last][0, -1]).float().cpu()
    top = logits.topk(5).indices
    return top[0].item(), [tok.decode([t]) for t in top]


def _random_pair(d, seed):
    g = torch.Generator().manual_seed(seed)
    return lambda layer: torch.randn(2, d, generator=g)


@pytest.mark.real_model
def test_multihop_swap_flips_currency(qwen):
    model, tok = qwen.model, qwen.tokenizer
    device = next(qwen.hf.parameters()).device
    src = tok.encode(" Italy", add_special_tokens=False)
    tgt = tok.encode(" Japan", add_special_tokens=False)
    assert len(src) == 1 and len(tgt) == 1
    euro_ids, yen_ids = _token_ids(tok, EURO_WORDS), _token_ids(tok, YEN_WORDS)
    ids = model.encode(BOOT_PROMPT, max_length=64)
    U2 = _unembed_rows(qwen, [src[0], tgt[0]])
    jac = qwen.lens.jacobians

    base_id, base_top = _readout(model, tok, ids, [])
    print(f"\n  baseline top5={base_top}", flush=True)
    assert base_id in euro_ids, f"baseline should answer Italy's currency, got {base_top}"

    flipped = []
    for lo, hi in BANDS:
        for normalize in SCALINGS:
            for alpha in ALPHAS:
                edits = _pair_edits(range(lo, hi + 1), lambda l: U2 @ jac[l], alpha, device, normalize)
                top_id, top = _readout(model, tok, ids, edits)
                ok = top_id in yen_ids
                if ok:
                    flipped.append((lo, hi, normalize, alpha))
                print(f"  L{lo}-{hi} unit={normalize!s:<5} a={alpha:<4} {'FLIP' if ok else '    '} top5={top}", flush=True)

    # control: identical edit machinery, random directions instead of Italy/Japan
    lo, hi = flipped[0][:2] if flipped else BANDS[0]
    for seed in range(3):
        edits = _pair_edits(range(lo, hi + 1), _random_pair(model.d_model, seed), 1.0, device, True)
        top_id, top = _readout(model, tok, ids, edits)
        print(f"  random control L{lo}-{hi} seed={seed}: top5={top}", flush=True)
        assert top_id not in yen_ids

    print(f"  configs flipping to a yen token at top-1: {len(flipped)}/{len(BANDS) * len(SCALINGS) * len(ALPHAS)}", flush=True)
    assert flipped, "no swap configuration moved the top-1 answer to Japan's currency"


def _probe_items(tok):
    items = []
    for it in load_probe_swap(str(PROBE_SWAP)):
        s = tok.encode(" " + it.intermediate, add_special_tokens=False)
        t = tok.encode(" " + it.swap_to, add_special_tokens=False)
        if len(s) == 1 and len(t) == 1:  # lens vectors are per single token
            items.append((it, s[0], t[0]))
    return items


@pytest.mark.real_model
def test_probe_swap_sweep(qwen):
    """probe-swap.json: baseline greedy next token == `answer`; after swapping
    `intermediate` -> `swap_to` at every position over the band, next token ==
    `swap_answer`. Rates are over baseline-correct items (the paper does not say
    what its denominator is). The sweep is tuned on these same items, so the best
    config shows whether the paper's regime is reachable, not an unbiased rate."""
    model, tok = qwen.model, qwen.tokenizer
    device = next(qwen.hf.parameters()).device
    jac = qwen.lens.jacobians

    items = _probe_items(tok)
    tokens = sorted({x for _, s, t in items for x in (s, t)})
    U, row = _unembed_rows(qwen, tokens), {t: i for i, t in enumerate(tokens)}

    todo = []  # baseline-correct items
    for it, s, t in items:
        ids = model.encode(it.prompt, max_length=128)
        ans = {tok.encode(w, add_special_tokens=False)[0] for w in (it.answer, " " + it.answer)}
        swap_ans = {tok.encode(w, add_special_tokens=False)[0] for w in (it.swap_answer, " " + it.swap_answer)}
        if _readout(model, tok, ids, [])[0] in ans:
            todo.append((it, ids, U[[row[s], row[t]]], ans, swap_ans))
    n = len(todo)
    print(f"\n{len(items)}/90 items single-token; baseline correct: {n}/{len(items)}", flush=True)

    def run(layers, alpha, normalize, vec_for=None):
        hits, moved, per_item = 0, 0, []
        for i, (it, ids, U2, ans, swap_ans) in enumerate(todo):
            vec = vec_for(i) if vec_for else (lambda l, U2=U2: U2 @ jac[l])
            top_id = _readout(model, tok, ids, _pair_edits(layers, vec, alpha, device, normalize))[0]
            hit = top_id in swap_ans
            hits += hit
            moved += top_id not in ans
            per_item.append((it.category, hit))
        return hits, moved, per_item

    results = {}
    for lo, hi in BANDS:
        for normalize in SCALINGS:
            for alpha in ALPHAS:
                hits, moved, per_item = run(range(lo, hi + 1), alpha, normalize)
                results[(lo, hi, normalize, alpha)] = (hits, moved, per_item)
                print(f"  L{lo}-{hi} unit={normalize!s:<5} a={alpha:<4} hit {hits:>2}/{n} ({hits / n:.0%})  moved off answer {moved:>2}/{n}", flush=True)

    best = max(results, key=lambda k: results[k][0])
    lo, hi, normalize, alpha = best
    hits, moved, per_item = results[best]
    print(f"\nbest config: L{lo}-{hi} unit={normalize} alpha={alpha}: {hits}/{n} ({hits / n:.0%})", flush=True)
    by_cat = {}
    for cat, hit in per_item:
        c = by_cat.setdefault(cat, [0, 0])
        c[0] += 1
        c[1] += hit
    print("  by category (hit/n):", {k: f"{v[1]}/{v[0]}" for k, v in sorted(by_cat.items())}, flush=True)

    ctrl = [run(range(lo, hi + 1), alpha, normalize, vec_for=lambda i, s=s: _random_pair(model.d_model, 1000 * s + i))[0] for s in range(3)]
    print(f"  random-direction control at best config (3 seeds): hits {ctrl}", flush=True)

    # Thresholds set from the observed run (best 18/53 = 34%, random control 1/53 at
    # that config), with margin: the swap must be clearly above chance, not match the paper.
    assert n >= 20, "baseline should answer a decent share of the suite"
    assert hits / n >= 0.25, f"best swap config only reached {hits}/{n}"
    assert hits >= 5 * max(max(ctrl), 1), f"swap ({hits}) not clearly above random control {ctrl}"

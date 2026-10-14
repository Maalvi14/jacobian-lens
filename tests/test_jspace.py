import time

import pytest
import torch

from hlens.jspace import lens_vectors, sparse_pursuit, decompose, k_variation
from jlens import JacobianLens
from jlens.examples import load_wikitext_prompts
from jlens.fitting import fit, valid_position_mask
from jlens.hooks import ActivationRecorder

from .tiny import TinyDecoder


def test_lens_vectors_ratio_consistent_across_tokens():
    model = TinyDecoder(n_layers=4, d_model=8)
    prompts = ["abcdefghij " * 5, "klmnopqrst " * 5]
    lens = fit(model, prompts, source_layers=[0, 1, 2], dim_batch=4, max_seq_len=64)

    layer = 1
    token_ids = [3, 7, 12]
    v = lens_vectors(lens, model, layer, token_ids)

    torch.manual_seed(0)
    for _ in range(3):
        h = torch.randn(model.d_model)
        lens_logits = model.unembed(lens.transport(h, layer))

        ratios = torch.stack(
            [torch.dot(v[i], h) / lens_logits[t] for i, t in enumerate(token_ids)]
        )
        relative_spread = (ratios.std() / ratios.mean().abs()).item()
        assert relative_spread < 0.05, f"ratios={ratios.tolist()}"

def test_sparse_pursuit_recovers_exact_orthonormal_combination():
    torch.manual_seed(0)
    d_model = 6
    n_atoms = 6
    dictionary, _ = torch.linalg.qr(torch.randn(d_model, n_atoms))
    dictionary = dictionary.T  # rows are orthonormal atoms: [n_atoms, d_model]

    h = 2.0 * dictionary[3] + 1.5 * dictionary[1]

    indices, coeffs = sparse_pursuit(h, dictionary, k=5)

    assert set(indices) == {3, 1}
    recovered = dict(zip(indices, coeffs.tolist()))
    assert abs(recovered[3] - 2.0) < 1e-3
    assert abs(recovered[1] - 1.5) < 1e-3

    reconstruction = dictionary[indices].T @ coeffs
    torch.testing.assert_close(reconstruction, h, rtol=1e-3, atol=1e-3)


def test_decompose_frac_variance_exact():
    torch.manual_seed(0)
    d_model = 6
    n_atoms = 6
    dictionary, _ = torch.linalg.qr(torch.randn(d_model, n_atoms))
    dictionary = dictionary.T

    h = 2.0 * dictionary[3] + 1.5 * dictionary[1] + 1.0 * dictionary[5]

    result = decompose(h, dictionary, k=2)

    assert set(result.indices) == {3, 1}
    expected_jspace = 2.0 * dictionary[3] + 1.5 * dictionary[1]
    expected_remainder = 1.0 * dictionary[5]

    torch.testing.assert_close(result.jspace_component, expected_jspace, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(result.remainder, expected_remainder, rtol=1e-3, atol=1e-3)

    expected_frac_variance = (2.0**2 + 1.5**2) / (2.0**2 + 1.5**2 + 1.0**2)
    assert abs(result.frac_variance - expected_frac_variance) < 1e-3

    # Pythagorean cross-check from last turn: these two formulas must agree
    # exactly, since remainder is orthogonal to jspace_component by NNLS optimality.
    alt_frac_variance = 1 - (result.remainder.norm() ** 2 / h.norm() ** 2).item()
    assert abs(result.frac_variance - alt_frac_variance) < 1e-6


def test_k_variation_prefix_consistency():
    torch.manual_seed(0)
    d_model = 6
    n_atoms = 6
    dictionary, _ = torch.linalg.qr(torch.randn(d_model, n_atoms))
    dictionary = dictionary.T

    h = 2.0 * dictionary[3] + 1.5 * dictionary[1] + 1.0 * dictionary[5]

    results = k_variation(h, dictionary, k_values=[2, 4])

    assert results[2].indices == results[4].indices[: len(results[2].indices)]


# ---------------------------------------------------------------------------
# Paper-reproduction gate (plan §3), real Qwen3.5-4B in bf16 (the precision the
# lens was fitted in): J-space share of variance at mid layers, atoms per
# position, and sensible atoms on the boot prompt.
# Run: uv run pytest tests/test_jspace.py -m real_model -s
# ---------------------------------------------------------------------------

GATE_LAYERS = [12, 16, 20]
N_PROMPTS = 50
POSITIONS_PER_PROMPT = 3
K = 25
BOOT_PROMPT = "Fact: The currency used in the country shaped like a boot is"


@pytest.fixture(scope="module")
def unembed_directions(qwen):
    """u_t = d logit_t / d h_final for every token: [vocab, d_model]. Independent
    of the lens layer, so compute once. (lens_vectors with an identity lens
    returns u @ I = u.)"""
    d = qwen.model.d_model
    vocab = qwen.hf.config.get_text_config().vocab_size
    identity_lens = JacobianLens({0: torch.eye(d)}, n_prompts=0, d_model=d)
    t0 = time.time()
    u = lens_vectors(identity_lens, qwen.model, 0, list(range(vocab)), dim_batch=512)
    print(f"\nunembed directions {tuple(u.shape)} in {time.time() - t0:.0f}s", flush=True)
    return u


@pytest.fixture(scope="module")
def dictionary_for(qwen, unembed_directions):
    """layer -> (unit-norm J-lens dictionary [n_atoms, d_model], token id per
    atom). Keeps one 2.5 GB dictionary alive at a time."""
    cache = {}

    def get(layer):
        if layer not in cache:
            cache.clear()
            V = unembed_directions @ qwen.lens.jacobians[layer]
            norms = V.norm(dim=1)
            keep = norms > 1e-6  # unused vocab rows would break unit-norm
            cache[layer] = (V[keep] / norms[keep, None], keep.nonzero().squeeze(1))
        return cache[layer]

    return get


def _collect_residuals(model, prompts, layers):
    """{layer: [n_prompts * POSITIONS_PER_PROMPT, d_model]} fp32 on CPU, at
    evenly spaced positions that pass jlens's own valid_position_mask."""
    out = {layer: [] for layer in layers}
    for prompt in prompts:
        ids = model.encode(prompt, max_length=128)
        with torch.no_grad(), ActivationRecorder(model.layers, at=layers) as rec:
            model.forward(ids)
            acts = {l: rec.activations[l][0].float().cpu() for l in layers}
        valid = valid_position_mask(ids.shape[1]).nonzero().squeeze(1)
        pick = valid[torch.linspace(0, len(valid) - 1, POSITIONS_PER_PROMPT).round().long()]
        for layer in layers:
            out[layer].append(acts[layer][pick])
    return {layer: torch.cat(v) for layer, v in out.items()}


def _pursuit_stats(hs, dictionary):
    fracs, n_selected, n_active = [], [], []
    for h in hs:
        d = decompose(h, dictionary, k=K)
        fracs.append(d.frac_variance)
        n_selected.append(len(d.indices))
        # "meaningfully active": coefficient within 5% of the largest one
        n_active.append(int((d.coeffs >= 0.05 * d.coeffs.max()).sum()))
    return torch.tensor(fracs), torch.tensor(n_selected).float(), torch.tensor(n_active).float()


def _summarize(name, fracs, n_selected, n_active):
    q = torch.quantile(fracs, torch.tensor([0.5, 0.9]))
    print(
        f"  {name}: frac_variance mean={fracs.mean():.3f} median={q[0]:.3f} "
        f"p90={q[1]:.3f} max={fracs.max():.3f} | atoms selected={n_selected.mean():.1f} "
        f"active(>=5% of max coef)={n_active.mean():.1f}",
        flush=True,
    )


@pytest.mark.real_model
def test_jspace_paper_gate(qwen, dictionary_for):
    model, tok = qwen.model, qwen.tokenizer
    prompts = load_wikitext_prompts(N_PROMPTS)
    residuals = _collect_residuals(model, prompts, GATE_LAYERS)
    boot_ids = model.encode(BOOT_PROMPT, max_length=64)
    with torch.no_grad(), ActivationRecorder(model.layers, at=GATE_LAYERS) as rec:
        model.forward(boot_ids)
        boot_acts = {l: rec.activations[l][0].float().cpu() for l in GATE_LAYERS}

    medians = {}
    for layer in GATE_LAYERS:
        D, token_ids = dictionary_for(layer)
        print(f"\nlayer {layer}: dictionary {tuple(D.shape)}", flush=True)
        t0 = time.time()
        stats = _pursuit_stats(residuals[layer], D)
        _summarize(f"WikiText ({len(stats[0])} positions, {time.time() - t0:.0f}s)", *stats)
        medians[layer] = torch.median(stats[0]).item()

        if layer == GATE_LAYERS[len(GATE_LAYERS) // 2]:
            # control: same pursuit against random unit atoms of the same count
            g = torch.Generator().manual_seed(0)
            R = torch.randn(D.shape, generator=g)
            R /= R.norm(dim=1, keepdim=True)
            _summarize("random-dictionary control (40 positions)", *_pursuit_stats(residuals[layer][:40], R))
            del R

        for pos in (-2, -1):
            d = decompose(boot_acts[layer][pos], D, k=K)
            order = d.coeffs.argsort(descending=True)[:8]
            atoms = [tok.decode([token_ids[d.indices[i]].item()]) for i in order]
            print(f"  boot prompt pos {pos} ({tok.decode(boot_ids[0, pos:][:1])!r}) "
                  f"frac={d.frac_variance:.3f} top atoms: {atoms}", flush=True)

    for layer, m in medians.items():
        assert m <= 0.10, f"layer {layer}: median J-space fraction {m:.3f} > 0.10"

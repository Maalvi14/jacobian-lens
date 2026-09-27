import torch

from hlens.jspace import lens_vectors, sparse_pursuit, decompose, k_variation
from jlens.fitting import fit

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
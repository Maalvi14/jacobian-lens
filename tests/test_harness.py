import pytest
import torch

from hlens.contexts import at_position, capture_context, lens_direction_bank, logit_fn
from hlens.harness import find_plateau, eps_sweep, estimator_agreement, ms_hvp, validity_score
from hlens.hvp import hvp_fr
from hlens.jspace import unembed_directions

def test_find_plateau_detects_flat_region():
    eps_grid = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    base = torch.tensor([1.0, 2.0, 3.0, 4.0])
    curve = torch.stack([
        base + torch.tensor([5.0, 0.0, 0.0, 0.0]),    # far off — noisy tiny-eps end
        base,                                          # flat region starts
        base + torch.tensor([0.001, 0.0, 0.0, 0.0]),  # flat
        base + torch.tensor([0.002, 0.0, 0.0, 0.0]),  # flat
        base + torch.tensor([10.0, 0.0, 0.0, 0.0]),   # diverges — large-eps breakdown
        base + torch.tensor([20.0, 0.0, 0.0, 0.0]),
    ])

    result = find_plateau(curve, eps_grid, tol=0.05, min_run_length=2)

    assert result.has_plateau
    assert result.eps_range == (1e-4, 1e-2)
    assert result.representative_eps == 1e-3


def test_find_plateau_no_plateau_when_always_changing():
    eps_grid = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    base = torch.tensor([1.0, 2.0, 3.0, 4.0])
    curve = torch.stack([base * (2**i) for i in range(len(eps_grid))])

    result = find_plateau(curve, eps_grid, tol=0.05, min_run_length=2)

    assert not result.has_plateau
    assert result.eps_range is None
    assert result.representative_eps is None


def test_eps_sweep_on_quadratic_form_is_flat_everywhere():
    torch.manual_seed(0)
    d = 6
    # float64: a central difference of a linear gradient is exact, so float32
    # rounding (amplified by 1/(2*eps)) would be the only error left
    A = torch.randn(d, d, dtype=torch.float64)
    A = (A + A.T) / 2

    h = torch.randn(d, dtype=torch.float64)
    v = torch.randn(d, dtype=torch.float64)

    def f(h: torch.Tensor) -> torch.Tensor:
        return 0.5 * h @ A @ h

    eps_grid = [1e-4, 1e-3, 1e-2, 1e-1]
    curve = eps_sweep(f, h, v, eps_grid)

    expected = A @ v
    for row in curve:
        torch.testing.assert_close(row, expected, rtol=1e-6, atol=1e-6)

    result = find_plateau(curve, eps_grid, tol=0.01, min_run_length=2)
    assert result.has_plateau


def test_estimator_agreement_quadratic_form_usable():
    torch.manual_seed(0)
    d = 6
    A = torch.randn(d, d)
    A = (A + A.T) / 2
    h = torch.randn(d)
    v = torch.randn(d)

    def f(h: torch.Tensor) -> torch.Tensor:
        return 0.5 * h @ A @ h

    result = estimator_agreement(f, h, v, tol=0.01)

    assert result.usable
    for pair, err in result.pairwise.items():
        assert err < 0.01, f"{pair} disagreed by {err}"


def test_estimator_agreement_flags_fd_disagreement_at_tight_tolerance():
    torch.manual_seed(1)
    d = 6
    A = torch.randn(d, d)
    A = (A + A.T) / 2
    h = torch.randn(d)
    v = torch.randn(d)

    def f(h: torch.Tensor) -> torch.Tensor:
        return 0.5 * h @ A @ h

    # rr and fr are both exact autodiff methods — they agree almost to floating-
    # point precision regardless of tolerance. fd has genuine O(eps^2)
    # discretization error, however small; an unreasonably tight tol exposes it.
    result = estimator_agreement(f, h, v, tol=1e-8)

    assert not result.usable
    assert result.pairwise[("rr", "fr")] < 1e-6
    assert result.pairwise[("rr", "fd")] > 1e-8


def test_ms_hvp_large_step_diverges_more_than_small_step():
    torch.manual_seed(0)
    d = 4
    h = torch.randn(d)

    def f(h: torch.Tensor) -> torch.Tensor:
        return (h**4).sum() / 4

    v_large = torch.randn(d) * 2.0
    v_small = v_large * 0.001  # same direction, much smaller

    result_large = ms_hvp(f, h, v_large, n_steps=200)
    result_small = ms_hvp(f, h, v_small, n_steps=200)

    assert result_large.reliability > 0.05
    assert result_small.reliability < 0.01


def test_ms_hvp_multi_step_converges_to_true_gradient_difference():
    torch.manual_seed(1)
    d = 4
    h = torch.randn(d)
    v = torch.randn(d) * 2.0

    def f(h: torch.Tensor) -> torch.Tensor:
        return (h**4).sum() / 4

    result = ms_hvp(f, h, v, n_steps=200)

    true_diff = (h + v) ** 3 - h**3  # exact, closed-form: grad(f)(h+v) - grad(f)(h)
    torch.testing.assert_close(result.multi_step, true_diff, rtol=1e-2, atol=1e-2)

    single_step_error = (result.single_step - true_diff).norm()
    multi_step_error = (result.multi_step - true_diff).norm()
    assert multi_step_error < single_step_error


def test_validity_score_valid_when_perturbation_within_radius():
    torch.manual_seed(0)
    d = 6
    A = torch.randn(d, d)
    A = (A + A.T) / 2
    h = torch.randn(d)

    v_raw = torch.randn(d)
    v = v_raw / v_raw.norm() * 0.05  # small perturbation, well inside the plateau

    def f(h: torch.Tensor) -> torch.Tensor:
        return 0.5 * h @ A @ h

    eps_grid = [1e-4, 1e-3, 1e-2, 1e-1]
    result = validity_score(f, h, v, eps_grid=eps_grid)

    assert result.valid
    assert result.taylor_radius == 1e-1
    assert result.validity_score > 1.0


def test_validity_score_invalid_when_perturbation_exceeds_radius():
    torch.manual_seed(0)
    d = 6
    A = torch.randn(d, d)
    A = (A + A.T) / 2
    h = torch.randn(d)

    v_raw = torch.randn(d)
    v = v_raw / v_raw.norm() * 5.0  # large perturbation, well outside the plateau

    def f(h: torch.Tensor) -> torch.Tensor:
        return 0.5 * h @ A @ h

    eps_grid = [1e-4, 1e-3, 1e-2, 1e-1]
    result = validity_score(f, h, v, eps_grid=eps_grid)

    assert not result.valid
    assert result.taylor_radius is not None
    assert result.validity_score is not None
    assert result.validity_score < 1.0

# --- harness on the real model (fp32 Qwen3.5-4B) -----------------------------------
# Scalar f = a logit at the boot prompt's " boot" token read from layer 16; the
# perturbation direction is the unit J-lens vector for " Italy".


def _real_scalar_problem(qwen_fp32, layer=16):
    model, tok = qwen_fp32.model, qwen_fp32.tokenizer
    point = capture_context(
        model, "Fact: The currency used in the country shaped like a boot is", layer, position=-2
    )
    top = model.unembed(point.h[0, -1]).argmax().item()
    f1, x0 = at_position(logit_fn(model, point, top), point)
    italy = tok.encode(" Italy", add_special_tokens=False)[0]
    v = lens_direction_bank(unembed_directions(model, [italy]).to(x0.device), qwen_fp32.lens, layer)[0]
    return f1, x0, v


@pytest.mark.real_model
def test_eps_sweep_plateau_and_estimator_agreement_real(qwen_fp32):
    f, x0, v = _real_scalar_problem(qwen_fp32)
    exact = hvp_fr(f, x0, v)
    grid = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0]  # absolute step along a unit v
    curve = eps_sweep(f, x0, v, grid)
    plateau = find_plateau(curve, grid, tol=0.05)
    errs = [((c - exact).norm() / exact.norm()).item() for c in curve]
    print("\nfd error vs exact by eps:", {e: f"{x:.1e}" for e, x in zip(grid, errs)}, flush=True)
    print("plateau:", plateau, flush=True)
    assert plateau.has_plateau
    lo, hi = plateau.eps_range
    assert 1e-4 < lo and hi < 3.0  # a real interior plateau: noisy at tiny eps, breaks down at large eps
    assert ((curve[grid.index(plateau.representative_eps)] - exact).norm() / exact.norm()).item() < 0.05

    # both ends of the U-shaped fd error are flagged unusable; the plateau eps is usable
    noisy = estimator_agreement(f, x0, v, tol=0.05, eps=1e-4)  # rounding noise (~7%)
    coarse = estimator_agreement(f, x0, v, tol=0.05, eps=3.0)  # truncation error (~40%)
    good = estimator_agreement(f, x0, v, tol=0.05, eps=plateau.representative_eps)
    print("agreement eps=1e-4:", noisy.pairwise, "\n  eps=3:", coarse.pairwise, "\n  plateau eps:", good.pairwise, flush=True)
    assert not noisy.usable
    assert not coarse.usable
    assert good.usable
    assert good.pairwise[("rr", "fr")] < 1e-4


@pytest.mark.real_model
def test_ms_hvp_converges_to_exact_secant_real(qwen_fp32):
    f, x0, v = _real_scalar_problem(qwen_fp32)
    grad = torch.func.grad(f)

    def secant_err(size, n_steps):
        vv = size * v
        exact = grad(x0 + vv) - grad(x0)  # integral of H(x0 + t vv) vv dt, exactly
        r = ms_hvp(f, x0, vv, n_steps=n_steps)
        return ((r.multi_step - exact).norm() / exact.norm()).item(), ((r.single_step - exact).norm() / exact.norm()).item(), r.reliability

    size = 0.5 * x0.norm().item()  # a large perturbation: half the residual norm
    errs = {n: secant_err(size, n)[0] for n in (1, 2, 4, 8, 16)}
    print(f"\nmulti-step error vs exact secant (|v|=0.5|x0|): {errs}", flush=True)
    assert errs[16] < errs[4] < errs[1]
    assert errs[16] < errs[1] / 3

    small = secant_err(0.01 * x0.norm().item(), 4)
    large = secant_err(size, 4)
    print(f"reliability small={small[2]:.3e} large={large[2]:.3e}", flush=True)
    assert large[2] > 3 * small[2]  # divergence between 1-step and multi-step grows with the perturbation


@pytest.mark.real_model
def test_validity_score_flags_oversized_perturbations_real(qwen_fp32):
    f, x0, v = _real_scalar_problem(qwen_fp32)
    grid = [1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0]
    small = validity_score(f, x0, 0.01 * v, eps_grid=grid)
    big = validity_score(f, x0, 100.0 * v, eps_grid=grid)
    print("\nsmall:", small, "\nbig:", big, flush=True)
    assert small.valid and small.validity_score >= 1
    assert not big.valid


# --- stage-C harness pieces, unit-tested on a tiny CPU Qwen3.5 ---------------------

from hlens.harness import (  # noqa: E402
    ablation_table, collect_restricted_hessians, load_null, null_distributions, randomize_model,
)
from hlens.variants import frozen_attention  # noqa: E402

from .tiny_qwen import _ByteTokenizer, tiny_hf, tiny_lens, tiny_model  # noqa: E402


def _state(hf):
    return {k: v.detach().clone() for k, v in hf.state_dict().items()}


def test_randomize_model_reinit_is_deterministic_and_changes_weights():
    ref = _state(tiny_hf(0))
    a, b = tiny_hf(0), tiny_hf(0)
    randomize_model(a, _ByteTokenizer(), mode="reinit", seed=1)
    randomize_model(b, _ByteTokenizer(), mode="reinit", seed=1)
    c = tiny_hf(0)
    randomize_model(c, _ByteTokenizer(), mode="reinit", seed=2)
    sa, sb, sc = _state(a), _state(b), _state(c)
    big = [k for k, v in ref.items() if v.dim() == 2 and v.numel() > 100]
    assert big and all(not torch.equal(ref[k], sa[k]) for k in big), "weights must actually change"
    assert all(torch.equal(sa[k], sb[k]) for k in sa), "same seed -> identical weights"
    assert any(not torch.equal(sa[k], sc[k]) for k in big), "different seed -> different weights"
    ids = torch.randint(0, 64, (1, 8))
    assert not torch.allclose(tiny_hf(0)(input_ids=ids).logits, a(input_ids=ids).logits)
    # HF's own rule: RMSNorm weights are re-zeroed ((1 + w) convention)
    norms = [k for k in sa if "layernorm" in k and k.endswith("weight")]
    assert norms and all(sa[k].abs().max() == 0 for k in norms)


def test_randomize_model_shuffle_preserves_each_tensors_values():
    ref = _state(tiny_hf(0))
    hf = tiny_hf(0)
    model = randomize_model(hf, _ByteTokenizer(), mode="shuffle", seed=3)
    out = _state(hf)
    assert model.n_layers == 4
    big = [k for k, v in ref.items() if v.dim() == 2 and v.numel() > 100]
    for k in ref:
        assert torch.equal(ref[k].flatten().sort().values, out[k].flatten().sort().values), k
    assert all(not torch.equal(ref[k], out[k]) for k in big)
    with pytest.raises(ValueError):
        randomize_model(tiny_hf(0), _ByteTokenizer(), mode="bogus")


def test_ablation_table_runs_the_full_grid_and_writes_csv(tmp_path):
    seen = []

    def run(a, b, scale):
        seen.append((a, b))
        return {"metric": scale * a + b}

    path = tmp_path / "t.csv"
    rows = ablation_table(run, {"a": [1, 2, 3], "b": [10, 20]}, fixed={"scale": 2}, csv_path=str(path))
    assert len(rows) == 6 and len(set(seen)) == 6
    assert rows[0] == {"a": 1, "b": 10, "metric": 12}
    lines = path.read_text().strip().splitlines()
    assert lines[0] == "a,b,metric" and len(lines) == 7


def test_collect_restricted_hessians_and_null_roundtrip(tmp_path):
    hf, model = tiny_model(0)
    lens = tiny_lens(model)
    prompts = ["the quick brown fox jumps over", "pack my box with five dozen jugs", "how vexingly quick"]
    c = collect_restricted_hessians(model, lens, prompts, layer=1, token_ids=[3, 5, 7], out_token=3, max_length=24)
    H, g, norms = c.H, c.g, c.norms
    assert H.shape == (3, 3, 3) and g.shape == (3, 3) and len(norms) == 3 and c.asymmetry.shape == (3,)
    assert torch.isfinite(H).all()
    assert (H - H.transpose(1, 2)).abs().max() < 1e-4 * H.abs().max().clamp_min(1e-6)

    nd = null_distributions(model, lens, prompts, name="toy", layer=1, token_ids=[3, 5, 7], out_token=3,
                            out_dir=str(tmp_path), max_length=24, meta={"mode": "toy"})
    assert nd.diag.numel() == 3 * 3 and nd.cross.numel() == 3 * 6
    back = load_null("toy", out_dir=str(tmp_path))
    assert torch.equal(back.diag, nd.diag) and back.meta["mode"] == "toy"
    assert nd.above(float(nd.cross.abs().max()) * 10) and not nd.above(0.0)
    assert (tmp_path / "toy.json").exists()


def test_frozen_attention_changes_curvature_not_the_forward_pass():
    import transformers.models.qwen3_5.modeling_qwen3_5 as m
    from hlens.contexts import at_position, capture_context, logit_fn
    from hlens.hvp import hvp_rr

    hf, model = tiny_model(0)
    original = m.eager_attention_forward
    prompt = "the quick brown fox jumps over"
    ids = model.encode(prompt)
    with torch.no_grad():
        plain = hf(input_ids=ids).logits
        with frozen_attention():
            frozen_out = hf(input_ids=ids).logits
    assert torch.allclose(plain, frozen_out, atol=1e-6), "freezing must not change values"
    assert m.eager_attention_forward is original, "patch must be undone on exit"

    point = capture_context(model, prompt, 0, position=-2, max_length=32)
    f1, x0 = at_position(logit_fn(model, point, 3), point)
    v = torch.randn_like(x0)
    normal = hvp_fr(f1, x0, v)
    with frozen_attention():
        f1f, _ = at_position(logit_fn(model, point, 3), point)
        frozen_fr = hvp_fr(f1f, x0, v)
        frozen_rr = hvp_rr(f1f, x0.clone().requires_grad_(True), v)
    torch.testing.assert_close(frozen_fr, frozen_rr, rtol=1e-4, atol=1e-6)
    assert not torch.allclose(normal, frozen_fr, rtol=1e-3, atol=1e-6), "frozen pattern must change the Hessian"


def test_hessian_battery_runs_on_tiny_model(tmp_path):
    from types import SimpleNamespace

    from hlens.experiments.ablation_battery import hessian_battery

    hf, model = tiny_model(0)
    qwen = SimpleNamespace(model=model, lens=tiny_lens(model))
    prompts = ["the quick brown fox jumps over", "pack my box with five dozen jugs", "how vexingly quick daft zebras", "sphinx of black quartz judge my vow"]
    grid = {"attention": ["full", "frozen"], "skip_first": [0, 2], "direction_set": ["lens", "random"],
            "estimator": ["fr"], "aggregation": ["mean", "median"], "nsigma": [None, 1.0]}
    path = tmp_path / "battery.csv"
    rows = hessian_battery(qwen, prompts, layer=1, token_ids=[3, 5, 7], out_token=3, grid=grid, max_length=24, csv_path=str(path))
    assert len(rows) == 2 * 2 * 2 * 1 * 2 * 2
    assert all(r["n_kept"] >= 1 and torch.isfinite(torch.tensor(r["ratio"])) for r in rows)
    full = next(r for r in rows if r["attention"] == "full" and r["skip_first"] == 0 and r["direction_set"] == "lens" and r["aggregation"] == "mean" and r["nsigma"] is None)
    frozen = next(r for r in rows if r["attention"] == "frozen" and r["skip_first"] == 0 and r["direction_set"] == "lens" and r["aggregation"] == "mean" and r["nsigma"] is None)
    assert full["mean_frob"] != frozen["mean_frob"], "frozen attention must change the Hessians"
    assert path.read_text().splitlines()[0].startswith("attention,skip_first")

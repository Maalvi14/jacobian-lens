# H-Lens Session Handoff

**Purpose:** this document is a record of the working sessions building `hlens`
against `hlens_plan.md` / `research_notes.md`. It exists so work can continue with full
context, without re-deriving decisions already made. Read this before re-reading the plan
docs — it tells you what's already done and *why* things were built the way they were,
which the plan docs alone don't capture.

- **Session 1** (off the Thor, toy model only) built the library modules (§3).
- **Session 2** (on the Jetson Thor) stood up the real Qwen3.5-4B, loaded a pre-fitted
  lens, added the real-model test infrastructure, and ran the first real-model validation
  gate (`functional.py`), which found and fixed a real bug (§4). Its results are folded
  into the sections below; the environment and model state is in §6, next steps in §7.

---

## 0. How this session worked (read this first)

The user drives; the assistant guides. Concretely, for the entire session:

- The assistant explained concepts, proposed designs, and reviewed code — **it did not
  write library code into the repo.** The user typed every line of `hlens/*.py` and
  `tests/*.py` themselves, using Read to check files.
- The user ran every `pytest` invocation themselves and pasted the output back.
- The assistant used `Read`/`Bash`(read-only: `find`, `grep`) freely to inspect files,
  but was corrected twice this session for overstepping into *running* code (an
  inline Python analysis script, and a `python -c` JSON inspector) — the boundary is
  "read files, don't execute exploratory scripts on the user's behalf."
- This document is an explicit exception: the user asked for it to be written directly.

Session 2 followed the same pattern: the assistant showed code in chat (`loading.py`,
`conftest.py`, the real-model tests) for the user to type into the repo, inspected files
and installed-package source read-only, and looked things up online (Hugging Face,
Neuronpedia's GitHub); the user ran every `pytest` command and the torch/CUDA check.

**If continuing this pattern**: keep guiding, not authoring library code.
This handoff doc and any similar written-for-you docs are fine to author directly.

---

## 1. Project goal, in one paragraph

`hlens` (H-Lens) extends the `jlens` reference implementation (Jacobian Lens, a
first-order linear readout of transformer residual streams) with **second-order
(curvature) tooling**: Hessian-vector products, sparse J-space decomposition, write
interventions (steering/ablation/coordinate-swap), and a reliability harness that
gates every curvature number behind validity checks. It's built as a sibling package
that imports `jlens` rather than forking it. Full scientific motivation is in
`hlens/docs/research_notes.md`; the build checklist is `hlens/docs/hlens_plan.md`.

---

## 2. Overall status

| Module | Status | Notes |
|---|---|---|
| `hlens/functional.py` (§1) | ✅ Complete; **validated on real Qwen3.5-4B** | Exact match vs hooked forward, both layer types (session 2) |
| `hlens/loading.py` (new, session 2) | ✅ Complete | `load_hf` / `load_qwen` take `dtype` (default bf16); fp32 copy is the curvature model |
| `hlens/hvp.py` (§2) | ✅ Complete; **validated on real Qwen3.5-4B in fp32 on CUDA** | `rr`≈`fr` to ~1e-6; bf16 measured, not validated (§3.2). `restricted_hessian` / `hvp_norm_form` not yet run on the real model |
| `hlens/jspace.py` (§3) | ✅ Complete, unit-tested; **paper gate passed on real Qwen3.5-4B (bf16)** | Median J-space variance 7–9% at layers 12/16/20; two of the three gate criteria are weaker than they look — see §3.3 |
| `hlens/interventions.py` (§4) | ✅ Complete, unit-tested; **swap reproduced only partially on real Qwen3.5-4B (bf16)** | Causal and specific (34% best vs 2% random control on probe-swap) but below the paper's rates and sensitive to α/band — see §3.4 |
| `hlens/harness.py` (§5) | ✅ Complete (buildable parts), unit-tested | 4 items still unbuilt, see §3.5 |
| `hlens/data.py` (§6) | ✅ Complete, unit-tested | Validated against real shipped JSON files |
| `hlens/experiments/trust_score.py` (§7, first script) | 🟡 Core math done and tested; orchestration not built | Needs real HVPs on the model — this is where we're headed |
| `hlens/experiments/{gain,binding,automatization,remainder_coupling,ignition}.py` | ⬜ Not started | |
| `hlens/vis_ext.py` (§8) | ⬜ Not started | Lowest priority (Phase 5) |

Test files live in `tests/`, one per module: `test_functional.py`, `test_hvp.py`,
`test_jspace.py`, `test_interventions.py`, `test_harness.py`, `test_data.py`,
`test_trust_score.py`, plus `test_real_qwen.py` (real-model smoke checks) and
`conftest.py`. The toy tests all reuse `tests/tiny.py`'s `TinyDecoder` (a 4-layer,
`d_model=8`, CPU-only toy model).

**Real-model test infrastructure (session 2):**
- Real-model tests are marked `@pytest.mark.real_model`. `conftest.py` registers the
  marker and **skips them unless a `-m` expression is given** — so plain `pytest` is the
  fast toy suite, and `pytest -m real_model -s` runs only the Qwen tests.
- They share two session-scoped fixtures: `qwen` (`load_qwen()`, bf16, 9 GB) and
  `qwen_fp32` (`load_qwen(dtype=torch.float32)`, ~16–18 GB), each loaded once per run and
  only if a test requests it. Both skip themselves if CUDA is unavailable.
- `test_functional.py` and `test_hvp.py` hold both the toy tests and the real-model ones
  (`..._qwen` for bf16, `..._qwen_fp32` for fp32). The marker there is **per-test, not
  module-level**, because a module-level `pytestmark` would also mark (and so skip) the
  toy tests in those files.
- `tests/test_real_qwen.py` has a module-level `pytestmark = pytest.mark.real_model`
  (it was briefly commented out; restored, so plain `pytest` is the fast toy suite again).
  `tests/test_jspace.py` follows the per-test-marker convention.
- **Running things: use `uv`.** Everything runs in the uv-managed `.venv`; use
  `uv run pytest ...`. Session 2 used `uv run --no-sync` so uv could not touch the
  environment (it keeps torch pinned to the `+cu130` wheel). Real-model runs:
  `uv run pytest -m real_model -s`.
- Full plain `pytest` on the Thor after the `functional.py` change: **69 passed, 4
  skipped** (the 4 are the Qwen slice tests in `test_functional.py`). One toy test needed
  a fix: `test_eps_sweep_on_quadratic_form_is_flat_everywhere` failed on float32 rounding
  (error ~`1e-3`, amplified by `1/(2·eps)`), so it now runs in float64 with tolerance
  `1e-6`. A small but real data point for the fp32-vs-bf16 decision in §7.

**What has touched the real model so far**: `functional.py` (slice-vs-hooked forward, bf16
and fp32), `hvp.py` (`hvp_rr` / `hvp_fr` / `hvp_fd`, scalar functions only, fp32 validated
and bf16 measured), `jspace.py` (paper gate, §3.3), `interventions.py` (swap sweeps, §3.4),
and the load/lens smoke checks. `harness.py`, `restricted_hessian`, `hvp_norm_form`, and
`trust_score.py` have **not** been run on the real model.

Last runs (Thor, `uv run --no-sync`): `pytest -m "not real_model"` → **66 passed**
(23 real-model tests deselected). The 23 real-model tests were each run and passed
individually: 8 `functional` slice tests (bf16 + fp32), 9 HVP tests, 3 smoke tests, 1
`jspace` gate, 2 swap tests (~5 min for the two swap tests, ~10 min for the `jspace` gate).
They were not run as one combined `-m real_model` session after the last edits, so
re-run `uv run pytest` and `uv run pytest -m real_model -s` before trusting this.

---

## 3. Module-by-module detail

### 3.1 `hlens/functional.py`

Pure layer-slice wrapper so `torch.func` transforms have something to operate on.

- **`_BlockKwargsCapture`** (private class) — a forward-**pre**-hook context manager
  (mirrors `jlens.hooks.ActivationRecorder`'s post-hook pattern, but captures block
  *inputs* instead of outputs). Registers with `with_kwargs=True`. Asserts it only
  fires once per capture (`assert self.kwargs is None` inside the hook). No
  `try`/`except` in `__enter__` — deliberately: unlike `ActivationRecorder`, this
  hooks exactly one block, so there's no partial-registration state to unwind if
  registration fails (that's *why* `ActivationRecorder` needs the try/except — it
  loops over multiple blocks — and this doesn't).
- **`_capture_block_kwargs(model, block_index, seq_len)`** — runs one throwaway
  `model.forward(dummy_ids)` (token id `0`, batch size 1, length `seq_len`) under
  `_BlockKwargsCapture` and `torch.no_grad()` to steal the real
  `position_embeddings`/`attention_mask`/`position_ids` HF's decoder stack computes
  for that layer. **Assumption, stated in-code**: these kwargs are
  content-independent (only `seq_len`-dependent) for a padding-free causal decoder.
  Trivially true for `TinyDecoder` (its blocks take just `hidden`); **verified in
  session 2 on real Qwen3.5-4B** for both layer types (see Validated below), including
  that kwargs captured from a zeros-token throwaway pass reproduce a real-token forward.
  (`HFLensModel.forward` calls the text module with `use_cache=False`, so no cache
  object ends up in the captured kwargs.)
- **`make_slice_fn(model, l_start, l_end)`** — returns a pure closure `h -> h'`
  running blocks `l_start+1..l_end`. Caches captured kwargs by **`(seq_len,
  block_index)`** in a closure-local dict: each block gets its *own* kwargs, captured
  the first time it is needed (see the session-2 bug in §4 — one shared kwargs dict is
  wrong on hybrid models). Bounds check: `0 <= l_start < l_end < model.n_layers` (strict
  `<` on the upper bound — this was a real bug we caught and fixed: the original `<=`
  allowed `l_end == n_layers`, which Python's forgiving list-slicing would silently
  accept and silently produce the wrong thing instead of erroring). Tuple-vs-tensor block
  output handled with `out if torch.is_tensor(out) else out[0]`.
- **`make_output_fn(model, l_start, contraction)`** — `make_slice_fn` through the
  *last* block (`model.n_layers - 1`), composed with a `contraction` callable
  (currently just `Callable[[Tensor], Tensor]`, e.g. `model.unembed`).

**Validated (toy)**: `test_make_slice_fn_matches_hooked_forward` and
`test_make_output_fn_matches_hooked_forward` both compare against a real
`ActivationRecorder`-hooked forward pass on `TinyDecoder`, tight tolerance (`1e-5`).
Proves the block-loop/tuple-unwrap/cache mechanics, not kwargs-threading (`TinyDecoder`
has no such kwargs).

**Validated (real model, session 2)**: `test_slice_fn_matches_hooked_forward_qwen` in
`tests/test_functional.py`, parametrized over slices `(2,3)` (a single full-attention
layer), `(3,4)` (a single linear-attention layer), `(3,12)` (mixed), `(0,30)` (whole
stack). Prompt = the boot prompt, model in bf16, `rtol=atol=0`. **All four give
`max |diff| = 0` exactly.** Background on the architecture this exercises: Qwen3.5-4B is
a *hybrid* — `layer_types` repeat 3× `linear_attention` (Gated DeltaNet) then 1×
`full_attention` (full-attention at layers 3, 7, 11, …; `full_attention_interval=4`);
decoder layers return a plain tensor, not a tuple; for an unpadded batch the
linear-attention layers receive `attention_mask=None` while full-attention layers receive a
real causal-mask tensor.

**Resolved (session 2): first-call capture inside a `torch.func` transform.** `slice_fn`
runs inside `torch.func.jvp`/`grad`, and on a cache miss it runs a throwaway model forward
from inside that transform. That did turn out to be a bug: the captured rotary `cos`/`sin`
were wrapped at the transform's level, got cached, and "escaped" (`INTERNAL ASSERT FAILED
... escaped?` in `apply_rotary_pos_emb`) when reused under a later transform — e.g.
`hvp_fr` as the first call, then `hvp_fd`. Fix: `_capture_block_kwargs` runs its forward
inside `torch._C._DisableFuncTorch()` so captured tensors are always plain. Caveats:
that is a private PyTorch API (present in torch 2.12.0; a future torch could change it,
which would show up as the regression test failing), and the diagnosis was inferred from
the failure pattern plus the fix working, not from a separate warm-up experiment.
Regression test: `test_hvp_fr_first_call_does_not_poison_kwargs_cache` in
`tests/test_hvp.py`.

### 3.2 `hlens/hvp.py`

The HVP core. Order built: `Contraction`/`make_scalar_fn` → `hvp_rr` → `hvp_fr` →
`hvp_fd` → `restricted_hessian` → `hvp_norm_form` (added later, for `trust_score.py`).

- **`Contraction = Cotangent | Logit | Norm`** (tagged union of three frozen
  dataclasses) — `Cotangent(c)` and `Logit(token_id)` reduce a vector-valued `f` to a
  scalar; `Norm()` is a marker with **no scalar form** (see `hvp_norm_form` below).
- **`make_scalar_fn(f, contraction)`** — pattern-matches the `Contraction`, returns
  `lambda h: dot(c, f(h))` for `Cotangent`, `lambda h: f(h)[..., t]` for `Logit`, and
  `raise NotImplementedError(...)` for `Norm` (by design — `Norm`'s actual
  computation is `hvp_norm_form`, a completely different code path, not something
  `make_scalar_fn` can produce).
- **`hvp_rr(f, h, v, *, create_graph=False)`** — reverse-over-reverse. Two nested
  `torch.autograd.grad` calls (`create_graph=True` on the first). **Requires**
  `h.requires_grad == True` (asserted, with a message pointing at
  `ActivationRecorder`'s `start_graph_at`). Reference implementation — works
  directly on hook-rooted graphs, no `functional.py` needed.
- **`hvp_fr(f, h, v)`** — forward-over-reverse, `torch.func.jvp(torch.func.grad(f), (h,), (v,))[1]`.
  Production path. **Does not** need `h.requires_grad` — `torch.func` is
  fully functional, no leaf/graph machinery.
- **`hvp_fd(f, h, v, eps=1e-3)`** — central finite difference on the *gradient*
  (`(grad_f(h+εv) - grad_f(h-εv)) / 2ε`), not on `f` directly — this is deliberately
  more numerically stable than double-finite-differencing `f` itself.
- **`RestrictedHessian(H, asymmetry)`** / **`restricted_hessian(f, h, directions, *,
  hvp_fn=hvp_fr)`** — `H_ij = v_i^T H v_j` via `k` HVPs (not `k²`): one HVP per
  column `j`, then one `directions @ Hv_j` matmul gives the whole column at once.
  Symmetrized after the full loop; `asymmetry` is the pre-symmetrization
  `‖H - Hᵀ‖_max`, a free estimator-error diagnostic.
- **`hvp_norm_form(f, h, delta_h)`** — the "no contraction" case for `Norm`. A
  **JVP of a JVP**, not a grad-then-jvp: differentiates the JVP function itself
  (`jvp_fn(x) = J(x)·δh`) in direction `δh`, at `h`. Works on genuinely
  vector-valued `f` (no scalar reduction). **Critical numerical detail, verified
  by hand-derivation and by test**: this returns the **full** contraction
  `H[δh, δh, :]`, i.e. `δhᵀHδh` as a vector — **not** `½δhᵀHδh`. Any caller wanting
  the `½` factor (e.g. the trust score formula) must apply it themselves. This was
  caught via the `f(h)=h²` closed-form test — output came out `2·δh²`, not `δh²`,
  confirming the factor-of-2 and preventing a silent trust-score bug.

**Validated (toy only)**: every estimator against an exact analytic quadratic-form ground
truth (`f(h) = 0.5 hᵀAh`, constant Hessian `A`); `hvp_rr` additionally against a
hook-rooted `TinyDecoder` activation; `hvp_rr`/`hvp_fr` cross-checked against each
other; `estimator_agreement`-style tight-tolerance test (in `harness.py`, see below)
shows `rr`/`fr` agree near machine precision while `fd` has real, expected
discretization error. `hvp_norm_form` validated on a linear map (exact zero) and
`h²` (exact nonzero closed form). All of this ran on CPU.

**Validated (real model, session 2, CUDA)** — `tests/test_hvp.py`, boot prompt, unit-norm
random `v`, scalar `f` read at the last position (`torch.func.grad` needs a 0-dim output,
so a raw `Logit(t)` contraction on `[1, seq, vocab]` output cannot be used directly; the
`logit-from-20` case wraps the output fn so `Logit` applies to a 1-D `unembed` result).
Cases: single full-attention layer `(2,3)`, single linear-attention layer `(3,4)`, mixed
`(3,12)`, and `(20, None)` = through the last block to one logit (the model's own top
next token). This also answers plan §2's "run it on the Thor" check: `torch.func` + nested
`grad` work on this GPU build, and **double backward works through the plain-PyTorch
Gated-DeltaNet fallback** (the biggest risk going in).

*fp32 model (`qwen_fp32`)* — relative error against `hvp_fr`:

| case | `rr` | `fd` eps=1e-3 | `fd` eps=1e-2 |
|---|---|---|---|
| full-attn layer | 4e-7 | 5e-3 | 5e-4 |
| linear-attn layer | 6e-7 | 6e-3 | 6e-4 |
| mixed (3→12) | 2e-6 | 1.3e-2 | 1.3e-3 |
| logit from 20 | 1e-6 | 5.6e-2 | 6.8e-3 |

`rr` and `fr` agree to float32 precision (test asserts `< 1e-4`; observed max 1.7e-6).
`fd` is rounding-noise-limited in fp32: *worse* at the smaller `eps`, better at 1e-2 —
the same effect that broke the toy eps-sweep test. Expect the harness's `eps` sweep to
matter on the real model.

*bf16 model (`qwen`)* — relative error against the fp32 reference (one prompt, one seed;
a first look, not a calibrated noise floor):

| case | `hvp_fr` bf16 | `hvp_fd` bf16 (eps=1e-2) |
|---|---|---|
| full-attn layer | 1.2% | 366% |
| linear-attn layer | 1.0% | 429% |
| mixed (3→12) | 3.5% | 1475% |
| logit from 20 | 1.6% | 5142% |

**Precision decision (made): curvature work uses the fp32 model.** bf16 `hvp_fr` is
usable for a rough read (1–3.5% here) but bf16 finite differences are unusable (errors of
several hundred percent or more), so the `eps` sweep, estimator agreement and anything
that divides by a small step must run in fp32. The bf16 test
(`test_hvp_bf16_noise_vs_fp32_qwen`) is a measurement and only asserts finiteness.

**Not yet run on the real model**: `hvp_norm_form` (needed by `trust_score`),
`restricted_hessian` (note it allocates its result with `torch.zeros(k, k)` in the default
dtype/device — check that behaves with CUDA/fp32 inputs), and vector-valued `f`.

### 3.3 `hlens/jspace.py`

Sparse decomposition. **Naming note**: the plan calls this `gradient_pursuit`; it
was renamed to **`sparse_pursuit`** mid-session because the actual algorithm
implemented is Non-Negative Orthogonal Matching Pursuit (NN-OMP — select, then full
NNLS refit over the whole support each iteration), not the different, cheaper
"Gradient Pursuit" algorithm (Blumensath & Davies) the plan's function name refers
to. The plan's *prose* description ("select atom with largest positive correlation,
refit nonnegative coefficients over the selected set") matches NN-OMP, not gradient
pursuit — the name in the plan is likely a slight misnomer, and we went with prose
over name.

- **`lens_vectors(lens, model, layer, token_ids, *, h_ref=None, dim_batch=8)`** — the
  "J-lens vector" `v_t = J_l^T @ u_t` for requested tokens. `u_t` is computed via
  **autograd on `model.unembed`** at a synthetic reference point (default
  `h_ref = torch.ones(d_model)`), *not* by reaching into the model's LM-head weight
  matrix directly — a deliberate choice to stay protocol-agnostic (`LensModel`
  doesn't expose raw weights) and consistent with how `jlens` estimates everything
  via autodiff rather than weight-peeking. **`h_ref` can't be zero** — `RMSNorm`/
  `LayerNorm` divides by the input's own norm, so `h_ref=0` gives `0/0`.
  `token_ids` is **required**, not optional-defaulting-to-all-vocab (raises if you
  try) — the full `[n_vocab, d_model]` matrix is ~1.5GB per layer for Qwen3.5-4B,
  and silently computing that by default was judged too dangerous a default.
  Batches multiple tokens per backward pass via one-hot cotangents, mirroring
  `jlens.fitting.jacobian_for_prompt`'s own batching trick.
  **Important validated caveat**: `lens_vectors`' correctness was confirmed via a
  *ratio-consistency* test, not a direct value match — `⟨v_t, h⟩` and the real lens
  logit for token `t` differ by a **shared, h-dependent scale factor** (the
  "data-dependent normalization factor" the paper itself names), because
  `RMSNorm`/`LayerNorm` is scale-invariant while a dot product is linear. The test
  checks that this scale factor is the *same* across different tokens for one `h`
  (confirms the transpose orientation is right), not that raw values match.
  **Real-model note**: `HFLensModel.unembed` casts its input to the LM head's dtype
  (bf16 on the current model), so `lens_vectors` on the real model is computed through a
  bf16 head unless the model is loaded in fp32 (see the precision decision in §7).
- **`sparse_pursuit(h, dictionary, *, k=25, tol=1e-6)`** — NN-OMP. Requires
  `dictionary` rows to be **unit-norm** (raises `ValueError` naming the actual max
  deviation otherwise — matching-pursuit correlation selection is scale-sensitive).
  Uses `scipy.optimize.nnls` for the refit step — **`scipy` is now a real declared
  dependency** (added via `uv add scipy`), not just an incidentally-installed
  package. Stops early on `correlations[best] <= 0` (no atom left with positive
  correlation) or `residual.norm() < tol`.
- **`Decomposition(indices, coeffs, jspace_component, remainder, frac_variance)`** /
  **`decompose(h, dictionary, *, k=25)`** — wraps `sparse_pursuit`.
  `frac_variance = ‖jspace_component‖² / ‖h‖²` (the *J-space's* share of variance,
  expected **small** — ~10% at mid layers per the paper — most of `h`'s variance
  lives in `remainder`, which is what carries most of the causal swap effect
  per the paper's finding). `remainder` is exactly what the plan calls "2F's `v_j`".
  NNLS's optimality (KKT) guarantees `remainder ⊥ jspace_component` exactly, so
  `frac_variance` and `1 - ‖remainder‖²/‖h‖²` agree — verified by test to `1e-6`.
- **`k_variation(h, dictionary, *, k_values)`** — thin wrapper, `{k: decompose(...)
  for k in k_values}`. Deliberately minimal ("a one-flag rerun, not a rewrite" per
  the plan). **Structural fact worth knowing**: because `sparse_pursuit`'s loop only
  depends on the current residual (not on `k` itself), increasing `k` can *never*
  change earlier selections — smaller-`k` results are an exact list-prefix of
  larger-`k` results. This is a guarantee, not something that needs probabilistic
  testing, and is tested via exact list equality on a prefix, not a set/fuzzy check.
  **"Seed" variation from the plan's "survives pursuit seed and k variation" is
  NOT implemented** — `sparse_pursuit` is fully deterministic, no RNG anywhere in
  it, so there's no seed to vary. This was a deliberate scope decision (Option 2 of
  two discussed): don't invent a noise-injection mechanism to manufacture
  "seed-ness" where none exists in the actual algorithm; document the gap instead.

**Validated**: `lens_vectors` (ratio-consistency, 3 trials), `sparse_pursuit` (exact
recovery on an orthonormal dictionary via QR decomposition — zero selection
ambiguity), `decompose` (exact `frac_variance` + Pythagorean cross-check on a
3-component orthonormal case with `k=2` forcing a real, predictable remainder),
`k_variation` (exact prefix consistency).

**Paper-reproduction gate (real model, session 2)** — `test_jspace_paper_gate` in
`tests/test_jspace.py`; bf16 `qwen` model + Neuronpedia lens. Setup: 50 WikiText-103
prompts (`jlens.examples.load_wikitext_prompts`), 128 tokens each, 3 evenly spaced
positions per prompt chosen from `jlens.fitting.valid_position_mask` (so the first
positions — attention sinks — and the last are excluded, as in the lens fit), layers 12 /
16 / 20, `k=25`, dictionary = all 248,320 token J-lens vectors, unit-normalised (rows with
norm < 1e-6 dropped). The layer-independent unembed directions `u_t = d logit_t / d h_final`
are computed once (via `lens_vectors` with an identity lens — 19 s; full-vocab
`lens_vectors` is O(vocab²) through autodiff, ~300 TFLOP) and each layer's dictionary is
`u @ J_l`. 150 positions per layer; ~165 s per layer (~1.1 s per pursuit — memory-bound
scans of a 2.5 GB dictionary); whole test 623 s.

| layer | median `frac_variance` | p90 | max | atoms selected | "active" (≥5% of max coef) |
|---|---|---|---|---|---|
| 12 | 0.080 | 0.108 | 0.130 | 25.0 | 24.7 |
| 16 | 0.072 | 0.096 | 0.112 | 25.0 | 24.7 |
| 20 | 0.094 | 0.118 | 0.145 | 25.0 | 24.9 |

The test asserts median ≤ 0.10 at each layer and passes. **Read the result with these
caveats (all discussed, none hidden):**
- **Random-dictionary control explains *more* variance than the J-lens dictionary.** At
  layer 16, 25 random unit atoms (same atom count, 40 positions) capture a median 0.151,
  vs 0.072 for the J-lens atoms. So "≤10%" is a correct reproduction of the paper's
  number but is **not by itself evidence that J-lens atoms align with the residual** — the
  real atoms are likely more redundant (clusters of near-duplicate tokens), so greedy
  pursuit gains less per atom. The paper's claim is about *causal* effect, not variance.
  Control was one layer, 40 positions, one seed.
- **"~25 meaningfully active atoms per position" is not actually tested.** Pursuit always
  selects exactly `k` atoms (it only stops early if no atom has positive correlation, which
  never happens here), so 25.0 is by construction. To test the criterion we would need to
  vary `k` or look at how quickly explained variance saturates.
- **Layer 20 is close to the line** (median 0.094 vs the 0.10 threshold; p90 0.118 and max
  0.145 exceed it). Only the median is asserted.
- **Boot prompt, top atoms by coefficient (qualitative):** at position −2 (` boot`) the
  atoms are sensible at all three layers — L12 `意大利`, ` Boots`, ` Ita`, ` heel`,
  `/boot`; L16 `/boot`, ` Ital`, `意大利`, ` Naples`; L20 `-shaped`, ` Italia`, ` Shoe`,
  ` ITAL`, ` heel`. At position −1 (` is`) L12/L16 are noisy (` US`, `boa`, stray symbols)
  and only L20 shows ` Italy`, ` Fiat`, ` currency`, ` Pago`, ` Macron`. **No `euro`/`lira`
  in the top 8**, so the plan's "lira/euro-ish tokens" expectation is only loosely met.
  Several top atoms at every layer are junk (stray CJK characters, `\u`, `；`).
- **Not tested:** the `h_ref = ones` caveat of `lens_vectors` — RMSNorm's Jacobian at
  `ones` projects out the all-ones direction, so the vectors differ slightly from raw
  `W_U` rows. If a later result is unexpectedly far from the paper, check this first.

### 3.4 `hlens/interventions.py`

Write operations — the "sibling" of `jlens.hooks.ActivationRecorder`.

- **`ActivationEditor(blocks, at, edit_fn)`** — forward-hook (not pre-hook) context
  manager that *replaces* a block's output via `edit_fn`. Unlike
  `_BlockKwargsCapture`, this **does** need the `try`/`except` cleanup in
  `__enter__` (mirrors `ActivationRecorder` fully) because it genuinely loops over
  multiple `at` indices — partial-registration failure is a real scenario here.
  One shared `edit_fn` is applied identically at every hooked block. **Correction
  (session 2):** that is *not* enough for lens-vector swaps — `J_l`, and so every lens
  vector, differs per layer, so a single shared `edit_fn` cannot carry layer-specific
  `v_s`/`v_t`. The swap tests work around it at the call site with **one
  `ActivationEditor` per layer, stacked with `contextlib.ExitStack`** (`_pair_edits` in
  `tests/test_interventions.py`); no library change was needed. Tuple-output rebuild:
  `(edited, *output[1:])`. Qwen3.5's decoder layers return a plain tensor, so the tensor
  branch is the one exercised on the real model (the swap tests run through it).
- **`steer(h, v, alpha)`** — `h + alpha * v`.
- **`ablate(h, v)`** — projects out the `v`-component: `h - (h·v̂)v̂`, `v̂ = v/‖v‖`.
  Uses `(h * v_hat).sum(dim=-1, keepdim=True)` (not `torch.dot`) so it works on
  batched `[..., d_model]` tensors, not just flat vectors.
- **`coordinate_swap(h, v_s, v_t, *, alpha=1.0)`** — exact §2.5 formula, confirmed
  against the paper's own definition mid-session (the user supplied the precise
  spec: `σ` is a **coordinate swap/permutation**, not a sigmoid — `σ(c) = (c_t,
  c_s)` for `c = (c_s, c_t)`). Implementation: `V = stack([v_s, v_t], dim=1)`,
  `c = h @ pinv(V).T`, `c_swapped = c.flip(-1)` (reversing a 2-element last dim
  *is* the swap — a neat one-liner for this specific case), `delta = alpha *
  (c_swapped - c)`, return `h + delta @ V.T`. Row-vector convention throughout,
  matching `JacobianLens.transport`'s own convention.

**Validated**: `coordinate_swap` is a true involution at `alpha=1.0` (swap twice ==
identity, tight tolerance), and the orthogonal complement (built via
`torch.linalg.qr(V, mode="complete")`, columns 2+ guaranteed ⊥ to `span(v_s,v_t)`)
is provably untouched by the swap.

**What the paper actually says about swaps (read from the PDF in `hlens/docs/` in
session 2 — earlier notes had this second-hand):**
- §2.5 definition is identical to ours: `V = [v_s v_t]`, `c = V†h`,
  `h_patched = h + V(σ(c) − c)`, σ swaps the two entries "optionally scaled by α". Applied
  at **every token position** over "the full workspace layer range".
- **Workspace range ≈ L38–L92 on the paper's 0–100 depth scale** (sub-ranges: early
  L38–54, late L75–92; L38–79 used in some analyses). On Qwen's 32 layers (indices 0–31)
  that maps to **about layers 12–28** *if the percentages transfer* — an assumption; the
  paper identifies the range per model (CKA block structure etc.) and only for Claude.
- **Only Claude models are reported** (Haiku 4.5, Sonnet 4.5, Opus 4.5/4.6); no open-weights
  numbers. Top-1 flip rates: **54% / 70% / 70%** on 50 two-hop prompts (Haiku / Sonnet /
  Opus 4.5); on the 90-prompt set, **60%** for raw J-lens vector swaps and **61%** for the
  probe's J-space component (**28%** for the non-J-space remainder; 6% with J-space
  coordinates clamped); flexible-generalization experiment **76/192 at α=1 → 101/192 at
  α=2**; Figure 68: **country facts 42/48 off-diagonal cells reach rank 1, number relations
  0/48**, months/animals in between. Failures concentrate where the source concept is only
  weakly present ("workspace loading" = cosine of the residual with the lens vector).
  Multi-token concepts degrade sharply (Fig 62) — we restrict to single-token entities.
- **Not stated:** the denominator (all prompts vs baseline-correct), and whether swap
  vectors are unit-norm or raw (§3.1's "add an *equal-magnitude* projection" weakly favours
  unit-norm). `research_notes.md` H1's "country facts ≈ always succeed; number relations ≈
  never at α=1" is consistent with Fig 68 (87.5% / 0%); its "recovering at α=2" is **not**
  verified from the text (only the pooled 76→101 is).

**Real-model swap tests (session 2)** — `tests/test_interventions.py`, bf16 `qwen`,
unit-norm vs raw lens vectors swept, one editor per layer, swap at all positions. Layer
band is a parameter (paper's Qwen band unknown): swept `(12,20)`, `(12,24)`, `(12,28)`.
Lens vectors for the swap are `u_t @ J_l` with `u_t` from `_unembed_rows` (identity-lens
trick, as in the `jspace` gate).
- `test_multihop_swap_flips_currency` — boot prompt, Italy → Japan, strict top-1 flip.
  **The prompt must end in "…is the"**: with a bare "is" the model answers ` the`/` called`,
  with "is " it answers digits (`'1','5','2'`), so neither can show a currency flip.
  Baseline top-5 `[' Euro',' euro',' Italian',' pound',' dollar']`. **11 of 36 configs
  (3 bands × 6 α × 2 scalings) flip the top-1 to ` yen`/` Japanese`**; on `(12,20)` every α
  in 0.7–1.25 flips; wider bands flip only at α=1.25 (and not at all for raw vectors on
  `(12,28)`); random-direction controls (3 seeds) leave the answer at baseline. At α ≥ 1.5
  the model says ` Italy`/`意大利` (overshoot — the Italy coordinate is negated).
- `test_probe_swap_sweep` — `data/experiments/probe-swap.json`, 88/90 items have
  single-token `intermediate` and `swap_to`, **53/88 baseline-correct** (denominator below).
  Hit = top-1 equals the first token of `swap_answer` (with or without leading space).

  | band (raw vectors) | α=0.7 | 0.9 | 1.0 | 1.25 | 1.5 | 2.0 |
  |---|---|---|---|---|---|---|
  | L12–20 | 8% | 9% | 17% | **34%** | 4% | 2% |
  | L12–24 | 9% | 11% | 21% | 9% | 2% | 0% |
  | L12–28 | 9% | 11% | 17% | 4% | 0% | 0% |

  Unit-norm vectors are within a couple of points (best 32%, i.e. 17/53 on L12–20 at α=1.25).
  **Best: L12–20, raw, α=1.25 → 18/53 (34%); random-direction control 1/53 in each of 3
  seeds.** Category pattern at the best config: `city-language` 3/3, `city-capital` 4/6,
  `city-continent` 2/3, `language-capital` 2/4, `river-capital` 1/1, but `multihop` 1/11
  and `element-*` 0/5 — country-style facts do best, as in the paper's Fig 68.
  Assertions (set from the observed run, with margin): best rate ≥ 25% and ≥ 5× the random
  control. They check "clearly above chance", not agreement with the paper.

**How to read this (caveats):**
- The swap is **causal and specific** (random directions at the same strength: 0–1 hits),
  and wired correctly end to end (editor + `lens_vectors` + `coordinate_swap`).
- It is **below the paper's numbers** (34% vs 54–70% on much larger Claude models) and the
  best config was **tuned on these same 53 items**, so 34% is a diagnostic that the regime
  is reachable, not an unbiased rate. The denominator (baseline-correct) is our choice.
- **Narrow α window, sharp transition.** On `(12,20)`, "moved off the original answer"
  jumps from 14/53 at α=1.0 to 43/53 at α=1.25 (both scalings), and hits collapse above that
  (4% at 1.5). The paper's α=2 *helps*; ours destroys the answer (all 53 move off it, none
  land on the target; the model tends to output the entity name). Unexplained — candidates:
  model size, the band mapping, the untested `h_ref = ones` caveat in `lens_vectors`
  (§3.3), or the swap hitting position 0 (the attention sink), which the paper also does.
- My intermediate "total strength" hypothesis (wider band needs lower α) is **refuted**:
  `(12,28)` at α=0.7 gives 9%.
- Raw vs unit-norm vectors barely matter, so the paper's unstated choice is not a likely
  cause of the gap.
- Worth keeping in mind for H6 (ignition as a curvature event), as a hypothesis only: the
  swap outcome is sharply nonlinear in α, which is what a curvature-based trust score would
  have to explain.

### 3.5 `hlens/harness.py`

Reliability harness. Everything buildable without hardware is done.

- **`eps_sweep(f, h, v, eps_grid)`** — `torch.stack([hvp_fd(f, h, v, eps=e) for e in
  eps_grid])`. Deliberately minimal — single `(h,v)` case; batching over
  positions/directions is a future outer-loop wrapper, not built into this.
- **`PlateauResult(has_plateau, eps_range, representative_eps)`** /
  **`find_plateau(curve, eps_grid, *, tol=0.05, min_run_length=2)`** — finds the
  **longest contiguous run** of small (`< tol`) relative diffs between consecutive
  curve points. `has_plateau=False` with explicit `None` fields when no run of at
  least `min_run_length` consecutive small diffs exists — "flagged, not reported"
  per the plan, verified by a synthetic no-plateau curve (constant-ratio doubling,
  where every consecutive diff is provably exactly `1.0`).
- **`EstimatorAgreement(pairwise, usable)`** / **`estimator_agreement(f, h, v, *,
  tol=0.05)`** — builds its **own** independent leaf copy of `h`
  (`h.clone().requires_grad_(True)`) for `hvp_rr`, uses `h.detach()` for `hvp_fr`/
  `hvp_fd` — centralizes the different-estimators'-different-preconditions handling
  so callers don't need to know about it. Three pairwise relative errors
  (`rr`-`fr`, `rr`-`fd`, `fr`-`fd`); `usable = all(err < tol)`.
  **Verified discriminating, not just always-True**: at an absurdly tight
  `tol=1e-8`, `rr`/`fr` (both exact autodiff) still agree to `<1e-6`, while `fd`
  (genuine finite-difference discretization error) trips `usable=False` — proves
  the boolean logic actually distinguishes estimator quality, not just passes
  trivially.
- **`MSHVPResult(single_step, multi_step, reliability)`** / **`ms_hvp(f, h, v, *,
  n_steps=4, hvp_fn=hvp_fr)`** — multi-step HVP correction. Mechanism: a **left-
  endpoint Riemann sum** approximating `∫₀¹ H(h+tv)·v dt = ∇f(h+v) - ∇f(h)` exactly
  (fundamental theorem of calculus). `single_step` is the naive 1-point
  approximation; `multi_step` walks `n_steps` sub-steps, recomputing local
  curvature at each. `reliability` is the relative divergence between the two.
  **Note on convergence rate**: this is a left-Riemann sum, `O(1/n_steps)`
  convergence, not the faster `O(1/n_steps²)` a midpoint rule would give — worth
  knowing if a future validation test's tolerance seems to need surprisingly large
  `n_steps` to pass.
  **This is structurally identical to how flow-matching models integrate a
  learned velocity field at sampling time** (discretize a straight-line path,
  Euler-style accumulate a per-step quantity) — same numerical machinery, applied
  to a curvature-correction term instead of a learned velocity, no learned/trained
  component here at all. Also structurally connects to the plan's `ignition.py`
  (H6) interpolation protocol, which studies curvature along the *same kind* of
  straight-line path (`(1-α)e_B + αe_A`) — flow-matching's canonical linear
  interpolant, used here as a fixed diagnostic probe, not a generative target.
- **`ValidityResult(valid, taylor_radius, perturbation_size, validity_score)`** /
  **`validity_score(f, h, v, *, eps_grid, plateau_tol=0.05)`** — composes
  `eps_sweep`+`find_plateau`: probes a *unit* direction `v/‖v‖` across `eps_grid`,
  takes the plateau's upper edge as an estimated "Taylor radius," compares against
  the real `‖v‖`. `score = taylor_radius / perturbation_size`; `valid = score >=
  1.0`. **Known, documented limitation** (this is explicitly the plan's "proto-2C"
  — not meant to be final): the reported `taylor_radius` is capped by whatever
  `eps_grid` you happened to test, not a true intrinsic property of `f` — for an
  exactly-quadratic `f` (infinite real Taylor radius), this proto method still
  reports a finite radius bounded by the grid's max. A future refinement
  (`experiments/trust_score.py`'s eventual productionized version) should probably
  dynamically extend the grid rather than use a fixed one.

**Not built**: `randomize_model` (needs `hlens.loading.load_hf()` for a *fresh* raw
copy — never randomize the session fixture's model — plus a small `jlens.fit` re-run on
the randomized model), `null_distributions` (needs a real data pipeline), `ablation_table`
(needs the above), `benchmark` (`nvpmodel -m 0` + `jetson_clocks`; the Thor is now
available), the early-averaging test, the pathological stress test.

### 3.6 `hlens/data.py`

Typed loaders for the shipped JSON data. **Important path note**: the real files
live at the **top-level** `data/experiments/*.json` and `data/evaluations/*.json`
— **not** `jlens/data/evaluations` as `research_notes.md` §5 mistakenly says
(`jlens/data/` only has `README.md`, `slice_vis.html`, `blackmail.json`). This is a
stale reference in the research notes doc itself, not a code issue — `data.py`
correctly points at the real top-level paths, matching `hlens_plan.md`'s own
(correct) references.

- **`LensEvalItem(name, prompt, intermediates, target=None)`** /
  **`load_lens_eval(path)`** — one shared loader for **all six**
  `lens-eval-{multihop,multilingual,poetry,order-ops,association,typo}.json` files,
  confirmed they share one identical schema. `target` genuinely optional — present
  in multihop/multilingual/order-ops, absent in poetry/association/typo (verified
  against real files, not assumed from the README prose alone).
- **`ProbeSwapItem(name, category, prompt, intermediate, answer, swap_to,
  swap_answer)`** / **`load_probe_swap(path)`** — for `probe-swap.json` (90
  two-hop factual prompts, the swap-failure set `trust_score.py` needs).
- **`IgnitionConfig(countries_12, alt_words, ctx_templates, noun_ctx_templates,
  idiom_pairs, scrambled_pairs)`** / **`load_ignition(path)`** — deliberately a
  **generative config loader, not a flat-trial loader**: `ignition.json` doesn't
  store literal trials, it stores ingredients (word lists, templates, pairs) that
  `ignition.py` (not yet built) will combinatorially expand into trials (every
  `carrier × pair × α`). Building that combinatorial trial list is scoped to the
  future experiment script, not `data.py`. Pairs converted to `tuple[str,str]`
  (not left as `list[str]`) for immutability/semantic clarity.
- **`CandidatePool(name, pool, proto)`** / **`CapacityConfig(block_families,
  targets_per_family, candidate_pools)`** / **`load_capacity(path)`** — same
  generative-config category as ignition. First nested-dataclass case in the file.

**Validated**: all four loaders against the actual real shipped JSON files (not
synthetic data) — specific real values checked, e.g. `probe-swap.json`'s first item
(`amazon-language`), `ignition.json`'s `idiom_pairs[0] == ("bread","butter")`,
`capacity.json`'s `candidate_pools[0].proto == ["names"]`.

**Priority note for the remaining experiment scripts** (from `research_notes.md`
§5): `trust_score.py` needs `probe-swap.json` (built, first priority); `binding.py`
needs 3 of the 6 `lens-eval-*` files (built); `ignition.py` needs `ignition.json`
(built, but scheduled last); `capacity.json` only feeds the **secondary** H9
hypothesis — lowest priority of the four.

### 3.7 `hlens/experiments/trust_score.py`

First experiment script (2C / H1), per the plan's "C → B/E → D → F → A" ordering.
Core math is done and tested; the actual experiment orchestration (loop over
`probe-swap.json`, compute τ per case, correlate/AUC against real swap success, null
overlay) is not built. The model and lens it needs are now available (§6), but it also
depends on the real-model HVP path being validated first (§7).

- **`swap_delta(h, v_s, v_t, *, alpha=1.0)`** — `coordinate_swap(h, v_s, v_t,
  alpha=alpha) - h`. The literal perturbation the swap intervention applies.
- **`trust_score(f, h, delta_h)`** — `τ = ‖0.5 * hvp_norm_form(f,h,delta_h)‖ /
  ‖jvp(f,h,delta_h)‖`. **The `0.5` factor is applied explicitly here** — recall
  `hvp_norm_form` returns the *full* contraction, not the half — this was a real
  bug risk caught and fixed before it ever shipped.

**Validated**: `swap_delta` against a direct `coordinate_swap` difference;
`trust_score` against an independent hand-derived closed form on `f(h)=h²`
(closed-form numerator `‖δh²‖`, closed-form denominator `‖2h⊙δh‖`).

**Protocol to use for the swaps (decided from §3.4):** layer band `(12, 20)`, `α = 1.25`,
raw (unnormalised) lens vectors, one editor per layer, all positions, bf16 model for the
swap and fp32 for the HVPs. Expect roughly a third of the 53 baseline-correct items to
succeed, so there will be both outcomes to explain, but the α/band choice came from a sweep
on these same items — say so in the writeup, and consider reporting τ's AUC at α=1.0 as
well (17% success) as a robustness check.

**What's left to build** (needs the real model + lens, now available):
1. Load probe-swap cases via `load_probe_swap`.
2. For each case: run the model, get real `h` at the relevant layer/position, get
   `v_s`/`v_t` via `lens_vectors` (source = `intermediate`, target = `swap_to`),
   compute `delta_h = swap_delta(...)`, compute `τ` via `trust_score(...)`.
3. Determine actual swap success (does the model's real greedy output match
   `swap_answer`?) — needs a real forward pass, not just `τ`.
4. Correlate `τ` against success/failure (AUC, per the plan).
5. Null overlay — needs `harness.py`'s `randomize_model` (not built) or some other
   null construction.

---

## 4. Bugs caught and fixed (worth knowing the *pattern*, not just the list)

A recurring theme: several bugs were `:` vs `=` typos (Python's variable-annotation
syntax silently accepting `name: expr` and discarding the expression's value instead
of assigning it) — caught in `hvp_fr` (`grad_f: torch.func.grad(f)` → should be
`=`) and `sparse_pursuit` (`coeffs: torch.zeros(0)` → should be `=`). Worth double-
checking for this specific typo pattern if something mysteriously stays
`NameError`-undefined despite looking assigned.

Other real bugs: a stray unrelated import (`from sympy.logic import false`, `from
numpy import require`, `from unittest import removeHandler` — all autocomplete
accidents, none used); `make_slice_fn`'s off-by-one bounds check (`<=` should be
`<` on `l_end`); a module-level stray statement in `trust_score.py` referencing
undefined names at import time (leftover from drafting, would crash on import);
`torch.rand` vs `torch.randn` inconsistency in one test (not a bug, just style
drift); import-path mistakes (`from hlens import X` instead of `from hlens.X_module
import X` — `hlens` has no `__init__.py`, unlike `jlens`, so nothing is re-exported
at the package level; same mistake happened twice, once for `hlens.data` and once
for `hlens.experiments`).

**Session 2 — the important one (found only by the real-model test):**
`make_slice_fn` originally captured kwargs **once, from block `l_start+1`**, and passed
that same dict to *every* block in the slice. On Qwen3.5 that is wrong, because the
attention mask differs by layer type: linear-attention layers get `attention_mask=None`
(unpadded input), full-attention layers get a causal-mask tensor. A slice whose first
block was linear therefore ran its full-attention layers with **no causal mask** (eager
attention only adds a mask `if attention_mask is not None`), so tokens could see the
future. Symptom: single-layer slices matched exactly, but mixed/long slices `(3,12)` and
`(0,30)` had max |diff| of ~4–5.5. Fix: capture and cache kwargs **per block**, keyed
`(seq_len, block_index)`. The toy test could never have caught this because `TinyDecoder`
blocks take no kwargs — which is exactly why the real-model gate mattered. Lesson: any
test of a "replay a captured layer context" wrapper must include slices that span every
distinct layer type in the model.

**Session 2 — second real-model-only bug:** the first call of a fresh `make_slice_fn`
inside `torch.func.jvp` cached transform-wrapped rotary tensors that "escaped" on the next
transform (full details and fix in §3.1). Symptom: `hvp_fr` followed by `hvp_fd` on any
slice containing a full-attention layer raised a PyTorch `INTERNAL ASSERT FAILED ...
escaped?`; the fp32 tests had passed only because `hvp_rr` ran first, outside any
transform, and warmed the cache. Linear-attention-only slices never hit it because they
don't read the rotary tensors. Lesson: test the *order* of first calls, not just that each
estimator works once the cache is warm.

Minor, same session: duplicate `load_hf` / `load_qwen` definitions in `loading.py` (old
and new versions both present; the later one silently won) — removed.

---

## 5. Research detour: Neuronpedia

Investigated `github.com/hijohnnylin/neuronpedia` (user's question: could its code
be reused?). Findings:

- Neuronpedia's "Jacobian Lens" feature (`/v1/lens/prompt` endpoint) is **first-
  order only** — transport + unembed, same scope as `jlens.fitting`/`jlens.lens`.
  It **vendors an unchanged copy of Anthropic's `jlens`** (its PR #247 says "the
  vendored `jlens/` library is untouched") and saves lenses in `JacobianLens.save`
  layout so upstream `load()` reads them unchanged.
- It has **no** Hessian/curvature/sparse-decomposition/coordinate-swap tooling at
  all — nothing in Neuronpedia overlaps with what `hlens` actually builds.
- `interp-engine` (Neuronpedia's standalone hook/capture library) was considered
  and **rejected** for this project — it's built for production serving (vLLM-backed
  hooking limited to pre-declared static points, incompatible with arbitrary
  runtime hooking and definitely incompatible with second-order autodiff through
  CUDA graphs), not for the differentiable-through-hooks pattern `hlens` needs.
  `jlens.hooks.ActivationRecorder` + `hlens.interventions.ActivationEditor` are
  smaller, purpose-built, and already validated for exactly this project's needs.
- **Resolved in session 2**: the pre-fitted-lens lead panned out. HF repo
  `neuronpedia/jacobian-lens` has a Qwen3.5-4B lens that loads through this repo's own
  `JacobianLens.from_pretrained` — details in §6. No `jlens.fit` run was needed for the
  main lens. (Neuronpedia issue #235 — transformers 5.0–5.12 applied OLMo-3's YaRN RoPE
  scaling to the wrong layers — affects OLMo-3 lenses only, not Qwen3.5.)

---

## 6. Environment and model state (updated in session 2)

**Machine and workflow.** Everything now runs on the Jetson Thor (Linux 6.8.12-tegra,
aarch64, NVIDIA Thor, driver 580, CUDA 13.0; 122 GiB unified memory, ~117 GiB free with
nothing else running). The repo is at `~/repos/jacobian-lens`, edited over VS Code
Remote-SSH, with the Claude Code session running on the Thor itself — this resolves the
earlier remote-dev-vs-split question in favor of one unified environment.

**Python environment** (`uv`, `.venv`, Python 3.11.14):
- `torch 2.12.0+cu130` — stock PyPI wheel, not an NGC build. Confirmed at runtime:
  `cuda.is_available() == True`, device capability `(11, 0)`, arch list
  `['sm_80', 'sm_90', 'sm_100', 'sm_110', 'sm_120']`, so it runs natively on the Thor.
  `uv.lock` currently pins this; watch for `uv sync` replacing it.
- `transformers 5.17.0`; `pyproject.toml` floor raised to `>=5.11` to match the version
  the lens was fitted under. Qwen3.5 (`transformers/models/qwen3_5/`) is included; the
  model card's "install from main" advice predates the release that added it.
- `huggingface-hub 1.16.1`, `numpy 2.4.6`, `scipy 1.17.1`.
- **Not installed — keep it that way**: `flash-linear-attention`, `causal-conv1d`, and the
  `kernels` package. transformers' priority is HF-hub kernels (only if requested) →
  the original package if installed → plain PyTorch. With none present, the
  Gated-DeltaNet layers use the reference PyTorch implementation, which supports the
  double backward HVPs need. Every run prints two transformers warnings saying so
  (`causal_conv1d_fn` and `chunk_gated_delta_rule` fall back to reference PyTorch) —
  expected and wanted, but it means linear-attention layers are slow; account for that in
  `benchmark`.

**Model.** `Qwen/Qwen3.5-4B` (9.3 GB, bf16) in the HF cache, loaded by
`hlens.loading.load_hf()` as
`AutoModelForCausalLM.from_pretrained(..., dtype=torch.bfloat16,
attn_implementation="eager")` → `Qwen3_5ForCausalLM` (text-only; layers at
`model.model.layers`), then `.to("cuda").eval()`. 32 layers, `d_model=2560`, vocab
248320. `jlens.from_hf` finds the decoder with its default `Layout("model")`, uses
`compile=False`, and freezes all parameters (`requires_grad_(False)`) — so gradients only
flow through the activation `h` you make differentiable. **Eager attention is required**:
SDPA may dispatch to flash/mem-efficient kernels that lack double backward. Peak GPU
memory after load + `lens.apply`: 7.96 GiB. Whole real-model test run: ~16 s.

**Lens.** No fitting needed for the main lens. HF repo `neuronpedia/jacobian-lens`,
folder `qwen3.5-4b/jlens/Salesforce-wikitext/`:
- `Qwen3.5-4B_jacobian_lens.pt` — `n_prompts=417` (fit stopped on its convergence
  criterion). This is `LENS_FILE`, the default in `load_qwen`.
- `Qwen3.5-4B_jacobian_lens_n1000.pt` — `n_prompts=1000`. Available as `LENS_FILE_N1000`.
- Both: `d_model=2560`, `source_layers=[0..30]` (31 layers). Fitted by Anthropic
  (credit: @mntss) on `Qwen/Qwen3.5-4B` in bf16, WikiText-103 train, 128-token sequences,
  batch 64, transformers 5.11 on a B200. The lens is held in fp32 on CPU;
  `JacobianLens.transport` moves each matrix to the input's device on use.
- Loaded with `jlens.JacobianLens.from_pretrained("neuronpedia/jacobian-lens",
  filename=...)` — no Neuronpedia code needed.

**Lens sanity check (boot prompt, position −2 = " boot")**: top tokens read `Italy` /
`意大利` at layers 9–12, `-shaped` / `shape` at 22–25, `boot` at 26–29, and the final
layer agrees with the model's own next-token ` is`. That reproduces the expected
intermediate-concept structure. (Reading out at position −1 instead should show the
currency.)

**Cache hygiene.** The unused NVFP4 checkpoint was removed; a ~6.6 MB root-owned
leftover directory remains (vLLM probably ran in Docker as root against the shared HF
cache) — `sudo rm -rf ~/.cache/huggingface/hub/models--AxionML--Qwen3.5-4B-NVFP4`.
`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` (80 GB) is still cached and unrelated.

**Why not vLLM (unchanged conclusion).** The vLLM endpoint only serves text generation;
`hlens` needs in-process access to the `nn.Module` layers and backprop through the
forward pass *twice*. vLLM's CUDA graphs and paged attention rule that out, and the
NVFP4 checkpoint it served is a different (quantized) model than the bf16 one the lens
was fitted on. Don't run vLLM at the same time as `hlens` work — unified memory is shared.

**Rules to keep**: eager attention; `compile=False`; no flash-linear-attention /
causal-conv1d / `kernels`; single device only (no NCCL, no `device_map`);
`nvpmodel -m 0` + `jetson_clocks` before any `benchmark` run (put it in the script's
preamble as a check).

**Code added:** `hlens/loading.py` — constants `MODEL_ID`, `LENS_REPO`, `LENS_DIR`,
`LENS_FILE`, `LENS_FILE_N1000`; frozen dataclass `Loaded(hf, tokenizer, model, lens)`;
`load_hf(*, dtype=torch.bfloat16, device="cuda")` (the raw model, eager attention — kept
separate so `randomize_model` can get a fresh copy); `load_qwen(*, dtype=torch.bfloat16,
lens_file, device)` (everything wrapped and ready). Pass `dtype=torch.float32` for
curvature work: the checkpoint is stored in bf16 and every bf16 value is exactly
representable in fp32, so the fp32 model has *identical weights* and differs only in
arithmetic precision. `HFLensModel.unembed` casts to the LM head's dtype, so an fp32 model
also gets an fp32 contraction. The lens is fp32 regardless.

**Git state at time of writing (nothing committed in session 2):** modified —
`hlens/functional.py`, `pyproject.toml`, `uv.lock`, `tests/test_functional.py`,
`tests/test_hvp.py`, `tests/test_harness.py`, `tests/test_jspace.py`,
`tests/test_interventions.py`; untracked —
`hlens/loading.py`, `tests/conftest.py`, `tests/test_real_qwen.py`, this doc. The `datasets`
package is now installed (needed for the WikiText loader).

---

## 7. Next steps, in order

Steps 1–5 of the old list (get onto the Thor, find the model, load it and wrap it, run the
`functional.py` gate, get a lens) are **done**. Remaining:

0. **Housekeeping.**
   - ~~Re-run plain `pytest` after the `functional.py` change~~ — done, green (§2).
   - ~~Decide whether to restore `pytestmark` in `tests/test_real_qwen.py`~~ — done,
     restored.
   - Commit the session-2 work when ready.
   - Add a README note that the torch in `uv.lock` is stock PyPI `+cu130` (with `sm_110`)
     and must not be silently replaced.
1. ~~**Precision policy for HVPs**~~ — **decided and done** (§3.2): fp32 copy of the model
   (`dtype` argument on the loaders, `qwen_fp32` fixture); bf16 measured, not used for
   curvature. The plan's "cast `h`/`v` to fp32 while the model runs bf16" could not work
   as written (bf16 weights reject fp32 input), which is why a full fp32 model was needed.
2. ~~**First real-model HVP tests**~~ — **done** (§3.2): `rr` / `fr` agree to ~1e-6 in
   fp32 on single-layer, mixed and logit-tail slices; double backward works through the
   Gated-DeltaNet fallback; the first-call capture bug was found and fixed (§3.1, §4).
   Small follow-ups: an `eps`-sweep + plateau check on the real model (the harness
   functions have only seen toy `f`), and `hvp_norm_form` / `restricted_hessian` on real
   slices (§3.2 "Not yet run").
3. ~~**`jspace` paper gate**~~ — **passed, with caveats** (§3.3): median variance 7–9%,
   atoms sensible on the boot prompt at position −2, but the random control beats the real
   dictionary on variance and the "~25 atoms" criterion is pinned by `k`. Optional
   follow-ups: a `k` sweep / variance-saturation curve; the control at more layers and
   positions.
4. ~~**`interventions` multihop swap**~~ — **done, partially reproduced** (§3.4): the boot
   prompt flips Italy→Japan's currency at top-1 in 11/36 configs, and the probe-swap suite
   reaches 34% at best (random control 1/53), below the paper's 54–70%. Optional
   follow-ups: investigate the α=2 divergence from the paper (test `h_ref = ones` vs raw
   `W_U` rows in `lens_vectors`; try excluding position 0); a held-out split so the α/band
   choice isn't tuned on the evaluation items; per-category analysis with "workspace
   loading" (the paper's predictor of swap failure) — which is also close to what H1's
   trust score will be compared against.
5. **`restricted_hessian` on the real model, then the rest of `harness.py`**
   (`randomize_model` + small `jlens.fit`, `null_distributions`, `ablation_table`,
   `benchmark`, early-averaging test, stress test).
6. **Finish `experiments/trust_score.py`'s orchestration** (§3.7's list) — the first
   genuinely new scientific result the whole project works toward (H1).

---

## 8. Key source documents (for re-reading, not re-deriving)

- `hlens/docs/hlens_plan.md` — the build checklist the sessions worked from.
- `hlens/docs/research_notes.md` — full scientific motivation, hypotheses (H1–H11),
  methods detail, reproduction gates, deliverables table. Contains the one known
  stale path reference noted in §3.6 above.
- `hlens/docs/overview.md`, `hlens/docs/thingstocheck.md` — not read in depth;
  check if still relevant.
- `jlens/*.py` — the upstream package `hlens` builds on. Read in full: `hooks.py`,
  `protocol.py`, `hf.py`, `fitting.py`, `lens.py`. Not read: `_logging.py`,
  `examples.py`, `vis.py`.
- Installed `transformers` source consulted in session 2:
  `models/qwen3_5/modeling_qwen3_5.py` (decoder layer, text-model layer loop, attention,
  Gated DeltaNet fallbacks), `masking_utils.py` (`create_recurrent_attention_mask`),
  `integrations/hub_kernels.py` (kernel/fallback priority).
- `hlens/docs/Verbalizable Representations Form a Global Workspace in Language Models.pdf`
  — the Anthropic paper (117 pages). Read in session 2: §2.5 (reading/writing), §3.1,
  §3.3 (two-hop swaps), §3.4 (flexible generalization, Fig 18/19), §4.1 (workspace range),
  A.13 (Fig 68 per-trial swap grids), the multi-token swap passage (Fig 62), and the lens
  comparison (Fig 53/54). **Not read:** the rest (e.g. §4.2 capacity/occupancy, §4.3
  gain, §5 automatization). `pdftotext -layout` gives usable text; the Read tool can show a
  single page as an image (Fig 68 is on PDF page 82).
- `tests/tiny.py` — the `TinyDecoder` fixture every toy test uses.
- `hlens/loading.py`, `tests/conftest.py`, `tests/test_real_qwen.py` — the real-model
  entry points (session 2).

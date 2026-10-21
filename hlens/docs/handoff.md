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
- **Session 2, second half** ("build everything from A, B and C" — the real-model checks of
  the HVP/harness pieces, the reproduction gates and early averaging test, and the rest of
  the reliability harness) is §3.8; this time the assistant **wrote the library and test
  code** at the user's request (the §0 pattern was lifted for this stretch).
- **Session 3** (2026-10-15, still on the Thor) moved the two claims worth pursuing (A = H9 inhibition at the ignition threshold,
  B = H2 curvature along lens directions) onto a **second architecture, Gemma 3 12B**, with held-out Qwen3.5-4B stimuli and new null seeds. It
  also wrote and ran two pre-registrations (`preregistration_replication.md`, `preregistration_b_prime.md`). Everything is in §3.11; the
  environment notes (memory behaviour, reboots, downloaded models) are at the end of §6.

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
| `hlens/hvp.py` (§2) | ✅ Complete; **validated on real Qwen3.5-4B in fp32 on CUDA** | `rr`≈`fr` ~1e-6; `hvp_norm_form` matches a second difference to 0.2%; `restricted_hessian` symmetric to 7e-7 (§3.2, §3.8). bf16 measured, not validated |
| `hlens/jspace.py` (§3) | ✅ Complete, unit-tested; **paper gate passed on real Qwen3.5-4B (bf16)** | Median J-space variance 7–9% at layers 12/16/20; two of the three gate criteria are weaker than they look — see §3.3 |
| `hlens/interventions.py` (§4) | ✅ Complete, unit-tested; **swap reproduced only partially on real Qwen3.5-4B (bf16)** | Causal and specific (35% best vs 0/52 random control on probe-swap, after the first-token fix) but below the paper's rates and sensitive to α/band — see §3.4 |
| `hlens/harness.py` (§5) | ✅ **Complete; every plan item built** and exercised on the real model | Plateau / agreement / `ms_hvp` / validity checked on Qwen; randomized null, null distributions, ablation table, benchmark built; stress test (Phase-2 gate) **passes** — §3.5, §3.8 |
| `hlens/contexts.py`, `landmarks.py`, `gain.py`, `interpolation.py`, `variants.py` (new) | ✅ Built, real-model validated | Shared context helpers; layer-landmark gate (workspace ≈ depth 0.3–0.9); MLP-gain gate (**not reproduced**); ignition interpolation; frozen-attention variant — §3.8 |
| `hlens/data.py` (§6) | ✅ Complete, unit-tested | Validated against real shipped JSON files |
| `hlens/experiments/trust_score.py` (§7, first script) | ✅ **Built and run on probe-swap (H1), incl. per-position and margin variants** | **H1 not supported**: τ is a valid linearisation-error score (ρ = 0.95 with the true first-order error) but linearisation error does not predict swap failure — it goes with success; §3.7 |
| `hlens/swaps.py`, `hlens/stats.py` (new) | ✅ Built, tested | Swap helpers promoted from the tests; AUC / bootstrap / permutation / stratified AUC / LOO logistic |
| `hlens/experiments/{early_averaging,build_nulls,ablation_battery,stress_test}.py` (new) | ✅ Built and **run** | Early averaging (N=100), randomized null, 64-row ablation table, stress test — §3.8 |
| `hlens/experiments/gain.py` (H2), `remainder_coupling.py` (H5), `binding.py` (H3), `ignition.py` (H6) | ✅ **Built and run** (§3.9) | H2 mixed (curvature above random from L16, 2.1× at L28–30; gain not amplified); **H5 not supported** (twice); **H3 not supported**; H6 pre-registered criteria met but weak (L4/L20/L24, not the onset band; Taylor radius collapses at the knee) |
| `hlens/experiments/{engagement,early_curvature,motor_collapse,competition,register_persona}.py` (E1, H7–H10) | ✅ **Built and run** (§3.10) | **E1 not supported** (first-order prediction, not nonlinear gain, predicts success); **H7 not supported**; **H8 not supported** (`J→I` confirmed but per-block curvature rises); **H9**: precondition met, list-loading part not supported; **H10 proxy not supported** (persona pairs reproduce the H6 signature, register pairs do not) |
| `hlens/experiments/automatization.py` (H4), H11 | ⬜ Blocked on checkpoint access | |
| `hlens/arch.py`, `hlens/loading.py` (`MODELS`, `load_model`, `matched_layer`), `tests/test_multimodel.py`, `tests/test_real_gemma.py` (session 3) | ✅ Built; Gemma 3 12B **validated in fp32** | Architecture adapters (Gemma's extra MLP post-norm, scaled embedding); tiny-CPU Gemma/untied-Qwen checks; real Gemma slice/HVP/lens checks pass — §3.11 |
| `hlens/experiments/{gates,h9_replication,h2_replication}.py` (session 3) | ✅ Built and **run** on Gemma and Qwen | Gates G1/G2; claim A (H9 precondition) and claim B/B′ (H2) replication — §3.11 |
| **Claim A** (H9 precondition: inhibition peaks at the ignition threshold) | 🟡 **Holds on Qwen3.5-4B only** | Held-out 120 trials pass on Qwen3.5-4B (both spacings); **fails on Gemma 3 12B** (30% both-negative) and **on Qwen3.6-27B** (17% both-negative, P0a-c all fail) |
| **Claim B′** (curvature along lens directions above random, equivalence-tested null, fresh data) | ✅ **Holds on Qwen3.5-4B, Gemma 3 12B and Llama 3.1 8B** | Original B **failed its own rules** (T2 on Qwen, T3 as written on both); B′ is the pre-registered follow-up — §3.11 |
| Qwen3.6-27B | 🟡 Claim A run (mixed precision, validated on the 4B): **fails**; claim B′ not run | B′ would need a null lens fit on a randomized 27B — §3.11 |
| `hlens/experiments/{h6_replication,h8_replication}.py` (session 3) | ✅ Built and **run** on Gemma 3 12B | **H6 not supported on Gemma** (no curvature peak at the threshold: T2 p = 0.90, T3 CI includes 0); **H8 not supported and the Qwen finding generalizes** (per-block nonlinearity rises toward the output, ratio 5.05) — §3.11 |
| `hlens/experiments/{h10_extended,h10_control,h10_scaled}.py` (session 3) | ✅ **Built and run** (Qwen3.5-4B) | H10 scaled to 100 pairs per family: **all four families (persona, near-synonym, matched control, countries) reproduce the ignition signature; no persona-specific effect** (matched difference −0.043, p = 0.08); distance explains within-family variation, not between-family differences — §3.11 |
| `hlens/swap_suite.py`, `hlens/experiments/swap_heldout.py`, `data/experiments/probe-swap-heldout.json` (session 3) | ✅ **Built and run** | Held-out swap suite (165 generated items) + `preregistration_swap_heldout.md`. **Qwen3.5-4B passes at α = 1.25 (52.6 %, controls 0/0/0); Gemma 3 12B fails at the pre-registered α = 1.25 (3.2 %) and passes at α = 1.0 (44.0 %)** — §3.11 |
| `hlens/vis_ext.py` (§8) | ⬜ Not started | Lowest priority (Phase 5) |

Test files live in `tests/`, one per module: `test_functional.py`, `test_hvp.py`,
`test_jspace.py`, `test_interventions.py`, `test_harness.py`, `test_data.py`,
`test_trust_score.py`, `test_landmarks.py`, `test_gain.py`, `test_interpolation.py`,
`test_early_averaging.py`, `test_null_pipeline.py`, `test_benchmark.py`, `test_stress.py`, plus
`test_real_qwen.py` (real-model smoke checks) and `conftest.py`. `tests/tiny_qwen.py` is a
tiny **CPU Qwen3.5** (hybrid linear/full attention, fp32, eager) used to unit-test the
HF-specific harness code (`randomize_model`, `frozen_attention`, null collection) in the fast
suite — it must be eager: SDPA has no forward-mode autodiff. The toy tests all reuse `tests/tiny.py`'s `TinyDecoder` (a 4-layer,
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

**What has touched the real model**: everything except `trust_score.py` (core math only) and
the `shuffle` randomization mode (unit-tested, no null built). See §3.8 for the stage A/B/C
results.

Last runs (Thor, `uv run --no-sync`): `pytest -m "not real_model"` → **73 passed**
(36 real-model tests deselected). **`uv run --no-sync pytest -m real_model`, run as ONE session
after all stage A/B/C edits: 36 passed in 26 min 41 s** (this exercises the `qwen` bf16, `qwen_fp32`
and a third freshly loaded randomized model together). Re-run both after further edits. Runtimes: jspace gate ~10 min, swap tests ~5 min,
stress test ~8–13 min, landmarks ~1.5 min, everything else seconds to ~2 min.

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

**Validated on the real model, stage A (fp32)** — `tests/test_hvp.py`:
- `hvp_norm_form` on vector-valued slices `(2,5)` and `(3,12)` vs the central second
  difference `(f(h+εd) + f(h−εd) − 2f(h))/ε²`: best relative error **1.6e-3 / 2.7e-3** (at
  ε=0.3 with |h| ≈ 13–17; error 11–21% at ε=0.03, 1–2% at ε=1: the usual U shape). This
  confirms on the real model that it returns the **full** contraction `dᵀH d`, not the half.
- `restricted_hessian` (layer 16, position −2, k=6 country lens vectors, scalar = a logit):
  asymmetry/|H|max = **6.9e-7**; diagonal within ~5% of independent second differences.
  **Library fix:** it allocated its result with `torch.zeros(k, k)` (default device/dtype);
  it now uses `directions.new_zeros(k, k)`, so CUDA/fp32 inputs work.

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

**Re-run after the first-token fix (§4):** 87/90 items usable, **52/87 baseline-correct**;
best config unchanged — **L12–20, raw vectors, α=1.25 → 18/52 (35%)**, random-direction control
**0/52 in each of 3 seeds** (was 1/53); `multihop` 1/10. The numbers quoted in the rest of this
subsection (53 baseline-correct, 34%, 1/53) are from the pre-fix run and are off by ≤ 1 item.

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

**Built in session 2 (stage C, details and results in §3.8):** `randomize_model`
(`reinit` / `shuffle`), `fit_null_lens` (cached, resumable), `collect_restricted_hessians`,
`null_distributions` / `load_null`, `ablation_table`, `benchmark`, `averaging_ratio`;
`estimator_agreement` gained an `eps` parameter (its hard-coded 1e-3 was wrong to assume
for real models). **Validated on the real model (stage A)** — `tests/test_harness.py`, fp32,
scalar logit at layer 16:
- **`eps_sweep` / `find_plateau`:** the fd error against `hvp_fr` is U-shaped: 6.8e-2
  (ε=1e-4, rounding noise) → 6.1e-3 (1e-3) → 2.2e-4 (3e-2, best) → 5.6e-2 (1.0) → 0.40
  (3.0, truncation). `find_plateau` found `(3e-4, 0.3)`, representative ε=0.01.
- **`estimator_agreement`:** both ends of the U are flagged unusable (ε=1e-4: ~7%; ε=3: ~40%);
  ε=1e-3 is already usable here (0.6%); at the plateau ε `rr`/`fr` agree to 1e-6.
  (A first version of the test wrongly assumed ε=1e-3 was noise-dominated.)
- **`ms_hvp`:** error vs the exact secant `∇f(x+v) − ∇f(x)` at |v| = 0.5|x|: n=1: 1.02, 2: 0.435,
  4: 0.200, 8: 0.096, 16: 0.047 — halving per doubling, i.e. the O(1/n) of the left-endpoint
  Riemann sum, as predicted. Its `reliability` is 2% at |v|=0.01|x| and 78% at 0.5|x|.
- **`validity_score`:** valid (score 30) for a 0.01-scale perturbation, invalid for 100×.

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

**Update (stage C):** the weight-randomized null now exists (§3.8: `results/nulls/`), as does
the validity gate that should accompany every τ. For the null overlay, the H1 pipeline can
call `load_null(...)` / `NullDistributions.above(...)`; `trust_score` itself still needs a
null built with the *same* protocol (swap deltas, not the random-context Hessian entries
recorded so far).

**H1 orchestration — BUILT AND RUN (session 2, after stage C).** `run_trust_score` /
`analyze` in `hlens/experiments/trust_score.py` (CLI: `uv run python -m hlens.experiments.trust_score`,
~19 min for 88 items; `--reanalyze` recomputes the analysis from the saved rows); helpers in
`hlens/swaps.py` (`pair_edits`, `run_top1`, `probe_items`, defaults band (12,20) / α=1.25) and
`hlens/stats.py`; results in `results/trust_score_qwen3.5-4b.json` (per-item rows + analyses).
- **Definition used.** For each (item, layer l ∈ {12, 16, 20}): `δ_l` = the α=1 swap delta
  (raw lens vectors `u_t @ J_l`) applied to the clean residual at layer l at **every position**
  (the real intervention, not a per-position perturbation); `f_l` = the final-layer residual at
  the last position as a function of the layer-l residual; `τ_l = ‖½ δᵀHδ‖ / ‖Jδ‖`
  (`trust_terms` returns both). **τ scales exactly linearly with α** (quadratic numerator /
  linear denominator; verified to 0.2% on the real model), so the ranking is α-independent and
  only the labels change. Outcome = top-1 flips to `swap_answer` under the real multi-layer swap
  (band (12,20), one editor per layer); the population is items answered correctly at baseline
  (**52 of 88**). Labels were computed in **both** fp32 (the model τ is computed in) and bf16
  (the precision the swap sweep was tuned in); they agree on 98.9% of items.
- **Baselines the score must beat:** workspace loading (mean cosine of the residual with the
  source lens vector; paper §3.4), `‖Jδ‖`, `‖δ‖`; **null:** the same τ at layer 16 on the
  weight-randomized model with its own lens (`results/nulls/`); permutation tests; bootstrap CIs.
- **Result (α=1.25, fp32 labels; corrected, n=51: 33 failures, 18 successes).** AUC for predicting
  *failure* from τ: **0.22** (L12 0.24, L16 0.22, L20 0.23; CI for L16 [0.10, 0.35]) — the
  **opposite** of H1: **high τ predicts swap SUCCESS, AUC 0.78**. Same sign at α=1.0 (fail-AUC
  0.12, 8 successes) and with bf16 labels (0.24). Workspace loading (low → failure) AUC 0.71
  [0.57, 0.85], p = 0.006; `−‖Jδ‖` 0.68; `‖δ‖` ~0.47 (nothing). **Null: τ from the randomized
  model has AUC 0.53 [0.35, 0.70]**. Spearman(τ16, loading16) = 0.41, (τ16, ‖Jδ‖) = 0.75. Leave-one-
  out logistic AUC: loading 0.68, log τ 0.74, both 0.71. (The pre-fix numbers were 0.22 / 0.71 /
  0.525 / n=52; the fix changed nothing material.)
- **But the association is mostly category structure (post-hoc stratification, defined after
  seeing the first analysis).** The geography-style categories (`city-*`, `language-capital`,
  `river-capital`; 19 items, 13 successes) have τ16 ≈ 0.7–1.1 and succeed; `multihop`,
  `element-*`, `animal-*`, `planet-*` have τ16 ≈ 0.04–0.5 and fail. Within groups, AUC(τ → failure)
  is 0.45 (geography) and 0.29 (other); the **stratified common AUC is 0.34 (permutation p = 0.07
  for the reversed direction)** — suggestive at best. The same stratification leaves **workspace
  loading at 0.72 (p = 0.017)**: the paper's own predictor survives, τ's reversed signal mostly
  does not.
- **Variants (per-position, margin, true error)** — `score_variants` / `run_variants` /
  `analyze_variants`, CLI `--variants`, results `results/trust_score_variants_qwen3.5-4b.json`
  (51 baseline-correct items, labels reused, layer 16 for the per-position ones, randomized-model
  null for all). Built to separate the two links of H1's chain, **τ → true first-order error →
  swap failure**, using the exact Taylor polynomial of a single-layer swap `f(h+αδ) ≈ f(h) +
  αJδ + α²·½δᵀHδ` (`taylor_terms`):
  - **Link 1 holds, strongly.** Spearman(τ, the *true* first-order error `‖f(h+αδ) − f(h) − αJδ‖ /
    ‖f(h+αδ) − f(h)‖`) = **0.95** (final-residual function) and **0.81** (margin
    `logit(swap_answer) − logit(answer)`), and the second-order term earns its keep: median
    relative error of the Taylor prediction falls **0.60 → 0.35** (residual) and **0.52 → 0.13**
    (margin) when `α²·½δᵀHδ` is added. So τ is a good validity score and the Hessian machinery
    measures real model nonlinearity.
  - **Link 2 fails.** The *true* first-order error does not predict failure: AUC(failure) 0.25
    (residual) and 0.33 (margin) — i.e. large linearisation error goes with swap SUCCESS (stratified
    0.40 / 0.47, not significant).
  - **Per-position τ** (perturb only one position; `last` = the answer position, `pstar` = the
    position with maximal source loading): `τ_last` AUC(failure) 0.39; **`τ_pstar` 0.58 [0.41,
    0.75], p = 0.18**; margin `τ_pstar` 0.60 (p = 0.12, stratified 0.64, p = 0.095): a **weak,
    non-significant hint in H1's predicted direction** only for the argument-like position.
    Margin τ with all positions: 0.29 / 0.32 / 0.36 at L16 / L12 / L20 (reversed, stratified ≈ 0.5).
  - **Does the second-order term improve outcome prediction?** Taylor-predicted post-swap margin
    (single layer 16, α=1.25): first order flips the sign for 3.9% of items, second order for 19.6%
    (actual multi-layer success 35%; it is a single-layer prediction of a 9-layer intervention);
    AUC(failure) 0.56 [0.38, 0.73] first order vs **0.64 [0.47, 0.79] second order (p = 0.05)** —
    a modest improvement that does not survive stratification (0.58 for both).
  - **Caution from the null:** the randomized model's margin-τ has AUC(failure) 0.66 (p = 0.03)
    and its `pstar` residual-τ 0.36; so some of these association sizes can arise without any
    learned structure (token/prompt effects), and only the learned-model-specific part (residual-τ,
    all positions: null 0.53 vs real 0.22) is clean.
  - Baseline for comparison: loading at the max-loading position `loading_max` AUC 0.50 (nothing);
    mean loading 0.71 is the best predictor of failure on this suite.
- **Verdict.** H1 *as stated* (τ high ⇒ the first-order readout is untrustworthy ⇒ swaps fail)
  is **not supported** — and the variants show *where* the chain breaks: τ **is** a good validity
  score (it tracks the true first-order error at ρ = 0.95 / 0.81, and the second-order term makes
  the Taylor prediction much better), but **linearisation error is not what makes swaps fail**: it
  is larger for swaps that succeed. The pre-registered falsifier ("uncorrelated with outcome, or
  fully explained by activation strength") is not cleanly met either: τ is strongly (reversely)
  associated with outcome, largely through category, and the paper's own workspace loading is the
  better predictor of failure. A plausible reading — successful swaps need the network to respond
  strongly and nonlinearly to the swapped concept (an *engaged*, ignition-like regime) — is a
  hypothesis, not a result. What H-lens does deliver is a validated second-order measurement
  of that nonlinearity.
- **Caveats.** The all-position τ perturbs every position jointly with f = the final residual at
  the last position; the variants cover the margin scalar and per-position (last / max-loading)
  perturbations at **layer 16 only** (all-position variants also at 12 and 20; the null only at
  16). n = 51 with 18 successes, so CIs are wide (±0.13); the α/band were tuned on the same items;
  the geography grouping is post-hoc; labels are first-token matches; `err1`/`err2` are for a
  *single-layer* perturbation while the swap labels come from a 9-layer one. The null is a single
  randomized model with a rough 30-prompt lens and, for some variants, shows association sizes
  comparable to the real model (see above). `tests/test_interventions.py` still has its own
  copies of the boot-prompt swap helpers now in `hlens/swaps.py`.

**Original plan for the orchestration (done above; kept for reference):**
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

### 3.8 Stages B and C (session 2, second half): gates, early averaging, harness build-out

New library modules and what they do. All curvature work uses the **fp32** model; the
landmark/gain/jspace-style measurements use whichever the test names (`qwen` bf16 for
landmarks, `qwen_fp32` for gain). Everything below was run on the Thor; raw outputs are in
`results/` (small JSON is meant to be committed; `*.pt` is gitignored).

- **`hlens/contexts.py`** — `ContextPoint` (input ids, layer, position, residual `h`),
  `capture_context`, `logit_fn` / `cotangent_fn` (scalar of the output at the point's position),
  `at_position(f, point)` → `(f1, x0)` restricting a function of the full residual `[1,seq,d]`
  to its own position so directions live in plain `d_model` space, and `lens_direction_bank`.
  `hlens/jspace.py` gained `unembed_directions(model, token_ids)` (the layer-independent
  `u_t = ∂logit_t/∂h_final`; `lens_vectors` now calls it and applies `J_l`).
- **`hlens/landmarks.py` — layer-landmark gate (paper §4.1, Fig 27/28).** `compute_landmark_curves`
  (30 WikiText prompts × 128 tokens, bf16) computes per source layer: top-k lens accuracy vs the
  model's real next token, excess kurtosis of the lens readout, autocorrelation of the top-1 lens
  token vs a position-shuffled null, effective dimensionality of the lens vectors (fraction of
  `d_model` for 90% of variance, 4096-token sample), and the CKA matrix between layers' lens-vector
  geometry. `workspace_range` reports per-metric onset/offset (rules documented in the docstring;
  per-metric estimates stay visible) and a lower-median consensus. **Qwen3.5-4B result**
  (`results/landmarks_qwen3.5-4b.json`): top-1 accuracy ≤ 3.5% to L15, 20% by L26, then
  29% → 59% over L27–L30; autocorrelation negative early, **jumps at L10** (0.08 → 0.29), peaks
  ~0.9 at L20–L26 and falls to 0.31 at L30; excess kurtosis is *not* near zero early (2.4 at
  L0), falls to ~1.0 at L6–L14 and climbs to ~2.0 by L26 (so the paper's "near zero early" shape
  does not transfer — onset must be anchored at the minimum); effective dimension grows steadily
  0.03 → 0.53; CKA adjacent-layer mean 0.979 vs 0.516 for layers > 10 apart, with a long middle
  block (L12 has CKA ≥ 0.9 with L6–L17). **Onset by metric: autocorr L9, eff-dim L14, kurtosis
  L18; offset: top-1 accuracy L28, autocorr fall L30.** So the workspace on Qwen is roughly
  **depth 0.3–0.9 (≈ L9–L28)**, consistent with the L12–L28 band mapped from the paper's
  L38–L92 and with the best swap band (12, 20) being its early half. The metrics disagree on
  onset by ~9 layers; do not quote a single onset as sharp. Lower-median consensus (lower median) layers 14–28 = depth 0.45–0.90.
- **`hlens/gain.py` — MLP-gain gate (paper §4.3.1, Fig 32 left) — NOT reproduced.** gain(v) =
  ‖MLP_{l+1}(·)‖ on a unit direction, normalised by the median over isotropic random unit
  directions; populations of 2000: J-lens vectors vs MLP-neuron output directions (`down_proj`
  columns of layer l). The paper's input convention is unspecified, so three were tried
  (`prenorm` = the block's own pre-norm then the MLP; `raw` = MLP on the unit vector; `rms` =
  scaled to unit RMS) and three J-lens token populations (uniform over the 248k vocabulary;
  low ids; tokens actually in the lens top-10 on real text — `active_token_pool`). **None gives
  the paper's ~10× in the workspace band.** Best case (active tokens, `prenorm`): J-lens gain
  0.4–0.9 for L3–L15, 1.2–1.8 for L16–L25, 2.0–2.8 for L26–L28, then **4.6 (L29) and 8.3 (L30)**;
  uniform-vocab J-lens never exceeds 1.2 before L26 (3.06 at L30). MLP-neuron gain is
  1.1–1.9 (peak 1.9 at L15) — *above* the J-lens curve through L3–L20, where the paper has
  neurons ≈ 1. The paper's Fig 32 (read from the PDF): J-lens gain ~1.2–1.5 early, ~2–4× by
  depth 40–60, **~10× at depth ~85 then falling to ~2.3 at 100**; neurons ≤ ~1.3. What we see
  is a comparable *peak size* (8×) but only in the last layer we can measure (L30; no J is
  fitted for layer 31), i.e. at the "motor" end, and weak gain in the workspace band. Per
  notes §4.4 this is first evidence about scale (or about an unstated convention), not a
  pipeline bug — the lens vectors it uses are the ones validated in the jspace gate. The test
  (`tests/test_gain.py`, results in `results/gain_qwen3.5-4b.json`) asserts only what we
  observe (neurons in (0.5, 2.2), J > neuron for L ≥ 21, late > 3× mid) as a regression guard.
- **Early averaging test (notes §4.5).** `averaging_ratio(Hs)` = ‖E[H]‖ / E‖H‖ with a
  sign-shuffled null (independent ±1 per context; ≈ 1/√N for equal norms). Protocol
  (`experiments/early_averaging.py`, `collect_restricted_hessians`): layer 16, 100 WikiText
  prompts (64 tokens), one random valid position per prompt, scalar = the ` Italy` logit at
  that position, **fixed** 8 country lens-vector directions (so matrices are comparable),
  `H_p[i,j] = v_iᵀ Hess v_j`, fp32, ~12 s per context. **Result:** second-order ratio **0.348**
  (sign-null 0.109, p95 0.244); first-order ratio of `g_p[i] = v_i·∇f` **0.995** (null 0.085).
  **Randomized-weights control (reinit, seed 0, own lens fitted on 30 prompts): second order
  0.294 (null 0.100, p95 0.132), first order 0.997** — and the randomized model's curvature is
  ~1300× smaller (mean ‖H‖_F 3.3e-4 vs 0.433). **Reading:** per context, about two thirds of the
  Hessian norm varies and about a third survives averaging, which is above the independent-
  context null but **close to what a randomized network gives**, so there is no evidence of a
  *learned* averaged H-lens object; write claims per-context and test any averaged object
  against the randomized null. The first-order ratio ≈ 1 in both models means that metric
  cannot discriminate (∇logit projected on fixed directions is dominated by the residual skip
  path); do not use it as evidence that `J_l` "exists". Caveats: one layer, one output token,
  one direction set, positions on generic WikiText text unrelated to countries, one random seed.
  Raw H/g stacks: `results/early_averaging_qwen3.5-4b_L16.pt` (+ `.json`).
- **`hlens/harness.py` stage C.** `randomize_model(hf, tokenizer, mode, seed)` randomizes **in
  place** (pass a fresh `load_hf()` copy) and re-wraps with `from_hf`: `reinit` clears every
  `_is_hf_initialized` flag (module, parameter, buffer — loaded weights carry it and would be
  skipped) then calls `init_weights()` under a forked RNG; `shuffle` permutes entries within each
  parameter tensor (tied weights appear once in `named_parameters`). `fit_null_lens` wraps
  `jlens.fit` (cached, resumable via `path + '.ckpt'`). **Cost: ~62 s per 64-token prompt
  regardless of `dim_batch` (32 and 64 identical → compute-bound)**, so the plan's "~100 prompts,
  cheap at 4B" is ~1.7 h; the null used **30 prompts, layer 16 only** (~25 min). `null_distributions`
  records diagonal (directional-curvature) and off-diagonal (cross-term) samples to
  `results/nulls/<name>.{pt,json}`; `load_null`, `NullDistributions.above(value, which, q)`. Built:
  `randomized_reinit_s0_L16` (`experiments/build_nulls.py`, same protocol as the real early-
  averaging run): |diag| quantiles 0.5/0.9/0.99 = 4.9e-5 / 1.2e-4 / 1.9e-4, |cross| =
  2.4e-5 / 5.8e-5 / 9.3e-5 (the real model's entries are ~1e-2). **Only `reinit`, one seed**;
  `shuffle` and more seeds are not built. The null lens is a rough fit (30 prompts, random
  weights), so "the same 8 tokens" give unrelated directions there.
- **`ablation_table` + `experiments/ablation_battery.py`** — generic Cartesian-product runner
  (tidy rows, optional CSV) and the second-order A.7 battery: attention (full | frozen),
  `skip_first` (0 | 4), direction set (lens | random), estimator (fr | fd), aggregation
  (mean | median), Nσ filter (none | 2σ); 64 rows, 16 contexts, k=6, layer 16
  (`results/ablation_battery_qwen3.5-4b.csv`). **Not implemented** (they change how `J_l` is
  fitted, not how a Hessian is evaluated): target layer, self-only / future-only positions,
  corpus size/distribution. Findings: `fr` and `fd` agree to 4 digits on the full-attention
  model (fd asymmetry ~2e-4); **mean ‖H‖_F along J-lens directions is ~9× that along random unit
  directions (0.258 vs 0.029)** — partly expected, since the lens directions include ` Italy`'s
  own vector and the scalar is the ` Italy` logit; skipping the first 4 positions raises mean
  ‖H‖ 0.121 → 0.167; Nσ=2 drops no contexts; mean vs median aggregation differ by ~0.02 in the
  ratio. **At N=16 the averaging ratio sits at its null (0.27 vs 0.28), so the battery is not
  powered for that quantity.**
- **`hlens/variants.py` — `frozen_attention()`** monkeypatches Qwen3.5's eager attention so the
  softmax pattern is a constant (value pathway only; second-order frozen-QK; linear-attention
  layers unaffected). **It is not estimator-invariant:** `hvp_fr` treats the pattern as constant
  including its tangent (asymmetry 8e-7), `hvp_fd` re-evaluates the pattern at x ± εv (asymmetry
  4e-2); they compute different operators, so "frozen" must be defined per estimator and
  `estimator_agreement` will (correctly) flag them. Effect on curvature size is tiny at layer 16
  (mean ‖H‖ 0.142 vs 0.145).
- **`benchmark`** (per HVP, per restricted H, per `lens.apply`; refuses to run unless
  `nvpmodel -q` says MAXN — `sudo nvpmodel -m 0 && sudo jetson_clocks`, which needs root). **The
  Thor is currently in the `120W` mode (id 1; MAXN is id 0), so these timings are not
  comparable:** HVP 0.44 s, restricted H (k=8) 3.5 s, `lens.apply` 0.48 s → **7.2×; the plan's
  "restricted H ≤ 2–3× lens.apply" milestone is not met** — an HVP costs about one
  `lens.apply`, so the milestone can only hold for k ≲ 3. Re-run in MAXN before quoting.
- **`hlens/interpolation.py` + `experiments/stress_test.py` — pathological stress test (the
  Phase-2 gate): PASSES.** Ignition protocol from `data/experiments/README.md`: replace the
  `{W}` token's embedding by `α·emb(A) + (1−α)·emb(B)`; A's reciprocal-rank share crosses 0.5
  at α* ≈ 0.35–0.54 and rises from ~0.2–0.3 at α=0.40 to ~0.8–0.9 at α=0.50–0.55 at every
  read layer from L14 to L30 (width ≈ 0.1; shallower layers are broader or more erratic) — a
  genuine sharp flip. Stress construction: scalar = lens-logit difference A−B read at layer 20
  as a function of the layer-8 residual at the `{W}` position; *knee* path α* ± 0.04 vs *control*
  path 0 → 0.08 (same α-length, far from the knee); score single-step, multi-step (n=4, 16) and
  the validity gate against the exact secant `∇f(x_hi) − ∇f(x_lo)`. **15 trials (3 carriers × 5
  France pairs): median single-step error 72% at the knee (range 32–129%) vs 6.2% on controls;
  multi-step (n=16) 4.8% vs 0.4%; the validity gate flags 15/15 knee paths and 0/15 control
  paths; `ms_hvp`'s own single-vs-multi `reliability` is 0.3–1.2 at the knee vs ~0.05**
  (usable without ground truth). Assertions (thresholds set from this run with margin) are in
  `tests/test_stress.py`; per-trial JSON in `results/stress_test_qwen3.5-4b.json`. Caveats: one
  (src, read) layer pair, one `half_width`, France-centred pairs, and the error is measured on
  a straight line in residual space (what the HVP integrates), not along the true α-path.

### 3.9 Option 3 — H2, H5, H3 (session 2, third part)

(Tracked open hypothesis **E1**, the engagement hypothesis from the H1 result, is in §7b and is
deliberately deferred to H6.) Everything here is fp32 on the Thor; raw outputs in `results/`.

**H2 — MLP amplification and curvature of J-space directions (`hlens/experiments/gain.py`,
`results/gain_h2_qwen3.5-4b.json`, null `results/gain_h2_null_qwen3.5-4b.json`).** Because the
context-free gain gate did not reproduce (§3.8), H2 was run as the *in-context* diagnostic the paper's
footnote 5 says matters. At real activations `r` (the residual entering the MLP of layer `ℓ+1`,
pre-norm included; 30 WikiText prompts × 16 random valid positions = 480 contexts) and unit
directions `v` defined at layer `ℓ`: `g1 = ‖J v‖` (in-context first-order gain), `κ = ‖½ vᵀHv‖`
(curvature along `v`), `ρ = κ‖r‖/g1`. The MLP is position-wise, so all (context, direction) pairs are
batched through one `jvp` / `jvp`-of-`jvp` (`mlp_taylor`; checked against second differences on the tiny
model). Three fixed 64-direction populations per layer: J-lens vectors of actively-used tokens,
MLP-neuron output directions (`down_proj` columns), isotropic random. Ratios to the random population,
prompt-cluster bootstrap CIs (300 resamples; CIs are tight, ±0.01–0.05, because most variance is across
directions, not prompts).
- **Curvature κ, J-lens / random:** 0.69–0.98 for L2–L15 (min 0.69 at L7) — *below* random; 1.11 at
  L16, 1.2–1.3 at L17–L21, 1.37 L22–L23, 1.43 L24, 1.55 L25, 1.71 L26, 1.75 L27, **2.1 at L28–L30**.
  Against the neuron controls (κ/random 1.1–1.7 through L17, ~1.15–1.3 after): neuron > J-lens for
  L2–L17 and L20; **J-lens > neuron from L21 onward**.
- **First-order gain g1, J-lens / random:** 0.54–0.99 for L2–L24 (min 0.54 at L6–L7), 1.06–1.09 L25–L27,
  1.15 L28, 1.31 L29, **3.05 at L30**. So **the in-context gain does not show the paper's amplification
  either** (consistent with the failed gate): J-aligned directions are amplified *less* than random
  through most of the network at first order.
- **Second-to-first-order ratio ρ, J-lens / random:** 1.2–1.6 over L4–L29 (1.64 at L28; 0.68 at L30):
  J-aligned directions are consistently more nonlinear *per unit of gain*.
- **Across-context variability of the gain** (coefficient of variation of g1 over contexts for fixed
  directions, the stand-in for ∂gain/∂context): ≤ 0.3 for every population and layer; J-lens slightly
  higher at L0–L13 (e.g. L5 0.18 vs 0.08), equal to the controls in L22–L28.
- **Randomized-weights control (layer 16): every ratio is exactly 1.00** (κ, g1, ρ; both populations), so the
  structure above is learned, not architectural.
- **Verdict (diagnostic only, as the notes frame it).** H2's prediction — curvature along J-aligned
  directions elevated relative to isotropic *and* neuron controls in the workspace layers — is
  **supported for L21–L29 (late workspace into the motor end)**, only vs the isotropic control for
  L16–L20, and **not** for the early workspace (L2–L15), where J-lens κ is below random. The falsifier
  ("curvature at the isotropic baseline, gain reproduced by linearised statistics alone") is not met —
  but there is no first-order gain to reproduce until L25+. The `∂gain/∂context`-variance prediction
  is not supported. Caveats: 30 prompts; "J-lens vectors" are those of *actively-used* tokens, which are
  not independent of the residual's content while isotropic directions are (some of the effect may be
  alignment with high-variance residual directions); ρ is evaluated at a perturbation as large as `r`.

**H5 — remainder coupling (`hlens/experiments/remainder_coupling.py`, pre-registered in
`hlens/docs/preregistration_h5.md`; results `results/h5/`).** Design (fixed beforehand): 200 WikiText
contexts (prompts 200–239, 5 random valid positions each, layer 16, fp32); sparse pursuit (k=25) against the
full-vocabulary J-lens dictionary; remainder direction `r̂`; scalar `f` = the top-1 next-token logit at the
position; coupling vector `w = H_f r̂` (one HVP); `s(a) = |⟨w, a⟩|/‖w‖`; control atoms = the 25 best-correlated
atoms *not* selected; primary statistic `T = mean_c [mean_sel s − mean_ctrl s]`; Test A = one-sided sign-flip
(p < 0.01), Test B = bootstrap CI of `T_real − T_null` > 0 against a weight-randomized model; supported iff
both. The randomized-model arm was run first (the script refuses otherwise).
- **First run — pre-registered verdict: NOT SUPPORTED.** `T_real = −0.0024`, `T_null = +0.0010`; test A
  p = 0.9999; test B CI [−0.0049, −0.0019] (the real model is *below* the null). 200/200 contexts kept.
  `‖H r̂‖/‖∇f‖` is 0.115 in the real model vs 0.007 in the randomized one (the remainder curves the gradient
  ~16× more in the trained network); mean J-space variance fraction 0.068 real vs 0.148 randomized.
- **A flaw in the pre-registered control, found after seeing the negative sign.** The pursuit makes the remainder
  *exactly orthogonal* to the selected atoms (KKT; verified in a unit test), but not to the control atoms (they are
  correlated with `x ≈ ρ`). Any component of `w` along `r̂` therefore inflates controls and not selected atoms,
  so a negative `T` is what the geometry predicts even under real coupling. Post-hoc (exploratory), against the
  analytic random-direction level `E|cos| = 0.0158`: **selected atoms couple at 1.45× chance [1.39, 1.51] in the
  real model vs 1.14× [1.11, 1.18] in the null** (real − null = +0.0048, CI [0.0038, 0.0059]); controls 1.60× vs
  1.08×; no dependence on the pursuit coefficient (Spearman 0.045). So the remainder's curvature response is
  aligned with lens-vector directions modestly beyond chance and beyond the randomized model, but **not
  specifically with the selected atoms**.
- **Corrected follow-up (Addendum 1 to the pre-registration, labelled post-hoc because it was designed after
  the first result):** project `r̂` out of both `w` and the atoms (`s⊥`), same tests A⊥ / B⊥; arms `*_v2`
  (`--tag _v2`). It cannot overturn the first verdict by itself; it tests whether that verdict was an artifact of
  the control.
- **Corrected follow-up result: the artifact explanation was WRONG.** `T⊥_real = −0.00234` vs `T_null = +0.00093`; A⊥
  p = 0.9998 (fail), B⊥ CI of `T⊥_real − T⊥_null` = [−0.0048, −0.0018] (fail) — essentially identical to the first run
  (−0.00238), so projecting `r̂` out of `w` and the atoms changed almost nothing (the component of `w` along `r̂` is
  small). **H5 is not supported in either form**, and the "orthogonality bias" I proposed after seeing the negative sign
  does not explain it: selected atoms really do couple slightly *less* to the remainder's curvature response than the
  best-correlated unselected atoms (those correlate with `x` itself, where the curvature structure is). What remains of
  the post-hoc finding is only that lens-vector directions in general couple to `H r̂` above chance in the trained model
  (1.45× / 1.60× for selected / control atoms vs 1.14× / 1.08× in the randomized model). Lesson: a story invented to
  explain a surprising sign is a hypothesis; the follow-up was designed to test it and it failed.

**H3 — binding in cross-terms (`hlens/experiments/binding.py`, `hlens/synonyms.py`, pre-registered in
`hlens/docs/preregistration_h3.md`; results `results/h3/`).** Suite `lens-eval-order-ops` (55 items, each a
number + an operation as intermediates), readout at the last prompt token, scalar = the logit of the target's
first token; direction set per item = the two intermediates' lens vectors + the 6 most output-relevant
(`|⟨∇f, v⟩|`) co-present pursuit atoms as controls; primary statistic = the rank of `|H_AB|` among the other 27
pairs; legs: (i) measured ablation interaction `I = Δ_AB − Δ_A − Δ_B` ranks higher for the binding pair,
(ii) stable across `k ∈ {10, 40}` and layers 14/18, (iii) above the randomized null; all four legs plus the
primary test must pass. Unit tests cover the bookkeeping, e.g. **on an exactly quadratic `f` the measured
interaction equals the Taylor-2 prediction exactly**. **H3 result (pre-registered analysis, `results/h3/analysis.json`): NOT SUPPORTED.** Null arm first (the script
enforces it), then the real arm; **35 of 55 items included** (19 excluded because the 4B model does not answer
the arithmetic correctly at the readout position, 1 for lacking a single-token synonym, e.g. 24). Primary
(Hessian) statistic `T = mean rank-fraction of |H_AB|` = **0.496** (p = 0.55): the binding pair is at the median
of the 28 pairs. Leg (i), measured ablation interaction: `T_I = 0.54`, p = 0.25. Leg (ii): `T` = 0.54 / 0.49 / 0.45 /
0.52 at (k=10, k=40, L14, L18), none significant (the measured-interaction statistic is 0.59 at k=10, p = 0.046, and
0.61 at L14, p = 0.014, but not at the primary setting). Leg (iii): `T_null` = 0.35, and `T_real − T_null` has CI
[0.018, 0.275] — passes, but only because the randomized model sits *below* 0.5, so it does not rescue the claim.
The falsifier's second clause ("high for arbitrary co-present pairs") is what the data say: a binding pair is no
larger in `|H_ij|` than equally output-relevant co-present pairs. **Useful by-products (exploratory):** the second-order
picture is valid at the ablation scale — the Taylor-2 prediction of the measured pairwise interaction is within 50%
for **93%** of pairs and `|H_ij|` correlates with `|I_ij|` (Spearman 0.28); only 17% of items show a super-additive
joint ablation for the binding pair. Caveats: 35 items; one scalar (the target's first-token logit); the realised
synonym is chosen by lens logit; controls are chosen by first-order relevance, which may itself correlate with
curvature; arithmetic is hard for a 4B model, which is why a third of the items drop out.

**H6 (`hlens/experiments/ignition.py`, pre-registered in `hlens/docs/preregistration_h6.md`).** Built and
unit-tested (projection share, transition width, Lanczos leading |eigenvalue|, decision logic). **A single-trial
pilot on a reserved carrier (excluded from the confirmatory set) shaped the design:** transition widths of the lens-free
projection share shrink *gradually* with depth (0.73 at L0–2, 0.52 at L10–18, ~0.28 at L26–30; thresholds ≈ 0.46–0.49 at
all layers) rather than collapsing at one layer, and the second-to-first-order ratio along the path is *smallest* at the
threshold (a sigmoid's inflection) — so a curvature peak exactly at α* is not guaranteed even if H6 holds; this is
written into the pre-registration. Confirmatory design: 24 trials (6 carriers × 4 pairs), layers {4,8,10,12,14,16,20,24},
gap 4, 5 α offsets around α*(L16), Lanczos 8 iterations; tests T1 (the layer of peak median λ lies within ±2 of the
width-collapse window, in ≥ 80% of bootstraps), T2 (λ larger at the threshold than 0.2 away, one-sided Wilcoxon p <
0.01), T3 (above the randomized-model null); supported iff all three. The knee-vs-Taylor-radius diagnosis is
recorded as exploratory.

**H6 result (pre-registered analysis, `results/h6/analysis.json`): the pre-registered λ criteria are met (T1, T2, T3 all
pass) — but the support is much weaker than that headline.** 24 trials, null arm first. Numbers:
- *T1 (passes, trivially).* Median λ at the threshold: L4 15.2, L8 1.64, L10 1.16, **L12 0.97 (minimum)**, L14 1.15, L16 2.89,
  L20 2.30, L24 0.41; the peak layer is **L4**, the median transition width falls gradually (0.67 at L4, 0.56 L8, 0.51 at
  L10–L14, 0.49 L16, 0.46 L20, 0.37 L24; no collapse at one layer), so the "width-collapse window" is the earliest interval
  [L4, L8] and the peak falls inside it (bootstrap fraction 0.98). Both are early-layer effects; the notes' onset band
  (≈ L6–L13 on Qwen) has the *lowest* λ.
- *T2 (passes, but concentrated).* mean D = 1.00, p = 8e-7; **78% of the pooled D comes from L4** (λ larger at the
  threshold than 0.2 away in 100% of trials, +71%), 19% from L20 (92% of trials, +132%), L24 +23% (100%); **L10–L16 show no
  effect and L12/L14 are reversed** (center > flank in 38% / 12% of trials). Excluding L4: mean D = 0.25, p = 0.008. A
  scale-free (relative-gap) version passes strongly (p = 2.6e-8) but is also carried by L4/L20/L24.
- *T3 (passes).* The null model shows no threshold effect (mean D = −0.0005; λ values 0.005–0.05); the CI of the real−null
  difference is [0.84, 1.16].
- *Scale-free τ_path (exploratory) fails everything:* peak layer L20 (outside the window, bootstrap 0.02), T2 negative
  (mean D −0.003, p = 0.99), T3 reversed (null larger).
- *Knee-vs-Taylor-radius diagnosis (mandatory; `results/h6/knee_vs_radius.json`, 5 trials):* at the threshold the quadratic
  picture of the commitment map has already broken down for an α-step of 0.05 at **L16, L20, L24 (0% of cells valid, median
  validity score 0.3)** and is mostly broken at L4 (40% valid); it is valid at L12 and valid at every layer 0.2 away from
  the threshold. So the layers where λ is elevated at the knee (L4, L20, L24) are exactly the layers where the Taylor radius
  has collapsed at the knee — the situation the notes warn about: the curvature estimate there cannot be cleanly separated
  from a loss of validity.
- *Interpretation.* H6's literal prediction ("curvature peaks at the width-collapse layers and near the threshold α") is
  met by the pre-registered criteria, but what the data support is narrower and different: the commitment map is strongly
  nonlinear *at the ambiguity threshold* at early (L4) and late-workspace (L20–L24) layers, **not** at the middle layers
  where the workspace begins, and the early-layer λ magnitude is plausibly a residual-norm scale effect (scale-free τ does
  not show it). Caveats: λ from 8 Lanczos iterations (not checked against the model), one gap (4), one layer set,
  24 trials, knee-vs-radius on 5 trials only, and the width-collapse criterion is weak when widths shrink gradually.

**H5 and H3** are pre-registered (`hlens/docs/preregistration_h5.md`, `preregistration_h3.md`; the
experiment scripts record each file's SHA-256 in every result file and the real-model arm refuses to
run unless the randomized-model arm ran first under the same hash). H3's pre-registration has a dated
addendum (the suite has two operations, `squared` and `mod`, that the first synonym list missed; found
when the null arm crashed during item resolution, before any measurement existed). Results below
when available.

### 3.10 E1 and the secondary hypotheses H7–H11 (session 2, fourth part)

All pre-registered (`hlens/docs/preregistration_{e1,h7,h8,h9,h10}.md`; each experiment records its file's SHA-256; H9 has a dated
addendum written after its precondition result). fp32 on the Thor; outputs in `results/`.

**E1 — the engagement hypothesis (`hlens/experiments/engagement.py`, `results/e1_engagement_qwen3.5-4b.json`): NOT SUPPORTED, and the
data point the other way.** Exact forward passes along the real multi-layer swap path (band L12–L20, raw vectors, α ∈ {−0.05 … 2.0}); margin
`m(α) = logit(swap) − logit(answer)`; first-order prediction `FO(α) = m₀ + α·m′(0)`; nonlinear gain `N = m(1.25) − FO(1.25)`; 51 items, 18
successes. **P1 fails:** AUC(N → success) = **0.39** (CI [0.23, 0.56]), stratified 0.51 — and the mean `N` is *lower* for successes (−8.8) than
failures (−4.6): the response saturates, it does not amplify. **P2 fails:** only **2 of 18** successes would not have flipped under the first-order
prediction alone (p = 0.9999 for "most flips are nonlinearly driven"). **P4 fails:** the single-layer (L16) `N16` AUC is 0.75 in the real model vs
0.56 in the randomized model, but the CI of the difference [−0.04, 0.41] includes 0. Extras (exploratory): **the first-order prediction predicts
success** (AUC(`FO(1.25)`) = 0.75, AUC(`g = m′(0)`) = 0.74) and the actual margin even better (AUC(`m(1.25)`) = 0.85); `N` and `FO` are
strongly anti-correlated (Spearman −0.87). Knee alignment (exploratory): `α_n ≤ α_c` in 89% of successes but Spearman(`α_n`, `α_c`) = −0.34.
**Reading:** on the real intervention, swap success is a large first-order effect on the margin that saturates at α = 1.25; the earlier
association of high τ / large single-layer first-order error with success (§3.7) looks like an effect-size correlate (bigger first-order pushes
give bigger absolute errors), **not** evidence of nonlinear engagement. E1 is retired as stated; the H1 "reversed sign" is best explained by
effect size. (The single-layer L16 `N16` association, 0.75, is not specific to the trained network under the pre-registered bootstrap.)

**H7 — early-layer content in the second-order term (`early_curvature.py`): NOT SUPPORTED.** Multihop suite, 30 baseline-correct items,
true intermediate vs 15 distractors, per-candidate scores `s0 = ⟨v,x⟩` (lens presence), `s1 = |⟨∇f,v⟩|` (sensitivity), `s2 = |vᵀHv|`
(self-curvature). Mean rank-fractions `T`: L4: s0 0.52, s1 0.43, **s2 0.55 (p = 0.19)**; L8: s0 **0.72**, s1 0.53, s2 0.62 (p = 0.026);
L12: 0.70 / 0.44 / 0.61; L16: 0.70 / 0.45 / 0.54. At the only early layer where the first-order lens is blind (L4) the second-order term is
not significant; at L8 the first-order lens already carries content (so condition 1 fails) and `s2` is not significant at the pre-registered
0.01; `s2` never beats sensitivity (CI [−0.07, 0.27]). Note the lens *presence* signal appears by **L8**, earlier than the landmark onset
estimates (L9–L18) — and sensitivity `s1` is at or below chance throughout.

**H8 — motor-regime collapse (`motor_collapse.py`): NOT SUPPORTED.** *Part A passes:* the distance of `J_ℓ` from a scaled identity falls
0.996 (L0), 0.975 (L10), 0.944 (L15), 0.783 (L20), 0.628 (L25), 0.544 (L28), **0.461 (L30)** — `J_ℓ → I` is confirmed. *Part B fails:* the
per-remaining-block nonlinearity of the layer-to-output map (`τ_map/(31 − ℓ)` along random perturbations of 10% of the residual norm,
60 contexts) **rises** toward the output (0.0046 at L6 → 0.0089 at L22 → 0.047 at L30); the motor/workspace ratio is **3.73 [3.52, 3.98]**, not
< 0.5. Raw `τ_map` does fall with depth (0.116 → 0.047), but only because the map has fewer blocks. *Part C:* the randomized model shows the same
rise (ratio 2.06); the real model's rise is steeper (real − null CI [1.46, 1.93]). Spearman(`d_I`, `τ_map`) = 1.0 (a monotone-in-depth
confound, uninformative). Consistent with H2 (random-direction MLP-level ρ does not collapse at L28–L30).

**H9 — capacity as competition (`competition.py`).** *Precondition (H6 sub-question (b)) MET:* at the ignition threshold the cross-terms
`v_Aᵀ H_{r_A} v_B` and `v_Aᵀ H_{r_B} v_B` (readouts = lens vectors at L20, directions = lens vectors at L12) are **negative in 24/24 trials**;
inhibition `I = −(c_A + c_B)/2` has median **0.134** at the threshold (p = 6e-8), exceeds the flanks (0.088 / 0.083; paired p = 3e-7) and the
orthogonal random control (0.0009; p = 6e-8). The first-order cross-effect is *positive* (+0.26): the competition is a second-order,
gain-control-type effect. *Part 2 (list loading, Addendum 1; `results/h9_list_loading_qwen3.5-4b.json`): NOT SUPPORTED.* 2 families (countries, cities) × 8 lists of 32 words,
evaluated at the comma after word k ∈ {4,…,32}; inhibition among the last four listed words' lens directions. **Q1 fails in the opposite
direction:** inhibition *falls* with load — Spearman(k, I) = **−0.51** for k ≤ 24 (median I 0.0107 at k = 4, 0.0062 at 8, 0.0040 at 12, 0.0028 at
16, 0.0026 at 20, 0.0010 at 24, 0.0008 at 28, 0.0035 at 32); the orthogonal control stays flat (ρ = 0.10, p = 0.17). **Q3 fails:** the association of I
with the fraction of listed words absent from the lens top-25 is ρ = 0.20, p = 0.08. Caveats: the pairwise inhibition here is 10–100× smaller than
at the ignition threshold (0.13); the occupancy measure is single-layer (L20, rank ≤ 25 over the whole vocabulary) and counts only 1–5 words
(median 5 at k = 32), far from the paper's ~25, whose measure is the **minimum rank over the workspace band** — so Q3 is under-powered by a
protocol difference; a 4-word window, 2 families, 16 lists. **Net for H9:** the precondition (competition at the ignition threshold) holds, but
it does not scale with load in this test, so "capacity as competition" is not supported.

**H10 — register and persona pairs, H6 protocol (exploratory proxy; `register_persona.py`): NOT SUPPORTED as a proxy (needs both families).**
24 trials per family, layers {4,12,16,20,24}. *Persona pairs reproduce the H6 signature:* threshold at L16 in 100% of trials, widths shrink
with depth (0.67 → 0.35; 96%), and λ is larger at the threshold than 0.2 away at L4 (100% of trials), **L20 (96%, p ≈ 0) and L24 (88%, p ≈ 0)**,
with the same L12 reversal (17%) as the countries. *Register-contrast pairs (formal vs informal nouns) do not:* a threshold exists (100%) and
widths shrink (0.70 → 0.58; 100%), but the curvature effect is absent at L20 (46% of trials, p = 0.73) and L24 (58%, p = 0.054) and at L4 it
is reversed (29%). The country reference reproduces its own earlier result (§3.9). A natural, untested reading: formal/informal synonyms
denote the *same* concept, so there is no competition to resolve; the curvature-at-threshold pattern may be a property of competing concepts.
Caveats: noun pairs only; 24 trials; carriers are noun-slot templates; register/persona operationalised as word pairs, not prompts.

**H11 — reflection training changes curvature: BLOCKED.** It needs a counterfactual-reflection-trained checkpoint and its base model, which we
do not have; no test was run. The tooling that would apply (ignition protocol, restricted Hessian, validity gate, randomized null) is in place.

### 3.11 Second-model replication: Gemma 3 12B (session 3)

**Why.** The research writeup plan (`hlens/docs/writeup_plan.md`) ranks four claims; A (H9 inhibition) and B (H2 curvature) are the ones that need a
second architecture before anyone should believe them. The user asked for Qwen3.6-27B and Gemma 3 12B (both have Neuronpedia lenses).
Pre-registrations: `preregistration_replication.md` (+ Addendum 1) and `preregistration_b_prime.md`. Results are summarised in
`writeup_plan.md` §6–§7 and live in `results/{gates_gemma-3-12b.json, h9_replication/, h2_replication/}`.

**What exists for each candidate model** (checked against `neuronpedia/jacobian-lens`):

| | Qwen3.6-27B | Gemma 3 12B |
|---|---|---|
| Lens file | `qwen3.6-27b/jlens/.../Qwen3.6-27B_jacobian_lens_n1000.pt` (3.3 GB; n=1000 only) | `gemma-3-12b/.../gemma-3-12b-pt_jacobian_lens.pt` (1.4 GB, 775 prompts); `-it` lens exists too |
| Layers | 64 (48 linear + 16 full attention), d=5120, untied head; lens L0-62 | 48 (sliding + global), d=3840; lens L0-46 |
| Weights | 55.6 GB bf16, open — **downloaded** | 24.4 GB bf16, gated (access granted) — **downloaded** |
| fp32 | ~108 GB: does not fit | ~45.5 GiB resident: fits |

**Code added.**
- `hlens/loading.py`: `MODELS` registry (`ModelSpec`), `load_model(key, dtype=, device=)`, `matched_layer(n_layers, ref_layer)` (depth-fraction mapping from Qwen3.5-4B:
  Gemma 18/24/30 for 12/16/20; Qwen3.6-27B 24/33/41). `load_hf` takes `model_id`.
- `hlens/arch.py`: `mlp_prenorm`, `mlp_branch` (Gemma blocks add a **post-feedforward norm**: `x + post_ff_norm(mlp(pre_ff_norm(x)))`), `embed_rows` (Gemma's
  embedding module multiplies by sqrt(d_model); reading the raw weight row for the ignition endpoints would have been wrong by 62×). `gain.py`, `experiments/gain.py`,
  `interpolation.py` now go through these.
- `make_setup(..., clean=False)`: the original stimulus builder writes `prefix(+trailing space) + " " + word`, a double space. Qwen tokenizes it `[' ', ' Word']`;
  Gemma merges it into `['  ', 'Word']` (the first Gemma H9 run aborted with "could not locate the {W} slot", no data). `clean=True` writes one space. Gemma uses it; Qwen was
  re-run with it too (Addendum 1).
- `hlens/experiments/gates.py` (G1 landmarks, G2 swap), `h9_replication.py` (O and HO trial sets, validity sensitivity, `--clean`), `h2_replication.py`
  (null arm + real arm + `analyze`, `--config A|B2`, `analyze_b_prime`). `competition.run` takes `src/tgt/mid/trials/validity/clean`.
- Tests: `tests/tiny_variants.py` + `tests/test_multimodel.py` (tiny CPU Gemma 3 and an untied-head Qwen: slice = hooked forward, fr = rr HVP, MLP branch, embedding scale),
  `tests/test_real_gemma.py` (real fp32: slice exact (diff 0), fr vs rr 5.5e-7, lens reads ` euro` at L36), `tests/test_replication.py` (decision rules, trial sets, prompt ranges).
  Fast suite: 148 passed, 2 skipped, 45 deselected at the last full run (2026-10-18).

**Results.**
- **G1 (landmarks)**: passed on Gemma; workspace range L20-43 (depth 0.43-0.91). **G2 (swap)**: **not passed at the pre-registered α = 1.25**: 8/48 (17%) vs random controls 1/0/0;
  α = 1.0 (post hoc) 14/48 (29%) vs controls 2/0/3; α ≥ 1.5 gives 0 hits.
- **Claim A (H9 precondition)**: Qwen3.5-4B original set (clean spacing) 24/24 both-negative, all three tests pass; Qwen held-out 120 trials: passes (old spacing 97.5% both-negative,
  clean spacing 88%; the validity-gated subset is too small to test with clean spacing: 3 of 120). **Gemma original and held-out: P0b fails** (inhibition is slightly positive in
  76% of trials but is larger on the flank than at the threshold; only ~30% have both cross-terms negative; values are ~1e-4, ~1000× smaller than Qwen's — scale not compared). By the
  pre-registered rule, A is a **Qwen-family finding**, not architecture-general.
- **Claim B (H2) under `preregistration_replication.md`**: **failed its own rules.** T1 passes on both; T2 (median ratio ≥ 1.5) fails on Qwen (1.37; 2.1 at L28-30), passes on Gemma (9.0);
  T3 ("the null ratio's 95% CI contains 1") fails in 7 of 8 null arms because the bootstrap CIs are ~±0.0005 wide while the ratios are 0.997-1.002. The rule was a poor choice (tests
  exact equality); it was **not** changed after the fact.
- **Claim B′** (`preregistration_b_prime.md`, SHA-256 `2b459bc2…`; fresh prompts 200-229 and fit prompts 300-329, new seeds, equivalence margin δ = 0.05): **holds on both models.**
  Qwen: 5/5 nulls within 0.9998-1.0014, real lower bound > 1.05 at 8/8 layers (ratios 1.15 → 2.20, Spearman 0.93), median 1.33. Gemma: 2/2 nulls (1.001, 1.0032), 6/6 layers
  (4.09 → 12.99, Spearman 1.00), median 9.37. B′ is a narrower hypothesis with a different rule, not a rescue of B; the write-up reports both.

**Mixed-precision path (for Qwen3.6-27B, which cannot be held in fp32).** `preregistration_mixed_precision.md` (SHA-256 `6b0f33b8…`) fixed acceptance criteria V1-V5 before any
measurement; `hlens/experiments/mixed_precision_check.py` ran them on Qwen3.5-4B against the existing fp32 H9 run (same α values; bf16 model, fp32 slice and fp32 unembedding directions;
`functional.make_mixed_slice_fn`, `Fp32Unembed`, rotary cos/sin recomputed in fp32). **All five pass**: sign agreement 98.6% (c_A) / 100% (c_B); Spearman of I 0.9987; median relative error
0.34% (90th percentile 0.96%); P0a-c verdicts identical; α* within 0.05 in 24/24 trials (median 0.0007); median I 0.0773 (fp32) vs 0.0769 (mixed); 24 trials in 223 s. Scope: this validates the **claim-A
measurement** only; the H2 curvature measurement (MLP-branch jvp) needs its own check; the 27B has ~4× more blocks in its slice, so the 4B result does not prove it. A bug found on the way: the fp32
rotary recompute ran inside the first `hvp_fr` call and "escaped" a functorch level on reuse; fixed with `torch._C._DisableFuncTorch()` and a cold-cache regression test (`tests/test_mixed_precision.py`).

**Claim A on Qwen3.6-27B (`preregistration_27b_claim_a.md`, SHA-256 `6b4cfcee…`; mixed precision; held-out set, 120 trials; 2.6 h): FAILS.** P0a p = 0.098, P0b p = 0.99 (inhibition at the
threshold 0.0011 is *smaller* than on the flanks, 0.0015/0.0013), P0c p = 0.13; both cross-terms negative in 17% of trials; first-order cross-effect +0.01; **0 of 120 trials have a Taylor-validity score ≥ 1**
(so the validity sensitivity analysis is not testable). Per the decision table A is **specific to Qwen3.5-4B** (it fails on a 27B of the same family and on Gemma 3 12B). Caveats stated in the
pre-registration: the mixed path was validated on the 4B only (8-block slice; the 27B slice has 17 blocks), the lens is the n=1000 fit, and absolute magnitudes are not compared across models; the original
24-trial set was run afterwards (2026-10-19; 0.6 h): it agrees with the held-out set, see below. Process notes: the first 27B attempt computed all 120 trials (4.9 h) and then died before saving, with no traceback in its log; the second attempt (checkpointed) ended with a
`TypeError` in `analyze` (a validity score of `None` for one trial, compared with `>= 1`), which is probably also what ended the first attempt (not proven). All 120 trials were in the
checkpoint (`HO_qwen3.6-27b_clean_mixed.partial.jsonl`) and `--from-checkpoint` rebuilt the result without reloading the model. While the 27B runs, torch holds 79 GiB (allocated = reserved) and the system shows
1-9 GiB available (about 30 GB outside torch's count is unexplained); `MemAvailable` stays near zero but the run completes; `drop_caches` is needed before each large job because the
memory is not returned at process exit.

**H6 and H8 on Gemma 3 12B (`preregistration_h6_h8_gemma.md`, SHA-256 `4cec1c9b…`; depth-matched layers; same statistics): neither is supported.**
- **H6** (24 trials, randomized arm first): T1 passes only trivially (the λ peak is at layer 6, the earliest layer tested, inside the width-collapse window [6, 12]; λ falls monotonically with depth from 6.3 at L6 to 0.02 at L36);
  **T2 fails** (mean flank difference +0.07, p = 0.90) and **T3 fails** (real-minus-null CI [−0.024, 0.175] includes 0); the `τ_path` variant fails too. So there is no curvature peak at the threshold on Gemma.
  Taylor-validity diagnosis: 10 of 50 cells (5 trials × 5 layers × 2 offsets) have score < 1 (median 1.0; L6 and L30 worst, 30%). Qwen's H6 passed the pre-registered tests with weak support; Gemma does not.
- **H8** (60 contexts): **A fails** (`d_I` falls only from 1.00 to 0.83 at the last lens layer 46; the median over layers 15-30 is 0.99), **B fails** (per-block nonlinearity *rises* toward the output: motor/workspace ratio 5.05, CI [4.73, 5.58], vs
  3.73 on Qwen), **C fails** (null ratio 2.33; the randomized model also rises). Per-block nonlinearity grows from 0.04-0.06 at L9-27 to 0.40 at L45 in the real model and from 0.004 to 0.012 in the null.
  `J_ℓ → I` is much weaker on Gemma than on Qwen (0.83 vs 0.46 at the last lens layer). **The Qwen H8 finding generalizes: nonlinearity per block rises toward the output; there is no motor-regime collapse.**
- Run time: H6 ≈ 73 min per arm; H8 ≈ 4 h (the Part B slices run to the output layer).

**Held-out swap suite (built; NOT yet run).** Motivation: G2's Qwen rate (35 %) came from a sweep on the same 87 items it was scored on, and on Gemma the pre-registered α = 1.25 gave 17 % while α = 1.0 (post hoc) gave 29 %.
`hlens/swap_suite.py` generates 165 items from fact tables (SHA-256 of the data file `003ae0cd…`; 123 city-based, 33 element-based, 9 language-based; disjoint from `probe-swap.json`; single-token entities and non-ambiguous
answers under both tokenizers; tests in `tests/test_swap_suite.py` with a fake tokenizer). `hlens/experiments/swap_heldout.py` runs the G2 protocol (bf16; band depth-matched: Qwen L12-20, Gemma L18-30; α = 1.25 primary and
1.0 secondary; three random-direction controls; Wilson intervals, by-category rates, per-item hits) and writes `results/swap_heldout_<model>.json`. The decision rule and outcome table are in `preregistration_swap_heldout.md`.
`run_g2` in `experiments/gates.py` now takes a `path` and returns per-item hits. To run: `.venv/bin/python -m hlens.experiments.swap_heldout --model qwen3.5-4b` then `--model gemma-3-12b` (each minutes, bf16; drop the cache first).

**Held-out swap suite: RESULTS (run 2026-10-17, bf16, `results/swap_heldout_<model>.json`; decision table in `preregistration_swap_heldout.md`).**
| | baseline-correct | α = 1.25 (primary) | α = 1.0 (secondary) |
|---|---|---|---|
| Qwen3.5-4B, band L12-20 | 114 / 165 | **60/114 = 52.6 %** (Wilson [0.44, 0.62]), controls 0/0/0 → **passes** | 38/114 = 33.3 % ([0.25, 0.42]), controls 0/0/0 → passes |
| Gemma 3 12B, band L18-30 | 125 / 165 | **4/125 = 3.2 %** ([0.01, 0.08]), controls 0/0/0 → **fails** | **55/125 = 44.0 %** ([0.36, 0.53]), controls 0/0/0 → passes |
Reading by the pre-registered table: (1) Qwen's dev-suite 35 % was **not** inflated by tuning (held-out 52.6 %; the held-out suite may simply be easier); (2) on Gemma the pre-registered setting (α = 1.25) **fails, and the dev-suite
17 % was not an item-sampling accident** (3.2 % on 125 fresh items); (3) Gemma's α = 1.0 result is **confirmed out of sample (44 %)**, i.e. the dev look at α = 1.0 was not a post-hoc accident, but it is reported as "the pre-registered α was
the wrong setting for Gemma", not as a pass of the pre-registered setting. The swap is a causal, control-separated effect on both models; the useful α is model-dependent (Gemma: ≥ 1.5 gave 0 hits on the dev suite, 1.25 almost none, 1.0 works), consistent
with raw lens vectors being on a different scale there. Categories (hit/baseline-correct): Qwen α 1.25: city-capital 19/32, city-language 22/29, city-continent 11/25, element-symbol 4/18, language-capital 4/8; Gemma α 1.0: city-continent 19/25,
city-language 19/29, city-capital 12/32, language-capital 4/8, element-state 1/13, element-symbol 0/18. **Coverage caveat (not in the pre-registration):** almost no `city-currency` items (2/34 on Qwen, 0 on Gemma) and no `element-state` items on Qwen survive the
baseline-correct filter: scoring is a case-sensitive first-token match, and the models answer "Euro" / "Solid" etc., so the evaluated suite is effectively capital / language / continent / symbol. Tuning α per model on a further split would be a new pre-registration.

**[Superseded by 'H10 at scale' below: the dissociation did not survive 100 pairs per family.] H10 extended (Qwen3.5-4B; `preregistration_h10_extended.md`, SHA-256 `bcd22322…`; 10 fresh pairs per family, fresh carriers 6-11, 60 trials per family, about 1 h): the persona/register dissociation REPLICATES.**
Persona: threshold in 100 % of trials, widths shrink in 100 %, λ larger at the threshold than at ±0.2 at L20 and L24 (both p ≈ 0) → reproduces the signature. Register: threshold 100 %, widths 97 %, but L20 p = 0.039 and L24 p = 0.28 → does not.
Contrast (persona > register, Mann-Whitney on D): L20 p = 5e-10, L24 p = 2e-15. The original H10 proxy statement ("both families reproduce") is still **not supported**. Per pair (median D over L20/L24): all 10 persona pairs are positive (0.09-0.59);
register pairs are mostly near 0 (-0.06 to 0.07) with two exceptions (attorney-lawyer 0.28, footwear-shoes 0.14). Caveats: only 10 distinct pairs per family, so the p-values are optimistic; **the likely reading is semantic distance, not "persona"**: persona pairs
(judge-clown, monk-sailor) are two distinct concepts, register pairs are near-synonyms (car/automobile) whose two endpoints produce almost the same commitment, and the country pairs (distinct concepts) also show the signature. The experiment cannot separate "persona"
from "any pair of distinct concepts"; a control family of distinct non-persona nouns, or an endpoint-distance covariate, would. Gemma was not used because H6 is not supported there (no signature to compare against).

**[Partly superseded by 'H10 at scale' below: persona is not special, but distance does not explain the differences between families.] H10 control (Qwen3.5-4B; `preregistration_h10_control.md`, SHA-256 `b604b954…`; distinct non-persona object nouns matched on endpoint distance): the persona advantage is a DISTINCT-CONCEPTS effect, not a persona effect.**
Matching phase (no curvature): endpoint distance d = ‖h_A − h_B‖ / mean norm at layer 24, mean over carriers 6-11; persona median d = 1.074, register median 0.731 (range 0.49-0.86), 29 candidate pairs; the 10 matched control pairs have median d = 1.075 and median |Δd| = 0.002 (max 0.028, 2.6 % of the persona median), so the match is adequate by the pre-registered rule.
Control trials (60): threshold 100 %, widths shrink 100 %, λ larger at the threshold at L20 and L24 (both p ≈ 0) → **the control reproduces the signature**. Pair-level primary contrast (Mann-Whitney, 10 vs 10 pairs): persona > control **p = 0.85**
(median pair D̄ 0.40 persona vs 0.52 control); control > register p = 0.0014 (median register 0.017). Across all 30 pairs Spearman(d, D̄) = 0.48; the persona indicator in `D̄ ~ d + persona` (20 persona + control pairs) is −0.29, 95 % pair-bootstrap interval [−0.77, 0.04]: no persona-specific component.
Pre-registered outcome: "control reproduces the signature; persona not larger". Reading: the extended-H10 dissociation is explained by endpoint distance (register pairs are measurably closer, d 0.73 vs 1.07), and "persona" nouns are not special among distinct concepts in this protocol. Caveats: 10 pairs per family; the control's pair
D̄ is heterogeneous (ladder-pizza 2.2, rope-lemon 0.95, anchor-pencil 0.90, kettle-compass −0.03); the register pairs are not exact synonyms (d still 0.5-0.86); one distance measure at one layer; Qwen3.5-4B only.

**Case-insensitive scoring + H1/E1 on the held-out items (pre-registered and fully RUN: swap re-scoring, H1, E1): `preregistration_heldout_scoring_h1_e1.md` (SHA-256 `a6dc897a…`).** `hlens.swaps.item_token_ids(tok, item, fold_case=True)` also accepts the lower-case and capitalised forms of an answer
(not for `element-symbol`), threaded through `probe_items`, `is_ambiguous`, `gates.run_g2`, `trust_score.score_item/run_trust_score` and `engagement.run` (all default to the old case-sensitive behaviour, so earlier results reproduce). Plan: (1) re-score the held-out swap suite on both models
(`python -m hlens.experiments.swap_heldout --model <m> --fold-case`, writes `results/swap_heldout_foldcase_<m>.json`); (2) H1 on the folded baseline-correct Qwen items (`python -m hlens.experiments.heldout_h1_e1 h1`): H1-notes (high τ → failure), H1-dev (high τ → success) and
H1-loading, with the original stratified tests; (3) E1 on the same items (`... e1`, then `... analyze`): P1, P2, P4 as in `preregistration_e1.md` plus a new P5 (the first-order margin prediction beats the nonlinear gain). Needs ≥ 15 successes and ≥ 15 failures. Unit tests: `tests/test_swap_suite.py`.

**Held-out swap suite RE-SCORED with case-insensitive matching (run 2026-10-18; `results/swap_heldout_foldcase_<model>.json`; `preregistration_heldout_scoring_h1_e1.md` §1-2): the verdicts do not change.**
Baseline-correct items rise from 114 to 125 (Qwen) and from 125 to 138 (Gemma); `city-currency` recovers 13 items on each model. Qwen3.5-4B (L12-20): α = 1.25 61/125 = 48.8 % ([0.40, 0.57], controls 0/0/0, passes; was 52.6 %), α = 1.0 39/125 = 31.2 % (passes). Gemma 3 12B (L18-30):
α = 1.25 5/138 = 3.6 % (fails; was 3.2 %), α = 1.0 55/138 = 39.9 % ([0.32, 0.48], controls 0/0/0, passes; was 44.0 %). The recovered currency items barely swap (Qwen 1/13, Gemma 1/13 and 0/13), and `element-state` still contributes no baseline-correct item on Qwen (13 on Gemma, 0-1 hits),
so the earlier coverage caveat shrinks but does not vanish. The case-sensitive results above remain valid as the first scoring; the folded numbers are the ones to cite for coverage.

**H10 at scale (Qwen3.5-4B; `preregistration_h10_scaled.md`, SHA-256 `5eb8e6f3…`; 100 persona, 100 near-synonym, 100 distance-matched object-noun control and 66 country pairs, 2 fresh carriers per pair, 732 trials, 6.2 h): the persona/register dissociation does NOT survive scaling.**
- **All four families reproduce the ignition signature** at the trial level (threshold ≥ 99 %, widths shrink ≥ 99 %, λ larger at the threshold at L20 and L24: persona p ≈ 0 / 0, near-synonym p = 2e-4 / ≈ 0, control ≈ 0 / 0, countries ≈ 0 / 0). The earlier "register does not" (10 pairs) was a small-sample result.
  Effect size does differ (median pair D̄: countries 0.49, control 0.36, persona 0.30, **near-synonym 0.058**).
- **T1 matched contrast (100 persona–control pairs, matching adequate: median |Δd| 0.0004, max 19 % of the persona median):** median D̄ difference −0.043, two-sided Wilcoxon p = 0.08 → no persona effect (persona is, if anything, slightly smaller).
- **T2 regression `D̄ ~ d + persona + synonym` (300 pairs, control = reference, pair bootstrap):** distance +1.00 [0.54, 1.51]; persona −0.093 [−0.18, −0.006]; synonym −0.086 [−0.23, 0.07]; equivalence margin 0.091 (25 % of the median control D̄): the persona interval is not inside ±margin (it reaches −0.18), so persona is **not shown equivalent**,
  but it is not above 0 either. **Persona effect beyond distance: no.**
- **T3 dose-response:** Spearman(d, D̄) = 0.185 over all 300 non-country pairs (pre-registered threshold 0.3 not met); within family 0.30 (persona), 0.33 (synonym), 0.33 (control), 0.23 (countries). Distance matters within a family, but between families it does not order the effect: countries have the smallest median d (0.74) and the largest D̄,
  near-synonyms have a larger d (0.87) than countries and the smallest D̄.
- **Pre-registered outcome ("overall Spearman ≤ 0.3"):** endpoint distance as measured does not explain the signature across families, and the earlier dissociation is not a clean persona effect either. Net: "persona" is not special; family labels carry little; pair-level variation (and something other than this distance, e.g. how the noun fits the carrier) carries more.
- Caveats: two carriers per pair; the pools are my choices (the persona pool includes mythical creatures; the near-synonym list mixes register, dialect and plain synonyms); one distance measure at one layer; Qwen3.5-4B only.

**H1 on the held-out items (Qwen3.5-4B fp32, case-folded labels; `preregistration_heldout_scoring_h1_e1.md` §3; `results/trust_score_heldout_qwen3.5-4b.json`; 52 min): the dev-suite association does NOT replicate at the primary label (α = 1.25); a weaker version of it appears at the secondary label (α = 1.0).**
Population at α = 1.25: 126 baseline-correct items, 61 successes / 65 failures (testable). Failure is the positive class.
- **α = 1.25 (primary).** `tau_L16`: naive AUC(fail) 0.507 (permutation p = 0.46); stratified AUC 0.614 (p_greater 0.017, p_less 0.98); loading (`baseline_neg_loading_L16`): naive 0.563 (p = 0.11), stratified 0.489 (p_greater 0.59).
  **H1-notes not met** (it requires both the naive permutation p < 0.05 and the stratified p < 0.05; the naive test fails although the stratified one passes), **H1-dev not met** (stratified AUC is above 0.5, not below), **H1-loading not met**. Table row "no / no": τ does not discriminate failures on the held-out items
  by the pre-registered rule; the stratified test alone leans in the notes' direction (0.61, p = 0.017), driven by the geography stratum (107 items, AUC 0.62) while the "other" stratum (19 items, 4 successes) points the other way (0.37) - the strata disagree, so I do not read it as support.
- **α = 1.0 (secondary).** 126 items, 39 successes / 87 failures. `tau_L16` naive AUC(fail) **0.289** (permutation p = 1.0, i.e. high τ marks success), stratified 0.370 (p_less 0.012): the dev-suite direction (**H1-dev met at this label**). Loading: naive 0.655 (p = 0.001) but stratified 0.576 (p_greater 0.10): not met.
  The first-order effect size does best: `baseline_neg_jdelta_mean` AUC(fail) 0.696 (p = 0.0005), stratified 0.625 (p = 0.015). Spearman(τ_L16, ‖Jδ‖-based `den`) = 0.59.
- **Reading.** τ is mostly a proxy for the size of the first-order push (a larger push gives a larger τ and a larger chance of flipping the answer): with weak swaps (α = 1.0, 31 % success) high τ marks success; with stronger swaps (α = 1.25, 49 %) the association disappears. This is the "effect-size correlate" explanation that E1 supported on the dev suite; the held-out E1 run is the next step.
- Category success at α = 1.25 (hits / baseline-correct): city-capital 19/32, city-continent 11/25, city-language 22/29, city-currency 1/13, element-symbol 4/18, language-capital 4/8, element-state 0/1.

**E1 on the held-out items (Qwen3.5-4B fp32, same 126 baseline-correct items and labels as H1; `preregistration_heldout_scoring_h1_e1.md` §4; `results/e1_engagement_heldout_qwen3.5-4b.json`; 38 min): E1 is NOT supported, and the first-order lens suffices for the swap (P5 passes).**
- **P1 fails:** AUC of the nonlinear gain N for success 0.505 (95 % CI [0.40, 0.61], permutation p = 0.46); stratified AUC 0.559 (p = 0.14).
- **P2 fails:** only 5 of the 61 successes have a first-order prediction that would not have flipped the margin (FO(1.25) < 0); the other 56 flips are already predicted by the first-order extrapolation (binomial p ≈ 1).
- **P4 fails:** layer-16 AUC of N is 0.562 in the real model vs 0.429 in the randomized one; the bootstrap CI of the difference is [−0.013, 0.278], which includes 0.
- **P5 passes (new in this repeat):** AUC of the first-order margin prediction FO(1.25) for success is 0.645 (permutation p = 0.0016), above the AUC of N (0.505); the realised margin m(1.25) has AUC 0.806. On the dev suite the same quantities were 0.75 (FO), 0.85 (margin) and 0.39 (N): the pattern replicates, weaker.
- Pre-registered statement "the first-order lens suffices for the swap" (E1 not supported and P5 passes): **supported.** Mean N is negative for successes and failures alike (−8.2 vs −7.6): the margin path saturates below its first-order extrapolation. Mean FO is larger for successes (17.4 vs 9.1). Exploratory: in 82 % of successes the first-order approximation is judged to have failed (`α_n ≤ α_c`), but Spearman(α_n, α_c) = −0.03.
- Together with H1 (τ is an effect-size proxy), the held-out repeat reproduces the dev-suite picture: swap success is a first-order effect that saturates, and second-order measures add nothing beyond the size of the first-order push.

**Claim A on Qwen3.6-27B, ORIGINAL 24-trial set (caution run, `preregistration_27b_claim_a.md`; mixed precision; run 2026-10-19, 24 trials, finished in one pass with no checkpoint fallback): FAILS, consistent with the held-out set.** P0a p = 0.95 (median inhibition at the threshold **−0.0010**, i.e. not positive), P0b p = 0.997 (flank medians +0.0012 / +0.0015),
P0c p = 0.93; both cross-terms negative in 2 of 24 trials (8 %); first-order cross-effect −0.04; 0 of 24 trials with a Taylor-validity score ≥ 1. The pre-registered decision table row "holds on HO but not on O (or the reverse)" does not apply: both sets fail, so claim A is specific to Qwen3.5-4B (fails on the 27B, on both of its trial sets, and on Gemma 3 12B).

**EXPLORATORY (preregistration_exploratory_a_h10.md, SHA-256 `08f8fe46…`; descriptive, no verdicts) — Study 2: what explains the H10 differences between families (Qwen3.5-4B, 366 pairs, `results/h10_covariates_qwen3.5-4b.json`).**
Associated covariates (pre-stated rule: overall Spearman with D̄ significant at p < 1e-3 and the sign agreeing in ≥ 3 of 4 families): endpoint distance at layer 24 (+0.185, p = 4e-4; positive in 4/4 families), input-embedding cosine (−0.194; 4/4 negative), unembedding cosine (−0.193; 4/4 negative),
**slot surprisal (−0.250, p = 1e-6; 4/4 negative: nouns that fit the carrier slot better have a larger effect)** and word length (−0.227; 3/4 negative, probably a frequency proxy). **Not associated:** output divergence `js_output` (−0.011), `slot_logp_gap`, d8, d16 (d28 +0.165, p = 0.0016, just misses the 1e-3 threshold).
Explanatory power is low: R² = 0.113 (family indicators), 0.147 (covariates), 0.236 (both). **The family gap is NOT largely explained** (rule: > 50 % shrinkage): persona vs control −0.097 → −0.076 (22 % shrinkage), near-synonym −0.234 → −0.162 (31 %), and the countries advantage *grows* from +0.145 to +0.315 once the covariates are in the model
(countries have the smallest distance and unfavourable covariates yet the largest effect). The two pre-stated candidates split: slot fit is associated, output divergence is not. Whatever makes country pairs strongest is not captured by these covariates; correlational, two carriers per pair, words reused across pairs.

**EXPLORATORY Study 1: claim A scan over layer pairs and direction sources (`preregistration_exploratory_a_h10.md`; original 24 trials; `results/h9_scan/<model>.json`; Qwen 1 h, Gemma 2.6 h).**
- **Qwen3.5-4B (positive control):** the sanity check holds (lens, 12 → 20: all three tests p ≤ 3e-7, 24/24 both-negative). **14 of 18 configurations pass with the Bonferroni threshold (0.01/18):** all lens configurations with gap ≤ 8 except 12 → 24 (peak test p = 0.11, 33 % both-negative), 16 → 24 (peak p = 0.18) and 16 → 28 (nothing significant) fail,
  i.e. the effect disappears when the target layer is near the output (m ≥ 24); identity (logit-lens, no J) directions pass at 8 of 9 (both-negative falls from 100 % to 42-50 % at the longer gaps; 16 → 28 misses the corrected threshold). So on the 4B the effect is broad in layer pair and does not depend on the J-lens.
- **Gemma 3 12B:** **no configuration passes, with or without correction** (the peak-at-threshold test P0b never gets below p = 0.048, usually 0.4-1.0). Inhibition is weakly positive and beats the random control in 14 of 18 configurations (P0a and P0c both p < 0.01 in all 9 lens configurations and 5 identity ones), without peaking at the threshold; the share of trials with both cross-terms negative is
  29-83 % with lens directions (best 12 → 18: 83 %, 24 → 42: 79 %) and 4-67 % with identity directions (the lens-free directions do worse). Reading by the pre-registered table: **"no Gemma config passes, the Qwen sanity check holds": no support for a layer-pair or direction explanation within this grid; claim A stays Qwen3.5-4B-specific.**
  What does exist on Gemma is a weak, threshold-unspecific positive inhibition. Limits: 24 trials per config; a coarse grid (3 source layers × 3 gaps, one threshold layer, one stimulus set); identity is one lens-free alternative.

**Combined `-m real_model` run (2026-10-19, after all session-3 edits; `results/logs/combined_real_model.log`): 45 passed in 27 min 43 s, 0 failed.** One session loaded the Qwen bf16 and fp32 fixtures, a freshly randomized model and (module-scoped) Gemma 3 12B fp32. Slowest: the jspace paper gate 558 s, the stress test 333 s, the probe-swap sweep 199 s, the null-pipeline timing test 127 s,
the Gemma fixture setup 94 s. The three result files the tests rewrite (`gain_qwen3.5-4b.json`, `landmarks_qwen3.5-4b.json`, `stress_test_qwen3.5-4b.json`) were regenerated **byte-identical** to the backups in `results/old/pre_combined_run/`. Fast suite at the same point: 151 passed, 2 skipped.

**Lanczos convergence check for the H6 leading-eigenvalue estimates (Qwen3.5-4B; `preregistration_lanczos_check.md`, SHA-256 `b9d98791…` with Addendum 1; 8 of the 24 trials × 8 layers × 5 offsets = 320 cells, 40-iteration runs, two start vectors; `results/lanczos_check_qwen3.5-4b.json`; 41 min): the 8-iteration estimates used for H6 are adequate.**
- **Reproduction check passes:** the stored H6 value equals the seed-0 8-step value in 320/320 cells.
- **C1 holds:** |λ₈ − λ₄₀| / λ₄₀ has median 0.01 %, 90th percentile 0.7 %, maximum 17.8 % (worst cells: L8 at +0.2 in trial 3, L14 at 0 in trial 15, L20 at −0.1 in trial 6). The 16-step estimate equals the 40-step one to numerical precision, so the converged reference is genuinely converged. Start vectors 0 and 1 differ by a median 0.01 % (90th percentile 1.5 %) at 8 steps.
- **C2 holds:** the sign of the decision statistic D agrees between λ₈ and λ₄₀ in 100 % of the (trial, layer) pairs; mean D is 1.101 (λ₈) vs 1.099 (λ₄₀), Wilcoxon p = 0.0011 vs 0.0013. **So the Qwen H6 results are not an artifact of Lanczos truncation.**
- **Residual-scale effect (descriptive):** median λ₄₀ by layer: L4 14.07, L8 2.18, L10 1.13, L12 0.95, L14 1.24, L16 3.33, L20 2.62, L24 0.48. Scaled by the residual norm (λ₄₀ × ‖x‖): L4 78.4, L8 17.6, L10 11.3, L12 9.3, L14 14.7, L16 39.1, L20 49.8, L24 14.1. L4 is still the maximum, but its lead over L20 shrinks from 5.4× to 1.6× and L16 and L20 come up:
  the "L4 peak" of H6 is partly a residual-scale effect (small residuals at early layers), and the pre-registered row for a differing profile applies. This does not change any H6 verdict.

**[RUN - see the B′ result below] B′ on a third model: Llama 3.1 8B (prepared, then run).** `preregistration_b_prime_llama.md` (SHA-256 recorded by `h2_replication.py --config B3`): B′ rules unchanged; null seeds 20, 21, 22 at layer 16 and seed 20 at layers 8 and 24; real arm L8, 16-30; fresh prompts 200-229 (eval) and 300-329 (fit).
Code: `MODELS["llama3.1-8b"]` in `loading.py`, `CONFIGS["B3"]` in `h2_replication.py` (prefix `B3_`), a tiny dense-Llama CPU model in `tests/tiny_variants.py` (slice = hooked forward, fr = rr HVP, MLP branch, unembedding all pass). The Neuronpedia lens (`llama3.1-8b/.../Llama-3.1-8B_jacobian_lens.pt`, layers 0-30, d = 4096, 539 prompts, finite) is downloaded.
**Blocker:** `meta-llama/Llama-3.1-8B` is gated with manual approval (16.1 GB of safetensors); the HF account in use gets a 403 until access is granted. After access: `hf download meta-llama/Llama-3.1-8B`, then the null arms (`python -m hlens.experiments.h2_replication null --model llama3.1-8b --seed 20 --layers 16 --config B3`, seeds 21 and 22 at 16, seed 20 at `--layers 8 24`), then `real --model llama3.1-8b --config B3`, then `analyze --config B3`.

**E1 on Gemma 3 12B, held-out swap suite (`preregistration_e1_gemma.md`, SHA-256 `3df0e5fa…`; α_ref = 1.0; band L18-30, single layer 24; `results/e1_gemma_heldout.json`; real stage 1 h 10 min, null stage 33 min): E1 is NOT supported and the first-order lens suffices for the swap on Gemma too (P5 passes).**
138 baseline-correct items, 57 successes at α = 1.0 (case-folded labels). **P1 fails:** the AUC of the nonlinear gain N for success is **0.247** (95 % CI [0.17, 0.33]; permutation p = 1.0; stratified 0.313): successes have the *more negative* N (mean −19.9 vs −7.5), i.e. they saturate further below the first-order extrapolation.
**P2 fails:** only 2 of the 57 successes have FO(1.0) < 0 (they would not have flipped under the first-order prediction). **P4 passes on its own** (layer-24 AUC of N 0.736 real vs 0.425 in the seed-10 randomized model; difference CI [0.183, 0.437]), but E1 needs P1 ∧ P2 ∧ P4. **P5 passes:** the AUC of the first-order prediction FO(1.0) is **0.807** (permutation p = 1e-4), against 0.247 for N, and the realised margin m(1.0) reaches **0.956**.
Pre-registered statement "the first-order lens suffices for the swap on Gemma" (E1 not supported and P5 passes): **supported** - now on two architectures (Qwen3.5-4B: FO 0.645, margin 0.806; Gemma 3 12B: FO 0.807, margin 0.956). Mean FO is larger for successes (22.2 vs 3.8). Null-lens check passed (transported cosine 0.825 vs identity 0.726, so the rebuilt randomized model matches its lens).
Category hits: city-continent 20/25, city-language 19/29, city-capital 13/32, language-capital 4/8, element-state 1/13, city-currency 0/13, element-symbol 0/18. Procedure note: the first `analyze` call failed with a `KeyError: 'loading_mean'` because `analyze_e1` assumed the workspace-loading column that the Gemma run does not compute (exploratory only); `analyze_e1` now skips that logistic comparison when the column is absent and the analysis was rerun; no pre-registered test was affected.
Not run: H1 (τ) on Gemma (needs HVPs, separate longer run).

**B′ on a third architecture: Llama 3.1 8B (dense decoder; `preregistration_b_prime_llama.md`, SHA-256 `684bf465…`; fresh prompts 200-229 / 300-329; null seeds 20, 21, 22 at L16 and seed 20 at L8 and L24; `results/h2_replication/B3_*.json`; nulls 23-36 min each, real arm 12 min, 2 h in total): B′ HOLDS.**
- **N:** all five null arms lie inside [0.95, 1.05]: seed 20 L16 1.0016 ([1.0012, 1.0019]), seed 20 L8 1.0008, seed 20 L24 0.9994 ([0.9991, 0.9998]), seed 21 L16 1.0007, seed 22 L16 1.0009. **R:** the real lower bound is above 1.05 at 8 of 8 layers at depth ≥ 0.5 (L16-30). **C:** L16 real lower bound 1.385 vs the largest null upper bound 1.002; L24 1.305 vs 1.0.
- **Real ratios (J-lens / random κ):** L16 1.40, L18 1.27, L20 1.40, L22 1.39, L24 1.34, L26 1.84, L28 2.11, **L30 2.83**; median 1.40 (below 1.5, so not "large", like Qwen); Spearman(layer, ratio) = 0.67 (0.93 on Qwen, 1.00 on Gemma): the effect is present from L16 but rises less smoothly with depth here. L8 (depth 0.26, exploratory): 1.58 (CI lower bound), also above the null region.
- **Gain (g1) lens/random:** 0.88-0.99 up to L24, 1.04 at L26, 1.19 at L28, 2.52 at L30: curvature is raised while first-order gain is not (as on Qwen; on Gemma both are raised). Neuron-direction κ ratios are 1.06-1.40.
- **B′ now holds on three architectures** (hybrid linear-attention Qwen3.5-4B, sliding + global Gemma 3 12B, dense full-attention Llama 3.1 8B); null ratios across 20 randomized-model arms: 0.994-1.0032. Real-weight checks before the run (`tests/test_real_llama.py`, 6 tests): shapes, lens reads "Paris" at L20 and L26, slice = hooked forward exactly, MLP branch, fr vs rr HVP 1.2e-6.
Hash note: the first hash reported (`429b368f…`) was computed before I fixed garbled wording in the limits paragraph; the file was edited before any run, and the result files record `684bf465…`.

**H1 on Gemma 3 12B, held-out swap suite (`preregistration_h1_gemma.md`, SHA-256 `7087d444…`; τ at layers 18, 24, 30, mid layer 24; band L18-30; α = 1.0 primary; `results/trust_score_heldout_gemma-3-12b.json`; 1 h 27 min): the dev-suite reversal REPLICATES on a second architecture; H1 is not supported.**
Population at α = 1.0: 138 baseline-correct items, 57 successes / 81 failures (testable). **H1-notes not met** (`tau_L24` AUC for failure 0.259, permutation p = 1.0: high τ marks success, not failure). **H1-dev met:** stratified AUC 0.364, p_less = 0.007. **H1-loading not met** (negated loading: naive AUC 0.691, p = 5e-4, but stratified 0.561, p_greater = 0.14).
Pre-registered row "no / yes": the reversal (τ marks success) replicates on Gemma. Descriptive: the first-order effect size predicts failure (`baseline_neg_jdelta_mean` naive AUC 0.646, p = 0.0035; stratified 0.615, p = 0.015; Qwen at α = 1.0: 0.70); Spearman(τ, ‖Jδ‖-based `den`) = 0.437 (Qwen 0.59); layer-wise τ AUC for failure 0.289 (L18), 0.259 (L24), 0.446 (L30);
within strata: geography 0.359, other 0.867 (31 items, 1 success: noise). α = 1.25: only 5 successes in 138, untestable (reported only). Compared with Qwen: on Qwen the reversal appeared at the secondary α = 1.0 and vanished at the primary α = 1.25; on Gemma it appears at its working strength. In both models τ behaves as a proxy for the size of the first-order push.
Procedure: `trust_score.analyze` now takes `mid_layer` (the stored Qwen analyses are reproduced exactly) and `run_trust_score` can checkpoint per item.

**Things that went wrong (patterns worth knowing).**
- **Memory.** The Thor shows ~54-72 GB "used" with no visible process (page cache + pinned/leaked allocations); `MemAvailable` then is ~50-65 GB and the fp32 Gemma null arm was killed by the
  kernel OOM killer (no Python traceback; the log just stops; confirm with `dmesg | grep -i "killed process"`). `sudo sysctl vm.drop_caches=3` (the user ran it) restores ~117 GB. Run
  one large job at a time.
- **Reboots.** The Thor rebooted itself twice during long jobs (`last -x` shows "crash"). A reboot kills background queues and clears `/tmp`, so logs there are lost; `results/` survives.
  The Gemma null lens fit (hours) is cached in `results/h2_replication/` and is reused on relaunch.
- **Null construction without two resident copies.** `null_arm` randomizes in fp32 on the CPU, rounds weights to bf16 precision, fits the lens on the bf16 model, then upcasts to fp32 (exact).
- **Shell traps.** `pgrep -f "pattern"` inside a wait loop matches the loop's own command line (and `pkill -f` kills the calling shell, exit 144). Wait on a result file
  (`until [ -f ... ]`) or use the `[x]pattern` bracket trick. A foreground `sleep` is blocked by the harness.
- `uv run` fails with "No environment file found at: .env" if `UV_ENV_FILE=.env` is exported in `~/.bashrc` and no `.env` exists; `touch .env` or call `hf`/`python` directly in the activated venv.

**Still open from this stretch.** (done since this paragraph was first written: the Qwen3.6-27B claim-A run [fails], H6 and H8 on Gemma [neither supported], the mixed-precision validation, and the held-out swap suite build.)
Remaining: ~~the held-out swap suite~~ (run, and re-scored case-insensitively; §3.11); ~~H10~~ (extended, control and 100-pair scale-up all done: the persona/register dissociation does not survive scaling; no persona effect); ~~H1 and E1 on the held-out items~~ (done: H1 does not replicate at α = 1.25 and shows the dev direction at α = 1.0; E1 not supported; the first-order lens suffices, P5 passes); ~~the 27B original 24-trial set for claim A~~ (done: fails, consistent with the held-out set); ~~why claim A fails off Qwen3.5-4B~~ (scan done: no layer-pair or direction explanation within the grid); ~~what explains the H10 between-family differences~~ (covariates done: slot fit, distance, embedding similarity and word length are associated; the family gaps are not explained); optional 27B real-only H2 arm (needs the mixed path validated for the curvature measurement; the 27B null lens fit is not recommended); the `preregistration_replication.md` table still lists 32 for the 27B mid layer where `matched_layer` gives 33 (the 27B pre-registration uses 33; a dated addendum would settle it); housekeeping (commit, combined `-m real_model` run, MAXN benchmark; the README note is done).
pre-registration); optional 27B real-only H2 arm (needs the mixed path validated for the curvature measurement; the 27B null lens fit is not recommended); the `preregistration_replication.md` table still lists 32 for the
27B mid layer where `matched_layer` gives 33 (the 27B pre-registration uses 33; a dated addendum would settle it); housekeeping: commit (yours), the MAXN benchmark (optional, needs your sudo); DONE: the combined `-m real_model` run (45 passed), the README torch-wheel note, the Lanczos checks.

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

**Session 2, second half — things worth remembering:**
- **SDPA attention has no forward-mode autodiff** (`NotImplementedError: Trying to use forward AD
  with _scaled_dot_product_flash_attention...`), which `hvp_fr` / `hvp_norm_form` need. Always
  `attn_implementation="eager"`; the tiny test model too.
- **`randomize_model(reinit)` is a no-op unless the `_is_hf_initialized` flags are cleared**
  (loaded weights carry them and `init_weights()` skips flagged modules/params/buffers).
- **`unembed_directions` needs autograd:** calling it inside a `torch.no_grad()` block raises "element
  0 of tensors does not require grad" (hit in the gain code); wrap in `torch.enable_grad()`.
- **`jlens.fit`'s atomic checkpoint needs its directory to exist** — created the first run and lost
  ~6 minutes of fitting; `fit_null_lens` now makes it.
- **Assumptions about noise scales were wrong twice:** fd at ε=1e-3 is already usable for a scalar
  logit (the "noise" ε is 1e-4); and the toy `eps_sweep` test failed on float32 rounding.
- **Gate rules that assumed Claude's curve shapes misfired** (kurtosis starts *high* on Qwen);
  anchor "onset" at each metric's minimum and keep per-metric estimates visible.
- **A calibration metric can be vacuous:** the first-order averaging ratio is ~1 for a randomized
  model too (residual skip path), so it proves nothing about `J_l`.
- **Two models of one process share the GPU fine, but timing is only meaningful when alone.**
- **A tokenizer bug contaminated the swap labels (found when the margin variant produced NaNs).**
  Qwen tokenises `" 5"` as `[" ", "5"]` (digits do not merge with a preceding space), and the
  first version of the answer-token sets took the first token of both `word` and `" " + word`, so
  every numeric answer contained the bare-space token: 14 of 88 items "overlapped", the margin
  `logit(swap) − logit(answer)` was identically 0 (NaN τ), and a space top-1 would have counted as
  a correct answer and as a swap hit. Fixed in `hlens/swaps.py` (`first_token_ids` skips
  whitespace-only tokens; `is_ambiguous` / `probe_items(drop_ambiguous=True)` drop the one item
  whose answers still share a first token, `chem-bones-Z` '20' vs '26'; 88 → 87 usable items).
  **Everything depending on the ids was re-run** (swap sweep, full H1 run, variants); the
  headline numbers moved by ≤ 0.03 AUC / 1 item (old files are in `results/old/`). Lesson: a
  `NaN` in a new metric is a data signal — look for degenerate inputs before touching the stats.

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

**Git state at time of writing:** the first half of session 2 is committed (`0a9eb9d "Added core
for hlens"`). **The stage A/B/C work is uncommitted:** modified — `hlens/harness.py`,
`hlens/hvp.py`, `hlens/jspace.py`, `tests/test_harness.py`, `tests/test_hvp.py`, `tests/test_functional.py`
(if touched), this doc; untracked — `hlens/{contexts,gain,interpolation,landmarks,variants}.py`,
`hlens/experiments/{ablation_battery,build_nulls,early_averaging,stress_test}.py`,
`tests/{tiny_qwen,test_benchmark,test_early_averaging,test_gain,test_interpolation,test_landmarks,
test_null_pipeline,test_stress}.py`, `results/`.

**Stage A/B/C environment notes.** `datasets` is installed (WikiText loader). `results/` holds
the artifacts above; `*.pt` is already gitignored, the JSON/CSV files are small. The Thor is in
the `120W` power mode — MAXN needs `sudo nvpmodel -m 0 && sudo jetson_clocks`, and only
`nvpmodel -q` can be read without root (clocks cannot), so `benchmark` verifies the mode only.
`jlens.fit` on a 4B model: ~62 s per 64-token prompt. Memory with two jobs (a fit + a battery)
was ~65 GB of 122 GB.

---

### 6b. Environment and model state added in session 3

- Cached models (`hf cache ls`): `Qwen/Qwen3.5-4B` (9.3 GB), `Qwen/Qwen3.6-27B` (55.6 GB, complete: 29/29 files), `google/gemma-3-12b-pt` (24.4 GB, complete), `neuronpedia/jacobian-lens`
  (5.5 GB: includes the Qwen3.6-27B and Gemma 3 12B lenses), plus an unrelated Nemotron 120B (80 GB). `Qwen/Qwen3.5-27B` has only config stubs. Gemma needs HF access approval.
- Run Gemma and Qwen jobs with `uv run --no-sync python -m hlens.experiments.<name>`; model keys: `qwen3.5-4b`, `qwen3.6-27b`, `gemma-3-12b`, `gemma-3-12b-it` (`hlens.loading.MODELS`).
- Memory: see §3.11 (OOM kills, `vm.drop_caches`, reboots). A full fp32 Gemma 3 12B is ~45.5 GiB; fp32 Qwen3.6-27B (~108 GB) does not fit.

## 7. Next steps, in order

Done: everything in the old steps 1–5; **stages A, B, C** (§3.8): real-model checks of
`hvp_norm_form` / `restricted_hessian` / the harness, the layer-landmark and MLP-gain gates, the
early averaging test, `randomize_model` + null lens + null distributions, `ablation_table` + battery,
`benchmark`, and the pathological stress test (**the Phase-2 gate passes**).

Gate status after stage B: sparse decomposition ✅ (caveats, §3.3); coordinate swap 🟡 partial
(§3.4); layer landmarks ✅ (workspace ≈ depth 0.3–0.9); **MLP gain ❌ not reproduced** (§3.8).

0. **Housekeeping.** Commit the stage A/B/C work. (The combined `-m real_model` session passed: 36 tests,
   26 min 41 s.) Add a README note that
   the torch in `uv.lock` is stock PyPI `+cu130` (with `sm_110`) and must not be silently replaced.
   For a comparable benchmark: `sudo nvpmodel -m 0 && sudo jetson_clocks`, then
   `uv run pytest tests/test_benchmark.py -m real_model -s` (and decide whether the "≤ 2–3×
   `lens.apply`" milestone should be restated per-HVP, since a restricted H costs k HVPs).
1. ~~**`experiments/trust_score.py` orchestration (H1)**~~ — **built and run** (§3.7): H1 not
   supported (τ high → swap success; AUC 0.78 naive, ~0.66 stratified, p ≈ 0.07; workspace loading is
   the better predictor of failure, stratified AUC 0.72, p = 0.017), **variants done**: τ tracks the
   true first-order error (ρ 0.95 / 0.81) but first-order error does not predict failure. **Next on
   H1, if pursued:** (a) a held-out protocol (α/band were tuned on the same items) and more
   items/categories — n = 51 is the binding limit; (b) per-position variants at more layers with a
   null at each; (c) test the "engagement" reading directly (does τ / true error predict the *size*
   of the margin change, or the ignition knee of the interpolation experiment?) — this connects to
   H6; (d) accept the result and move on to H2/H3/H5/H6, where τ's validity role (gating curvature
   claims) is what matters.
2. ~~**`gain.py` (H2)**~~ — **done** (§3.9). ~~**`binding.py` (H3)**~~ — **done, not supported**. ~~**`remainder_coupling.py`
   (H5)**~~ — **done, not supported (twice)**. ~~**`ignition.py` (H6)**~~ — **done**: pre-registered criteria met, interpretation
   weak (§3.9). ~~**H7–H10, E1**~~ — **done** (§3.10). Still open: **H4 / `automatization.py`** and **H11** (both blocked on checkpoints).
3. **Across all results (H1–H3, H5–H10, E1):** nothing so far supports the idea that second-order structure adds explanatory power beyond
   the first-order J-lens on this model: τ is a valid nonlinearity measure but marks swap *success* (H1); binding pairs
   are not special (H3); the remainder is not specifically coupled to selected atoms (H5); H6/H2 find real, learned
   curvature structure (null-controlled) but not at the places the hypotheses named; E1 shows swap success on the real intervention is a first-order
   effect that saturates; H8 shows `J_ℓ → I` but no curvature collapse; H9/H10 find a second-order competition effect at the ignition threshold that
   is specific to persona/country pairs and absent for register synonyms, but does not scale with list load. Reasonable next moves: (b) a larger/held-out swap suite so H1/E1 are not limited by n = 51; (c) check
   the Lanczos leading eigenvalue against a converged one on a few cells and normalise λ by residual norm; (d) decide
   whether to continue with H7–H11 or to write up the validated tooling plus these null/negative results.
4. **Open threads from earlier gates:** the swap α=2 divergence from the paper and a held-out
   split for the α/band choice (§3.4); `k`-sweep and more-layers control for the jspace gate
   (§3.3); the `shuffle` null and more seeds (§3.8); a larger-N / more-layers / more-output-token
   early-averaging run, since one layer and one token is thin; replace the 30-prompt null lens by
   a larger fit if the null becomes load-bearing (~1 min per prompt per layer set).
5. **`vis_ext.py`** (pairwise-interaction heatmap with validity opacity and a randomized-null
   panel) — lowest priority, after the first experiments.

6. **Session 3 follow-ups (second model).** Order suggested: (a) ~~Qwen3.6-27B~~ done: claim A **fails** on the 27B (mixed precision validated on the 4B first); B′ on the 27B is not recommended (needs a null lens fit; the nulls have all been flat); (b) ~~pre-register and run H6 (ignition) and H8 on Gemma~~ done, neither supported; (c) ~~held-out swap suite~~ run: Qwen passes at α = 1.25, Gemma fails at 1.25 and passes at 1.0 (§3.11); (d) H10 on
   Gemma with more pairs; (e) decide how to write up: claims A (Qwen only) and B′ (two models) with the negatives (`hlens/docs/writeup_plan.md`). Housekeeping: commit (~90 files; the README torch-wheel note is DONE), re-run the combined
   `-m real_model` suite (it now includes `test_real_gemma.py`), `sudo nvpmodel -m 0 && sudo jetson_clocks` then the benchmark, README torch-wheel note, Lanczos convergence checks.

---

## 7b. Hypotheses being tracked (open, not yet tested)

**E1 — the engagement hypothesis (arose from the H1 result; to be tested with H6).**
*Statement.* A lens-coordinate swap succeeds when it pushes the network into a strongly
nonlinear, *engaged* regime (an ignition-like commitment to the swapped concept); the
second-order measures (τ, the true first-order error) mark that regime, which is why they
correlate with swap **success**, not failure (§3.7).
*Evidence so far.* τ tracks the true first-order error at ρ = 0.95 / 0.81; large first-order
error goes with success (AUC(failure) 0.25 / 0.33); the high-τ categories are exactly the
best-swapping ones (geography facts); `moved off the answer` jumps sharply between α = 1.0 and
α = 1.25 on the best band (14/53 → 43/53); in the stress test (§3.8) the interpolation knee has
72% single-step HVP error vs 6% on control paths. *None of this is a direct test*: it is all
correlational, on n = 51, and the per-layer errors are for single-layer perturbations.
*Predictions to test (design, cheap, mostly exact forward passes):*
 (a) along an item's swap path `h + αδ` (single layer, then the 9-layer band), the margin
     `logit(swap_answer) − logit(answer)` as a function of α has a **knee**, and the true
     first-order error `err1(α)` rises steeply at that knee; successful swaps are those whose
     α = 1.25 lies past the knee, failed ones stop short of it;
 (b) per item, the minimal α at which the swap succeeds coincides with the err1 knee (the
     sharp 14 → 43 "moved off" jump should be the population version of this);
 (c) in the ignition interpolation (`hlens/interpolation.py`), swapped-concept lens share
     crosses 0.5 exactly where the layer-to-layer curvature peaks (H6's own prediction), and the
     probe-swap items that succeed are those whose swap delta crosses an interpolation-type knee;
 (d) both must fail the randomized-weights null, and survive stratification by category and
     conditioning on workspace loading (the paper's predictor).
*Falsifier.* Swap success is independent of whether the item's α = 1.25 lies past its err1/margin
knee, once workspace loading and category are accounted for; or the knee/curvature relation also
appears in the randomized model.
*Where it runs.* Natural home: `experiments/ignition.py` (H6), which already needs the interpolation,
share curves, stress-test machinery and validity gate. The per-item α-scan in (a)/(b) needs only
forward passes and could be run earlier if wanted.
*Update after H6 (2026-10-14):* the H6 pilot and results bear on (c). Along the interpolation path the scale-free
second-to-first-order ratio τ is *smallest* at the threshold (the sigmoid's inflection) and the Taylor radius *collapses* at
the threshold at L16–L24 (validity score 0.3), so "curvature peaks at the knee" should be restated for E1 as "the knee
is where the quadratic picture fails / where the response is strongly nonlinear". That is consistent with, but does not
test, E1: it does not link swap success to crossing a knee. Parts (a), (b), (d) remain untested.
*Status:* **TESTED 2026-10-14 — NOT SUPPORTED** (§3.10): the first-order prediction, not the nonlinear gain, predicts swap success on the real intervention.

---

## 8. Key source documents (for re-reading, not re-deriving)

- `hlens/docs/paper_primer.md` — the PRIMER for drafting the paper (2026-10-20): methods and why, hypotheses, every experiment and result with numbers, claims ledger, narratives, pitfalls, limitations, file map, suggested outline.
- `hlens/docs/writeup_draft.md` — the DRAFT write-up (session 3): summary, methods, results by hypothesis, limitations, where each number lives. `hlens/docs/writeup_plan.md` — claim ranking, decision rules and result tables.
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
- Paper pieces additionally read in stage B/C: Fig 27/28 text (landmarks), §4.3.1 + Fig 32 (gain;
  read as an image, PDF page 35), A.7 (ablation design space), the ignition protocol in
  `data/experiments/README.md`. Still unread: §4.1.1 (ignition interpretation), §4.2 (capacity),
  §A.15/A.17 (needed for H6/H9), §5 (automatization).
- `tests/tiny.py` — the `TinyDecoder` fixture every toy test uses; `tests/tiny_qwen.py` — a tiny CPU
  Qwen3.5 for HF-specific unit tests.
- `hlens/loading.py`, `tests/conftest.py`, `tests/test_real_qwen.py` — the real-model
  entry points (session 2).

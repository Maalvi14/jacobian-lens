# H-Lens Code TODO

Everything below is derived from reading the real `jlens` source (not the overview), the paper
(§2.3, §2.5, §4.3, §A.7, §A.8, §A.24.2), and Roadmap v2. Items are grouped by module, in
dependency order. Roadmap phase tags in brackets.

**Package decision (do first):** build `hlens` as a *sibling package* that imports `jlens`,
mirroring its patterns (keyword-only args, Protocol typing, validation at edges, lazy imports).
Don't fork `jlens` — you want to stay mergeable with upstream fixes, and `jacobian_for_prompt`,
`ActivationRecorder`, `valid_position_mask`, and `JacobianLens` are all public API you can
import directly.

```
hlens/
├── functional.py      # pure layer-slice callables (unlocks torch.func)
├── hvp.py             # three HVP estimators + restricted bilinear forms
├── jspace.py          # gradient pursuit, J-lens vectors, remainder   ← missing from jlens
├── interventions.py   # steering / ablation / coordinate swaps        ← missing from jlens
├── harness.py         # ε-sweeps, MS-HVP, validity, randomized null
├── data.py            # loaders for jlens/data/experiments/*.json
├── experiments/       # one script per Phase-2 letter
└── vis_ext.py         # validity-overlaid heatmaps + null panels
```

---

## 0. Corrections to earlier analysis (affects scope)

- [x] **Per-context Jacobians need NO patch.** `jacobian_for_prompt` is exported in
  `jlens.__all__` and returns per-prompt `J_ℓ` directly — `fit` discards them, but you can call
  the per-prompt function yourself for 2B/2C. Drop the "export patch" item from the roadmap.
- [ ] **Interventions are missing from `jlens` entirely** — I under-flagged this earlier.
  `hooks.py` only *records*; there is no steering, ablation, or coordinate-swap code anywhere in
  the repo, yet 2C (swap-failure validation) and 2E (causal three-leg criterion) both depend on
  the paper's §2.5 write operations. This is a required new module (see §4 below).

---

## 1. `functional.py` — pure layer-slice wrapper  [Phase 1a, blocking for forward-over-reverse]

`torch.func.jvp(torch.func.grad(...))` requires a pure function; the hook-based
`ActivationRecorder` does not compose with `torch.func` transforms.

- [ ] `make_slice_fn(model: LensModel, l_start: int, l_end: int) -> Callable[[Tensor], Tensor]`
  — a pure callable `h_l → h_l'` built from `model.layers[l_start+1 : l_end+1]`.
  - Thread per-block kwargs correctly: modern Qwen/Llama blocks need `position_embeddings`
    (rotary cos/sin), `attention_mask`, `position_ids`. Capture these once from a real hooked
    forward (record them in a setup pass) and close over them; they're input-shape-dependent,
    so key the cache by `seq_len`.
  - Handle tuple-returning blocks the same way `hooks.py` does
    (`output if torch.is_tensor(output) else output[0]`).
- [ ] `make_output_fn(model, l_start, contraction)` — same, but composed with `unembed` (or a
  chosen contraction, §2 below) for activation→output Hessians.
- [ ] **Validation gate:** for random prompts, `slice_fn(h_l)` must match the hooked forward's
  `h_l'` to tolerance (bf16 → expect ~1e-2 relative; run the check in fp32 to separate wrapper
  bugs from precision). This test is the tripwire for missed kwargs.

## 2. `hvp.py` — HVP core  [Phase 1a]

**First, pin the output contraction** (the roadmap never does; every downstream number depends
on it). For a vector-valued map `f: h ↦ h'`, "the Hessian" is a 3-tensor; all estimators take a
`contraction` argument:

- [ ] `Contraction` (frozen dataclass): one of
  - `cotangent(c)` — H of the scalar `⟨c, f(h)⟩` (use for 2A layer-to-layer spectra; `c` from
    e.g. the top J-lens direction at the target);
  - `logit(t)` — H of logit `t` after `unembed` (labeled-coordinate curvature, 2E);
  - `norm` — for the trust score's `‖½ δhᵀH δh‖`, computed vector-valued as
    `d/dε [J(h+εδh) δh]` at ε=0 (one jvp of the jvp — no contraction choice needed; document
    this as the exception).
- [ ] `hvp_rr(f, h, v, *, create_graph=False)` — reverse-over-reverse: two nested
  `torch.autograd.grad` calls with `create_graph=True` on the inner one. **Build this first** —
  it works directly on the existing hook infrastructure (`start_graph_at` makes `h_l` the leaf),
  no functional wrapper needed. Reference implementation.
- [ ] `hvp_fr(f, h, v)` — forward-over-reverse via `torch.func.jvp(torch.func.grad(f_scalar))`
  on the `functional.py` slice. Production path (memory-cheaper: no double graph).
- [ ] `hvp_fd(f, h, v, eps)` — central finite difference `(∇f(h+εv) − ∇f(h−εv)) / 2ε`.
  Validation only; ε exposed for the harness's ε-sweep.
- [ ] `restricted_hessian(f, h, directions: Tensor[k, d_model]) -> Tensor[k, k]` — the paper
  object: `H_ij = v_iᵀ H v_j`. Batching per roadmap: loop j over k HVPs (fix `v_j`, one
  `H·v_j`), then a single `directions @ Hv_j` matmul per column. **k HVPs, not k².** Symmetrize
  `(H + Hᵀ)/2` and *record* the asymmetry — it's a free estimator-error diagnostic.
- [ ] Precision policy: model runs bf16; cast `h`, `v`, and all HVP accumulation to fp32
  (mirror `jlens`'s fp32-in-memory convention). Second derivatives through bf16 forward
  activations are the noise floor the harness will measure; make the dtype explicit, not
  ambient.
- [ ] **Toy-model tests (the Phase 1a milestone):**
  - analytic check: a function with known Hessian (quadratic form, then a 2-layer tanh MLP with
    autograd-computed full Hessian via `torch.autograd.functional.hessian`) — all three
    estimators within tolerance;
  - three-way agreement on `tests/tiny.py`'s minimal `LensModel` (reuse it — it exists exactly
    for this);
  - **run this on the Thor first**: it doubles as the check that the NGC PyTorch build's
    `torch.func` + nested-`grad` support is sound on Blackwell.

## 3. `jspace.py` — sparse decomposition  [Phase 1a; blocks 2E, 2F, and restricted forms]

Confirmed absent from the repo. Built from paper §2.3/§2.5/§A.8.

- [ ] `lens_vectors(lens: JacobianLens, model: LensModel, layer: int, token_ids=None)
  -> Tensor[n, d_model]` — the J-lens vector for token `t` at layer `ℓ` is the direction whose
  inner product with `h_ℓ` drives token `t`'s lens logit: `v_t ≈ J_ℓᵀ u_t` with `u_t` the
  unembedding row (paper: logits are "determined, approximately, up to a data-dependent
  normalization factor, by ⟨v_t, h_ℓ⟩" — the final RMSNorm is the approximation; document it).
  Compute lazily / for requested token subsets — the full `[n_vocab, d_model]` matrix is
  ~150k × 2560 fp32 ≈ 1.5 GB per layer for Qwen3.5-4B; fine in 128 GB unified memory but don't
  materialize per-layer copies casually.
- [ ] `sparse_pursuit(h, dictionary, *, k=25) -> (indices, coeffs)` — greedy nonnegative
  sparse pursuit (matching-pursuit variant per the paper's citation): iteratively select the
  atom with the largest positive correlation with the residual, refit nonnegative coefficients
  over the selected set, repeat to k atoms or tolerance.
- [ ] `decompose(h, dictionary, *, k=25) -> Decomposition` — frozen dataclass:
  `indices, coeffs, jspace_component, remainder, frac_variance`. `remainder` is 2F's `v_j`.
- [ ] **Validation gate (reproduces the paper before you trust it):** on ~50 WikiText prompts
  through Qwen3.5-4B, J-space fraction-of-variance ≤ ~10% at mid layers, ~25 meaningfully
  active vectors per position, and top pursuit atoms qualitatively sensible on the walkthrough's
  multihop prompt (`lira`/`euro`-ish tokens at the boot position). If these don't roughly
  reproduce, your pursuit differs from theirs — stop and reconcile.
- [ ] Pursuit-seed / k-variation utilities (2E leg (ii) needs "survives seed and k variation"
  as a one-flag rerun, not a rewrite).

## 4. `interventions.py` — write operations  [needed by 2C validation and 2E leg (i)]

- [ ] `ActivationEditor` — the write-side sibling of `ActivationRecorder`: a context manager
  registering forward hooks that *return a modified output* (PyTorch replaces a block's output
  with a hook's non-None return). Same deferred-registration / guaranteed-removal semantics as
  `hooks.py`; handle the tuple-output case by rebuilding the tuple with the edited hidden
  state.
- [ ] `steer(h, v, alpha)`, `ablate(h, v)` — §2.5 basics.
- [ ] `coordinate_swap(h, v_s, v_t, *, alpha=1.0)` — exactly §2.5: `V = [v_s v_t]`,
  `c = V⁺h`, `h ← h + V(σ(c) − c)`; orthogonal component untouched. Support "every position
  over a fixed layer range" application (the paper's standard swap protocol).
- [ ] Test: swap then swap back is identity to tolerance; swap on the multihop prompt flips the
  currency answer (qualitative reproduction of the paper's headline intervention — also your
  first end-to-end validation that editor + lens vectors are wired correctly).

## 5. `harness.py` — reliability harness  [Phase 1b; gates all of Phase 2]

- [ ] `eps_sweep(hvp_fd, ..., eps_grid)` → per-(position, direction) stability curve +
  `find_plateau` (flat region detection; no plateau ⇒ flagged, not reported).
- [ ] `estimator_agreement(...)` — pairwise relative error among rr / fr / fd per position and
  direction; tolerance ⇒ boolean usability mask.
- [ ] `ms_hvp(f, h, v, *, n_steps)` — composed small-step correction along the perturbation
  path, with a reliability score = divergence between single-step and multi-step. (The AtP-
  correction citation is still ⚠ VERIFY per the roadmap; the *mechanism* is implementable and
  useful regardless of whether that paper exists.)
- [ ] `validity_score(...)` — estimated local Taylor radius (from the ε-sweep plateau edge and
  the second-order term's magnitude) vs. intended perturbation size. This *is* proto-2C: build
  it here, productionize the ratio form in `experiments/trust_score.py`.
- [ ] `randomize_model(hf_model, *, mode="reinit", seed)` — weight-randomized null:
  per-module `reset_parameters()` (or within-tensor shuffle as a second mode), then re-wrap
  with `from_hf`. **Two design notes:** (a) the pre-fitted lens is meaningless on the
  randomized model — a faithful Adebayo-style null re-runs the *whole* pipeline, so budget a
  small `jlens.fit` (~100 WikiText prompts, cheap at 4B) on the randomized model; (b) never
  randomize your only in-memory copy — reload from the HF cache.
- [ ] `null_distributions(...)` — record cross-term and directional-curvature nulls to disk;
  every Phase-2 script loads these rather than recomputing.
- [ ] `ablation_table(...)` — the second-order §A.7 battery as a config-sweep runner: Nσ
  filters, mean/median, stop-grad variants, position schemes (`skip_first` is already a
  parameter everywhere in `jlens` — sweep it), direction sets, ε, step count. Output: one
  tidy table = the standalone methods deliverable.
- [ ] `benchmark(...)` — wall-clock per HVP / per restricted-H, run **after**
  `nvpmodel -m 0` + `jetson_clocks`, asserting against the (restated) cost milestone:
  per-context restricted H at one position ≤ 2–3× one `lens.apply` call.
- [ ] Early averaging test (roadmap 1a checklist): `‖E[H_restricted]‖ / E‖H_restricted‖` over
  ~100 prompts — settles whether an *averaged* "H-lens" object exists at all. One script,
  run once, early.
- [ ] Pathological stress test: a constructed high-curvature case (e.g. positions straddling a
  sharp top-1 flip under interpolation) where single-step visibly degrades and MS-HVP + gating
  recovers. **This is the Phase-2 gate.**

## 6. `data.py` — parse the shipped experiment data  [cheap, do early]

The repo ships `data/experiments/*.json` — likely much of what the roadmap says to
"identify/reconstruct":

- [ ] Inspect schemas of `probe-swap.json` (→ 2C swap-failure set), `ignition.json` (→ 2A
  interpolation protocol), `capacity.json`, and `data/evaluations/lens-eval-{multihop,poetry,
  order-ops}.json` (→ 2E task suites). Write typed loaders for whichever contain per-case
  prompts/outcomes; flag whichever are aggregate-only (those revert to reconstruction tasks).
- [ ] Read both `README.md`s in `data/` — they presumably document exactly this.

## 7. `experiments/` — one script per Phase-2 letter  [in v2 order: C → B/E → D → F → A]

- [ ] `trust_score.py` (2C): per-position `‖½ δhᵀH δh‖ / ‖J δh‖` with `δh` from the actual
  intervention family (swap deltas from `interventions.py`, not random directions — the score
  should be evaluated on the perturbations you actually apply); correlation/AUC against the
  swap-failure set; null overlay.
- [ ] `gain.py` (2B): **reimplement the §4.3.1 measurement first** (median MLP-block output
  norm on direction populations, normalized by isotropic-random gain — not in the repo), since
  it's also the Phase-1b "reproduce ~10×" milestone; then ∂gain/∂context via per-context
  `jacobian_for_prompt` + HVP. Report as diagnostic only (local-vs-network caveat).
- [ ] `binding.py` (2E): `restricted_hessian` on the loaded task suites; three-leg criterion
  (causal-beyond-first-order via `coordinate_swap`, seed/k stability via `jspace` utilities,
  null comparison via `harness`); report validity-gated fraction.
- [ ] `automatization.py` (2D): curvature along J-space vs. automatic-circuit directions across
  RL checkpoints — **blocked on checkpoint access; keep last in the B/E/D group.**
- [ ] `remainder_coupling.py` (2F): `v_j = decompose(h).remainder`; pre-registered null
  comparison (write the pre-registration file into the repo before running).
- [ ] `ignition.py` (2A, last): interpolation from `data/experiments/ignition.json` (or
  reimplemented from §4.1.1); layer-to-layer spectra via power iteration / Lanczos over the
  `hvp` operator with a `cotangent` contraction, layers ~21–42 scaled to Qwen's depth; MS-HVP
  near knees; validity + null overlays; knee-vs-convergence-radius diagnosis in the writeup.

## 8. `vis_ext.py`  [Phase 5, alongside 2E]

- [ ] Pairwise-interaction heatmap per position: cell = `H_ij`, opacity = validity score,
  side-by-side randomized-null panel. Reuse `jlens.vis`'s page/iframe plumbing where possible.

## 9. Environment / infra checklist (Thor-specific)

- [ ] `compile=False` everywhere (already the default); add an assertion in `hvp.py` that
  refuses compiled blocks unless an `allow_compiled=True` flag is passed *and* the three-way
  agreement test has been run with compile on.
- [ ] The "fast path not available" transformers warning is fine — the plain-torch attention
  path is actually preferable for higher-order autodiff; do **not** install
  flash-linear-attention for this work.
- [ ] `nvpmodel -m 0` + `jetson_clocks` before any `benchmark` run (put it in the script's
  preamble as a check, not a comment).
- [ ] Single-device only (no NCCL on Jetson) — irrelevant for 4B, and 27B fits unified memory
  anyway; keep `device_map` out of the picture.
- [ ] Pin the environment: `uv add` torch-from-NGC constraint notes in the README so the
  Blackwell-compatible wheel doesn't get silently replaced by a PyPI wheel on `uv sync`.

## 10. Suggested build order (dependency-sorted)

1. `hvp.py::hvp_rr` + analytic toy tests (run on Thor — validates the PyTorch build too)
2. `functional.py` + slice-vs-hooked validation → `hvp_fr`, `hvp_fd`, three-way agreement
3. `jspace.py` + paper-reproduction gate (≤10% variance, k≈25)
4. `interventions.py` + multihop-swap reproduction
5. `hvp.py::restricted_hessian` (needs 3)
6. `data.py` schema inspection (parallel, anytime)
7. `harness.py` (needs 1–2; null pipeline needs a small `jlens.fit` run)
8. Milestone: gain reproduction (`gain.py` measurement half) + stress test → **Phase-2 gate**
9. `experiments/trust_score.py` → the rest in v2 order
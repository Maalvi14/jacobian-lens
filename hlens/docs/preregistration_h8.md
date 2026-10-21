# Pre-registration: H8 (motor-regime collapse)

Written 2026-10-14, **before any H8 measurement**. Notes §3 H8: "If the late dimensionality rise is `J_ℓ → I`,
curvature in that band should collapse toward the trivial case." `hlens/experiments/motor_collapse.py` records this
file's SHA-256 in its result file.

## Part A — is `J_ℓ` approaching the identity? (CPU, the pre-fitted lens)
`d_I(ℓ) = min_c ‖J_ℓ − c I‖_F / ‖J_ℓ‖_F` for every fitted layer ℓ = 0…30 (`c` = the best scalar). Reported as a profile;
criterion A: `d_I(30) < 0.5 × median_{ℓ ∈ 10..20} d_I(ℓ)`.

## Part B — does the layer-to-output map lose curvature faster than its length? (fp32 real model; the same code on the
weight-randomized model is the architectural control)
- Contexts: WikiText prompts 300–329 (`load_wikitext_prompts(330)[300:]`), 2 random valid positions each (seed 0) → 60.
- Layers ℓ ∈ {6, 10, 14, 18, 22, 26, 28, 29, 30}. At each context and layer: residual `h_ℓ` at the position `p`; a perturbation
  `δ` at `p` only: a random unit direction (seed 0) scaled to `0.1 ‖h_ℓ[p]‖`; `f` = the final-layer residual at `p` as a function of
  `h_ℓ` (blocks ℓ+1…31). `τ_map(ℓ) = ‖½ δᵀHδ‖ / ‖J δ‖` (`trust_terms`, fp32).
- Per-block nonlinearity `n(ℓ) = τ_map(ℓ) / (31 − ℓ)` (nonlinearity per remaining block, so a map that is merely short is not
  counted as "collapsed"). Workspace band `{14, 18, 22}`, motor band `{28, 29, 30}`.
- Criterion B: `median n(motor) < 0.5 × median n(workspace)`, and the 95% bootstrap CI (2,000 context resamples) of the ratio
  `median n(motor) / median n(workspace)` lies entirely below 1.
- Criterion C (null): the ratio in the randomized model is NOT below 0.5 (i.e. the same collapse does not appear without learned
  structure); formally the bootstrap CI of `ratio_real − ratio_null` is entirely below 0.

**H8 is supported iff A ∧ B ∧ C.**

## Secondary, exploratory
Spearman(`d_I(ℓ)`, `τ_map(ℓ)`) across the evaluated layers; `τ_map` itself (not per block); the same with δ along the top
J-lens direction instead of a random one.

## Known limits (stated in advance)
Random directions at 10% of the residual norm are not the perturbations the model sees; `τ_map` at ℓ = 30 is a single block;
the lens covers layers ≤ 30 only; one perturbation scale.

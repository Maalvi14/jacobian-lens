# Pre-registration: H6 (ignition is a curvature event)

Written 2026-10-14, **before any confirmatory H6 measurement**. A single-trial pilot was run beforehand
(carrier index 39 = "The history lecture focused on medieval {W}", pair France–Germany, 41 α values, a few
curvature evaluations) only to calibrate the grid, the layer gap and the cost; **that carrier is excluded from
the confirmatory set**. The pilot suggested two things this document builds in rather than hides: transition
widths shrink gradually with depth (0.73 at L0–2, 0.52 at L10–18, ~0.28 at L26–30) instead of collapsing at one
layer, and the second-to-first-order ratio along the path is *smallest* at the threshold (a sigmoid's inflection)
— so a curvature peak exactly at α* is not guaranteed even if the hypothesis is true. The tests below follow the
notes' wording (leading Hessian eigenvalue), not the pilot. `hlens/experiments/ignition.py` records this file's
SHA-256 in every result file; the real arm refuses to run before the randomized-model arm under the same hash.

## Hypothesis (research notes §3, H6)
Along the country-pair interpolation `(1−α) e_B + α e_A`, layer-to-layer curvature spikes are concentrated at the
layers where the transition width collapses, and near the α where the projection share crosses 0.5.
Falsifier: curvature is flat, or its peaks are uncorrelated with transition-width collapse (then the sharpening is a
property of the trajectory across many small steps and the local Hessian is the wrong object).

## Setup (fixed)
- Model Qwen3.5-4B fp32 eager (real arm); weight-randomized copy (`reinit`, seed 0), no lens needed (null arm).
- Trials: `data/experiments/ignition.json`; carriers = `ctx_templates[0..5]`; pairs = the first 4 of
  `itertools.combinations(single-token countries_12, 2)` after a seed-0 shuffle → **24 trials**. `{W}` as in
  `hlens/interpolation.py`. α grid for shares: 41 points in [0,1].
- Transition readout: projection share `⟨h(α) − h(0), h(1) − h(0)⟩ / ‖h(1) − h(0)‖²` at the `{W}` position, every
  layer 0–31; width = α-distance between share 0.1 and 0.9; threshold α*_t = share-0.5 crossing **at layer 16**.
- Curvature: layers `ℓ ∈ L = {4, 8, 10, 12, 14, 16, 20, 24}`, gap 4 (target `m = ℓ + 4`); `f(h_ℓ) = ⟨c_m, h_m⟩`,
  `c_m` = unit vector from the pure-B to the pure-A activation at layer m (position `{W}`); α offsets
  `o ∈ {−0.2, −0.1, 0, +0.1, +0.2}` around α*_t; `λ(t, ℓ, o)` = leading |eigenvalue| of the Hessian of `f` over the
  `{W}`-position residual (Lanczos, 8 iterations, fp32). `τ_path` (second-to-first-order ratio of an α-step of 0.05) is
  recorded as well.

## Tests (the decision)
- **T1 (layer localisation).** `Λ(ℓ)` = median over trials of `λ(t, ℓ, 0)`; `ℓ_λ = argmax_{ℓ∈L} Λ(ℓ)`. `W(ℓ)` = median
  width at layer ℓ. The width-collapse interval is `[ℓ_i, ℓ_{i+1}]` for consecutive members of `L` with the most
  negative `W(ℓ_{i+1}) − W(ℓ_i)`. Criterion: `ℓ_λ ∈ [ℓ_i − 2, ℓ_{i+1} + 2]` **and** this holds in ≥ 80% of 2,000
  trial-bootstrap resamples.
- **T2 (α localisation).** Per (trial, ℓ ∈ L): `D = λ(o = 0) − mean(λ(o = ±0.2))`. Criterion: one-sided Wilcoxon
  signed-rank, H1 `D > 0`, over all (trial, ℓ) pairs, **p < 0.01**.
- **T3 (above the weight-randomized null).** The T2 statistic `mean D` in the null arm; criterion: the 95%
  percentile-bootstrap CI (10,000 resamples of trials, seed 0) of `mean D_real − mean D_null` is **> 0**.

**H6 is supported iff T1 ∧ T2 ∧ T3.** Otherwise not supported; every number is reported.

## Secondary, exploratory (never decide)
1. T1 with the notes' depth band (paper L21–42 of 100 → Qwen layers 6–13) instead of the width window.
2. The same tests with `τ_path` in place of `λ`.
3. **Knee-vs-Taylor-radius diagnosis (mandatory in the write-up).** For each (trial, ℓ, o = 0) the validity score
   of the commitment map along the path; the fraction of the largest-λ cells that have validity score < 1 (a spike that
   coincides with the Taylor-radius boundary is an estimator artifact, not a model property).
4. One commitment layer or several: the layers where the median width drops by > 25% of its total range.
5. Engagement hypothesis E1 (handoff §7b), parts (a)/(b)/(d), are *not* part of this pre-registration; they get
   their own.

## Known limits (stated in advance)
24 trials; one gap (4) and one layer set; Lanczos with 8 iterations gives the leading |eigenvalue| to a few percent at
best (checked on a toy operator, not on the model); the commitment axis `c_m` is defined from the same trial's endpoint
activations; the null arm shares the same protocol but has no ignition to find, so T3 mainly guards against
architecture-level artifacts.

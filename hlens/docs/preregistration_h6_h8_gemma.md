# Pre-registration: H6 (ignition as a curvature event) and H8 (motor-regime collapse) on Gemma 3 12B

Written 2026-10-17, **before any Gemma measurement of H6 or H8**. Both hypotheses were tested on Qwen3.5-4B under `preregistration_h6.md` and
`preregistration_h8.md`. This file repeats those tests, unchanged in their statistics and thresholds, on Gemma 3 12B with depth-matched layers, to ask whether
the two Qwen results are properties of the architecture family or of that model. `hlens/experiments/{h6_replication,h8_replication}.py` record this file's SHA-256 in
every result file; for H6 the real arm refuses to run before the weight-randomized arm under the same hash.

## What the Qwen runs found (context, not evidence for these tests)
- **H6:** the three pre-registered tests passed, but the support rested on three layers (L4, L20, L24) rather than the onset band, and the Taylor radius collapsed at the
  knee (a possible estimator artifact).
- **H8:** `J_ℓ → I` was confirmed (distance 0.996 → 0.461) but per-block nonlinearity *rose* toward the output (ratio 3.73), so H8 was **not supported**.

## Model, precision, stimuli
`google/gemma-3-12b-pt` in fp32, eager attention; the weight-randomized control is `reinit` (seed 0), no lens fitted (neither test needs a lens in the randomized arm).
The pre-fitted `gemma-3-12b-pt` lens is used for H8 part A only. H6 stimuli: the 24 original trials (carriers `ctx_templates[0..5]` × the first 4 shuffled
single-token country pairs), with the clean one-space construction required by Gemma's tokenizer (Addendum 1 to `preregistration_replication.md`).

## Depth-matched layers (Qwen layer ↦ Gemma layer by depth fraction, 48 layers)
- **H6:** `L = {6, 12, 15, 18, 21, 24, 30, 36}` (Qwen {4, 8, 10, 12, 14, 16, 20, 24}); the threshold layer `α*_t` is the share-0.5 crossing at **layer 24** (Qwen 16);
  the commitment gap is **6** (Qwen 4); offsets `{−0.2, −0.1, 0, +0.1, +0.2}`; Lanczos with 8 iterations; transition widths at layers 0-44.
- **H8:** layers `{9, 15, 21, 27, 33, 39, 42, 44, 45}`; workspace band `{21, 27, 33}`, motor band `{42, 44, 45}`; the output layer is 47, so the per-block
  nonlinearity is `τ_map(ℓ) / (47 − ℓ)`; part A compares `d_I(46)` (the last lens layer) with the median of `d_I(ℓ)` for ℓ in 15..30; contexts: WikiText prompts 300-329,
  two random valid positions each (seed 0) → 60; a random unit direction at 0.1 of the residual norm.

## Tests (identical to the Qwen pre-registrations)
- **H6:** T1 (the peak of the median λ lies within the width-collapse window ± 2 layers, in ≥ 80 % of 2,000 trial bootstraps), T2 (λ at the threshold exceeds the mean of the
  ±0.2 flanks, one-sided Wilcoxon p < 0.01 over (trial, layer) pairs), T3 (the 95 % bootstrap CI of the real-minus-null mean flank difference is > 0).
  **Supported iff T1 ∧ T2 ∧ T3.**
- **H8:** A (`d_I(last lens) < 0.5 × median d_I(mid range)`), B (the median per-block nonlinearity in the motor band < 0.5 × that in the workspace band, with the bootstrap CI of the
  ratio entirely below 1), C (the randomized model's ratio ≥ 0.5 and the CI of the real-minus-null difference entirely below 0). **Supported iff A ∧ B ∧ C.**
- Decision table: a Gemma pass or fail is reported as such, next to the Qwen outcome. For H6, "passes on both models" says the transition event is not Qwen-specific; the Taylor-validity
  diagnosis below decides how much the curvature reading is trusted. For H8, "not supported on both" generalizes the Qwen finding (nonlinearity per block rises toward the output).

## Secondary (reported, never decide)
- **H6 mandatory knee-vs-Taylor-radius diagnosis:** the validity score along the path for 5 trials (indices 0, 5, 10, 15, 20) at layers `{6, 18, 24, 30, 36}`, offsets `{0, +0.2}`, gap 6;
  and the fraction of the largest-λ cells with score < 1. Also the same H6 tests with `τ_path` in place of λ, and the layer where the median width drops by > 25 % of its range.
- **H8:** the Spearman correlation of `d_I(ℓ)` with `τ_map(ℓ)`; `τ_map` itself.

## Procedure and resources
One large job at a time (Gemma fp32 is about 45 GiB); drop the page cache before launching (`sudo sysctl vm.drop_caches=3`). Every H6 trial is checkpointed to a `.jsonl` file and a rerun resumes. Order: H6 null,
H6 real, H8 (real arm then randomized arm). Expected cost: hours for H6 (24 trials × 8 layers × 5 offsets of Lanczos), about an hour for H8.

## Known limits (stated in advance)
All the Qwen limits (24 trials; one gap and one layer set; Lanczos accurate to a few percent at best; random perturbation directions for H8); a single weight-randomization seed; depth matching is
approximate (the architectures differ, Gemma has sliding and global layers); the Gemma `-pt` model only.

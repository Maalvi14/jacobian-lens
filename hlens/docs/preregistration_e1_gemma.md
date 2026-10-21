# Pre-registration: E1 (engagement) on Gemma 3 12B, held-out swap suite

Written 2026-10-19, **before any E1 measurement on Gemma**. On Qwen3.5-4B, E1 was not supported on the dev suite (`preregistration_e1.md`) nor on the held-out suite (`preregistration_heldout_scoring_h1_e1.md`), and the first-order prediction was the better predictor of swap success (P5).
This repeats the test on Gemma 3 12B so that "the first-order lens suffices for the swap" is a statement about two architectures or fails to be. `hlens/experiments/e1_gemma.py` records this file's SHA-256.

## What is fixed
- **Model, items, scoring:** Gemma 3 12B-pt in fp32, eager attention, its Neuronpedia lens; the 165 held-out swap items (`data/experiments/probe-swap-heldout.json`, SHA-256 `003ae0cd…`) with the case-insensitive scoring of `preregistration_heldout_scoring_h1_e1.md` §1; the items that are baseline-correct for this model are used.
- **Reference strength α_ref = 1.0** (not 1.25). Reason, stated openly: on the held-out suite the pre-registered α = 1.25 gave only 5/138 hits on Gemma (3.6 %), which leaves too few successes to test anything, while α = 1.0 gave 55/138 (39.9 %). The choice therefore uses the marginal hit rates of the earlier swap runs on these same items; it does not use any margin-path quantity.
  The label is "the real band swap at α = 1.0 flips the top-1 to a (folded) swap-answer id".
- **Swap path:** the band L18-30 (depth-matched), raw lens vectors, one editor per layer, margins `m(α) = logit(swap_id) − logit(ans_id)` at the last position for α ∈ {−0.05, 0, 0.05, …, 2.0}; `ans_id` = the clean top-1, `swap_id` = the folded swap-answer id with the higher clean logit. `FO(α) = m(0) + α g` with `g = (m(0.05) − m(−0.05)) / 0.1`; **`N = m(1.0) − FO(1.0)`**.
- **Single-layer variant** for the null comparison: the same path with only the layer-24 editor (the depth-matched mid layer, where the null lens exists) in the real model and in the seed-10 weight-randomized Gemma (`reinit`, fp32 on the CPU, weights rounded to bf16 precision, as in the B′ null arm), using the real model's `ans_id` and `swap_id`.
  The randomized model is rebuilt from its seed; a check that its lens belongs to the rebuilt weights is part of the procedure (below).

## Tests (as in the Qwen E1 pre-registration, with α_ref = 1.0)
- **P1:** AUC of `N` for success > 0.5 with one-sided permutation p < 0.01 **and** stratified AUC (geography vs other) > 0.5 with one-sided stratified-permutation p < 0.05.
- **P2:** among successes, the fraction with `FO(1.0) < 0` exceeds one half (one-sided binomial p < 0.01).
- **P4:** AUC(`N`, layer 24) in the real model minus that in the randomized model: 95 % percentile bootstrap interval (10,000 item resamples, seed 0) > 0.
- **E1 is supported iff P1 ∧ P2 ∧ P4.**
- **P5:** AUC of `FO(1.0)` for success > 0.5 (one-sided permutation p < 0.01) **and** greater than the AUC of `N`; the AUC of the realised margin `m(1.0)` is reported.
- **"The first-order lens suffices for the swap" is supported on Gemma iff E1 is not supported and P5 passes.**
- **Null-lens check (procedural):** the layer-24 lens of the seed-10 randomized model must transport layer-24 residuals closer (mean cosine) to the final residual than the identity map does. If it does not, the rebuilt randomized model does not match the lens, P4 is declared untestable, and the other tests stand.

| Outcome | Statement |
|---|---|
| E1 not supported and P5 passes | the first-order lens suffices for the swap on two architectures (Qwen3.5-4B, Gemma 3 12B) |
| E1 not supported, P5 fails | the swap is not explained by nonlinear gain either; first-order extrapolation does not predict success on Gemma; reported as a difference between models |
| E1 supported | nonlinear amplification drives swap success on Gemma: a second-order effect on one model, in contrast with Qwen; reported with the per-test results |
| fewer than 15 successes or 15 failures | not testable; the numbers are reported |

## Not part of this pre-registration
H1 (τ) on Gemma (it needs HVPs through three layers for every item and is a separate, longer run); other values of α; other bands.

## Known limits (stated in advance)
One null seed and one layer for P4; the null is rebuilt from its seed rather than kept in memory (guarded by the lens check); α_ref was chosen from earlier swap marginals on the same items; the population is dominated by city-to-country relations; items sharing a relation are not independent; the label is a top-1 flip while `N` is a pairwise margin quantity.

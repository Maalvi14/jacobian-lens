# Pre-registration: E1 (the engagement hypothesis)

Written 2026-10-14, **before any E1 measurement**. Origin: the H1 result (handoff §3.7/§7b). Not required by the
research notes; pre-registered anyway because the quantities were chosen after seeing H1 and need a fixed decision
rule. `hlens/experiments/engagement.py` records this file's SHA-256 in its result file.

## Hypothesis
A lens-coordinate swap succeeds when the network amplifies it *nonlinearly*: the swap's effect on the decisive logit
margin exceeds what its first-order effect predicts, and that excess is what carries the margin across zero. The
second-order measures (τ, the true first-order error) therefore track success, not failure (as seen in H1).

## Quantities (exact forward passes; no HVPs)
Items: the 51 baseline-correct probe-swap items of `results/trust_score_qwen3.5-4b.json` (fp32 model; their labels
`hit_a1.25` = the real multi-layer swap flips the top-1 to the swap answer). Margin `m(α) = logit(swap_id) − logit(ans_id)`
at the last position, where `ans_id` = the clean top-1 and `swap_id` = the first-token id of the swap answer with the
higher clean logit (as in `run_variants`). Path: the real swap protocol (band L12–L20, raw lens vectors, one editor per layer)
at scale α ∈ {−0.05, 0, 0.05, …, 2.0}. `m₀ = m(0)`; `g = (m(0.05) − m(−0.05)) / 0.1`; first-order prediction
`FO(α) = m₀ + α g`; **nonlinear gain `N = m(1.25) − FO(1.25)`**.
Single-layer variant for the null comparison: the same path with only the layer-16 editor (`N16`), computed in the real model
and in the weight-randomized model (reinit, seed 0; its lens exists at layer 16 only), same `ans_id`/`swap_id`, same labels.

## Tests (the decision)
- **P1 (nonlinear gain predicts success beyond category).** AUC of `N` for success (positive = success), one-sided
  permutation p < 0.01; **and** the stratified AUC (strata: geography vs other, as defined in H1) > 0.5 with one-sided
  stratified-permutation p < 0.05.
- **P2 (flips are nonlinearly driven).** Among successes, the fraction with `FO(1.25) < 0` (the first-order
  prediction alone would NOT have flipped the margin) must exceed one half: one-sided binomial test, p < 0.01.
- **P4 (specific to the trained network).** AUC(`N16`) in the real model minus AUC(`N16`) in the randomized model:
  95% percentile-bootstrap CI (10,000 item resamples, seed 0) > 0.

**E1 is supported iff P1 ∧ P2 ∧ P4.**

## Secondary, exploratory (never decide)
1. Knee alignment: `α_c` = the first α > 0 with `m(α) > 0` (linear interpolation); `α_n` = the first α ≥ 0.2 at which
   `|m − FO| > 0.25 |m − m₀|` (the first-order approximation has failed). Among successes: the fraction with
   `α_n ≤ α_c`, and Spearman(`α_n`, `α_c`).
2. Leave-one-out logistic AUC of [workspace loading] vs [loading, `N`] (does `N` add to the paper's predictor?).
3. `N` and `FO` by category; success defined as `m(1.25) > 0` instead of the top-1 flip.
4. Per-item margin curves `m(α)`, to look for a sharp rise (a knee) in successes vs a smooth drift in failures.

## Known limits (stated in advance)
n = 51 (about 18 successes); the α/band come from the sweep tuned on these same items; `N` is the nonlinear part of the
margin only along this one swap path; single-layer null at L16 only; "success" is the top-1 flip while `m` is a pairwise margin.

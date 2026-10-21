# Pre-registration: H1 (trust score) on Gemma 3 12B, held-out swap suite

Written 2026-10-20, **before any H1 measurement on Gemma**. On Qwen3.5-4B the trust score τ (the second-to-first-order ratio of the swap's effect, `‖½ δᵀHδ‖ / ‖Jδ‖` through the layer-to-output map) did not predict swap failure as the notes expected: on the dev suite it marked success;
on the held-out suite (`preregistration_heldout_scoring_h1_e1.md` §3) it marked nothing at α = 1.25 and marked success at α = 1.0, behaving as a proxy for the size of the first-order push. E1 on Gemma (`preregistration_e1_gemma.md`) found that first-order extrapolation explains the swap there too.
This repeats H1 on Gemma so the τ result is a statement about two architectures or fails to be. `hlens/experiments/h1_gemma.py` records this file's SHA-256.

## What is fixed
- **Model, items, scoring:** Gemma 3 12B-pt in fp32, eager attention, its Neuronpedia lens; the 165 held-out swap items (SHA-256 `003ae0cd…`) with the case-insensitive scoring; the items that are baseline-correct for this model (expected 138) form the population. No weight-randomized column is computed (the decision statements do not use it).
- **Swap and label:** the real multi-layer swap over the depth-matched band L18-30, raw lens vectors, one editor per layer. **Primary label: α = 1.0 flips the top-1 to a (folded) swap-answer id** (Gemma's working strength; the pre-registered α = 1.25 gives about 4 % hits and is reported as a secondary, expected to be untestable). Failure is the positive class.
- **τ layers (depth-matched from Qwen's 12, 16, 20):** layers **18, 24, 30**; the mid layer is **24** (the analysis calls it "L16" in key names for compatibility with the Qwen files). For each layer ℓ, τ_ℓ = ‖½ δᵀHδ‖ / ‖Jδ‖ with δ the α = 1 swap delta at every position of the layer-ℓ residual and f the final-layer residual at the last position (`hlens.experiments.trust_score._tau_at_layer`, forward-over-reverse HVP in fp32).
  Baselines as in the Qwen analysis: workspace loading at the mid layer (cosine of the residual with the source lens vector, mean over positions) and the first-order effect ‖Jδ‖.
- **Strata:** geography categories (the six `GEOGRAPHY` categories) vs other, as in the Qwen analysis.
- **Testable iff the population has at least 15 successes and at least 15 failures.**

## Statements (identical to the Qwen held-out pre-registration; evaluated at the primary label)
- **H1-notes (high τ marks failure):** AUC of `tau_L24` for failure > 0.5 with one-sided permutation p < 0.05 **and** stratified AUC > 0.5 with stratified p_greater < 0.05.
- **H1-dev (high τ marks success):** stratified AUC of `tau_L24` for failure < 0.5 with stratified p_less < 0.05.
- **H1-loading (workspace loading predicts failure):** stratified AUC of the negated loading at layer 24 for failure > 0.5 with stratified p_greater < 0.05.

| H1-notes | H1-dev | Statement |
|---|---|---|
| yes | no | the notes' prediction holds on Gemma; the Qwen reversal was not general |
| no | yes | the dev-suite reversal (τ marks success) replicates on a second architecture; H1 is not supported |
| no | no | τ does not discriminate on Gemma by the pre-registered rule |

## Reported but not decisive
The first-order effect size as a predictor of failure (`baseline_neg_jdelta_mean`, AUC and stratified AUC; on Qwen at α = 1.0 it was 0.70), the Spearman correlation of τ with ‖Jδ‖ (0.59 on Qwen), the layer-wise τ AUCs, the by-category success rates, and the α = 1.25 label if it is testable.

## Known limits (stated in advance)
The population is dominated by city-to-country relations and the "other" stratum is small; items sharing a relation are not independent; τ is evaluated along one particular intervention path and at three layers; no randomized-model control for τ on Gemma; the held-out suite's α = 1.0 hit rate on Gemma (about 40 %) leaves roughly 55 successes.

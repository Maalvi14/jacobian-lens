# Pre-registration: case-insensitive scoring on the held-out swap suite, and H1 / E1 on its items

Written 2026-10-18, **before any run with case-insensitive scoring** and before any H1 or E1 measurement on the held-out items. The held-out suite (`data/experiments/probe-swap-heldout.json`, SHA-256 `003ae0cd…`) was scored once with
case-sensitive first-token matching (`preregistration_swap_heldout.md`); that run showed almost no currency or state-of-matter item passing the baseline-correct filter, because the models answer "Euro" and "Solid" while the
answers are written "euro" and "solid". `hlens/experiments/{swap_heldout,heldout_h1_e1}.py` record this file's SHA-256.

## 1. The scoring fix (fixed here)
`hlens.swaps.item_token_ids(tok, item, fold_case=True)`: the first-token ids of the answer and of the swap answer include the lower-case and capitalised forms of the word, each with and without a leading space.
**Exception: `element-symbol` items stay case-sensitive** (the answer "He" is a symbol; "he" is a pronoun). Ambiguity (answer and swap answer sharing a first token) is recomputed under the folded ids, and ambiguous items are dropped.
Nothing else in the swap protocol changes (bf16 for the swap suite, fp32 for H1/E1; band L12-20 Qwen, L18-30 Gemma; α = 1.25 primary and 1.0 secondary; three random-direction controls).

## 2. Swap suite re-scored (both models; the swap-suite decision rule unchanged)
Per model and per α: **pass iff baseline-correct items ≥ 30, hit rate ≥ 25 %, and hits ≥ 5 × max(1, max over the three controls)**. Reported next to the case-sensitive numbers: the baseline-correct count and hit rate by category.
Specifically recorded: how many `city-currency` and `element-state` items now pass the baseline-correct filter.

## 3. H1 (trust score) on the held-out items — Qwen3.5-4B fp32
Population: the held-out items that are baseline-correct under folded scoring; label: the real band swap at α = 1.25 flips the top-1 to a folded swap-answer id (`hit_a1.25`; α = 1.0 is secondary). Failure is the positive class.
Scores and strata as in the original H1 (`trust_score.analyze`; τ at L16; baseline workspace loading at L16; strata: geography categories vs other). **Testable iff the population has ≥ 15 successes and ≥ 15 failures.**
- **H1-notes (the notes' prediction: high τ marks failure):** AUC of `tau_L16` for failure > 0.5 with one-sided permutation p < 0.05 **and** stratified AUC > 0.5 with stratified p_greater < 0.05.
- **H1-dev (the dev-suite observation: high τ marks success):** stratified AUC of `tau_L16` for failure < 0.5 with stratified p_less < 0.05.
- **H1-loading (workspace loading predicts failure, as on the dev suite):** stratified AUC of `baseline_neg_loading_L16` for failure > 0.5 with stratified p_greater < 0.05.

| H1-notes | H1-dev | Statement |
|---|---|---|
| yes | no | the notes' prediction holds on held-out items; the dev-suite reversal was not general |
| no | yes | the dev-suite reversal (τ marks success) replicates; H1 is not supported |
| no | no | τ does not discriminate on held-out items; the dev-suite association was not stable |
| yes | yes | impossible by construction (opposite signs) |

## 4. E1 (engagement) on the held-out items — Qwen3.5-4B fp32
Exactly `preregistration_e1.md` (margin paths at α in −0.05…2.0, `N = m(1.25) − FO(1.25)`, P1, P2, P4 with the L16 weight-randomized control and its seed-0 lens), on the baseline-correct held-out items with the folded labels.
**E1 is supported iff P1 ∧ P2 ∧ P4.** Added here (it was exploratory on the dev suite):
- **P5 (the first-order prediction is the better predictor):** AUC of `FO(1.25)` for success > 0.5 with one-sided permutation p < 0.01 **and** greater than the AUC of `N`. Also reported: the AUC of the realised margin `m(1.25)`.
The statement "the first-order lens suffices for the swap" is supported iff E1 is not supported and P5 passes.

## Not part of this pre-registration
Gemma H1/E1 (no null lens at a matched layer and the HVP cost); α other than 1.25 and 1.0; any change to the band.

## Known limits (stated in advance)
The population is dominated by city-to-country relations; case folding can credit a different word with the same folded first token (checked only for ambiguity between the answer and the swap answer); the band and α are those tuned on the dev suite (fixed here);
items from the same relation are not independent; the stratified tests use two strata; E1's P4 uses a single randomization seed and a null lens at layer 16 only; one model for H1/E1.

# Pre-registration (EXPLORATORY): why claim A fails off Qwen3.5-4B, and what explains the H10 differences between families

Written 2026-10-19, **before any run of either study**. Both are **exploratory and descriptive**: they generate hypotheses, they do not confirm claims, and nothing here changes the verdicts of earlier pre-registrations.
`hlens/experiments/{h9_scan,h10_covariates}.py` record this file's SHA-256.

## Study 1 — claim A scan (`h9_scan.py`)
Claim A (inhibition at the ignition threshold: both cross-terms negative, peaking at the threshold) holds on Qwen3.5-4B at one layer pair (ℓ = 12, m = 20) chosen in advance, and fails on Gemma 3 12B (depth-matched 18 → 30) and Qwen3.6-27B (24 → 41).
Possible reasons that are cheap to separate: (a) the layer pair that the depth mapping gives is not where the effect lives in the other models; (b) the J-lens directions (which depend on the fitted lens) are poor there, whereas a lens-free direction would work; (c) neither, i.e. the effect is specific to Qwen3.5-4B.
- **Models:** Qwen3.5-4B (positive control) and Gemma 3 12B, fp32. Not the 27B (cost and the mixed-precision caveat).
- **Stimuli:** the 24 original H9 trials (`trial_sets(...)["O"]`; Gemma with the clean one-space construction); the threshold α* is the share-0.5 crossing at the model's depth-matched mid layer (Qwen 16, Gemma 24), unchanged.
- **Grid (18 configs per model):** directions ∈ {J-lens vectors (as in claim A), identity (the unembedding rows themselves, i.e. logit-lens directions, no J)} × source ℓ ∈ {Qwen 8, 12, 16; Gemma 12, 18, 24} × gap ∈ {Qwen 4, 8, 12; Gemma 6, 12, 18} (target m = ℓ + gap).
- **Per config:** the P0a, P0b, P0c tests of `preregistration_h9.md` (one-sided Wilcoxon), the fraction of trials with both cross-terms negative at the threshold, the median inhibition and the median first-order cross-effect.
  A config "passes with Bonferroni" iff all three tests have p < 0.01 / 18 ≈ 5.6e-4.
- **Sanity check (must hold or the scan is not trusted):** on Qwen3.5-4B the config (lens, ℓ = 12, m = 20) reproduces claim A: all three tests pass and at least 20 of 24 trials have both cross-terms negative (the earlier run had 24/24).
- **Descriptive questions, no verdict:** (1) On Gemma, does any config pass with Bonferroni, and with which directions and layers? (2) On Qwen, how narrow is the (ℓ, m) region where claim A holds, and do identity directions work there? (3) Is the fraction of both-negative cross-terms above 70 % anywhere on Gemma?
- **Reading rules (written in advance):**

| Observation | Reading |
|---|---|
| a Gemma config passes with Bonferroni (any directions) | the failure of the depth-matched pair is at least partly a choice of layers or directions; claim A is not shown to be Qwen-only (a new pre-registered confirmation on held-out trials would be needed before saying so) |
| only identity configs pass on Gemma | the J-lens directions are the problem on Gemma, not the effect |
| no Gemma config passes, the Qwen sanity check holds | no support for a layer-pair or direction explanation within this grid; claim A stays Qwen3.5-4B-specific |
| the Qwen sanity check fails | the scan code or protocol is not reproducing claim A; the scan is discarded |

## Study 2 — what explains the H10 between-family differences (`h10_covariates.py`)
At 100 pairs per family (`preregistration_h10_scaled.md`) all four families show the signature with different strengths (median pair D̄: countries 0.49, control 0.36, persona 0.30, near-synonym 0.058), and endpoint distance at layer 24 does not order them
(countries have the smallest distance and the largest effect). This study asks which pair-level properties do.
- **Data:** the 366 pairs and their D̄ values of `results/h10_scaled_qwen3.5-4b.json` (D̄ = mean over L20 and L24 of the median over the pair's two trials of `D = λ(center) − mean λ(±0.2)`), carriers `noun_ctx_templates[12..13]`, Qwen3.5-4B fp32, forward passes only.
- **Covariates (fixed list, computed per pair and averaged over the two carriers):**
  `d8, d16, d24, d28` (endpoint distance at those layers: ‖h_A − h_B‖ / mean norm at the `{W}` position); `js_output` (Jensen–Shannon divergence of the next-token distributions at the last position for pure A vs pure B);
  `emb_cos` (cosine of the two words' input embeddings); `unemb_cos` (cosine of their unembedding directions); `slot_surprisal` (mean of −log p(word | text before the slot) — how well the nouns fit the carrier);
  `slot_logp_gap` (|log p(A) − log p(B)| at the slot); `word_len` (mean characters).
- **Analyses (descriptive):** Spearman correlation of each covariate with D̄, over all pairs and within each family (flagged "significant" at p < 0.01 / 10 = 1e-3); a least-squares fit of D̄ on family indicators (control as reference) alone, on the z-scored covariates alone, and on both, with 95 % pair-bootstrap intervals (2,000 resamples, seed 0);
  the R² of each; and, for each family, how much its coefficient against the control shrinks when the covariates are added.
- **Reading (descriptive, no verdict):** a covariate is "associated" if its overall Spearman is significant and its sign agrees in at least three of the four families. The family gap is "largely explained" if the coefficients of countries and near-synonym against the control shrink by more than 50 % with the covariates in the model.
  The `slot_surprisal` and `js_output` covariates are the pre-stated candidates (how well the noun fits the carrier; how much the word changes the model's output); the rest are controls on them.

## Procedure and resources
One large job at a time; drop the page cache before the Gemma scan (`sudo sysctl vm.drop_caches=3`). The Qwen scan and Study 2 are small (about an hour each); the Gemma scan is the long one (18 configs × 24 trials, estimated a few hours). Every config of the scan is checkpointed (a `.jsonl` per config) and the summary file is updated after each config.

## Known limits (stated in advance)
24 trials per config, so each Wilcoxon test is coarse; 18 configs per model and Bonferroni correction for them, but not across the three descriptive questions; the grid is small and coarse (three source layers, three gaps, one threshold layer, one set of country stimuli); identity directions are one lens-free alternative among several; the Gemma stimuli use a different
spacing construction from the Qwen ones; the 27B is not scanned; Study 2 covariates are my choices, computed at two carriers, correlational, with 366 non-independent pairs (words are reused); "associated" does not mean causal.

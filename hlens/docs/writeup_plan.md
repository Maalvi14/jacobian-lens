# Write-up plan: what to claim, what to re-test first

Status: DRAFT. Written after H1–H10 and E1 were run (see `handoff.md` §3, §7b).
Nothing here has been run yet.

## 1. Claims, ranked by how much a reviewer would test them

| # | Claim | Evidence now | Weakness |
|---|---|---|---|
| A | Second-order lateral inhibition at the ignition threshold (H9 precondition): both cross-terms negative, while first-order cross-effects are positive. | 24/24 trials, p = 6e-8, one model | One model; 24 trials; word-pair stimuli only; ignition threshold depends on the Taylor-validity gate (H6 radius collapses at the knee) |
| B | Curvature along J-lens directions exceeds random directions from L16 (2.1x at L28-30); the weight-randomized control is exactly flat (H2). | Null lens at layer 16 only | One model; one seed for the randomized null; the null lens exists at a single layer |
| C | Cross-architecture replication on a hybrid Gated-DeltaNet/attention model: sparse decomposition and landmarks transfer, swap partially (35% vs 54-70%), MLP gain does not. | Gates run | Swap suite is 87 usable items, tuned on the same items |
| D | Method: tau / Taylor-radius / per-block nonlinearity as a validity check for the linear lens; tokenizer pitfall for swap scoring. | Tooling tested | Not a finding; must not be framed as predicting outcomes |
| N | Negatives: H1, E1, H3, H5, H7, H8, H9 part 2, H10. | Pre-registered | Report as-is |

Rule for the write-up: A and B are only claimed as far as the re-tests below
support. If a re-test fails, the claim is downgraded to "observed on Qwen3.5-4B
only" or moved to the negatives section.

## 2. Re-tests, in order

Run serially (three fp32 jobs at once reached ~120/122 GB). Each gets a
pre-registration file with its SHA-256 recorded in results, and the real arm
refuses to run before the randomized-model null arm under the same hash, as for
H3-H10. Any analysis choice made after seeing data goes in a dated addendum.

### R1. H2 seeds and layers (cheapest; do first)
- Re-initialize the randomized model with seeds 0-4 (currently seed 0 only).
- Fit null lenses at more layers than L16 (e.g. L8, L16, L24). Fitting is the
  cost: ~6 minutes per layer was the observed order of magnitude for L16.
- Prediction that can fail: lens-direction curvature / random-direction
  curvature > 1 from L16 on, with the null ratio within noise of 1 at every
  layer and seed.
- Failure means: the L16-only flat null was luck, and B is downgraded.

### R2. H9 precondition, larger and more varied stimuli
- Expand from 24 trials to a held-out set (target >= 100) with new concept
  pairs not used to set the 0.134 threshold. Pre-register the threshold and
  the pair list before running.
- Add non-noun pairs (the H10 limits note noun-only stimuli).
- Report the Taylor-validity gate result per trial; count a trial toward the
  claim only if the gate passes at the threshold.
- Prediction that can fail: both cross-terms negative in a clear majority of
  held-out trials, and the randomized-model arm shows no such sign pattern.
- Keep part 2 (load dependence) out of the claim. If it is retried, use the
  multi-layer minimum-rank occupancy measure, pre-registered, as its own test.

### R3. Second models (checked 2026-10-15)
Both requested models have pre-fitted lenses in `neuronpedia/jacobian-lens`
(downloaded; finite, correct shapes):

| | Qwen3.6-27B | Gemma 3 12B |
|---|---|---|
| Lens file | `qwen3.6-27b/jlens/.../Qwen3.6-27B_jacobian_lens_n1000.pt` (3.3 GB; only the n=1000 fit exists) | `gemma-3-12b/.../gemma-3-12b-pt_jacobian_lens.pt` (1.4 GB, 775 prompts) and the `-it` variant (844 prompts) |
| Source layers | 0-62 of 64, d=5120 | 0-46 of 48, d=3840 |
| Fitted on | `Qwen/Qwen3.6-27B` | `google/gemma-3-12b-pt` (the `-it` lens is for `-it`) |
| Architecture | Same hybrid family as the 4B: 48 linear + 16 full attention, untied lm_head | Standard transformer, sliding + global attention |
| Weights | 55.6 GB bf16, ungated | 24.4 GB bf16, **gated (manual approval)** |
| fp32 weights | ~108 GB: **does not fit** in 122 GB with HVP activations | ~49 GB: fits |
| Role | Same-family scaling check | Different-architecture replication |

Consequences:
- Gemma 3 12B is the cleaner replication (full fp32, different architecture class)
  and should go first. It needs HF access approval for `google/gemma-3-12b-pt`.
- Qwen3.6-27B cannot be run in fp32 as the 4B was. It needs a pre-registered
  mixed-precision design (bf16 weights, only the layers inside each HVP slice
  cast to fp32) and a check that this reproduces the 4B fp32 results on a
  Qwen3.5-4B subset before it is trusted. Until that check passes, 27B results
  are not comparable with the 4B ones.
- Every result must state which lens was used (n=1000 for the 27B).
- Readiness already verified on tiny CPU stand-ins (`tests/test_multimodel.py`):
  slice function vs hooked forward and forward-over-reverse vs reverse-over-reverse
  HVP, for Gemma 3 (sliding/global layers) and an untied-head Qwen. Real-weight
  checks are still required (see the handoff).

Run on each new model: R1-style flat-null check at one layer, R2's held-out
protocol, and the three paper gates (landmarks, swap, sparse pursuit).
Decision rule written in advance: A and B are stated as general only if they
replicate on the second model; otherwise stated as Qwen3.5-4B findings.

### R4. Larger swap suite (supports C and the negatives)
- Build a held-out set of swap items disjoint from the 87 used to choose the
  band and alpha. Fix band L12-20 and alpha 1.25 beforehand.
- Report the hit rate with a bootstrap interval against the random-direction
  control.
- Re-run the E1 first-order vs actual-margin comparison on it, since H1 and E1
  are both limited by suite size.

## 3. Housekeeping before any numbers go in a paper
- Commit the current work (57 changed or new files; the user commits).
- Re-run the combined `-m real_model` session after the latest edits.
- Re-run `benchmark` in MAXN mode (needs `sudo nvpmodel -m 0 && sudo jetson_clocks`
  by the user); timing numbers taken in the 120W mode are not reportable.
- README note about the pinned torch wheel.
- Lanczos convergence checks and eigenvalue normalization by residual norm
  for the ignition results.

## 4. Outline of the write-up
1. Setup and replication of the paper's gates on a hybrid architecture (C).
2. Curvature is privileged along lens directions (B), with the flat null.
3. A second-order effect the first-order lens misses (A).
4. Validity tooling and its limits (D): tau, Taylor radius, nonlinearity.
5. Negative results (N), with pre-registrations linked.
6. Limitations: single model unless R3 replicates, small swap suite, post-hoc
   addenda (H5, H9 part 2, H3 synonyms) listed explicitly.

## 5. Open decisions for the user
- Gemma access request on HF (needed before anything else on Gemma).
- Compute budget: R1 and R2 are hours; R3 may be days if a lens must be fitted.
- Venue and framing: findings paper with a negatives section, or a tooling
  and negative-results report.

## 6. Results so far on the second model (2026-10-15)
Pre-registration: `preregistration_replication.md` (+ Addendum 1). Raw results: `results/h9_replication/`, `results/h2_replication/`, `results/gates_gemma-3-12b.json`.

| Item | Qwen3.5-4B | Gemma 3 12B |
|---|---|---|
| G1 landmarks | (earlier) | passed; workspace range L20-43 |
| G2 swap at pre-registered a=1.25 | 35% (earlier suite) | **not passed**: 8/48 = 17% (controls 1/0/0); a=1.0, post hoc: 14/48 |
| Claim A (H9 precondition) | holds on held-out set (120 trials), both spacings | **fails**: no peak at the threshold (P0b p~0.8), 30% both-negative |
| Claim B (H2) T1 (CI > 1 in >= 75% of layers, depth >= 0.5) | passes | passes |
| Claim B T2 (median ratio >= 1.5) | **fails**: median over L16-30 is 1.37 (rises to 2.1 at L28-30) | passes: median 9.0 (4.1 at L24 to 11.3 at L44) |
| Claim B T3 (null CI contains 1) | **fails as written** in 6 of 7 arms | **fails as written** (1 arm) |

Reading of T3: the null ratios are 0.997-1.002, i.e. within 0.5% of 1, against real effects of 1.1x-11x. The rule failed because the
bootstrap intervals are extremely narrow (about +-0.0005), so an interval around 1.001 excludes 1. The rule as written was a poor choice
(it tests exact equality, not "no practical effect"). It is **not changed after the fact**: by the pre-registered rule, B is not
supported on either model. An equivalence criterion (e.g. null ratio within 5% of 1) is an exploratory reading only and would need a
new pre-registration before it can count.

## 7. B′ results (2026-10-15; `preregistration_b_prime.md`, SHA-256 2b459bc2…, fresh prompts 200-229, new seeds)
| | Qwen3.5-4B | Gemma 3 12B |
|---|---|---|
| N: null arms inside [0.95, 1.05] | 5/5 (0.9998-1.0014; seeds 10-12 at L16, 10 at L8 and L24) | 2/2 (seed 10, L24: 1.001; seed 11, L24: 1.0032, interval [1.0026, 1.0039]) |
| R: real lower bound > 1.05 in >= 75% of layers (depth >= 0.5) | 8/8 layers | 6/6 layers |
| C: real lower bound > every null upper bound at null layers | L16 1.142 vs 1.002; L24 1.313 vs 1.001 | L24 4.027 vs 1.002 |
| Median ratio (not a test) | 1.33 (below 1.5, so not "large") | 9.37 (large) |
| Ratio vs depth (Spearman) | 0.93 | 1.00 |
| **B′ holds** | **yes** | **yes** |

B (H2) itself remains reported as failed under `preregistration_replication.md` (T2 on Qwen, T3 as written on both). B′ holds on both models,
so by the decision table it is stated as holding on Qwen3.5-4B and Gemma 3 12B; two models only. The optional Gemma seed-11 null also lies inside
[0.95, 1.05], so N holds with both Gemma nulls.

## 8. Claim A on Qwen3.6-27B (2026-10-17; `preregistration_27b_claim_a.md`, mixed precision validated on the 4B)
Held-out set, 120 trials: P0a p = 0.098, P0b p = 0.99, P0c p = 0.13: **A fails.** Both cross-terms negative in 17% of trials; inhibition at the threshold (0.0011) is below the flank values;
0/120 trials pass the Taylor-validity gate. A therefore holds on Qwen3.5-4B only (fails on Gemma 3 12B and on the 27B of the same family). Claim B′ holds on Qwen3.5-4B and Gemma 3 12B and was not tested on the 27B.
Caveats: mixed precision validated on the 4B only (8-block slice vs 17 here); n=1000 lens; the original 24-trial set was not run on the 27B.

## 9. H6 and H8 on Gemma 3 12B (2026-10-17; `preregistration_h6_h8_gemma.md`)
- **H6 not supported:** T2 fails (p = 0.90), T3 fails (CI [−0.024, 0.175]); λ is largest at the earliest layer tested and falls with depth. Qwen3.5-4B passed the pre-registered tests with weak support, so H6 is Qwen-specific at best.
- **H8 not supported, and the Qwen finding generalizes:** per-block nonlinearity rises toward the output (ratio 5.05 vs 3.73 on Qwen); the randomized model also rises (2.33). `J → I` is weaker on Gemma (0.83).

## 10. Held-out swap suite (built and run 2026-10-17)
`data/experiments/probe-swap-heldout.json` (165 generated items, SHA-256 `003ae0cd…`) with `preregistration_swap_heldout.md` fixes the G2 protocol (Qwen L12-20, Gemma L18-30, α = 1.25 primary and 1.0 secondary) and its decision table before any run.
It addresses claim C's weakness (settings tuned on the scored items; Gemma 17 % at the pre-registered α) and is the suite on which H1 and E1 could later be repeated (separate pre-registration).

**Results.** Qwen3.5-4B (L12-20): α = 1.25 60/114 = 52.6 % (controls 0/0/0) → passes; α = 1.0 38/114 = 33.3 % → passes. Gemma 3 12B (L18-30): α = 1.25 4/125 = 3.2 % → **fails** (the pre-registered setting);
α = 1.0 55/125 = 44.0 % (controls 0/0/0) → passes. Claim C update: the swap reproduces causally on both models on held-out items, Qwen at the original setting (the 35 % dev rate was not inflated by tuning), Gemma only at
α = 1.0 (confirmed out of sample; the pre-registered α = 1.25 was wrong for it). Coverage caveat: currency and state items almost never pass the baseline-correct filter (case-sensitive first-token scoring).

## 11. H10 extended (2026-10-17; SUPERSEDED by §14 - the dissociation did not survive scaling; `preregistration_h10_extended.md`, Qwen3.5-4B)
10 fresh pairs and fresh carriers per family (60 trials each): persona reproduces the H6 signature (L20 and L24 p ≈ 0), register does not (L20 p = 0.039, L24 p = 0.28); contrast p = 5e-10 / 2e-15 → the dissociation replicates; the proxy statement
(both families) is still not supported. Likely reading: distinct concepts vs near-synonyms (the country pairs behave like persona pairs), not persona per se; 10 distinct pairs per family. Not run on Gemma (no H6 signature there).

## 12. H10 control (2026-10-18; partly superseded by §14; `preregistration_h10_control.md`, Qwen3.5-4B)
Ten object-noun pairs matched to the persona pairs on layer-24 endpoint distance (median |Δd| 0.002): the control reproduces the signature (L20, L24 p ≈ 0); persona is not larger (pair-level p = 0.85; persona coefficient −0.29, 95 % CI [−0.77, 0.04]). The
persona/register dissociation is therefore a distinct-concepts (endpoint distance) effect, not a persona effect. Register pairs are closer (median d 0.73 vs 1.07).

## 13. Held-out swap suite re-scored with case-insensitive matching (2026-10-18)
Baseline-correct 125 (Qwen) / 138 (Gemma); Qwen α = 1.25 61/125 = 48.8 % (pass), Gemma α = 1.25 5/138 = 3.6 % (fail), Gemma α = 1.0 55/138 = 39.9 % (pass), controls 0/0/0. Verdicts unchanged; recovered currency items hardly swap (1/13 on both models).

## 14. H10 at scale (2026-10-18; `preregistration_h10_scaled.md`, Qwen3.5-4B, 100 pairs per family)
All four families (persona, near-synonym, matched object-noun control, countries) reproduce the signature; the "register does not" result of the 10-pair run was a small-sample result. Matched persona vs control: median −0.043, p = 0.08 (no persona effect); regression persona −0.093 [−0.18, −0.006];
overall Spearman(d, D̄) 0.185 (< 0.3); near-synonyms show a much smaller effect (median D̄ 0.058 vs 0.30-0.49). Conclusion: no persona-specific effect, and endpoint distance explains the variation within families but not the differences between them.

## 15. H1 on the held-out items (2026-10-18; `preregistration_heldout_scoring_h1_e1.md` §3, Qwen3.5-4B, 126 baseline-correct items)
α = 1.25 (primary; 61 successes / 65 failures): τ_L16 naive AUC(fail) 0.507 (p = 0.46), stratified 0.614 (p = 0.017); loading stratified 0.489 → H1-notes, H1-dev and H1-loading all not met. α = 1.0 (secondary; 39/87): τ_L16 AUC(fail) 0.289, stratified 0.370 (p_less 0.012) → the dev direction (high τ marks success) appears;
first-order effect size predicts failure best (AUC 0.70). τ behaves as an effect-size proxy; E1 on the held-out items is next.

## 16. E1 on the held-out items (2026-10-18; `preregistration_heldout_scoring_h1_e1.md` §4, Qwen3.5-4B, 126 items, 61 successes)
P1 fails (AUC of the nonlinear gain 0.505, p = 0.46), P2 fails (5 of 61 successes need nonlinearity), P4 fails (null-model difference CI [−0.013, 0.278]) → E1 not supported. P5 passes: AUC of the first-order prediction 0.645 (p = 0.0016) > AUC of N 0.505; margin AUC 0.806
→ "the first-order lens suffices for the swap" is supported. The dev-suite pattern (FO 0.75, margin 0.85, N 0.39) replicates, weaker.

## 17. Claim A on Qwen3.6-27B, original 24-trial set (2026-10-19)
P0a p = 0.95 (median I at the threshold −0.0010), P0b p = 0.997, P0c p = 0.93; both cross-terms negative in 8 % of trials; no trial passes the validity gate. Agrees with the held-out set (120 trials, fails): A fails on the 27B on both sets. A is specific to Qwen3.5-4B.

## 18. Exploratory H10 covariates (2026-10-19; `preregistration_exploratory_a_h10.md` Study 2)
Pair-level D̄ (366 pairs) is associated with endpoint distance at L24 (+0.19), input and unembedding cosine (−0.19), slot surprisal (−0.25: better-fitting nouns have a larger effect) and word length (−0.23); not with output divergence. R² 0.24 with family indicators and all covariates. The family gaps are not explained
(persona 22 % and near-synonym 31 % shrinkage; the countries advantage grows from +0.145 to +0.315).

## 19. Exploratory claim A scan (2026-10-19; `preregistration_exploratory_a_h10.md` Study 1)
Qwen3.5-4B: 14 of 18 configurations pass with the Bonferroni threshold (sanity check holds); the effect needs the target layer to be well below the output (m < 24) and does not need the J-lens (identity directions work in 8 of 9). Gemma 3 12B: no configuration of 18 passes even uncorrected (the peak-at-threshold test never below p = 0.048);
inhibition is weakly positive and beats the control in 14 of 18, but is not threshold-specific; both-negative at most 83 %. No layer-pair or direction explanation; claim A stays Qwen3.5-4B-specific.

## 20. Lanczos convergence check (2026-10-19; `preregistration_lanczos_check.md`)
8-iteration eigenvalue estimates vs 40-iteration references over 320 cells: median relative error 0.01 %, 90th percentile 0.7 %, maximum 17.8 %; D sign agreement 100 %; the stored H6 values are reproduced in all cells. The H6 results are not a Lanczos artifact. Scaled by the residual norm the layer profile flattens (L4 lead over L20: 5.4× → 1.6×).

## 21. E1 on Gemma 3 12B, held-out swap suite (2026-10-19; `preregistration_e1_gemma.md`)
138 baseline-correct items, 57 successes at α = 1.0. P1 fails (AUC of the nonlinear gain 0.247, p = 1.0), P2 fails (2 of 57 successes need nonlinearity), P4 passes on its own (0.736 vs 0.425; CI [0.183, 0.437]) → E1 not supported. P5 passes: first-order AUC 0.807 (p = 1e-4), margin AUC 0.956 → "the first-order lens suffices for the swap" holds on two architectures.

## 22. B′ on Llama 3.1 8B (2026-10-20; `preregistration_b_prime_llama.md`)
N, R and C hold: five null arms 0.9994-1.0016 (inside [0.95, 1.05]); real lower bound above 1.05 at 8/8 layers (ratios 1.40 at L16 to 2.83 at L30, median 1.40, Spearman 0.67); C: 1.385 vs 1.002 at L16, 1.305 vs 1.0 at L24. B′ holds on three architectures (hybrid, sliding + global, dense). First-order gain ratio ≈ 1 up to L28 (curvature raised, gain not, as on Qwen).

## 23. H1 on Gemma 3 12B, held-out swap suite (2026-10-20; `preregistration_h1_gemma.md`)
138 baseline-correct items, 57 successes at α = 1.0: H1-notes not met (τ_L24 AUC for failure 0.259), H1-dev met (stratified 0.364, p_less = 0.007), H1-loading not met. The reversal (high τ marks success) replicates on a second architecture; the first-order effect size predicts failure (AUC 0.646, stratified 0.615, p = 0.015). τ behaves as a proxy for the size of the first-order push on both models.

# Does second-order structure add to the Jacobian lens? Pre-registered tests on three language models

**Status: DRAFT (2026-10-17).** Written from the handoff (`handoff.md`), `writeup_plan.md`, the pre-registrations in `hlens/docs/preregistration_*.md` and the result files in `results/`.
Every number below comes from those; the section "Where each number lives" at the end maps claims to files. Nothing here has been peer reviewed, and the authors of the hypotheses and the analyst are the same project.

## 1. Summary

The Jacobian lens (jlens, Anthropic) reads what a residual stream "means" with a first-order, per-layer linear map. `hlens` asks whether the second-order (curvature) structure carries information the first-order lens misses.
We built the tooling (Hessian-vector products through arbitrary layer slices, restricted Hessians, a trust score, validity checks, randomized-model controls), reproduced the paper's checks on a hybrid linear-attention model
(Qwen3.5-4B) and a standard transformer (Gemma 3 12B), and tested eleven hypotheses, each pre-registered with a hash-recorded protocol and a randomized-weights control where one applies.

**What we found, briefly.**
1. **One positive, replicated result (claim B′).** The mean curvature of the MLP branch along lens directions exceeds its curvature along random directions in the upper layers, and a weight-randomized model shows no such difference
   (null ratios 0.994-1.0032 across 20 randomized-model arms). It holds on Qwen3.5-4B (ratio 1.15 → 2.20 from L16 to L30), on Gemma 3 12B (4.1 → 13.0 from L24 to L44) and on Llama 3.1 8B (1.40 → 2.83 from L16 to L30), three different architectures, on prompts and seeds not used to design the test. The original, stricter version (B) failed its own rules, see 4.2.
2. **One narrow result (claim A).** Near the point where an interpolated word embedding flips the model's answer, raising one country concept lowers the sensitivity of the other (negative second-order cross-terms, with a positive first-order cross-effect). It holds on Qwen3.5-4B
   (24/24 trials; 120 held-out trials) and **fails on Gemma 3 12B and on Qwen3.6-27B**. We report it as specific to Qwen3.5-4B.
3. **The first-order lens is sufficient for what we could test, on two architectures for the swap.** The second-order trust score marks swap *success*, not failure, on both Qwen3.5-4B and Gemma 3 12B (H1); the first-order margin shift predicts swap success better than the nonlinear gain (E1);
   binding pairs, the remainder, early-layer content, and list loading show no second-order signal (H3, H5, H7, H9 part 2). A persona-vs-register dissociation seen in a 10-pair run did not survive 100 pairs per family (H10): every family shows the ignition signature, and persona is not special.
4. **Two structural findings generalize.** Per-block nonlinearity *rises* toward the output on both models (H8; no "motor collapse"), and the coordinate swap is a causal, control-separated effect on both (held-out suite), with a model-dependent best strength.

## 2. What was built and how it was tested

- **Tooling** (`hlens/`): differentiable layer slices (`functional.py`), reverse-over-reverse, forward-over-reverse and finite-difference HVPs agreeing to ~1e-6 in fp32 (`hvp.py`), restricted Hessians, the linearization-error score
  τ = ‖½ δᵀHδ‖ / ‖Jδ‖, plateau/agreement/validity checks (`harness.py`), weight randomization with a re-fitted lens (`reinit` of HF's own initialization), sparse pursuit over the lens dictionary, interventions (coordinate swap).
  Unit tests run on tiny CPU models (Qwen3.5-like, Gemma 3-like, untied-head Qwen) so architecture-specific code is checked without weights; real-model tests (`-m real_model`) check slice = hooked forward exactly and fp32 HVP agreement on the real weights.
- **Models.** Qwen3.5-4B (32 layers; 3:1 Gated-DeltaNet : full attention; fp32, eager). Gemma 3 12B-pt (48 layers; sliding + global attention; fp32, eager, ~45 GiB). Qwen3.6-27B (64 layers; used only for claim A; bf16 weights with an fp32 slice, see 2.3).
  Lenses: Neuronpedia's pre-fitted jlens files (Qwen3.5-4B n = 417, Gemma 3 12B-pt 775 prompts, Qwen3.6-27B n = 1000).
- **Pre-registration discipline.** Each hypothesis has a file in `hlens/docs/` fixing setup, statistics and decision rule before data; result files store the file's SHA-256; a real arm refuses to run before the weight-randomized arm under the same hash. Deviations are dated addenda
  (H3 synonyms, H5 and H9 part 2, the Gemma stimulus spacing). Replications use depth-matched layers (`matched_layer`: the Qwen3.5-4B layer ↦ the same depth fraction).
- **2.3 Mixed precision.** fp32 curvature of the 27B does not fit (≈ 108 GB). The model is kept in bf16 and only the HVP slice (17 blocks), the final norm and the head are fp32 copies, with rotary cos/sin recomputed in fp32. On Qwen3.5-4B against the full-fp32 H9 run this reproduced the result
  (sign agreement 98.6 %/100 %, Spearman 0.9987, median relative error 0.34 %, identical verdicts, threshold α* within 0.05 in 24/24 trials). It was validated for that one measurement, with an 8-block slice (the 27B uses 17).

## 3. Reproducing the paper's checks

| Check | Qwen3.5-4B | Gemma 3 12B |
|---|---|---|
| Layer landmarks (workspace range) | measured; workspace ≈ depth 0.3-0.9 | passed; L20-43 (depth 0.43-0.91) |
| Sparse decomposition of the residual | passed, with caveats | — |
| MLP-gain gate (context-free) | **not reproduced** | — |
| Coordinate swap, original 87-item suite | 35 % (paper: 54-70 % on Claude models); random controls 0/52 | pre-registered α = 1.25: 8/48 = 17 % (**not passed**); α = 1.0 (post hoc) 14/48 |
| Coordinate swap, held-out 165-item suite | α = 1.25: **60/114 = 52.6 %**, controls 0/0/0 (passes) | α = 1.25: **4/125 = 3.2 %** (fails); α = 1.0: **55/125 = 44.0 %**, controls 0/0/0 (passes) |

The held-out suite (`hlens/swap_suite.py`, generated from fact tables, disjoint from the original, fixed protocol) shows (i) Qwen's original rate was not inflated by tuning, (ii) Gemma's pre-registered strength was the wrong one and its α = 1.0 effect is real out of sample.
Scoring first used a case-sensitive first-token match, so currency items (the models answer "Euro") mostly dropped out at the baseline-correct filter; a pre-registered case-insensitive re-scoring recovers 13 currency items per model (baseline-correct 125 Qwen, 138 Gemma) and changes no verdict (Qwen α = 1.25 61/125 = 48.8 %; Gemma α = 1.25 5/138 = 3.6 %, α = 1.0 55/138 = 39.9 %); the recovered items hardly swap (1/13 on both models), and state-of-matter items still do not pass the baseline filter on Qwen.
A tokenizer bug found on the way (`first_token_ids` counted the bare-space token, so numeric answers falsely overlapped) was fixed and every swap result re-run (headline numbers moved ≤ 0.03 AUC).

## 4. Results by hypothesis

### 4.1 Claim A: second-order mutual inhibition at the ignition threshold (H9 precondition)
Setup: interpolate the `{W}` embedding between two countries; ℓ → m slice (Qwen: 12 → 20), readouts along the two lens vectors, cross-terms c_A, c_B of the restricted Hessian; inhibition I = −(c_A + c_B)/2; tests P0a (I > 0), P0b (peak at the threshold), P0c (beats a random orthogonal pair).

| Run | Trials | I at threshold | P0a / P0b / P0c | Both cross-terms negative |
|---|---|---|---|---|
| Qwen3.5-4B, original (clean spacing) | 24 | 0.110 | pass / pass / pass | 24/24 |
| Qwen3.5-4B, held-out (old spacing) | 120 | 0.118 | pass / pass / pass | 97.5 % |
| Qwen3.5-4B, held-out (clean spacing) | 120 | 0.083 | pass / pass / pass | 88 % |
| Gemma 3 12B, original / held-out | 24 / 120 | ~1e-4 | P0a pass; **P0b fails** (p ≈ 0.7-0.8) | 29 % / 30 % |
| Qwen3.6-27B (mixed precision), held-out | 120 | 0.0011 | **fail / fail / fail** (p = 0.098 / 0.99 / 0.13) | 17 % |
| Qwen3.6-27B (mixed precision), original set | 24 | −0.0010 | **fail / fail / fail** (p = 0.95 / 0.997 / 0.93) | 8 % |

The first-order cross-effect is *positive* where the second-order one is negative, so the J-lens (first-order) cannot show this. On Qwen3.5-4B the Taylor-validity gate (score ≥ 1) leaves 15 of 120 trials (old spacing), all of which still pass, with a smaller threshold-vs-flank margin; on Gemma 19 of 24 and 83 of 120 are valid and the verdict is unchanged;
on the 27B none of the 120 is valid. An exploratory scan (18 layer-pair and direction configurations per model, 24 trials) found the effect on Qwen3.5-4B in 14 of 18 configurations, with or without the J-lens (identity directions work in 8 of 9), but needing the target layer to be well below the output; on Gemma no configuration passes even without correction (the peak-at-threshold test never below p = 0.048), although a weak, threshold-unspecific positive inhibition exists. We state claim A as **a finding on Qwen3.5-4B** and not as a property of the lens, of the family, or of language models. Part 2 of H9 (inhibition grows with list length) was not supported (Spearman −0.51; Q3 p = 0.08; an under-powered single-layer occupancy measure).

### 4.2 Claim B and B′: curvature along lens directions (H2)
κ = ‖½ vᵀHv‖ of the MLP branch (own norms; Gemma's post-feedforward norm included) at layer l + 1, for unit directions v in three populations: lens vectors of actively used tokens, MLP-neuron output directions, and isotropic random directions; 30 prompts × 16 positions × 64 directions, prompt-cluster bootstrap.
- **B (pre-registered 2026-10-15) failed its own rules.** T1 passed on both models; T2 (median ratio ≥ 1.5) failed on Qwen (1.37); T3 ("the null's 95 % interval contains 1") failed in 7 of 8 null arms because bootstrap intervals were about ±0.0005 wide around ratios of 0.997-1.002 - an exact-equality rule is a poor test of "no effect".
- **B′ (new pre-registration with an equivalence margin δ = 0.05, fresh prompts, new seeds)** holds on all three models (Qwen3.5-4B, Gemma 3 12B; Llama 3.1 8B from a third pre-registration with the same rules and fresh seeds): all null arms inside [0.95, 1.05] (Qwen: 5 arms, 0.9998-1.0014; Gemma: 2 arms, 1.001 and 1.0032), the real lower bound above 1.05 at every layer tested at depth ≥ 0.5 (Qwen 8/8, Gemma 6/6), and the real lower bound above every null's upper bound at the matched layers.
  Qwen ratios by layer: L16 1.15, L18 1.29, L20 1.14, L22 1.32, L24 1.34, L26 1.58, L28 1.85, L30 2.20 (median 1.33); Gemma: L24 4.1, L28 5.3, L32 9.1, L36 9.7, L40 12.0, L44 13.0 (median 9.4); the ratio rises with depth (Spearman 0.93, 1.00).
- Llama 3.1 8B (dense full attention; 5 null arms 0.9994-1.0016; real lower bound above 1.05 at 8/8 layers; ratios L16 1.40, L18 1.27, L20 1.40, L22 1.39, L24 1.34, L26 1.84, L28 2.11, L30 2.83; median 1.40; Spearman 0.67).
- What it does and does not say: the lens directions are a preferred direction for MLP-branch curvature in the trained models and not in randomized ones. It does not say the model *uses* these directions, nor that the effect is large on Qwen (the median ratio is below 1.5). The first-order gain ratio (‖Jv‖, lens vs random) behaves differently on the two models: on Qwen it is about 0.8-1.1 for L16-28 (curvature is raised while gain is not) and 3.1 at L30; on Gemma it is also raised (1.8 at L24 → 3.8 at L44), so on Gemma curvature and gain are not dissociated.

### 4.3 What the first-order lens already covers (negative results)

| Hypothesis | Question | Outcome (Qwen3.5-4B unless stated) |
|---|---|---|
| H1 (trust score) | does τ flag swap failures? | τ is a valid linearization-error score (ρ = 0.95 with the true first-order error) but marks swap **success** on the dev suite (AUC 0.78 naive, ~0.66 stratified, p ≈ 0.07); not supported. **Held-out repeat (126 baseline-correct items; case-insensitive labels):** at α = 1.25 τ does not discriminate (naive AUC for failure 0.507, p = 0.46; stratified 0.614, p = 0.017 but the two strata disagree), at α = 1.0 the dev direction appears (AUC for failure 0.289, stratified 0.370, p = 0.012); the first-order effect size predicts failure best (AUC 0.70). τ behaves as a proxy for the size of the first-order push. **Gemma 3 12B, held-out suite (α = 1.0; 138 items, 57 successes; pre-registered): the reversal replicates** - AUC of τ at layer 24 for failure 0.259 (stratified 0.364, p = 0.007: high τ marks success), loading not significant after stratification, first-order effect size predicts failure (AUC 0.646, stratified 0.615, p = 0.015) |
| E1 (engagement) | does nonlinear gain explain swap success? | no. Dev suite: the first-order margin shift predicts success (AUC 0.75), the actual margin better (0.85), the nonlinear gain the opposite way (0.39; Spearman −0.87 with the first-order shift); only 2 of 18 successes need nonlinearity. **Held-out repeat (126 items, 61 successes; pre-registered): P1 fails (AUC of the nonlinear gain 0.505, p = 0.46), P2 fails (5 of 61 successes need nonlinearity), P4 fails (CI [−0.013, 0.278]); P5 passes (first-order AUC 0.645, p = 0.0016, above the gain's 0.505; margin AUC 0.806)** - the first-order lens suffices for the swap. **Gemma 3 12B, held-out suite (α = 1.0; 138 items, 57 successes; pre-registered): P1 fails (AUC of the nonlinear gain 0.247, successes saturate further below the first-order extrapolation), P2 fails (2 of 57 successes need nonlinearity), P4 passes alone (0.736 vs 0.425), so E1 is not supported; P5 passes (first-order AUC 0.807, margin AUC 0.956)** - the first-order lens suffices for the swap on two architectures |
| H3 (binding) | are binding pairs special in the Hessian cross-terms? | no: mean rank-fraction 0.496 (p = 0.55) over 35 usable items |
| H5 (remainder coupling) | do selected atoms couple to the remainder's curvature? | no, in two forms; my post-hoc "orthogonality bias" explanation was refuted by the follow-up |
| H6 (ignition as a curvature event) | does curvature peak at the width-collapse layers and the threshold? | Qwen: the three pre-registered tests pass but the support is three layers (L4, L20, L24; L4 carries 78 % of the effect) and the Taylor radius has collapsed at the knee there; a Lanczos convergence check (320 cells) shows the 8-iteration eigenvalues are converged (median error 0.01 %, decision statistic stable in 100 % of cases), and scaling λ by the residual norm shrinks the L4 lead from 5.4× to 1.6× over L20, so part of the L4 peak is a residual-scale effect. **Gemma: not supported** (T2 p = 0.90, T3 CI [−0.024, 0.175]; λ falls monotonically with depth) |
| H7 (early layers) | does curvature carry content where the first-order lens is blind? | no (first-order T = 0.72 at L8; curvature at L4 T = 0.55, p = 0.19; 30 items) |
| H8 (motor collapse) | does the layer-to-output map lose curvature near the output? | **no on both models**: `J_ℓ → I` holds on Qwen (distance 0.996 → 0.461) and is weak on Gemma (1.00 → 0.83), but per-block nonlinearity rises toward the output (ratio 3.73 Qwen, 5.05 Gemma; the randomized models also rise, 2.33 on Gemma) |
| H10 (register vs persona) | do persona/register pairs show the ignition signature? | exploratory proxy. A 10-pair run suggested persona does and register does not, and a 10-pair distance-matched control showed the control does too. **At 100 pairs per family (732 trials; pair-level tests) all four families reproduce the signature** (persona, near-synonym, matched object-noun control, countries), with different strengths (median pair D̄ 0.49 countries, 0.36 control, 0.30 persona, 0.058 near-synonym). Persona is not larger than its matched control (median difference −0.043, p = 0.08; regression coefficient −0.09, 95 % CI [−0.18, −0.006]); endpoint distance correlates with the effect within families (Spearman ≈ 0.3) but not across them (overall 0.185). So: no persona-specific effect; the earlier register result was a small-sample effect |
| H4, H11 | automatization; reflection-trained checkpoints | not run: need checkpoints we do not have |

## 5. Limitations (stated plainly)

- **Few models.** Three architectures for B′ (Qwen3.5-4B, Gemma 3 12B, Llama 3.1 8B) and two for the swap and E1; A rests on one model; the 27B was run only for A, in mixed precision validated on a smaller slice, with a lens fitted on 1000 prompts (the 4B's was 417) and with 0/120 trials passing the validity gate.
- **Small and templated stimuli.** 24-trial and 120-trial country sets; 87-item hand-written and 165-item generated swap suites; 100 pairs per H10 family (2 carriers each; the earlier 10-pair runs are superseded); trials sharing a pair or a relation are not independent, so reported p-values and Wilson intervals are optimistic.
- **Post-hoc elements, all dated and flagged:** the H5 follow-up, H9 part 2, the H3 synonym addendum, the Gemma stimulus-spacing option, the reading of Gemma's α = 1.0 swap result (confirmed out of sample), and B′ itself (designed after seeing B's failure; tested on fresh data).
- **Measurement scope.** κ is the MLP-branch curvature at one layer at a time; "lens directions" are those of actively used tokens, which are not independent of the residual's content (some of the effect may be alignment with high-variance directions); randomized-model nulls use one lens-fitting recipe (30 prompts).
- **Compute and environment.** Single machine (Jetson Thor, 122 GiB unified memory). Several long runs needed relaunches (a kernel/driver memory event, two reboots, an analysis bug that discarded a 4.9 h run before checkpointing was added); nothing reported depends on a partially completed run.
- **Interpretation.** Nothing here shows second-order structure *explains* model behaviour beyond the first-order lens; the evidence for "the first-order lens suffices" is the failure of the hypotheses we tested, not a direct proof.

## 6. What to do next (if this goes further)
What explains the H10 effect beyond slot fit, distance and embedding similarity (the covariates explain 24 % of pair-level variance and not the family gaps; country pairs are strongest);
B′ on a fourth model or size (the 27B needs a null lens fit and a mixed-precision check of the curvature measurement); more than two seeds on Gemma nulls.

## Where each number lives
Pre-registrations: `hlens/docs/preregistration_{h3,h5,h6,e1,h7,h8,h9,h10}.md`, `_replication.md`, `_b_prime.md`, `_mixed_precision.md`, `_27b_claim_a.md`, `_h6_h8_gemma.md`, `_swap_heldout.md`, `_h10_extended.md`, `_h10_control.md`, `_h10_scaled.md`, `_heldout_scoring_h1_e1.md`, `_e1_gemma.md`, `_h1_gemma.md`, `_b_prime_llama.md`.
Results: `results/` (`h9_replication/`, `h2_replication/`, `h6_*`, `h8_motor_collapse_*`, `swap_heldout_*.json`, `gates_gemma-3-12b.json`, `mixed_precision_check_qwen3.5-4b.json`, `h10_extended_qwen3.5-4b.json`, earlier Qwen files); narrative and bug history: `handoff.md`; plan and result tables: `writeup_plan.md`.
Code: `hlens/` (library), `hlens/experiments/` (one script per experiment), `tests/` (fast suite 148 passed, 2 skipped, 45 real-model tests deselected by default).

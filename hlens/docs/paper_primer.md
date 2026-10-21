# Paper primer: the H-lens program (methods, reasons, experiments, results, hypotheses)

**Purpose:** one document to draft the paper from. It explains what each method is and why it was used, lists every experiment and hypothesis, gives the results with the numbers, says how strong each claim is and how to word it, and records the pitfalls found on the way.
**Status:** written 2026-10-20 from `handoff.md`, `writeup_plan.md`, `writeup_draft.md`, `research_notes.md` and the result files. Every number comes from `results/` (map in section 10). Where a number is exploratory or post hoc, the text says so.
**How it relates to `writeup_draft.md`:** the draft is a paper-shaped summary; this primer is the longer reference behind it (reasons, glossary, claims ledger, pitfalls, wording advice).

---

## 0. The whole thing in one page

**Question.** The Jacobian lens (J-lens, Anthropic's `jlens`) reads an activation using the *first-order* (linear) effect of that activation on the model's output, averaged over contexts. Many things it was used for (swaps, ignition, capacity, binding) are nonlinear phenomena measured with a linear tool. The H-lens program asks: **does second-order structure (curvature, Hessians) add anything the first-order lens misses?**

**Method in one line.** Compute Hessian-vector products (HVPs) through layer slices of real models, restrict them to J-lens directions, validate every number with reliability checks and weight-randomized controls, and pre-register every test.

**Models.** Qwen3.5-4B (hybrid linear/full attention), Gemma 3 12B (sliding + global attention), Llama 3.1 8B (dense full attention); Qwen3.6-27B only for one claim (mixed precision). All on one Jetson Thor.

**What we found.**
1. **Claim B′ (positive, replicated on three architectures):** the curvature of the MLP branch along J-lens directions exceeds its curvature along random directions in the upper layers; weight-randomized models show no such difference (20 null arms, ratios 0.994-1.0032). Ratios: Qwen 1.15 → 2.20 (L16 → L30), Gemma 4.1 → 13.0 (L24 → L44), Llama 1.40 → 2.83 (L16 → L30).
2. **Claim A (narrow, Qwen3.5-4B only):** near the point where an interpolated word embedding flips the answer, two competing concepts show negative *second-order* cross-terms (while the first-order cross-effect is positive). It holds on Qwen3.5-4B and fails on Gemma 3 12B and Qwen3.6-27B.
3. **The first-order lens suffices for the swap (two architectures):** the nonlinear gain does not predict swap success; the first-order extrapolation does; the trust score τ behaves as a proxy for the size of the first-order push. Binding, the remainder, early-layer content, and capacity-as-competition show no second-order signal.
4. **Structural findings:** per-block nonlinearity *rises* toward the output (no "motor collapse"); the coordinate swap is a causal, control-separated effect on both models, with a model-dependent best strength; the ignition-as-curvature-event hypothesis (H6) is weak on Qwen and absent on Gemma; a persona-vs-register effect seen with 10 pairs did not survive 100 pairs.

**What we did not show.** That the model *uses* the curvature along lens directions. B′ shows structure exists, not that it matters for behaviour.

---

## 1. Background and motivation

### 1.1 The J-lens
For each layer ℓ the J-lens is one matrix, the corpus-averaged Jacobian of the final residual with respect to the layer-ℓ residual:
`J_ℓ = E_{t, t'≥t, prompt}[ ∂h_final,t' / ∂h_ℓ,t ]`. An activation is read as `lens(h_ℓ) = softmax(W_U · norm(J_ℓ h_ℓ))`. The rows of `W_U J_ℓ` are the **J-lens vectors**, one per vocabulary token. The **J-space** is the set of sparse non-negative combinations of at most about 25 of these vectors; it typically explains ≤ 10 % of an activation's variance but carries most of the causal effect in swaps.
Operations on the J-space (steering, ablation, the **coordinate swap** `h ← h + V(σ(c) − c)` with `c = V⁺h`) are all *linear* in the activation. We used Neuronpedia's pre-fitted lenses.

### 1.2 Why second order
The corpus average discards context-specific structure, and the lens is a first-order object. Open questions in the original paper are nonlinear in nature: what "ignition" (a sudden commitment) is, why capacity saturates near 25 concepts, whether broadcast gain is weight alignment or nonlinear amplification, whether early layers are empty or just linearly inaccessible, how concepts are *bound*, and how comprehensively the lens covers reasoning (including automatized behaviour). The H-lens adds the next Taylor term.

### 1.3 What the H-lens object is
For a map `f` (layer slice, or layer-to-output) the Hessian is a 3-tensor that we never materialize. We fix a **contraction** to a scalar and compute HVPs:
- **cotangent contraction** `⟨c, f(h)⟩` for layer-to-layer measurements;
- **logit contraction** (Hessian of one labelled logit);
- **norm form** for the trust score: `‖½ δᵀHδ‖` computed as one JVP of a JVP.
The **restricted Hessian** over `k` directions `{v_i}` is `H_ij = v_iᵀ H v_j` (k HVPs, symmetrized; the asymmetry is a free estimator-error check).
The **trust score** is `τ = ‖½ δᵀHδ‖ / ‖Jδ‖` for the perturbation `δ` actually applied (the swap delta): how large the second-order change is relative to the first-order one.

---

## 2. Glossary (use these definitions in the paper)

| Term | Meaning |
|---|---|
| J_ℓ, J-lens vector | corpus-averaged Jacobian at layer ℓ; rows of `W_U J_ℓ` |
| coordinate swap | replace the coordinate of the source concept by that of the target in the J-lens basis, at a band of layers, at every position; scale α |
| band | contiguous layers of the swap (Qwen L12-20, Gemma L18-30, Llama not used for swaps) |
| HVP | Hessian-vector product; `hvp_rr` reverse-over-reverse (reference), `hvp_fr` forward-over-reverse (production), `hvp_fd` finite difference (validation) |
| restricted Hessian | the matrix of `vᵢᵀ H vⱼ` over a few directions |
| τ (trust score) | `‖½ δᵀHδ‖ / ‖Jδ‖` along the applied perturbation |
| κ | `‖½ vᵀ H v‖` of the MLP branch (its own norms included) along a unit direction `v`; the B/B′ measurement |
| κ ratio | median κ over lens directions divided by median κ over isotropic random directions, per layer |
| g1 | `‖J v‖`, the first-order gain of the MLP branch along `v` |
| λ | leading |eigenvalue| of the commitment map's Hessian (Lanczos), the H6 measurement |
| ignition protocol | interpolate the `{W}` embedding `(1-α) e_B + α e_A` between two words; the model's answer flips at a threshold α* (projection-share 0.5 crossing at a mid layer) |
| projection share | `⟨h(α)-h(0), h(1)-h(0)⟩ / ‖h(1)-h(0)‖²` at the `{W}` position |
| inhibition I | `−(c_A + c_B)/2`, where `c_A, c_B` are the cross-terms of the restricted Hessian of two readouts over the two concept directions (claim A) |
| margin path, N, FO | `m(α) = logit(swap) − logit(answer)` along the swap path; `FO(α) = m(0) + α·g` the first-order extrapolation; `N = m(α_ref) − FO(α_ref)` the nonlinear gain (E1) |
| null arm / real arm | the same pipeline on a weight-randomized copy (with a lens re-fitted on it) / on the real model |
| matched layer | the layer at the same depth fraction as a reference layer of Qwen3.5-4B (`matched_layer`) |
| pre-registration | a file fixing the setup, tests and decision rule before data; its SHA-256 is stored in every result file |
| equivalence margin δ | the B′ null rule: the whole 95 % interval of the null ratio must lie in [0.95, 1.05] |

---

## 3. Methods, and why each was used

### 3.1 Estimators and the reliability harness
- **Three HVP estimators** (`hvp.py`), built in this order and required to agree on analytic quadratics, a tanh MLP against `torch.autograd.functional.hessian`, and a tiny decoder before touching a language model. On real models in fp32: forward-over-reverse agrees with reverse-over-reverse to about 1e-6 (Qwen), 5.5e-7 (Gemma), 1.2e-6 (Llama).
  *Why:* a curvature number from a single estimator is not trustworthy; agreement is the first validity gate. Forward-over-reverse is the production path (no double graph in memory); finite difference is validation only.
- **Functional layer slices** (`functional.py`): the blocks l+1…m as a pure function of the layer-l residual, with rotary embeddings, masks and position ids captured per block and cached. Validated to match the hooked forward *exactly* (difference 0) on every model.
  *Why:* HVPs need a pure function; mixed full/linear attention (Qwen) and sliding/global attention (Gemma) need per-block kwargs. A bug here (one block's kwargs reused for all) gave errors of 4-5.5 on mixed slices before it was fixed.
- **Plateau, agreement mask, multi-step HVP, validity score** (`harness.py`): no curvature value without a flat finite-difference plateau; the Taylor-validity score compares the estimated Taylor radius with the intended step. Score < 1 means the local quadratic picture has broken down.
  *Why:* curvature spikes can be estimator artifacts; the notes made a knee-versus-Taylor-radius diagnosis mandatory.
- **Lanczos** (8 iterations) for the leading eigenvalue in H6; a later convergence check (320 cells, 40 iterations) showed the 8-iteration values are converged (median relative error 0.01 %, 90th percentile 0.7 %, decision statistic stable in 100 % of cases).
- **Stress test** (a constructed sharp top-1 flip) passes: single-step HVP degrades and multi-step plus gating recovers.

### 3.2 Controls: weight-randomized nulls and random directions
- **Weight-randomized null:** re-initialize the model with HF's own initialization (`reinit`; seeds differ per arm), round the weights to bf16 precision, **re-fit a small lens on the randomized model** (30 prompts), and run the identical measurement. *Why:* the pre-fitted lens is meaningless on random weights; a faithful null re-runs the whole pipeline. It tells us whether a result needs *learned* structure.
- **Random-direction controls:** isotropic random directions (B/B′), random orthogonal pairs (claim A), random-direction swaps (swap suites; 0 hits in nearly all runs).
- **A real arm refuses to run before its null arm** under the same pre-registration hash (H3, H5, H6, H2 replications).

### 3.3 Pre-registration and statistics
- **Discipline:** every hypothesis has a file in `hlens/docs/preregistration_*.md` written before data, stating setup, tests, decision rule and known limits; the result files store its SHA-256. Deviations are dated addenda (H3 synonyms, H5 post-hoc follow-up, H9 part 2, the Gemma stimulus spacing, the Lanczos cell count). These are internal (same team), not externally registered.
- **Tests used:** AUC with bootstrap intervals; one- and two-sided permutation tests; **stratified** AUC and stratified permutation tests (strata geography vs other, defined post hoc after H1); one-sided Wilcoxon signed-rank (paired); Mann-Whitney; binomial test; Spearman; Wilson intervals for rates; pair- and prompt-cluster bootstraps; Bonferroni for multi-configuration scans; **equivalence testing** (B′) in place of "interval contains 1".
- **Unit of analysis:** the pair (H10 scaled) or the item/prompt (cluster bootstrap) rather than the trial, to avoid inflating significance with non-independent trials. Remaining p-values and Wilson intervals are still optimistic where items share relations.
- **Lesson learned (B):** a null rule of "95 % interval contains exactly 1" fails when intervals are ±0.0005 wide. B′ replaced it with an equivalence margin chosen from reasoning (δ = 0.05, ten times below the smallest effect of interest), not from the data.

### 3.4 Models, lenses, precision, hardware
| Model | Layers / d | Notes |
|---|---|---|
| Qwen3.5-4B | 32 / 2560 | 3:1 Gated-DeltaNet : full attention; plain-PyTorch fallback (flash-linear-attention, causal-conv1d and `kernels` deliberately not installed); lens n = 417 prompts |
| Gemma 3 12B-pt | 48 / 3840 | sliding + global attention; embedding scaled by √d; extra post-feedforward norm; lens 775 prompts; gated on HF |
| Llama 3.1 8B | 32 / 4096 | dense GQA/SwiGLU; lens 539 prompts; gated on HF |
| Qwen3.6-27B | 64 / 5120 | claim A only; lens n = 1000; fp32 does not fit (≈ 108 GB) |
- **fp32 for all curvature work** (bf16 is only a measurement; bf16 rounding swamps second differences), **eager attention** (fused SDPA kernels lack forward-mode autodiff), compile disabled.
- **Depth-matched layers** (`matched_layer`): absolute layer numbers from the paper or from Qwen are never carried over; a layer is mapped by its depth fraction. (Qwen3.5-4B and Llama 3.1 8B both have 32 layers, so their indices coincide.)
- **Mixed precision for the 27B:** bf16 model, with fp32 copies only of the HVP slice blocks (17), the final norm and the head; rotary cos/sin recomputed in fp32. Validated **only for the claim-A measurement** on Qwen3.5-4B against full fp32 (sign agreement 98.6 %/100 %, Spearman 0.9987, median relative error 0.34 %, identical verdicts, thresholds within 0.05 in 24/24); the 4B slice had 8 blocks, the 27B slice 17.
- **Hardware:** one Jetson Thor (122 GiB unified memory shared by CPU and GPU). Practical constraints that shaped the design: one large job at a time; page-cache drops between large jobs; per-trial checkpointing after a 4.9-hour run was lost.

### 3.5 Stimuli and datasets
- **Swap suites:** the 90-item `probe-swap.json` from the paper (87 usable on Qwen, 89 on Gemma, single-token entities) as the *dev* suite; a **held-out suite** of 165 generated two-hop items (`swap_suite.py`: city → country → capital/language/continent/currency, element → symbol/state, language → country → capital), disjoint from the dev suite, single-token entities in both tokenizers. Scoring: greedy next token equals the swap answer (first-token match), over baseline-correct items; **case-insensitive scoring** (except element symbols) added in a pre-registered rescoring.
- **Ignition stimuli** (`data/experiments/ignition.json`): country pairs in carrier sentences with a `{W}` slot; 24 original trials; 120 held-out trials (new carriers and pairs); for H10: noun pairs in noun-slot carriers.
- **Text prompts:** WikiText prompts for contexts (evaluation prompts 0-29 and 200-229; lens-fitting prompts 100-129 and 300-329; all disjoint).

---

## 4. Hypotheses and experiments (the full ledger)

Legend for strength: **C** = confirmatory (pre-registered, fixed rule, fresh data), **R** = pre-registered replication, **E** = exploratory (descriptive; no verdict), **P** = post hoc.

| ID | Hypothesis | Why it was asked | Experiment (short) | Outcome |
|---|---|---|---|---|
| Gates | the pipeline reproduces the paper's checks on an open model | nothing new is trusted otherwise | sparse decomposition; coordinate swap; layer landmarks; MLP gain | decomposition passed (with caveats); swap partial; landmarks measured; **MLP gain not reproduced** |
| Early averaging | an *averaged* H-lens object exists like `J_ℓ` | decides whether claims are per-context | restricted Hessians of 100 contexts at L16 | second-order ratio 0.348 vs sign-null 0.109; randomized model 0.294: **no evidence of a learned averaged object** |
| H1 (C, R) | τ predicts swap failure | notes' first test; targets the lens's blind spot | τ at three layers on the swap suites | **not supported**; τ marks success / effect size (Qwen dev, Qwen held-out, Gemma held-out) |
| E1 (C, R) | swap success is carried by nonlinear amplification | follow-up to H1's reversal | margin paths; N vs first-order | **not supported** (Qwen dev + held-out, Gemma held-out); first-order lens suffices (P5) |
| H2 → B → B′ | curvature along J-lens directions exceeds random | tests whether broadcast gain is partly nonlinear | κ ratio per layer with randomized nulls | B failed its own rules; **B′ holds on Qwen, Gemma, Llama** |
| H3 (C) | binding lives in Hessian cross-terms | notes' relational-structure test | cross-terms of binding vs matched pairs | **not supported** |
| H4 | automatization reduces curvature | alignment-relevant | needs reward-hacking checkpoints | **blocked** (no checkpoints) |
| H5 (C) | the remainder is coupled to selected atoms through curvature | tests whether the remainder is inert | remainder cross-terms vs control atoms | **not supported** (two forms) |
| H6 (C, R) | ignition is a curvature event | flagship mechanistic hypothesis | Lanczos λ along the interpolation path | Qwen: tests pass but weak; **Gemma: not supported** |
| claim A / H9 precondition (C, R) | competing concepts inhibit each other near threshold at second order | "sharp, competitive ignition" | restricted-Hessian cross-terms at threshold | **holds on Qwen3.5-4B only** |
| H7 (C) | curvature carries content where the first-order lens is blind (early layers) | early layers empty or inaccessible? | content probe along lens directions | **not supported** |
| H8 (C, R) | motor-regime collapse (curvature collapses near output) | cheap test of an inference | `J_ℓ → I`, per-block nonlinearity | **not supported on Qwen and Gemma** |
| H9 part 2 (C, P) | competition grows with list load | capacity ceiling as dynamics | list-loading protocol | **not supported** |
| H10 (E, C) | persona/register pairs show the ignition signature | persona flagging in the paper | 10 pairs; control; 100 pairs per family | **no persona effect**; all families show the signature |
| H11 | reflection training lowers curvature barriers | alignment payoff | needs two checkpoints | **blocked** |

---

## 5. Results in detail

### 5.1 Reproduction gates
- **Sparse decomposition (Qwen):** median J-space share of variance 7-9 % at layers 12, 16, 20 (paper: ≤ 10 %). Caveats: a *random* dictionary of the same size explains more variance (0.151 vs 0.072 at L16), so the number is a correct reproduction but not evidence of alignment; "about 25 active atoms" is not actually tested.
- **Coordinate swap, dev suite:** Qwen best 35 % (band L12-20, α = 1.25, raw lens vectors) against 0/52 random-control hits; the paper reports 54-70 % on Claude models. Gemma at the depth-matched band L18-30: α = 1.25 8/48 = 17 % (not passed), α = 1.0 (post hoc) 14/48.
- **Coordinate swap, held-out suite:** Qwen α = 1.25 **60/114 = 52.6 %** (controls 0/0/0); Gemma α = 1.25 **4/125 = 3.2 %** (fails), α = 1.0 **55/125 = 44.0 %** (controls 0/0/0). After case-insensitive rescoring: Qwen 61/125 = 48.8 %, Gemma α = 1.25 5/138 = 3.6 %, α = 1.0 55/138 = 39.9 %; the recovered currency items hardly swap (1/13). Conclusions: Qwen's dev rate was not inflated by tuning; the swap is real on both models; the useful α is model-dependent.
- **Layer landmarks:** Qwen workspace ≈ depth 0.3-0.9 (autocorrelation onset L9, effective-dimension onset L14, kurtosis onset L18; offsets L28-L30); the metrics disagree by about 9 layers, so no sharp onset is quoted. Gemma: workspace range L20-43 (depth 0.43-0.91); landmark check passed.
- **MLP gain (context-free), Qwen: not reproduced.** The paper reports ~10× J-lens gain in the workspace band; we see 1.2-1.8 at L16-25 and a peak of 8.3 only at L30 (the motor end); neuron directions ≈ 1.1-1.9. Reported as a negative reproduction (possibly scale or an unstated convention). H2 then asked the *in-context* version.
- **Early averaging:** see the ledger.

### 5.2 Claim B′ (the positive result) in detail
**Measurement.** κ = ‖½ vᵀ H v‖ of the MLP branch at layer l+1 for three populations of unit directions: J-lens vectors of actively used tokens, MLP-neuron output directions, isotropic random directions; 30 prompts × 16 random valid positions × 64 directions; prompt-cluster bootstrap (300 resamples); contexts are the residuals entering the MLP.
**B (first version, failed its own rules):** T1 passed on both models, T2 (median ratio ≥ 1.5) failed on Qwen (1.37), and T3 (null interval contains 1) failed in 7 of 8 null arms because the intervals were extremely narrow.
**B′ (new pre-registration; equivalence margin 0.05; fresh prompts 200-229 and 300-329; new seeds):**
- **N:** every null arm's whole interval in [0.95, 1.05]. **R:** the real lower bound above 1.05 in ≥ 75 % of layers at depth ≥ 0.5. **C:** at each null-lens layer in that range, the real lower bound exceeds every null arm's upper bound.

| | Null arms (ratio) | R: layers above 1.05 | Real ratio by layer (median over depth ≥ 0.5) | Spearman(layer, ratio) |
|---|---|---|---|---|
| Qwen3.5-4B | 5 arms: 0.9998-1.0014 | 8/8 (L16-30) | L16 1.15, L18 1.29, L20 1.14, L22 1.32, L24 1.34, L26 1.58, L28 1.85, L30 2.20 (median 1.33) | 0.93 |
| Gemma 3 12B | 2 arms: 1.001, 1.0032 | 6/6 (L24-44) | L24 4.1, L28 5.3, L32 9.1, L36 9.7, L40 12.0, L44 13.0 (median 9.4) | 1.00 |
| Llama 3.1 8B | 5 arms: 0.9994-1.0016 | 8/8 (L16-30) | L16 1.40, L18 1.27, L20 1.40, L22 1.39, L24 1.34, L26 1.84, L28 2.11, L30 2.83 (median 1.40) | 0.67 |

Across 20 randomized-model arms the ratio is 0.994-1.0032.
**First-order gain (g1) ratio** (descriptive): Qwen ≈ 0.8-1.1 for L16-28 and 3.1 at L30; Llama 0.88-0.99 to L24, 1.19 at L28, 2.52 at L30 (curvature raised while gain is not); Gemma 1.8 at L24 rising to 3.8 at L44 (both raised). Neuron-direction κ ratios: 1.06-1.40 on Llama.
**What to claim:** MLP-branch curvature along lens directions is above random in the upper layers of three architectures and absent in weight-randomized controls. **What not to claim:** that the model uses it, that the effect is large (median ratios 1.33 and 1.40 on two models are below the 1.5 "large" label), or anything about other layers or other branches.
**Honesty point:** B′ was designed after B failed, using knowledge of B's data; the fresh data and pre-registered margin are what make it a test.

### 5.3 Claim A (the narrow result) in detail
**Measurement.** Interpolate the `{W}` embedding between two countries; find the threshold α* (projection share 0.5 at the mid layer); at α* and α* ± 0.2 compute the restricted Hessian over the two J-lens directions at layer ℓ for two readouts (the unit J-lens vectors of the two countries at layer m); I = −(c_A + c_B)/2. Tests: **P0a** (I > 0), **P0b** (I at threshold > mean of flanks), **P0c** (I beats a random orthogonal pair), each one-sided Wilcoxon p < 0.01.

| Run | n | I at threshold | Tests | Both cross-terms negative |
|---|---|---|---|---|
| Qwen3.5-4B, original set | 24 | 0.110 | all pass | 24/24 |
| Qwen3.5-4B, held-out (old / clean spacing) | 120 | 0.118 / 0.083 | all pass | 97.5 % / 88 % |
| Gemma 3 12B original / held-out | 24 / 120 | ~1e-4 | P0b fails (p ≈ 0.7-0.8) | 29 % / 30 % |
| Qwen3.6-27B held-out / original | 120 / 24 | 0.0011 / −0.0010 | all three fail | 17 % / 8 % |

The first-order cross-effect is positive (+0.26 to +0.33 on Qwen) where the second-order one is negative; hence the J-lens cannot show this. **Exploratory scan** (18 layer-pair × direction-source configurations per model, 24 trials): Qwen passes in 14 of 18 (the effect needs the target layer well below the output, m < 24, and does not need the J-lens: identity directions pass in 8 of 9); Gemma passes in 0 of 18 (the peak test never below p = 0.048), though a weak, threshold-unspecific positive inhibition exists in 14 of 18.
**Validity gate:** Qwen 15 of 120 trials valid (all pass); Gemma 19 of 24 and 83 of 120 valid (verdict unchanged); the 27B 0 of 120.
**What to claim:** claim A is a finding on Qwen3.5-4B. **What not to claim:** a property of the J-lens, of the Qwen family (the 27B fails), or of language models. The 27B used mixed precision validated on a shorter slice and a different lens fit; that caveat remains.
**Part 2 (list loading, H9):** inhibition falls with list length (Spearman −0.51; Q3 p = 0.08) with a single-layer, under-powered occupancy measure; not supported (a post hoc addendum written after the precondition result).

### 5.4 The first-order lens suffices: H1 and E1
**H1 (trust score).** On the dev suite τ is a valid linearization-error score (ρ = 0.95 with the true first-order error) but marks swap **success**: AUC for failure 0.78 naive and ~0.66 stratified (p ≈ 0.07); workspace loading was the better failure predictor (stratified 0.72, p = 0.017).
- **Qwen held-out (126 items; 61 successes at α = 1.25):** τ_L16 AUC for failure 0.507 (p = 0.46), stratified 0.614 (p = 0.017; the strata disagree) → H1-notes, H1-dev, H1-loading all not met. At α = 1.0 (39 successes): AUC 0.289, stratified 0.370 (p_less = 0.012): the dev direction appears. First-order effect size predicts failure best (AUC 0.70).
- **Gemma held-out (138 items; 57 successes at α = 1.0):** τ_L24 AUC for failure 0.259, stratified 0.364 (p_less = 0.007) → H1-dev **met**; loading not met after stratification; first-order size AUC 0.646 (stratified 0.615, p = 0.015); Spearman(τ, first-order size) 0.437 (Qwen 0.59).
- **Reading:** τ is largely a proxy for the size of the first-order push; the notes' prediction (high τ marks failure) is not supported on either model.
**E1 (engagement).** Hypothesis: swap success is carried by nonlinear amplification. Quantities: margin `m(α)` along the swap path; first-order prediction FO; nonlinear gain `N = m(α_ref) − FO(α_ref)`. Tests: P1 (N predicts success), P2 (most successes would not flip under the first-order prediction), P4 (specific to the trained model vs randomized), P5 (first-order prediction beats N; added in the held-out repeat).

| | n / successes | AUC of N | successes needing nonlinearity | first-order AUC | margin AUC | E1 | P5 |
|---|---|---|---|---|---|---|---|
| Qwen dev | 51 / 18 | 0.39 | 2 of 18 | 0.75 | 0.85 | not supported | (exploratory) |
| Qwen held-out (α 1.25) | 126 / 61 | 0.505 | 5 of 61 | 0.645 | 0.806 | not supported (P4 CI [−0.013, 0.278]) | pass |
| Gemma held-out (α 1.0) | 138 / 57 | 0.247 | 2 of 57 | 0.807 | 0.956 | not supported (P4 passes alone: 0.736 vs 0.425) | pass |

Successes saturate *below* their first-order extrapolation (mean N negative; −19.9 vs −7.5 on Gemma). **Conclusion:** the swap's effect is first order and saturating; curvature of the swap path does not explain success.

### 5.5 H3, H5, H7 (negative results on specific second-order claims)
- **H3 (binding in cross-terms):** 35 of 55 items usable (arithmetic must be answered correctly); the binding pair's mean rank-fraction of |H_ij| is 0.496 (p = 0.55), i.e. at the median of 28 pairs; high |H_ij| for arbitrary co-present pairs, so it measures co-activation. Useful by-product: the Taylor-2 prediction of the measured pairwise interaction is within 50 % for 93 % of pairs.
- **H5 (remainder coupling):** selected atoms couple slightly *less* to the remainder's curvature response than best-correlated unselected atoms; not supported in two forms. A post-hoc explanation I proposed ("orthogonality bias") was refuted by its own follow-up (T⊥ ≈ T) and is documented as such. What remains: lens directions in general couple to the remainder's curvature above chance in the trained model (1.45×/1.60× vs 1.14×/1.08× randomized).
- **H7 (early layers):** first-order lens already carries content at L8 (T = 0.72); at L4, where it is blind, second-order curvature is not significant (T = 0.55, p = 0.19; 30 items).

### 5.6 H6 (ignition as a curvature event) and H8 (motor collapse)
- **H6, Qwen (24 trials, randomized arm first):** the three pre-registered tests (peak in the width-collapse window; curvature larger at the threshold than at ±0.2; above the null) pass, but weakly: support rests on L4, L20, L24 (L4 carries 78 % of the pooled statistic); L10-L16 show no effect and L12/L14 reverse; at the threshold the Taylor radius has already collapsed at L16-L24 (0 % of cells valid); scale-free τ_path fails everything. A Lanczos check shows the eigenvalues are converged; scaling by the residual norm reduces the L4 lead over L20 from 5.4× to 1.6× (part of the L4 peak is residual scale).
- **H6, Gemma (depth-matched, 24 trials):** **not supported**: T2 p = 0.90, T3 interval [−0.024, 0.175]; λ falls monotonically with depth (6.3 at L6 to 0.02 at L36). Taylor validity: 10 of 50 cells below 1.
- **H8, Qwen:** `J_ℓ → I` is confirmed (distance to scaled identity 0.996 at L0 → 0.461 at L30) but per-block nonlinearity *rises* toward the output (motor/workspace ratio 3.73).
- **H8, Gemma:** `J_ℓ → I` is weak (1.00 → 0.83), and the ratio is 5.05 (randomized model also rises, 2.33). **No motor-regime collapse; the finding generalizes.**

### 5.7 H10: persona and register (a cautionary tale about small samples)
1. 10-pair run (4 pairs per family, 24 trials): persona pairs reproduced the ignition signature, register pairs did not → "proxy not supported" (both required).
2. Extended run (10 fresh pairs, 60 trials per family): persona reproduces (L20, L24 p ≈ 0); register does not (L20 p = 0.039, L24 p = 0.28); contrast p = 5e-10 / 2e-15.
3. Control (10 object-noun pairs matched to persona on endpoint distance, median |Δd| 0.002): the control reproduces the signature; persona not larger (p = 0.85; coefficient −0.29, CI [−0.77, 0.04]).
4. **Scaled run (100 pairs per family; countries 66; 732 trials; pair-level tests):** **all four families reproduce the signature** (persona, near-synonym, matched control, countries); median pair effect D̄: countries 0.49, control 0.36, persona 0.30, near-synonym 0.058. Matched persona − control: median −0.043, p = 0.08; regression persona −0.093 [−0.18, −0.006], near-synonym −0.086 [−0.23, 0.07], distance +1.00 [0.54, 1.51]; overall Spearman(distance, effect) 0.185 (within families ≈ 0.3).
5. **Exploratory covariates (366 pairs):** associated with the effect: endpoint distance at L24 (+0.19), input and unembedding cosine (−0.19), slot surprisal (−0.25: nouns that fit the carrier better have a larger effect), word length (−0.23); not output divergence. R² = 0.24; **the family gaps are not explained** (persona 22 %, near-synonym 31 % shrinkage; the countries advantage grows).
**Message:** "persona is special" was a small-sample artifact; this is worth a short, honest paragraph or an appendix.

### 5.8 Mixed precision, Lanczos, and other methodological results
See 3.1 and 3.4 for the numbers. These belong in a methods section or appendix.

---

## 6. How the findings fit together (candidate narratives)

**Narrative 1 (recommended): "Second-order structure exists and is aligned with the lens, but the first-order lens is enough for what we can test."**
- Lead with B′ (structure exists, three architectures, randomized controls) as the positive result.
- Then the negatives that bound it: swap success is first-order and saturating (H1, E1 on two architectures), binding/remainder/early-layer/competition show nothing.
- Present claim A as a narrow, model-specific finding (honest about its failure to transfer) and H6/H8 as "second-order is not where the ignition phenomenology lives".
- Emphasize methods: reliability harness, randomized nulls, pre-registration, replication on depth-matched layers.

**Narrative 2: "A reliability toolkit and a careful negative-results study."** If the venue values rigor over novelty: lead with the harness, pre-registration discipline, bug catalog and the stack of negative results; B′ as the single positive.

**Claims ledger (what each claim is worth)**

| Claim | Strength | Wording |
|---|---|---|
| B′ | confirmatory, three architectures, pre-registered after B failed | "curvature along lens directions exceeds random in upper layers; absent under weight randomization" |
| The first-order lens suffices for the swap | confirmatory (E1 + P5), two architectures | "the swap's effect on the answer margin is well predicted by its first-order extrapolation; nonlinear gain does not predict success" |
| τ is an effect-size proxy | confirmatory direction, two architectures | "τ tracks the size of the first-order push and does not flag failures" |
| Claim A | confirmatory on one model, fails elsewhere | "on Qwen3.5-4B ... fails on Gemma 3 12B and Qwen3.6-27B" |
| Per-block nonlinearity rises toward the output | confirmatory, two models | "no evidence of motor-regime collapse" |
| Swap works on both models at different strengths | confirmatory held-out | "a causal, control-separated effect; the best strength is model-dependent" |
| H6 / H3 / H5 / H7 / H9-2 | negative or weak | one sentence each in a negative-results table |
| H10 | negative after scaling | brief; shows the danger of 10-pair conclusions |
| Persona/register dissociation | **do not claim** | only as the cautionary example |

---

## 7. Pitfalls found (worth a methods/appendix paragraph, and a checklist for reviewers)

| Pitfall | What happened | Fix |
|---|---|---|
| First-token scoring bug | `first_token_ids` counted the bare-space token (Qwen tokenizes " 5" as [" ", "5"]), so numeric answers falsely overlapped; margins were identically 0 | skip whitespace-only tokens; all swap results re-run (headline AUC moved ≤ 0.03) |
| Case-sensitive scoring | "Euro"/"Solid" not credited, dropping currency items | pre-registered case-insensitive rescoring; no verdict changed |
| Per-block kwargs | one block's rotary/mask kwargs reused for the whole slice (errors 4-5.5 on mixed slices) | cache by (seq, block) |
| functorch "escape" | kwargs first captured inside a transform and reused later crashed | capture at base level (`_DisableFuncTorch`) + regression tests (twice: slices and fp32 rotary) |
| Gemma embedding scale | Gemma multiplies embeddings by √d; reading the raw weight row for interpolation endpoints would be off by 62× | use the module's output (`embed_rows`) |
| Gemma extra MLP post-norm | Gemma blocks have `post_feedforward_layernorm` | `mlp_branch` adapter |
| Double-space stimuli | `prefix + " " + word` gives a double space; Qwen splits it, Gemma merges it into a different token | `clean=True` option; Qwen re-run both ways |
| B's null rule | "interval contains 1" fails with ±0.0005 intervals | equivalence margin (B′) |
| Simpson-like strata | stratified and naive tests disagree (H1) | pre-registered rules require both; strata disagreements are reported |
| Small-sample dissociation | H10's persona/register result vanished at 100 pairs | scale-up before claiming |
| Memory | Thor's shared pool: OOM kills, phantom "used" memory; unsaved 4.9 h run lost | checkpoint every trial; drop caches; one big job at a time; allocation caps |
| Wait/monitor commands | foreground waits interrupted; unclear status | background waits; always state what is running |
| Post-hoc stories | H5's "orthogonality bias" explanation was refuted by its own follow-up | write explanations as hypotheses and test them |

---

## 8. Limitations and threats to validity (for the paper's limitations section)
- **Few models, few sizes.** Three architectures for B′; two for the swap/E1/H1; claim A on one model; the 27B only for A with a validated-on-smaller-slice mixed-precision path and a different lens fit.
- **Small, templated stimuli.** 24- and 120-trial country sets; 87- and 165-item swap suites (the held-out items are templated and city-dominated); pair-level n = 100; trials/items sharing a relation or word are not independent so p-values and Wilson intervals are optimistic.
- **What the measurements are.** κ is MLP-branch curvature at one layer at a time; J-lens vectors of *actively used* tokens are not independent of the residual's content (some of B′ may be alignment with high-variance directions); τ is evaluated along one intervention path; null arms use one lens-fitting recipe (30 prompts) and few seeds on Gemma (2).
- **Lens quality differs by model** (n = 417, 775, 539, 1000 prompts); lens readout quality was spot-checked only (e.g. the 27B lens reads blanks at its mid layer on one prompt).
- **Post-hoc elements** are dated and flagged (see the ledger); B′ and the stratification were designed after seeing earlier data.
- **No causal test** that curvature along lens directions matters for behaviour; "the first-order lens suffices" is supported by the failure of the hypotheses we tested, not proven.
- **Internal pre-registration** (same team, hash-recorded files, not an external registry).
- **Single machine/environment**; several runs needed relaunches (none of the reported results depends on a partially completed run).

---

## 9. Open questions and what a reviewer might ask
1. **Does the model use the curvature along lens directions?** A causal test (perturb along high-curvature lens directions, measure behavioural effect, with matched random controls) is the most valuable missing experiment.
2. **Why does B′ vary so much in size across models (1.3-1.4× vs 9×)?** Not explained; depends on layer, lens quality, architecture?
3. **Why does claim A fail off Qwen3.5-4B?** The exploratory scan excludes a layer-pair or lens explanation within its grid; model-specific?
4. **What makes country pairs the strongest ignition pairs** (H10 covariates explain 24 %)?
5. **B′ on a larger model** (needs a null lens fit and a mixed-precision check of the curvature measurement); more Gemma null seeds.
6. **The blocked hypotheses H4 and H11** (automatization, reflection training) need checkpoints.

---

## 10. Where everything lives

- **Pre-registrations** (`hlens/docs/`): `preregistration_{h3,h5,h6,e1,h7,h8,h9,h10}.md`; `_replication.md` (gates + H2/H9 replication; Addendum 1: stimulus spacing); `_b_prime.md`; `_b_prime_llama.md`; `_mixed_precision.md`; `_27b_claim_a.md`; `_h6_h8_gemma.md`; `_swap_heldout.md`; `_heldout_scoring_h1_e1.md`; `_e1_gemma.md`; `_h1_gemma.md`; `_h10_extended.md`; `_h10_control.md`; `_h10_scaled.md`; `_exploratory_a_h10.md`; `_lanczos_check.md`.
- **Result files** (`results/`): `h2_replication/` (B, B′, Llama: `B2_*`, `B3_*`), `h9_replication/` (claim A), `h9_scan/`, `h6/`, `h6_gemma-3-12b/`, `h8_motor_collapse_*`, `swap_heldout*_*.json`, `trust_score_*`, `e1_*`, `h10_*`, `lanczos_check_*`, `mixed_precision_check_*`, `gates_gemma-3-12b.json`; older Qwen files for H1-H8.
- **Narrative and history:** `handoff.md` (every experiment, every bug); `writeup_plan.md` (numbered result sections 6-23); `writeup_draft.md` (paper-shaped summary); `research_notes.md` (original program).
- **Code:** `hlens/` library (`functional.py`, `hvp.py`, `jspace.py`, `interventions.py`, `harness.py`, `landmarks.py`, `gain.py`, `interpolation.py`, `swaps.py`, `swap_suite.py`, `arch.py`, `loading.py`, `stats.py`), `hlens/experiments/` (one script per experiment), `tests/` (fast suite 160 passed, 4 skipped; 53 real-model tests: 45 of them, including the Gemma ones, passed in one combined session in 27 min 43 s; the 8 Llama tests passed in their own run before the Llama experiment).
- **Reproducing a result:** each experiment script's docstring gives its command (`.venv/bin/python -m hlens.experiments.<name> ...`); real-model tests: `uv run --no-sync pytest -m real_model`.

---

## 11. Suggested paper outline (for drafting)
1. Introduction: the first-order limitation of the J-lens; the question; contributions (a reliability toolkit; B′; the swap/first-order result; a bounded set of negative results).
2. Background: J-lens, J-space, swaps; second-order objects (HVP, restricted Hessian, τ, κ).
3. Methods: estimators and validation; slices; controls (randomized nulls); depth matching; pre-registration and statistics; models and precision; stimuli.
4. Reproducing the first-order checks (gates).
5. Main result: B′ (curvature along lens directions) on three architectures.
6. The first-order lens suffices for the swap: H1 and E1.
7. A model-specific second-order effect: claim A (and its failure to transfer).
8. What second order does not show: H3, H5, H6, H7, H8, H9-2, H10 (compact table; H10 as a cautionary tale).
9. Limitations; what a causal follow-up would need.
10. Appendix: mixed precision, Lanczos check, bug catalogue, pre-registration list with hashes, full tables.

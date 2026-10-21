# Pre-registration: H9 precondition (mutual inhibition near the ignition threshold; H6 sub-question (b))

Written 2026-10-14, **before any H9 measurement**. Notes §3 H9: "If H6 confirms inhibitory cross-terms, reframe the
~25-vector ceiling as a dynamical property." H6's sub-question (b) — "do simultaneously near-threshold candidates show
mutual-inhibition cross-terms?" — was not tested in H6. This file pre-registers that precondition. **If it fails, H9 is
reported as "precondition not met" and the list-loading part is not run as a test of H9.**
`hlens/experiments/competition.py` records this file's SHA-256 in its result file.

## Setup (fixed)
- The 24 H6 trials (`hlens.experiments.ignition.choose_trials`), fp32 real model; α*_t = the projection-share 0.5 crossing at
  layer 16; points `α ∈ {α* − 0.2, α*, α* + 0.2}`.
- Source layer ℓ = 12 (residual at `{W}`), target layer m = 20. Readouts `r_A(h) = ⟨w_A, h_m⟩`, `r_B(h) = ⟨w_B, h_m⟩` with `w_A, w_B` the unit
  J-lens vectors of the two countries at layer m (position `{W}`, through blocks ℓ+1…m). Directions `v_A, v_B` = the unit J-lens
  vectors of the two countries at layer ℓ.
- Cross-terms `c_A = (v_Aᵀ H_{r_A} v_B)`, `c_B = (v_Aᵀ H_{r_B} v_B)` from `restricted_hessian` over `{v_A, v_B}`.
  **Inhibition `I = −(c_A + c_B)/2`** (positive when raising one concept's activation reduces the other readout's sensitivity
  to the first: a competitive, divisive interaction).
- Control: the same computation for two random unit directions orthogonal to each other and to `v_A, v_B` (seed 0), per trial.

## Tests (precondition; all must hold)
- **P0a:** `I(α*) > 0`: one-sided Wilcoxon over trials, **p < 0.01**.
- **P0b:** `I(α*) > mean(I(α* ± 0.2))`: paired one-sided Wilcoxon, **p < 0.01** (inhibition is strongest at the threshold).
- **P0c:** `I(α*) − I_control(α*)`: one-sided Wilcoxon `> 0`, **p < 0.01** (not a generic property of any direction pair).

## Secondary, exploratory
Fraction of trials with both `c_A < 0` and `c_B < 0` at α*; the same at ℓ = 8 and 16; first-order cross-effects
`w_Bᵀ J v_A` (Jacobian-level inhibition); the list-loading protocol of the notes (`data/experiments/capacity.json`) only if P0a–c hold.

## Known limits (stated in advance)
One layer pair; two-direction restricted Hessian only; country pairs only; the readout axes are lens vectors, so "inhibition" is
relative to the lens's own vocabulary directions.

## Addendum 1 (2026-10-14; the precondition run is complete and MET; this part is written before any list-loading measurement)
Precondition result (`results/h9_competition_qwen3.5-4b.json`): P0a, P0b, P0c all pass (median I at the threshold 0.134, p = 6e-8; threshold vs
flank +0.047, p = 3e-7; control pair 0.0009); both cross-terms negative in 24/24 trials. (The first-order cross-effect was +0.26, i.e.
*facilitatory* at first order: the competition is a second-order, gain-control-type effect.) Per the notes the capacity question follows.

### H9 part 2 — does competition grow with load, and is it related to what stays in the workspace?
- Stimuli: `data/experiments/capacity.json` pools, families **countries** and **cities**; for each family 8 lists (seed 0) of 32
  single-token words (with a leading space), written `" w1, w2, …, wk,"`; evaluated at the comma after word `k ∈ {4, 8, 12, 16, 20, 24, 28, 32}`.
- At each (list, k): residual at layer 12 at the comma; directions = unit J-lens vectors (layer 12) of the **last 4** listed words (a fixed
  number of words, so the number of pairs does not grow with k); readouts `r_i = ⟨w_i, h_20⟩` (w_i = unit lens vector of word i at layer 20);
  `c_i = mean_{j≠i} H_{r_i}[i, j]`; **inhibition `I = −mean_i c_i`** (as in the precondition). Control: 4 random orthonormal directions orthogonal
  to those, same readouts.
- Occupancy `C(k)` = how many of the k listed words have lens rank ≤ 25 at layer 20 at that comma.
- **Q1 (competition grows with load):** one-sided Spearman between `k` and `I` over all (list, k) with `k ≤ 24`, **p < 0.01**; the same for the control
  must have p > 0.05.
- **Q3 (competition is associated with fewer words kept):** at each `k ∈ {24, 28, 32}`, one-sided Spearman between `I` and the fraction of listed
  words *not* in the lens top-25, pooled over lists after z-scoring within `k` and family: **p < 0.01**.
- **H9 is supported iff the precondition ∧ Q1 ∧ Q3.**
- Exploratory: `I(k)` and `C(k)` curves; whether `C(k)` plateaus near 25; `I(32) − I(24)`; first-order cross-effects.
- Limits: two families; 16 lists; a 4-word window; the layer pair (12, 20); lens-vector readouts; one sample of list order.

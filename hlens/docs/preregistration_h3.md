# Pre-registration: H3 (binding lives in cross-terms)

Written 2026-10-14, **before any H3 measurement was run on any model**. `hlens/experiments/binding.py`
records this file's SHA-256 in every result file and refuses to run the real arm unless the
randomized-model arm was already run under the same hash. Do not edit after the first run; amend
with a dated addendum.

## Hypothesis (research notes §3, H3)
Concepts that are jointly operated on (the number and the operation feeding an arithmetic
result) have elevated `|H_ij|` relative to co-present, equally output-relevant but non-interacting
pairs. A finding survives only if it passes the **three-leg criterion**: (i) causal beyond first
order — the joint effect exceeds the sum of single effects, more so for binding pairs; (ii) stable
under pursuit-`k` and layer variation; (iii) above the weight-randomized null.
Falsifier: `|H_ij|` indistinguishable from the null, or equally high for arbitrary co-present pairs
(then it measures co-activation, not binding).

## Setup (fixed)
- Suite: `data/evaluations/lens-eval-order-ops.json`, all 55 items; each has two intermediates:
  a number (A) and an operation (B). Readout position = the last prompt token. Scalar `f` = the logit
  of the first token of the item's `target`, as a function of the residual `x` at (layer ℓ, position).
  **Included items** (real arm): those where the model's own top-1 at that position is that token
  (it actually computes the answer) and for which both intermediates have a single-token synonym.
  The null arm uses the same item names (a random network cannot be filtered on correctness).
- Synonyms (single-token forms, with and without a leading space): numbers → digits and English word
  (0–20, tens up to 90); multiplication → `*`, `×`, multiplication, times, multiply; addition → `+`,
  addition, plus, add, sum; subtraction → `-`, subtraction, minus, subtract; division → `/`, `÷`,
  division, divide, divided. The realised synonym of each intermediate = the one with the highest
  lens logit at (ℓ, position) in the clean pass.
- Model: Qwen3.5-4B fp32 eager (real arm); weight-randomized copy (`reinit`, seed 0) with its lens
  `results/nulls/lens_randomized_reinit_s0_L16.pt` (null arm; layer 16 only).
- Primary layer ℓ = 16, pursuit k = 25, dictionary = unit-norm J-lens vectors of the arm's whole vocabulary.
- **Direction set `S`** per item: the unit lens vectors of A and B (realised synonyms) plus **6 control
  atoms**: from the pursuit-selected atoms, excluding every synonym of A and B and the target's first
  token, the 6 with the largest first-order relevance `|⟨∇f, v_k⟩|`. (Controls are co-present and as
  output-relevant as it gets, so a pair involving them cannot be "high" merely for being relevant.)
- `H` = the restricted Hessian of `f` over `S` (`hlens.hvp.restricted_hessian`, forward-over-reverse).

## Primary statistic (Leg 0: the Hessian claim)
Per item, `P_item` = the fraction of the other 27 pairs in `S` (excluding (A,B)) whose `|H_ij|` is smaller
than `|H_AB|` (ties count 1/2). `T = mean P_item`. Test: one-sided Wilcoxon signed-rank on
`P_item − 0.5`, H1: `T > 0.5`, **p < 0.01**.

## The three legs
- **(i) causal, beyond first order.** Ablation = projecting the component along the direction(s) out of
  `x` (a pair: projection onto their span). `Δ_i = f(x ablated along i) − f(x)`, `Δ_ij` likewise,
  interaction `I_ij = Δ_ij − Δ_i − Δ_j` (forward passes only). `Q_item` = the fraction of the other 27 pairs
  whose `|I|` is smaller than `|I_AB|`. Criterion: one-sided Wilcoxon on `Q_item − 0.5`, **p < 0.01**.
  (The second-order expansion predicts `I_ij ≈ ½ (a_i a_j) · 2 H_ij`; agreement is a diagnostic below.)
- **(ii) stability.** The primary test repeated at pursuit `k = 10` and `k = 40` and at layers 14 and 18
  (four variants): each must have `T > 0.5` with one-sided **p < 0.05**.
- **(iii) above the randomized null.** `T_null` from the null arm (layer 16, k = 25): the 95%
  percentile-bootstrap CI (10,000 resamples of items in each arm, seed 0) of `T_real − T_null` must be
  **> 0**.

**H3 is supported iff Leg 0 ∧ (i) ∧ (ii) ∧ (iii).** Otherwise not supported; every number is reported.

## Secondary, exploratory (never decide)
1. Spearman correlation of `|H_ij|` with `|I_ij|` over all pairs (is the second-order picture valid at
   the intervention scale?), and the **validity-gated fraction**: pairs whose measured interaction is
   within 50% of the Taylor prediction.
2. The same statistics on `lens-eval-multihop` items with ≥ 2 intermediates (9 items) and on the pair
   (number, operation) with `S` restricted to the two.
3. Super-additivity: the fraction of items with `|Δ_AB| > |Δ_A + Δ_B|` for the binding pair vs controls.
4. Results by operation type.

## Known limits (stated in advance)
55 items (fewer after filtering) is the unit of inference; arithmetic in a 4B model is imperfect, so
the correctness filter may leave a modest sample; one layer for the null; the ablation is a single
intervention scale (the atom's own coefficient); "binding" is operationalised as co-membership in an
item's intermediates, which the dataset provides but does not prove.

## Addendum 1 (2026-10-14, before any H3 measurement existed)
The first launch of the null arm crashed while *resolving item tokens* (a `KeyError`), before any
model forward pass or measurement. The suite contains two operation keys the synonym list above
did not cover: `squared` (5 items) and `mod` (3 items). Added, as additional single-token forms:
squared → `^2`, `²`, squared, square; mod → `%`, mod, modulo, remainder. Number keys without a
single-token form in the lists (e.g. 24) are excluded, per the rule above ("both intermediates must
have a single-token synonym"). Nothing else changes; the new SHA-256 is what the arms record.

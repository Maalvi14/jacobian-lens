# Pre-registration: H7 (early-layer content in the second-order term)

Written 2026-10-14, **before any H7 measurement**. Notes §3 H7: "Where `W_U J_ℓ` is near-degenerate (Fig 28d), test
whether the second-order term restricted to J-lens directions carries above-null signal about downstream verbalizable
content." `hlens/experiments/early_curvature.py` records this file's SHA-256 in its result file.

## Setup (fixed)
- Suite: `data/evaluations/lens-eval-multihop.json`; item = (prompt, target, intermediate(s)); readout position = the last
  prompt token; scalar `f` = the logit of the target's first token at that position as a function of the residual at layer ℓ
  (position-restricted, `hlens.contexts`). **Included items:** baseline-correct (the model's own top-1 is the target's
  first token), and whose first intermediate has a single-token form (with/without leading space, `hlens.synonyms`).
- Candidates per item: the true intermediate's token plus **15 distractors** sampled (seed 0) from the pool of all other
  items' resolved intermediate tokens (excluding the true token). Resolved token of a word = the single-token form with the
  highest lens logit at (ℓ, position), as in H3.
- Directions: unit J-lens vectors `v_t = u_t J_ℓ` at layer ℓ (fp32 model, the shipped lens).
- Scores per candidate `t`: `s0(t) = ⟨v_t, x⟩` (first-order lens *presence*), `s1(t) = |⟨∇f, v_t⟩|` (first-order *sensitivity*),
  `s2(t) = |v_tᵀ H v_t|` (self-curvature; one forward-over-reverse HVP per candidate).
- `P_s` per item = fraction of the 15 distractors scoring lower than the true intermediate (ties ½); `T_s(ℓ)` = mean over items.
- Layers: early `E = {4, 8}`; reference `{12, 16}` (where the first-order lens works).

## Decision
H7 is supported iff for at least one early layer `ℓ ∈ E`, **all** hold:
1. the first-order lens fails there: `T_{s0}(ℓ) ≤ 0.55`;
2. the second-order term carries content: `T_{s2}(ℓ) ≥ 0.60` with one-sided Wilcoxon (H1: `P_{s2} > 0.5`) **p < 0.01**;
3. it beats first-order sensitivity: the 95% bootstrap CI (10,000 item resamples, seed 0) of `T_{s2}(ℓ) − T_{s1}(ℓ)` is **> 0**.
Also reported: `T_{s0}, T_{s1}, T_{s2}` at all four layers.

## Secondary, exploratory
1. Whether `s2`'s signal tracks the lens-vector geometry: Spearman between `|s2|` and the candidate's lens-vector norm / its
   cosine with the mean lens vector (a content-free shortcut if high).
2. The same at layer 16 in the weight-randomized model (the only layer with a null lens).
3. Item-level agreement of `s2` rank with `s0` rank at the reference layers.

## Known limits (stated in advance)
Roughly 40–60 included items; distractors come from a mixed pool (countries, planets, months, numbers) so category cues are not
controlled; self-curvature along one lens direction is only one slice of the second-order term; "downstream verbalizable content"
is operationalised as the identity of the dataset's intermediate.

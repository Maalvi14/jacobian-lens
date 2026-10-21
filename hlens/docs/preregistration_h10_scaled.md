# Pre-registration: H10 at scale — 100 pairs per family; is the signature a function of endpoint distance, not of persona or register?

Written 2026-10-18, **before any endpoint distance or curvature was computed for the scaled pair sets** (the pool sizes and the single-token counts were checked; nothing else).
Follows `preregistration_h10_extended.md` (persona reproduces the signature, register does not) and `preregistration_h10_control.md` (a distance-matched control of 10 object-noun pairs reproduces it as strongly as persona; persona
coefficient −0.29, 95 % CI [−0.77, 0.04]). Both rest on 10 pairs per family. `hlens/experiments/h10_scaled.py` records this file's SHA-256 in every result file.

## Question
With 100 pairs per family, (i) does the persona advantage disappear once endpoint distance is matched, (ii) does the signature strength follow endpoint distance continuously across families, and (iii) are the earlier results about "persona" and "register" a property of the families or of the pairs chosen?

## Design (fixed)
- **Model and protocol:** Qwen3.5-4B fp32; the H10 protocol (`run_trial`; layers {4, 12, 16, 20, 24}, offsets {−0.2, 0, +0.2}; original double-space construction). **Carriers: `noun_ctx_templates[12..13]`** (fresh; the earlier H10 runs used 0-5 and 6-11) → **2 trials per pair**.
  The pair is the unit of analysis, so more pairs matter more than more carriers per pair.
- **Endpoint distance** `d`: as in `preregistration_h10_control.md` (layer 24, `‖h_A − h_B‖ / mean norm`), averaged over the two carriers.
- **Families (all fixed by seed and lists in `h10_scaled.py`; pairs must be single tokens with a leading space):**
  - **persona:** 100 random distinct pairs from a pool of role / character nouns (65 single-token words of 76 listed: doctor, pirate, wizard, queen, vampire, ghost, samurai, …), each word used at most 4 times (seed 0).
  - **synonym:** the first 100 single-token pairs of a fixed list of 143 near-synonym / register-variant noun pairs (formal–informal, British–American, plain synonyms: lavatory–toilet, lorry–truck, rubbish–trash, …). This replaces the earlier "register" family, which was 10 formal–informal pairs.
  - **control:** 100 object-noun pairs matched one-to-one to the persona pairs on `d` (nearest neighbour without replacement, persona pairs in order of decreasing `d`) from a pool of 400 random pairs of 115 single-token everyday object nouns (each word at most 8 times in the pool; seed 1).
    **Adequate match iff median `|Δd|` ≤ 10 % of the median persona `d` and the maximum ≤ 25 %**; otherwise the matched tests are descriptive only.
  - **countries (reference):** all pairs of the single-token countries of `countries_12`.
  Trials: 2 per pair → 200 + 200 + 200 + about 132.

## Tests (pair-level; `D̄` = for each pair, the mean over L20 and L24 of the median over its trials of `D = λ(center) − mean λ(±0.2)`)
- **T1 (matched contrast):** two-sided Wilcoxon signed-rank on the 100 matched differences `D̄_persona − D̄_control`, **p < 0.01**. "Persona effect" additionally requires a positive median difference.
- **T2 (adjusted contrast):** least squares `D̄ ~ d + persona + synonym` on the 300 pairs of the three non-country families (control is the reference), 95 % pair bootstrap interval (2,000 resamples, seed 0) for each coefficient.
  **Persona is practically equivalent to control iff the persona interval lies within ±25 % of the median control `D̄`.** **A persona effect beyond distance iff the persona interval is above 0 and T1 passes with a positive median difference.**
- **T3 (dose-response):** Spearman correlation of `d` with `D̄` over all 300 pairs, and within each family (countries included as a reference).
- **"Endpoint distance explains the signature" iff the overall Spearman is > 0.3 and there is no persona effect beyond distance.**
- **Context (not decisive):** per-family signature criteria (a: threshold exists; b: widths shrink; c: λ larger at the threshold at L20 and L24, p < 0.01), at the trial level.

| Outcome | Statement |
|---|---|
| distance explains; persona equivalent to control | the persona/register results of H10 are a distance effect, now on 100 pairs per family; "persona" is not special in this protocol |
| distance explains; persona neither equivalent nor beyond distance | consistent with distance and with a small persona component; not resolved at this sample size |
| persona effect beyond distance | persona pairs show a larger signature than matched distinct objects; weak evidence of a persona-specific component |
| overall Spearman ≤ 0.3 | endpoint distance (as measured here) does not track the signature; the earlier dissociation is not explained by it and is not a clean persona effect either |
| match inadequate | descriptive only |

## Secondary, exploratory
`D̄` against `d` by family (binned); the synonym family split into formal–informal vs British–American pairs; the same analysis with `d` at layer 16; the fraction of pairs with `D̄ > 0` by family.

## Known limits (stated in advance)
Qwen3.5-4B only (H6 is absent on Gemma); two carriers per pair, so `D̄` is noisy per pair (averaged over pairs); random pairs from word pools are not validated categories (the persona pool includes mythical creatures, the synonym list mixes register, dialect and plain synonyms);
the distance is one measure at one layer; words in a family are reused across pairs, so pairs are not fully independent; "control" matches on distance but not on frequency or concreteness; about 6 hours of compute.

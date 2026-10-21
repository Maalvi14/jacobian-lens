# Pre-registration: H10 control — is the persona advantage a persona effect or a distinct-concepts effect?

Written 2026-10-18, **before any measurement for this control** (no endpoint distance, no curvature). Follows `preregistration_h10_extended.md`, whose result was: persona pairs reproduce the H6 ignition
signature and register pairs do not (contrast p = 5e-10 / 2e-15), with the likely confound that persona pairs are two *distinct* concepts while register pairs are near-synonyms.
`hlens/experiments/h10_control.py` records this file's SHA-256 in every result file.

## Question
Do distinct, non-persona object nouns, matched to the persona pairs on how far apart the two endpoints are, show the signature as strongly as persona pairs do?

## Design (fixed)
- **Model and protocol:** Qwen3.5-4B fp32; the H10 protocol (`hlens.experiments.ignition.run_trial`; layers {4, 12, 16, 20, 24}, offsets {−0.2, 0, +0.2}; original double-space construction), carriers `noun_ctx_templates[6..11]` (the carriers of the extended H10) → 6 trials per pair.
- **Endpoint distance (the matching variable, computed before any curvature):** for a pair (A, B) and a carrier, `d = ‖h_A − h_B‖ / (½(‖h_A‖ + ‖h_B‖))`, with h the layer-24 residual at the `{W}` position for the pure-A and pure-B inputs (α = 1 and α = 0); layer 24 is where the commitment axis of the
  deepest curvature layer (L20 + gap 4) is defined. The pair's `d` is the mean over the six carriers. Computed for the 10 extended persona pairs, the 10 extended register pairs and every candidate control pair.
- **Candidate control pool (fixed here, 32 pairs of unrelated everyday objects; no roles, people, characters or animals):**
  anchor–pencil, river–sandwich, mountain–violin, candle–hammer, ladder–pizza, window–banana, bottle–saddle, blanket–trumpet, lantern–carrot, umbrella–helmet, bridge–cookie, pillow–wrench, basket–rocket, mirror–pumpkin,
  kettle–compass, garden–telescope, button–volcano, spoon–castle, feather–engine, clock–melon, shovel–piano, rope–lemon, bucket–guitar, needle–tractor, sofa–cherry, tent–microscope, stamp–trousers, magnet–onion, drum–sock, truck–lamp,
  tower–biscuit, anvil–carpet. Pairs failing the single-token test (leading space) are dropped.
- **Matching (deterministic):** persona pairs in order of decreasing `d`; each takes the unmatched candidate with the closest `d`. The first 10 matched candidates are the control family. Achieved quality is reported: median and maximum `|d_persona − d_control|`.
  **The match is adequate iff the median `|Δd|` ≤ 10 % of the median persona `d` and the maximum ≤ 25 %.** If it is not, the control is reported as "not adequately matched" and its test results are descriptive only.
- **Trials:** 10 control pairs × 6 carriers = 60.

## Tests
- **Per-family signature criteria** (a, b, c) exactly as `preregistration_h10.md`, for the control family.
- **Primary contrast, at the pair level** (the unit is the pair, not the trial, which removes the trial-dependence of the earlier tests): per pair, `D̄ = mean over L20 and L24 of the median over the pair's 6 trials of D` (`D = λ(center) − mean λ(±0.2)`).
  One-sided Mann–Whitney on `D̄`, persona > control, 10 vs 10 pairs, **p < 0.01**. (The same for control > register is reported, not decisive.)
- **Distance covariate (pair level, all 30 pairs: persona, register, control):** Spearman correlation of `d` with `D̄`; and the least-squares coefficient of the persona indicator in `D̄ ~ d + persona` fitted on the 20 persona and control pairs, with a 95 % pair bootstrap interval (2,000 resamples, seed 0).

| Outcome | Statement |
|---|---|
| matched control reproduces the signature and the persona-vs-control contrast is not significant | the persona/register dissociation is explained by endpoint distance / distinct concepts; "persona" is not special in this protocol |
| matched control reproduces the signature but persona is significantly larger (p < 0.01) | distinct concepts show the signature, and persona pairs show it more strongly even at matched distance; weak evidence of a persona-specific component, limited to these pairs |
| matched control does not reproduce the signature and persona is larger | the signature is specific to role/persona nouns among distinct concepts (the original reading), within the limits below |
| control does not reproduce the signature and persona is not larger | neither family is distinguishable from the control; the extended-H10 persona result is not reproduced under this contrast |
| match inadequate | descriptive only; the pool or the distance definition is the limit |

## Secondary, exploratory
Per-pair `D̄` and `d` listed; the same contrast with `d` measured at layer 16 instead of 24; the Spearman correlation of `d` with `D̄` within each family.

## Known limits (stated in advance)
Ten pairs per family and a pool of 32 candidates (the matching is only as good as the pool); "role / persona" versus "object" is my categorisation; the distance is one particular measure at one layer and a different measure could rank pairs differently; the control words differ from the persona words in
frequency, concreteness and part of speech mix in ways not matched; Qwen3.5-4B only (H6 is absent on Gemma); a match on distance does not match on everything else.

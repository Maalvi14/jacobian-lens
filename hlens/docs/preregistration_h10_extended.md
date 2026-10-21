# Pre-registration: H10 extended — does the persona/register dissociation replicate with more, fresh pairs?

Written 2026-10-17, **before any measurement of the extended H10**. `hlens/experiments/h10_extended.py` records this file's SHA-256 in its result file.

## Why this, and why Qwen3.5-4B only
The original H10 proxy (`preregistration_h10.md`) found that persona noun pairs reproduce the H6 ignition signature (a threshold exists, widths shrink with depth, the leading Hessian eigenvalue is larger at the
threshold than 0.2 away at L20 and L24) while register pairs do not (L20 p = 0.73, L24 p = 0.054). That rests on **4 pairs per family** (24 trials). This file repeats it with **10 fresh pairs per family and fresh carriers**.
Gemma 3 12B is not used: H6 itself is not supported there (`preregistration_h6_h8_gemma.md`: T2 p = 0.90, no curvature peak at the threshold), so the reference signature the H10 criteria are defined against is absent in
that model, and a Gemma run could only reproduce "no signature in any family".

## Design (fixed)
- Model, protocol, layers `{4, 12, 16, 20, 24}`, offsets `{−0.2, 0, +0.2}`, readouts and statistics: exactly `preregistration_h10.md` (`hlens.experiments.ignition.run_trial`; Qwen3.5-4B fp32; original double-space construction, as in the original H10).
- **Pairs (fresh; none overlaps the original eight):** the first 10 single-token pairs of each list, in this order.
  register (formal vs informal noun): attorney–lawyer, infant–baby, footwear–shoes, employment–job, spectacles–glasses, gentleman–guy, youngster–kid, telephone–phone, currency–money, canine–dog
  (spares if one fails the single-token test: stomach–belly, purchase–buy);
  persona (role nouns): judge–clown, monk–sailor, queen–witch, chef–knight, scientist–ghost, princess–vampire, mayor–dragon, banker–hunter, surgeon–outlaw, professor–giant (spares: banker–outlaw, mayor–witch).
- **Carriers (fresh):** `noun_ctx_templates[6..11]` of `data/experiments/ignition.json` → **60 trials per family** (the original carriers 0-5 are not reused).
- Reference: the Qwen3.5-4B H6 real-arm country trials (`results/h6/real.json`), restricted to the same layers and offsets.

## Tests
- **Per family, the original criteria (all three for the family to "reproduce the signature"):** (a) a share-0.5 threshold exists at layer 16 for ≥ 80 % of trials; (b) `width(24) < width(8)` in ≥ 80 % of trials where both exist;
  (c) λ larger at the threshold than 0.2 away at **both L20 and L24** (one-sided Wilcoxon, p < 0.01 each).
- **Contrast (new):** at L20 and L24, one-sided Mann–Whitney on `D = λ(center) − mean λ(flank)`, persona > register, p < 0.01 at **both**.
- **The dissociation replicates iff** persona reproduces the signature ∧ register does not ∧ the contrast holds at L20 and L24.
- **The original H10 proxy statement** ("supported iff both families reproduce the signature") is also evaluated and reported as before.

| Outcome | Statement |
|---|---|
| persona reproduces, register does not, contrast significant | the persona/register dissociation replicates on fresh pairs and carriers |
| both families reproduce | the original register result (p = 0.054 at L24) was a small-sample miss; the proxy is supported (both families show the signature) |
| neither reproduces | the original persona result did not replicate; reported as such |
| persona fails, register reproduces | the reverse of the original; reported as such |
| persona and register both reproduce or both fail but the contrast is significant | reported as inconsistent with the dissociation rule; the contrast p-values are shown |

## Secondary, exploratory
Per-layer fractions of trials with `λ(center) > λ(flank)` for all five layers and both families; median widths by layer; Mann–Whitney of `D` against the country reference per layer; per-pair results (does any one pair drive a family).

## Known limits (stated in advance)
Noun pairs only; "register" and "persona" are operationalised as word pairs, not prompts or system messages; lists are my choices (formal-vs-informal and role nouns), not validated categories; 60 trials per family but only 10 distinct pairs
(trials sharing a pair are not independent, so the Wilcoxon p-values are optimistic); a single model; no randomized-model arm (the H6 null already shows the signature absent without learned structure).

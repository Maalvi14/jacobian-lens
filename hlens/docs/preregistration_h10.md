# Pre-registration: H10 (register and persona transitions; exploratory proxy)

Written 2026-10-14, **before any H10 measurement**. Notes §3 H10: the felt-vs-mechanistic register and the
default-vs-roleplay flagging "are candidates for the same ambiguous-interpolation protocol as H6, with activation directions
or persona pairs replacing concept pairs." No persona data ships with the repo and Qwen3.5-4B is not the paper's model, so this
is an **exploratory proxy**: the H6 protocol on (i) register-contrast noun pairs and (ii) persona noun pairs, with the country
pairs of H6 as the reference family. It tests whether the ignition signature found for concept pairs generalises to
register/persona contrasts, not any claim about the paper's assistant persona. `hlens/experiments/register_persona.py` records
this file's SHA-256.

## Setup (fixed)
- Families (4 pairs each, single-token with a leading space, seed-0 order):
  register = (automobile, car), (residence, house), (physician, doctor), (attire, clothes) [formal vs informal noun];
  persona = (doctor, pirate), (teacher, soldier), (king, beggar), (wizard, farmer) [role nouns].
  Words failing the single-token test are replaced by the next candidates in the lists: register — (beverage, drink), (feline, cat);
  persona — (nurse, pilot), (robot, poet).
- Carriers: `noun_ctx_templates[0..5]` of `data/experiments/ignition.json` → 24 trials per family.
- Protocol, readouts and statistics exactly as H6 (`hlens.experiments.ignition.run_trial`): projection share, threshold at layer 16,
  Lanczos leading |eigenvalue| `λ` of the commitment map, gap 4, **layers {4, 12, 16, 20, 24}, offsets {−0.2, 0, +0.2}**.
- Reference: the H6 real-arm country trials (`results/h6/real.json`), restricted to the same layers/offsets.

## Decision (per family, all three must hold for the family to "reproduce the signature")
(a) a share-0.5 threshold exists at layer 16 for ≥ 80% of trials;
(b) widths shrink with depth: `width(24) < width(8)` in ≥ 80% of trials where both exist (width at layer 8 is computed for the
    share curve, not the curvature layers);
(c) `λ` is larger at the threshold than 0.2 away at **both L20 and L24**: one-sided Wilcoxon `p < 0.01` at each.
**H10 (proxy) is supported iff both the register and the persona family reproduce the signature.**

## Secondary, exploratory
Per-layer fraction of trials with `λ(center) > λ(flank)`, for all five layers and all three families; median widths by layer;
Mann–Whitney comparisons of `D = λ(center) − mean λ(flank)` between families per layer; the validity score at the threshold
(`path_validity`) on 4 trials per family.

## Known limits (stated in advance)
Noun pairs only; 24 trials per family; the carriers are noun-slot templates; "register" and "persona" contrasts are
operationalised as word pairs, not as prompts or system messages; no null arm (the H6 null already shows the signature absent in a
randomized network).

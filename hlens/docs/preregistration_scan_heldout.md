# Pre-registration: claim-A scan on the 120 held-out trials (Qwen3.5-4B, J-lens directions)

Written 2026-10-20, **before the run**. The exploratory scan of claim A over layer pairs and direction sources (`preregistration_exploratory_a_h10.md`) used 24 trials per configuration. A review of the draft paper asked for the scan on more trials. This repeats the scan on the **120 held-out trials** (the set HO of `h9_replication.trial_sets`: templates 6–35 × pairs 4–7, disjoint from the original 24) for the **nine J-lens configurations** of Qwen3.5-4B (source layers 8, 12, 16 × gaps +4, +8, +12; threshold layer 16). Identity-direction configurations are not repeated (cost).

## What is fixed
- Everything of the original scan except the trial set: fp32, eager attention, the same direction construction and per-configuration protocol (`hlens/experiments/h9_scan.py --set HO --directions lens`), results in `results/h9_scan_ho/`.
- **Statistics per configuration:** the three precondition tests of claim A (inhibition at the threshold above 0, above its flanks, above a random orthogonal pair), each at p < 0.01 / 9 (Bonferroni over the 9 configurations); the fraction of trials with both cross-terms negative.
- **Prediction (stated in advance, from the 24-trial scan):** the six configurations that passed on 24 trials (8→12, 8→16, 8→20, 12→16, 12→20, 16→20) pass again; the three with a target layer near the output (12→24, 16→24, 16→28) do not.

## Outcome table
| Result | Statement |
|---|---|
| the six pass, the three fail | the 24-trial scan replicates: the effect needs a target layer well below the output |
| some of the six fail | the 24-trial result was partly sampling noise; the paper reports the count |
| any of the three passes | the layer-pair boundary of the 24-trial scan was noise; reported |

## Known limits
One model, one threshold layer, templated stimuli, trials share templates and country words (p-values optimistic); identity directions not repeated.

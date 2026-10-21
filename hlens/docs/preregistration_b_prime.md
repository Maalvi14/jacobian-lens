# Pre-registration: B′ — curvature along J-lens directions exceeds random directions, with an equivalence-tested null

Written 2026-10-15, **before any B′ measurement**. The experiments record this file's SHA-256 in every result file; a real arm refuses to
run before a null arm under the same hash (`hlens/experiments/h2_replication.py --config B2`).

## Why this file exists
B (H2) was pre-registered in `preregistration_replication.md` and **failed its own rules**: T2 failed on Qwen3.5-4B (median ratio 1.37 < 1.5), and T3
("the null ratio's 95 % interval contains 1") failed in 7 of 8 null arms, because the bootstrap intervals are about ±0.0005 wide, so an interval
around 1.001 excludes 1. That result stands and is reported as a failure of B. **B′ is a new, narrower hypothesis, not a rescue of B**: it replaces
the exact-equality null rule with an equivalence rule and drops the 1.5× size threshold from the pass/fail rule. Because the existing data informed
this design, **B′ is tested only on data not used so far** (new prompts, new weight-randomization seeds).

## What the old data already show (so the reader can judge the design)
Existing runs: null ratios 0.997-1.002 (8 arms); real ratios 1.11-2.13 on Qwen (L16-30) and 4.1-11.3 on Gemma (L24-44). The choices below were
made knowing this. Their purpose is that the margin and the data are fixed *before* the new runs, and that a failure on new data is possible.

## Hypothesis B′
For a model with a lens fitted on it, in layers with depth ≥ 0.5, the median over fixed context rows of
κ(J-lens directions)/κ(random directions) is above what a weight-randomized model gives, and the weight-randomized model gives a ratio equivalent to 1.
κ is as defined in `hlens/experiments/gain.py` (norm of ½ vᵀHv of the MLP branch, own norms included, at layer l+1).

## Fixed design
- **Fresh data.** Evaluation prompts: wikitext prompts with indices 200-229 (`load_wikitext_prompts(230)[200:]`). Lens-fitting prompts for the null models:
  indices 300-329. (Earlier work used 0-29 for evaluation and 100-129 for fitting; the four ranges are disjoint.)
- **Everything else as in `preregistration_replication.md`**: 16 random valid positions per prompt, 64 directions per population, prompt-cluster bootstrap
  (300 resamples), fp32, depth-matched layers, the null built by `reinit` randomization, bf16 rounding, and a lens fit with `dim_batch` 32 / 64 tokens.
- **Equivalence margin δ = 0.05.** Chosen because (i) it is ten times smaller than the smallest real effect the project cares about (1.5×, a 50 % increase),
  and (ii) it is large enough that estimator noise at these sample sizes is far below it. It is not derived from the observed null values.
- **Null arms (new seeds).**
  - Qwen3.5-4B: seeds 10, 11, 12 at layer 16; seed 10 at layers 8 and 24 (one fit).
  - Gemma 3 12B: seed 10 at layer 24 (required); seed 11 at layer 24 (reported if time allows, not required).
- **Real arms (fresh prompts).**
  - Qwen3.5-4B: layers {8, 16, 18, 20, 22, 24, 26, 28, 30}.
  - Gemma 3 12B: layers {4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44}.
  - **Depth range for the tests:** layers with depth ≥ 0.5 (Qwen: 16-30; Gemma: 24-44).

## Tests
- **N (null equivalence):** for every null arm of the model, the whole 95 % interval of the J-lens/random κ ratio lies inside **[0.95, 1.05]**.
- **R (real exceeds the equivalence region):** the real arm's 95 % interval **lower bound > 1.05** in at least **75 %** of the measured layers in the depth range.
- **C (direct contrast at null layers):** at each null-lens layer in the depth range (Qwen L16, L24; Gemma L24), the real arm's lower bound is greater than the
  upper bound of every null arm at that layer.
- **B′ holds on a model iff N ∧ R ∧ C on that model.** B′ is stated as architecture-general only if it holds on both models.

## Reported but not tests
- The median ratio over the depth range, and whether it reaches 1.5 (the label "large effect" is used only if it does; Qwen is expected not to on the old numbers).
- Spearman correlation between layer and ratio within the depth range (does the effect grow with depth).
- The neuron-direction population and the gain (g1) ratios.
- Layer L8 on Qwen (depth 0.26, outside the depth range): the same N test, reported as exploratory.

## Decision table
| Outcome | Statement in the write-up |
|---|---|
| N ∧ R ∧ C on both models | B′ holds on Qwen3.5-4B and Gemma 3 12B; B itself still reported as having failed its pre-registered rules |
| holds on one model only | B′ is a single-model finding |
| N fails on any arm | the randomized control is not flat; the J-lens-direction effect cannot be attributed to learned structure on that model |
| R fails | the effect is not distinguishable from the null region on fresh data; the earlier ratios are reported as not reproduced |
| C fails | the real arm is not separable from the nulls at the matched layers |

## Known limits (stated in advance)
Two models; one layer (Gemma) or three (Qwen) of null evidence; seeds 3 (Qwen) and 1-2 (Gemma); a small lens-fitting recipe for the null (30 prompts);
the margin is a judgment, not an estimate; the claim concerns MLP-branch curvature along lens directions and says nothing about whether the directions are
used by the model.

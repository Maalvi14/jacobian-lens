# Pre-registration: replication of H2 (claim B) and the H9 precondition (claim A) on a second architecture

Written 2026-10-15, **before any Gemma 3 12B measurement and before the held-out Qwen3.5-4B run**. Implements R1, R2 and the
gates of `writeup_plan.md`. The experiments record this file's SHA-256 in every result file; real arms refuse to run before the
matching null arm under the same hash (H2), as in H3-H10.

## Models and depth-matched layers
Layers are defined by depth fraction of the Qwen3.5-4B values (32 layers, `depth = layer/(n_layers-1)`) and rounded:

| role | Qwen3.5-4B | Gemma 3 12B (48 layers, lens L0-46) | Qwen3.6-27B (64 layers, lens L0-62; **only if run**) |
|---|---|---|---|
| source ℓ (H9, swap onset) | 12 | 18 | 24 |
| target m (H9) | 20 | 30 | 41 |
| middle / null-lens layer | 16 | 24 | 32 |

- Gemma 3 12B: `google/gemma-3-12b-pt`, Neuronpedia lens `gemma-3-12b-pt` (775 prompts), fp32 for all curvature work, bf16 for the
  gates (the lens was fitted in bf16). Eager attention.
- Qwen3.6-27B fp32 does not fit in memory; it is not part of this pre-registration (a separate one is needed for a mixed-precision design).

## Gates (replication of the paper's checks; report only, not tests of A/B)
- **G1 landmarks:** Gemma, 30 wikitext prompts, bf16. Pass if: top-1 lens accuracy at the last lens layer > 5 × that at depth 0.39
  (layer 18); mean autocorrelation of layers 0-2 < 0.05; max autocorrelation > 0.5. (Same thresholds as the Qwen test.)
- **G2 swap:** `probe-swap.json` items re-resolved with the Gemma tokenizer (single-token, non-ambiguous, baseline-correct), band L18-30,
  α = 1.25, raw lens vectors, one editor per layer. Pass if hit rate ≥ 25 % and ≥ 5 × max(1, hits of 3 random-direction controls).
  The band and α are fixed here (mapped from Qwen); a sweep around them is exploratory only.

## H2 replication (claim B): is curvature along J-lens directions above random, with a flat randomized-weight control?
Protocol = `hlens/experiments/gain.py` (`run_h2`): 30 wikitext prompts, 16 random valid positions each, 64 directions per population,
cluster (prompt) bootstrap, κ = ‖½ vᵀHv‖ of the MLP branch (own norms included), at layer l+1 for lens source layer l.
- **Gemma real arm:** layers {4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44}.
- **Depth range for the tests:** layers with depth ≥ 0.5 (Gemma: 24-44; the Qwen original is L16-30).
- **T1:** the J-lens/random κ ratio has a 95 % CI lower bound > 1 at ≥ 75 % of the measured layers in the depth range.
- **T2:** the median over the depth range of the J-lens/random κ ratio ≥ 1.5.
- **T3 (null):** in the weight-randomized model (`reinit`) with a lens fitted on it, the ratio's 95 % CI contains 1 at every
  null-lens layer and seed.
- **B holds on a model iff T1 ∧ T2 ∧ T3 there.** If T3 fails anywhere, B is withdrawn for that model and the original Qwen finding is flagged as
  possibly an artifact of the (single, seed-0, layer-16) null.
- **Qwen3.5-4B null extension (R1):** seeds 1-4 at layer 16, and seed 0 at layers 8 and 24 (lens fitted on the randomized model for each).
- **Gemma null:** seed 0 at layer 24; seed 1 at layer 24 if time allows (reported if run, not required).
- The lens fitted on a randomized model uses the same recipe as `build_nulls.py` (30 prompts disjoint from the evaluation prompts, 64 tokens).
- Exploratory: the neuron-direction population; g1 ratios (gain); whether the ratio rises with depth.

## H9 precondition replication (claim A)
Protocol = `hlens/experiments/competition.py` (`run`, `analyze_h9`) with the layers above, run at ℓ, m, mid per model.
- **Original set (O):** `choose_trials` as defined for H6/H9 (6 carrier templates `ctx_templates[:6]` × the first 4 pairs of the seed-0
  shuffled single-token country pairs). On Qwen this is the existing result; on Gemma it is a direct replication.
- **Held-out set (HO):** carrier templates `ctx_templates[6:36]` × the pairs at shuffled positions 4-7 (not used in O) → up to 120 trials
  after tokenizer filtering. **The same HO set is run on Qwen3.5-4B** (new stimuli, same model) and on Gemma.
- Thresholds are not re-tuned: α* is the projection-share 0.5 crossing at the mid layer, as in H9; points α* + {−0.2, 0, +0.2}.
- **Tests on the HO set (each model):** P0a, P0b, P0c exactly as `preregistration_h9.md` (one-sided Wilcoxon p < 0.01).
  **A holds on a model iff P0a ∧ P0b ∧ P0c on HO.** Fraction of trials with both cross-terms negative is reported (Qwen O: 24/24).
- **Validity-gate sensitivity (pre-registered, secondary):** `path_validity` at α* (ℓ → m); the three tests are repeated on trials whose
  score ≥ 1. If A holds on all trials but fails on the valid ones, A is reported as "holds only where the local quadratic picture is invalid".
- **Not run here:** H9 part 2 (list loading), the randomized-weight arm for H9 (as for Qwen the control is a random orthogonal pair).
- **Decision rule for generality:** A and B are stated as architecture-general only if they hold on both Qwen3.5-4B (HO / null extension)
  and Gemma. If they hold on Qwen only, they are Qwen-family findings; if they fail on Qwen HO / null extension, they are withdrawn.

## Known limits (stated in advance)
Two models; country pairs only for H9; fp32 only for Gemma; the Gemma `-pt` model and lens are used (no instruction-tuned arm); the lens-fitting
recipe on randomized models is small (30 prompts) and the same for both models; no correction for the number of layers beyond the ≥ 75 % rule.

## Addendum 1 (2026-10-15; written after the Qwen3.5-4B held-out H9 run and a crash of the first Gemma run, before any Gemma H9 data)
- **Stimulus spacing.** The original `make_setup` writes `prefix(with its trailing space) + " " + word`, i.e. a double space before the slot.
  Qwen tokenizes this as `[' ', ' Word']`; Gemma merges it into `['  ', 'Word']`, so the slot token is not `' Word'` and the run aborted
  (`could not locate the {W} slot`; no data produced). A `clean=True` option removes the double space. **Gemma is run with `clean=True`.**
  To check that the spacing is not what drives the result, Qwen3.5-4B O and HO are re-run with `clean=True` as well; Qwen's `clean=False`
  HO result (already in hand: all three tests pass, 117/120 both-negative, 15/120 valid) stays reported. The decision rule for claim A on
  Qwen uses the `clean=True` HO run (same construction as Gemma); the `clean=False` run is reported beside it.
- **Mid-layer for Qwen3.6-27B** is 33 (not 32) by `matched_layer`; unused so far.
- Gate results (G1 passed; G2 not passed at the pre-registered α = 1.25: 8/48 vs controls 1/0/0) were obtained before this addendum and are unchanged.

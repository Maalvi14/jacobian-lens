# Pre-registration: claim A (H9 precondition) on Qwen3.6-27B, mixed precision

Written 2026-10-15, **before any Qwen3.6-27B measurement**. `hlens/experiments/h9_replication.py --model qwen3.6-27b` records this file's SHA-256 in its result file.

## Why this test
Claim A (inhibition near the ignition threshold is a second-order effect that peaks at the threshold) held on Qwen3.5-4B (120 held-out trials, both spacings) and
**failed on Gemma 3 12B** (no peak at the threshold; 30 % both-negative). It is therefore a candidate Qwen-family finding. Qwen3.6-27B is the same family at a larger
size: if A holds, it is a Qwen-family finding across sizes; if it fails, A is specific to the 4B and the family reading is withdrawn.

## What is being run
- **Model and lens:** `Qwen/Qwen3.6-27B` (64 layers, d = 5120; 48 linear + 16 full attention; untied head) with the Neuronpedia lens `qwen3.6-27b/.../Qwen3.6-27B_jacobian_lens_n1000.pt`
  (the n = 1000 fit is the only one available; every statement about this model says so). The lens was fitted in bf16; it is used as is, no new lens is fitted.
- **Precision:** fp32 weights do not fit (≈ 108 GB). The model is held in bf16; the slice blocks (layers 25-41) and the unembedding directions' final norm and head are fp32 copies
  (`make_mixed_slice_fn`, `Fp32Unembed`). This path was validated for **this measurement only** on Qwen3.5-4B against fp32 (`preregistration_mixed_precision.md`, all five criteria passed;
  median relative error of the inhibition 0.34 %). A dry run measured 53.8 GB after loading and 84.8 GB (peak 87.3 GB) with the mixed slice built.
- **Layers** (depth-matched by `matched_layer`): source ℓ = 24, target m = 41, middle / threshold layer 33. (The table in `preregistration_replication.md` lists 32 for the middle layer; the
  code's rounding gives 33, and 33 is used.)
- **Stimuli:** as in the replication pre-registration, with the clean one-space construction. **Primary set: the held-out set HO** (30 carrier templates `ctx_templates[6:36]` × the pairs at shuffled
  positions 4-7 = 120 trials; all 120 are single-token under this tokenizer). Secondary: the original set O (24 trials). Both are run; HO decides.
- **Measurement:** unchanged from `competition.py`: α* = projection-share 0.5 crossing at the middle layer (from the bf16 forward passes; the mixed-precision check found this within
  0.05 of fp32 in 24/24 trials), points α* + {−0.2, 0, +0.2}, cross-terms c_A and c_B of the restricted Hessian of the two readouts over the two lens directions, inhibition
  I = −(c_A + c_B)/2, and the random orthogonal control pair.

## Tests (on the HO set)
Exactly P0a, P0b, P0c of `preregistration_h9.md` (one-sided Wilcoxon, p < 0.01 each):
- **P0a:** I at the threshold > 0.
- **P0b:** I at the threshold exceeds the mean of the two flanks.
- **P0c:** I at the threshold exceeds the control pair's.
**A holds on the 27B iff P0a ∧ P0b ∧ P0c on HO.** Also reported: the fraction of trials with both cross-terms negative at the threshold (Qwen3.5-4B: 24/24 on O, 88-97.5 % on HO),
the first-order cross-effect, the same tests on O, and the same tests on the trials whose Taylor-validity score is ≥ 1 (validity is computed with the same mixed slice; if fewer than 8
trials are valid, the sensitivity analysis is reported as not testable).

## Decision table
| Outcome on HO | Statement |
|---|---|
| holds on 27B (and on 4B) | A holds across Qwen sizes (4B, 27B) and fails on Gemma 3 12B: a Qwen-family finding, not architecture-general |
| fails on 27B | A is specific to Qwen3.5-4B; it is not even a family-level finding |
| holds on HO but not on O (or the reverse) | reported as inconsistent; HO decides, O is a caution |
| a run fails to complete | reported as not tested (no partial result is used) |

## Procedure and resources
- One large job at a time. Before launching: drop the page cache (`sudo sysctl vm.drop_caches=3`) and confirm `MemAvailable` ≥ 100 GB. A kernel out-of-memory kill leaves no traceback; check `dmesg | grep -i "killed process"`.
- Order: HO first, then O. The expected cost is hours (the 4B mixed run took 223 s for 24 trials with an 8-block slice; this slice has 17 blocks of twice the width).

## Not claimed / known limits (stated in advance)
- The mixed-precision path is validated on the 4B for this quantity only. The 27B slice has 17 blocks (the 4B check had 8), so more rounding error may accumulate, and the 4B check cannot show
  that the 27B behaves the same. No hypothesis test on the 27B is run in full fp32.
- The n = 1000 lens is a different fit from the n = 417 lens used on the 4B. On the one prompt tried in the dry run the lens reads blanks at the middle layer (33) and "Currency" at layer 50:
  the lens-direction bank is not checked for quality beyond that.
- Country pairs only; absolute magnitudes are not compared across models (Gemma's were ~1000× smaller than Qwen's).
- Claim B′ is not tested on the 27B here; that needs a weight-randomized null model with a new lens fit and its own pre-registration.

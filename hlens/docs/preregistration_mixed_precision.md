# Pre-registration: does a mixed-precision path reproduce fp32 curvature results? (validation on Qwen3.5-4B)

Written 2026-10-15, **before any mixed-precision measurement**. Purpose: Qwen3.6-27B does not fit in fp32 (~108 GB), so a candidate
path keeps the whole model in bf16 and casts only what a curvature measurement touches to fp32. This file fixes, in advance, what counts as
"reproduces fp32". The check runs on Qwen3.5-4B, where a full-fp32 reference exists. `mixed_precision_check.py` records this file's SHA-256.

## The mixed path (what is being validated)
- The model is loaded in bf16. Activations entering a measurement come from the **bf16 forward pass**, then are cast to fp32.
- `hlens.functional.make_mixed_slice_fn(model, l_start, l_end)`: fp32 deep copies of blocks l_start+1 … l_end; block kwargs captured from the bf16
  model and cast to fp32, with the rotary cos/sin **recomputed in fp32** when the text model exposes `rotary_emb`.
- `hlens.functional.Fp32Unembed(model)`: fp32 copies of the final norm and lm_head, used to build the unembedding directions and nothing else.
- Everything downstream (restricted Hessian, `jvp`) is unchanged and runs in fp32.

## Reference
The existing full-fp32 run of the H9 precondition on the 24 original trials with the clean spacing:
`results/h9_replication/O_qwen3.5-4b_clean.json` (src 12, target 20, mid 16; offsets −0.2, 0, +0.2 around each trial's threshold α*).
The mixed arm evaluates the **same α values** (the reference α* ± offsets), so only precision differs in what is compared.

## Acceptance criteria (all must hold; 24 trials × 3 offsets = 72 cells)
- **V1 sign agreement:** the sign of the cross-term `c_A` and of `c_B` agrees with fp32 in ≥ 95 % of cells (for each of the two).
- **V2 rank agreement:** Spearman correlation of the inhibition `I` over the 72 cells, mixed vs fp32, ≥ 0.95.
- **V3 magnitude:** median of |I_mixed − I_fp32| / |I_fp32| over the cells ≤ 0.10, and its 90th percentile ≤ 0.30.
- **V4 verdicts:** the three precondition tests (P0a, P0b, P0c) of `preregistration_h9.md`, computed on the mixed cells, give the same pass/fail as fp32 (all pass).
- **V5 threshold:** the mixed pipeline's own α* (projection-share crossing from bf16 forward passes) is within ±0.05 of the fp32 α* in ≥ 90 % of trials.

## Decision
- All five hold → the mixed path may be used for the **claim-A measurement** (inhibition at the ignition threshold) on a larger model. It is **not** thereby
  validated for the H2 curvature measurement, which needs its own check (different quantity, MLP-branch jvp).
- Any criterion fails → the mixed path is not used; the failing criterion is reported, and the 27B is not run for claim A through this path.
- Whatever happens, the numbers are reported with the 4B comparison next to any 27B result.

## Known limits (stated in advance)
One model (a smaller one than the target), one layer pair, 24 trials, the same stimuli that produced the reference; the 27B has 4× more blocks in its slice
(more accumulated error) and a different width; the test cannot show the 27B behaves like the 4B.

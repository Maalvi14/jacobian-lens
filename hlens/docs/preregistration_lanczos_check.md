# Pre-registration: Lanczos convergence check for the H6 leading-eigenvalue estimates

Written 2026-10-19, **before the check was run**. A robustness check on an existing result, not a new hypothesis: `preregistration_h6.md` used the leading |eigenvalue| λ of the commitment map's Hessian from **8 Lanczos iterations** ("accurate to a few percent at best,
checked on a toy operator, not on the model" - its stated limit). This file fixes how that is now checked on the model. `hlens/experiments/lanczos_check.py` records this file's SHA-256.

## Design (fixed)
- Qwen3.5-4B fp32, the Qwen H6 real-arm results (`results/h6/real.json`): trial indices {0, 3, 6, 9, 12, 15, 18, 21} of the 24 trials (8 trials); layers {4, 8, 10, 12, 14, 16, 20, 24}; gap 4; offsets {−0.2, 0, +0.2} around each trial's stored threshold α*; i.e. 192 cells.
- Per cell: one Lanczos run of **40 iterations** (full re-orthogonalisation, forward-over-reverse HVP, the same commitment function as H6), with the eigenvalue read off after 8, 16, 24 and 40 steps; start vector seed 0 (the H6 seed - at 8 steps this must reproduce the stored H6 value) and seed 1.
  The converged reference is the 40-step estimate of seed 0.

## Criteria (all descriptive; none changes the H6 verdict)
- **C1 (the 8-step estimates are adequate):** the median of |λ₈ − λ₄₀| / λ₄₀ over the cells ≤ 5 % and its 90th percentile ≤ 20 %.
- **C2 (the H6 decision statistic is stable):** the sign of `D = λ(center) − mean λ(±0.2)` per (trial, layer) agrees between λ₈ and λ₄₀ in ≥ 90 % of the (trial, layer) pairs.
- **Reproduction check:** the stored H6 value equals the seed-0 8-step value (relative difference ≤ 1e-3) in all cells that have a stored value; if not, the check is not trusted.
- **Reported, not decisive:** the layer profile of the median λ₈, λ₄₀ and λ₄₀ × ‖x‖ (the Hessian eigenvalue scaled by the residual norm at the perturbed position, which puts layers with very different residual scales on one footing); the mean D and its one-sided Wilcoxon p for λ₈, λ₄₀ and λ₄₀ × ‖x‖;
  the 16-step error and the difference between the two start vectors at 8 steps.

| Outcome | Statement |
|---|---|
| C1 and C2 hold | the 8-iteration estimates used for H6 are adequate on this subset; the H6 results are not an artifact of Lanczos truncation |
| C1 fails, C2 holds | individual λ values are off by more than a few percent but the decision statistic is stable |
| C2 fails | the H6 decision statistic is not robust to Lanczos truncation; the Qwen H6 tests (T1, T2) are not trusted until recomputed with the converged values |
| the reproduction check fails | the check is discarded and the code is debugged |
| the layer profile of λ₄₀ × ‖x‖ differs from that of λ₄₀ | reported as a scale effect: the "L4 peak" of H6 is partly a residual-scale effect (descriptive only) |

## Known limits (stated in advance)
8 of 24 trials; 40 iterations may itself be short of convergence for clustered top eigenvalues (the 16/24/40 estimates are reported so the trend is visible); Qwen3.5-4B only (the Gemma H6 was not supported and is not rechecked); λ is the leading |eigenvalue| of one scalar map along the
`{W}` position only.

## Addendum 1 (2026-10-19; written after the run had started and the first trial's 40 cells showed the discrepancy; no result had been analysed)
The design above says "offsets {−0.2, 0, +0.2}" and "192 cells", but the script uses the H6 offset list `OFFSETS` = {−0.2, −0.1, 0, +0.1, +0.2} (all five offsets of the original H6 run), i.e. **8 trials × 8 layers × 5 offsets = 320 cells** (40 per trial).
The run continues unchanged. C1, the reproduction check and the layer profiles use all 320 cells (the layer profiles use the offset-0 cells); C2 and the decision statistic D use the offsets −0.2, 0, +0.2 exactly as in H6. The "192" was a counting error in this file.

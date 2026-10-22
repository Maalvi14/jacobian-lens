# Pre-registration: B′ control populations (variance, token directions, lens-fit recipe) and more Gemma null seeds

Written 2026-10-20, **before any of these controls was run**. B′ holds on three models (`preregistration_b_prime.md`, `preregistration_b_prime_llama.md`): the median κ along J-lens directions exceeds the median κ along isotropic random directions, and weight-randomized models show no effect. A review of the draft paper named four alternative explanations that the isotropic baseline cannot exclude. This file fixes the tests for them.
`hlens/experiments/curvature_controls.py` records this file's SHA-256 in every result file. These controls were designed after seeing the B′ results (stated openly); the thresholds below are the B′ ones.

## What is fixed
- **Models and layers:** Qwen3.5-4B, Gemma 3 12B, Llama 3.1 8B, fp32, eager attention, the Neuronpedia lenses of B′. Real arm only (the null question is answered by B′). Layers: the B′ real layers at depth ≥ 0.5 (Qwen and Llama L16–30 as in B′; Gemma L24–44).
- **Prompts and sampling:** the B′ evaluation prompts (WikiText 200–229), 16 contexts per prompt, 64 directions per population, seed 0, the B′ prompt-cluster bootstrap (300 resamples). The `j_lens` and `random` populations are regenerated with the same seeds as B′ and must reproduce the stored B′ ratios to within 0.02 (procedural check).
- **New direction populations** (unit vectors at source layer ℓ, same measurement as B′: κ = ‖½ vᵀHv‖ of the MLP branch at layer ℓ+1):
  - `cov`: directions sampled from the second-moment matrix of the residual stream, v ∝ Σᵢ gᵢ rᵢ with rᵢ activations entering the layer-(ℓ+1) MLP at 1,200 positions of disjoint prompts (WikiText 400–449, 24 positions each) and gᵢ ~ N(0,1).
  - `varmatch`: for each of the 64 lens directions, the unused direction from a pool of 1,024 `cov` and 1,024 isotropic directions whose variance vᵀΣv (same activations) is closest in log scale. Matching quality (median |log ratio|) is reported. **This is the variance-matched control.**
  - `embed`: unit rows of the input-embedding matrix for 64 of the active tokens used for `j_lens`.
  - `unembed`: unit unembedding gradient directions uₜ (the lens vectors without Jℓ) for the same 64 tokens.
- **Statistic:** for each population P, the ratio κ(j_lens)/κ(P) as in B′ (median over contexts and directions, prompt-cluster bootstrap 95% interval).

## Tests (per model, over its layers at depth ≥ 0.5)
- **V (variance):** the lower bound of κ(j_lens)/κ(varmatch) exceeds 1.05 at ≥ 75% of layers. *If V holds, the enrichment is not explained by the lens directions' alignment with high-variance directions of the residual stream.* If V fails, the B′ effect is at least partly a variance-alignment effect and the paper's claim is narrowed accordingly.
- **K (token directions):** the lower bound of κ(j_lens)/κ(embed) and of κ(j_lens)/κ(unembed) exceeds 1.05 at ≥ 75% of layers (each reported separately). *If K holds, the Jacobian map adds curvature structure beyond the token directions it is built from.*
- **C (cov, descriptive):** κ(j_lens)/κ(cov) is reported without a threshold.
- **Decision table.**

| V | K | Statement |
|---|---|---|
| holds | holds | the enrichment is specific to J-lens directions, beyond variance and token directions |
| holds | fails | beyond variance, but token directions (embedding or unembedding) carry similar curvature; the claim becomes "token-aligned" rather than "J-lens-specific" |
| fails | any | the enrichment is explained (partly or wholly) by high-variance alignment; the B′ statement is narrowed |

## Lens-fit recipe control (Q3; Qwen3.5-4B and Gemma 3 12B)
The null lenses are fitted with 30 prompts (WikiText 300–329, `dim_batch` 32, `max_seq_len` 64), the real lenses with 417–775 prompts. The real model's lens is refitted with the **null recipe** at Qwen L16, L24 and Gemma L24, and the ratio κ(j_lens)/κ(random) is measured with it (same B′ evaluation prompts).
- **Q3 holds** if the lower bound of the refitted-lens ratio exceeds 1.05 at every one of these layers (the effect does not depend on the lens-fit length). Report the refitted-lens ratio next to the original one. If the refitted ratio is below 1.05, the null arms are not comparable with the real arm in lens quality and the paper says so.

## More Gemma null seeds (Q4)
Seeds 12, 13, 14 at layer 24 under the B′ configuration (`h2_replication.py null --config B2`; a dated addendum to the B′ registration, not a new rule): the B′ rule N applies unchanged (every null interval within [0.95, 1.05]). Gemma then has 5 null arms.

## Not part of this registration
Gain-normalised ratios ρ (already stored; reported descriptively); a causal test of use; other direction sources.

## Known limits (stated in advance)
`cov` directions are random combinations of 1,200 activations, so they live in the span of activations seen; the matched pool is finite; embedding rows of Gemma are scaled by √d in the model (the direction is unit-normalised). One seed and 64 directions per population.

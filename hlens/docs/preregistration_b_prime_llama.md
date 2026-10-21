# Pre-registration: B′ on a third model — Llama 3.1 8B

Written 2026-10-19, **before any Llama 3.1 8B measurement** (the weights are not yet downloaded; only the Neuronpedia lens file has been fetched and its shape checked: layers 0-30, d = 4096, 539 prompts, finite).
This repeats `preregistration_b_prime.md` — hypothesis, margin, tests and decision logic unchanged — on a dense decoder-only transformer, to ask whether B′ is a property of trained transformers or of the two models it was found on
(Qwen3.5-4B, a hybrid with Gated-DeltaNet layers; Gemma 3 12B, sliding + global attention). `hlens/experiments/h2_replication.py --config B3` records this file's SHA-256 in every result file; the real arm refuses to run before a null arm under the same hash.

## Hypothesis B′ (unchanged)
For a model with a lens fitted on it, in layers with depth ≥ 0.5, the median J-lens/random ratio of κ = ‖½ vᵀHv‖ of the MLP branch (own norms included) is above what a weight-randomized model gives, and the weight-randomized model gives a ratio equivalent to 1.

## Model and lens
`meta-llama/Llama-3.1-8B` (base; 32 layers, d = 4096, GQA, SwiGLU; fp32, eager attention; ~32 GB in fp32; gated on HF, manual approval) with the Neuronpedia lens `llama3.1-8b/jlens/Salesforce-wikitext/Llama-3.1-8B_jacobian_lens.pt`
(layers 0-30; fitted in bf16 on 539 prompts, the fit having stopped on its convergence criterion). The base model and its own lens are used, matching the Gemma `-pt` choice. With 32 layers the Qwen3.5-4B layer indices are already depth-matched (`matched_layer(32, l) = l`).

## Fixed design (identical to B′ except the model and the seeds)
- **Fresh data:** evaluation prompts = wikitext indices 200-229; lens-fitting prompts for the null models = indices 300-329 (the same ranges as the Qwen and Gemma B′ runs; the prompts are model-independent and were not tuned on).
- **Equivalence margin δ = 0.05**; 16 random valid positions per prompt, 64 directions per population, prompt-cluster bootstrap (300 resamples), fp32; the null by `reinit` randomization, bf16 rounding, a lens fit with `dim_batch` 32 / 64 tokens (30 prompts).
- **Null arms (new seeds):** seeds 20, 21, 22 at layer 16; seed 20 at layers 8 and 24 (one fit). (The Qwen B′ arms were seeds 10-12 at layer 16 and seed 10 at layers 8 and 24.)
- **Real arm:** layers {8, 16, 18, 20, 22, 24, 26, 28, 30}; **depth range for the tests:** layers with depth ≥ 0.5, i.e. 16-30 (8 layers).

## Tests (unchanged)
- **N:** every null arm's whole 95 % interval of the J-lens/random κ ratio lies inside [0.95, 1.05].
- **R:** the real arm's 95 % interval lower bound > 1.05 in at least 75 % of the measured layers in the depth range.
- **C:** at each null-lens layer in the depth range (L16, L24), the real arm's lower bound is greater than the upper bound of every null arm at that layer.
- **B′ holds on Llama iff N ∧ R ∧ C.**

## Reported but not tests
The median ratio over the depth range and whether it reaches 1.5 ("large effect"); the Spearman correlation of ratio with layer; the neuron-direction population and the first-order gain (g1) ratio; layer 8 (depth 0.26) as an exploratory N check.

## Decision table
| Outcome | Statement |
|---|---|
| holds on Llama | B′ holds on three architectures (hybrid linear-attention, sliding + global attention, dense full attention) |
| N fails on any arm | the randomized control is not flat on this model; the effect cannot be attributed to learned structure there |
| N holds, R fails | the effect is not distinguishable from the null region on Llama; B′ is reported as holding on the two models and failing to extend |
| N and R hold, C fails | the real arm is not separable from the nulls at the matched layers; reported as such |
| the lens or a run fails | not tested; no partial result is used |

## Known limits (stated in advance)
The lens was fitted on 539 prompts (the Qwen3.5-4B lens on 417, the Gemma one on 775); one randomization recipe (30 prompts) and four null fits; the base model only; the J-lens/random ratio is a statement about MLP-branch curvature at one layer at a time, not about use by the model;
Llama's 128k-token vocabulary and untied output head make the lens fit more expensive than the Qwen one, but do not change the protocol.

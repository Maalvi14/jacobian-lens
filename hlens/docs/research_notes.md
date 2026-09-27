# The H-lens: Second-Order Structure of the Verbalizable Workspace

**A research program extending the Jacobian lens (Gurnee et al., 2026) from first-order readout to local curvature**

*Draft research summary · September 2026*

---

## TL;DR

The Jacobian lens (J-lens) characterizes an activation by its **first-order** causal effect on model outputs, averaged over contexts. Nearly every open question the paper leaves on the table — what ignition mechanistically *is*, why capacity saturates at ~25 concepts, whether broadcast is a weight-alignment or a nonlinear-amplification phenomenon, whether early layers are genuinely empty or merely linearly inaccessible, how concepts are *bound* rather than merely present, and above all how comprehensively the lens covers alignment-relevant reasoning — is a place where a **nonlinear** phenomenon was measured with a **linear** tool and the residual was flagged as a limitation.

The H-lens program adds the next Taylor term. Concretely, we compute Hessian-vector products (HVPs) of layer-to-layer and layer-to-output maps, restricted to J-lens directions and their non-J-space remainders, and use them for three things:

1. **A per-position trust score** for the J-lens readout: the ratio of the second-order to first-order Taylor term along the perturbations we actually apply. This turns "we don't know what the lens misses" into a measured quantity, and directly targets the paper's stated blind spot (§9.2) that automatized misaligned behavior may bypass the J-space entirely.
2. **A restricted Hessian** `H_ij = v_iᵀ H v_j` over J-lens vectors and remainders, which is the natural object for testing whether workspace contents are *bound* (relational structure lives in cross-terms), whether they *compete* (ignition and capacity as winner-take-all dynamics), and whether broadcast is nonlinear.
3. **A reliability harness** — estimator agreement, ε-sweeps, multi-step HVP, and weight-randomized nulls — so that no curvature number is reported without a validity gate. This is the methodological deliverable regardless of how the scientific hypotheses land.

We build on the released `jlens` reference implementation as a sibling package (`hlens`), reproduce the paper's headline numbers on an open model (Qwen3.5-4B) as gates before trusting any new measurement, and order experiments so that the cheapest alignment-relevant result (the trust score) lands first and the flagship mechanistic result (ignition curvature) lands last on a validated pipeline.

---

## 1. Where the paper stops, and why it stops there

### 1.1 What the J-lens is

The paper defines, for each layer ℓ, a single corpus-averaged matrix

```
J_ℓ = E_{t, t'≥t, prompt} [ ∂h_final,t' / ∂h_ℓ,t ]
```

and reads an activation via `lens(h_ℓ) = softmax(W_U norm(J_ℓ h_ℓ))`. The rows of `W_U J_ℓ` are the **J-lens vectors**, one per vocabulary token. The **J-space** is the set of sparse nonnegative combinations of at most k≈25 of these vectors (§2.3, §A.8); an activation's J-space component is found by gradient pursuit and typically explains ≤10% of variance (§4.2), yet carries most of the causal effect in swap experiments (§3.3, Figure 16). Writing is by steering, ablation, or the coordinate swap `h ← h + V(σ(c) − c)` with `c = V⁺h` (§2.5).

Every one of these operations is linear in `h` or in the perturbation. The corpus-averaging of `J_ℓ` deliberately discards the context-specific component of the Jacobian (§2.1), and the paper's own footnote on the MLP-gain metric (§4.3.1) concedes that the gain "may be context-dependent" in ways the linearized measurement cannot resolve.

### 1.2 The open questions are second-order questions

Mapping the paper's explicit limitations and future-work items (§4.1, §9.1, §9.2, §A.15, §A.24.2) to what a second-order object could say about them:

| Paper gap | Where flagged | First-order tool used | What curvature adds |
|---|---|---|---|
| Automatized behavior bypasses the J-space; lens coverage of alignment-relevant reasoning is uncharacterized | §3.5, §5, §9.2 | J-lens readout | A per-position **faithfulness score** that flags where the linear readout is locally untrustworthy, *including where the readout shows nothing* |
| Gradient pursuit is greedy, non-unique, locks onto near-miss correlates | §2.3, §A.24.2 | Sparse pursuit | Cross-terms `v_remainderᵀ H v_j` quantify entanglement between the chosen atoms and the discarded remainder |
| "Beyond a bag of concepts": binding/relational structure invisible to a flat readout | §9.1 | Ranked token list | Off-diagonal restricted Hessian entries are the local operationalization of "these two concepts interact" |
| Ignition defined statistically (transition-width collapse over ~L21–42, bimodality) but never located mechanistically | §4.1.1, §A.15 | Projection share along a line | Curvature of the layer map along the interpolation path; mutual-inhibition cross-terms between near-threshold candidates |
| Capacity ceiling (~25 vectors, ~6 unrelated items) measured, not explained | §4.2, §A.17 | Occupancy counts | Test whether capacity is a *dynamical* (competitive-inhibition) property rather than a storage budget |
| Broadcast gain (~10×) is itself a linearized metric | §4.3.1, footnote 5; §A.19 | Norm gain, cosine alignment | Does the MLP's local curvature favor J-aligned directions beyond what weight alignment predicts? |
| Early-layer absence: degenerate lens vs. genuinely empty | §4.1, Figure 28(d) | Effective rank of `W_U J_ℓ` | If the linear map is near-degenerate, test whether the second-order term still carries above-null verbalizable signal |
| Late dimensionality rise interpreted as `J_ℓ → I` | §4.1 | Effective rank | Curvature should collapse toward trivial in the motor regime if the interpretation is right; a direct test rather than an inference |
| No mechanistic account of how contents enter the workspace | §9.1 | — | Curvature barriers around entry are a candidate mechanism, and a candidate training target |

The unifying observation: the paper measures a family of sharp, all-or-nothing, competitive phenomena (ignition, capacity displacement, selective routing, persona flagging) and the tool it uses is by construction blind to sharpness.

---

## 2. The H-lens object

### 2.1 Definitions

For a map `f: h_ℓ ↦ h_ℓ'` (layer slice) or `f: h_ℓ ↦ logits` (layer-to-output), the Hessian is a 3-tensor. We never materialize it. We fix a **contraction** to a scalar and compute HVPs:

- **Cotangent contraction** `⟨c, f(h)⟩` — used for layer-to-layer spectra, with `c` chosen from the top J-lens direction at the target layer. This is the natural second-order analogue of the paper's "backpropagate from the target-layer residual stream" construction.
- **Logit contraction** — Hessian of a single labeled logit after unembedding; used for labeled-coordinate curvature in binding experiments.
- **Norm form** — for the trust score, `‖½ δhᵀ H δh‖` is computed vector-valued as `d/dε [J(h+εδh) δh]|₀`, one JVP of a JVP, with no contraction choice.

The **restricted Hessian** over a set of `k` directions `{v_i}` is

```
H_ij = v_iᵀ H v_j,     k HVPs (one per column), then one matmul per column
```

symmetrized as `(H + Hᵀ)/2`, with the asymmetry recorded as a free estimator-error diagnostic. The directions are J-lens vectors, pursuit-selected atoms, and the non-J-space remainder, so `H` is indexed by vocabulary tokens plus one "remainder" row — the same node vocabulary the paper's attribution graphs use (§A.24.2), now with an interaction matrix instead of only edges.

### 2.2 The trust (faithfulness) score

At position `t`, layer `ℓ`, for a perturbation `δh`:

```
τ(t, ℓ, δh) = ‖ ½ δhᵀ H δh ‖ / ‖ J δh ‖
```

The critical design decision is that `δh` is drawn from the **intervention family actually used** — the swap deltas `V(σ(c) − c)` and ablation deltas from `interventions.py` — not from random directions. A trust score evaluated on perturbations nobody applies measures nothing about the reliability of the interventions the paper's causal claims rest on.

High `τ` means the first-order readout at this site is locally untrustworthy. This is not the same as "the readout is wrong": it is a statement that the linear approximation underlying `J_ℓ h_ℓ` has left its validity radius for perturbations of the relevant size.

### 2.3 Relationship to the J-lens

The H-lens does not replace the J-lens. Per-context Jacobians `J_ℓ^(p)` are already computed by `jacobian_for_prompt` and discarded by `fit`; the H-lens is what you get when you stop discarding the context-specific structure and ask about its rate of change. Three regimes are worth distinguishing:

- **`τ` small everywhere**: the paper's linear story is locally adequate; curvature is a null result and the remaining work is characterizing where the corpus-average `J_ℓ` differs from `J_ℓ^(p)`.
- **`τ` large at positions where the J-lens shows clean content**: the readout is right about *what* is present but the causal inference from swaps is out of its validity range — expect this to correlate with the paper's swap-failure categories.
- **`τ` large where the J-lens shows nothing**: the candidate signature of automatized, non-verbalized computation. This is the regime that matters for §9.2.

---

## 3. Hypotheses

Each hypothesis is stated with the paper section it targets, a concrete prediction, and what result would falsify it. Ordering follows the build order (§5), not importance.

**H1 (Faithfulness / coverage; §3.5, §9.2).** The trust score predicts J-lens intervention failure. *Prediction:* on the paper's two-hop swap suite, `τ` at the swapped positions separates successful from failed swaps with AUC substantially above the weight-randomized null, and its category profile matches the known heterogeneity (country facts ≈ always succeed; number relations ≈ never at α=1, recovering at α=2). *Falsifier:* `τ` is uncorrelated with swap outcome, or is fully explained by the source vector's activation strength (which §9.1 already identifies as the dominant failure predictor).

**H2 (Gain is partly nonlinear; §4.3.1, §A.19).** The ~10× MLP gain on J-lens vectors is not fully explained by first-order weight alignment. *Prediction:* the second-order term of the MLP block along J-aligned directions is elevated relative to isotropic and neuron-direction controls in the workspace layers, and `∂gain/∂context` (from per-context Jacobians) has larger variance for J-aligned directions. *Falsifier:* curvature along J-aligned directions is at the isotropic baseline and the gain is reproduced by the linearized weight statistics of §A.19 alone. This is reported as a diagnostic, since the paper's footnote already concedes that gain in a network is not the gain of an isolated block.

**H3 (Binding lives in cross-terms; §9.1).** Concepts the paper's swaps show to be jointly operated on (e.g. `21` and `doubled` feeding `42` in the arithmetic graph, §A.24.2) have elevated `|H_ij|` relative to co-present but non-interacting pairs. *Prediction:* off-diagonal restricted-Hessian mass concentrates on pairs whose joint swap has a super-additive effect. *Falsifier:* `|H_ij|` is indistinguishable from the null, or is high for arbitrary co-present pairs (in which case it measures co-activation, not binding). A finding survives only if it passes a **three-leg criterion**: (i) causal beyond first order — the joint swap effect exceeds the sum of single swaps; (ii) stable under pursuit seed and `k` variation; (iii) above the weight-randomized null.

**H4 (Automatization reduces curvature; §3.5.2, §5.4).** As a behavior becomes automatic through RL, curvature along the direction separating "J-space-routed" from "circuit-routed" processing decreases. *Prediction:* across the reward-hacking organism's checkpoints (SDF → phase 1 → phase 2), the direction identified by §3.5.2's ablation contrast shows attenuating curvature that tracks the attenuating J-lens signal. *Status:* blocked on checkpoint access; the measurement is specified so that it can run the moment access exists.

**H5 (Remainder is not inert; §2.3, §A.24.2, Figure 16).** The non-J-space remainder, which carries ~90% of variance but only ~28% of swap effect, is coupled to the selected atoms through second-order terms. *Prediction:* `v_remainderᵀ H v_j` is above null for the atoms the pursuit selected, i.e. the remainder gates or amplifies the sparse decomposition rather than being orthogonally separable. *Pre-registered falsifier:* the null distribution is written to disk before the experiment runs; the hypothesis fails if the observed cross-terms fall within it.

**H6 (Ignition is a curvature event; §4.1.1, §A.15).** Along the paper's country-pair interpolation `(1−α)e_B + αe_A`, layer-to-layer curvature spikes are concentrated at the layers where transition width collapses. *Prediction:* the leading eigenvalue of the layer-to-layer Hessian (via power iteration or Lanczos over the HVP operator with a cotangent contraction) peaks in the depth-percentile band corresponding to L21–42 in the paper, and near the α at which per-trial projection shares are bimodal. *Two sub-questions the measurement resolves regardless of outcome:* (a) is there one commitment layer or several (the §A.14 naming-vs-avoidance asymmetry suggests early- and late-workspace roles differ); (b) do simultaneously near-threshold candidates show mutual-inhibition cross-terms, i.e. is "sharp, competitive ignition" literally competitive. *Falsifier:* curvature is flat or its peaks are uncorrelated with transition-width collapse — in which case the sharpening is a property of the trajectory across many small steps and the local Hessian is the wrong object. A knee-versus-convergence-radius diagnosis is mandatory in the writeup: a curvature spike that coincides with the boundary of the local Taylor radius is an artifact of the estimator, not a property of the model.

**Secondary hypotheses (from the brief; scheduled after the core program):**
- **H7 (Early-layer content; §4.1).** Where `W_U J_ℓ` is near-degenerate (Figure 28(d)), test whether the second-order term restricted to J-lens directions carries above-null signal about downstream verbalizable content. Whether the curvature onset tracks a fixed depth percentile or the accumulation of MLP nonlinearity is a secondary question.
- **H8 (Motor-regime collapse; §4.1).** If the late dimensionality rise is `J_ℓ → I`, curvature in that band should collapse toward the trivial case. This is the cheapest direct test of an interpretation the paper reaches by inference.
- **H9 (Capacity as competition; §4.2, §A.17).** If H6 confirms inhibitory cross-terms, reframe the ~25-vector ceiling as a dynamical property; the list-loading protocol of §4.2 gives a ready stimulus set.
- **H10 (Register and persona transitions; §3.5.3, §6).** The felt-vs-mechanistic register and the default-Claude-vs-roleplay flagging (`disclaimer`, `fictional`) are candidates for the same ambiguous-interpolation protocol as H6, with activation directions or persona pairs replacing concept pairs.
- **H11 (Reflection training changes curvature; §7).** Beyond whether ethics-related lens vectors appear, test whether counterfactual reflection training lowers the curvature barrier around them in the original context — a mechanistic account of "installed disposition" and, if confirmed, a candidate training signal.

---

## 4. Methods

### 4.1 Estimators

Three HVP estimators, built in this order:

1. **Reverse-over-reverse** (`hvp_rr`): nested `torch.autograd.grad` with `create_graph=True` on the inner call. Works directly on `jlens`'s hook infrastructure (`start_graph_at` makes `h_ℓ` the leaf). Reference implementation.
2. **Forward-over-reverse** (`hvp_fr`): `torch.func.jvp(torch.func.grad(f_scalar))` on a pure functional layer slice. Production path — no double graph in memory. Requires the `functional.py` wrapper, which threads rotary `position_embeddings`, `attention_mask`, and `position_ids` through the block stack and is validated against the hooked forward to tolerance before any HVP runs through it.
3. **Central finite difference** (`hvp_fd`): `(∇f(h+εv) − ∇f(h−εv)) / 2ε`. Validation only; `ε` is exposed for the sweep.

All three must agree on (a) analytic quadratic forms, (b) a small tanh MLP against `torch.autograd.functional.hessian`, and (c) `jlens`'s own `tests/tiny.py` model, before they are pointed at a language model. Precision is explicit: the model runs in bf16, all HVP inputs and accumulations are cast to fp32 (mirroring `jlens`'s fp32-in-memory convention), and the bf16-induced noise floor is a *measured* quantity, not an assumption.

### 4.2 Reliability harness

This is the part of the program most directly informed by the recent second-order attribution literature. Zhang & Wang (2026) show that the dominant error in attribution patching comes from downstream nonlinearities rather than local curvature at the patched site, and derive (i) a reliability score, (ii) error bounds, and (iii) an HVP correction, with a multi-step HVP variant that matches integrated gradients at lower cost and a "Screen-Flag-Fix" workflow that spends second-order compute only where the first-order estimate is flagged. Kramár et al. (2024) had earlier catalogued the two main first-order failure modes — attention saturation and cancellation between direct and indirect effects — and proposed AtP*.

The harness adopts this structure wholesale:

- **ε-sweep with plateau detection.** No curvature value is reported without a flat region in the finite-difference sweep; no plateau ⇒ flagged, not reported.
- **Estimator agreement mask.** Pairwise relative error among `rr`/`fr`/`fd` per (position, direction); disagreement ⇒ that cell is masked in every downstream figure.
- **Multi-step HVP.** Composed small-step correction along the perturbation path; divergence between single-step and multi-step is the reliability score. This matters most near ignition knees, which is exactly where H6 is looking.
- **Validity score.** Estimated local Taylor radius (from the plateau edge and the magnitude of the second-order term) versus the intended perturbation size. This is the prototype of the trust score in §2.2.
- **Weight-randomized null.** Following Adebayo et al.'s sanity-check logic: re-initialize the model, **refit a small `jlens` lens on the randomized weights** (the pre-fitted lens is meaningless there; a faithful null re-runs the whole pipeline), and record cross-term and directional-curvature nulls to disk once. Every Phase-2 script loads these rather than recomputing.
- **Ablation table.** The second-order counterpart of the paper's §A.7 battery: Nσ outlier filters, mean vs. median aggregation, stop-gradient variants (frozen-QK, self-only, future-only), position schemes, direction sets, `ε`, and step count, as a single config-sweep runner. This table is a standalone deliverable.
- **Pathological stress test.** A constructed high-curvature case (positions straddling a sharp top-1 flip under interpolation) in which single-step HVP visibly degrades and multi-step plus gating recovers. Passing this is the gate for all of Phase 2.

### 4.3 Reproduction gates

Nothing new is trusted until the pipeline reproduces the paper on the open model:

| Gate | Paper target | Pass criterion |
|---|---|---|
| Sparse decomposition | §2.3, §4.2 | J-space fraction of variance ≤ ~10% at mid layers; ~25 meaningfully active vectors; sensible atoms on the walkthrough's multihop prompt |
| Coordinate swap | §2.5, §3.3 | Swap-then-swap-back is identity to tolerance; the multihop swap flips the currency answer |
| MLP gain | §4.3.1, Figure 32 (left) | J-lens vectors reach ~10× normalized gain in the workspace band; neuron directions stay near 1 |
| Layer landmarks | §4.1 | Workspace onset/offset located as depth *percentiles*, not absolute layer indices, before any curvature is plotted against depth |

The SAE-stratified gain panel (Figure 32, right) is out of scope unless Qwen SAEs are available; the neuron-direction comparison is sufficient for the gate.

### 4.4 Model, hardware, and what does not transfer

All work runs on Qwen3.5-4B (and opportunistically a ~27B checkpoint) on a single Jetson Thor with 128 GB unified memory, compile disabled, plain-torch attention (the "fast path" is deliberately not installed — higher-order autodiff through fused kernels is a known hazard). Costs are budgeted against a restated milestone: one per-context restricted Hessian at one position ≤ 2–3× the cost of one `lens.apply` call.

The paper's numbers come from Claude-scale production models, and it explicitly does not know how the workspace scales with model size (§9.1). Three consequences:

- Any absolute layer number in the paper is treated as a percentile of depth.
- A failure to reproduce a gate on a 4B model is first evidence about scaling, not about the pipeline — but only after the analytic tests pass.
- Results are reported as "on Qwen3.5-4B" with the scaling caveat stated once, prominently, not hedged in every sentence.

### 4.5 Early averaging test

Before any of the above, one script answers a question the roadmap cannot skip: over ~100 pretraining-like prompts, is `‖E[H_restricted]‖ / E‖H_restricted‖` large enough that an *averaged* H-lens object exists at all, in the way `J_ℓ` exists as a meaningful average? If the answer is no, the program is purely per-context and every claim is phrased accordingly. This is run once, early, and its result reshapes the writeup.

---

## 5. Experimental program

Ordered by dependency and by how soon each yields an alignment-relevant result:

1. **`trust_score.py` (H1).** Per-position `τ` on the swap suite, using swap deltas; AUC against outcomes; null overlay; category breakdown.
2. **`gain.py` (H2).** Reproduce §4.3.1 (gate), then `∂gain/∂context`. Diagnostic only.
3. **`binding.py` (H3).** Restricted Hessian on the multihop / poetry / order-of-operations suites shipped in `jlens/data/evaluations`; three-leg criterion; report the *validity-gated* fraction of pairs that pass.
4. **`automatization.py` (H4).** Specified now, run when checkpoints are available.
5. **`remainder_coupling.py` (H5).** Pre-registration file committed before the run.
6. **`ignition.py` (H6).** Last, on a fully validated pipeline: interpolation sweep from `ignition.json` (or reimplemented from §4.1.1), Lanczos spectra over the HVP operator, multi-step HVP at knees, validity and null overlays, knee-vs-radius diagnosis.
7. **Secondary (H7–H11).** Scheduled after the core program; each reuses the ignition protocol or the restricted Hessian with a different direction set.

A pairwise-interaction heatmap per position — cell = `H_ij`, opacity = validity score, randomized-null panel alongside — is the standard figure, built on `jlens.vis`'s page plumbing.

---

## 6. Alignment payoff

The paper is careful (§9.2) not to claim the J-lens is sufficient for monitoring, and names two escape routes for concerning mechanisms: automatized behavior that never enters the J-space, and concepts without single-token names. The H-lens addresses the first directly and the second partially.

- **Measured blind spot.** The trust score gives a per-position flag that is *independent of readout content*. A monitoring pipeline can escalate flagged positions to SAEs or attribution graphs (the paper's own recommended complements, §A.24) rather than escalating everything or nothing.
- **Automatization tracking.** If H4 holds, curvature along the routing direction is a trainable, checkpoint-comparable signal for "this behavior is becoming a fixed circuit," which is exactly the case the paper says the J-lens will miss.
- **Training signals.** If H6/H9 establish curvature barriers around workspace entry, and H11 shows reflection training lowers them, then "make desired concepts win the competition for workspace access" becomes a concrete objective — a mechanistic refinement of what §7 achieves through verbal supervision alone. This is the most speculative item in the program and is stated as a direction, not a plan.

---

## 7. What would make us wrong

Anticipated failure modes, stated up front so the writeup can be honest about them:

- **Curvature is dominated by bf16 noise.** The fp32 casting and three-way agreement tests exist for this; if the noise floor swamps the signal at 4B scale, the honest result is "second-order structure is not resolvable at this precision/scale," reported as such.
- **`τ` is a proxy for activation strength.** §9.1 already attributes most swap failures to weak source-vector activation. If `τ` adds no information beyond that, H1 is a null result and the trust score is redundant with a cheaper first-order quantity.
- **Cross-terms measure co-activation, not binding.** The three-leg criterion is designed to catch this; a restricted Hessian that lights up for arbitrary co-present pairs is a covariance artifact.
- **Ignition knees coincide with convergence-radius boundaries.** Then the "curvature spike" is the estimator failing, not the model committing. The knee-vs-radius diagnosis is mandatory for this reason.
- **No averaged H-lens exists.** Then all claims are per-context and the program is a reliability tool plus a set of case studies, not a new corpus-level object.

---

## 8. Relation to prior work

- **Gurnee et al. (2026), *Verbalizable Representations Form a Global Workspace in Language Models*.** The base paper; all reproduction gates, direction sets, intervention definitions, and open questions are drawn from it. The companion `jlens` repository is the substrate.
- **Zhang & Wang (2026), *When Attribution Patching Lies*.** Establishes that downstream nonlinearity is the dominant source of first-order attribution error and supplies the reliability-score / error-bound / HVP-correction / multi-step-HVP toolkit the harness adopts. Their Screen-Flag-Fix workflow is the template for how the trust score should be used in monitoring.
- **Kramár et al. (2024), *AtP\**.** Catalogue of first-order failure modes (attention saturation, cancellation) and the verification-after-prefiltering discipline.
- **Ameisen et al. (2025) / Lindsey et al. (2025), circuit tracing and attribution graphs.** The paper's §A.24.2 graphs use J-lens atoms plus a remainder node as the node vocabulary; the restricted Hessian is indexed by the same nodes.
- **Adebayo et al. (2018), sanity checks for saliency maps.** The weight-randomization null, applied here with the lens refit on randomized weights.
- **Pearlmutter (1994).** The HVP trick itself.
- **Recent Hessian-based attribution in autoregressive LMs (e.g. HETA, ICLR 2026) and Hessian-guided feature-interaction discovery (CVPR 2026).** Independent evidence that second-order sensitivity is informative where gradients vanish, and that the Hessian is better used for *detecting* interactions than for scoring their magnitude — a caution the three-leg criterion respects.

---

## 9. Deliverables and milestones

| Phase | Deliverable | Gate |
|---|---|---|
| 1a | `hvp.py` three estimators; analytic and tiny-model tests pass on Thor | Validates the NGC PyTorch build's `torch.func` and nested-grad support on Blackwell |
| 1a | `functional.py`, `jspace.py`, `interventions.py` | Paper reproduction gates (§4.3) |
| 1a | Early averaging test | Decides per-context vs. averaged framing |
| 1b | `harness.py` with nulls on disk; ablation table; gain reproduction; stress test | **Phase-2 gate** |
| 2 | `trust_score.py` → `gain.py` / `binding.py` → `remainder_coupling.py` → `ignition.py` | Each script loads shared nulls and reports validity-gated results only |
| 3 | Writeup: methods paper (harness + ablation table) and results paper (H1–H6), with H7–H11 as follow-ups | Scaling caveat stated once; knee-vs-radius diagnosis included |

The methods deliverable stands on its own even if every scientific hypothesis returns null: a validated, cost-budgeted way to compute and gate second-order quantities on the J-lens's own direction set is a contribution the paper's authors explicitly asked for when they wrote that "further work on addressing the J-lens's limitations, and characterizing how comprehensively it covers models' alignment-relevant reasoning, could elevate it to a more load-bearing tool."
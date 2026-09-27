# H-Lens TODO — Explanation in Simplified Technical English

This document explains each item of `hlens_code_todo.md`. It uses ASD-STE100 style: short
sentences, simple words, one idea for each sentence. Technical names stay as they are. Each
technical name has a definition when it appears the first time.

## Definitions of the important words

- **Residual**: the vector that a transformer layer sends to the next layer. It has
  `d_model` numbers.
- **Jacobian (J)**: a matrix of first derivatives. It shows this: if the input changes a
  small amount, how much does the output change? The J-lens uses this matrix.
- **Hessian (H)**: the second derivatives. It shows this: how does the Jacobian itself
  change when the input changes? The Hessian measures the curvature of the model function.
- **HVP (Hessian-vector product)**: the result `H · v` for one direction `v`. We never
  compute the full Hessian. The full Hessian is too large. We only compute `H · v` for the
  directions we need.
- **Direction**: a vector with length 1 in the residual space. It points at one "concept".
- **J-lens vector**: the direction for one vocabulary token, made with the Jacobian. The
  paper found ~25 of these active at one token position.
- **Contraction**: the step that turns a vector output into one number. The Hessian of a
  vector function is a 3-dimensional object. We must select one number to take derivatives
  of. This selection is the contraction.
- **Null test**: the same measurement on a model with random weights. If the random model
  gives the same result, the result is not real. It is an artifact.

---

## Item 0 — Corrections

**Per-context Jacobians.** Before, we thought we must change the `jlens` code to get the
Jacobian for one prompt. This was wrong. The function `jacobian_for_prompt` is public. It
gives the Jacobian for one prompt. We can use it directly. No change to `jlens` is
necessary.

**Interventions are missing.** The `jlens` code can only read activations. It cannot change
them. But two of our experiments must change activations. So we must write this code
ourselves. See Item 4.

## Item 1 — `functional.py` (the pure function wrapper)

**What it is.** A tool that makes a "pure function" from a part of the model. The pure
function receives a residual at layer `l`. It returns the residual at layer `l'`. Nothing
else happens.

**Why we need it.** PyTorch has two systems for derivatives. The old system uses hooks. The
new system (`torch.func`) needs pure functions. The fast HVP method (forward-over-reverse)
only works with the new system. The `jlens` hooks do not work with the new system. So we
must make this wrapper.

**The hard part.** Modern transformer layers need extra inputs: position data and the
attention mask. The wrapper must supply these inputs correctly. If we forget one input, the
result is wrong.

**The test.** We compare the wrapper output with the normal model output. The two outputs
must agree. If they do not agree, the wrapper has a defect.

## Item 2 — `hvp.py` (the Hessian-vector product core)

**What it is.** The mathematical center of the project. It computes `H · v` in three
different ways.

**The three ways:**
1. **Reverse-over-reverse (`hvp_rr`)**: take the derivative two times with the old PyTorch
   system. This works with the `jlens` hooks today. We build this first.
2. **Forward-over-reverse (`hvp_fr`)**: the fast way. It uses `torch.func` and the wrapper
   from Item 1. It uses less memory. This is the production method.
3. **Finite difference (`hvp_fd`)**: compute the gradient at two near points. Subtract.
   Divide by the distance. This is slow and simple. We use it only to check the other two.

**Why three ways?** The three methods must give the same answer. If they agree, our code is
correct. If they do not agree, we have a defect or a numerical problem. This "three-way
agreement" is our most important safety test.

**The contraction decision.** The map from layer `l` to layer `l'` gives a vector, not a
number. Second derivatives of a vector are a 3-dimensional object. We cannot use it
directly. We must first select one number. The code makes this selection explicit with a
`Contraction` object. The roadmap did not make this decision. The code forces the decision.
This is good. A hidden decision causes confusion later.

**The restricted Hessian.** The full Hessian has `d_model × d_model` numbers. Too many. We
only compute `v_i · H · v_j` for the ~25 active J-lens directions. This gives a small
25 × 25 table. The table shows how each pair of concepts interacts. The cost is only 25
HVPs, not 625.

**The precision rule.** The model computes in bf16. This number format is not precise.
Second derivatives make the noise larger. So we convert to fp32 for all Hessian
mathematics. This copies the `jlens` rule.

**The toy tests.** We first test on a tiny model. For a tiny model, we can compute the
true Hessian exactly. Our three methods must match the true Hessian. We run this test on
the Thor first. The test also shows if the Thor PyTorch build is correct.

## Item 3 — `jspace.py` (the sparse decomposition)

**What it is.** The tool that finds the ~25 active J-lens vectors at one position. It also
computes the remainder: the part of the residual that is not in the J-space.

**Why we must build it.** The paper describes this tool. But the released code does not
contain it. We must build it from the paper text.

**How it works (gradient pursuit).** The dictionary contains one direction for each
vocabulary token. That is more than 150,000 directions. The pursuit is a greedy loop:
1. Find the direction that matches the residual best.
2. Add it to the selected set.
3. Fit positive weights for the selected set.
4. Subtract the fitted part. Repeat with the rest.
The loop stops at ~25 directions.

**The memory note.** The full dictionary for one layer uses about 1.5 GB. The Thor has 128
GB. This is not a problem. But we do not keep copies for many layers at the same time.

**The validation gate.** The paper reports two numbers: the J-space holds ≤ 10% of the
variance, and ~25 vectors are active. Our code must reproduce these numbers. If it does
not, our pursuit is different from theirs. Then we stop and find the difference. We do not
continue with a wrong tool.

## Item 4 — `interventions.py` (the write operations)

**What it is.** Code that changes activations during a forward pass. The `jlens` hooks can
only read. This module can write.

**The parts:**
- **ActivationEditor**: a context manager, like `ActivationRecorder`. But its hooks return
  a changed tensor. PyTorch then uses the changed tensor.
- **Steer**: add a direction to the residual.
- **Ablate**: remove a direction from the residual.
- **Coordinate swap**: exchange two concepts. Example: replace "Italy" with "France" inside
  the residual. The paper gives the exact formula.

**Why we need it.** Experiment 2C asks: when does a swap fail? To answer, we must do swaps.
Experiment 2E asks: does a pair of concepts have a causal interaction? To answer, we must
change one concept and watch the other. Both need write access.

**The test.** A swap, then the reverse swap, must give the original residual back. Also: on
the "country shaped like a boot" prompt, a swap of the country must change the currency
answer. The paper shows this result. We must reproduce it.

## Item 5 — `harness.py` (the reliability harness)

**What it is.** The quality-control system. Second derivatives in bf16 are noisy. Every
number we report must pass these checks first.

**The parts:**
- **Epsilon sweep**: run the finite-difference method with many step sizes. Plot the
  result against the step size. A correct measurement shows a flat region (a plateau). No
  plateau means: do not trust this number.
- **Estimator agreement**: compare the three HVP methods at each position. Mark the
  positions where they agree. Use only these positions.
- **MS-HVP (multi-step HVP)**: for large perturbations, one Hessian step is not enough.
  We walk the path in many small steps. We also compute a score: how different is the
  one-step result from the many-step result? A large difference is a warning.
- **Validity score**: the Taylor approximation is only correct inside a small radius. We
  estimate this radius. We compare it with the size of our perturbation. If the
  perturbation is larger than the radius, the measurement is not valid.
- **Randomized null**: make a copy of the model with random weights. Run the full
  pipeline on it. This includes a new small J-lens fit, because the old lens has no
  meaning for a random model. Important: never randomize your only loaded model. Load a
  fresh copy from the cache first.
- **Ablation table**: run the pipeline many times with different settings (filters, step
  sizes, position rules). The table shows which results are stable and which depend on
  settings. This table is a deliverable by itself.
- **Benchmark**: measure the time cost. Lock the Thor clocks first (`nvpmodel -m 0`,
  `jetson_clocks`). Without locked clocks the times are random.
- **Early averaging test**: compute the mean Hessian over 100 prompts. Compare its size
  with the mean size of one Hessian. If the mean is much smaller, the Hessians point in
  different directions for each prompt. Then an "average H-lens" object does not exist.
  This one cheap test decides a large part of our approach. Run it early.
- **Stress test**: build one input where we know the curvature is extreme. The simple
  method must fail there. The MS-HVP and the gates must catch the failure. This test is
  the gate for Phase 2. If the harness does not catch a known failure, the harness is not
  ready.

## Item 6 — `data.py` (the shipped experiment data)

**What it is.** The repository contains JSON files from the paper's experiments. Examples:
`probe-swap.json`, `ignition.json`, the evaluation suites.

**Why it matters.** Our roadmap says: "reconstruct the paper's test cases". Maybe we do
not need to. Maybe the files already contain the cases. We open the files and look at
their structure. If a file has the full cases, we only write a loader. That is much less
work. If a file has only summary numbers, we must reconstruct the cases. We find out
early.

## Item 7 — `experiments/` (the science scripts)

One script for each experiment. The order follows Roadmap v2: C first, then B and E, then
D, then F, then A last.

- **`trust_score.py` (2C)**: for each position, compute a ratio. Top: the size of the
  second-order term. Bottom: the size of the first-order term. A large ratio means: the
  linear lens is not trustworthy here. Prediction: swaps fail at positions with a large
  ratio. We test this prediction against the real swap failures. This is our fastest path
  to a useful result.
- **`gain.py` (2B)**: first, rebuild the paper's gain measurement. The measurement: how
  strongly does the next MLP block amplify a direction? The paper reports ~10× for J-lens
  vectors. We must reproduce this number. Then we add the new part: how does the gain
  change with context? The Hessian gives this answer.
- **`binding.py` (2E)**: compute the 25 × 25 interaction table on task suites. A large
  off-diagonal cell means: two concepts interact. Example hope: "42" and "doubled"
  interact during arithmetic. Every claimed interaction must pass three checks: a causal
  test with swaps, stability across settings, and the null comparison.
- **`automatization.py` (2D)**: compare curvature along J-space directions and along
  automatic-circuit directions, across training checkpoints. Blocked: we need access to
  the checkpoints. Keep this last in its group.
- **`remainder_coupling.py` (2F)**: does the J-space part interact with the remainder
  part? We write the prediction in a file before we run the test. This prevents us from
  fooling ourselves.
- **`ignition.py` (2A)**: the hardest experiment. The paper shows "ignition": a sudden
  jump when an ambiguous input crosses a boundary. We measure the curvature spectrum
  along the interpolation path. A sharp jump should appear as a curvature spike. Danger:
  at the jump, our own approximations also break. This is why 2A is last. The harness
  must be strong before we try it.

## Item 8 — `vis_ext.py` (the pictures)

Heat maps of the 25 × 25 interaction tables. Each cell shows the interaction strength.
The cell opacity shows the validity score. A pale cell means: do not trust this number.
Next to each map: the same map from the random-weights model. The reader can compare the
real result with the null directly.

## Item 9 — Environment rules for the Thor

- Keep `compile=False`. Compiled blocks and second derivatives do not mix safely.
- The "fast path not available" warning is good for us. The simple attention code is
  safer for second derivatives. Do not install the fast attention package.
- Lock the clocks before every benchmark.
- Use one device only. The 4B model and the 27B model both fit in the 128 GB memory.
- Write down the PyTorch source (the NGC container). Prevent `uv` from replacing it with
  a normal PyPI wheel. The normal wheel does not work on the Thor GPU.

## Item 10 — The build order

The order follows the dependencies:
1. `hvp_rr` with toy tests. This also validates the Thor PyTorch build.
2. The wrapper, then the other two HVP methods, then the three-way agreement.
3. The pursuit, with the paper-reproduction gate.
4. The interventions, with the swap reproduction.
5. The restricted Hessian (needs the pursuit).
6. The data inspection (any time, in parallel).
7. The harness (needs 1 and 2).
8. The milestone: the ~10× gain reproduction and the stress test. This opens Phase 2.
9. The trust score, then the other experiments in order.

The rule behind the order: **build the measuring instrument first, calibrate it, then
measure.** Every early item is instrument work. The science starts at step 9, and by then
every number passes through checks that we trust.
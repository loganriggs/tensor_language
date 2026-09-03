# Hourly strategic review — 2026-09-03 09:09 UTC

## Circuit interpretation targets

A useful decomposition must ultimately provide all of the following, not merely a low-rank reconstruction:

1. **Computational specification:** what information is read, what operation or interaction is computed, what is
   written, and which later computations use the write.
2. **Grouping across modules and splitting within modules:** merge pieces of different attention heads or MLPs when
   downstream computation treats them as one variable, and split a native module when its pieces do different jobs.
3. **Held-out and shifted-input prediction:** predict activations and signed behavioral effects on unseen documents,
   task variants, and distribution shifts.
4. **Extraction or sufficiency:** an executable extracted circuit, or explicit interface plus background, reproduces
   the target computation or signed causal effect.
5. **Selective manipulation:** removing, swapping, or editing the circuit changes its intended behavior while
   preserving unrelated behaviors, with redundancy and interactions included explicitly.
6. **Composition and reuse:** shared computations can serve multiple tasks/modules and their joint effects remain
   predictable.
7. **Stable identification:** the units survive document splits, fitting restarts, and valid basis rotations, or are
   defined by an operational downstream equivalence that is itself stable.

The program goal is a smaller transparent tensor program that is simultaneously predictive, composable,
manipulable, and literally cheaper in storage, computation, edges, and states. Compression, rank, reconstruction,
and CE alone cannot satisfy that goal.

## What changed since the previous Codex checkpoint

Rung 522 completed after 4,981.154 seconds. Its independent audit proves that TEST never opened, no removal ran, the
103 saved projectors match the result, and the model-call ledger is exactly 20,600 optimization forward/backward
pairs plus 5,029 inference-only forwards. Only 8/103 fits were healthy and none of the 15 required real fits was
healthy. This is an invalid optimizer instrument, not a circuit null.

The post-hoc spike localization was corrected. Although 110 exact target/member/control/map patterns spike in
multiple objectives, all share a random seed. Only 43/13,668 exact patterns appear under two seeds, and only two
spike-producing patterns have a cross-seed comparison opportunity. The archive therefore cannot identify
row-specific normalization versus learning rate.

Rung 523 was preregistered before prospective model work, implemented, tested, frozen, and queued through the
managed runner. It is live as PID `1823268`; at this checkpoint it is still in CPU preflight and has not allocated
the GPU. It tests the missing three cells of row-specific/fixed FIT scale by learning rate `.03/.003`, using only the
15 real fits and only FIT/VALIDATION. TEST and omitted-circuit evaluation are executable errors.

## Is the current route still the highest-information move?

Yes, narrowly and conditionally. R523 is not circuit evidence, but it is the cheapest discriminating test of whether
the already-built attention8 intervention can become a valid instrument. It costs 9,000 optimization calls rather
than repeating all 103 fits. Its outcomes choose among normalization repair, learning-rate repair, both, or closing
raw-Adam-through-QR. Skipping it would leave the attempted circuit test uninterpretable.

If R523 passes, one healthy sealed scientific repeat remains worth doing despite the weak prior from the MLP10
chapter. It directly tests cross-boundary grouping, held-out circuit prediction, selectivity, reuse on a fourth
circuit, and removal. The old `>=4x` member/control and `+1.0` improvement over whole-attention8 bars remain frozen.
A clean but weak result must be scored as the registered circuit null, not rescued by the known expectation of broad
attention8 effects.

The route stops being best after one healthy scientific null. Repeated rank changes or increasingly permissive
projector objectives would be rank drift and would not improve the seven circuit targets.

## Confound audit

- **Baseline subtraction:** all candidate and common-health quantities use signed CE differences from the same native
  float32 execution. R523 changes only the denominator or optimizer step.
- **Frame mixing:** each target/member/control/map record remains labeled; no reshape infers semantic axes.
- **Nonlinear loss composition:** every effect is measured by a physical model forward. CE changes are not added
  across interventions.
- **Shared token difficulty:** controls retain the frozen matching strata. Fixed scales use only member positions and
  therefore do not alter which controls are compared.
- **Leakage:** fixed target/map scales use FIT only; candidate choice uses common VALIDATION health; TEST and the
  omitted target are blocked by code.
- **Dead settings:** both changed factors affect the numerical objective; tests show the explicit denominator changes
  the loss and the complete 200-update candidate path runs.
- **Precision/noise floor:** rung 522's failures are many orders of magnitude above numerical precision. Later
  circuit conclusions must still respect the measured low cross-document signal of per-source fingerprints.
- **Post-selection:** the three candidate cells, health limits, and minimal-change decision order were written before
  prospective results. No extra learning rate may be added after seeing them.

## Genuinely different routes

1. **Finish R523 and, only if licensed, one unchanged sealed attention8 rerun.** This is the only current route that
   can directly turn the stable-but-broad attention8 effect into a cross-circuit selective intervention. Kill it if
   no R523 cell is healthy, or if the repaired run fails the frozen held-out selectivity gate.
2. **Exact MLP0 token/context interaction decomposition.** Fold the complete vocabulary embedding into MLP0 only as
   an analysis device, separate token-only, token-by-context, and context-only terms, and group token features by
   equal downstream effects rather than weight similarity. This attacks computational specification and within-MLP
   splitting using unusually complete input information. Kill a proposed group if it does not predict held-out
   token/task effects or support selective removal.
3. **Higher-signal downstream-response vocabulary across head pieces.** Define units by equality under a chosen set of
   later readers and finite interventions, allowing pieces from several heads to merge and one head to split. Use
   task-conditioned response functions rather than noisy per-source 32-circuit fingerprints. Kill it if equivalence
   classes do not transfer across documents/donors or their joint effects fail composition tests.
4. **Direct optimization on the Grassmann manifold.** If R523 closes raw Adam through differentiable QR, optimize the
   projector by tangent-space gradients followed by a retraction, while keeping the scientific objective and gates
   fixed. This is an instrument repair, not a new rank search. Kill it if a planted exact model fails or the 15 real
   FIT/VALIDATION health fits remain unstable.
5. **Raise document count for MLP10 per-source effects.** This can distinguish a noise-limited 12% held-out coverage
   from fundamentally small signal, but costs about 122,000 forwards and does not itself group or split circuits.
   Defer unless a downstream-response route genuinely requires those noisy coordinates.

## Ranked next actions and falsifiers

1. Score R523 exactly. Any prospective cell must have 15/15 healthy fits, no loss above 1,000, at most 3/3,000 above
   100, and common VALIDATION improvement for every fit. Otherwise that cell cannot be adopted.
2. If an arm is adopted, freeze a new sealed scientific runner with the original rung-522 A--D gates unchanged.
   Prediction A failing under a healthy instrument kills the shared rank-4 attention8 route.
3. If no arm is adopted, implement the already-declared direct-subspace optimizer toy falsifier before touching the
   model again. A planted failure kills that optimizer family.
4. At the attention8 terminal boundary, pivot to MLP0 exact token/context structure rather than changing rank.

R523 survives this review because it resolves a specific invalid instrument at modest cost and has opposing
predictions. Rank four is held constant and earns no evidence by itself.

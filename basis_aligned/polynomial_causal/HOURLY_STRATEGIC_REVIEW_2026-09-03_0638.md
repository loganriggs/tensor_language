# Hourly strategic review — 2026-09-03 06:38 UTC (Codex)

## Goal check

The target is a transparent tensor program whose circuit parts:

1. predict held-out data, donors, directions, and circuit instances;
2. can be extracted as the same reusable computation when different heads or MLP paths share it;
3. can be changed or removed without changing unrelated circuits; and
4. compose with other extracted parts so their joint computation is predictable.

Low rank, fewer parameters, quantization, and reconstruction error do not establish any of these. In rung522 rank 4
is only a matched capacity held constant across task-conditioned, recovery-only, random, and shuffled-label fits.
The claim is held-out selective causal reuse and removal, not rank reduction.

## What changed this hour

- Rung521 was closed honestly: the full attention8 response is reproducible but broad.
- The rung522 instrument smoke passed at the exact batch sizes with no model-weight gradients and no scientific
  outcome retained.
- A red-team audit moved all 103 learned frames before TEST and made TEST a one-way sealed sweep.
- CPU analysis proved the first label-null movement gate impossible. Under its exact strata, only 1,189 of 1,442
  nonzero four-bit codes can move. An exact minimum-cost assignment now reaches that algebraic maximum for every
  null seed rather than relaxing to an outcome-chosen percentage.
- The partial science runner remains behind an explicit kill switch. Its frame inventory, state guard, scheduler,
  statistics, and current training-call components have 44 passing focused tests.

## Mathematical step-back and alternative routes

The constrained-permutation failure is evidence that tensor-network work benefits from solving the finite
combinatorial problem exactly before optimizing. The same stance suggests four alternatives/checks for rung522:

- view the task-conditioned frame as a quotient under the internal `Q -> QR` gauge and compare projectors, never
  named basis columns;
- treat the 32 downstream circuit responses as observables that identify an equivalence class of attention8
  directions, instead of assuming attention heads are the natural basis;
- use factorial/Möbius interaction terms if a single projected attention8 action remains broad, so shared pairwise
  or higher-order paths are separated rather than assigned to one source module; and
- if one linear projector fails but the response is reproducible, test a small input-conditioned family only under a
  new held-out mechanistic hypothesis—not by increasing rank until something passes.

The next action remains the simplest decisive one: finish the exact causal response trainer and labeled 16-arm
evaluator for the frozen linear-projector hypothesis. The alternatives are successors only after its registered
failure mode is known.

## Anti-Goodhart check

No simplicity objective is allowed to select itself on the final circuit set. Success must be evaluated on withheld
circuits/data and by targeted removal with broad collateral controls. A method that merely makes storage smaller or
causal responses easier to approximate but fails component reuse or selective removal gets no interpretability
credit.

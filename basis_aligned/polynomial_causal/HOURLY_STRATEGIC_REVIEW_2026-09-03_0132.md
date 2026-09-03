# Hourly strategic review — 2026-09-03 01:32 UTC

## Circuit interpretation targets

A useful decomposition must eventually provide all of the following:

1. **Computation:** what information is read, what operation is performed, what is written, and what later
   computation uses it.
2. **Grouping and splitting across native modules:** merge parts of different heads/MLPs when downstream use treats
   them as one variable, and split one native module when its parts serve different computations.
3. **Held-out and OOD prediction:** predict activations and signed behavioral effects on unseen documents, tasks, and
   data shifts.
4. **Extraction or sufficiency:** an executable isolated circuit, or a precisely specified interface and background,
   reproduces the target computation or causal effect.
5. **Selective manipulation:** removals, swaps, and edits change the intended behavior while preserving unrelated
   circuits, including redundancy and interaction effects.
6. **Composition and reuse:** shared subcomputations work in multiple tasks/modules and combine predictably with
   task-specific branches.
7. **Stable identification:** units survive data splits, plausible gauges, and fitting restarts, or are defined by
   downstream operational equivalence.

The whole-program target remains a smaller transparent tensor program that is predictive, composable, manipulable,
and literally cheaper in storage, compute, edges, states, or program instructions. Rank, reconstruction, or CE alone
cannot substitute for the circuit targets.

## What changed since 00:32

Rung514 exhaustively rejected48 fixed factor allocations and113,520 signed two/three-term consumer programs. Its
planted search recovered all8 known programs, but no real program passed either independent search. That closed
small linear combinations of exact attention11/MLP11 interactions as the cross-implementation variable.

Rung515 then changed the observation rather than widening the program. It physically removed all816 exact
`(MLP10 branch, score implementation, attention11/MLP11 term)` nodes and recomputed the nonlinear suffix. The
managed run used52,452 forwards and50,592 term patches; all exactness, liveness, source-relation, calibration, and
planted-pair checks passed. There were791 material nodes but `0 / 17,460` cross-implementation pairs. The best quality
margin was`-1.1546`, far from the registered boundary. Thus different exact terms are not made equivalent by the
actual downstream task/circuit effects at this consumer boundary. Confirmation and physical substitution correctly
remained unopened.

The task-space SVD result was also corrected. The four-dimensional implementation loadings, not right singular
vectors over disjoint document coordinates, are the comparable cross-half object and have cosine`.99973`. Raw
top-one energy is`.964/.972`, but after subtracting the shared document-coordinate mean it is only`.587/.464`.
This is evidence for a large aggregate shared task component, not a one-state realization or identified circuit.

The 01:20 mathematical review maps the desired endpoint to a finite observational quotient. Exact bisimulation or
partition refinement would require a closed action/transition table and complete future observations; our sampled
interventions can identify only a registered restricted quotient. Hankel-rank results do not apply to the
four-by-document SVD matrix or the normalized nonlinear model.

Rung516 is preregistered and implemented as the cheap interpretation of the rung515 null. It asks which of the32
known circuit coordinates, if any, stably force task-compatible term pairs apart across document halves. A scope
audit removed two invalid promises before execution: the zero-pair artifact contains no responses for the unopened30
circuits, and the registered bipartite relation graph cannot yield a useful clique state-count bound.

## Is this still the highest-information route?

The exact-term equality-score descent is no longer the best place for another GPU experiment. Rungs506--515 have
changed observation, grain, term vocabulary, and local consumer while preserving nulls. Rung516 remains worthwhile
because it is already paid for, takes seconds on CPU, and directly answers whether known circuits rather than task
effects caused the final split. After that receipt, this arc should close unless it names a compact, stable observer
set that defines a genuinely different task-state experiment.

The active program should then pivot. The aggregate shared task component is potentially useful, but only if a
task-defined state can be extracted and selectively exchanged; fitting a rank-one approximation would repeat the
error the user has flagged. The alternative is a different documented program gap, with MLP0's token/context
decomposition especially attractive because its finite token input set supplies unusually complete information.

## Confound audit

- **Baseline subtraction:** rung515 compares removed minus intact within the same score implementation and reports
  circuit member-minus-control effects; the score-absent state is not mixed into the pair signature.
- **Frame mixing:** every pair stays at one consumer site. Attention11-to-MLP11 swaps are excluded because they occur
  on opposite sides of a nonlinear operation.
- **Nonlinear loss composition:** this is why rung515 recomputed the actual suffix; the result is a finite CE effect,
  not an additive interpretation of local writes.
- **Shared token difficulty:** member-minus-control circuit coordinates and two document halves reduce this, but the
  task coordinates retain a large shared mean; centered SVD shows that mean cannot be called a complete circuit.
- **Leakage and post-selection:** fixed32/30 circuit partitions, two document halves, no best-k pair selection, a
  1--16 cap, and16 permutations were frozen. Since zero pairs passed, confirmation stayed sealed.
- **Dead edits and precision:** all50,592 term removals were nonzero; captured patches matched requests exactly;
  attention/MLP corner identities and BF16 deployment closure passed.
- **Multiplicity controls:** all16 controls also returned zero. As in rung514, this floor is vacuous rather than a
  favorable margin; the real null rests on the absolute gates.
- **Rung516 scope:** it can validate named circuit witnesses over documents, not over new circuit identities. It is
  descriptive even on a full pass and cannot retroactively create a substitution-valid group.

## Genuinely different next approaches

1. **Circuit separation cover (rung516, execute now):** determine whether a small, stable set of already-known
   circuits is what forces task-compatible exact terms apart. Kill it if fewer than64 task-compatible pairs exist,
   circuits reject fewer than half, or a half0-selected top eight fails the frozen half1/control bars.
2. **Task-defined finite state transition:** treat the four equality implementations as alternative realizations of
   one proposed task state and test equality under a closed set of later removals/swaps/compositions, rather than
   matching local terms. It changes computation/grouping/manipulation only if the same state predicts held-out circuit
   effects and can be exchanged bidirectionally. Kill it if the aggregate shared task component fails after removing
   the shared document mean or if later interventions distinguish the states.
3. **MLP0 finite-input interpretation:** use the complete embedding/token input table to identify token-only groups,
   then separately model token-by-context and context-only finite differences. Candidate variables must group tokens
   by downstream effect and predict held-out contexts before removal. This changes computation, grouping, prediction,
   and manipulation rather than rank. Kill a token group if downstream effects do not interchange across tokens or
   contexts.
4. **Causal interaction atlas for a documented circuit:** start from one of the62 circuits and decompose the input to
   a later bilinear MLP into exact earlier-write self/cross terms, using gradients only to shortlist and finite
   removals to identify the terms. This directly follows the multiple-mediators concern. Kill it if no small set of
   interactions predicts the circuit on held-out documents or removal harms unrelated circuits equally.
5. **Frontier gaps at MLP16 or attention5:** revisit only with a task/circuit-conditioned object such as a
   downstream-response equivalence or exact interaction interface. A new rank allocation, Tucker fit, quantizer, or
   variance curve is rejected because it cannot identify or manipulate a computation.
6. **Hankel/tensor canonicalization:** retain as a mathematical route only after constructing a prefix/suffix-closed
   experiment table or proving the normalized transition lies in the required realization class. The current SVD and
   local CP tensors violate those assumptions.

## Ranked actions

1. Finalize hashes and execute rung516 now; it is the cheapest way to extract circuit-level meaning from the valid
   rung515 null.
2. If rung516 names a compact stable circuit set, use it to preregister a task-defined state-transition test with
   physical exchange and unrelated-circuit controls. If it nulls, close this equality exact-term arc.
3. At closure, compare a task-defined state transition against the MLP0 finite-input route using expected circuit
   information per forward; prefer the one with a concrete held-out interchange test, not the lower-rank promise.
4. Keep MLP16/attention5 pricing gaps behind those causal routes until a new circuit-conditioned object exists.

Rung515's strong null changes the route. Continuing to finer supports or softer similarity thresholds would be a
cosmetic repeat; completing rung516 and then changing the object is now the disciplined path.

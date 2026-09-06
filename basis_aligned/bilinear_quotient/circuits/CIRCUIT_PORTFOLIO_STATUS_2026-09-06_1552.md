# Circuit portfolio status — 2026-09-06 15:52 UTC

## Count the evidence objects, not all of them as circuits

The repository currently exposes three different counts:

- `circuits/registry.json` contains **79 canonical legacy circuit records**. Most are
  census/localization records created before the current tier rubric; none carries an
  explicit current mechanistic or counterfactual tier, so 79 is not a count of completed
  explanations.
- The latest result for each `circuit_fast_screen_result_v1` candidate gives **64 unique
  behavioral candidates**: **49 screens** and **15 nulls**. A screen identifies a useful
  causal site under its registered controls; it is not a complete circuit.
- There are presently **three strongest computation-level linguistic program lines**:
  `aspectual_anchor.has_vs_had`, `tense_auxiliary.is_vs_was`, and
  `temporal_auxiliary.will_vs_had`. The compact Task14+bracket release is an additional
  useful 22-scalar executable behavioral program, but its own boundary certificate does
  not claim recursive causal/weight realization.

No line has earned Tier 5 / CF5. The portfolio is broad at localization and narrow at
full explanation.

## Current quality frontier

The temporal auxiliary line is the clearest current Tier-4 candidate:

- exact upstream writer: block 8 head 1 cue terms write the subject-onset state;
- localized readers: block 9 heads 1/4, block 11 head 3, and block 15 heads 5/1;
- operation identity: changed value vectors transported by near-native patterns account
  for about 100% of the reader response on both original and fresh cue families;
- actual-joint sparse response program: the three head groups plus MLP11/MLP14 reproduce
  **96.65%** of held-out A1 and **96.00%** of untouched A2 writer effect, with every row
  donorward and about tenfold lower rowwise error than the best singleton;
- remaining boundary: the replay is writer-conditioned/cached. The subspaces have not
  yet been converted into a compact weight-derived program and recursively connected to
  token/position primitives.

The aspectual and `is/was` lines have executable transparent programs and prospective
lexical transfer. Their causal recovery and circuit-boundary claims differ, so a single
ordinal comparison would hide important weaknesses. They should be re-audited under the
same Tier/CF rubric before being counted beside the temporal program.

## Incorporating subspace-to-weight translation

For an orthonormal residual subspace `U`, translate every attention head into exact
weight factors:

`Q_h U`, `K_h U`, `Q2_h U`, `K2_h U`, `V_h U`, `U' O_h`, and
`U' O_h V_h U`.

These distinguish routing reads, value reads, residual writes, and direct source-to-target
value transport. For a bilinear MLP, restrict its weights to the subspace tensor

`T[a,i,j] = sum_n (U' Down)[a,n] (Left U)[n,i] (Right U)[n,j]`.

The tensor exactly replays the quadratic MLP on inputs inside `U`; its Frobenius norm and
singular spectrum are invariant to orthogonal rotations of the chosen basis. Therefore
two tasks can share a weight-level circuit even if their fitted activation coordinates
are rotated.

The correct workflow is:

1. identify a causal subspace at one valid live boundary with full-rank closure;
2. map head-local axes through the exact output projection into residual space;
3. rank upstream writers and downstream readers using the contracted weight objects,
   against dimension-matched random-subspace controls;
4. intervene on the predicted weight-mediated edges and require held-out causal replay;
5. compare the gauge-invariant contracted tensors across tasks, then test the shared
   intersection and each task-specific complement separately.

This turns DAS from the final answer into a hypothesis generator for a literal tensor
program. It also diagnoses the scalar-objective cheat: a constrained-DAS direction that
mainly aligns with the final answer readout but lacks corresponding writer/reader weight
incidence should not be promoted as the circuit subspace.

The reusable implementation is `ops/subspace_weight_atlas.py`; its exact OV contraction,
bilinear replay, and basis-rotation invariance tests pass. The next experiment should
apply it to both the constrained-DAS and difference-in-means block11H3 axes, then test
whether the weight-predicted edges explain centered full-vocabulary causal effects.
